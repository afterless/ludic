"""Sequence-parallel forward + scalerl/CISPO loss (model-agnostic, reusable).

Two modes, dispatched by `parallel_compute_loss(..., mode=)`:

* **ulysses** (default, recommended): inputs are PLAIN contiguous seq-shards (no
  DTensor); the model's attention is `ulysses_sp` (head-sharded all-to-all over the
  full seq), so any mask works (causal + sliding-window). The loss runs on the local
  `[B, S/P, V]` logits and is reduced over the sp group. Per-rank loss is scaled by
  `sp_size` so FSDP's AVG over the (dp×sp) world yields the correct loss/gradient.
* **ring** (experimental): torch `context_parallel` ring attention. Causal-only and
  carries a DTensor-mixing issue on this stack — kept for completeness, not the path.

At `sp_size == 1` callers should use the normal non-CP loss; these helpers assume
sp_size > 1. The batch is REPLICATED across sp ranks (ludic FSDPBatchSource), and
labels are pre-shifted here (the canonical CP/SP requirement).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ludic.training.loss import (
    CISPOLoss,
    CompositeLoss,
    TokenKLLoss,
    selective_log_softmax,
)


def _extract_scalerl_terms(loss) -> tuple[CISPOLoss, float]:
    """Pull (cispo_loss, kl_multiplier) out of a scalerl loss; raise otherwise."""
    if isinstance(loss, CISPOLoss):
        return loss, 0.0
    if isinstance(loss, CompositeLoss):
        cispo: CISPOLoss | None = None
        kl_mult = 0.0
        for term in loss.terms:
            if isinstance(term.loss, CISPOLoss):
                cispo = term.loss
            elif isinstance(term.loss, TokenKLLoss):
                kl_mult = term.weight * term.loss.coeff
        if cispo is None:
            raise NotImplementedError("Sequence parallelism requires a CISPO term (scalerl).")
        return cispo, kl_mult
    raise NotImplementedError(
        f"Sequence parallelism supports only the scalerl/CISPO loss, got {type(loss).__name__}."
    )


def _preshift(input_ids, action_mask, actor_logps):
    """Per-position alignment: position t pairs logit_t with id/mask/actor_logp_{t+1}."""
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    pos_mask = torch.roll(action_mask, shifts=-1, dims=1).to(torch.float32)
    pos_mask[:, -1] = 0.0
    pos_actor_logps = torch.roll(actor_logps, shifts=-1, dims=1)
    pos_actor_logps[:, -1] = 0.0
    return target_ids, pos_mask, pos_actor_logps


def _cispo_from_shard(token_logp, pos_actor_logps, mask, adv, counts, cispo, kl_mult, P, B):
    """CISPO(+TokenKL) on a local seq shard; ×P scaling so FSDP-AVG over the sp dim is correct."""
    log_ratio = token_logp - pos_actor_logps
    ratio = torch.exp(log_ratio)
    is_weight = torch.clamp(ratio, 1.0 - cispo.clip_eps_low, 1.0 + cispo.clip_eps_high).detach()
    cispo_partial = (is_weight * adv.unsqueeze(-1) * token_logp * mask).sum(dim=-1)
    cispo_obj = cispo_partial / counts if cispo.length_normalize else cispo_partial
    loss = -(P / B) * cispo_obj.sum()
    if kl_mult > 0:
        kl_partial = (log_ratio * mask).sum(dim=-1)
        loss = loss + kl_mult * (P / B) * (kl_partial / counts).sum()
    return loss, ratio, log_ratio


def _sp_stats(loss, token_logp, mask, ratio, log_ratio, adv, group):
    """Logging stats, token-weighted means reduced over the sp group."""
    with torch.no_grad():
        tok = mask.sum()
        dist.all_reduce(tok, op=dist.ReduceOp.SUM, group=group)
        tok = tok.clamp(min=1.0)

        def _mm(x):
            s = (x * mask).sum()
            dist.all_reduce(s, op=dist.ReduceOp.SUM, group=group)
            return s / tok

        loss_disp = loss.detach().clone()
        dist.all_reduce(loss_disp, op=dist.ReduceOp.AVG, group=group)
        return {
            "loss": loss_disp,
            "ratio_mean": _mm(ratio).detach(),
            "kl_actor_policy": _mm(ratio - log_ratio - 1.0).detach(),
            "logp_mean": _mm(token_logp).detach(),
            "adv_mean": adv.mean().detach(),
        }


def ulysses_compute_loss(model, batch, algo, sp_mesh, *, cast_logits_to_fp32=False):
    """Plain seq-shard forward (ulysses attention) + scalerl loss, reduced over sp."""
    cispo, kl_mult = _extract_scalerl_terms(algo.loss)
    group = sp_mesh.get_group()
    P = sp_mesh.size()
    sp_rank = sp_mesh.get_local_rank()

    input_ids = batch["input_ids"]
    action_mask = batch["action_mask"]
    actor_logps = batch["actor_logps"]
    adv = batch["weight"]
    B, T0 = input_ids.shape

    target_ids, pos_mask, pos_actor_logps = _preshift(input_ids, action_mask, actor_logps)

    # pad seq to a multiple of P so it shards evenly (padded positions: pos_mask=0)
    T = ((T0 + P - 1) // P) * P
    if T != T0:
        npad = T - T0
        input_ids = F.pad(input_ids, (0, npad), value=0)
        target_ids = F.pad(target_ids, (0, npad), value=0)
        pos_mask = F.pad(pos_mask, (0, npad), value=0.0)
        pos_actor_logps = F.pad(pos_actor_logps, (0, npad), value=0.0)
    position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

    # contiguous seq-shard for this rank (plain tensors — no DTensor)
    Sl = T // P
    sl = slice(sp_rank * Sl, (sp_rank + 1) * Sl)
    ids_shard = input_ids[:, sl].contiguous()
    pos_shard = position_ids[:, sl].contiguous()
    tgt_shard = target_ids[:, sl].contiguous()
    mask_shard = pos_mask[:, sl].contiguous()
    actor_shard = pos_actor_logps[:, sl].contiguous()

    outputs = model(input_ids=ids_shard, position_ids=pos_shard)
    logits = outputs.logits
    if cast_logits_to_fp32:
        logits = logits.float()

    token_logp = selective_log_softmax(logits, tgt_shard)  # [B, Sl]
    mask = mask_shard.to(token_logp.dtype)
    counts = mask.sum(dim=-1)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)  # global per-sample token count
    counts = counts.clamp(min=1.0)

    loss, ratio, log_ratio = _cispo_from_shard(
        token_logp, actor_shard, mask, adv, counts, cispo, kl_mult, P, B
    )
    stats = _sp_stats(loss, token_logp, mask, ratio, log_ratio, adv, group)
    return loss, stats


def cp_compute_loss(model, batch, algo, cp_mesh, *, cast_logits_to_fp32=False):
    """EXPERIMENTAL ring-attention (torch context_parallel) loss — causal-only; carries a
    DTensor-mixing issue on this stack. Prefer `ulysses_compute_loss`."""
    from torch.distributed.tensor.experimental import context_parallel

    cispo, kl_mult = _extract_scalerl_terms(algo.loss)
    group = cp_mesh.get_group()
    P = cp_mesh.size()

    input_ids = batch["input_ids"]
    action_mask = batch["action_mask"]
    actor_logps = batch["actor_logps"]
    adv = batch["weight"]
    B, T0 = input_ids.shape

    target_ids, pos_mask, pos_actor_logps = _preshift(input_ids, action_mask, actor_logps)
    chunk = 2 * P
    T = ((T0 + chunk - 1) // chunk) * chunk
    if T != T0:
        npad = T - T0
        input_ids = F.pad(input_ids, (0, npad), value=0)
        target_ids = F.pad(target_ids, (0, npad), value=0)
        pos_mask = F.pad(pos_mask, (0, npad), value=0.0)
        pos_actor_logps = F.pad(pos_actor_logps, (0, npad), value=0.0)
    position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1).contiguous()

    buffers = [input_ids, position_ids, target_ids, pos_mask, pos_actor_logps]
    with context_parallel(
        cp_mesh, buffers=buffers, buffer_seq_dims=[1, 1, 1, 1, 1],
        no_restore_buffers=set(buffers),
    ):
        outputs = model(input_ids=input_ids, position_ids=position_ids)
        logits = outputs.logits.float() if cast_logits_to_fp32 else outputs.logits
        token_logp = selective_log_softmax(logits, target_ids)
        mask = pos_mask.to(token_logp.dtype)
        counts = mask.sum(dim=-1)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        counts = counts.clamp(min=1.0)
        loss, ratio, log_ratio = _cispo_from_shard(
            token_logp, pos_actor_logps, mask, adv, counts, cispo, kl_mult, P, B
        )
    stats = _sp_stats(loss, token_logp, mask, ratio, log_ratio, adv, group)
    return loss, stats


def parallel_compute_loss(model, batch, algo, mesh, *, mode="ulysses", cast_logits_to_fp32=False):
    """Dispatch the sequence-parallel loss by `mode` ('ulysses' | 'ring')."""
    if mode == "ulysses":
        return ulysses_compute_loss(model, batch, algo, mesh, cast_logits_to_fp32=cast_logits_to_fp32)
    if mode == "ring":
        return cp_compute_loss(model, batch, algo, mesh, cast_logits_to_fp32=cast_logits_to_fp32)
    raise ValueError(f"unknown parallel mode {mode!r} (expected 'ulysses' or 'ring')")
