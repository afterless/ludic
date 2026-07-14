"""Fused linear + log-softmax for per-token log-probs without materializing (B,T,V).

The model forward stops at hidden states; this computes token log-probs by tiling
``hidden @ lm_head_weight.T`` over tokens, applying Gemma's final-logit softcap, then
log_softmax + gather per tile. Each tile is wrapped in ``torch.utils.checkpoint`` so the
``(tile, V)`` logits are recomputed in backward rather than stored — peak memory is one
tile, not the full ``(B,T,V)`` logits plus its gradient. The lm_head is frozen (no LoRA),
so only ``grad_hidden`` flows back; the gathered weight is treated as a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint


def _tile_logp(hidden_tile, weight, target_tile, softcap):
    """Per-token logp for one token tile: softcap(hidden@W.T) -> fp32 log_softmax -> gather."""
    z = hidden_tile @ weight.t()  # [n, V]
    if softcap is not None:
        z = softcap * torch.tanh(z / softcap)
    z = z.float()  # fp32 for IS-ratio stability (matches cast_logits_to_fp32)
    lse = torch.logsumexp(z, dim=-1)  # [n]
    target_logit = z.gather(-1, target_tile.unsqueeze(-1)).squeeze(-1)  # [n]
    return target_logit - lse  # [n]


def fused_token_logp(hidden, weight, target_ids, softcap=None, tile_tokens=512):
    """Per-token log pi(target | hidden) as ``[B, S]``, tiling so ``(B*S, V)`` never materializes.

    hidden: ``[B, S, H]``; weight: ``[V, H]`` (gathered, frozen); target_ids: ``[B, S]`` long.
    """
    B, S, H = hidden.shape
    n = B * S
    hidden_flat = hidden.reshape(n, H)
    target_flat = target_ids.reshape(n)
    tile = max(1, int(tile_tokens))
    out = []
    for start in range(0, n, tile):
        end = min(start + tile, n)
        h = hidden_flat[start:end]
        t = target_flat[start:end]
        if torch.is_grad_enabled() and h.requires_grad:
            lp = checkpoint(_tile_logp, h, weight, t, softcap, use_reentrant=False)
        else:
            lp = _tile_logp(h, weight, t, softcap)
        out.append(lp)
    return torch.cat(out, dim=0).reshape(B, S)


@dataclass
class FusedLMHead:
    """Stand-in for ``logits`` carrying hidden states + the frozen output weight, so the loss
    computes ``token_logp`` via the fused path instead of a materialized ``(B,T,V)`` tensor.

    Only ``_compute_token_logp_raw`` / ``_compute_logp_action_raw`` unpack it; every other
    loss consumer reaches token_logp through those, so the rest of the stack is untouched.
    """

    hidden: torch.Tensor  # [B, S, H]
    weight: torch.Tensor  # [V, H] gathered, frozen
    softcap: Optional[float] = None
    tile_tokens: int = 512

    def token_logp(self, input_ids) -> torch.Tensor:
        """Shifted next-token logp ``[B, S-1]`` (matches ``_compute_token_logp_raw``)."""
        return fused_token_logp(
            self.hidden[:, :-1, :],
            self.weight,
            input_ids[:, 1:],
            softcap=self.softcap,
            tile_tokens=self.tile_tokens,
        )


def final_logit_softcap(model) -> Optional[float]:
    """Gemma's ``final_logit_softcapping`` (read from the text sub-config), or None."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    get_text_config = getattr(cfg, "get_text_config", None)
    text_cfg = get_text_config() if callable(get_text_config) else cfg
    return getattr(text_cfg, "final_logit_softcapping", None)


def _resolve_backbone(model):
    """The submodule whose forward returns ``last_hidden_state`` (everything before lm_head)."""
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise AttributeError(
            f"No `.model` backbone on {type(model).__name__} for the fused LM head."
        )
    return backbone


def build_fused_lm_head(
    model, input_ids, *, attention_mask=None, position_ids=None, tile_tokens=512
) -> FusedLMHead:
    """Run the backbone (skipping lm_head), gather the frozen output weight, wrap in FusedLMHead."""
    backbone = _resolve_backbone(model)
    out = backbone(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
    )
    hidden = out.last_hidden_state
    weight = model.get_output_embeddings().weight
    if weight.requires_grad:
        raise RuntimeError(
            "fused LM head assumes a frozen lm_head (no LoRA / modules_to_save); "
            "weight.requires_grad=True would make the fused backward drop its gradient."
        )
    full_weight = weight.full_tensor() if hasattr(weight, "full_tensor") else weight
    return FusedLMHead(
        hidden=hidden,
        weight=full_weight,
        softcap=final_logit_softcap(model),
        tile_tokens=tile_tokens,
    )
