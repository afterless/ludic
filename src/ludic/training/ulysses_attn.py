"""Ulysses sequence-parallel attention for HF models (model-agnostic, reusable).

Each rank starts with q/k/v seq-sharded `[B, H, S/P, D]` (RoPE already applied with
GLOBAL positions). all-to-all -> head-sharded `[B, H/P, S, D]` (full seq), compute
attention NORMALLY (full mask), all-to-all back -> `[B, H, S/P, D]`. Because each rank
sees the FULL sequence per head, masks are mask-agnostic: causal (global) and
sliding-window layers both work — the mask is rebuilt over the full seq here (the
shard-sized mask HF passes is ignored). This supports "exotic" attention patterns
(sliding window, etc.) under sequence parallelism that ring-attention CP cannot.

Register as `attn_implementation="ulysses_sp"` (done at import). The sp ProcessGroup
is installed by the trainer via `set_sp_group()` before the first forward; at sp_size
== 1 the fn falls back to plain SDPA (no-op).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

_SP_GROUP = None
_SP_SIZE = 1
# Keyed by (sliding_window, device) — NOT by S. Each entry holds one mask sized to
# the largest S seen so far; smaller S slice its top-left. Keying by S leaked ~1GB/step
# because every distinct full-seq length permanently cached a fresh [S, S] bool tensor.
_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def set_sp_group(group, size: int) -> None:
    """Install the sequence-parallel ProcessGroup the attention all-to-alls over."""
    global _SP_GROUP, _SP_SIZE
    _SP_GROUP, _SP_SIZE = group, size


class _AllToAll(torch.autograd.Function):
    """Autograd-aware `all_to_all_single` (even split on dim 0).

    The raw collective is not differentiable; gradients must flow back across ranks
    through attention. all_to_all with even splits is its own adjoint, so backward is
    another all_to_all of the gradient.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        x = x.contiguous()  # contiguous BEFORE empty_like so the output is contiguous too
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()  # all_to_all_single requires a contiguous output tensor
        gin = torch.empty_like(grad)
        dist.all_to_all_single(gin, grad, group=ctx.group)
        return gin, None


def _seq_to_head(x: torch.Tensor, P: int, group) -> torch.Tensor:
    """`[B, H, S/P, D]` (seq-sharded) -> `[B, H/P, S, D]` (head-sharded, full seq)."""
    B, H, Sl, D = x.shape
    inp = x.view(B, P, H // P, Sl, D).permute(1, 0, 2, 3, 4).contiguous()  # [P, B, H/P, Sl, D]
    out = _AllToAll.apply(inp, group)  # dim0 now indexes source rank (seq shard)
    return out.permute(1, 2, 0, 3, 4).reshape(B, H // P, P * Sl, D).contiguous()


def _head_to_seq(x: torch.Tensor, P: int, group) -> torch.Tensor:
    """`[B, H/P, S, D]` (head-sharded, full seq) -> `[B, H, S/P, D]` (seq-sharded)."""
    B, Hp, S, D = x.shape
    Sl = S // P
    inp = x.view(B, Hp, P, Sl, D).permute(2, 0, 1, 3, 4).contiguous()  # [P, B, H/P, Sl, D]
    out = _AllToAll.apply(inp, group)  # dim0 now indexes source rank (head group)
    return out.permute(1, 0, 2, 3, 4).reshape(B, P * Hp, Sl, D).contiguous()


def _sliding_window_mask(S: int, sw: int, device) -> torch.Tensor:
    """Boolean [S, S] sliding causal band: attend iff (j <= i) and (i - j < sw).

    Cache one mask per (sw, device) sized to the largest S seen; the band is
    origin-anchored, so any smaller S is exactly the top-left [:S, :S] slice. This
    keeps the cache at O(1) tensors instead of one [S, S] per distinct seq length.
    """
    key = (sw, device)
    m = _MASK_CACHE.get(key)
    if m is None or m.shape[0] < S:
        i = torch.arange(S, device=device).unsqueeze(1)
        j = torch.arange(S, device=device).unsqueeze(0)
        m = (j <= i) & (i - j < sw)
        _MASK_CACHE[key] = m
    return m[:S, :S]


def ulysses_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    group, P = _SP_GROUP, _SP_SIZE
    if group is None or P == 1:
        # No SP active → behave exactly like standard SDPA (correct causal/mask/GQA).
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
        )

    # seq-sharded [B, H, S/P, D] -> head-sharded [B, H/P, S, D] (full seq)
    q = _seq_to_head(query, P, group)
    k = _seq_to_head(key, P, group)
    v = _seq_to_head(value, P, group)
    k = repeat_kv(k, module.num_key_value_groups)  # GQA: [B, H_kv/P, S, D] -> [B, H/P, S, D]
    v = repeat_kv(v, module.num_key_value_groups)

    S = q.shape[2]
    if getattr(module, "is_sliding", False) and getattr(module, "sliding_window", None):
        attn_mask = _sliding_window_mask(S, module.sliding_window, q.device)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout, scale=scaling,
        )
    else:
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout, scale=scaling, is_causal=True,
        )

    # head-sharded [B, H/P, S, D] -> seq-sharded [B, H, S/P, D], then HF layout [B, S/P, H, D]
    out = _head_to_seq(out, P, group)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("ulysses_sp", ulysses_attention_forward)
