"""Numerical check for the CISPO seq-parallel logging stats (clip_frac, ess_frac).

These gate off-policy / clip-saturation calls during RL, so the math must be exact.
Runs a single-rank (world_size=1) gloo group so the sp all-reduces in `_sp_stats`
are identity, then checks against hand-computed values.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ludic.training.seq_parallel_loss import _sp_stats


def test_sp_stats_clip_frac_and_ess(tmp_path):
    if not dist.is_initialized():
        store = dist.FileStore(str(tmp_path / "store"), 1)
        dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        # 3 action tokens (ratio 0.5/1.0/1.5) + 1 padding (mask 0; ratio 99 must be ignored).
        ratio = torch.tensor([[0.5, 1.0, 1.5, 99.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        log_ratio = ratio.log()
        token_logp = torch.full_like(ratio, -1.0)
        adv = torch.tensor([0.0])
        loss = torch.tensor(0.0)

        clip_low, clip_high = 0.20, 0.28  # clip band [0.80, 1.28]
        stats = _sp_stats(
            loss, token_logp, mask, ratio, log_ratio, adv,
            dist.group.WORLD, clip_low, clip_high,
        )

        # 0.5 < 0.80 (clip), 1.0 in band, 1.5 > 1.28 (clip) -> 2/3; padding's 99 excluded by mask.
        assert abs(float(stats["clip_frac"]) - 2 / 3) < 1e-6
        # E[r]=1.0, E[r^2]=3.5/3 -> ess_frac = 1.0^2 / (3.5/3) = 6/7.
        assert abs(float(stats["ess_frac"]) - 6 / 7) < 1e-6
        # ratio_mean over the 3 action tokens only (1.0, not ~25) confirms the mask holds.
        assert abs(float(stats["ratio_mean"]) - 1.0) < 1e-6
    finally:
        dist.destroy_process_group()
