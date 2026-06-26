"""Regression tests for the Ulysses sliding-window mask cache.

The cache was previously keyed by sequence length S, which varies every
micro-batch, so it accumulated one [S, S] bool mask per distinct length and
never evicted them (~1 GB/step GPU leak → OOM ~step 60). The fix keys by
(sliding_window, device), keeps one mask sized to the largest S, and returns the
origin-anchored top-left [:S, :S] slice. These tests lock in (1) slice
correctness and (2) the O(1) cache bound.
"""

import torch

from ludic.training.ulysses_attn import _MASK_CACHE, _sliding_window_mask


def _fresh(S: int, sw: int) -> torch.Tensor:
    i = torch.arange(S).unsqueeze(1)
    j = torch.arange(S).unsqueeze(0)
    return (j <= i) & (i - j < sw)


def test_slice_is_bit_identical_to_fresh():
    """Sliced masks must equal a freshly built mask at every S (no grad corruption)."""
    _MASK_CACHE.clear()
    sw, dev = 1024, torch.device("cpu")
    _sliding_window_mask(4096, sw, dev)  # seed the cache at a large S
    for S in [1, 2, 7, 64, 513, 1000, 4095, 4096]:
        got = _sliding_window_mask(S, sw, dev)
        assert torch.equal(got, _fresh(S, sw)), f"mask mismatch at S={S}"


def test_cache_is_bounded_across_distinct_lengths():
    """Distinct sequence lengths must NOT grow the cache (the leak fix)."""
    _MASK_CACHE.clear()
    sw, dev = 512, torch.device("cpu")
    for S in range(100, 2001, 4):  # 475 distinct lengths, like a real run
        _sliding_window_mask(S, sw, dev)
    assert len(_MASK_CACHE) == 1, f"cache grew to {len(_MASK_CACHE)} entries (leaked)"


def test_cache_grows_mask_to_largest_S():
    """The single cached mask tracks the largest S requested so far."""
    _MASK_CACHE.clear()
    sw, dev = 256, torch.device("cpu")
    _sliding_window_mask(500, sw, dev)
    assert _MASK_CACHE[(sw, dev)].shape[0] == 500
    _sliding_window_mask(1200, sw, dev)  # larger → rebuild
    assert _MASK_CACHE[(sw, dev)].shape[0] == 1200
    _sliding_window_mask(300, sw, dev)  # smaller → reuse+slice, no shrink
    assert _MASK_CACHE[(sw, dev)].shape[0] == 1200
