"""Unit tests for fused_token_logp.

Forward values and grad_hidden must match the reference selective path
(softcap(hidden@W.T) -> log_softmax -> gather) to fp32 precision, across softcap
on/off and tile sizes (1, ragged, single-tile larger than B*S). Run on the cluster
(needs torch): `.venv/bin/python -m pytest ludic/tests/test_fused_lm_head.py`.
"""

import torch

from ludic.training.fused_lm_head import fused_token_logp


def _reference_token_logp(hidden, weight, target_ids, softcap):
    z = hidden @ weight.t()
    if softcap is not None:
        z = softcap * torch.tanh(z / softcap)
    z = z.float()
    logp = torch.log_softmax(z, dim=-1)
    return logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def _check(B, S, H, V, softcap, tile, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(B, S, H, generator=g, dtype=torch.float32)
    weight = torch.randn(V, H, generator=g, dtype=torch.float32) * 0.02
    target = torch.randint(0, V, (B, S), generator=g)

    h1 = hidden.clone().requires_grad_(True)
    h2 = hidden.clone().requires_grad_(True)

    fused = fused_token_logp(h1, weight, target, softcap=softcap, tile_tokens=tile)
    ref = _reference_token_logp(h2, weight, target, softcap)

    assert fused.shape == (B, S)
    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-5), (fused - ref).abs().max().item()

    fused.sum().backward()
    ref.sum().backward()
    assert torch.allclose(h1.grad, h2.grad, atol=1e-5, rtol=1e-5), (
        (h1.grad - h2.grad).abs().max().item()
    )


def test_fused_matches_reference_fwd_bwd():
    for softcap in (None, 30.0):
        for tile in (1, 4, 13, 100_000):  # tile=1, ragged, single-tile (> B*S)
            _check(2, 6, 8, 19, softcap, tile)


def test_shapes_and_seeds():
    _check(1, 2, 4, 7, 30.0, 1)
    _check(3, 10, 16, 50, None, 8, seed=1)
    _check(4, 5, 32, 128, 30.0, 7, seed=2)


def test_no_grad_path_matches():
    # Inference path (no checkpoint) must equal the reference too.
    g = torch.Generator().manual_seed(3)
    hidden = torch.randn(2, 5, 8, generator=g)
    weight = torch.randn(17, 8, generator=g) * 0.02
    target = torch.randint(0, 17, (2, 5), generator=g)
    with torch.no_grad():
        fused = fused_token_logp(hidden, weight, target, softcap=30.0, tile_tokens=4)
        ref = _reference_token_logp(hidden, weight, target, 30.0)
    assert torch.allclose(fused, ref, atol=1e-5, rtol=1e-5)
