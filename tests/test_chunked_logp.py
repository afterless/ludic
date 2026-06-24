"""Parity + memory checks for the chunked selective_log_softmax (OOM fix).

Run standalone: `python ludic/tests/test_chunked_logp.py` (parity on CPU; memory
test runs only if CUDA is present). Also pytest-compatible.
"""

import torch

from ludic.training.loss import selective_log_softmax, _ChunkedSelectiveLogSoftmax


def _ref(logits, index):
    return torch.gather(
        logits.log_softmax(dim=-1), dim=-1, index=index.unsqueeze(-1)
    ).squeeze(-1)


def _check_fn(B, T, V, chunk, atol=2e-5, rtol=1e-4):
    torch.manual_seed(0)
    base = torch.randn(B, T, V, dtype=torch.float32, requires_grad=True)
    idx_full = torch.randint(0, V, (B, T))
    # mirror the real caller: a non-contiguous shifted slice
    logits, index = base[:, :-1, :], idx_full[:, 1:]

    ref = _ref(logits, index)
    got = _ChunkedSelectiveLogSoftmax.apply(logits, index, chunk)
    assert torch.allclose(ref, got, atol=atol, rtol=rtol), f"fwd mismatch chunk={chunk}"

    gref = torch.autograd.grad(ref.sum(), base, retain_graph=True)[0]
    ggot = torch.autograd.grad(got.sum(), base)[0]
    assert torch.allclose(gref, ggot, atol=atol, rtol=rtol), f"bwd mismatch chunk={chunk}"


def test_parity_chunk_regimes():
    # ragged last chunk, tiny chunk, single chunk (chunk >= B*S), B=1
    for chunk in [1, 7, 64, 1000, 10**9]:
        _check_fn(2, 130, 4096, chunk)
    _check_fn(1, 200, 4096, 37)
    _check_fn(3, 96, 2048, 50)


def test_public_routing_matches_eager():
    # B*S > chunk -> chunked branch; result must equal the eager reference
    torch.manual_seed(1)
    B, T, V = 2, 300, 2048
    base = torch.randn(B, T, V, requires_grad=True)
    idx = torch.randint(0, V, (B, T))
    logits, index = base[:, :-1, :], idx[:, 1:]
    got = selective_log_softmax(logits, index, chunk_size=64)
    eager = selective_log_softmax(logits, index, chunk_size=0)
    assert torch.allclose(got, eager, atol=2e-5, rtol=1e-4)


def test_memory_peak_drops():
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping memory test")
        return
    dev, V, B, S = "cuda", 262144, 1, 3072
    peaks = {}
    for label in ("eager", "chunked"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        logits = torch.randn(B, S, V, device=dev, dtype=torch.float32, requires_grad=True)
        index = torch.randint(0, V, (B, S), device=dev)
        out = (
            _ChunkedSelectiveLogSoftmax.apply(logits, index, 1024)
            if label == "chunked"
            else _ref(logits, index)
        )
        out.sum().backward()
        peaks[label] = torch.cuda.max_memory_allocated() / 2**30
        del logits, index, out
    print(f"peak GiB: eager={peaks['eager']:.2f} chunked={peaks['chunked']:.2f}")
    assert peaks["chunked"] < peaks["eager"] - 3.0, peaks


if __name__ == "__main__":
    test_parity_chunk_regimes()
    test_public_routing_matches_eager()
    print("parity OK")
    test_memory_peak_drops()
