from __future__ import annotations

import gc

import pytest
import torch


@pytest.mark.gpu
def test_shared_context_reduces_cuda_allocations():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for shared-context perf check")

    import ludic.training.loss as loss_mod

    device = torch.device("cuda")
    B, T, V = 16, 1024, 8192

    def run_once(mode: str) -> int:
        logits = torch.randn(B, T, V, dtype=torch.float32, device=device, requires_grad=True)
        input_ids = torch.randint(0, V, (B, T), dtype=torch.long, device=device)
        batch = {
            "input_ids": input_ids,
            "action_mask": torch.ones(B, T, dtype=torch.float32, device=device),
        }

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        if mode == "shared":
            shared = loss_mod.SharedContext(logits, batch)
            with loss_mod._use_shared_context(shared):
                token_logp_a = loss_mod.compute_token_logp(logits, input_ids)
                token_logp_b = loss_mod.compute_token_logp(logits, input_ids)
                loss = token_logp_a.sum() + token_logp_b.sum()
        elif mode == "unshared":
            token_logp_a = loss_mod.compute_token_logp(logits, input_ids)
            token_logp_b = loss_mod.compute_token_logp(logits, input_ids)
            loss = token_logp_a.sum() + token_logp_b.sum()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)

        del logits, input_ids, batch, token_logp_a, token_logp_b, loss
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return peak

    # Force eager selective_log_softmax so the savings are measurable and stable.
    original_compiled = loss_mod._USE_COMPILED_SELECTIVE_LOG_SOFTMAX
    loss_mod._USE_COMPILED_SELECTIVE_LOG_SOFTMAX = False
    try:
        run_once("shared")
        run_once("unshared")

        shared_bytes = run_once("shared")
        unshared_bytes = run_once("unshared")
    finally:
        loss_mod._USE_COMPILED_SELECTIVE_LOG_SOFTMAX = original_compiled

    assert unshared_bytes > 0
    assert shared_bytes <= unshared_bytes * 0.8 + 8 * 1024**2
