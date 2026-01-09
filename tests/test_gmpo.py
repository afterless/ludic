"""
Tests for GMPO (Geometric-Mean Policy Optimization) implementation.
"""
from __future__ import annotations

import pytest
import torch
import numpy as np

from ludic.training.loss import GMPOLoss, TokenClippedSurrogateLoss
from ludic.training.algorithm import make_gmpo, make_grpo


class TestGMPOLoss:
    """Tests for the GMPOLoss class."""

    def test_gmpo_loss_basic(self):
        """Test basic GMPO loss computation."""
        loss_fn = GMPOLoss(clip_eps_low=0.4, clip_eps_high=0.4, length_normalize=True)

        # Create simple batch with 2 samples, 4 tokens each
        B, T, V = 2, 4, 10
        logits = torch.randn(B, T, V, dtype=torch.float32)
        input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
        action_mask = torch.tensor(
            [[0, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.float32
        )  # Variable action lengths
        advantages = torch.tensor([1.5, -0.8], dtype=torch.float32)

        # Create actor logprobs (simulating behavior policy)
        actor_logps = torch.randn(B, T, dtype=torch.float32)

        batch = {
            "input_ids": input_ids,
            "action_mask": action_mask,
            "weight": advantages,
            "attention_mask": torch.ones(B, T, dtype=torch.float32),
            "actor_logps": actor_logps,
        }

        loss, stats = loss_fn.compute(logits, batch)

        # Verify outputs
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Verify stats
        assert "loss" in stats
        assert "ratio_mean" in stats
        assert "ratio_std" in stats
        assert "geom_mean_ratio_mean" in stats
        assert "geom_mean_ratio_std" in stats
        assert "clip_frac" in stats
        assert "kl_actor_policy" in stats
        assert "adv_mean" in stats
        assert "adv_std" in stats
        assert "avg_action_tokens" in stats

        # Check that all stats are scalars
        for key, val in stats.items():
            assert val.ndim == 0, f"Stat {key} should be scalar, got shape {val.shape}"

    def test_gmpo_log_space_stability(self):
        """Test that GMPO correctly handles operations in log-space."""
        loss_fn = GMPOLoss(clip_eps_low=0.4, clip_eps_high=0.4, length_normalize=True)

        B, T, V = 1, 10, 50
        logits = torch.randn(B, T, V, dtype=torch.float32)
        input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
        action_mask = torch.ones(B, T, dtype=torch.float32)
        action_mask[:, 0] = 0  # First token is prompt

        # Create very small/large log probs to test numerical stability
        advantages = torch.tensor([2.0], dtype=torch.float32)
        actor_logps = torch.randn(B, T, dtype=torch.float32) * 5  # Wider range

        batch = {
            "input_ids": input_ids,
            "action_mask": action_mask,
            "weight": advantages,
            "attention_mask": torch.ones(B, T, dtype=torch.float32),
            "actor_logps": actor_logps,
        }

        loss, stats = loss_fn.compute(logits, batch)

        # Should not produce NaN or Inf even with extreme values
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert not torch.isnan(stats["geom_mean_ratio_mean"])
        assert not torch.isinf(stats["geom_mean_ratio_mean"])

    def test_gmpo_sign_handling(self):
        """Test that GMPO correctly handles advantage sign."""
        loss_fn = GMPOLoss(clip_eps_low=0.4, clip_eps_high=0.4, length_normalize=True)

        B, T, V = 2, 5, 20
        logits = torch.randn(B, T, V, dtype=torch.float32)
        input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
        action_mask = torch.tensor(
            [[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.float32
        )

        # One positive, one negative advantage
        advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)
        actor_logps = torch.randn(B, T, dtype=torch.float32)

        batch = {
            "input_ids": input_ids,
            "action_mask": action_mask,
            "weight": advantages,
            "attention_mask": torch.ones(B, T, dtype=torch.float32),
            "actor_logps": actor_logps,
        }

        loss, stats = loss_fn.compute(logits, batch)

        # Loss should be finite
        assert torch.isfinite(loss)
        # Advantages should have mean close to 0
        assert torch.abs(stats["adv_mean"]) < 1e-6

    def test_gmpo_length_normalization(self):
        """Test that length normalization works correctly."""
        loss_fn_norm = GMPOLoss(
            clip_eps_low=0.4, clip_eps_high=0.4, length_normalize=True
        )
        loss_fn_no_norm = GMPOLoss(
            clip_eps_low=0.4, clip_eps_high=0.4, length_normalize=False
        )

        B, T, V = 1, 6, 15
        logits = torch.randn(B, T, V, dtype=torch.float32)
        input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
        action_mask = torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.float32)
        advantages = torch.tensor([1.0], dtype=torch.float32)
        actor_logps = torch.randn(B, T, dtype=torch.float32)

        batch = {
            "input_ids": input_ids,
            "action_mask": action_mask,
            "weight": advantages,
            "attention_mask": torch.ones(B, T, dtype=torch.float32),
            "actor_logps": actor_logps,
        }

        loss_norm, _ = loss_fn_norm.compute(logits, batch)
        loss_no_norm, _ = loss_fn_no_norm.compute(logits, batch)

        # Both should be finite
        assert torch.isfinite(loss_norm)
        assert torch.isfinite(loss_no_norm)

    def test_gmpo_vs_grpo_outlier_robustness(self):
        """Test that GMPO is more robust to outliers than GRPO."""
        # This is a conceptual test showing the difference
        gmpo_loss = GMPOLoss(clip_eps_low=0.4, clip_eps_high=0.4, length_normalize=True)
        grpo_loss = TokenClippedSurrogateLoss(
            clip_eps_low=0.2, clip_eps_high=0.27, length_normalize=False
        )

        B, T, V = 1, 5, 10
        logits = torch.randn(B, T, V, dtype=torch.float32)
        input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
        action_mask = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.float32)
        advantages = torch.tensor([2.0], dtype=torch.float32)

        # Create actor logprobs with one outlier token
        actor_logps = torch.tensor([[-1.0, -1.0, -10.0, -1.0, -1.0]], dtype=torch.float32)

        batch = {
            "input_ids": input_ids,
            "action_mask": action_mask,
            "weight": advantages,
            "attention_mask": torch.ones(B, T, dtype=torch.float32),
            "actor_logps": actor_logps,
        }

        gmpo_loss_val, gmpo_stats = gmpo_loss.compute(logits, batch)
        grpo_loss_val, grpo_stats = grpo_loss.compute(logits, batch)

        # Both should produce valid losses
        assert torch.isfinite(gmpo_loss_val)
        assert torch.isfinite(grpo_loss_val)

        # GMPO should have more stable geometric mean ratio
        # (This is more of a sanity check; the real test is in training dynamics)
        assert torch.isfinite(gmpo_stats["geom_mean_ratio_mean"])


class TestMakeGMPO:
    """Tests for the make_gmpo algorithm preset."""

    def test_make_gmpo_basic(self):
        """Test basic GMPO algorithm creation."""
        algo = make_gmpo(group_size=4)

        assert algo.name == "gmpo"
        assert algo.credit_assigner is not None
        assert algo.loss is not None
        assert algo.preprocess is not None
        assert isinstance(algo.loss, GMPOLoss)

    def test_make_gmpo_parameters(self):
        """Test GMPO with various parameter configurations."""
        # Test with custom clipping
        algo = make_gmpo(
            group_size=8,
            clip_eps_low=0.3,
            clip_eps_high=0.5,
            length_normalize=False,
        )

        assert isinstance(algo.loss, GMPOLoss)
        assert algo.loss.clip_eps_low == 0.3
        assert algo.loss.clip_eps_high == 0.5
        assert algo.loss.length_normalize is False

        # Test with ratio clipping
        algo_clip = make_gmpo(group_size=4, ratio_clip=5.0)
        assert algo_clip.loss.ratio_clip == 5.0

        # Test with positive_only
        algo_pos = make_gmpo(group_size=4, positive_only=True)
        assert algo_pos.credit_assigner.positive_only is True

    def test_gmpo_vs_grpo_config_differences(self):
        """Test that GMPO and GRPO have appropriate default differences."""
        gmpo = make_gmpo(group_size=4)
        grpo = make_grpo(group_size=4)

        # GMPO should have wider clipping by default
        assert gmpo.loss.clip_eps_low == 0.4  # e^-0.4 ≈ 0.67
        assert gmpo.loss.clip_eps_high == 0.4  # e^0.4 ≈ 1.49

        # GRPO has tighter clipping
        assert grpo.loss.clip_eps_low == 0.2  # 0.8
        assert grpo.loss.clip_eps_high == 0.27  # 1.27

        # GMPO should use length normalization by default
        assert gmpo.loss.length_normalize is True
        assert grpo.loss.length_normalize is False

    def test_gmpo_missing_actor_logps_raises(self):
        """Test that GMPO raises error when actor_logps are missing."""
        from ludic.training.types import SAWItem, SAWBatch, SampleAttachments

        algo = make_gmpo(group_size=4)

        # Create a batch without actor_logps
        item = SAWItem(
            input_ids=[1, 2, 3, 4, 5],
            attention_mask=[1, 1, 1, 1, 1],
            action_mask=[0, 0, 0, 1, 1],
            weight=1.0,
            meta={},
            attachments=SampleAttachments(actor_logps=None),  # Missing!
        )

        batch = SAWBatch(items=[item])

        # Preprocess should raise error
        with pytest.raises(ValueError, match="Missing.*actor_logps"):
            algo.preprocess(batch)


class TestGMPONumericalStability:
    """Test numerical stability of GMPO implementation."""

    def test_extreme_advantages(self):
        """Test GMPO with extreme advantage values."""
        loss_fn = GMPOLoss(clip_eps_low=0.4, clip_eps_high=0.4, length_normalize=True)

        B, T, V = 2, 5, 10
        logits = torch.randn(B, T, V, dtype=torch.float32)
        input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
        action_mask = torch.tensor(
            [[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.float32
        )

        # Extreme advantages
        advantages = torch.tensor([100.0, -100.0], dtype=torch.float32)
        actor_logps = torch.randn(B, T, dtype=torch.float32)

        batch = {
            "input_ids": input_ids,
            "action_mask": action_mask,
            "weight": advantages,
            "attention_mask": torch.ones(B, T, dtype=torch.float32),
            "actor_logps": actor_logps,
        }

        loss, stats = loss_fn.compute(logits, batch)

        # Should still be stable
        assert torch.isfinite(loss)
        assert torch.isfinite(stats["geom_mean_ratio_mean"])

    def test_zero_advantage(self):
        """Test GMPO with zero advantages."""
        loss_fn = GMPOLoss(clip_eps_low=0.4, clip_eps_high=0.4, length_normalize=True)

        B, T, V = 1, 5, 10
        logits = torch.randn(B, T, V, dtype=torch.float32)
        input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
        action_mask = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.float32)

        # Zero advantage
        advantages = torch.tensor([0.0], dtype=torch.float32)
        actor_logps = torch.randn(B, T, dtype=torch.float32)

        batch = {
            "input_ids": input_ids,
            "action_mask": action_mask,
            "weight": advantages,
            "attention_mask": torch.ones(B, T, dtype=torch.float32),
            "actor_logps": actor_logps,
        }

        loss, stats = loss_fn.compute(logits, batch)

        # Loss should be near zero (since advantage is zero)
        assert torch.isfinite(loss)
        # With zero advantage, objective should be zero
        assert abs(loss.item()) < 1e-5

    def test_very_long_sequences(self):
        """Test GMPO with long sequences (stress test)."""
        loss_fn = GMPOLoss(clip_eps_low=0.4, clip_eps_high=0.4, length_normalize=True)

        B, T, V = 1, 512, 100  # Long sequence
        logits = torch.randn(B, T, V, dtype=torch.float32)
        input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
        action_mask = torch.ones(B, T, dtype=torch.float32)
        action_mask[:, :10] = 0  # First 10 tokens are prompt

        advantages = torch.tensor([1.5], dtype=torch.float32)
        actor_logps = torch.randn(B, T, dtype=torch.float32)

        batch = {
            "input_ids": input_ids,
            "action_mask": action_mask,
            "weight": advantages,
            "attention_mask": torch.ones(B, T, dtype=torch.float32),
            "actor_logps": actor_logps,
        }

        loss, stats = loss_fn.compute(logits, batch)

        # Should handle long sequences without numerical issues
        assert torch.isfinite(loss)
        assert torch.isfinite(stats["geom_mean_ratio_mean"])
        assert stats["avg_action_tokens"].item() == 502  # T - 10
