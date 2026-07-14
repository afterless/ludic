"""Unit tests for Trainer._fetch_macro_batch (consistent-size backfill).

The method only touches item.meta + algo.preprocess, so we drive it with a stub
self (SimpleNamespace) and a fake batch source — no model/GPU needed.
"""
import asyncio
from types import SimpleNamespace

from ludic.training.trainer import Trainer
from ludic.training.types import SAWBatch, SAWItem


def _item(pv, adv=1.0, reward=1.0, rid=0, clen=10):
    return SAWItem(
        input_ids=[1],
        attention_mask=[1],
        action_mask=[1],
        weight=1.0,
        meta={
            "policy_version": pv,
            "advantage": adv,
            "total_reward": reward,
            "rollout_id": rid,
            "completion_length": clen,
        },
    )


def _drop_zero(batch):
    """Mimic drop_zero_weight: drop zero-advantage samples."""
    return SAWBatch(
        items=[it for it in batch.items if it.meta.get("advantage", 0.0) != 0.0],
        meta=batch.meta,
    )


class _FakeSource:
    def __init__(self, batches, batch_size):
        self._batches = list(batches)
        self.batch_size = batch_size
        self.pulls = 0

    async def next_batch(self):
        # Repeat the last prepared batch if the backfill needs more pulls.
        batch = self._batches[min(self.pulls, len(self._batches) - 1)]
        self.pulls += 1
        return batch


def _stub(source, max_lag=2, step=10, preprocess=_drop_zero):
    return SimpleNamespace(
        _batch_source=source,
        cfg=SimpleNamespace(max_lag=max_lag),
        _train_step_idx=step,
        algo=SimpleNamespace(preprocess=preprocess),
    )


def test_backfill_reaches_target_and_trims():
    # target=4. pull1: 2 fresh (pv10) + 2 stale (pv5) -> 2 trainable. pull2: 4 fresh
    # -> +4 = 6, trim to 4. Needs exactly 2 pulls.
    pull1 = SAWBatch(items=[_item(10, rid=1), _item(10, rid=2), _item(5, rid=3), _item(5, rid=4)])
    pull2 = SAWBatch(items=[_item(10, rid=5), _item(10, rid=6), _item(10, rid=7), _item(10, rid=8)])
    src = _FakeSource([pull1, pull2], batch_size=4)
    macro, outcome = asyncio.run(Trainer._fetch_macro_batch(_stub(src)))

    assert len(macro.items) == 4                       # exact target (trimmed from 6)
    assert macro.meta["num_samples"] == 4
    assert macro.meta["effective_rollouts"] == 4
    assert src.pulls == 2                              # backfilled across 2 pulls
    assert abs(macro.meta["max_lag_drop_rate"] - 0.25) < 1e-9   # 2 stale / 8 seen
    assert len(outcome.items) == 6                     # post-max_lag, pre-drop_zero


def test_drop_zero_excluded_from_trainable_kept_in_outcome():
    # All fresh; 2 have zero advantage -> dropped from gradient batch, kept in outcome.
    pull1 = SAWBatch(items=[_item(10, adv=1, rid=1), _item(10, adv=0, rid=2),
                            _item(10, adv=1, rid=3), _item(10, adv=0, rid=4)])
    pull2 = SAWBatch(items=[_item(10, adv=1, rid=5), _item(10, adv=1, rid=6)])
    src = _FakeSource([pull1, pull2], batch_size=3)
    macro, outcome = asyncio.run(Trainer._fetch_macro_batch(_stub(src)))

    assert len(macro.items) == 3                       # 2 + 2 = 4 trainable, trimmed to 3
    assert macro.meta["max_lag_drop_rate"] == 0.0      # nothing stale
    assert len(outcome.items) == 6                     # all fresh kept pre-drop_zero


def test_no_max_lag_single_pull():
    # SFT-like: max_lag None, no preprocess -> one pull fills target, no lag metrics.
    pull = SAWBatch(items=[_item(0, rid=i) for i in range(5)])
    src = _FakeSource([pull], batch_size=5)
    macro, outcome = asyncio.run(Trainer._fetch_macro_batch(_stub(src, max_lag=None, preprocess=None)))

    assert len(macro.items) == 5
    assert src.pulls == 1
    assert "max_lag_drop_rate" not in macro.meta       # no lag tracking when max_lag None


def test_exact_fill_no_overshoot():
    # target=4, one pull of 4 fresh -> exact, single pull, no trim loss.
    pull = SAWBatch(items=[_item(10, rid=i) for i in range(4)])
    src = _FakeSource([pull], batch_size=4)
    macro, _ = asyncio.run(Trainer._fetch_macro_batch(_stub(src)))

    assert len(macro.items) == 4
    assert src.pulls == 1
