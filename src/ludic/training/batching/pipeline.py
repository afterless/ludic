from __future__ import annotations

import pickle
import logging
import asyncio
import time
from typing import List, Callable, Optional
from dataclasses import replace

from ludic.training.types import (
    BatchSource, 
    SAWBatch, 
    SAWItem, 
    RolloutRequest, 
    CreditAssigner,
)
from ludic.inference.client import VersionedClient
from .rollout_engine import RolloutEngine

logger = logging.getLogger(__name__)

class PipelineBatchSource(BatchSource):
    """
    Trainer-side component.
    Pulls completed, pre-processed SAWItems from a Redis queue.
    
    This decouples the Trainer from the generation latency. The Trainer
    simply blocks on the queue until a macro-batch is assembled.
    """
    def __init__(
        self, 
        redis_url: str, 
        queue_key: str = "ludic_queue", 
        batch_size: int = 4,
        poll_timeout: int = 1
    ):
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PipelineBatchSource requires the 'redis' package. Install with: uv sync --extra pipelinerl"
            ) from exc

        self.r = redis.from_url(redis_url)
        self.queue_key = queue_key
        self.batch_size = batch_size
        self.poll_timeout = poll_timeout

    async def next_batch(self) -> SAWBatch:
        """
        Blocking fetch from Redis. Returns a SAWBatch once enough items are pulled.
        """
        items: List[SAWItem] = []
        
        while len(items) < self.batch_size:
            # BLPOP blocks until data is available, preventing busy-loops.
            # We use a short timeout to allow the loop to check for exit signals/cancellation.
            raw_data = self.r.blpop(self.queue_key, timeout=self.poll_timeout)
            
            if raw_data:
                # raw_data is tuple (queue_name, payload_bytes)
                payload = raw_data[1]
                try:
                    # The Actor has already done the tokenization and credit assignment.
                    # We just deserialize the final training sample.
                    saw_item: SAWItem = pickle.loads(payload)
                    items.append(saw_item)
                except Exception as e:
                    logger.error(f"Failed to deserialize SAWItem from Redis: {e}")
            else:
                # Timeout occurred, loop again or yield to event loop
                await asyncio.sleep(0.01)
                continue

        # Calculate basic batch stats for logging
        avg_reward = 0.0
        if items:
            total_r = sum(it.meta.get("total_reward", 0.0) for it in items)
            avg_reward = total_r / len(items)

        meta = {
            "target_rollouts": len(items),
            "num_samples": len(items),
            "avg_total_reward": avg_reward,
            "source": "pipeline_redis"
        }

        return SAWBatch(items=items, meta=meta)


# -------------------------------------------------------------------------
# The Actor Loop
# -------------------------------------------------------------------------

async def run_pipeline_actor(
    engine: RolloutEngine,
    requests_fn: Callable[[], List[RolloutRequest]],
    credit_assigner: CreditAssigner,
    redis_url: str,
    queue_key: str = "ludic_queue",
    max_steps: int = 10,
    concurrency: int = 4,
    client: Optional[VersionedClient] = None,
    pipeline_depth: int = 1,
):
    """
    Actor-side component. Runs an infinite loop to:
    1. Fetch intent via requests_fn.
    2. Poll the runtime (via client) for the current policy version.
    3. Tag requests with that version.
    4. Delegate generation AND collation to the shared Engine (one SAWItem per turn).
    5. Push the resulting SAWItems to Redis.

    pipeline_depth>1 runs that many batches concurrently so the cores stay fed through a batch's
    straggler. Each batch keeps one policy version, so depth never mixes versions within a batch.
    """
    try:
        import redis  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "run_pipeline_actor requires the 'redis' package. Install with: uv sync --extra pipelinerl"
        ) from exc

    r_conn = redis.from_url(redis_url)
    logger.info(f"Pipeline Actor connected to Redis at {redis_url}")

    total_pushed = 0

    async def _gen_and_push(batch_num: int) -> int:
        requests = requests_fn()
        if not requests:
            return 0
        num_episodes = sum(r.num_episodes for r in requests)
        current_ver = await client.get_policy_version() if client else 0
        tagged_requests = [
            replace(req, meta={**req.meta, "policy_version": current_ver}) for req in requests
        ]
        logger.info(
            f"[batch {batch_num}] Generating {num_episodes} episodes from {len(requests)} "
            f"requests (concurrency={concurrency}, max_steps={max_steps}, v{current_ver})"
        )
        batch_start = time.monotonic()
        try:
            saw_batch = await engine.generate_batch(
                requests=tagged_requests,
                max_steps=max_steps,
                credit_assigner=credit_assigner,
                concurrency=concurrency,
            )
        except Exception as e:
            logger.error(f"[batch {batch_num}] generation error: {e}", exc_info=True)
            return 0
        elapsed = time.monotonic() - batch_start
        if not saw_batch.items:
            logger.info(f"[batch {batch_num}] Empty batch after {elapsed:.1f}s")
            return 0
        logger.info(
            f"[batch {batch_num}] Generated {len(saw_batch.items)} SAWItems in {elapsed:.1f}s "
            f"(avg_reward={saw_batch.meta.get('avg_total_reward', 0):.3f})"
        )
        pipe = r_conn.pipeline()
        for item in saw_batch.items:
            pipe.rpush(queue_key, pickle.dumps(item))
        try:
            pipe.execute()
        except redis.RedisError as e:
            logger.error(f"[batch {batch_num}] Redis pipeline error: {e}")
            return 0
        logger.info(
            f"[batch {batch_num}] Pushed {len(saw_batch.items)} items to Redis (v{current_ver})"
        )
        return len(saw_batch.items)

    depth = max(1, pipeline_depth)
    batch_num = 0
    inflight: set[asyncio.Task[int]] = set()
    while True:
        while len(inflight) < depth:
            batch_num += 1
            inflight.add(asyncio.create_task(_gen_and_push(batch_num)))
        done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
        round_pushed = 0
        for task in done:
            try:
                round_pushed += task.result()
            except Exception as e:
                logger.error(f"batch task crashed: {e}", exc_info=True)
        total_pushed += round_pushed
        if round_pushed == 0:
            await asyncio.sleep(0.5)
