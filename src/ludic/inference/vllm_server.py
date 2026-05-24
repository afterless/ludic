import asyncio
import os
import signal
import sys
from argparse import Namespace
from typing import Any, Set, Optional
from collections.abc import Coroutine

# Use V1 engine explicitly.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"

import torch
import uvloop
from fastapi import FastAPI, Request
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit
from vllm.tokenizers import cached_tokenizer_from_config

# V1 logits-processor interface
from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor as V1LogitsProcessor,
    BatchUpdate,
    MoveDirectionality,
)

# ---------------------------------------------------------------------------
# Global state for weight updates & background tasks
# ---------------------------------------------------------------------------

weight_update_lock = asyncio.Lock()

background_tasks: Set[asyncio.Task[Any]] = set()

RUNTIME_VERSION: int = 0
RUNTIME_VERSION_LOCK = asyncio.Lock()


def create_background_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Create an async task and track it so we can wait/cancel on shutdown."""
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task


# ---------------------------------------------------------------------------
# Custom logits processor: inject "</think>" after N tokens (pure V1)
# ---------------------------------------------------------------------------


class GlobalThinkProcessor(V1LogitsProcessor):
    """
    Single V1 logits processor instance per worker.

    For each request in the batch:
      - On BatchUpdate.added, we inspect SamplingParams.extra_args["max_think"].
      - If present and > 0, we remember:
          * a live reference to that request's output_ids (list[int])
          * its trigger_len
      - On apply(logits), we walk each row and, if that request is within
        its think-window, we force the next token of the '</think>' sequence.

    In other words: we force the model to emit '</think>' after a
    user-chosen number of generated tokens.
    """

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        # Per-request state: req_idx -> {"output_ids": list[int], "trigger_len": int}
        self.req_state: dict[int, dict[str, Any]] = {}
        # Pre-tokenized think_ids injected into vllm_config BEFORE engine spawn
        self.think_ids = vllm_config.additional_config.get("think_ids", [])

    # ---- required by V1 interface ----

    def is_argmax_invariant(self) -> bool:
        # We overwrite logits and hence argmax, so no.
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        """
        Called whenever the persistent batch changes (add/remove/move),
        *before* each forward pass.
        """
        if batch_update is None:
            return

        # 1) Handle removals
        for ridx in batch_update.removed:
            if ridx in self.req_state:
                self.req_state.pop(ridx, None)

        # 2) Handle additions
        for (req_idx, params, prompt_ids, output_ids) in batch_update.added:
            assert isinstance(params, SamplingParams)
            extra_args = getattr(params, "extra_args", None)

            trigger_len = None
            if isinstance(extra_args, dict):
                trigger_len = extra_args.get("max_think")

            if not isinstance(trigger_len, int) or trigger_len <= 0:
                self.req_state.pop(req_idx, None)
                continue

            self.req_state[req_idx] = {
                "output_ids": output_ids,
                "trigger_len": trigger_len,
            }

        # 3) Handle moves
        for (src, dst, direction) in batch_update.moved:
            if direction == MoveDirectionality.UNIDIRECTIONAL:
                state = self.req_state.pop(src, None)
                if state is not None:
                    self.req_state[dst] = state
            else:
                s1 = self.req_state.get(src)
                s2 = self.req_state.get(dst)
                if s1 is not None or s2 is not None:
                    self.req_state[src], self.req_state[dst] = s2, s1

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [batch_size, vocab_size]
        We mutate rows in-place where we want to force '</think>'.
        """
        if not self.think_ids or not self.req_state:
            return logits

        batch_size = logits.shape[0]
        think_ids = self.think_ids

        for req_idx in range(batch_size):
            state = self.req_state.get(req_idx)
            if state is None:
                continue

            output_ids: list[int] = state["output_ids"]
            trigger_len: int = state["trigger_len"]

            seq_len = len(output_ids)
            pos = seq_len - trigger_len

            if pos < 0 or pos >= len(think_ids):
                continue

            forced_id = think_ids[pos]

            row = logits[req_idx]
            row.fill_(float("-inf"))
            row[forced_id] = 0.0

        return logits


# ---------------------------------------------------------------------------
# Server / app setup
# ---------------------------------------------------------------------------


def _thinking_close_ids(tokenizer) -> list[int]:
    """Token id(s) that close the model's reasoning channel, used by
    GlobalThinkProcessor to force-stop thinking at the max_think budget.

    Model-agnostic: prefer the model's NATIVE end-of-reasoning marker that maps
    to a single special token (e.g. Gemma's `<channel|>` closes the thought
    channel = token 101; Qwen/Hermes `</think>`). We force ONLY the close token
    and inject NO natural-language text — the previous hardcoded
    "Okay, time is up ... </think>" was Qwen-only syntax (mere text for Gemma, so
    it never closed the channel) and its lowercase words corrupted
    reasoning-format side tasks (e.g. uppercase-CoT) by polluting reasoning_content.
    """
    for marker in ("<channel|>", "</think>", "<|im_end|>", "<|eot_id|>"):
        try:
            ids = tokenizer.encode(marker, add_special_tokens=False)
        except Exception:
            continue
        if len(ids) == 1:  # native single special token for this chat template
            return ids
    # Fallback: legacy textual close when no native single-token marker exists.
    return tokenizer.encode("</think>", add_special_tokens=False)


async def run_server(args: Namespace) -> None:
    sock_addr = (args.host or "0.0.0.0", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    # ----------------------------------------------------------------------
    # 1) Build engine_args from CLI and inject our extension + logits proc
    # ----------------------------------------------------------------------
    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Wire our GlobalThinkProcessor into the engine-wide logits processor list.
    # If user already passed --logits-processors, append ours.
    think_proc = "ludic.inference.vllm_server:GlobalThinkProcessor"
    if engine_args.logits_processors:
        if think_proc not in engine_args.logits_processors:
            engine_args.logits_processors.append(think_proc)
    else:
        engine_args.logits_processors = [think_proc]

    # ----------------------------------------------------------------------
    # 2) Build VllmConfig from engine_args (now containing logits_processors)
    # ----------------------------------------------------------------------
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    # --------------------------------------------------------------
    # 3) Pre-tokenize '</think>' using the *same* tokenizer config
    #    the engine will use. This is controller-side only.
    # --------------------------------------------------------------
    try:
        tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        think_ids = _thinking_close_ids(tokenizer)
        vllm_config.additional_config["think_ids"] = think_ids
    except Exception as e:
        raise RuntimeError(
            f"Failed to resolve the reasoning-close token for model "
            f"{vllm_config.model_config.model}: {e}"
        ) from e

    # ----------------------------------------------------------------------
    # 4) Build AsyncLLM engine from the prepared config.
    #    At this point, vllm_config already knows about GlobalThinkProcessor
    #    and think_ids. V1 will instantiate our logits processor on workers.
    # ----------------------------------------------------------------------
    engine = AsyncLLMEngine.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        enable_log_requests=engine_args.enable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )

    app: FastAPI = build_app(args)

    # ------------------------ control-plane endpoints ---------------------

    # TODO: override
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runtime_version")
    async def runtime_version() -> dict[str, int]:
        return {"version": RUNTIME_VERSION}

    @app.post("/update_lora")
    async def update_lora(request: Request) -> dict[str, str]:
        """Hot-swap the served 'policy' LoRA adapter from a PEFT dir on shared FS.
        Body: {lora_name, lora_path, version}. Needs --enable-lora + VLLM_ALLOW_RUNTIME_LORA_UPDATING."""
        from vllm.entrypoints.serve.lora.protocol import (
            LoadLoRAAdapterRequest,
            UnloadLoRAAdapterRequest,
        )

        data = await request.json()
        name = data.get("lora_name", "policy")
        path = data.get("lora_path")
        forced_version = data.get("version")
        if not path:
            return {"status": "error", "detail": "missing 'lora_path'"}

        sm = app.state.openai_serving_models
        await engine.pause_generation(
            wait_for_inflight_requests=True,
            clear_cache=False,
        )
        async with weight_update_lock:
            try:
                # Unload the previous version (no-op/ignored on the first sync).
                try:
                    await sm.unload_lora_adapter(UnloadLoRAAdapterRequest(lora_name=name))
                except Exception:
                    pass
                await sm.load_lora_adapter(
                    LoadLoRAAdapterRequest(lora_name=name, lora_path=path)
                )
                await engine.reset_prefix_cache()
                global RUNTIME_VERSION
                async with RUNTIME_VERSION_LOCK:
                    if forced_version is not None:
                        RUNTIME_VERSION = int(forced_version)
                    else:
                        RUNTIME_VERSION += 1
            finally:
                await engine.resume_generation()
        return {"status": "ok"}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache() -> dict[str, str]:
        """
        Reset any KV/prefix caches on the engine.
        """
        create_background_task(engine.reset_prefix_cache())
        return {"status": "ok"}

    @app.post("/get_num_background_tasks")
    async def get_num_background_tasks() -> dict[str, int]:
        return {"num_background_tasks": len(background_tasks)}

    @app.post("/shutdown")
    async def shutdown() -> dict[str, str]:
        # Self-SIGTERM after the response flushes; trips the graceful handler above.
        loop = asyncio.get_running_loop()
        loop.call_later(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
        return {"status": "shutting down"}

    # ------------------------ start HTTP server --------------------------

    # vLLM 0.13: engine_client exposes vllm_config as an attribute
    print(engine.vllm_config)

    # vLLM 0.13: init_app_state(engine_client, state, args)
    await init_app_state(engine, app.state, args)

    shutdown_task = await serve_http(
        app,
        sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )
    await shutdown_task

    # graceful shutdown of background tasks
    for task in list(background_tasks):
        task.cancel()
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

    sock.close()


def main() -> None:
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-compatible server with weight synchronization"
    )
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--batch-invariant",
        action="store_true",
        help="Enable vLLM batch-invariant kernels (sets VLLM_BATCH_INVARIANT=1).",
    )
    argv = sys.argv[1:]
    # vLLM can silently override sampling params using the model's Hugging Face
    # `generation_config` unless `--generation-config vllm` is set. Defaulting
    # to `vllm` makes Ludic's SamplingParams the source of truth.
    if not any(a == "--generation-config" or a.startswith("--generation-config=") for a in argv):
        argv = [*argv, "--generation-config", "vllm"]
    args = parser.parse_args(argv)
    assert args is not None
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    validate_parsed_serve_args(args)
    print(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
