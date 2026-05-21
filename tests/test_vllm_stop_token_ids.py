from __future__ import annotations

import asyncio

from ludic.inference.request import ReturnSpec, TokenCompletionRequest
from ludic.inference.sampling import SamplingParams
from ludic.inference.vllm_client import VLLMChatClient
from ludic.types import ChatResponse


class FakeChoice:
    text = "ok"
    finish_reason = "stop"
    token_ids = [3, 4]
    logprobs = None


class FakeResponse:
    choices = [FakeChoice()]
    prompt_token_ids = [1, 2]

    def model_dump(self, exclude_none: bool = True):
        return {}


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


def test_vllm_client_forwards_stop_token_ids(monkeypatch) -> None:
    monkeypatch.setattr(VLLMChatClient, "_check_server", lambda self, timeout: None)
    client = VLLMChatClient(host="127.0.0.1", port=1234)
    completions = FakeCompletions()
    client._async_client.completions = completions

    async def run():
        return await client.complete_tokens(
            TokenCompletionRequest(
                model="mock",
                prompt_token_ids=[1, 2],
                stop_token_ids=[9001, 9002],
                sampling=SamplingParams(max_tokens=8),
                return_=ReturnSpec.for_eval(return_token_ids=True),
            )
        )

    resp, _ = asyncio.run(run())

    assert isinstance(resp, ChatResponse)
    assert completions.kwargs["extra_body"]["stop_token_ids"] == [9001, 9002]
