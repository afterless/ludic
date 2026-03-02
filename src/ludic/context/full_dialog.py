from __future__ import annotations
from typing import List
from ludic.context.base import ContextStrategy
from ludic.inference.client import ChatResponse
from ludic.types import Message, Observation, Info

class FullDialog(ContextStrategy):
    def on_env_reset(self, obs: Observation, info: Info) -> None:
        self._messages.append({"role": "user", "content": obs})

    def on_before_act(self) -> List[Message]:
        return list(self._messages)

    def on_after_act(self, assistant_response: ChatResponse) -> None:
        assistant_text = assistant_response.text
        self._messages.append({"role": "assistant", "content": assistant_text})

    def on_after_step(self, next_obs: Observation, info: Info) -> None:
        if "tool_name" in info:
            self.add_tool_result(
                str(info.get("tool_call_id", "")),
                str(info["tool_name"]),
                next_obs,
            )
        else:
            self._messages.append({"role": "user", "content": next_obs})
