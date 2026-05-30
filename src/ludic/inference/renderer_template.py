"""Renderer-backed chat templates for token-in inference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ludic.inference.chat_template import TemplateResult
from ludic.inference.tool_parser import ToolParseResult
from ludic.types import Message


def _content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif part.get("type") == "thinking":
                    parts.append(str(part.get("thinking", "")))
        return "".join(parts)
    return str(content)


def _strip_xml_tool_instructions(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            continue
        if line.strip() == "Tool-call syntax:":
            skip_next = True
            continue
        if "<tool_call>" in line or "</tool_call>" in line:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _json_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = {"name": name, "arguments": arguments}
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


def _openai_tool_to_simple(tool: Dict[str, Any]) -> Dict[str, Any]:
    function = tool.get("function", tool)
    return {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {}),
    }


class PrimeRendererChatTemplate:
    """Adapter from Prime Intellect renderers to Ludic's ChatTemplate protocol."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        renderer_name: str = "auto",
        default_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        try:
            from renderers import create_renderer
        except ImportError as exc:
            raise ImportError(
                "renderer_name requires the Prime Intellect `renderers` package. "
                "Install the project with the gpu extra or add `renderers` to the environment."
            ) from exc

        self._tokenizer = tokenizer
        self._renderer = create_renderer(tokenizer, renderer=renderer_name)
        self._default_tools = list(default_tools or [])

    def apply(
        self,
        messages: List[Message],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> TemplateResult:
        effective_tools = tools if tools is not None else self._default_tools
        prompt_ids = self._render_ids(
            messages,
            tools=effective_tools,
            add_generation_prompt=add_generation_prompt,
        )
        return TemplateResult(
            prompt_token_ids=prompt_ids,
            prompt_text=self._tokenizer.decode(prompt_ids),
            stop_token_ids=self._stop_token_ids(),
        )

    def _render_ids(
        self,
        messages: List[Message],
        *,
        tools: Optional[List[Dict[str, Any]]],
        add_generation_prompt: bool,
    ) -> List[int]:
        try:
            return list(
                self._renderer.render_ids(
                    messages,
                    tools=tools or None,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        except TypeError:
            return list(
                self._renderer.render_ids(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                )
            )

    def _stop_token_ids(self) -> Optional[List[int]]:
        getter = getattr(self._renderer, "get_stop_token_ids", None)
        if not callable(getter):
            return None
        stop_ids = getter()
        if not stop_ids:
            return None
        return [int(token_id) for token_id in stop_ids]

    def parse_completion(
        self,
        *,
        completion_token_ids: List[int],
        completion_text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        parser = getattr(self._renderer, "parse_response", None)
        if not callable(parser):
            return completion_text, {"success": False, "reason": "no_parse_response"}

        try:
            parsed = parser(list(completion_token_ids))
        except Exception as exc:
            return completion_text, {
                "success": False,
                "reason": "parse_exception",
                "error": str(exc),
            }

        content = getattr(parsed, "content", None)
        if content is None and isinstance(parsed, dict):
            content = parsed.get("content")
        reasoning = getattr(parsed, "reasoning_content", None)
        if reasoning is None and isinstance(parsed, dict):
            reasoning = parsed.get("reasoning_content")
        tool_calls = getattr(parsed, "tool_calls", None)
        if tool_calls is None and isinstance(parsed, dict):
            tool_calls = parsed.get("tool_calls")

        pieces = []
        if reasoning:
            pieces.append(f"<think>{reasoning}</think>")
        if content:
            pieces.append(_content_to_text(content))
        if tool_calls:
            pieces.extend(_canonical_tool_calls(tool_calls))

        if not pieces:
            return completion_text, {"success": True, "empty": True}
        return "\n".join(pieces), {"success": True, "tool_calls": bool(tool_calls)}

    def parse_tool_calls(self, completion_text: str) -> ToolParseResult:
        return ToolParseResult(tool_calls=None, parse_error=False)

    def supports_tools(self) -> bool:
        return False


def _canonical_tool_calls(tool_calls: object) -> List[str]:
    canonical = []
    if not isinstance(tool_calls, list):
        return canonical
    for call in tool_calls:
        function = getattr(call, "function", None)
        if function is None and isinstance(call, dict):
            function = call.get("function", {})
        name = getattr(function, "name", None)
        if name is None and isinstance(function, dict):
            name = function.get("name")
        arguments = getattr(function, "arguments", None)
        if arguments is None and isinstance(function, dict):
            arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if name:
            canonical.append(_json_tool_call(str(name), dict(arguments or {})))
    return canonical


@dataclass(frozen=True)
class Gemma4ParseResult:
    text: str
    info: Dict[str, Any]


class Gemma4ChatTemplate:
    """Text-only Gemma 4 renderer adapter based on vLLM's public tool template."""

    _TOOL_CALL_RE = re.compile(
        r"(?:<\|tool_call>)?call:(?P<name>[A-Za-z_][\w.-]*)\{(?P<args>.*?)\}(?=<\|tool_call\||<\|tool_response>|<\|turn>|$)",
        re.DOTALL,
    )
    _THOUGHT_RE = re.compile(
        r"<\|channel>thought\n(?P<thought>.*?)(?=<\|tool_call>|<\|turn>|$)",
        re.DOTALL,
    )
    _XML_TOOL_RE = re.compile(r"<tool_call>\s*(?P<payload>.*?)\s*</tool_call>", re.DOTALL)

    def __init__(
        self,
        tokenizer: Any,
        *,
        default_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._default_tools = [_openai_tool_to_simple(tool) for tool in default_tools or []]

    def apply(
        self,
        messages: List[Message],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> TemplateResult:
        effective_tools = [_openai_tool_to_simple(tool) for tool in tools or []]
        if not effective_tools:
            effective_tools = self._default_tools
        prompt_text = self._render_messages(
            messages,
            tools=effective_tools,
            add_generation_prompt=add_generation_prompt,
        )
        return TemplateResult(
            prompt_token_ids=self._encode(prompt_text),
            prompt_text=prompt_text,
            stop_token_ids=self._stop_token_ids(),
        )

    def _render_messages(
        self,
        messages: List[Message],
        *,
        tools: List[Dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        out = [getattr(self._tokenizer, "bos_token", None) or ""]
        remaining = list(messages)

        system_parts = []
        if remaining and remaining[0].get("role") == "system":
            system_parts.append(_content_to_text(remaining.pop(0).get("content")).strip())
        if tools:
            system_parts.extend(self._render_tool_spec(tool) for tool in tools)
        if system_parts:
            out.append("<|turn>system\n")
            out.append("\n".join(part for part in system_parts if part))
            out.append("\n")

        for message in remaining:
            role = message.get("role", "user")
            if role == "assistant":
                out.append("<|turn>model\n")
                out.append(self._render_assistant_message(message))
                out.append("\n")
            elif role == "tool":
                name = str(message.get("name") or "unknown")
                out.append(self._render_tool_response(name, _content_to_text(message.get("content"))))
            else:
                out.append(f"<|turn>{role}\n")
                content = _content_to_text(message.get("content")).strip()
                if role == "user":
                    content = _strip_xml_tool_instructions(content)
                out.append(content)
                out.append("\n")

        if add_generation_prompt:
            out.append("<|turn>model\n<|channel>thought\n")
        return "".join(out)

    def _render_tool_spec(self, tool: Dict[str, Any]) -> str:
        description = str(tool.get("description") or "")
        parameters = tool.get("parameters") or {}
        return (
            f"<|tool>declaration:{tool.get('name', '')}"
            f"{{description:<|\"|>{description}<|\"|>,"
            f"parameters:{json.dumps(parameters, separators=(',', ':'))}}}"
        )

    def _render_assistant_message(self, message: Message) -> str:
        # Past assistant turns must include the thought channel so multi-turn history
        # matches what the model emits naturally. No explicit <channel|> close — the
        # next piece (tool_call envelope or content) serves as the implicit boundary,
        # matching _THOUGHT_RE's lookahead so round-tripping recovers the same reasoning.
        pieces: List[str] = []
        if reasoning := message.get("reasoning_content"):
            pieces.append(f"<|channel>thought\n{reasoning}")
        content = _content_to_text(message.get("content"))
        if tool_calls := message.get("tool_calls"):
            pieces.append("".join(self._render_openai_tool_call(call) for call in tool_calls))
        elif xml_call := self._tool_call_from_xml(content):
            pieces.append(self._render_openai_tool_call(xml_call))
        else:
            pieces.append(content.strip())
        return "".join(pieces)

    def _render_openai_tool_call(self, call: Dict[str, Any]) -> str:
        function = call.get("function", call)
        name = str(function.get("name", ""))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        arg_str = self._format_args(dict(arguments or {}))
        return f"<|tool_call>call:{name}{{{arg_str}}}"

    def _render_tool_response(self, name: str, response: str) -> str:
        return f"<|tool_response>response:{name}{{value:{self._quote(response)}}}"

    def _tool_call_from_xml(self, content: str) -> Optional[Dict[str, Any]]:
        match = self._XML_TOOL_RE.search(content)
        if not match:
            return None
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            return None
        return {
            "function": {
                "name": payload.get("name", ""),
                "arguments": payload.get("arguments", {}),
            }
        }

    def _format_args(self, arguments: Dict[str, Any]) -> str:
        return ",".join(
            f"{key}:{self._format_value(value)}" for key, value in sorted(arguments.items())
        )

    def _format_value(self, value: Any) -> str:
        if isinstance(value, str):
            return self._quote(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(value, separators=(",", ":"))

    def _quote(self, value: Any) -> str:
        return f"<|\"|>{str(value)}<|\"|>"

    def _encode(self, text: str) -> List[int]:
        return [int(token_id) for token_id in self._tokenizer.encode(text, add_special_tokens=False)]

    def _stop_token_ids(self) -> Optional[List[int]]:
        ids = []
        for text in ("<|tool_response>", "<|turn>"):
            token_ids = self._encode(text)
            if len(token_ids) == 1:
                ids.append(token_ids[0])
        return ids or None

    def parse_completion(
        self,
        *,
        completion_token_ids: List[int],
        completion_text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        parsed = self._parse_native_completion(completion_text)
        return parsed.text, parsed.info

    def _parse_native_completion(self, text: str) -> Gemma4ParseResult:
        thought = ""
        if match := self._THOUGHT_RE.search(text):
            thought = self._strip_xml_think_wrappers(match.group("thought"))

        canonical_calls = []
        parse_errors = []
        tool_matches = list(self._TOOL_CALL_RE.finditer(text))
        if not thought and tool_matches:
            thought = self._extract_prompt_continuation_thought(
                text[: tool_matches[0].start()]
            )

        for match in tool_matches:
            name = match.group("name")
            try:
                args = self._parse_args(match.group("args"))
                canonical_calls.append(_json_tool_call(name, args))
            except ValueError as exc:
                parse_errors.append(str(exc))

        if canonical_calls:
            pieces = []
            if thought:
                pieces.append(f"<think>{thought}</think>")
            pieces.extend(canonical_calls)
            return Gemma4ParseResult(
                text="\n".join(pieces),
                info={
                    "success": not parse_errors,
                    "tool_calls": len(canonical_calls),
                    "parse_errors": parse_errors,
                },
            )

        visible = self._THOUGHT_RE.sub("", text).strip()
        if thought:
            visible = f"<think>{thought}</think>\n{visible}".strip()
        return Gemma4ParseResult(
            text=visible,
            info={
                "success": not parse_errors,
                "tool_calls": 0,
                "parse_errors": parse_errors,
            },
        )

    def _extract_prompt_continuation_thought(self, prefix: str) -> str:
        """Recover thought text generated after a prompt-side thought channel.

        The prompt ends with ``<|channel>thought\n``. vLLM completions usually do
        not include prompt tokens, so Gemma can return ``reasoning<|tool_call>``
        rather than ``<|channel>thought\nreasoning<|tool_call>``. Treat that
        prefix as the thought content instead of dropping it while canonicalizing
        the native tool call.
        """
        cleaned = prefix.strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"^<\|channel>thought\n?", "", cleaned).strip()
        cleaned = re.sub(r"^<\|channel>\w+\n?", "", cleaned).strip()
        cleaned = re.sub(r"<\|tool_call>\s*$", "", cleaned).strip()
        return self._strip_xml_think_wrappers(cleaned)

    def _strip_xml_think_wrappers(self, text: str) -> str:
        cleaned = text.strip()
        while cleaned.startswith("<think>") and cleaned.endswith("</think>"):
            cleaned = cleaned[len("<think>") : -len("</think>")].strip()
        return cleaned

    def _parse_args(self, arg_text: str) -> Dict[str, Any]:
        args = {}
        for item in self._split_top_level(arg_text):
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid Gemma tool argument: {item!r}")
            key, value = item.split(":", 1)
            args[key.strip()] = self._parse_value(value.strip())
        return args

    def _split_top_level(self, text: str) -> List[str]:
        parts = []
        start = 0
        depth = 0
        in_quote = False
        i = 0
        while i < len(text):
            if text.startswith('<|"|>', i):
                in_quote = not in_quote
                i += len('<|"|>')
                continue
            char = text[i]
            if not in_quote:
                if char in "[{":
                    depth += 1
                elif char in "]}":
                    depth -= 1
                elif char == "," and depth == 0:
                    parts.append(text[start:i].strip())
                    start = i + 1
            i += 1
        parts.append(text[start:].strip())
        return parts

    def _parse_value(self, value: str) -> Any:
        if value.startswith('<|"|>') and value.endswith('<|"|>'):
            return value[len('<|"|>') : -len('<|"|>')]
        if value == "true":
            return True
        if value == "false":
            return False
        if value == "null":
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def parse_tool_calls(self, completion_text: str) -> ToolParseResult:
        parsed = self._parse_native_completion(completion_text)
        return ToolParseResult(
            tool_calls=None,
            parse_error=not parsed.info.get("success", False),
        )

    def supports_tools(self) -> bool:
        return False
