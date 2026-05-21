"""
Chat template abstraction for token-in API.

This module provides control over chat template application, enabling
drift-free RL training by ensuring exact token alignment between
the library and inference backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

from ludic.types import Message
from ludic.inference.tool_parser import ToolParseResult, ToolParser

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TemplateResult:
    """
    Result of applying a chat template.

    Contains both the token IDs (canonical representation) and
    the text representation (for debugging/logging).
    """

    prompt_token_ids: List[int]
    prompt_text: str
    stop_token_ids: Optional[List[int]] = None


class ChatTemplate(Protocol):
    """
    Protocol for converting chat messages to token IDs.

    Implementations handle model-specific chat templates and tool formatting.
    This gives Ludic full control over what tokens the model sees,
    eliminating training-inference drift.
    """

    def apply(
        self,
        messages: List[Message],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> TemplateResult:
        """
        Apply the chat template to messages.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Optional tool schemas (OpenAI function calling format).
            add_generation_prompt: Whether to add the assistant turn prefix.

        Returns:
            TemplateResult with token IDs and text representation.
        """
        ...

    def parse_tool_calls(
        self,
        completion_text: str,
    ) -> ToolParseResult:
        """
        Parse tool calls from raw completion text.

        Model-specific parsing to extract tool_calls from the output.
        """
        ...

    def supports_tools(self) -> bool:
        """
        Whether this template can parse tool calls from model output.
        """
        ...


class HFChatTemplate:
    """
    ChatTemplate implementation using HuggingFace tokenizer's apply_chat_template.

    Uses HuggingFace's native tool formatting - the tokenizer's chat template
    already knows how to format tools for each model. We just pass them through.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        *,
        tool_parser: Optional[ToolParser] = None,
        enable_thinking: Optional[bool] = None,
        default_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Args:
            tokenizer: HuggingFace tokenizer with chat_template support.
            tool_parser: Parser for extracting tool calls from model output.
                        Required if using tools.
            enable_thinking: If set, passes enable_thinking to apply_chat_template.
                            Use False to disable chain-of-thought on models that support
                            it (e.g. Qwen3). None means the kwarg is not passed at all.
            default_tools: Tool schemas (OpenAI function-calling format) passed to
                            apply_chat_template when the caller doesn't supply tools
                            explicitly. Required for chat templates that drop role:tool
                            messages when tools= is unset (e.g. Gemma 4).
        """
        self._tokenizer = tokenizer
        self._tool_parser = tool_parser
        self._enable_thinking = enable_thinking
        self._default_tools = list(default_tools or [])

    def apply(
        self,
        messages: List[Message],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> TemplateResult:
        """Apply the chat template to messages using HuggingFace's native formatting."""
        extra_kwargs: Dict[str, Any] = {}
        if self._enable_thinking is not None:
            extra_kwargs["enable_thinking"] = self._enable_thinking

        effective_tools = tools if tools is not None else self._default_tools

        # Use HuggingFace's apply_chat_template directly - it handles
        # model-specific tool formatting automatically
        if effective_tools:
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tools=effective_tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                **extra_kwargs,
            )
        else:
            rendered = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                **extra_kwargs,
            )

        prompt_token_ids = self._normalize_template_output(rendered)
        prompt_text = self._tokenizer.decode(prompt_token_ids)

        return TemplateResult(
            prompt_token_ids=prompt_token_ids,
            prompt_text=prompt_text,
        )

    def _normalize_template_output(self, rendered: Any) -> List[int]:
        if isinstance(rendered, str):
            return self._tokenizer.encode(rendered, add_special_tokens=False)
        if isinstance(rendered, Mapping):
            rendered = rendered["input_ids"]
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if rendered and isinstance(rendered[0], list):
            if len(rendered) != 1:
                raise ValueError("Expected one chat template sequence, got a batch.")
            rendered = rendered[0]
        return [int(token_id) for token_id in rendered]

    def parse_tool_calls(
        self,
        completion_text: str,
    ) -> ToolParseResult:
        """Parse tool calls from completion text using the tool parser."""
        if self._tool_parser:
            return self._tool_parser.parse(completion_text)
        return ToolParseResult(tool_calls=None, parse_error=False)

    def supports_tools(self) -> bool:
        """Whether tool-call parsing is configured."""
        return self._tool_parser is not None
