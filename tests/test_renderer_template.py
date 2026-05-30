from __future__ import annotations

from ludic.inference.renderer_template import Gemma4ChatTemplate


class MockGemmaTokenizer:
    bos_token = "<bos>"

    def __init__(self) -> None:
        self.special = {"<|tool_response>": 9001, "<|turn>": 9002}

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self.special:
            return [self.special[text]]
        return [(ord(ch) % 251) + 1 for ch in text]

    def decode(self, token_ids):
        return "".join(chr((token_id - 1) % 251) for token_id in token_ids)


def test_gemma4_template_stops_on_native_boundaries() -> None:
    template = Gemma4ChatTemplate(MockGemmaTokenizer())

    result = template.apply([{"role": "user", "content": "hi"}])

    assert result.stop_token_ids == [9001, 9002]
    assert "<|turn>model\n<|channel>thought\n" in result.prompt_text


def test_gemma4_parse_native_tool_call_to_canonical_xml() -> None:
    template = Gemma4ChatTemplate(MockGemmaTokenizer())

    text, info = template.parse_completion(
        completion_token_ids=[],
        completion_text=(
            '<|channel>thought\nCHECK IT\n'
            '<|tool_call>call:check_math_solution{solution:<|"|>55<|"|>}'
        ),
    )

    assert info["success"] is True
    assert info["tool_calls"] == 1
    assert "<think>CHECK IT</think>" in text
    assert (
        text.splitlines()[-1]
        == '<tool_call>{"name":"check_math_solution","arguments":{"solution":"55"}}</tool_call>'
    )


def test_gemma4_preserves_prompt_continuation_thought_before_tool_call() -> None:
    template = Gemma4ChatTemplate(MockGemmaTokenizer())

    text, info = template.parse_completion(
        completion_token_ids=[],
        completion_text=(
            "Step 1: Identify the country.\n"
            "Step 2: France's capital is Paris."
            '<|tool_call>call:check_cot_task{answer:<|"|>A<|"|>}'
        ),
    )

    assert info["success"] is True
    assert info["tool_calls"] == 1
    assert (
        "<think>Step 1: Identify the country.\n"
        "Step 2: France's capital is Paris.</think>"
    ) in text
    assert (
        text.splitlines()[-1]
        == '<tool_call>{"name":"check_cot_task","arguments":{"answer":"A"}}</tool_call>'
    )


def test_gemma4_unwraps_xml_think_before_native_tool_call() -> None:
    template = Gemma4ChatTemplate(MockGemmaTokenizer())

    text, info = template.parse_completion(
        completion_token_ids=[],
        completion_text=(
            "<think>\n"
            "Step 1: Identify the country.\n"
            "Step 2: France's capital is Paris.\n"
            "</think>"
            '<|tool_call>call:check_cot_task{answer:<|"|>A<|"|>}'
        ),
    )

    assert info["success"] is True
    assert info["tool_calls"] == 1
    assert text.startswith(
        "<think>Step 1: Identify the country.\n"
        "Step 2: France's capital is Paris.</think>\n"
    )
    assert "<think><think>" not in text
    assert (
        text.splitlines()[-1]
        == '<tool_call>{"name":"check_cot_task","arguments":{"answer":"A"}}</tool_call>'
    )


def test_gemma4_parse_decoded_tool_call_without_special_prefix() -> None:
    template = Gemma4ChatTemplate(MockGemmaTokenizer())

    text, info = template.parse_completion(
        completion_token_ids=[],
        completion_text="call:check_math_solution{solution:63}",
    )

    assert info["success"] is True
    assert info["tool_calls"] == 1
    assert (
        text
        == '<tool_call>{"name":"check_math_solution","arguments":{"solution":63}}</tool_call>'
    )


def test_gemma4_renders_prior_xml_tool_call_as_native() -> None:
    template = Gemma4ChatTemplate(MockGemmaTokenizer())

    result = template.apply(
        [
            {"role": "assistant", "content": '<tool_call>{"name":"check_cot_task","arguments":{"answer":"A"}}</tool_call>'},
            {"role": "tool", "name": "check_cot_task", "content": '{"both":true}'},
        ]
    )

    assert '<|tool_call>call:check_cot_task{answer:<|"|>A<|"|>}' in result.prompt_text
    assert '<|tool_response>response:check_cot_task{value:<|"|>{"both":true}<|"|>}' in result.prompt_text


def test_gemma4_renders_past_thought_channel_before_tool_call() -> None:
    """Past assistant turns with reasoning_content must render the thought channel
    BEFORE the tool call so the model sees a history matching its own generation
    format (think → act). Round-tripping the rendered text through parse_completion
    must recover the original reasoning cleanly (no <channel|> close-marker leak)."""
    template = Gemma4ChatTemplate(MockGemmaTokenizer())
    reasoning = "Need to add 2 and 2."

    result = template.apply(
        [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "check_math_solution",
                            "arguments": {"solution": "4"},
                        },
                    }
                ],
            },
            {"role": "tool", "name": "check_math_solution", "content": '{"is_correct": true}'},
        ],
        add_generation_prompt=False,
    )

    thought_idx = result.prompt_text.find(f"<|channel>thought\n{reasoning}")
    tool_call_idx = result.prompt_text.find("<|tool_call>call:check_math_solution")
    assert thought_idx != -1, "past reasoning_content must be emitted in the thought channel"
    assert tool_call_idx != -1, "tool call must be emitted"
    assert thought_idx < tool_call_idx, "thought channel must precede the tool call in history"
    assert "<channel|>" not in result.prompt_text, (
        "renderer must not emit explicit <channel|> close — _THOUGHT_RE's lookahead "
        "uses the next piece as the boundary, and an explicit close would leak into "
        "parsed reasoning_content on round-trip."
    )

    # Round-trip: the rendered assistant block, parsed back, must recover the reasoning
    # as a <think>…</think> wrapper with no close-marker leak.
    assistant_start = result.prompt_text.find("<|channel>thought")
    assistant_end = result.prompt_text.find("<|tool_response>")
    assistant_block = result.prompt_text[assistant_start:assistant_end].rstrip()
    text, info = template.parse_completion(
        completion_token_ids=[],
        completion_text=assistant_block,
    )
    assert info["success"] is True
    assert info["tool_calls"] == 1
    assert f"<think>{reasoning}</think>" in text
    assert "<channel|>" not in text
