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
