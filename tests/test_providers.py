"""JSON salvage for local models.

Every failure mode here was observed from a real small model. These are the
cases that decide whether a 3B model can drive the extraction backends at all.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from arena.providers import coerce, extract_json, schema_instruction, strip_reasoning


class Verdictish(BaseModel):
    grade: str
    reason: str


class TestStripReasoning:
    def test_removes_think_block(self) -> None:
        raw = "<think>Let me consider...\nmaybe</think>\nThe answer is Paris."
        assert strip_reasoning(raw) == "The answer is Paris."

    def test_leaves_clean_text_alone(self) -> None:
        assert strip_reasoning("The answer is Paris.") == "The answer is Paris."


class TestExtractJson:
    def test_plain_object(self) -> None:
        assert extract_json('{"a": 1}') == '{"a": 1}'

    def test_markdown_fence(self) -> None:
        assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_leading_prose(self) -> None:
        got = extract_json('Sure! Here is the JSON:\n{"a": 1}\nHope that helps.')
        assert got == '{"a": 1}'

    def test_nested_objects(self) -> None:
        blob = '{"a": {"b": {"c": 1}}}'
        assert extract_json(f"noise {blob} noise") == blob

    def test_brace_inside_a_string_does_not_end_the_object(self) -> None:
        blob = '{"reason": "the value was {redacted}"}'
        assert extract_json(blob) == blob

    def test_escaped_quote_inside_a_string(self) -> None:
        blob = '{"reason": "he said \\"no\\" firmly"}'
        assert extract_json(blob) == blob

    def test_no_json_at_all(self) -> None:
        assert extract_json("I cannot help with that.") is None


class TestCoerce:
    def test_reasoning_model_output(self) -> None:
        raw = (
            "<think>The reference says Chapel Hill and so does the answer.</think>\n"
            '```json\n{"grade": "correct", "reason": "matches"}\n```'
        )
        got = coerce(raw, Verdictish)
        assert got is not None and got.grade == "correct"

    def test_returns_none_on_schema_mismatch(self) -> None:
        assert coerce('{"unrelated": true}', Verdictish) is None

    def test_returns_none_on_malformed_json(self) -> None:
        assert coerce('{"grade": "correct", }', Verdictish) is None


class TestSchemaInstruction:
    def test_includes_the_field_names(self) -> None:
        text = schema_instruction(Verdictish)
        assert "grade" in text and "reason" in text
        assert "json" in text.lower()


class TestOpenAICompatProvider:
    """Repair path, with the HTTP call faked."""

    def _provider(self, replies: list[str]):
        pytest.importorskip("openai")
        from arena.providers import Completion, OpenAICompatProvider

        provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
        provider.base_url = "http://fake"
        calls: list[str] = []

        def fake_chat(*, model, prompt, system, max_tokens, json_mode=False, schema=None):
            calls.append(prompt)
            return Completion(text=replies[len(calls) - 1], input_tokens=10, output_tokens=5)

        provider._chat = fake_chat
        return provider, calls

    def test_first_try_success_costs_one_call(self) -> None:
        provider, calls = self._provider(['{"grade": "correct", "reason": "ok"}'])
        parsed, usage = provider.parse(
            model="m", prompt="p", system=None, schema=Verdictish, max_tokens=100
        )
        assert parsed.grade == "correct"
        assert usage.calls == 1 and len(calls) == 1

    def test_repair_round_trip_is_billed(self) -> None:
        provider, calls = self._provider(
            ["I think it is correct.", '{"grade": "correct", "reason": "ok"}']
        )
        parsed, usage = provider.parse(
            model="m", prompt="p", system=None, schema=Verdictish, max_tokens=100
        )
        assert parsed.grade == "correct"
        assert usage.calls == 2
        assert usage.input_tokens == 20, "both attempts must be counted"
        assert "not valid JSON" in calls[1]

    def test_gives_up_loudly_after_the_repair(self) -> None:
        provider, _ = self._provider(["nope", "still nope"])
        with pytest.raises(ValueError, match="repair attempt"):
            provider.parse(
                model="m", prompt="p", system=None, schema=Verdictish, max_tokens=100
            )


class TestOpenAICompatToolCalls:
    """The transport the `loop` orchestrator runs on when the model is local.

    Ollama and friends return tool arguments as a JSON *string*, and small
    models get that string wrong in exactly the ways they get structured output
    wrong. Same salvage, same reason.
    """

    class FakeFunction:
        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    class FakeToolCall:
        def __init__(self, id, name, arguments) -> None:
            self.id = id
            self.function = TestOpenAICompatToolCalls.FakeFunction(name, arguments)

    def _provider(self, tool_calls, content=None):
        pytest.importorskip("openai")
        from arena.providers import OpenAICompatProvider

        provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
        provider.base_url = "http://fake"
        sent: dict = {}

        class FakeMessage:
            def __init__(self):
                self.content = content
                self.tool_calls = tool_calls

        class FakeResponse:
            def __init__(self):
                self.choices = [type("C", (), {"message": FakeMessage()})()]
                self.usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 3})()

        class FakeCompletions:
            def create(self, **kwargs):
                sent.update(kwargs)
                return FakeResponse()

        provider.client = type(
            "C", (), {"chat": type("Ch", (), {"completions": FakeCompletions()})()}
        )()
        return provider, sent

    def test_well_formed_arguments_become_a_tool_call(self) -> None:
        provider, _ = self._provider(
            [self.FakeToolCall("c1", "recall", '{"query": "where do I live"}')]
        )
        got = provider.chat(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            system=None,
            tools=[{"name": "recall", "description": "d", "parameters": {}}],
            max_tokens=100,
            effort="low",
        )
        assert got.wants_tools
        assert got.tool_calls[0].name == "recall"
        assert got.tool_calls[0].args == {"query": "where do I live"}

    def test_fenced_arguments_are_salvaged(self) -> None:
        """A small model wrapping its arguments in markdown is not a crash."""
        provider, _ = self._provider(
            [self.FakeToolCall("c1", "recall", '```json\n{"query": "cat"}\n```')]
        )
        got = provider.chat(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            system=None,
            tools=[{"name": "recall", "description": "d", "parameters": {}}],
            max_tokens=100,
            effort="low",
        )
        assert got.tool_calls[0].args == {"query": "cat"}

    def test_unrecoverable_arguments_drop_the_call(self) -> None:
        """A dropped call costs a wasted hop; a raised exception costs the run."""
        provider, _ = self._provider([self.FakeToolCall("c1", "recall", "not json at all")])
        got = provider.chat(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            system=None,
            tools=[{"name": "recall", "description": "d", "parameters": {}}],
            max_tokens=100,
            effort="low",
        )
        assert got.tool_calls == ()
        assert not got.wants_tools

    def test_the_transcript_is_translated_for_the_endpoint(self) -> None:
        provider, sent = self._provider(None, content="Chapel Hill.")
        got = provider.chat(
            model="m",
            messages=[
                {"role": "user", "content": "where?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "recall", "args": {"query": "home"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "Durham"},
            ],
            system="be terse",
            tools=[{"name": "recall", "description": "d", "parameters": {}}],
            max_tokens=100,
            effort="low",
        )
        roles = [m["role"] for m in sent["messages"]]
        assert roles == ["system", "user", "assistant", "tool"]
        # Arguments go over the wire as a JSON string, not a dict.
        assert sent["messages"][2]["tool_calls"][0]["function"]["arguments"] == (
            '{"query": "home"}'
        )
        assert sent["tools"][0]["type"] == "function"
        assert got.text == "Chapel Hill."

    def test_reasoning_blocks_are_stripped_from_the_answer(self) -> None:
        provider, _ = self._provider(None, content="<think>hmm</think>Chapel Hill.")
        got = provider.chat(
            model="m",
            messages=[{"role": "user", "content": "q"}],
            system=None,
            tools=None,
            max_tokens=100,
            effort="low",
        )
        assert got.text == "Chapel Hill."
