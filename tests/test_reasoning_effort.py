"""Reasoning control and context-window guarding on the local provider.

Both behaviours exist because of one measured failure: qwen3.5:9b spends
reasoning tokens out of the same `max_tokens` budget as the answer, so the
300-token answer cap returned an empty `content` and the judge scored it as a
wrong answer. A truncation artifact must never be reportable as a memory result.
"""

from __future__ import annotations

import pytest

from arena.providers import REASONING_EFFORT


class FakeUsage:
    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 10) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeMessage:
    def __init__(self, content: str = "ok") -> None:
        self.content = content
        self.tool_calls = None


def make_provider(prompt_tokens: int = 100, context_limit: int = 4096):
    """An OpenAICompatProvider with the HTTP layer replaced by a recorder."""
    pytest.importorskip("openai")
    from arena.providers import OpenAICompatProvider

    provider = OpenAICompatProvider.__new__(OpenAICompatProvider)
    provider.base_url = "http://fake"
    provider.context_limit = context_limit
    provider.peak_prompt_tokens = 0
    provider.context_warnings = 0
    sent: dict = {}

    class FakeResponse:
        def __init__(self) -> None:
            self.choices = [type("C", (), {"message": FakeMessage()})()]
            self.usage = FakeUsage(prompt_tokens=prompt_tokens)

    class FakeCompletions:
        def create(self, **kwargs):
            sent.update(kwargs)
            return FakeResponse()

    provider.client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()}
    )()
    return provider, sent


class TestReasoningEffortMapping:
    def test_benchmark_default_disables_reasoning(self) -> None:
        """`--effort low` is the default, and must not spend answer budget."""
        assert REASONING_EFFORT["low"] == "none"

    def test_scale_is_monotonic_and_total(self) -> None:
        for level in ("low", "medium", "high", "xhigh", "max"):
            assert level in REASONING_EFFORT, f"{level} must map to something"
        assert REASONING_EFFORT["max"] == "high"

    def test_unknown_effort_falls_back_to_none(self) -> None:
        """An unmapped value must not silently re-enable reasoning."""
        provider, sent = make_provider()
        provider.complete(
            model="m", prompt="p", system=None, max_tokens=300, effort="nonsense"
        )
        assert sent["reasoning_effort"] == "none"


class TestRequestBody:
    def test_complete_sends_mapped_effort(self) -> None:
        provider, sent = make_provider()
        provider.complete(model="m", prompt="p", system=None, max_tokens=300, effort="high")
        assert sent["reasoning_effort"] == "medium"
        assert sent["temperature"] == 0

    def test_effort_reaches_the_wire_at_all(self) -> None:
        """Regression: the local provider used to accept `effort` and drop it."""
        provider, sent = make_provider()
        provider.complete(model="m", prompt="p", system=None, max_tokens=300, effort="low")
        assert "reasoning_effort" in sent

    def test_structured_output_path_also_disables_reasoning(self) -> None:
        """Extraction and the judge go through parse(); same trap applies."""
        provider, sent = make_provider()
        provider._chat(model="m", prompt="p", system=None, max_tokens=300, json_mode=True)
        assert sent["reasoning_effort"] == "none"
        assert sent["response_format"] == {"type": "json_object"}


class TestContextGuard:
    def test_quiet_when_the_prompt_fits(self, capsys) -> None:
        provider, _ = make_provider(prompt_tokens=500, context_limit=4096)
        provider.complete(model="m", prompt="p", system=None, max_tokens=300, effort="low")
        assert provider.context_warnings == 0
        assert "[context]" not in capsys.readouterr().out

    def test_warns_when_the_prompt_approaches_the_window(self, capsys) -> None:
        provider, _ = make_provider(prompt_tokens=3500, context_limit=4096)
        provider.complete(model="m", prompt="p", system=None, max_tokens=300, effort="low")
        assert provider.context_warnings == 1
        assert "[context]" in capsys.readouterr().out

    def test_peak_is_tracked_for_reporting(self) -> None:
        provider, _ = make_provider(prompt_tokens=1234)
        provider.complete(model="m", prompt="p", system=None, max_tokens=300, effort="low")
        assert provider.peak_prompt_tokens == 1234

    def test_warning_does_not_spam(self, capsys) -> None:
        provider, _ = make_provider(prompt_tokens=4000, context_limit=4096)
        for _ in range(10):
            provider.complete(
                model="m", prompt="p", system=None, max_tokens=300, effort="low"
            )
        assert provider.context_warnings == 10, "every breach still counted"
        assert capsys.readouterr().out.count("[context]") == 3, "but printed 3 times"
