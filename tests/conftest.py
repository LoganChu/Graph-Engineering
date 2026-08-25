"""A stub LLM so the harness can be tested end-to-end with no API calls.

The stub still records Usage against the ledger, which means the cost
attribution path -- the part most likely to silently break -- is covered too.
"""

from __future__ import annotations

from datetime import date

import pytest

from arena.llm import Ledger
from arena.types import Event, Probe, Task, Usage


class StubLLM:
    """Mimics arena.llm.LLM. Canned extraction, echo answers, keyword judging."""

    def __init__(self, ledger: Ledger | None = None) -> None:
        self.ledger = ledger or Ledger()
        self.model = "stub"
        self.effort = "low"
        self.calls: list[tuple[str, str]] = []

    def _bill(self, phase: str, prompt: str, out: int = 20) -> None:
        self.ledger.add(
            Usage(
                phase=phase,
                model="stub",
                input_tokens=max(len(prompt) // 4, 1),
                output_tokens=out,
            )
        )

    def complete(self, prompt, *, phase, system=None, max_tokens=2000) -> str:
        self.calls.append((phase, prompt))
        self._bill(phase, prompt)
        if phase == "answer":
            # Echo the excerpt so grading depends on what recall actually returned.
            excerpt = prompt.split("QUESTION")[0]
            return excerpt.strip()[:400]
        return "stub summary"

    def parse(self, prompt, schema, *, phase, system=None, max_tokens=1000):
        self.calls.append((phase, prompt))
        self._bill(phase, prompt)
        fields = set(schema.model_fields)
        if fields == {"facts"}:
            return schema(facts=[])
        if fields == {"grade", "reason"}:
            return schema(grade="incorrect", reason="stub judge")
        raise AssertionError(f"StubLLM has no canned response for {schema!r}")


@pytest.fixture
def stub_factory():
    return lambda ledger: StubLLM(ledger)


@pytest.fixture
def tiny_task() -> Task:
    events = tuple(
        Event(turn_id=i + 1, speaker="user", text=text, at=at)
        for i, (text, at) in enumerate(
            [
                ("I live in Durham and my cat is called Miso.", date(2025, 1, 1)),
                ("Miso is allergic to chicken.", date(2025, 2, 1)),
                ("I moved to Chapel Hill today.", date(2025, 6, 1)),
            ]
        )
    )
    probes = (
        Probe(
            probe_id="tiny::1",
            after_turn=3,
            type="contradiction",
            question="Where do I live?",
            expected="Chapel Hill",
            must_not_contain=("Durham",),
        ),
    )
    return Task(task_id="tiny", description="fixture", events=events, probes=probes)
