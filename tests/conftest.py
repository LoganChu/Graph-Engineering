"""A stub LLM so the harness can be tested end-to-end with no API calls.

The stub still records Usage against the ledger, which means the cost
attribution path -- the part most likely to silently break -- is covered too.

It also speaks `chat()` with tool calls, so the loop and graph orchestrators run
offline against it. Four knobs drive the orchestration axis: `tool_hops` is how
many times the "model" asks to search before answering, `assess_hops` is how
many times the graph's assessment reports the evidence insufficient, and
`rewrites` / `plan_parts` are the search plans `fanout` and `plan_execute` write
up front. Scripting those is what lets the tests assert on control flow without
a model that might simply choose not to use it.
"""

from __future__ import annotations

from datetime import date

import pytest

from arena.llm import Ledger
from arena.providers import Completion, ToolCall
from arena.types import Event, Probe, Task, Usage


class StubLLM:
    """Mimics arena.llm.LLM. Canned extraction, echo answers, keyword judging."""

    def __init__(
        self,
        ledger: Ledger | None = None,
        *,
        tool_hops: int = 1,
        assess_hops: int = 0,
        rewrites: list[str] | None = None,
        plan_parts: list[str] | None = None,
    ) -> None:
        self.ledger = ledger or Ledger()
        self.model = "stub"
        self.effort = "low"
        self.calls: list[tuple[str, str]] = []
        self.tool_hops = tool_hops
        self.assess_hops = assess_hops
        self.assessments = 0
        # What the non-adaptive orchestrators get back when they plan their
        # searches up front. Lists rather than counts so a test can script
        # blanks and duplicates and watch them get dropped.
        self.rewrites = ["rewrite one", "rewrite two"] if rewrites is None else rewrites
        self.plan_parts = ["part one", "part two"] if plan_parts is None else plan_parts

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
        if fields == {"queries"}:
            # fanout: every query decided before any of them run.
            return schema(queries=list(self.rewrites))
        if fields == {"sub_questions"}:
            # plan_execute: the decomposition, also decided up front.
            return schema(sub_questions=list(self.plan_parts))
        if fields == {"sufficient", "missing", "next_query"}:
            # The graph orchestrator's routing decision. Report "not yet" for
            # the scripted number of cycles, then stop -- otherwise the only
            # thing ending the loop would be the hop cap, and the test could not
            # tell a working exit condition from a working cap.
            self.assessments += 1
            if self.assessments <= self.assess_hops:
                return schema(
                    sufficient=False,
                    missing="the other half",
                    next_query=f"reformulated query {self.assessments}",
                )
            return schema(sufficient=True, missing="", next_query="")
        raise AssertionError(f"StubLLM has no canned response for {schema!r}")

    def chat(self, messages, *, phase, system=None, tools=None, max_tokens=1000):
        """Mimics a tool-calling model: search `tool_hops` times, then answer."""
        self.calls.append(("chat", str(messages)))
        searched = [m for m in messages if m.get("role") == "tool"]
        prompt = str(messages)

        if tools and len(searched) < self.tool_hops:
            question = next(
                (m.get("content", "") for m in messages if m.get("role") == "user"), ""
            )
            got = Completion(
                text="",
                input_tokens=max(len(prompt) // 4, 1),
                output_tokens=20,
                tool_calls=(
                    ToolCall(
                        id=f"call_{len(searched)}",
                        name="recall",
                        args={"query": question},
                    ),
                ),
            )
        else:
            # Echo the evidence, so grading depends on what recall returned --
            # the same contract the `complete()` answer path honours.
            evidence = "\n".join(m.get("content", "") for m in searched)
            got = Completion(
                text=(evidence or "I don't know.")[:400],
                input_tokens=max(len(prompt) // 4, 1),
                output_tokens=20,
            )

        resolved = phase if isinstance(phase, str) else phase(got)
        self.ledger.add(
            Usage(
                phase=resolved,
                model="stub",
                input_tokens=got.input_tokens,
                output_tokens=got.output_tokens,
            )
        )
        return got


@pytest.fixture
def stub_factory():
    return lambda ledger: StubLLM(ledger)


@pytest.fixture
def stub_factory_for():
    """Build a stub factory with the cycle knobs set."""

    def make(**kwargs):
        return lambda ledger: StubLLM(ledger, **kwargs)

    return make


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
