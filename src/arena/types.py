"""Core data types shared across the harness.

Everything here is immutable (frozen dataclasses / pydantic models). Backends
return new objects rather than mutating shared state, which keeps a run's
provenance auditable after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

ProbeType = Literal[
    "simple_recall",      # stated once, asked back
    "multi_hop",          # requires joining two or more separate turns
    "contradiction",      # a later turn overrides an earlier one
    "temporal",           # "what was true as of <date>"
    "negation",           # something that was never said / was denied
    "aggregation",        # count or summarize across many turns
]

# "orchestrate" is a model call that decided *what to do next* rather than one
# that produced an answer: a tool-choice step in the loop orchestrator, an
# assess/reformulate node in the graph one. Splitting it out is what makes the
# orchestration axis measurable -- otherwise control-flow spend hides inside
# the answer column and every orchestrator looks equally cheap.
Phase = Literal["write", "read", "answer", "judge", "orchestrate"]


@dataclass(frozen=True)
class Event:
    """One observed turn of the session -- the unit a backend writes."""

    turn_id: int
    speaker: str
    text: str
    at: date

    def render(self) -> str:
        return f"[{self.at.isoformat()}] {self.speaker}: {self.text}"


@dataclass(frozen=True)
class Probe:
    """A question asked after `after_turn` events have been observed."""

    probe_id: str
    after_turn: int
    type: ProbeType
    question: str
    expected: str
    as_of: date | None = None
    must_not_contain: tuple[str, ...] = ()


@dataclass(frozen=True)
class Task:
    task_id: str
    description: str
    events: tuple[Event, ...]
    probes: tuple[Probe, ...]


@dataclass(frozen=True)
class Recall:
    """What a backend hands the agent for one probe."""

    context: str
    provenance: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Attempt:
    """The result of one orchestrator answering one probe.

    `hops` is the number of times the orchestrator went back to the store. It is
    the honest unit of orchestration effort: single-shot is always 1, and a loop
    or graph that needs four hops to answer what single-shot answered in one has
    to earn that cost back in accuracy.
    """

    text: str
    hops: int = 1
    context_chars: int = 0
    read_latency_s: float = 0.0
    queries: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Usage:
    """Token spend for a single API call, tagged with the phase that caused it."""

    phase: Phase
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_s: float = 0.0
    calls: int = 1  # >1 when structured output needed a repair round trip
    cached_locally: bool = False


class Verdict(BaseModel):
    """Structured judge output."""

    grade: Literal["correct", "partial", "incorrect"] = Field(
        description="correct = matches the expected answer; partial = right "
        "direction but incomplete or hedged; incorrect = wrong, refused, or "
        "contradicts the expected answer."
    )
    reason: str = Field(description="One sentence explaining the grade.")


@dataclass(frozen=True)
class ProbeResult:
    task_id: str
    probe: Probe
    answer: str
    grade: str
    reason: str
    hard_fail: bool = False       # tripped a must_not_contain guard
    context_chars: int = 0
    read_latency_s: float = 0.0
    hops: int = 1                 # store round trips the orchestrator made
    queries: tuple[str, ...] = ()  # what it actually searched for


@dataclass
class RunResult:
    """One (backend x orchestrator x task) cell of the matrix."""

    backend: str
    task_id: str
    orchestrator: str = "single_shot"
    probes: list[ProbeResult] = field(default_factory=list)
    usage: list[Usage] = field(default_factory=list)
    store_stats: dict = field(default_factory=dict)
    wall_s: float = 0.0
    error: str | None = None
