"""Core data types shared across the harness.

Everything here is immutable (frozen dataclasses / pydantic models). Backends
return new objects rather than mutating shared state, which keeps a run's
provenance auditable after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Sequence

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

# Why an orchestrator stopped searching. Without this the hop count is
# ambiguous: a run averaging 4.0 hops under a cap of 4 has told you about the
# cap, not about the policy. Reporting the mix is the only way to know whether
# `--max-hops` is binding the comparison being published.
Stop = Literal[
    "answered",         # single_shot: there was never a second hop to take
    "sufficient",       # the policy decided it had enough
    "hop_cap",          # ran out of budget with the question still open
    "recursion_limit",  # the loop would not terminate on its own
    "no_query",         # said "not enough" but could not phrase another search
    "error",
]

#: How a matrix cell ended. `partial` is the one that matters: before it
#: existed, a replay that died at turn 400 of 492 kept whatever probes had
#: already fired and was averaged into the headline score as if it were whole.
Outcome = Literal["complete", "partial", "error"]


@dataclass(frozen=True)
class Event:
    """One observed turn of the session -- the unit a backend writes."""

    turn_id: int
    speaker: str
    text: str
    at: date
    #: Which source session this turn came from. LongMemEval marks its gold
    #: evidence at session granularity, so this is what makes retrieval
    #: gradeable; hand-authored tasks leave it empty and are scored on the
    #: answer alone.
    session_id: str = ""

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
    must_not_contain: tuple[str, ...] = ()
    #: Sessions that actually carry the answer. Present for imported corpora
    #: that ship the annotation; empty means retrieval cannot be graded for
    #: this probe and only the answer is scored.
    gold_sessions: tuple[str, ...] = ()


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
class EvidenceScore:
    """Did the store retrieve the turns that actually hold the answer?

    Graded against the corpus's own gold annotation with no model involved.
    That is the point of it: every other number in this harness moves when the
    judge model changes, and this one does not.

    It also splits a failure the answer score cannot. A backend scoring badly
    with `recall` near 1.0 retrieved the evidence and the agent fumbled it; the
    same score with `recall` near 0.0 is a memory failure. Those are different
    bugs, and one number cannot tell them apart.

    Counts rather than ratios are stored so the report can micro-average across
    probes instead of averaging ratios over different denominators.
    """

    n_gold: int           # gold sessions for this probe
    n_hit: int            # gold sessions that appeared in what was retrieved
    n_sessions: int       # distinct sessions retrieved
    gold_chars: int       # chars of retrieved turns belonging to gold sessions
    retrieved_chars: int  # chars of every retrieved turn

    @property
    def recall(self) -> float:
        return self.n_hit / self.n_gold if self.n_gold else 0.0

    @property
    def precision(self) -> float:
        return self.n_hit / self.n_sessions if self.n_sessions else 0.0

    @property
    def efficiency(self) -> float:
        """Share of the retrieved material that was worth retrieving.

        Measured over selected turns rather than rendered context, so a graph
        backend that stores triples instead of prose is not credited or
        penalised for how compactly it renders.
        """
        return self.gold_chars / self.retrieved_chars if self.retrieved_chars else 0.0


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
    #: Union of every hop's `Recall.provenance`, deduplicated in first-seen
    #: order. Carried up so the runner can grade retrieval without any
    #: orchestrator needing to know that gold evidence exists.
    provenance: tuple[str, ...] = ()
    stop: Stop = "answered"
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
class Stat:
    """A reported number with its uncertainty attached.

    Existing at all is the point. Every headline figure here used to be a single
    sample rendered to two decimals, which reads as precision the run does not
    have: at 60 probes a score of 0.70 carries a standard error near 0.06, so
    most of the gaps in the backend table were inside the noise.
    """

    mean: float
    stderr: float
    n: int

    @classmethod
    def over(cls, values: Sequence[float]) -> "Stat":
        n = len(values)
        if n == 0:
            return cls(0.0, 0.0, 0)
        mean = sum(values) / n
        if n == 1:
            # One cluster is one observation: the spread is unknown, not zero.
            return cls(mean, float("nan"), 1)
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        return cls(mean, (variance / n) ** 0.5, n)

    @property
    def ci95(self) -> float:
        return 1.96 * self.stderr

    def render(self, places: int = 2) -> str:
        if self.n < 2:
            return f"{self.mean:.{places}f} +/-?"
        return f"{self.mean:.{places}f} +/-{self.ci95:.{places}f}"


@dataclass(frozen=True)
class PassK:
    """Fraction of probes answered correctly on *every* one of k trials.

    Mean accuracy and reliability are different questions, and for anything a
    person would depend on the second is the one that matters: a store right 75%
    of the time answers only 42% of probes right three times running. Strict --
    only `correct` counts, and a single `partial` breaks the streak.
    """

    k: int
    value: float
    n_probes: int


@dataclass(frozen=True)
class Failure:
    """Why a cell did not finish, and how far it got.

    Structured rather than a traceback string because the report has to make a
    decision with it -- an incomplete cell is excluded from the headline average
    -- and a decision cannot be made against free text.
    """

    where: Literal["setup", "write", "read", "judge"]
    detail: str
    turn_id: int | None = None       # how far the replay got
    probes_done: int = 0
    probes_expected: int = 0


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
    stop: Stop = "answered"       # why it stopped searching
    evidence: EvidenceScore | None = None  # None when the corpus ships no gold
    trial: int = 0

    @property
    def correct(self) -> bool:
        """Strict success, for pass^k. Partial credit is a reporting choice."""
        return self.grade == "correct"


@dataclass
class RunResult:
    """One (backend x orchestrator x task x trial) cell of the matrix."""

    backend: str
    task_id: str
    orchestrator: str = "single_shot"
    trial: int = 0
    probes: list[ProbeResult] = field(default_factory=list)
    usage: list[Usage] = field(default_factory=list)
    store_stats: dict = field(default_factory=dict)
    wall_s: float = 0.0
    failure: Failure | None = None

    @property
    def outcome(self) -> Outcome:
        if self.failure is None:
            return "complete"
        return "partial" if self.probes else "error"

    @property
    def error(self) -> str | None:
        """Flat view of `failure`, for display and for the raw JSON dump."""
        return self.failure.detail if self.failure else None
