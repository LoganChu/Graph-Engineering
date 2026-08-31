"""The second axis: who owns control flow between the model and the store.

`Memory` asks *what is worth remembering*. This asks *how hard the agent works
to find it*. They are orthogonal, and holding one fixed while varying the other
is the only way to attribute a score to either.

    single_shot   one retrieval, one answer. No control flow at all.
    fanout        one call writes every query up front, all of them retrieve,
                  one answer. Breadth WITHOUT feedback -- the ablation that
                  says whether adaptivity or merely more queries did the work.
    plan_execute  one call splits the question into sub-questions, each
                  retrieves on its own branch, one answer joins them. The only
                  orchestrator that asks DIFFERENT questions rather than the
                  same question reworded.
    loop          LangChain `create_agent`. The MODEL decides when to search
                  again, and with what wording. Control flow lives in the
                  prompt and in the model's judgement.
    graph         LangGraph `StateGraph`. YOU decide: an explicit
                  retrieve -> assess -> retrieve cycle with a declared exit
                  condition and a hard hop cap. Control flow lives in the
                  topology.

Two things vary across those five, not one, and the pair is what makes the set
worth running:

                    |  one query, reworded  |  several distinct questions
    ----------------+-----------------------+----------------------------
    no feedback     |  single_shot, fanout  |  plan_execute
    feedback        |  loop, graph          |  --

`loop` and `graph` are the two positions people mean by "loop engineering"
versus "graph engineering". Note that they are not two frameworks: since
LangChain v1.0, `create_agent` is itself compiled to a LangGraph graph. The
difference measured here is not which library runs -- it is whether the
iteration policy is written down as edges or delegated to the model.

`fanout` is what makes that comparison legible. It spends the same retrieval
budget as `loop` and `graph` and opens with the same query, and differs in
exactly one respect: its later queries are written before any evidence comes
back. If it ties them, the extra searches were the whole story and the
per-hop model call bought nothing.

Every orchestrator answers through the same closed-book prompt in `agent.py`, so
a score difference is a difference in the evidence that got assembled, never in
how the question was asked.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from .llm import LLM
from .memory import Memory
from .types import Attempt, Recall, Stop

#: Ceiling on store round trips per probe. Without one, a model that keeps
#: failing to find a fact will keep paying for the privilege.
DEFAULT_MAX_HOPS = 4

_REGISTRY: dict[str, type["Orchestrator"]] = {}


def register(name: str) -> Callable[[type["Orchestrator"]], type["Orchestrator"]]:
    def wrap(cls: type["Orchestrator"]) -> type["Orchestrator"]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_orchestrator(name: str) -> type["Orchestrator"]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown orchestrator {name!r}; available: {sorted(_REGISTRY)}. "
            "'loop', 'graph' and 'plan_execute' need the extra: "
            "uv sync --extra orchestration"
        )
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


@dataclass(frozen=True)
class Hop:
    """One store round trip, with the position it belongs at in the record.

    `order` exists because `plan_execute` fans its retrievals out across
    LangGraph branches, which run on worker threads and finish in whatever
    order the store happens to return. Recording arrival order would make
    `queries` and `provenance` come back shuffled differently on every run of
    an otherwise identical cell, which is error-analysis output -- `arena
    inspect` reads it to say what the orchestrator actually searched for, and a
    field that reshuffles makes two runs of the same cell diff for no reason.

    It does not affect the cache. The cache is keyed on prompt text, and the
    only ordering that reaches a prompt is the evidence block, which the join
    node sorts for itself. It does not affect the evidence scores either --
    `evidence.score` is set arithmetic over session ids. What the LOCK below
    protects is different and more serious: `hops` and `chars` were
    read-modify-write on shared ints, which concurrent branches can silently
    undercount, and an undercounted hop is a cost comparison that lies.
    """

    order: int
    query: str
    chars: int
    latency_s: float
    provenance: tuple[str, ...]


class Retriever:
    """One probe's worth of store access, with the bookkeeping attached.

    Every orchestrator goes through this rather than calling `store.recall`
    directly, which is what makes hop counts and retrieved-context volume
    comparable across five very different control-flow styles.

    Safe to call from several threads. Sequential orchestrators never needed
    that; a fan-out does, and the accounting is the part that has to stay
    correct for the cost comparison to mean anything.
    """

    def __init__(self, store: Memory) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._log: list[Hop] = []

    @property
    def hops(self) -> int:
        return len(self._log)

    @property
    def chars(self) -> int:
        return sum(hop.chars for hop in self._log)

    @property
    def latency_s(self) -> float:
        """Total time spent inside `recall`, summed over hops.

        Summed rather than wall-clocked, so a fan-out is charged for the
        retrieval work it actually did instead of being credited for running
        the hops concurrently. It is a work measure, not a latency measure,
        and it stays comparable to the sequential orchestrators.
        """
        return sum(hop.latency_s for hop in self._log)

    @property
    def queries(self) -> tuple[str, ...]:
        return tuple(hop.query for hop in self._ordered())

    @property
    def provenance(self) -> tuple[str, ...]:
        """Union across hops, first-seen order.

        Accumulated here so retrieval can be graded against gold evidence
        without any orchestrator knowing that gold evidence exists.
        """
        seen: set[str] = set()
        found: list[str] = []
        for hop in self._ordered():
            for entry in hop.provenance:
                if entry not in seen:
                    seen.add(entry)
                    found.append(entry)
        return tuple(found)

    def _ordered(self) -> list[Hop]:
        return sorted(self._log, key=lambda hop: hop.order)

    def fetch(self, query: str, *, order: int | None = None) -> Recall:
        """Search the store once and bill it.

        `order` pins where this hop lands in the record. Pass it when hops are
        dispatched concurrently; leave it out and hops are recorded in the
        order they arrive, which is what every sequential orchestrator wants.
        """
        started = time.perf_counter()
        recall = self.store.recall(query)
        elapsed = time.perf_counter() - started
        with self._lock:
            self._log.append(
                Hop(
                    order=len(self._log) if order is None else order,
                    query=query,
                    chars=len(recall.context),
                    latency_s=elapsed,
                    provenance=tuple(recall.provenance),
                )
            )
        return recall

    def finish(self, text: str, note: str = "", stop: Stop = "answered") -> Attempt:
        return Attempt(
            text=text,
            hops=self.hops,
            context_chars=self.chars,
            read_latency_s=self.latency_s,
            queries=self.queries,
            provenance=self.provenance,
            stop=stop,
            note=note,
        )


class Orchestrator(ABC):
    """A policy for turning a question plus a store into an answer."""

    name: str = "unnamed"
    #: Human-readable one-liner shown by `arena list` and in the report.
    blurb: str = ""

    def __init__(self, llm: LLM, *, max_hops: int = DEFAULT_MAX_HOPS) -> None:
        self.llm = llm
        self.max_hops = max_hops

    @abstractmethod
    def run(self, store: Memory, question: str) -> Attempt:
        """Answer one probe. Implementations must route retrieval through
        `Retriever` so the hop accounting stays honest."""
