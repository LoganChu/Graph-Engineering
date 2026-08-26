"""The second axis: who owns control flow between the model and the store.

`Memory` asks *what is worth remembering*. This asks *how hard the agent works
to find it*. They are orthogonal, and holding one fixed while varying the other
is the only way to attribute a score to either.

    single_shot   one retrieval, one answer. No control flow at all.
    loop          LangChain `create_agent`. The MODEL decides when to search
                  again, and with what wording. Control flow lives in the
                  prompt and in the model's judgement.
    graph         LangGraph `StateGraph`. YOU decide: an explicit
                  retrieve -> assess -> retrieve cycle with a declared exit
                  condition and a hard hop cap. Control flow lives in the
                  topology.

`loop` and `graph` are the same two positions people mean by "loop engineering"
versus "graph engineering". Note that they are not two frameworks: since
LangChain v1.0, `create_agent` is itself compiled to a LangGraph graph. The
difference measured here is not which library runs -- it is whether the
iteration policy is written down as edges or delegated to the model.

Every orchestrator answers through the same closed-book prompt in `agent.py`, so
a score difference is a difference in the evidence that got assembled, never in
how the question was asked.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import date
from typing import Callable

from .llm import LLM
from .memory import Memory
from .types import Attempt, Recall

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
            "'loop' and 'graph' need the extra: uv sync --extra orchestration"
        )
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


class Retriever:
    """One probe's worth of store access, with the bookkeeping attached.

    Every orchestrator goes through this rather than calling `store.recall`
    directly, which is what makes hop counts and retrieved-context volume
    comparable across three very different control-flow styles.
    """

    def __init__(self, store: Memory, as_of: date | None = None) -> None:
        self.store = store
        self.as_of = as_of
        self.hops = 0
        self.chars = 0
        self.latency_s = 0.0
        self.queries: list[str] = []

    def fetch(self, query: str) -> Recall:
        started = time.perf_counter()
        recall = self.store.recall(query, as_of=self.as_of)
        self.latency_s += time.perf_counter() - started
        self.hops += 1
        self.chars += len(recall.context)
        self.queries.append(query)
        return recall

    def finish(self, text: str, note: str = "") -> Attempt:
        return Attempt(
            text=text,
            hops=self.hops,
            context_chars=self.chars,
            read_latency_s=self.latency_s,
            queries=tuple(self.queries),
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
    def run(self, store: Memory, question: str, as_of: date | None = None) -> Attempt:
        """Answer one probe. Implementations must route retrieval through
        `Retriever` so the hop accounting stays honest."""
