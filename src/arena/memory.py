"""The one interface every memory architecture in the arena implements.

Deliberately small. Two methods carry the whole design space:

    observe(event)  -- the WRITE policy: what is worth storing, in what form,
                       and at what cost. This is where graph backends pay their
                       extraction tax and where naive backends pay nothing.

    recall(query)   -- the READ policy: what subset of the store earns a place
                       in the context window for this question.

`consolidate()` is the optional third leg -- sleep-time reorganization
(summarizing, merging duplicate entities, decaying stale facts). Backends that
do not reorganize simply inherit the no-op.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Callable

from .llm import LLM
from .types import Event, Recall

_REGISTRY: dict[str, type["Memory"]] = {}


def register(name: str) -> Callable[[type["Memory"]], type["Memory"]]:
    """Class decorator: make a backend addressable by name from the CLI."""

    def wrap(cls: type["Memory"]) -> type["Memory"]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_backend(name: str) -> type["Memory"]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


class Memory(ABC):
    """Base class for a memory architecture under test."""

    name: str = "unnamed"
    #: Human-readable one-liner shown in the report.
    blurb: str = ""

    def __init__(self, llm: LLM, token_budget: int = 2000) -> None:
        self.llm = llm
        self.token_budget = token_budget

    @abstractmethod
    def observe(self, event: Event) -> None:
        """Ingest one turn. May call the LLM (tag those calls phase='write')."""

    @abstractmethod
    def recall(self, query: str, as_of: date | None = None) -> Recall:
        """Return the context to put in front of the agent for this question.

        `as_of` is set only for temporal probes. Backends without a notion of
        time should ignore it -- and will be scored on that.
        """

    def consolidate(self) -> None:
        """Optional periodic reorganization. Default: nothing."""
        return None

    def stats(self) -> dict:
        """Store-size diagnostics for the report."""
        return {}


def budget_chars(token_budget: int) -> int:
    """Rough chars-per-token for trimming context before the answer call.

    A ~4 chars/token heuristic is fine here because the budget is a guardrail,
    not an accounting figure -- real spend comes from the API usage numbers.
    """
    return token_budget * 4
