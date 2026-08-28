"""Deterministic behavior of the stores -- no API involved."""

from __future__ import annotations

from datetime import date

import pytest

from arena import backends  # noqa: F401  (registers)
from arena.backends.temporal_graph import TemporalGraph
from arena.memory import get_backend
from arena.types import Event

from .conftest import StubLLM


def ev(turn_id: int, text: str, at: date, speaker: str = "user") -> Event:
    return Event(turn_id=turn_id, speaker=speaker, text=text, at=at)


class TestBM25:
    def test_ranks_the_matching_turn_first(self) -> None:
        store = get_backend("bm25")(StubLLM())
        store.observe(ev(1, "The capital of France is Paris.", date(2025, 1, 1)))
        store.observe(ev(2, "I had oatmeal for breakfast.", date(2025, 1, 2)))
        store.observe(ev(3, "My cat Miso is allergic to chicken.", date(2025, 1, 3)))

        recall = store.recall("what is my cat allergic to")
        assert "chicken" in recall.context
        assert "3" in recall.provenance

    def test_empty_store_does_not_crash(self) -> None:
        store = get_backend("bm25")(StubLLM())
        assert "no memories" in store.recall("anything").context


class TestFullTranscript:
    def test_returns_everything_in_order(self) -> None:
        store = get_backend("full_transcript")(StubLLM())
        for i in range(5):
            store.observe(ev(i + 1, f"turn {i}", date(2025, 1, i + 1)))
        recall = store.recall("q")
        assert all(f"turn {i}" in recall.context for i in range(5))
        assert store.stats()["events"] == 5


class TestTemporalGraph:
    """The supersession logic is the whole thesis of this backend."""

    @pytest.fixture
    def store(self) -> TemporalGraph:
        s = TemporalGraph(StubLLM())
        # Bypass LLM extraction: assert triples directly.
        s._assert(("user", "lives in", "durham"), ev(1, "", date(2025, 1, 1)))
        s._assert(("user", "lives in", "chapel hill"), ev(2, "", date(2025, 6, 1)))
        return s

    def test_new_value_closes_the_previous_one(self, store: TemporalGraph) -> None:
        history = store.versions[("user", "lives in")]
        assert len(history) == 2
        old, new = history
        assert old.valid_to == date(2025, 6, 1)
        assert new.open

    def test_current_recall_marks_the_old_value_superseded(
        self, store: TemporalGraph
    ) -> None:
        ctx = store.recall("where do I live").context
        assert "chapel hill" in ctx.lower()
        assert "NO LONGER TRUE" in ctx

    def test_the_retired_value_keeps_its_validity_interval(
        self, store: TemporalGraph
    ) -> None:
        """The as-of *read* is gone -- no corpus supplies a query timestamp --
        but the interval it read from is still what supersession records, and
        it is what the CURRENT / NO LONGER TRUE split is rendered from."""
        old, new = store.versions[("user", "lives in")]
        assert (old.valid_from, old.valid_to) == (date(2025, 1, 1), date(2025, 6, 1))
        assert new.valid_from == date(2025, 6, 1) and new.valid_to is None
        assert "2025-01-01 to 2025-06-01" in old.render()

    def test_multivalued_relations_do_not_supersede(self) -> None:
        s = TemporalGraph(StubLLM())
        s._assert(("user", "owns", "a bike"), ev(1, "", date(2025, 1, 1)))
        s._assert(("user", "owns", "a car"), ev(2, "", date(2025, 2, 1)))
        assert all(v.open for v in s.versions[("user", "owns")])
        assert s.supersessions == 0

    def test_restating_a_held_fact_is_a_no_op(self) -> None:
        s = TemporalGraph(StubLLM())
        s._assert(("user", "lives in", "durham"), ev(1, "", date(2025, 1, 1)))
        s._assert(("user", "lives in", "Durham"), ev(2, "", date(2025, 2, 1)))
        assert len(s.versions[("user", "lives in")]) == 1
        assert s.supersessions == 0


class TestWriteCostIsAttributed:
    """A backend that spends tokens on ingest must show up in the write phase."""

    def test_agent_notes_bills_writes(self) -> None:
        llm = StubLLM()
        store = get_backend("agent_notes")(llm)
        for i in range(4):
            store.observe(ev(i + 1, f"fact number {i}", date(2025, 1, i + 1)))
        assert llm.ledger.by_phase("write"), "notes rewrite should bill to 'write'"

    def test_bm25_writes_are_free(self) -> None:
        llm = StubLLM()
        store = get_backend("bm25")(llm)
        for i in range(4):
            store.observe(ev(i + 1, f"fact number {i}", date(2025, 1, i + 1)))
        assert not llm.ledger.by_phase("write")
