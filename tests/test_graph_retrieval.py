"""Regression tests for graph retrieval.

Both cases here come from a real local-model run where `temporal_graph` scored
0.00 across every probe type. Extraction had worked -- 36 nodes, 26 edges -- but
retrieval matched query terms against subjects only, so a question about an
entity that appears as an *object* retrieved nothing and the agent correctly
answered "I don't know" to everything.
"""

from __future__ import annotations

from datetime import date

from arena.backends.temporal_graph import TemporalGraph
from arena.types import Event

from .conftest import StubLLM


def ev(turn_id: int, at: date) -> Event:
    return Event(turn_id=turn_id, speaker="user", text="", at=at)


def build(triples: list[tuple[str, str, str]]) -> TemporalGraph:
    store = TemporalGraph(StubLLM())
    for i, (s, r, o) in enumerate(triples):
        event = ev(i + 1, date(2025, 1, i + 1))
        store.g.add_node(s, label=s)
        store.g.add_node(o, label=o)
        store.g.add_edge(s, o, relation=r, turn_id=event.turn_id, at=event.at)
        store._assert((s, r, o), event)
    return store


class TestObjectSideRetrieval:
    def test_query_matching_an_object_still_retrieves_the_fact(self) -> None:
        store = build([("sam", "owns", "checkout service")])
        # "checkout" appears only as an object; subject-only matching missed it.
        context = store.recall("who owns the checkout service?").context
        assert "sam" in context.lower()

    def test_query_matching_a_subject_retrieves_the_fact(self) -> None:
        store = build([("sam", "owns", "checkout service")])
        assert "checkout" in store.recall("what does sam own?").context.lower()

    def test_neighborhood_pulls_in_a_second_hop(self) -> None:
        store = build(
            [
                ("sam", "owns", "checkout service"),
                ("checkout service", "depends on", "payments api"),
            ]
        )
        context = store.recall("tell me about sam").context.lower()
        assert "checkout service" in context
        assert "payments api" in context, "second hop should be reachable"


class TestFallback:
    def test_unmatched_query_returns_everything_rather_than_nothing(self) -> None:
        store = build([("sam", "owns", "checkout service")])
        context = store.recall("completely unrelated zzzz").context
        assert "sam" in context.lower(), "an empty context guarantees a wrong answer"

    def test_empty_store_is_reported_as_empty(self) -> None:
        store = TemporalGraph(StubLLM())
        assert "no memories" in store.recall("anything").context


class TestSupersessionSurvivesRetrieval:
    def test_superseded_fact_is_labelled_not_dropped(self) -> None:
        store = build(
            [
                ("checkout service", "is owned by", "dana"),
                ("checkout service", "is owned by", "sam"),
            ]
        )
        context = store.recall("who owns checkout service?").context
        assert "NO LONGER TRUE" in context
        current = context.split("NO LONGER TRUE")[0]
        assert "sam" in current.lower() and "dana" not in current.lower()
