"""Bi-temporal graph memory: facts have a lifetime, not just a timestamp.

The claim being tested: memory is not storage, it is belief revision. Append-only
stores (vector, lexical, plain graph) answer "where do I live?" by handing the
model two contradictory sentences and hoping recency wins. This backend closes
the old fact instead.

Two clocks per fact:
    valid_from / valid_to -- when the fact was true in the world
    ingested_at           -- when the system learned it

The write path is what is measured: a differing assertion on a single-valued
relation closes the open version instead of stacking beside it, so `recall` can
hand the model a CURRENT block and a separate NO LONGER TRUE block rather than
two contradictory sentences and a hope that recency wins.

That second clock once also backed an as-of read ("what did the system believe
on March 1"). It is gone. No corpus in the arena -- and, as far as we could
find, no published memory benchmark -- ships a *query* timestamp as a field;
they all put temporal reference in the question text. It was unreachable code
claiming a capability nothing exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from ..memory import register
from ..types import Event, Recall
from .entity_graph import EntityGraph, norm


@dataclass(frozen=True)
class Version:
    """One assertion of (subject, relation) -> object over a validity interval."""

    subject: str
    relation: str
    object: str
    valid_from: date
    ingested_at: date
    turn_id: int
    valid_to: date | None = None

    @property
    def open(self) -> bool:
        return self.valid_to is None

    def render(self) -> str:
        if self.open:
            window = f"since {self.valid_from.isoformat()}"
        else:
            window = f"{self.valid_from.isoformat()} to {self.valid_to.isoformat()}"
        return f"- {self.subject} {self.relation} {self.object}  [{window}]"


@register("temporal_graph")
class TemporalGraph(EntityGraph):
    """Entity graph + validity intervals + supersession on write."""

    blurb = "Bi-temporal facts: new assertions close old ones instead of stacking."

    #: Relations where a subject can hold many values at once (owning two cars),
    #: as opposed to single-valued ones (living in one city).
    MULTIVALUED = {"owns", "likes", "prefers", "is allergic to", "knows", "visited"}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.versions: dict[tuple[str, str], list[Version]] = {}
        self.supersessions = 0

    # -- write ---------------------------------------------------------------

    def observe(self, event: Event) -> None:
        super().observe(event)  # keep the graph for neighborhood walks
        for fact in self._facts_of(event):
            self._assert(fact, event)

    def _facts_of(self, event: Event):
        """Re-read the triples super().observe() just wrote for this turn."""
        for src, dst, data in self.g.edges(data=True):
            if data["turn_id"] == event.turn_id:
                yield (src, data["relation"], dst)

    def _assert(self, fact: tuple[str, str, str], event: Event) -> None:
        subject, relation, obj = fact
        key = (subject, relation)
        history = self.versions.setdefault(key, [])

        if relation not in self.MULTIVALUED:
            for i, prior in enumerate(history):
                if prior.open and norm(prior.object) != norm(obj):
                    history[i] = replace(prior, valid_to=event.at)
                    self.supersessions += 1

        if any(p.open and norm(p.object) == norm(obj) for p in history):
            return  # restating a fact we already hold open

        history.append(
            Version(
                subject=subject,
                relation=relation,
                object=obj,
                valid_from=event.at,
                ingested_at=event.at,
                turn_id=event.turn_id,
            )
        )

    # -- read ----------------------------------------------------------------

    def _relevant(self, query: str) -> list[Version]:
        """Seed, walk the graph, then collect the versions those edges touch.

        Matching seeds against subjects alone silently drops every fact where
        the query term is the *object* -- asking "who owns checkout?" would miss
        `sam --owns--> checkout` entirely, because "checkout" is not a subject.
        Reusing the k-hop walk fixes the asymmetry and picks up neighbors of the
        matched entity for free, which is the whole reason to keep a graph.
        """
        seeds = self._seeds(query)
        touched: set[tuple[str, str]] = set()
        for src, data, dst in self._neighborhood(seeds):
            touched.add((src, data["relation"]))
            # An edge is evidence for its endpoints in both directions.
            touched.update(
                key for key in self.versions if key[0] == dst or key[0] == src
            )

        picked = [v for key in touched for v in self.versions.get(key, ())]
        if not picked:  # fall back to everything rather than answering blind
            picked = [v for h in self.versions.values() for v in h]
        return picked

    def recall(self, query: str) -> Recall:
        if not self.versions:
            return Recall(context="(no memories)", note="empty store")

        candidates = self._relevant(query)

        current = [v for v in candidates if v.open]
        retired = [v for v in candidates if not v.open]
        parts = ["CURRENT FACTS:"]
        parts.append("\n".join(v.render() for v in current) or "(none)")
        if retired:
            parts.append(
                "\nNO LONGER TRUE (superseded -- do not answer with these):\n"
                + "\n".join(v.render() for v in retired)
            )
        return Recall(
            context="\n".join(parts),
            provenance=tuple(str(v.turn_id) for v in current),
            note=f"{len(current)} current / {len(retired)} superseded",
        )

    def stats(self) -> dict:
        base = super().stats()
        total = sum(len(h) for h in self.versions.values())
        return base | {
            "fact_keys": len(self.versions),
            "versions": total,
            "supersessions": self.supersessions,
        }
