"""Bi-temporal graph memory: facts have a lifetime, not just a timestamp.

The claim being tested: memory is not storage, it is belief revision. Append-only
stores (vector, lexical, plain graph) answer "where do I live?" by handing the
model two contradictory sentences and hoping recency wins. This backend closes
the old fact instead.

Two clocks per fact:
    valid_from / valid_to -- when the fact was true in the world
    ingested_at           -- when the system learned it

That second clock is what lets you ask "what did the system believe on March 1"
as distinct from "what was actually true on March 1", which is the query no
similarity search can answer at all.
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

    def alive_at(self, when: date) -> bool:
        if when < self.valid_from:
            return False
        return self.valid_to is None or when < self.valid_to

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
        seeds = set(self._seeds(query))
        picked: list[Version] = []
        for (subject, _), history in self.versions.items():
            if subject in seeds:
                picked.extend(history)
        if not picked:  # fall back to everything rather than answering blind
            picked = [v for h in self.versions.values() for v in h]
        return picked

    def recall(self, query: str, as_of: date | None = None) -> Recall:
        if not self.versions:
            return Recall(context="(no memories)", note="empty store")

        candidates = self._relevant(query)

        if as_of is not None:
            live = [v for v in candidates if v.alive_at(as_of)]
            body = "\n".join(v.render() for v in sorted(live, key=lambda v: v.valid_from))
            return Recall(
                context=(
                    f"FACTS TRUE AS OF {as_of.isoformat()} "
                    "(the store was queried at that point in time):\n" + (body or "(none)")
                ),
                provenance=tuple(str(v.turn_id) for v in live),
                note=f"as-of query, {len(live)} live facts",
            )

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
