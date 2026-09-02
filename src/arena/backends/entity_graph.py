"""Knowledge-graph memory: turns are distilled into triples at write time.

This is the backend that pays an explicit extraction tax on every turn and hopes
to earn it back on multi-hop questions, where the answer lives in the *join*
between two turns that share no vocabulary. Retrieval is a k-hop neighborhood
walk rather than a similarity ranking, so a fact three turns apart from the
query terms can still surface -- which is exactly what BM25 and embeddings miss.

Known limitation, stated up front: entity linking is string-normalized rather
than semantic, so "my sister" and "Rachel" stay separate nodes unless the
extractor resolves them. `consolidate()` takes one LLM-assisted pass at merging
aliases; the residual error is real and shows up in the multi-hop scores.
"""

from __future__ import annotations

import re

import networkx as nx
from pydantic import BaseModel, Field

from ..memory import Memory, register
from ..types import Event, Recall

EXTRACT = """\
Extract factual triples from this conversation turn.

Turn (dated {when}), spoken by {speaker}: {text}

These are conversations in which one person talks about their own life, so most
facts are about {speaker}: what they did, went to, attended, bought, read,
learned, decided, prefer, or plan. A turn whose main purpose is asking a
question almost always states facts on the way to the question. Extract those;
the question itself is not a fact.

Rules:
- The SUBJECT is the entity the statement is about. Resolve first-person
  pronouns to {speaker}. Resolve other pronouns only when unambiguous, and skip
  the triple otherwise. When the statement is about something else, use that
  instead ("Sam owns checkout now" -> subject is sam).
- The RELATION is a short lowercase verb phrase drawn from the turn's own
  wording. Do not force the statement into a vocabulary the turn does not use.
- The OBJECT is a short noun phrase or literal value, under six words. Keep
  dates, durations and quantities verbatim -- they are usually the point of the
  fact, not decoration.
- Extract only what this turn asserts. No inference, no world knowledge. A
  technical or specialist topic is not world knowledge when the speaker says
  they did something with it: "I read the Raft paper" asserts that they read it.
- Split compound statements into separate triples.

Examples.

Turn: "I finally moved from Durham to New York last month, and started at Stripe."
  ({speaker}, moved from, Durham)
  ({speaker}, moved to, New York)
  ({speaker}, started at, Stripe)

Turn: "I went to my niece's graduation on the 3rd of June and ended up giving a
short speech, which I hadn't planned. Any tips for next time?"
  ({speaker}, went to, niece's graduation)
  ({speaker}, went to graduation on, 3rd of June)
  ({speaker}, gave, short speech)

Turn: "Thanks, that makes sense. I'll give it a try."
  no facts -- return an empty list"""


class Fact(BaseModel):
    subject: str = Field(description="Short noun phrase.")
    relation: str = Field(description="Short lowercase verb phrase.")
    object: str = Field(description="Short noun phrase or literal value.")


class Extraction(BaseModel):
    facts: list[Fact]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


@register("entity_graph")
class EntityGraph(Memory):
    """Triples in a MultiDiGraph; retrieval by k-hop neighborhood."""

    blurb = "LLM-extracted triples in a graph; k-hop neighborhood retrieval."

    HOPS = 2
    MAX_EDGES = 40

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.g = nx.MultiDiGraph()
        self.extract_calls = 0

    # -- write ---------------------------------------------------------------

    def _extract(self, event: Event) -> list[Fact]:
        prompt = EXTRACT.format(
            when=event.at.isoformat(), speaker=event.speaker, text=event.text
        )
        self.extract_calls += 1
        try:
            return self.llm.parse(prompt, Extraction, phase="write").facts
        except Exception:
            # A failed extraction loses the turn -- that is a real property of
            # this architecture, so record it rather than falling back.
            return []

    def observe(self, event: Event) -> None:
        for fact in self._extract(event):
            s, o = norm(fact.subject), norm(fact.object)
            if not s or not o:
                continue
            self.g.add_node(s, label=fact.subject)
            self.g.add_node(o, label=fact.object)
            self.g.add_edge(
                s,
                o,
                relation=norm(fact.relation),
                turn_id=event.turn_id,
                at=event.at,
            )

    # -- read ----------------------------------------------------------------

    def _seeds(self, query: str) -> list[str]:
        """Match query tokens against node names. Crude but LLM-free."""
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for node in self.g.nodes:
            tokens = set(re.findall(r"[a-z0-9]+", node))
            overlap = len(q & tokens)
            if overlap:
                scored.append((overlap / max(len(tokens), 1), overlap, node))
        scored.sort(reverse=True)
        seeds = [n for _, _, n in scored[:5]]
        # "user" anchors most personal questions; include it as a fallback hub.
        if "user" in self.g and "user" not in seeds:
            seeds.append("user")
        return seeds

    def _neighborhood(self, seeds: list[str]) -> list[tuple]:
        frontier = set(seeds)
        seen = set(seeds)
        edges: list[tuple] = []
        for _ in range(self.HOPS):
            nxt: set[str] = set()
            for node in frontier:
                if node not in self.g:
                    continue
                for _, dst, data in self.g.out_edges(node, data=True):
                    edges.append((node, data, dst))
                    if dst not in seen:
                        nxt.add(dst)
                for src, _, data in self.g.in_edges(node, data=True):
                    edges.append((src, data, node))
                    if src not in seen:
                        nxt.add(src)
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return edges

    def _render(self, edges: list[tuple]) -> str:
        lines, seen = [], set()
        for src, data, dst in edges:
            key = (src, data["relation"], dst)
            if key in seen:
                continue
            seen.add(key)
            s = self.g.nodes[src].get("label", src)
            o = self.g.nodes[dst].get("label", dst)
            lines.append(f"- ({data['at'].isoformat()}) {s} --[{data['relation']}]--> {o}")
        return "\n".join(lines[: self.MAX_EDGES])

    def recall(self, query: str) -> Recall:
        if self.g.number_of_edges() == 0:
            return Recall(context="(no memories)", note="empty graph")
        edges = self._neighborhood(self._seeds(query))
        body = self._render(edges)
        return Recall(
            context=f"KNOWN FACTS (subject --[relation]--> object):\n{body}",
            provenance=tuple(sorted({str(d["turn_id"]) for _, d, _ in edges})),
            note=f"{len(edges)} edges from {self.HOPS}-hop walk",
        )

    def stats(self) -> dict:
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "extract_calls": self.extract_calls,
        }
