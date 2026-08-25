"""BM25 retrieval over raw turns.

Worth including even though nobody markets it as "agent memory": lexical search
is a stubbornly strong baseline on recall tasks where the question reuses the
question-asker's own vocabulary, and it costs nothing at write time. If a
fancier backend does not beat this, it has not earned its complexity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date

from ..memory import Memory, register
from ..types import Event, Recall

TOKEN = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@register("bm25")
class BM25(Memory):
    """Classic Okapi BM25 over the turn log. Pure Python, no dependencies."""

    blurb = "Lexical BM25 over raw turns. Free writes, no semantics."

    TOP_K = 8

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[Event] = []
        self.docs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.df: Counter = Counter()

    def observe(self, event: Event) -> None:
        toks = tokenize(event.text)
        counts = Counter(toks)
        self.events.append(event)
        self.docs.append(counts)
        self.doc_lens.append(len(toks))
        for term in counts:
            self.df[term] += 1

    def _score(self, query_terms: list[str], idx: int) -> float:
        n = len(self.docs)
        avgdl = (sum(self.doc_lens) / n) if n else 1.0
        doc = self.docs[idx]
        dl = self.doc_lens[idx] or 1
        total = 0.0
        for term in query_terms:
            f = doc.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
            total += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / avgdl))
        return total

    def recall(self, query: str, as_of: date | None = None) -> Recall:
        if not self.events:
            return Recall(context="(no memories)", note="empty")
        terms = tokenize(query)
        ranked = sorted(
            range(len(self.events)),
            key=lambda i: self._score(terms, i),
            reverse=True,
        )[: self.TOP_K]
        ranked.sort()  # present in chronological order, not score order
        body = "\n".join(self.events[i].render() for i in ranked)
        return Recall(
            context=body,
            provenance=tuple(str(self.events[i].turn_id) for i in ranked),
            note=f"top-{len(ranked)} of {len(self.events)}",
        )

    def stats(self) -> dict:
        return {"events": len(self.events), "vocab": len(self.df)}
