"""Dense-embedding retrieval over turns -- the standard RAG baseline.

Embeddings run locally (sentence-transformers) rather than through an API so the
write path stays free and the comparison against BM25 isolates *semantics* from
*cost*. Install with:  uv sync --extra embeddings
"""

from __future__ import annotations


from ..memory import Memory, register
from ..types import Event, Recall

_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The vector backend needs the 'embeddings' extra. "
                "Run: uv sync --extra embeddings"
            ) from exc
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


@register("vector")
class VectorStore(Memory):
    """Cosine similarity over per-turn embeddings, top-k into context."""

    blurb = "Local dense embeddings over turns, top-k cosine retrieval."

    TOP_K = 8

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[Event] = []
        self.vectors = None

    def observe(self, event: Event) -> None:
        model = _load_model()
        import numpy as np

        vec = model.encode([event.text], normalize_embeddings=True)
        self.events.append(event)
        self.vectors = vec if self.vectors is None else np.vstack([self.vectors, vec])

    def recall(self, query: str) -> Recall:
        if not self.events:
            return Recall(context="(no memories)", note="empty")
        model = _load_model()
        q = model.encode([query], normalize_embeddings=True)
        scores = (self.vectors @ q.T).ravel()
        top = scores.argsort()[::-1][: self.TOP_K]
        order = sorted(int(i) for i in top)
        body = "\n".join(self.events[i].render() for i in order)
        return Recall(
            context=body,
            provenance=tuple(str(self.events[i].turn_id) for i in order),
            note=f"top-{len(order)} of {len(self.events)}",
        )

    def stats(self) -> dict:
        return {"events": len(self.events), "model": _MODEL_NAME}
