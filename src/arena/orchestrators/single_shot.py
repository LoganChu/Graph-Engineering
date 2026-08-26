"""One retrieval, one answer. The control that the other two have to beat.

This is what the harness did before there was an orchestration axis, kept
byte-identical so that every number published against the memory axis alone
remains comparable -- and so the existing response cache still hits.
"""

from __future__ import annotations

from datetime import date

from .. import agent
from ..memory import Memory
from ..orchestration import Orchestrator, Retriever, register
from ..types import Attempt


@register("single_shot")
class SingleShot(Orchestrator):
    """No control flow: retrieve once with the question as the query, answer."""

    blurb = "Control. One recall, one answer -- the model never gets a second look."

    def run(self, store: Memory, question: str, as_of: date | None = None) -> Attempt:
        retriever = Retriever(store, as_of)
        recall = retriever.fetch(question)
        return retriever.finish(
            agent.answer(self.llm, recall, question), note=recall.note
        )
