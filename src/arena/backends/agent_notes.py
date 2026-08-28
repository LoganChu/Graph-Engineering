"""Agent-authored memory: the model keeps its own notes file.

No schema, no index, no retrieval step -- the model decides what is worth
writing down and rewrites the document as understanding changes. This is the
approach coding agents converged on (a markdown memory directory), and it is the
strongest contradiction-handler in the set for a simple reason: revision is the
default operation rather than an append.

Its weakness is equally structural. The whole document goes into context on
every read, so cost scales with everything remembered rather than with what the
question needs, and a fact dropped during a rewrite is gone with no audit trail.
"""

from __future__ import annotations


from ..memory import Memory, budget_chars, register
from ..types import Event, Recall

REVISE = """\
You maintain a long-term memory document about a person you are assisting.

CURRENT MEMORY DOCUMENT:
{notes}

NEW TURNS (each is dated):
{turns}

Rewrite the document to incorporate the new turns. Rules:
- Keep it organized under markdown headings by topic.
- Record dates for anything that could change over time.
- If a new turn contradicts something already written, REPLACE the old
  statement and add a dated note about what changed -- do not keep both as if
  they were both true.
- Preserve details that seem incidental; you cannot re-read the raw turns later.
- Stay under {limit} words.

Output only the revised document."""


@register("agent_notes")
class AgentNotes(Memory):
    """A single markdown document the model revises as it goes."""

    blurb = "Model-authored markdown notes, rewritten as understanding changes."

    REVISE_EVERY = 4
    WORD_LIMIT = 400

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.notes = "(empty)"
        self.pending: list[Event] = []
        self.revisions = 0

    def observe(self, event: Event) -> None:
        self.pending.append(event)
        if len(self.pending) >= self.REVISE_EVERY:
            self._revise()

    def _revise(self) -> None:
        if not self.pending:
            return
        turns = "\n".join(e.render() for e in self.pending)
        self.notes = self.llm.complete(
            REVISE.format(notes=self.notes, turns=turns, limit=self.WORD_LIMIT),
            phase="write",
            max_tokens=1200,
        )
        self.pending = []
        self.revisions += 1

    def recall(self, query: str) -> Recall:
        self._revise()  # flush before answering; unwritten turns are invisible
        return Recall(
            context=f"MEMORY DOCUMENT:\n{self.notes}"[: budget_chars(self.token_budget)],
            provenance=("notes",),
            note=f"{self.revisions} revisions",
        )

    def stats(self) -> dict:
        return {"note_chars": len(self.notes), "revisions": self.revisions}
