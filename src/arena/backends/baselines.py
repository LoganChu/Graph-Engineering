"""Control and cheap-default backends.

`full_transcript` is the ceiling: everything ever said, no retrieval step. Any
architecture that cannot beat it on accuracy is only worth its cost savings.
`window_summary` is the default most production agents actually ship.
"""

from __future__ import annotations


from ..memory import Memory, budget_chars, register
from ..types import Event, Recall

SUMMARIZE = """\
You maintain a running summary of a conversation.

Existing summary:
{summary}

New turns to fold in:
{turns}

Rewrite the summary so it covers both. Preserve concrete facts: names, dates,
places, numbers, preferences, and any correction of an earlier statement (say
explicitly what changed and when). Drop small talk. Under 250 words.
Output only the summary."""


@register("full_transcript")
class FullTranscript(Memory):
    """No memory system at all -- replay the entire history every turn."""

    blurb = "Control. Whole transcript in context, no retrieval."

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[Event] = []

    def observe(self, event: Event) -> None:
        self.events.append(event)

    def recall(self, query: str) -> Recall:
        body = "\n".join(e.render() for e in self.events)
        return Recall(
            context=body,
            provenance=tuple(str(e.turn_id) for e in self.events),
            note="full history",
        )

    def stats(self) -> dict:
        return {"events": len(self.events), "chars": sum(len(e.text) for e in self.events)}


@register("window_summary")
class WindowSummary(Memory):
    """Last N turns verbatim; everything older folded into a running summary."""

    blurb = "Sliding window of recent turns + LLM running summary of the rest."

    WINDOW = 6
    FOLD_EVERY = 4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.recent: list[Event] = []
        self.summary = "(nothing yet)"
        self.pending: list[Event] = []
        self.folds = 0

    def observe(self, event: Event) -> None:
        self.recent.append(event)
        if len(self.recent) > self.WINDOW:
            self.pending.append(self.recent.pop(0))
        if len(self.pending) >= self.FOLD_EVERY:
            self._fold()

    def _fold(self) -> None:
        turns = "\n".join(e.render() for e in self.pending)
        self.summary = self.llm.complete(
            SUMMARIZE.format(summary=self.summary, turns=turns),
            phase="write",
            max_tokens=600,
        )
        self.pending = []
        self.folds += 1

    def recall(self, query: str) -> Recall:
        if self.pending:
            self._fold()
        recent = "\n".join(e.render() for e in self.recent)
        body = f"SUMMARY OF EARLIER CONVERSATION:\n{self.summary}\n\nRECENT TURNS:\n{recent}"
        return Recall(
            context=body[: budget_chars(self.token_budget)],
            provenance=("summary",) + tuple(str(e.turn_id) for e in self.recent),
            note=f"{self.folds} folds",
        )

    def stats(self) -> dict:
        return {
            "summary_chars": len(self.summary),
            "window": len(self.recent),
            "folds": self.folds,
        }
