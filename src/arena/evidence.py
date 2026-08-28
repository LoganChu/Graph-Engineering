"""Grading retrieval itself, against the corpus's own gold annotation.

The answer score confounds two things a memory benchmark exists to separate:
whether the store found the right turns, and whether the agent did anything
sensible with them. LongMemEval ships `answer_session_ids` -- the sessions that
actually carry the answer -- and every retrieval backend here already reports
which turns it handed over in `Recall.provenance`. Lining those up costs no
model calls and yields the one metric in this harness that does not move when
the judge model changes.

Two backends cannot be graded this way and that is a finding rather than a gap.
`agent_notes` reports `("notes",)` and `summary` reports `("summary",)`: both
answer from text they rewrote, so there is no turn to trace back to. They score
`None` here, which is the honest statement that their evidence is unauditable by
construction.
"""

from __future__ import annotations

from .types import Attempt, EvidenceScore, Event, Probe


def _resolve(provenance: tuple[str, ...], by_turn: dict[int, Event]) -> list[Event]:
    """Provenance entries are turn ids as strings. Anything else is unresolvable."""
    seen: set[int] = set()
    out: list[Event] = []
    for entry in provenance:
        try:
            turn_id = int(entry)
        except (TypeError, ValueError):
            continue
        if turn_id in seen:
            continue
        event = by_turn.get(turn_id)
        if event is not None:
            seen.add(turn_id)
            out.append(event)
    return out


def score(
    probe: Probe, events: tuple[Event, ...], attempt: Attempt
) -> EvidenceScore | None:
    """Grade one attempt's retrieval. `None` when it cannot honestly be graded.

    `events` is what the store had actually observed when the probe fired, not
    the whole task. Gold is intersected with it, so a probe asked before its
    evidence arrives is not graded on failing to retrieve turns that did not
    exist yet.

    Returns `None` in three cases, which mean different things:
      * the probe ships no gold annotation -- nothing to grade against;
      * none of its gold had been observed yet -- nothing retrievable;
      * the backend reported provenance that names no turns -- nothing to trace.

    An empty retrieval is *not* one of them. Finding nothing is a real result
    and scores zero recall.
    """
    observed = {e.session_id for e in events if e.session_id}
    gold = {s for s in probe.gold_sessions if s} & observed
    if not gold:
        return None

    by_turn = {e.turn_id: e for e in events}
    retrieved = _resolve(attempt.provenance, by_turn)
    if attempt.provenance and not retrieved:
        return None  # unauditable backend: it rewrote the evidence

    sessions = {e.session_id for e in retrieved if e.session_id}
    return EvidenceScore(
        n_gold=len(gold),
        n_hit=len(sessions & gold),
        n_sessions=len(sessions),
        gold_chars=sum(len(e.render()) for e in retrieved if e.session_id in gold),
        retrieved_chars=sum(len(e.render()) for e in retrieved),
    )


def aggregate(scores: list[EvidenceScore]) -> dict:
    """Micro-average across probes: sum the counts, then divide.

    Macro-averaging the per-probe ratios would weight a probe with one gold
    session the same as one with five, which is not the question being asked.
    """
    if not scores:
        return {}
    n_gold = sum(s.n_gold for s in scores)
    n_hit = sum(s.n_hit for s in scores)
    n_sessions = sum(s.n_sessions for s in scores)
    gold_chars = sum(s.gold_chars for s in scores)
    retrieved_chars = sum(s.retrieved_chars for s in scores)
    return {
        "evidence_recall": n_hit / n_gold if n_gold else 0.0,
        "evidence_precision": n_hit / n_sessions if n_sessions else 0.0,
        "evidence_efficiency": gold_chars / retrieved_chars if retrieved_chars else 0.0,
        "evidence_n": len(scores),
    }
