"""Grading retrieval itself, against the corpus's own gold annotation.

The answer score confounds two things a memory benchmark exists to separate:
whether the store found the right turns, and whether the agent did anything
sensible with them. Both corpora ship the annotation that separates them, and
every retrieval backend here already reports which turns it handed over in
`Recall.provenance`. Lining those up costs no model calls and yields the one
metric in this harness that does not move when the judge model changes.

Two grains are graded, and the finer one is the headline. LongMemEval ships
`answer_session_ids` (sessions) *and* a per-turn `has_answer` flag; LoCoMo ships
`evidence: ['D1:3']`, which is turn-level to begin with. Session grain alone was
too coarse to earn its place: on the `oracle` split every haystack session is a
gold session, so session precision is 1.00 for any backend that returns anything
and session recall asks only whether a store found the sessions it was handed.
The same split is 8.2% needle at turn grain. Session counts are still computed,
because a corpus that annotates only sessions can still be graded on them and
because they are what earlier runs were scored on -- but they are the coarse
column now, not the number.

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
    the whole task. Gold is intersected with it, at both grains, so a probe
    asked before its evidence arrives is not graded on failing to retrieve turns
    that did not exist yet.

    Returns `None` in three cases, which mean different things:
      * the probe ships no gold annotation at either grain -- nothing to grade;
      * none of its gold had been observed yet -- nothing retrievable;
      * the backend reported provenance that names no turns -- nothing to trace.

    An empty retrieval is *not* one of them. Finding nothing is a real result
    and scores zero recall.

    A probe annotated at only one grain is graded at that grain and carries
    zeroes at the other; `aggregate` keeps the two denominators apart rather
    than letting an ungraded grain average in as a miss.
    """
    observed_sessions = {e.session_id for e in events if e.session_id}
    observed_refs = {e.ref for e in events if e.ref}
    gold = {s for s in probe.gold_sessions if s} & observed_sessions
    gold_turns = {t for t in probe.gold_turns if t} & observed_refs
    if not gold and not gold_turns:
        return None

    by_turn = {e.turn_id: e for e in events}
    retrieved = _resolve(attempt.provenance, by_turn)
    if attempt.provenance and not retrieved:
        return None  # unauditable backend: it rewrote the evidence

    sessions = {e.session_id for e in retrieved if e.session_id}
    refs = {e.ref for e in retrieved if e.ref}
    return EvidenceScore(
        n_gold=len(gold),
        n_hit=len(sessions & gold),
        n_sessions=len(sessions),
        gold_chars=sum(len(e.render()) for e in retrieved if e.session_id in gold),
        retrieved_chars=sum(len(e.render()) for e in retrieved),
        n_gold_turns=len(gold_turns),
        n_hit_turns=len(refs & gold_turns),
        n_turns=len(refs),
        gold_turn_chars=sum(len(e.render()) for e in retrieved if e.ref in gold_turns),
    )


def aggregate(scores: list[EvidenceScore]) -> dict:
    """Micro-average across probes: sum the counts, then divide.

    Macro-averaging the per-probe ratios would weight a probe with one gold
    session the same as one with five, which is not the question being asked.

    The two grains are summed over different probe sets. A probe with no
    turn-level gold contributes nothing to the turn denominators -- counting it
    as a miss would report a corpus's silence as a backend's failure -- so the
    turn keys are absent entirely when nothing was annotated that finely, and
    `evidence_turn_n` says how many probes stand behind them.
    """
    if not scores:
        return {}
    out: dict = {}

    at_session = [s for s in scores if s.n_gold]
    if at_session:
        n_gold = sum(s.n_gold for s in at_session)
        n_hit = sum(s.n_hit for s in at_session)
        n_sessions = sum(s.n_sessions for s in at_session)
        gold_chars = sum(s.gold_chars for s in at_session)
        retrieved_chars = sum(s.retrieved_chars for s in at_session)
        out.update(
            {
                "evidence_recall": n_hit / n_gold if n_gold else 0.0,
                "evidence_precision": n_hit / n_sessions if n_sessions else 0.0,
                "evidence_efficiency": (
                    gold_chars / retrieved_chars if retrieved_chars else 0.0
                ),
                "evidence_n": len(at_session),
            }
        )

    at_turn = [s for s in scores if s.n_gold_turns]
    if at_turn:
        n_gold = sum(s.n_gold_turns for s in at_turn)
        n_hit = sum(s.n_hit_turns for s in at_turn)
        n_turns = sum(s.n_turns for s in at_turn)
        gold_chars = sum(s.gold_turn_chars for s in at_turn)
        retrieved_chars = sum(s.retrieved_chars for s in at_turn)
        out.update(
            {
                "evidence_turn_recall": n_hit / n_gold if n_gold else 0.0,
                "evidence_turn_precision": n_hit / n_turns if n_turns else 0.0,
                "evidence_turn_efficiency": (
                    gold_chars / retrieved_chars if retrieved_chars else 0.0
                ),
                "evidence_turn_n": len(at_turn),
            }
        )
    return out
