"""LongMemEval -> arena task files.

The hand-authored tasks were a harness demonstration: a 14-turn haystack with
nine probes hung off it, which is a needle density no real session history has.
At that size `full_transcript` fits everything in context and top-k retrieval
returns half the store, so the memory axis barely separates. LongMemEval fixes
the ratio -- a median 492-turn haystack per question in the `s` variant, of
which a handful of turns carry the evidence.

The impedance mismatch, stated up front: LongMemEval pairs ONE question with
its OWN haystack, while a `Task` here is one conversation carrying many probes.
So every converted instance is a task with exactly one probe, and the write path
is paid per question rather than amortized across them. That is not a conversion
bug, it is what the dataset is; it just means the write column dominates any run
over the `s` variant and the variant choice is really a budget choice:

    oracle   evidence sessions only.   ~23 turns/question (median), 10,960 total.
    s        ~50 sessions/question.   ~492 turns/question (median), 246,930 total.

`entity_graph` and `temporal_graph` spend one `llm.parse` per turn, so the full
`s` variant is a quarter of a million extraction calls per graph backend. Subset
it. `stratified()` exists so that subsetting does not quietly skew the probe-type
mix that the report buckets by.

One thing the dataset gives us that the hand-authored tasks never could:
`answer_session_ids`, the sessions that actually carry the answer. Those ride
through onto `Probe.gold_sessions`, and each turn keeps its `session` id, which
lets `evidence.py` grade *retrieval* against ground truth with no model in the
loop. It is the only metric here that does not move when the judge changes, and
it separates the two failures the answer score cannot: the store never found the
evidence, versus the store found it and the agent fumbled it.

Task files built before this existed still load -- they simply carry no session
ids and score `None` for evidence. Re-run the builder to pick it up.

One thing the dataset cannot give us: `must_not_contain`. The hard-fail guard
needs a distractor string and LongMemEval ships no such annotation, so imported
probes carry no guard and a `knowledge-update` answer naming the superseded fact
alongside the current one is graded by the judge alone. LoCoMo does supply
distractors -- see `locomo.py` -- which is part of why it is worth having both.

Temporal reference lives in the question text here ("...after its first
service") rather than in a field, which is true of every corpus we surveyed and
is why the harness no longer carries an `as_of` query parameter at all.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Iterable

from .types import ProbeType

#: LongMemEval's question_type -> our ProbeType. The three `single-session-*`
#: kinds all ask for a fact stated once in one session; what varies is whether
#: the user, the assistant, or a stated preference is the source, which is a
#: distinction about the corpus rather than about the store.
QUESTION_TYPES: dict[str, ProbeType] = {
    "single-session-user": "simple_recall",
    "single-session-assistant": "simple_recall",
    "single-session-preference": "simple_recall",
    "multi-session": "multi_hop",
    "knowledge-update": "contradiction",
    "temporal-reasoning": "temporal",
}

#: Abstention is marked by a suffix on question_id, not by question_type -- an
#: abstention instance keeps the type of the question it was derived from.
ABSTENTION_SUFFIX = "_abs"

#: '2023/04/10 (Mon) 23:07'
_DATE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")


def parse_date(value: str) -> date:
    m = _DATE.search(value)
    if not m:
        raise ValueError(f"cannot read {value!r} as a LongMemEval timestamp")
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def probe_type(instance: dict[str, Any]) -> ProbeType:
    """Abstention wins over the base type: the correct answer is 'I don't know'
    regardless of which category the question was spun out of."""
    if instance["question_id"].endswith(ABSTENTION_SUFFIX):
        return "negation"
    kind = instance["question_type"]
    if kind not in QUESTION_TYPES:
        raise ValueError(f"unmapped question_type {kind!r}")
    return QUESTION_TYPES[kind]


def to_task(instance: dict[str, Any]) -> dict[str, Any]:
    """One LongMemEval instance -> the YAML shape `tasks.load_task` reads.

    Sessions are replayed in chronological order, which the raw data is NOT.
    That matters beyond tidiness: `temporal_graph` closes an open fact when a
    differing assertion arrives, stamping `valid_to` from the event date. Feed
    it a June session before an February one and it will retire the newer fact
    with the older one, inverting exactly the belief revision it exists to test.
    """
    sessions = instance["haystack_sessions"]
    dates = [parse_date(d) for d in instance["haystack_dates"]]
    if len(dates) != len(sessions):
        raise ValueError(
            f"{instance['question_id']}: {len(sessions)} sessions but {len(dates)} dates"
        )

    evidence = set(instance.get("answer_session_ids") or ())
    ids = instance.get("haystack_session_ids") or [""] * len(sessions)

    order = sorted(range(len(sessions)), key=lambda i: (dates[i], i))

    turns: list[dict[str, Any]] = []
    for i in order:
        for turn in sessions[i]:
            turns.append(
                {
                    "t": dates[i],
                    "speaker": turn.get("role", "user"),
                    "text": turn["content"],
                    # Carried through so retrieval can be graded against
                    # `answer_session_ids` without a model in the loop.
                    "session": ids[i],
                }
            )

    kind = probe_type(instance)
    n_evidence = sum(1 for i in order if ids[i] in evidence)
    # Only sessions actually present in this haystack are gradeable. Asking a
    # store to retrieve a session it was never shown would score retrieval
    # against something no backend could ever satisfy.
    present = set(ids)
    gold = sorted(s for s in evidence if s in present and s)

    return {
        "id": instance["question_id"],
        "description": (
            f"LongMemEval {instance['question_type']}"
            + (" (abstention)" if kind == "negation" else "")
            + f". {len(sessions)} sessions, {len(turns)} turns, "
            f"{n_evidence} carrying evidence."
        ),
        "turns": turns,
        "probes": [
            {
                # LongMemEval asks only at the end of the haystack, so the
                # `after_turn` machinery -- asking before a contradiction lands
                # -- goes unexercised on imported tasks.
                "after_turn": len(turns),
                "type": kind,
                "question": instance["question"],
                # `answer` is an int on 32 of the 500 instances.
                "expected": str(instance["answer"]),
                "gold_sessions": gold,
            }
        ],
    }


def stratified(
    instances: Iterable[dict[str, Any]], limit: int | None, seed: int = 0
) -> list[dict[str, Any]]:
    """Take `limit` instances with the probe-type mix kept as even as possible.

    Taking the first N instead would hand the report a lopsided by-type table,
    and the by-type table is the entire point of importing this dataset.
    Deterministic given `seed`, so a run is reproducible.
    """
    items = list(instances)
    if limit is None or limit >= len(items):
        return items

    import random

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        buckets.setdefault(probe_type(item), []).append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked: list[dict[str, Any]] = []
    # Round-robin over the type buckets until the budget runs out, so a scarce
    # type (negation has only 30 instances) is never starved by a plentiful one.
    while len(picked) < limit and any(buckets.values()):
        for kind in sorted(buckets):
            if not buckets[kind] or len(picked) >= limit:
                continue
            picked.append(buckets[kind].pop())
    return picked
