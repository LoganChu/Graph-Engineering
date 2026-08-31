"""LoCoMo -> arena task YAML.

LoCoMo (Maharana et al., ACL 2024) is the second corpus, and it is here to fix
the limitation LongMemEval's `oracle` split cannot: needle density. Oracle ships
only the sessions carrying the answer -- a median of 24 turns per question,
nearly all needle. LoCoMo hands the same store 369-689 turns across 19-32
sessions and asks ~200 questions of it. That is a haystack.

Three structural differences from `longmemeval.py`, each of which matters:

  * **One task per conversation, not per question.** LongMemEval gives every
    question its own haystack, so the write path is paid once per question --
    1,378 turns bought 60 probes. Here 5,882 turns buy 1,986 probes, because a
    conversation is ingested once and then interrogated ~200 times. That is
    roughly 8x more probe per extraction call, and it is the reason the full
    LoCoMo port is affordable where the full LongMemEval `s` split is not.

  * **Gold arrives already at turn granularity.** LoCoMo annotates evidence as
    turn ids (`evidence: ['D1:3']`), where LongMemEval's `answer_session_ids` is
    session-level and its turn-level `has_answer` flag has to be read off the
    turns themselves. Both are threaded through to `evidence.py` now, so a
    LoCoMo `dia_id` rides onto the turn as its `ref` and onto the probe as
    `gold_turns`; `gold_sessions` is derived from the same ids and kept as the
    coarse column.

  * **It supplies distractors.** Category 5 questions are adversarial: the
    question attributes an event to the wrong speaker, and `adversarial_answer`
    holds the answer a fooled model gives. The reference evaluator scores these
    by checking for an abstention phrase, so the correct behaviour is to
    decline. That makes `adversarial_answer` exactly the string a correct answer
    must not contain -- which is what `Probe.must_not_contain` was built for and
    what no other corpus here has ever been able to fill.

One more thing to know when reading the scores: LoCoMo conversations are
*human-human* (two named speakers), not user-assistant. Questions are asked
about both participants, so "the user" is not a meaningful frame here the way it
is in LongMemEval.
"""

from __future__ import annotations

import random
import re
from datetime import date
from typing import Any, Iterable

from .types import ProbeType

#: LoCoMo's numeric category codes, verified against the published evaluator
#: rather than inferred: category 5 is scored by checking the output for an
#: abstention phrase, which is what fixes it as our `negation`.
CATEGORIES: dict[int, ProbeType] = {
    1: "multi_hop",
    2: "temporal",
    4: "simple_recall",
    5: "negation",
}

#: Category 3 ("open-domain") is deliberately dropped rather than mapped.
#: Its questions -- "Would Caroline still want to pursue counseling if she had
#: not been supported growing up?" -- are answered from world knowledge and
#: inference, not from anything the transcript asserts. Our agent is forced
#: closed-book: its prompt permits only the retrieved excerpt as a source and
#: instructs it to say "I don't know" otherwise. Every category-3 question would
#: therefore be graded wrong for obeying its instructions, and the resulting
#: number would measure prompt compliance rather than memory.
OPEN_DOMAIN = 3

#: Reference answer for abstention probes. Worded to match how LongMemEval
#: phrases its own `_abs` answers ("The information provided is not enough...")
#: so the judge applies the same standard to both corpora -- the rubric grades
#: "I don't know" as correct only when the reference itself says the fact was
#: never stated.
ABSTAIN = (
    "The information provided is not enough. "
    "The conversation does not state this, and the question assumes something "
    "that was never said."
)

#: '1:56 pm on 8 May, 2023'
_DATE = re.compile(r"on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        "january february march april may june july august september "
        "october november december".split()
    )
}
_SESSION = re.compile(r"session_(\d+)$")


def parse_date(value: str) -> date:
    """LoCoMo stamps a session with a wall-clock string, not an ISO date."""
    m = _DATE.search(value or "")
    if not m:
        raise ValueError(f"cannot read {value!r} as a LoCoMo session timestamp")
    month = _MONTHS.get(m.group(2).strip().lower())
    if month is None:
        raise ValueError(f"unknown month in {value!r}")
    return date(int(m.group(3)), month, int(m.group(1)))


def session_of(dia_id: str) -> str:
    """'D1:3' -> 'D1'. The part before the colon is the session."""
    return str(dia_id).split(":", 1)[0]


def turn_text(turn: dict[str, Any]) -> str:
    """Flatten a possibly-multimodal turn into the one thing we can store.

    Roughly a fifth of LoCoMo turns carry an image. We are a text harness, so
    the BLIP caption is folded into the text rather than dropped -- otherwise
    those turns arrive semantically empty and every backend is scored on a
    transcript with holes in it.
    """
    text = (turn.get("text") or "").strip()
    caption = (turn.get("blip_caption") or "").strip()
    if caption:
        shared = f"[shared an image: {caption}]"
        return f"{text} {shared}".strip() if text else shared
    return text


def sessions_in(conversation: dict[str, Any]) -> list[tuple[str, date, list]]:
    """Return (session_id, date, turns) in chronological order.

    Chronological rather than declaration order for the same reason as
    LongMemEval: `temporal_graph` stamps `valid_to` from the event date when it
    supersedes a fact, so replaying June before February retires the newer fact
    with the older one -- silently, in the category the backend exists to win.
    """
    found: list[tuple[str, date, list]] = []
    for key, turns in conversation.items():
        m = _SESSION.match(key)
        if not m or not isinstance(turns, list):
            continue
        stamp = conversation.get(f"{key}_date_time")
        if stamp is None:
            continue
        found.append((f"D{m.group(1)}", parse_date(stamp), turns))
    found.sort(key=lambda item: (item[1], int(item[0][1:])))
    return found


def to_task(instance: dict[str, Any]) -> dict[str, Any]:
    """One LoCoMo conversation -> the YAML shape `tasks.load_task` reads.

    Unlike `longmemeval.to_task` this emits many probes against one haystack,
    which is also the first corpus here that makes `report.score_stat`'s
    cluster-by-task standard error do real work: 200 probes sharing a
    conversation are emphatically not 200 independent observations of it.
    """
    conversation = instance["conversation"]
    ordered = sessions_in(conversation)

    turns: list[dict[str, Any]] = []
    present: set[str] = set()
    dia_to_session: dict[str, str] = {}
    for session_id, when, raw_turns in ordered:
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            text = turn_text(turn)
            if not text:
                continue
            dia_id = str(turn.get("dia_id", ""))
            if dia_id:
                dia_to_session[dia_id] = session_id
            present.add(session_id)
            turns.append(
                {
                    "t": when,
                    "speaker": turn.get("speaker", "user"),
                    "text": text,
                    "session": session_id,
                    # LoCoMo's own turn id, which is exactly what `evidence`
                    # names. No id has to be synthesised here the way it does
                    # for LongMemEval.
                    "ref": dia_id,
                }
            )

    probes: list[dict[str, Any]] = []
    for qa in instance.get("qa", []):
        category = qa.get("category")
        try:
            category = int(category)
        except (TypeError, ValueError):
            continue
        kind = CATEGORIES.get(category)
        if kind is None:  # category 3, or anything the dataset adds later
            continue

        # Gold is intersected with the turns actually built above: 9 of
        # LoCoMo's 2,815 evidence ids point at turns that are not in the
        # transcript, and grading a store on retrieving those would be scoring
        # it against something no backend could satisfy. That intersection
        # leaves exactly two of the 1,890 mapped questions with no gold at all.
        gold_turns = sorted(
            {str(e) for e in (qa.get("evidence") or []) if str(e) in dia_to_session}
        )
        gold = sorted({dia_to_session[e] for e in gold_turns})

        probe: dict[str, Any] = {
            "after_turn": len(turns),
            "type": kind,
            "question": qa["question"],
            "gold_sessions": gold,
            "gold_turns": gold_turns,
        }
        if kind == "negation":
            probe["expected"] = ABSTAIN
            # The trap answer. This is the only corpus that ships one, and it
            # is what lets the deterministic guard run before the judge.
            distractor = str(qa.get("adversarial_answer") or "").strip()
            if distractor:
                probe["must_not_contain"] = [distractor]
        else:
            probe["expected"] = str(qa.get("answer", ""))
        probes.append(probe)

    kinds = sorted({p["type"] for p in probes})
    return {
        "id": str(instance["sample_id"]),
        "description": (
            f"LoCoMo {instance['sample_id']}. {len(ordered)} sessions, "
            f"{len(turns)} turns, {len(probes)} probes ({', '.join(kinds)})."
        ),
        "turns": turns,
        "probes": probes,
    }


def subset(
    instances: Iterable[dict[str, Any]],
    conversations: int | None = None,
    probes_per_conversation: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Trim the corpus along both axes, keeping the probe-type mix even.

    Two separate budgets, because they cost different things. Dropping
    *conversations* is what saves write-path spend -- one fewer conversation is
    ~590 fewer extraction calls. Dropping *probes* saves only answer and judge
    calls, which are far cheaper, so the sensible small run keeps every
    conversation short rather than keeping fewer conversations whole.
    """
    picked = list(instances)
    if conversations is not None:
        picked = picked[:conversations]
    if probes_per_conversation is None:
        return picked

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for instance in picked:
        by_kind: dict[int, list] = {}
        for qa in instance.get("qa", []):
            try:
                category = int(qa.get("category"))
            except (TypeError, ValueError):
                continue
            if category in CATEGORIES:
                by_kind.setdefault(category, []).append(qa)

        for bucket in by_kind.values():
            rng.shuffle(bucket)

        kept: list = []
        # Round-robin across types so a cap never silently deletes a whole
        # probe category -- the by-type table is the point of the corpus.
        while len(kept) < probes_per_conversation and any(by_kind.values()):
            for category in sorted(by_kind):
                if not by_kind[category]:
                    continue
                kept.append(by_kind[category].pop())
                if len(kept) >= probes_per_conversation:
                    break
        out.append({**instance, "qa": kept})
    return out
