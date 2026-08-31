"""Task files: a dated conversation plus probes fired at specific points.

Probes carry `after_turn` so a question can be asked *before* the contradicting
turn arrives -- which is how you test that a store handled a change rather than
just that it holds the latest value.

Turns carry an optional `ref` -- the source corpus's own id for that turn -- and
probes an optional `gold_turns` list of those refs. That pairing is what lets
retrieval be graded at turn granularity instead of session granularity; a file
carrying neither still loads and is graded on whatever it does carry.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import yaml

from .types import Event, Probe, Task


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise ValueError(f"cannot read {value!r} as a date")


def load_task(path: Path) -> Task:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    events = tuple(
        Event(
            turn_id=i + 1,
            speaker=turn.get("speaker", "user"),
            text=turn["text"],
            at=_as_date(turn["t"]),
            session_id=str(turn.get("session", "")),
            ref=str(turn.get("ref", "")),
        )
        for i, turn in enumerate(raw["turns"])
    )

    probes = tuple(
        Probe(
            probe_id=f"{raw['id']}::{i + 1}",
            after_turn=int(p.get("after_turn", len(events))),
            type=p["type"],
            question=p["question"],
            expected=str(p["expected"]),
            must_not_contain=tuple(p.get("must_not_contain", ())),
            gold_sessions=tuple(str(s) for s in p.get("gold_sessions", ())),
            gold_turns=tuple(str(t) for t in p.get("gold_turns", ())),
        )
        for i, p in enumerate(raw["probes"])
    )

    refs = {e.ref for e in events if e.ref}
    for probe in probes:
        if not 1 <= probe.after_turn <= len(events):
            raise ValueError(
                f"{probe.probe_id}: after_turn={probe.after_turn} is outside "
                f"the {len(events)}-turn conversation"
            )
        # A gold turn naming nothing in the transcript is a build bug, and a
        # silent one: `evidence.py` would intersect it away and grade the probe
        # against whatever survived, reporting a recall that quietly asks less
        # than it says it does. Both builders already drop unresolvable ids
        # (LoCoMo ships nine), so anything reaching here is a real mismatch.
        dangling = sorted(t for t in probe.gold_turns if t not in refs)
        if dangling:
            raise ValueError(
                f"{probe.probe_id}: gold_turns name turns that are not in the "
                f"transcript: {', '.join(dangling[:5])}"
            )

    return Task(
        task_id=raw["id"],
        description=raw.get("description", ""),
        events=events,
        probes=probes,
    )


def load_all(directory: Path) -> list[Task]:
    return [load_task(p) for p in sorted(directory.glob("*.yaml"))]
