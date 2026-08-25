"""Task files: a dated conversation plus probes fired at specific points.

Probes carry `after_turn` so a question can be asked *before* the contradicting
turn arrives -- which is how you test that a store handled a change rather than
just that it holds the latest value.
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
            as_of=_as_date(p["as_of"]) if p.get("as_of") else None,
            must_not_contain=tuple(p.get("must_not_contain", ())),
        )
        for i, p in enumerate(raw["probes"])
    )

    for probe in probes:
        if not 1 <= probe.after_turn <= len(events):
            raise ValueError(
                f"{probe.probe_id}: after_turn={probe.after_turn} is outside "
                f"the {len(events)}-turn conversation"
            )

    return Task(
        task_id=raw["id"],
        description=raw.get("description", ""),
        events=events,
        probes=probes,
    )


def load_all(directory: Path) -> list[Task]:
    return [load_task(p) for p in sorted(directory.glob("*.yaml"))]
