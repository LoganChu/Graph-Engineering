"""Executes the (backend x task) matrix.

One fresh store per cell -- no state leaks between backends or between tasks.
Turns are replayed in order and probes fire at their declared position, so a
probe asked at turn 12 sees exactly what the store had learned by turn 12.
"""

from __future__ import annotations

import time
import traceback
from collections import defaultdict
from pathlib import Path

from . import agent, judge
from .llm import LLM, Ledger
from .memory import get_backend
from .types import ProbeResult, RunResult, Task


def run_cell(
    backend_name: str,
    task: Task,
    *,
    model: str,
    effort: str,
    token_budget: int,
    cache_dir: Path,
    on_probe=None,
    llm_factory=None,
) -> RunResult:
    """Replay one task against one backend.

    `llm_factory` takes a Ledger and returns an LLM-shaped object; tests pass a
    stub so the whole pipeline can run without touching the API.
    """
    ledger = Ledger()
    llm = (
        llm_factory(ledger)
        if llm_factory
        else LLM(model=model, effort=effort, cache_dir=cache_dir, ledger=ledger)
    )
    store = get_backend(backend_name)(llm, token_budget=token_budget)
    result = RunResult(backend=backend_name, task_id=task.task_id)
    started = time.perf_counter()

    probes_by_turn: dict[int, list] = defaultdict(list)
    for probe in task.probes:
        probes_by_turn[probe.after_turn].append(probe)

    try:
        for event in task.events:
            store.observe(event)

            for probe in probes_by_turn.get(event.turn_id, []):
                read_started = time.perf_counter()
                recall = store.recall(probe.question, as_of=probe.as_of)
                read_elapsed = time.perf_counter() - read_started

                text = agent.answer(llm, recall, probe.question)
                grade, reason, hard_fail = judge.grade(llm, probe, text)

                pr = ProbeResult(
                    task_id=task.task_id,
                    probe=probe,
                    answer=text,
                    grade=grade,
                    reason=reason,
                    hard_fail=hard_fail,
                    context_chars=len(recall.context),
                    read_latency_s=read_elapsed,
                )
                result.probes.append(pr)
                if on_probe:
                    on_probe(pr)

        result.store_stats = store.stats()
    except Exception:
        result.error = traceback.format_exc(limit=4)

    result.usage = ledger.records
    result.wall_s = time.perf_counter() - started
    return result


def run_matrix(
    backends: list[str],
    tasks: list[Task],
    *,
    model: str,
    effort: str,
    token_budget: int = 2000,
    cache_dir: Path = Path("results/cache"),
    on_cell=None,
    on_probe=None,
    llm_factory=None,
) -> list[RunResult]:
    results: list[RunResult] = []
    for backend_name in backends:
        for task in tasks:
            cell = run_cell(
                backend_name,
                task,
                model=model,
                effort=effort,
                token_budget=token_budget,
                cache_dir=cache_dir,
                on_probe=on_probe,
                llm_factory=llm_factory,
            )
            results.append(cell)
            if on_cell:
                on_cell(cell)
    return results
