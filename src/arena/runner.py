"""Executes the (backend x orchestrator x task) matrix.

One fresh store per cell -- no state leaks between backends, orchestrators, or
tasks. Turns are replayed in order and probes fire at their declared position,
so a probe asked at turn 12 sees exactly what the store had learned by turn 12.

The orchestrator is constructed once per cell and is stateless between probes:
whatever a loop or a graph learned answering probe 3 must not help it on probe
4, or the score stops being a measurement of the store.
"""

from __future__ import annotations

import time
import traceback
from collections import defaultdict
from pathlib import Path

from . import agent, backends, judge, orchestrators  # noqa: F401  (registration)
from .llm import Ledger, ModelConfig
from .memory import get_backend
from .orchestration import DEFAULT_MAX_HOPS, get_orchestrator
from .types import ProbeResult, RunResult, Task


def run_cell(
    backend_name: str,
    task: Task,
    *,
    config: ModelConfig,
    orchestrator: str = "single_shot",
    token_budget: int = 2000,
    max_hops: int = DEFAULT_MAX_HOPS,
    cache_dir: Path = Path("results/cache"),
    on_probe=None,
    llm_factory=None,
) -> RunResult:
    """Replay one task against one backend under one orchestrator.

    `llm_factory` takes a Ledger and returns an LLM-shaped object; tests pass a
    stub so the whole pipeline can run without touching a model at all.
    """
    ledger = Ledger()
    llm = llm_factory(ledger) if llm_factory else config.build(ledger, cache_dir)
    store = get_backend(backend_name)(llm, token_budget=token_budget)
    driver = get_orchestrator(orchestrator)(llm, max_hops=max_hops)
    result = RunResult(
        backend=backend_name, task_id=task.task_id, orchestrator=orchestrator
    )
    started = time.perf_counter()

    probes_by_turn: dict[int, list] = defaultdict(list)
    for probe in task.probes:
        probes_by_turn[probe.after_turn].append(probe)

    try:
        for event in task.events:
            store.observe(event)

            for probe in probes_by_turn.get(event.turn_id, []):
                attempt = driver.run(store, probe.question, as_of=probe.as_of)
                grade, reason, hard_fail = judge.grade(llm, probe, attempt.text)

                pr = ProbeResult(
                    task_id=task.task_id,
                    probe=probe,
                    answer=attempt.text,
                    grade=grade,
                    reason=reason,
                    hard_fail=hard_fail,
                    context_chars=attempt.context_chars,
                    read_latency_s=attempt.read_latency_s,
                    hops=attempt.hops,
                    queries=attempt.queries,
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
    config: ModelConfig,
    orchestrators: list[str] | None = None,
    token_budget: int = 2000,
    max_hops: int = DEFAULT_MAX_HOPS,
    cache_dir: Path = Path("results/cache"),
    on_cell=None,
    on_probe=None,
    llm_factory=None,
) -> list[RunResult]:
    results: list[RunResult] = []
    for orchestrator in orchestrators or ["single_shot"]:
        for backend_name in backends:
            for task in tasks:
                cell = run_cell(
                    backend_name,
                    task,
                    config=config,
                    orchestrator=orchestrator,
                    token_budget=token_budget,
                    max_hops=max_hops,
                    cache_dir=cache_dir,
                    on_probe=on_probe,
                    llm_factory=llm_factory,
                )
                results.append(cell)
                if on_cell:
                    on_cell(cell)
    return results
