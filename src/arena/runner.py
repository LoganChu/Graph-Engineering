"""Executes the (backend x orchestrator x task x trial) matrix.

One fresh store per cell -- no state leaks between backends, orchestrators,
tasks, or trials. Turns are replayed in order and probes fire at their declared
position, so a probe asked at turn 12 sees exactly what the store had learned by
turn 12.

The orchestrator is constructed once per cell and is stateless between probes:
whatever a loop or a graph learned answering probe 3 must not help it on probe
4, or the score stops being a measurement of the store.

Two things a cell records beyond the answer:

  * `Failure`, tagged with the phase that raised. A replay that dies at turn 400
    of 492 keeps the probes that already fired, and without a marker those
    partial results were averaged into the headline score as though the cell had
    finished. The report now excludes them and says how many it dropped.
  * `EvidenceScore`, when the corpus ships gold annotations. Free -- the
    backends already report which turns they handed over.
"""

from __future__ import annotations

import time
import traceback
from collections import defaultdict
from pathlib import Path

from . import agent, backends, evidence, judge, orchestrators  # noqa: F401  (registration)
from .llm import Ledger, ModelConfig
from .memory import get_backend
from .orchestration import DEFAULT_MAX_HOPS, get_orchestrator
from .types import Failure, ProbeResult, RunResult, Task


class _PhaseError(Exception):
    """Internal: a phase raised, and the replay cannot continue past it."""

    def __init__(self, where: str, detail: str) -> None:
        super().__init__(detail)
        self.where = where
        self.detail = detail


def _guard(where: str, fn, *args, **kwargs):
    """Run one phase, tagging anything it raises with the phase that raised it."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad; it is recorded
        raise _PhaseError(where, traceback.format_exc(limit=4)) from exc


def run_cell(
    backend_name: str,
    task: Task,
    *,
    config: ModelConfig,
    orchestrator: str = "single_shot",
    trial: int = 0,
    token_budget: int = 2000,
    max_hops: int = DEFAULT_MAX_HOPS,
    cache_dir: Path = Path("results/cache"),
    on_probe=None,
    llm_factory=None,
) -> RunResult:
    """Replay one task against one backend under one orchestrator.

    `llm_factory` takes a Ledger and returns an LLM-shaped object; tests pass a
    stub so the whole pipeline can run without touching a model at all.

    `trial` salts the response cache for everything except the judge, so repeats
    resample the agent instead of replaying trial 0. Trial 0 is unsalted, which
    keeps every previously cached response valid.
    """
    ledger = Ledger()
    llm = (
        llm_factory(ledger)
        if llm_factory
        else config.build(ledger, cache_dir, trial=trial)
    )
    result = RunResult(
        backend=backend_name,
        task_id=task.task_id,
        orchestrator=orchestrator,
        trial=trial,
    )
    expected = len(task.probes)
    started = time.perf_counter()
    turn_id: int | None = None

    probes_by_turn: dict[int, list] = defaultdict(list)
    for probe in task.probes:
        probes_by_turn[probe.after_turn].append(probe)

    try:
        # Lambdas so that an unknown name is caught here too, rather than
        # escaping as a bare KeyError from the registry lookup.
        store = _guard(
            "setup", lambda: get_backend(backend_name)(llm, token_budget=token_budget)
        )
        driver = _guard(
            "setup", lambda: get_orchestrator(orchestrator)(llm, max_hops=max_hops)
        )

        for index, event in enumerate(task.events):
            turn_id = event.turn_id
            _guard("write", store.observe, event)

            for probe in probes_by_turn.get(event.turn_id, []):
                attempt = _guard(
                    "read", driver.run, store, probe.question, as_of=probe.as_of
                )
                grade, reason, hard_fail = _guard(
                    "judge", judge.grade, llm, probe, attempt.text
                )

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
                    stop=attempt.stop,
                    # Only turns observed so far are retrievable, so that is
                    # what retrieval is graded against.
                    evidence=evidence.score(probe, task.events[: index + 1], attempt),
                    trial=trial,
                )
                result.probes.append(pr)
                if on_probe:
                    on_probe(pr)

        result.store_stats = _guard("write", store.stats)
    except _PhaseError as exc:
        result.failure = Failure(
            where=exc.where,  # type: ignore[arg-type]
            detail=exc.detail,
            turn_id=turn_id,
            probes_done=len(result.probes),
            probes_expected=expected,
        )

    result.usage = ledger.records
    result.wall_s = time.perf_counter() - started
    return result


def run_matrix(
    backends: list[str],
    tasks: list[Task],
    *,
    config: ModelConfig,
    orchestrators: list[str] | None = None,
    trials: int = 1,
    token_budget: int = 2000,
    max_hops: int = DEFAULT_MAX_HOPS,
    cache_dir: Path = Path("results/cache"),
    on_cell=None,
    on_probe=None,
    llm_factory=None,
) -> list[RunResult]:
    """Run the full matrix, `trials` times over.

    Trials are the outermost loop deliberately: one complete pass exists before
    any cell is repeated, so an interrupted run still has a balanced matrix to
    report rather than three trials of the first backend and none of the last.
    """
    results: list[RunResult] = []
    for trial in range(max(trials, 1)):
        for orchestrator in orchestrators or ["single_shot"]:
            for backend_name in backends:
                for task in tasks:
                    cell = run_cell(
                        backend_name,
                        task,
                        config=config,
                        orchestrator=orchestrator,
                        trial=trial,
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
