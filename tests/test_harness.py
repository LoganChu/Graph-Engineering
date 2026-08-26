"""Task loading, grading, cost accounting, and a full offline matrix run."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from arena import backends, report, runner, tasks  # noqa: F401
from arena.judge import grade, score
from arena.llm import Ledger, ModelConfig, cost_usd
from arena.types import Probe, Usage

from .conftest import StubLLM

# The hand-authored pair. The default corpus is now LongMemEval, generated into
# tasks/longmemeval/ and gitignored, so it cannot be what these tests assert on
# -- and `as_of` only ever appears here, since imported probes carry no date.
TASK_DIR = Path(__file__).resolve().parents[1] / "tasks" / "handwritten"


class TestTaskLoading:
    def test_every_shipped_task_loads(self) -> None:
        loaded = tasks.load_all(TASK_DIR)
        assert loaded, "no task files found"
        for task in loaded:
            assert task.events and task.probes

    def test_probe_positions_are_validated(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "id: bad\n"
            "turns:\n"
            "  - {t: 2025-01-01, text: hello}\n"
            "probes:\n"
            "  - {after_turn: 9, type: simple_recall, question: q, expected: a}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="outside"):
            tasks.load_task(bad)

    def test_temporal_probes_parse_as_of(self) -> None:
        task = tasks.load_task(TASK_DIR / "relocation.yaml")
        temporal = [p for p in task.probes if p.type == "temporal"]
        assert temporal and all(p.as_of is not None for p in temporal)


class TestGrading:
    def test_banned_string_fails_before_the_judge_runs(self) -> None:
        llm = StubLLM()
        probe = Probe(
            probe_id="p1",
            after_turn=1,
            type="contradiction",
            question="Where do I live?",
            expected="Chapel Hill",
            must_not_contain=("Durham",),
        )
        verdict, reason, hard = grade(llm, probe, "You live in Durham.")
        assert verdict == "incorrect" and hard
        assert not llm.ledger.by_phase("judge"), "should not spend a judge call"

    def test_partial_credit(self) -> None:
        assert score(["correct", "partial", "incorrect"]) == pytest.approx(0.5)
        assert score([]) == 0.0


class TestCostAccounting:
    def test_cache_reads_are_cheaper_than_fresh_input(self) -> None:
        fresh = Usage(phase="read", model="claude-opus-5", input_tokens=1000, output_tokens=0)
        cached = Usage(
            phase="read",
            model="claude-opus-5",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=1000,
        )
        assert cost_usd(cached) == pytest.approx(cost_usd(fresh) * 0.10)

    def test_ledger_separates_phases(self) -> None:
        ledger = Ledger()
        ledger.add(Usage(phase="write", model="claude-opus-5", input_tokens=100, output_tokens=10))
        ledger.add(Usage(phase="answer", model="claude-opus-5", input_tokens=200, output_tokens=20))
        assert ledger.tokens("write") == (100, 10)
        assert ledger.cost() > ledger.cost("write") > 0


class TestOfflineMatrix:
    """The end-to-end path, with the API stubbed out."""

    BACKENDS = ["full_transcript", "bm25", "window_summary"]

    def test_matrix_runs_and_summarizes(self, tiny_task, stub_factory, tmp_path) -> None:
        results = runner.run_matrix(
            self.BACKENDS,
            [tiny_task],
            config=ModelConfig(model="stub"),
            cache_dir=tmp_path,
            llm_factory=stub_factory,
        )
        assert len(results) == len(self.BACKENDS)
        assert all(r.error is None for r in results), [r.error for r in results]
        assert all(len(r.probes) == 1 for r in results)

        summary = report.summarize(results)
        assert set(summary) == set(self.BACKENDS)
        for metrics in summary.values():
            assert 0.0 <= metrics["score"] <= 1.0
            assert metrics["n_probes"] == 1

        report.save(results, summary, tmp_path / "out")
        assert (tmp_path / "out" / "summary.json").exists()
        assert (tmp_path / "out" / "table.md").exists()

    def test_probes_only_see_turns_observed_so_far(self, stub_factory, tmp_path) -> None:
        """A probe at turn 2 must not see turn 3."""
        from arena.types import Event, Probe, Task

        task = Task(
            task_id="ordering",
            description="",
            events=(
                Event(turn_id=1, speaker="user", text="alpha", at=date(2025, 1, 1)),
                Event(turn_id=2, speaker="user", text="bravo", at=date(2025, 1, 2)),
                Event(turn_id=3, speaker="user", text="charlie", at=date(2025, 1, 3)),
            ),
            probes=(
                Probe(
                    probe_id="ordering::1",
                    after_turn=2,
                    type="simple_recall",
                    question="what was said",
                    expected="alpha and bravo",
                ),
            ),
        )
        result = runner.run_cell(
            "full_transcript",
            task,
            config=ModelConfig(model="stub"),
            token_budget=2000,
            cache_dir=tmp_path,
            llm_factory=stub_factory,
        )
        # StubLLM echoes the retrieved excerpt as the answer.
        answer = result.probes[0].answer
        assert "bravo" in answer and "charlie" not in answer
