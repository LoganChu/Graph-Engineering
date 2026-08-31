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

# Both corpora are generated and gitignored, so nothing is *shipped* to assert
# on. What these tests can still guarantee is that whatever has been built
# locally loads -- and, offline, that a synthetic task round-trips the loader.
# Per-corpus conversion is covered in test_longmemeval.py and test_locomo.py.
CORPORA = [
    Path(__file__).resolve().parents[1] / "tasks" / name
    for name in ("longmemeval", "locomo")
]


class TestTaskLoading:
    @pytest.mark.parametrize("corpus", CORPORA, ids=lambda p: p.name)
    def test_a_built_corpus_loads(self, corpus: Path) -> None:
        """Opportunistic: skips when that corpus has not been built here."""
        if not corpus.is_dir() or not any(corpus.glob("*.yaml")):
            pytest.skip(f"{corpus.name} not built -- see scripts/build_{corpus.name}.py")
        loaded = tasks.load_all(corpus)
        assert loaded, "no task files found"
        for task in loaded:
            assert task.events and task.probes
            assert all(1 <= p.after_turn <= len(task.events) for p in task.probes)

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

    def test_a_synthetic_task_round_trips_offline(self, tmp_path: Path) -> None:
        """The one loader test that needs neither a network nor a built corpus."""
        path = tmp_path / "t.yaml"
        path.write_text(
            """
id: t
turns:
  - {t: 2025-01-01, speaker: user, text: hello, session: s1, ref: "s1:0"}
  - {t: 2025-01-02, speaker: user, text: goodbye, session: s2, ref: "s2:0"}
probes:
  - after_turn: 2
    type: negation
    question: q
    expected: not stated
    gold_sessions: [s1]
    gold_turns: ["s1:0"]
    must_not_contain: [Durham]
""".lstrip(),
            encoding="utf-8",
        )
        task = tasks.load_task(path)
        assert [e.session_id for e in task.events] == ["s1", "s2"]
        assert [e.ref for e in task.events] == ["s1:0", "s2:0"]
        probe = task.probes[0]
        assert probe.gold_sessions == ("s1",)
        assert probe.gold_turns == ("s1:0",)
        assert probe.must_not_contain == ("Durham",)

    def test_gold_turns_naming_nothing_in_the_transcript_are_rejected(
        self, tmp_path: Path
    ) -> None:
        """Otherwise the intersection in `evidence.score` quietly grades the
        probe against whatever survived, reporting a recall that asks less than
        it claims to."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            """
id: bad
turns:
  - {t: 2025-01-01, text: hello, session: s1, ref: "s1:0"}
probes:
  - after_turn: 1
    type: simple_recall
    question: q
    expected: a
    gold_turns: ["s1:7"]
""".lstrip(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="not in the transcript"):
            tasks.load_task(bad)

    def test_a_task_with_no_refs_at_all_still_loads(self, tmp_path: Path) -> None:
        """Hand-authored files predate all of this and are graded on the answer."""
        path = tmp_path / "t.yaml"
        path.write_text(
            """
id: t
turns:
  - {t: 2025-01-01, text: hello}
probes:
  - {after_turn: 1, type: simple_recall, question: q, expected: a}
""".lstrip(),
            encoding="utf-8",
        )
        task = tasks.load_task(path)
        assert task.events[0].ref == ""
        assert task.probes[0].gold_turns == ()




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
