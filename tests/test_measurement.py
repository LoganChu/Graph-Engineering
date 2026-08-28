"""The measurement layer: uncertainty, reliability, evidence, and failure.

These cover the parts of the harness that decide what a number *means* rather
than what it is. Each one guards a way the report used to be quietly wrong: a
score printed without an interval, a partial cell averaged in as though it had
finished, a repeat trial replaying the first trial's cached answer.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from arena import evidence, report, runner
from arena.llm import LLM, ModelConfig
from arena.orchestration import Retriever
from arena.types import (
    Attempt,
    Event,
    Failure,
    Probe,
    ProbeResult,
    Recall,
    RunResult,
    Stat,
    Task,
)


def make_event(turn_id: int, session: str, text: str = "hello there") -> Event:
    return Event(
        turn_id=turn_id,
        speaker="user",
        text=text,
        at=date(2025, 1, 1),
        session_id=session,
    )


def make_probe(probe_id: str = "p1", gold: tuple[str, ...] = ("s1",)) -> Probe:
    return Probe(
        probe_id=probe_id,
        after_turn=1,
        type="simple_recall",
        question="q",
        expected="a",
        gold_sessions=gold,
    )


def graded(probe_id: str, task_id: str, grade: str, trial: int = 0) -> ProbeResult:
    return ProbeResult(
        task_id=task_id,
        probe=make_probe(probe_id),
        answer="a",
        grade=grade,
        reason="",
        trial=trial,
    )


class TestStat:
    def test_single_observation_has_unknown_spread(self):
        stat = Stat.over([0.7])
        assert stat.mean == pytest.approx(0.7)
        assert stat.n == 1
        # Not zero. One sample tells you nothing about the variance, and
        # rendering +/-0.00 would claim certainty the run does not have.
        assert math.isnan(stat.stderr)
        assert "?" in stat.render()

    def test_stderr_shrinks_as_observations_accumulate(self):
        few = Stat.over([0.0, 1.0, 0.0, 1.0])
        many = Stat.over([0.0, 1.0] * 32)
        assert many.stderr < few.stderr

    def test_empty_is_zero_not_an_error(self):
        assert Stat.over([]).n == 0


class TestScoreStat:
    def test_clusters_by_task_not_by_probe(self):
        """Two probes in one task are one observation, not two.

        The same eight grades split across eight tasks carry far more
        information than split across two, and the interval has to say so.
        """
        spread = [
            graded(f"p{i}", f"task{i}", "correct" if i % 2 else "incorrect")
            for i in range(8)
        ]
        clumped = [
            graded(f"p{i}", f"task{i % 2}", "correct" if i % 2 else "incorrect")
            for i in range(8)
        ]
        assert report.score_stat(spread).n == 8
        assert report.score_stat(clumped).n == 2
        assert report.score_stat(clumped).stderr > report.score_stat(spread).stderr

    def test_trials_are_averaged_within_a_probe_first(self):
        """Three trials of one probe is still one observation of one task."""
        probes = [graded("p1", "t1", "correct", trial=i) for i in range(3)]
        stat = report.score_stat(probes)
        assert stat.n == 1
        assert stat.mean == pytest.approx(1.0)

    def test_partial_credit_is_honoured(self):
        stat = report.score_stat([graded("p1", "t1", "partial")])
        assert stat.mean == pytest.approx(0.5)


class TestPassK:
    def test_one_miss_breaks_the_streak(self):
        probes = [
            graded("p1", "t1", "correct", trial=0),
            graded("p1", "t1", "correct", trial=1),
            graded("p2", "t2", "correct", trial=0),
            graded("p2", "t2", "incorrect", trial=1),
        ]
        value, n = report.pass_k(probes, k=2)
        assert n == 2
        assert value == pytest.approx(0.5)

    def test_partial_does_not_count_as_success(self):
        """pass^k is strict on purpose -- half credit is a reporting choice."""
        probes = [graded("p1", "t1", "partial", trial=i) for i in range(2)]
        value, _ = report.pass_k(probes, k=2)
        assert value == 0.0

    def test_probes_missing_from_a_trial_are_skipped(self):
        probes = [
            graded("p1", "t1", "correct", trial=0),
            graded("p1", "t1", "correct", trial=1),
            graded("p2", "t2", "correct", trial=0),  # never ran in trial 1
        ]
        value, n = report.pass_k(probes, k=2)
        assert (value, n) == (1.0, 1)


class TestEvidence:
    events = (
        make_event(1, "s1"),
        make_event(2, "s1"),
        make_event(3, "s2"),
        make_event(4, "s3"),
    )

    def attempt(self, provenance: tuple[str, ...]) -> Attempt:
        return Attempt(text="a", provenance=provenance)

    def test_perfect_retrieval(self):
        score = evidence.score(
            make_probe(gold=("s1",)), self.events, self.attempt(("1", "2"))
        )
        assert score is not None
        assert score.recall == 1.0
        assert score.precision == 1.0
        assert score.efficiency == 1.0

    def test_precision_falls_when_extra_sessions_come_along(self):
        score = evidence.score(
            make_probe(gold=("s1",)), self.events, self.attempt(("1", "3", "4"))
        )
        assert score is not None
        assert score.recall == 1.0
        assert score.precision == pytest.approx(1 / 3)
        assert 0 < score.efficiency < 1

    def test_retrieving_nothing_is_a_real_result_not_an_ungradeable_one(self):
        score = evidence.score(make_probe(gold=("s1",)), self.events, self.attempt(()))
        assert score is not None
        assert score.recall == 0.0

    def test_unauditable_backend_scores_none(self):
        """`agent_notes` reports ('notes',): there is no turn to trace back to."""
        score = evidence.score(
            make_probe(gold=("s1",)), self.events, self.attempt(("notes",))
        )
        assert score is None

    def test_no_gold_annotation_scores_none(self):
        score = evidence.score(make_probe(gold=()), self.events, self.attempt(("1",)))
        assert score is None

    def test_gold_is_restricted_to_what_had_been_observed(self):
        """A probe fired before its evidence arrives is not graded on missing it."""
        early = self.events[:1]  # only session s1 seen so far
        assert evidence.score(make_probe(gold=("s3",)), early, self.attempt(("1",))) is None

    def test_aggregate_micro_averages(self):
        scores = [
            evidence.score(make_probe(gold=("s1",)), self.events, self.attempt(("1",))),
            evidence.score(make_probe(gold=("s2",)), self.events, self.attempt(("1",))),
        ]
        agg = evidence.aggregate([s for s in scores if s])
        assert agg["evidence_recall"] == pytest.approx(0.5)
        assert agg["evidence_n"] == 2


class TestRetrieverProvenance:
    class FakeStore:
        def __init__(self) -> None:
            self.calls = 0

        def recall(self, query, as_of=None):
            self.calls += 1
            return Recall(context="ctx", provenance=("1", "2") if self.calls == 1 else ("2", "3"))

    def test_provenance_is_unioned_across_hops_in_first_seen_order(self):
        retriever = Retriever(self.FakeStore())
        retriever.fetch("one")
        retriever.fetch("two")
        assert retriever.finish("answer").provenance == ("1", "2", "3")

    def test_stop_reason_is_carried(self):
        retriever = Retriever(self.FakeStore())
        retriever.fetch("one")
        assert retriever.finish("answer", stop="hop_cap").stop == "hop_cap"


class TestTrialCacheSalt:
    def test_trial_zero_is_unsalted(self, tmp_path):
        """Every response cached before trials existed has to keep hitting."""
        base = LLM(cache_dir=tmp_path, trial=0)
        payload = {"prompt": "hello"}
        assert base._key(payload, "answer") == base._key(payload)

    def test_later_trials_miss_the_cache(self, tmp_path):
        payload = {"prompt": "hello"}
        first = LLM(cache_dir=tmp_path, trial=0)._key(payload, "answer")
        second = LLM(cache_dir=tmp_path, trial=1)._key(payload, "answer")
        assert first != second

    def test_the_judge_is_never_salted(self, tmp_path):
        """The same answer must get the same grade in every trial."""
        payload = {"prompt": "grade this"}
        first = LLM(cache_dir=tmp_path, trial=0)._key(payload, "judge")
        second = LLM(cache_dir=tmp_path, trial=3)._key(payload, "judge")
        assert first == second


class ExplodingStore:
    """A backend that dies partway through the replay."""

    name = "exploding"
    blurb = ""

    def __init__(self, llm, token_budget=2000, die_at: int = 2) -> None:
        self.die_at = die_at

    def observe(self, event) -> None:
        if event.turn_id == self.die_at:
            raise RuntimeError("store blew up")

    def recall(self, query, as_of=None) -> Recall:
        return Recall(context="ctx", provenance=("1",))

    def stats(self) -> dict:
        return {}


class TestFailure:
    def task(self) -> Task:
        return Task(
            task_id="t",
            description="",
            events=(make_event(1, "s1"), make_event(2, "s1"), make_event(3, "s1")),
            probes=(
                Probe(probe_id="t::1", after_turn=1, type="simple_recall",
                      question="q", expected="a"),
                Probe(probe_id="t::2", after_turn=3, type="simple_recall",
                      question="q", expected="a"),
            ),
        )

    def test_a_dead_replay_is_recorded_with_its_phase_and_position(
        self, monkeypatch, stub_factory, tmp_path
    ):
        monkeypatch.setattr(
            runner, "get_backend", lambda name: ExplodingStore
        )
        cell = runner.run_cell(
            "exploding",
            self.task(),
            config=ModelConfig(),
            cache_dir=tmp_path,
            llm_factory=stub_factory,
        )
        assert cell.failure is not None
        assert cell.failure.where == "write"
        assert cell.failure.turn_id == 2
        # One probe fired before the crash: that is `partial`, not `complete`,
        # and it is the case that used to be averaged in silently.
        assert cell.outcome == "partial"
        assert cell.failure.probes_done == 1
        assert cell.failure.probes_expected == 2

    def test_partial_cells_are_excluded_from_the_score(self):
        good = RunResult(backend="bm25", task_id="a")
        good.probes = [graded("p1", "a", "correct")]
        bad = RunResult(backend="bm25", task_id="b")
        bad.probes = [graded("p2", "b", "correct")]
        bad.failure = Failure(
            where="write", detail="boom", turn_id=9, probes_done=1, probes_expected=4
        )

        summary = report.summarize([good, bad])["bm25"]
        assert summary["n_probes"] == 1
        assert summary["n_incomplete"] == 1
        assert summary["incomplete"][0]["turn_id"] == 9

    def test_the_report_says_what_it_dropped(self):
        bad = RunResult(backend="bm25", task_id="b")
        bad.failure = Failure(
            where="read", detail="boom", probes_done=0, probes_expected=3
        )
        rendered = "\n".join(report.incomplete_note(report.summarize([bad])))
        assert "Cells excluded" in rendered
        assert "read phase" in rendered


class TestMarkdown:
    def test_scores_never_print_without_an_interval(self):
        cells = [
            RunResult(backend="bm25", task_id=f"t{i}") for i in range(4)
        ]
        for i, cell in enumerate(cells):
            cell.probes = [graded(f"p{i}", f"t{i}", "correct" if i % 2 else "incorrect")]
        rendered = report.to_markdown(report.summarize(cells), priced=False)
        assert "+/-" in rendered

    def test_reliability_table_appears_only_with_repeats(self):
        single = RunResult(backend="bm25", task_id="t")
        single.probes = [graded("p1", "t", "correct")]
        assert report.reliability_table(report.summarize([single])) == []

        repeated = [RunResult(backend="bm25", task_id="t", trial=i) for i in range(2)]
        for i, cell in enumerate(repeated):
            cell.probes = [graded("p1", "t", "correct" if i == 0 else "incorrect", trial=i)]
        rendered = "\n".join(report.reliability_table(report.summarize(repeated)))
        assert "pass^2" in rendered
        assert "0.00" in rendered  # the probe failed one of its two trials
