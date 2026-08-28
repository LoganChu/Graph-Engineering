"""Conversion is the part that has to stay correct without a network.

The chronology test is the one that matters: LongMemEval does not ship sessions
in date order, and `temporal_graph` stamps `valid_to` from the event date when a
fact is superseded. Replay out of order and it retires the new fact with the old
one -- silently, and in exactly the probe category the backend exists to win.
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from arena.longmemeval import QUESTION_TYPES, parse_date, probe_type, stratified, to_task
from arena.tasks import load_task


def instance(**over):
    base = {
        "question_id": "gpt4_deadbeef",
        "question_type": "single-session-user",
        "question": "What degree did I graduate with?",
        "answer": "Business Administration",
        "question_date": "2023/06/01 (Thu) 09:00",
        "haystack_session_ids": ["s_a", "s_b"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/01/03 (Tue) 11:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "later turn", "has_answer": True}],
            [{"role": "assistant", "content": "earlier turn"}],
        ],
        "answer_session_ids": ["s_a"],
    }
    return base | over


class TestParseDate:
    def test_reads_the_slash_format(self):
        assert parse_date("2023/04/10 (Mon) 23:07") == date(2023, 4, 10)

    def test_tolerates_single_digit_fields(self):
        assert parse_date("2023/4/1 (Sat) 00:00") == date(2023, 4, 1)

    def test_rejects_junk(self):
        with pytest.raises(ValueError):
            parse_date("last tuesday")


class TestProbeType:
    @pytest.mark.parametrize("kind,expected", sorted(QUESTION_TYPES.items()))
    def test_every_dataset_type_maps(self, kind, expected):
        assert probe_type(instance(question_type=kind)) == expected

    def test_abstention_overrides_the_base_type(self):
        got = probe_type(instance(question_id="gpt4_x_abs", question_type="multi-session"))
        assert got == "negation"

    def test_unmapped_type_is_loud(self):
        with pytest.raises(ValueError):
            probe_type(instance(question_type="brand-new-category"))


class TestToTask:
    def test_sessions_are_replayed_in_date_order(self):
        task = to_task(instance())
        assert [t["text"] for t in task["turns"]] == ["earlier turn", "later turn"]
        assert [t["t"] for t in task["turns"]] == [date(2023, 1, 3), date(2023, 5, 20)]

    def test_role_becomes_speaker(self):
        task = to_task(instance())
        assert [t["speaker"] for t in task["turns"]] == ["assistant", "user"]

    def test_probe_fires_at_the_final_turn(self):
        task = to_task(instance())
        assert task["probes"][0]["after_turn"] == len(task["turns"]) == 2

    def test_integer_answers_survive_as_strings(self):
        task = to_task(instance(answer=3))
        assert task["probes"][0]["expected"] == "3"

    def test_no_guard_is_invented(self):
        assert "must_not_contain" not in to_task(instance())["probes"][0]

    def test_mismatched_dates_and_sessions_are_rejected(self):
        with pytest.raises(ValueError):
            to_task(instance(haystack_dates=["2023/05/20 (Sat) 02:21"]))

    def test_description_counts_evidence_sessions(self):
        assert "1 carrying evidence" in to_task(instance())["description"]

    def test_round_trips_through_the_real_loader(self, tmp_path):
        """The converter's only contract is that `tasks.load_task` accepts it."""
        task = to_task(instance())
        path = tmp_path / "t.yaml"
        path.write_text(
            yaml.safe_dump(task, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        loaded = load_task(path)
        assert loaded.task_id == "gpt4_deadbeef"
        assert [e.text for e in loaded.events] == ["earlier turn", "later turn"]
        assert loaded.events[0].at == date(2023, 1, 3)
        assert loaded.probes[0].type == "simple_recall"
        assert loaded.probes[0].must_not_contain == ()


class TestStratified:
    def _corpus(self):
        out = []
        for kind in ("single-session-user", "multi-session", "knowledge-update"):
            for i in range(20):
                out.append(instance(question_id=f"{kind}_{i}", question_type=kind))
        out += [instance(question_id=f"neg_{i}_abs") for i in range(3)]
        return out

    def test_keeps_everything_when_under_budget(self):
        corpus = self._corpus()
        assert len(stratified(corpus, 999)) == len(corpus)
        assert len(stratified(corpus, None)) == len(corpus)

    def test_honours_the_budget(self):
        assert len(stratified(self._corpus(), 12)) == 12

    def test_spreads_across_probe_types(self):
        picked = stratified(self._corpus(), 12)
        counts = {}
        for item in picked:
            counts[probe_type(item)] = counts.get(probe_type(item), 0) + 1
        assert len(counts) == 4, counts
        # Round-robin, so no type may run more than one ahead of another until
        # its bucket is exhausted. `negation` has only three instances here.
        assert counts["negation"] == 3
        assert max(counts[k] for k in counts if k != "negation") <= 3

    def test_is_deterministic_for_a_seed(self):
        corpus = self._corpus()
        a = [i["question_id"] for i in stratified(corpus, 10, seed=7)]
        b = [i["question_id"] for i in stratified(corpus, 10, seed=7)]
        assert a == b

    def test_a_scarce_type_is_never_starved(self):
        picked = stratified(self._corpus(), 8)
        assert any(probe_type(i) == "negation" for i in picked)
