"""LoCoMo conversion, which has to stay correct without a network.

Two of these guard claims the corpus is being imported *for*. The distractor
test protects the only source of `must_not_contain` any corpus here has ever
supplied; the chronology test protects `temporal_graph` from having its
supersession inverted by an out-of-order replay.
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from arena.locomo import (
    ABSTAIN,
    CATEGORIES,
    parse_date,
    session_of,
    subset,
    to_task,
    turn_text,
)
from arena.tasks import load_task


def turn(dia_id: str, speaker: str = "Caroline", text: str = "hello there", **over):
    return {"dia_id": dia_id, "speaker": speaker, "text": text} | over


def instance(**over):
    base = {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            # Declared out of order on purpose: session 1 is the *later* one.
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [turn("D1:1", text="later turn")],
            "session_2_date_time": "9:30 am on 3 January, 2023",
            "session_2": [turn("D2:1", text="earlier turn")],
        },
        "qa": [
            {
                "question": "What did Caroline research?",
                "answer": "Adoption agencies",
                "evidence": ["D2:1"],
                "category": 4,
            }
        ],
    }
    return base | over


class TestParseDate:
    def test_reads_the_wall_clock_format(self):
        assert parse_date("1:56 pm on 8 May, 2023") == date(2023, 5, 8)

    def test_tolerates_a_missing_comma(self):
        assert parse_date("9:30 am on 3 January 2024") == date(2024, 1, 3)

    def test_rejects_junk(self):
        with pytest.raises(ValueError):
            parse_date("some time last spring")

    def test_rejects_an_unknown_month(self):
        with pytest.raises(ValueError):
            parse_date("1:00 pm on 8 Smarch, 2023")


class TestTurnText:
    def test_plain_turn_is_unchanged(self):
        assert turn_text(turn("D1:1", text="hello")) == "hello"

    def test_image_caption_is_folded_in(self):
        got = turn_text(turn("D1:1", text="look!", blip_caption="a red bicycle"))
        assert "look!" in got and "a red bicycle" in got

    def test_image_only_turn_still_carries_content(self):
        """A fifth of LoCoMo turns are images. Dropping them would put holes in
        the transcript that every backend then gets scored on."""
        got = turn_text({"dia_id": "D1:1", "text": "", "blip_caption": "a sunset"})
        assert "a sunset" in got

    def test_empty_turn_stays_empty(self):
        assert turn_text({"dia_id": "D1:1", "text": "  "}) == ""


class TestSessionOf:
    def test_splits_the_dia_id(self):
        assert session_of("D12:34") == "D12"


class TestToTask:
    def test_sessions_are_replayed_in_date_order(self):
        """The whole reason this function sorts. Out-of-order replay makes
        temporal_graph retire the new fact with the old one."""
        task = to_task(instance())
        assert [t["text"] for t in task["turns"]] == ["earlier turn", "later turn"]
        assert [t["t"] for t in task["turns"]] == [date(2023, 1, 3), date(2023, 5, 8)]

    def test_turns_carry_their_session_id(self):
        task = to_task(instance())
        assert {t["session"] for t in task["turns"]} == {"D1", "D2"}

    def test_gold_sessions_come_from_evidence_dia_ids(self):
        task = to_task(instance())
        assert task["probes"][0]["gold_sessions"] == ["D2"]

    def test_turns_carry_their_dia_id_as_the_ref(self):
        """LoCoMo needs no synthesised turn id: `evidence` already names these."""
        task = to_task(instance())
        assert [t["ref"] for t in task["turns"]] == ["D2:1", "D1:1"]

    def test_evidence_rides_through_at_turn_grain(self):
        task = to_task(instance())
        assert task["probes"][0]["gold_turns"] == ["D2:1"]

    def test_unresolvable_evidence_is_dropped(self):
        """9 of LoCoMo's 2,815 evidence ids point at turns that are not in the
        transcript. Grading a store on retrieving those is grading it against
        something no backend could satisfy."""
        inst = instance()
        inst["qa"][0]["evidence"] = ["D2:1", "D99:1"]
        probe = to_task(inst)["probes"][0]
        assert probe["gold_sessions"] == ["D2"]
        assert probe["gold_turns"] == ["D2:1"]

    def test_a_turn_dropped_for_being_empty_cannot_be_gold(self):
        """Image-less, text-less turns never reach the store, so gold must not
        name them -- the loader rejects a probe that does."""
        inst = instance()
        inst["conversation"]["session_2"] = [turn("D2:1", text="")]
        probe = to_task(inst)["probes"][0]
        assert probe["gold_turns"] == []

    def test_probes_fire_after_every_turn(self):
        task = to_task(instance())
        assert all(p["after_turn"] == len(task["turns"]) for p in task["probes"])

    @pytest.mark.parametrize("category,expected", sorted(CATEGORIES.items()))
    def test_every_mapped_category_survives(self, category, expected):
        inst = instance()
        inst["qa"][0]["category"] = category
        inst["qa"][0]["adversarial_answer"] = "the trap"
        probes = to_task(inst)["probes"]
        assert [p["type"] for p in probes] == [expected]

    def test_open_domain_questions_are_dropped(self):
        """Category 3 needs world knowledge, and the agent is forced
        closed-book -- scoring them would measure prompt compliance."""
        inst = instance()
        inst["qa"][0]["category"] = 3
        assert to_task(inst)["probes"] == []

    def test_unknown_category_is_dropped_not_crashed(self):
        inst = instance()
        inst["qa"][0]["category"] = 99
        assert to_task(inst)["probes"] == []


class TestAdversarialProbes:
    """Category 5 is the only source of `must_not_contain` in any corpus here.

    LoCoMo's own evaluator scores these by checking the output for an abstention
    phrase, so the correct behaviour is to decline -- which makes
    `adversarial_answer` the string a correct answer must not contain.
    """

    def adversarial(self):
        inst = instance()
        inst["qa"][0] = {
            "question": "What did Melanie realize after her charity race?",
            "adversarial_answer": "self-care is important",
            "evidence": ["D2:1"],
            "category": 5,
        }
        return to_task(inst)["probes"][0]

    def test_maps_to_negation(self):
        assert self.adversarial()["type"] == "negation"

    def test_the_trap_answer_becomes_a_distractor(self):
        assert self.adversarial()["must_not_contain"] == ["self-care is important"]

    def test_expected_answer_is_an_abstention(self):
        assert self.adversarial()["expected"] == ABSTAIN

    def test_missing_distractor_is_tolerated(self):
        inst = instance()
        inst["qa"][0] = {
            "question": "q",
            "evidence": [],
            "category": 5,
        }
        probe = to_task(inst)["probes"][0]
        assert "must_not_contain" not in probe
        assert probe["expected"] == ABSTAIN

    def test_the_guard_actually_fires_end_to_end(self, tmp_path):
        """Round-trip through YAML and the loader, since that is the path a
        real run takes."""
        inst = instance()
        inst["qa"][0] = {
            "question": "q",
            "adversarial_answer": "Durham",
            "evidence": ["D2:1"],
            "category": 5,
        }
        path = tmp_path / "conv-1.yaml"
        path.write_text(yaml.safe_dump(to_task(inst), sort_keys=False), encoding="utf-8")
        probe = load_task(path).probes[0]
        assert probe.must_not_contain == ("Durham",)
        assert probe.type == "negation"


class TestSubset:
    def make(self, n_per_category=10):
        qa = [
            {"question": f"q{c}-{i}", "answer": "a", "evidence": ["D2:1"], "category": c}
            for c in CATEGORIES
            for i in range(n_per_category)
        ]
        return [instance(qa=qa), instance(sample_id="conv-2", qa=list(qa))]

    def test_conversation_cap_is_the_expensive_axis(self):
        assert len(subset(self.make(), conversations=1)) == 1

    def test_probe_cap_keeps_every_type(self):
        """A cap must never silently delete a whole probe category -- the
        by-type table is the point of the corpus."""
        got = subset(self.make(), probes_per_conversation=8)
        kinds = {q["category"] for q in got[0]["qa"]}
        assert kinds == set(CATEGORIES)
        assert len(got[0]["qa"]) == 8

    def test_cap_above_supply_keeps_everything(self):
        got = subset(self.make(n_per_category=2), probes_per_conversation=999)
        assert len(got[0]["qa"]) == 2 * len(CATEGORIES)

    def test_is_deterministic_for_a_seed(self):
        a = subset(self.make(), probes_per_conversation=6, seed=7)
        b = subset(self.make(), probes_per_conversation=6, seed=7)
        assert [q["question"] for q in a[0]["qa"]] == [
            q["question"] for q in b[0]["qa"]
        ]

    def test_no_cap_returns_the_corpus_untouched(self):
        made = self.make()
        assert subset(made) == made


class TestLoadsThroughTheHarness:
    def test_round_trips_to_a_real_task(self, tmp_path):
        path = tmp_path / "conv-1.yaml"
        path.write_text(yaml.safe_dump(to_task(instance()), sort_keys=False), encoding="utf-8")
        task = load_task(path)
        assert task.task_id == "conv-1"
        assert [e.text for e in task.events] == ["earlier turn", "later turn"]
        assert task.events[0].session_id == "D2"
        assert task.events[0].ref == "D2:1"
        assert task.probes[0].gold_sessions == ("D2",)
        assert task.probes[0].gold_turns == ("D2:1",)
