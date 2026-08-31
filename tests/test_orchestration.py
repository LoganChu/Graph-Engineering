"""The orchestration axis: control flow, hop accounting, and phase attribution.

Split in two. Everything above `TestLoop` runs on the base install, because the
accounting rules are the part that has to stay correct whether or not LangChain
is present. The framework-backed orchestrators skip when the extra is missing.
"""

from __future__ import annotations

import threading
import time
from datetime import date
from pathlib import Path

import pytest

from arena import backends, orchestrators, report, runner  # noqa: F401
from arena.llm import LLM, Ledger, ModelConfig
from arena.memory import get_backend
from arena.orchestration import Retriever, available, get_orchestrator
from arena.providers import AnthropicProvider, Completion, ToolCall
from arena.types import Event, Probe, Recall, Task

from .conftest import StubLLM

needs_langchain = pytest.mark.skipif(
    not {"loop", "graph"} <= set(available()),
    reason="needs the orchestration extra: uv sync --extra orchestration",
)


def build_store(name: str, llm, events: list[Event]):
    store = get_backend(name)(llm, token_budget=2000)
    for event in events:
        store.observe(event)
    return store


EVENTS = [
    Event(turn_id=1, speaker="user", text="I live in Durham.", at=date(2025, 1, 1)),
    Event(turn_id=2, speaker="user", text="My cat is Miso.", at=date(2025, 2, 1)),
    Event(
        turn_id=3, speaker="user", text="Miso is allergic to chicken.", at=date(2025, 3, 1)
    ),
]


class TestRegistry:
    def test_single_shot_is_always_available(self) -> None:
        """The control must not depend on an optional extra."""
        assert "single_shot" in available()

    def test_unknown_orchestrator_names_the_extra(self) -> None:
        with pytest.raises(KeyError, match="orchestration"):
            get_orchestrator("nonexistent")


class TestRetrieverAccounting:
    """Every orchestrator routes through this, so hop counts stay comparable."""

    def test_hops_and_queries_accumulate(self) -> None:
        llm = StubLLM()
        store = build_store("bm25", llm, EVENTS)
        retriever = Retriever(store)

        retriever.fetch("cat")
        retriever.fetch("Durham")
        attempt = retriever.finish("some answer")

        assert attempt.hops == 2
        assert attempt.queries == ("cat", "Durham")
        assert attempt.context_chars > 0
        assert attempt.text == "some answer"

    def test_a_fresh_retriever_starts_at_zero(self) -> None:
        llm = StubLLM()
        store = build_store("bm25", llm, EVENTS)
        assert Retriever(store).finish("x").hops == 0


class TestSingleShot:
    def test_one_hop_and_the_question_is_the_query(self) -> None:
        llm = StubLLM()
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("single_shot")(llm).run(store, "what cat do I have")

        assert attempt.hops == 1
        assert attempt.queries == ("what cat do I have",)

    def test_spends_nothing_on_control_flow(self) -> None:
        """The control has no control flow, so `orchestrate` must stay empty."""
        llm = StubLLM()
        store = build_store("bm25", llm, EVENTS)
        get_orchestrator("single_shot")(llm).run(store, "where do I live")

        assert llm.ledger.by_phase("orchestrate") == []
        assert llm.ledger.by_phase("answer")


class TestFanout:
    """Breadth without feedback -- the control the iterative rows are read against."""

    def test_the_question_is_searched_verbatim_first(self) -> None:
        """Its evidence has to be a superset of single_shot's, or the ablation
        is not measuring only the extra queries."""
        llm = StubLLM()
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("fanout")(llm).run(store, "what cat do I have")

        assert attempt.queries[0] == "what cat do I have"

    def test_it_spends_the_whole_budget_in_one_round(self) -> None:
        llm = StubLLM(rewrites=["the cat", "pet name", "allergies"])
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("fanout")(llm, max_hops=4).run(store, "cat")

        assert attempt.hops == 4, "the question plus three rewrites"
        assert attempt.queries == ("cat", "the cat", "pet name", "allergies")

    def test_one_control_flow_call_however_wide_the_fan(self) -> None:
        """The cost claim: one `orchestrate` call, not one per hop like `graph`."""
        llm = StubLLM(rewrites=["a", "b", "c", "d", "e"])
        store = build_store("bm25", llm, EVENTS)
        get_orchestrator("fanout")(llm, max_hops=6).run(store, "cat")

        assert len(llm.ledger.by_phase("orchestrate")) == 1
        assert len(llm.ledger.by_phase("answer")) == 1

    def test_a_rewrite_never_costs_more_than_the_budget(self) -> None:
        llm = StubLLM(rewrites=["a", "b", "c", "d", "e"])
        store = build_store("bm25", llm, EVENTS)
        assert get_orchestrator("fanout")(llm, max_hops=2).run(store, "cat").hops == 2

    def test_blank_and_repeated_rewrites_are_dropped(self) -> None:
        """A rewrite echoing the question would spend a hop on evidence in hand."""
        llm = StubLLM(rewrites=["  ", "CAT", "cat ", "the cat"])
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("fanout")(llm, max_hops=4).run(store, "cat")

        assert attempt.queries == ("cat", "the cat")

    def test_a_budget_of_one_pays_for_no_rewrite(self) -> None:
        """With no room to fan out it must degenerate to single_shot exactly,
        not to single_shot plus a wasted call."""
        llm = StubLLM()
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("fanout")(llm, max_hops=1).run(store, "cat")

        assert attempt.hops == 1
        assert llm.ledger.by_phase("orchestrate") == []

    def test_every_query_reaches_the_answer(self) -> None:
        llm = StubLLM(rewrites=["the cat", "pet name"])
        store = build_store("bm25", llm, EVENTS)
        get_orchestrator("fanout")(llm, max_hops=3).run(store, "cat")

        answers = [p for phase, p in llm.calls if phase == "answer"]
        assert len(answers) == 1
        assert answers[0].count("--- search ") == 3

    def test_a_spent_budget_is_not_reported_as_capped(self) -> None:
        """`hop_cap` means a policy was cut short. This one had no policy."""
        llm = StubLLM()
        store = build_store("bm25", llm, EVENTS)
        assert get_orchestrator("fanout")(llm).run(store, "cat").stop == "answered"

    def test_it_runs_without_the_orchestration_extra(self) -> None:
        assert "fanout" in available()


class TestChatPlumbing:
    """`LLM.chat` is what the frameworks run on: cache it, and bill it right."""

    class FakeProvider:
        name = "fake"

        def __init__(self, reply: Completion) -> None:
            self.reply = reply
            self.hits = 0

        def chat(self, **kwargs) -> Completion:
            self.hits += 1
            return self.reply

        def complete(self, **kwargs):  # pragma: no cover - unused here
            raise NotImplementedError

        def parse(self, **kwargs):  # pragma: no cover - unused here
            raise NotImplementedError

    def _llm(self, reply: Completion, tmp_path: Path) -> tuple[LLM, FakeProvider]:
        provider = self.FakeProvider(reply)
        return (
            LLM(model="fake", provider=provider, cache_dir=tmp_path, ledger=Ledger()),
            provider,
        )

    def test_a_tool_call_is_billed_as_control_flow(self, tmp_path: Path) -> None:
        from arena.orchestrators.adapter import phase_of

        reply = Completion(
            text="",
            input_tokens=10,
            output_tokens=5,
            tool_calls=(ToolCall(id="1", name="recall", args={"query": "x"}),),
        )
        llm, _ = self._llm(reply, tmp_path)
        llm.chat([{"role": "user", "content": "hi"}], phase=phase_of)

        assert [u.phase for u in llm.ledger.records] == ["orchestrate"]

    def test_prose_is_billed_as_the_answer(self, tmp_path: Path) -> None:
        from arena.orchestrators.adapter import phase_of

        llm, _ = self._llm(Completion(text="Durham.", input_tokens=10, output_tokens=5), tmp_path)
        llm.chat([{"role": "user", "content": "hi"}], phase=phase_of)

        assert [u.phase for u in llm.ledger.records] == ["answer"]

    def test_identical_turns_replay_from_cache(self, tmp_path: Path) -> None:
        reply = Completion(
            text="",
            input_tokens=10,
            output_tokens=5,
            tool_calls=(ToolCall(id="1", name="recall", args={"query": "x"}),),
        )
        llm, provider = self._llm(reply, tmp_path)
        messages = [{"role": "user", "content": "hi"}]

        first = llm.chat(messages, phase="answer")
        second = llm.chat(messages, phase="answer")

        assert provider.hits == 1, "second call should have replayed from disk"
        # Tool calls have to survive the round trip, or a cached loop replays as
        # a loop that never searched.
        assert second.tool_calls == first.tool_calls
        assert llm.ledger.records[1].cached_locally


class TestAnthropicTranscript:
    """Tool results are user-role blocks on this API, and must coalesce."""

    def test_parallel_tool_results_share_one_message(self) -> None:
        blocks = AnthropicProvider._blocks(
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "a", "name": "recall", "args": {"query": "1"}},
                        {"id": "b", "name": "recall", "args": {"query": "2"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "first"},
                {"role": "tool", "tool_call_id": "b", "content": "second"},
            ]
        )
        assert [b["role"] for b in blocks] == ["user", "assistant", "user"]
        assert len(blocks[2]["content"]) == 2, "two results, one user message"
        assert {b["type"] for b in blocks[2]["content"]} == {"tool_result"}

    def test_an_empty_assistant_turn_is_dropped(self) -> None:
        """Neither text nor tool calls is not a legal message; skip, don't send."""
        blocks = AnthropicProvider._blocks(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]
        )
        assert [b["role"] for b in blocks] == ["user"]


class TestReportAxis:
    def test_control_cells_keep_their_bare_name(self) -> None:
        """A default run must produce the table it always did."""
        assert report.cell_key("bm25", "single_shot") == "bm25"
        assert report.cell_key("bm25", "loop") == "bm25/loop"

    def test_the_lift_table_needs_something_to_compare(self) -> None:
        one = {"bm25": {"backend": "bm25", "orchestrator": "single_shot"}}
        assert report.orchestration_table(one) == []

    def test_control_flow_spend_lands_on_the_read_side(self) -> None:
        """`orchestrate` tokens must not vanish, and must not look like writes."""
        from arena.types import ProbeResult, RunResult, Usage

        probe = Probe(
            probe_id="p", after_turn=1, type="simple_recall", question="q", expected="a"
        )
        cell = RunResult(backend="bm25", task_id="t", orchestrator="loop")
        cell.probes = [ProbeResult(task_id="t", probe=probe, answer="a", grade="correct", reason="", hops=3)]
        cell.usage = [
            Usage(phase="orchestrate", model="claude-opus-5", input_tokens=100, output_tokens=10),
            Usage(phase="answer", model="claude-opus-5", input_tokens=50, output_tokens=10),
            Usage(phase="judge", model="claude-opus-5", input_tokens=999, output_tokens=99),
        ]

        metrics = report.summarize([cell])["bm25/loop"]

        assert metrics["orchestrate_tokens"] == 110
        assert metrics["read_tokens"] == 170, "orchestrate + answer, judge excluded"
        assert metrics["write_tokens"] == 0
        assert metrics["avg_hops"] == 3.0
        assert metrics["orchestrator"] == "loop"


@needs_langchain
class TestAdapter:
    def test_langchain_messages_become_a_transcript(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        from arena.orchestrators.adapter import to_transcript

        system, transcript = to_transcript(
            [
                SystemMessage(content="be terse"),
                HumanMessage(content="where?"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "recall", "args": {"query": "home"}, "id": "c1"}],
                ),
                ToolMessage(content="Durham", tool_call_id="c1"),
            ]
        )

        assert system == "be terse"
        assert [m["role"] for m in transcript] == ["user", "assistant", "tool"]
        assert transcript[1]["tool_calls"][0]["name"] == "recall"
        assert transcript[2]["tool_call_id"] == "c1"

    def test_tool_schemas_are_flattened_for_the_provider(self) -> None:
        from arena.orchestrators.adapter import to_tool_specs

        specs = to_tool_specs(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "recall",
                        "description": "search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )
        assert specs == [
            {
                "name": "recall",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        assert to_tool_specs(None) is None


@needs_langchain
class TestLoop:
    def test_the_model_drives_the_search(self) -> None:
        llm = StubLLM(tool_hops=1)
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("loop")(llm).run(store, "what is my cat called")

        assert attempt.hops == 1
        assert "Miso" in attempt.text, "the answer should carry the retrieved evidence"

    def test_two_searches_when_the_model_asks_for_two(self) -> None:
        llm = StubLLM(tool_hops=2)
        store = build_store("bm25", llm, EVENTS)
        assert get_orchestrator("loop")(llm).run(store, "cat").hops == 2

    def test_the_hop_cap_holds_against_a_model_that_will_not_stop(self) -> None:
        """An unbounded loop is the failure mode; the cap is what prevents it."""
        llm = StubLLM(tool_hops=99)
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("loop")(llm, max_hops=2).run(store, "cat")

        assert attempt.hops == 2
        assert "recursion limit" in attempt.note

    def test_control_flow_and_answer_are_billed_apart(self) -> None:
        llm = StubLLM(tool_hops=1)
        store = build_store("bm25", llm, EVENTS)
        get_orchestrator("loop")(llm).run(store, "cat")

        assert llm.ledger.by_phase("orchestrate"), "the tool-choice call"
        assert llm.ledger.by_phase("answer"), "the final prose call"


@needs_langchain
class TestGraph:
    def test_it_stops_when_the_evidence_is_sufficient(self) -> None:
        llm = StubLLM(assess_hops=0)
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("graph")(llm).run(store, "what is my cat called")

        assert attempt.hops == 1
        assert attempt.queries == ("what is my cat called",)

    def test_the_cycle_reformulates_and_retries(self) -> None:
        """The edge back to `retrieve` is the whole reason this is a graph."""
        llm = StubLLM(assess_hops=2)
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("graph")(llm).run(store, "what is my cat called")

        assert attempt.hops == 3
        assert attempt.queries[0] == "what is my cat called"
        assert attempt.queries[1:] == ("reformulated query 1", "reformulated query 2")

    def test_the_cap_is_a_property_of_the_graph(self) -> None:
        llm = StubLLM(assess_hops=99)
        store = build_store("bm25", llm, EVENTS)
        assert get_orchestrator("graph")(llm, max_hops=2).run(store, "cat").hops == 2

    def test_assessments_are_billed_as_control_flow(self) -> None:
        llm = StubLLM(assess_hops=1)
        store = build_store("bm25", llm, EVENTS)
        get_orchestrator("graph")(llm).run(store, "cat")

        assert len(llm.ledger.by_phase("orchestrate")) == 2, "one per cycle"

    def test_every_hop_reaches_the_answer(self) -> None:
        """Evidence accumulates across the cycle rather than being replaced."""
        llm = StubLLM(assess_hops=1)
        store = build_store("bm25", llm, EVENTS)
        get_orchestrator("graph")(llm).run(store, "cat")

        answer_prompts = [p for phase, p in llm.calls if phase == "answer"]
        assert len(answer_prompts) == 1
        assert answer_prompts[0].count("--- search ") == 2


class SlowStore:
    """A store whose earliest query is its slowest, and which records both the
    order it finished in and the threads it ran on.

    Exists so the fan-out's ordering guarantee can be tested deterministically
    instead of by running it repeatedly and hoping the race shows up.
    """

    name = "slow"

    def __init__(self, n: int, unit: float = 0.05) -> None:
        self.n = n
        self.unit = unit
        self.threads: set[int] = set()
        self.completed: list[str] = []
        self._lock = threading.Lock()

    def observe(self, event) -> None:  # pragma: no cover - never ingested
        raise NotImplementedError

    def recall(self, query: str) -> Recall:
        index = int(query.split()[-1])
        time.sleep(self.unit * (self.n - index))
        with self._lock:
            self.threads.add(threading.get_ident())
            self.completed.append(query)
        return Recall(context=f"body for {query}", provenance=(f"s{index}",))


@needs_langchain
class TestPlanExecute:
    """Fan-out/fan-in: several different questions, one join."""

    def test_each_part_gets_its_own_retrieval(self) -> None:
        llm = StubLLM(plan_parts=["where do I live", "what is my cat called"])
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("plan_execute")(llm).run(
            store, "where does my cat live"
        )

        assert attempt.hops == 2
        assert attempt.queries == ("where do I live", "what is my cat called")

    def test_branch_order_survives_reversed_completion(self) -> None:
        """The branches really do run on worker threads, so the store can
        return them in any order. If the record followed completion order, the
        queries `arena inspect` prints would reshuffle between two runs of the
        same cell -- error-analysis output that cannot be diffed is not much
        use.

        `SlowStore` makes the earliest-dispatched query the slowest, which
        reverses completion order outright rather than hoping to catch a race.
        """
        n = 4
        store = SlowStore(n)
        llm = StubLLM(plan_parts=[f"part {i}" for i in range(n)])

        attempt = get_orchestrator("plan_execute")(llm, max_hops=n).run(store, "q")

        assert store.completed == [f"part {i}" for i in reversed(range(n))], (
            "the fixture is supposed to invert completion order; if this fails "
            "the test below is passing for the wrong reason"
        )
        assert len(store.threads) > 1, "a fan-out that never forked proves nothing"
        assert attempt.queries == tuple(f"part {i}" for i in range(n))
        assert attempt.provenance == tuple(f"s{i}" for i in range(n))
        assert attempt.hops == n

    def test_the_join_reads_parts_in_plan_order(self) -> None:
        """Same argument one level up: the answer prompt is cache-keyed too."""
        store = SlowStore(3)
        llm = StubLLM(plan_parts=[f"part {i}" for i in range(3)])

        get_orchestrator("plan_execute")(llm, max_hops=3).run(store, "q")

        prompt = next(p for phase, p in llm.calls if phase == "answer")
        headers = [
            line for line in prompt.splitlines() if line.startswith("--- part ")
        ]

        assert [h.split("'")[1] for h in headers] == ["part 0", "part 1", "part 2"]

    def test_the_plan_cannot_outspend_the_budget(self) -> None:
        llm = StubLLM(plan_parts=["a", "b", "c", "d", "e"])
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("plan_execute")(llm, max_hops=3).run(store, "q")

        assert attempt.hops == 3

    def test_an_empty_plan_falls_back_to_the_question(self) -> None:
        """No branches means the join is never reached and nothing is answered.
        Degrading to single_shot is the only acceptable failure here."""
        llm = StubLLM(plan_parts=[])
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("plan_execute")(llm).run(store, "where do I live")

        assert attempt.queries == ("where do I live",)
        assert "Durham" in attempt.text

    def test_repeated_parts_are_dropped(self) -> None:
        llm = StubLLM(plan_parts=["the cat", "The Cat ", "where I live"])
        store = build_store("bm25", llm, EVENTS)
        attempt = get_orchestrator("plan_execute")(llm).run(store, "q")

        assert attempt.queries == ("the cat", "where I live")

    def test_one_plan_call_and_one_answer_call(self) -> None:
        """Branches must not answer their own sub-question: an orchestrator
        wins by assembling better evidence, not by buying extra reasoning."""
        llm = StubLLM(plan_parts=["a", "b", "c"])
        store = build_store("bm25", llm, EVENTS)
        get_orchestrator("plan_execute")(llm, max_hops=3).run(store, "q")

        assert len(llm.ledger.by_phase("orchestrate")) == 1
        assert len(llm.ledger.by_phase("answer")) == 1

    def test_every_branch_reaches_the_join(self) -> None:
        llm = StubLLM(plan_parts=["where do I live", "what is my cat called"])
        store = build_store("bm25", llm, EVENTS)
        get_orchestrator("plan_execute")(llm).run(store, "q")

        answers = [p for phase, p in llm.calls if phase == "answer"]
        assert len(answers) == 1
        assert answers[0].count("--- part ") == 2
        assert "Durham" in answers[0] and "Miso" in answers[0]


@needs_langchain
class TestMatrixAcrossBothAxes:
    def test_all_five_orchestrators_run_and_report(
        self, tiny_task: Task, stub_factory, tmp_path: Path
    ) -> None:
        orders = ["single_shot", "fanout", "loop", "graph", "plan_execute"]
        results = runner.run_matrix(
            ["bm25", "full_transcript"],
            [tiny_task],
            config=ModelConfig(model="stub"),
            orchestrators=orders,
            cache_dir=tmp_path,
            llm_factory=stub_factory,
        )

        assert len(results) == 10
        assert all(r.error is None for r in results), [r.error for r in results]
        assert {r.orchestrator for r in results} == set(orders)

        summary = report.summarize(results)
        assert set(summary) == {
            f"{backend}{suffix}"
            for backend in ("bm25", "full_transcript")
            for suffix in ("", "/fanout", "/loop", "/graph", "/plan_execute")
        }
        assert report.orchestrators_in(summary) == [
            "single_shot",
            "fanout",
            "graph",
            "loop",
            "plan_execute",
        ]

        table = report.to_markdown(summary)
        assert "Orchestration: does searching again pay for itself?" in table

        report.save(results, summary, tmp_path / "out")
        raw = (tmp_path / "out" / "runs.json").read_text(encoding="utf-8")
        assert '"orchestrator"' in raw and '"queries"' in raw

    def test_state_does_not_leak_between_probes(self, stub_factory_for, tmp_path: Path) -> None:
        """Probe 2 must not inherit probe 1's evidence, or the score is fiction."""
        task = Task(
            task_id="two",
            description="",
            events=(
                Event(turn_id=1, speaker="user", text="alpha", at=date(2025, 1, 1)),
                Event(turn_id=2, speaker="user", text="bravo", at=date(2025, 1, 2)),
            ),
            probes=(
                Probe(probe_id="two::1", after_turn=1, type="simple_recall", question="q1", expected="alpha"),
                Probe(probe_id="two::2", after_turn=2, type="simple_recall", question="q2", expected="bravo"),
            ),
        )
        result = runner.run_cell(
            "bm25",
            task,
            config=ModelConfig(model="stub"),
            orchestrator="graph",
            cache_dir=tmp_path,
            llm_factory=stub_factory_for(assess_hops=0),
        )

        assert result.error is None, result.error
        assert [p.hops for p in result.probes] == [1, 1]
        assert [p.queries for p in result.probes] == [("q1",), ("q2",)]


class TestCharts:
    """Chart code is easy to break silently, and it is the headline artifact."""

    def _metrics(self, backend: str, orchestrator: str, score: float) -> dict:
        return {
            "backend": backend,
            "orchestrator": orchestrator,
            "score": score,
            "by_type": {"simple_recall": score, "multi_hop": score},
            "cost_per_probe_usd": 0.002,
            "tokens_per_probe": 500.0,
            "avg_hops": 1.0,
        }

    def test_the_orchestration_chart_appears_with_a_second_style(
        self, tmp_path: Path
    ) -> None:
        summary = {
            "bm25": self._metrics("bm25", "single_shot", 0.5),
            "bm25/loop": self._metrics("bm25", "loop", 0.7),
        }
        written = report.write_charts(summary, tmp_path, priced=True)

        assert [p.name for p in written] == [
            "by_probe_type.png",
            "cost_vs_accuracy.png",
            "orchestration.png",
        ]
        assert all(p.stat().st_size > 0 for p in written)

    def test_a_single_orchestrator_run_draws_the_original_two(
        self, tmp_path: Path
    ) -> None:
        summary = {"bm25": self._metrics("bm25", "single_shot", 0.5)}
        written = report.write_charts(summary, tmp_path, priced=False)

        assert [p.name for p in written] == ["by_probe_type.png", "cost_vs_accuracy.png"]
