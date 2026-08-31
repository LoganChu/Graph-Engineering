"""Decompose first, then retrieve once per part. Map-reduce over the store.

                          +--> retrieve("where did I work in 2023") --+
    START -> plan --Send--+--> retrieve("when did I move")           -+--> answer -> END
                          +--> retrieve("what did I say about pay")  -+

The other four orchestrators all ask ONE question, and differ only in how many
times they reword it. This one asks several different questions. That is a
different capability, not a different policy, and it is the gap the axis had:

  * `multi_hop` probes are joins across turns that share no vocabulary. No
    rewording of the original question retrieves both halves, because the
    halves are not phrased alike -- but two sub-questions retrieve one each.
  * `aggregation` needs a fold rather than a lookup. Decomposition is the
    mechanism that makes a fold expressible at all here. It is not a full
    map-reduce over the store -- the branches are the sub-questions the model
    could name, not every event -- so this narrows the gap rather than closing
    it. Neither corpus currently populates that probe type anyway.

`Send` is what makes this a graph rather than a chain. The branch count is not
known until the plan node has run, so the fan-out cannot be drawn in advance --
which is precisely the case `add_conditional_edges` returning `Send` objects
exists for, and the reason this is worth writing on LangGraph instead of in a
list comprehension the way `fanout` is.

Cost is deliberately identical in shape to `fanout`: one `orchestrate` call for
the plan, N retrievals, one `answer` call. The branches do NOT each answer their
own sub-question. Per-branch answers would read well on a diagram and would
break the rule the whole axis rests on -- that an orchestrator wins only by
assembling better evidence, never by spending extra reasoning calls the others
were not given. The join is the answer model's job, on the same prompt everyone
else gets.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from .. import agent
from ..memory import Memory
from ..orchestration import Orchestrator, Retriever, register
from ..types import Attempt, Recall


class Plan(BaseModel):
    """The decomposition -- decided once, before anything is retrieved."""

    # Required for the reason `Rewrites.queries` is -- see fanout.py. A missing
    # key means the model never filled the schema in, and must reach the repair
    # path rather than being read as a decomposition into nothing. An empty list
    # is left legal: `run` degrades it to the question, which is the documented
    # failure mode and the thing `test_an_empty_plan_falls_back_to_the_question`
    # pins down.
    sub_questions: list[str] = Field(
        ...,
        description="Independently answerable parts, in the order they should "
        "be looked up. One item if the question does not decompose.",
    )


PLAN = """\
You are breaking a question into the separate lookups needed to answer it, \
against a store of things one person has said over many past conversations.

QUESTION
--------
{question}

Return at most {n} sub-questions. Each one must be answerable on its own, by a \
single search, without knowing the answer to any of the others -- you get one \
round of parallel lookups, not a chain.

A question that needs a join across two unrelated facts becomes one \
sub-question per fact. A question that counts or summarises becomes the \
lookups whose results you would then count.

If the question is already a single lookup, return it unchanged as the only \
item. Padding an atomic question into parts costs searches and finds nothing \
new."""


class State(TypedDict):
    question: str
    plan: list[str]
    # Branches finish in whatever order the store returns them, so each carries
    # the position it was dispatched at and the join sorts before rendering.
    # Without that the answer prompt reorders between identical runs, and a
    # content-addressed cache stops replaying.
    evidence: Annotated[list[tuple[int, str, str]], operator.add]
    answer: str


class Branch(TypedDict):
    """The payload one `Send` carries -- not the graph state."""

    order: int
    sub_question: str


@register("plan_execute")
class PlanExecute(Orchestrator):
    """LangGraph `Send`: split the question, retrieve each part, join once."""

    blurb = "Decompose into sub-questions, retrieve each on its own branch, join."

    def run(self, store: Memory, question: str) -> Attempt:
        retriever = Retriever(store)

        def plan(state: State) -> dict:
            drafted = self.llm.parse(
                PLAN.format(question=state["question"], n=self.max_hops),
                Plan,
                phase="orchestrate",
                max_tokens=400,
            )
            parts = _distinct(drafted.sub_questions, limit=self.max_hops)
            # An empty plan would emit no branches, and a fan-out with no
            # branches never reaches the join -- the graph would end without an
            # answer. Falling back to the question makes the failure mode
            # "behaves like single_shot" instead of "returns nothing".
            return {"plan": parts or [state["question"]]}

        def dispatch(state: State) -> list[Send]:
            return [
                Send("retrieve", Branch(order=i, sub_question=part))
                for i, part in enumerate(state["plan"])
            ]

        def retrieve(branch: Branch) -> dict:
            recall = retriever.fetch(
                branch["sub_question"], order=branch["order"]
            )
            body = recall.context or "(no matching memory found)"
            return {
                "evidence": [(branch["order"], branch["sub_question"], body)]
            }

        def answer(state: State) -> dict:
            evidence = "\n\n".join(
                f"--- part {order + 1}: {sub!r} ---\n{body}"
                for order, sub, body in sorted(state["evidence"])
            )
            return {
                "answer": agent.answer(
                    self.llm, Recall(context=evidence), state["question"]
                )
            }

        builder = StateGraph(State)
        builder.add_node("plan", plan)
        builder.add_node("retrieve", retrieve)
        builder.add_node("answer", answer)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges("plan", dispatch, ["retrieve"])
        builder.add_edge("retrieve", "answer")
        builder.add_edge("answer", END)

        final = builder.compile().invoke(
            {"question": question, "plan": [], "evidence": [], "answer": ""}
        )
        # `answered`, not `hop_cap`: the plan is as wide as the question needed,
        # and a budget spent by design is not a budget that ran out.
        return retriever.finish(final["answer"], stop="answered")


def _distinct(proposed: list[str], *, limit: int) -> list[str]:
    """Drop blanks and repeats, case- and whitespace-insensitively.

    A decomposition that names the same part twice would spend two hops to
    retrieve one thing, and the wasted hop would be billed as breadth.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for part in proposed:
        cleaned = part.strip()
        if not cleaned or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        kept.append(cleaned)
        if len(kept) == limit:
            break
    return kept
