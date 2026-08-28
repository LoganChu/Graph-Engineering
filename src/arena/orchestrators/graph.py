"""Graph engineering: write the iteration policy down as edges.

    START -> retrieve -> assess --(enough | out of hops)--> answer -> END
                 ^                        |
                 +------(not enough)------+

The cycle is the point. A DAG cannot express "search again with better terms",
which is why this is a `StateGraph` and not a chain -- and it is the capability
the loop orchestrator gets implicitly, from the model, instead of explicitly,
from the topology.

Three things you get here that the loop cannot give you:

  * a declared exit condition -- `sufficient`, checked by a structured call
    whose output is validated, rather than inferred from the model choosing not
    to emit another tool call;
  * a hop cap that is a property of the graph rather than a hope about the
    prompt;
  * inspectable state. `queries` and `evidence` are fields, so a bad run can be
    read after the fact instead of reconstructed from a message log.

The assessment is deliberately ONE model call that both judges sufficiency and
proposes the next query. Splitting it into separate assess and reformulate nodes
reads better on a diagram but would spend two calls per cycle against the loop's
one, and the comparison being made here is a cost comparison.

The answer node calls the same `agent.answer` as `single_shot`, with the same
prompt. Everything this orchestrator can win, it has to win by assembling better
evidence.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .. import agent
from ..memory import Memory
from ..orchestration import Orchestrator, Retriever, register
from ..types import Attempt, Recall, Stop


class Assessment(BaseModel):
    """Structured output of the assess node -- the graph's routing decision."""

    sufficient: bool = Field(
        description="True if the evidence so far is enough to answer the "
        "question, OR if the store clearly does not hold the answer and further "
        "searching will not help."
    )
    missing: str = Field(
        default="", description="When not sufficient: the one fact still needed."
    )
    next_query: str = Field(
        default="",
        description="When not sufficient: search terms for the next attempt.",
    )


ASSESS = """\
You are deciding whether a memory search has gathered enough to answer a question.

QUESTION
--------
{question}

SEARCHES ALREADY TRIED
----------------------
{queries}

EVIDENCE RETRIEVED SO FAR
-------------------------
{evidence}

Judge only what is above. Do not use outside knowledge.

Set `sufficient` to true if the evidence answers the question. Also set it true \
if the evidence makes clear the store simply never recorded this -- in that case \
"I don't know" is the correct answer and more searching only costs money.

Otherwise name the `missing` fact, and give a `next_query` that uses DIFFERENT \
wording from the searches already tried, phrased the way the stored text would \
put it rather than the way the question does."""


class State(TypedDict):
    question: str
    query: str
    evidence: list[str]
    queries: list[str]
    hops: int
    exhausted: bool
    stop: Stop
    answer: str


@register("graph")
class GraphAgent(Orchestrator):
    """LangGraph `StateGraph`: an explicit retrieve/assess cycle you can read."""

    blurb = "LangGraph StateGraph -- an explicit retrieve/assess cycle with a declared exit."

    def run(self, store: Memory, question: str) -> Attempt:
        retriever = Retriever(store)

        def retrieve(state: State) -> dict:
            recall = retriever.fetch(state["query"])
            body = recall.context or "(no matching memory found)"
            return {
                "evidence": state["evidence"]
                + [f"--- search {state['hops'] + 1}: {state['query']!r} ---\n{body}"],
                "queries": state["queries"] + [state["query"]],
                "hops": state["hops"] + 1,
            }

        def assess(state: State) -> dict:
            if state["hops"] >= self.max_hops:
                return {"exhausted": True, "stop": "hop_cap"}
            verdict = self.llm.parse(
                ASSESS.format(
                    question=state["question"],
                    queries="\n".join(f"- {q!r}" for q in state["queries"]),
                    evidence="\n\n".join(state["evidence"]),
                ),
                Assessment,
                phase="orchestrate",
                max_tokens=400,
            )
            if verdict.sufficient:
                return {"exhausted": True, "stop": "sufficient"}
            if not verdict.next_query.strip():
                # It judged the evidence insufficient and then could not say
                # what to look for. Recorded separately because it is a failure
                # of the assessor, not of the hop budget -- and the two used to
                # land in the same bucket.
                return {"exhausted": True, "stop": "no_query"}
            return {"query": verdict.next_query.strip(), "exhausted": False}

        def answer(state: State) -> dict:
            evidence = "\n\n".join(state["evidence"])
            return {
                "answer": agent.answer(
                    self.llm, Recall(context=evidence), state["question"]
                )
            }

        def route(state: State) -> str:
            return "answer" if state["exhausted"] else "retrieve"

        builder = StateGraph(State)
        builder.add_node("retrieve", retrieve)
        builder.add_node("assess", assess)
        builder.add_node("answer", answer)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "assess")
        builder.add_conditional_edges(
            "assess", route, {"retrieve": "retrieve", "answer": "answer"}
        )
        builder.add_edge("answer", END)

        final = builder.compile().invoke(
            {
                "question": question,
                "query": question,  # first hop asks the store the question itself
                "evidence": [],
                "queries": [],
                "hops": 0,
                "exhausted": False,
                "stop": "answered",  # always overwritten: assess runs first
                "answer": "",
            },
            # Every cycle is retrieve + assess; +2 covers the answer node and END.
            {"recursion_limit": 2 * self.max_hops + 2},
        )
        return retriever.finish(final["answer"], stop=final["stop"])
