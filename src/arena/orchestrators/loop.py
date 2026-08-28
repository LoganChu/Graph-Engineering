"""Loop engineering: hand the model a search tool and let it decide.

`create_agent` is LangChain v1's replacement for the deprecated `AgentExecutor`,
and since v1.0 it compiles to a LangGraph graph -- so this is not "LangChain
instead of LangGraph". It is the *implicit* end of the same runtime: there is
still a cycle, but nobody wrote it down. The model chooses how many times to
search, what to search for, and when it has enough.

What that buys: query reformulation for free. A store that misses on the
question's own wording gets a second chance at the user's phrasing without
anyone specifying how.

What it costs: a model call per decision, and no guarantee the loop converges.
The hop cap below is not a tuning knob, it is the thing standing between this
orchestrator and an unbounded bill.
"""

from __future__ import annotations


from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

from .. import agent
from ..memory import Memory
from ..orchestration import Orchestrator, Retriever, register
from ..types import Attempt, Stop
from .adapter import ArenaChatModel

TOOL_DESCRIPTION = """\
Search this person's long-term memory and return the matching excerpt.

Pass the terms you expect to appear in the stored text, not the question. If a \
search comes back empty or off-target, try again with different wording."""

BUDGET_SPENT = (
    "Search budget exhausted -- no further searches are available. "
    "Answer from the evidence you already have, or say you don't know."
)


@register("loop")
class LoopAgent(Orchestrator):
    """LangChain `create_agent`: the model owns the iteration policy."""

    blurb = "LangChain create_agent -- the model decides when to search again."

    def run(self, store: Memory, question: str) -> Attempt:
        retriever = Retriever(store)
        refused = False  # did the model ask for a search it could not have?

        def recall(query: str) -> str:
            # The cap is enforced here rather than by the framework so that an
            # over-eager model spends one wasted model call, not one wasted
            # retrieval plus a crash.
            nonlocal refused
            if retriever.hops >= self.max_hops:
                refused = True
                return BUDGET_SPENT
            return retriever.fetch(query).context or "(no matching memory found)"

        tools = [
            StructuredTool.from_function(
                func=recall, name="recall", description=TOOL_DESCRIPTION
            )
        ]

        graph = create_agent(
            ArenaChatModel(client=self.llm),
            tools,
            system_prompt=agent.TOOL_SYSTEM,
        )

        note = ""
        stop: Stop = "answered"
        try:
            # Each hop costs two graph steps (decide, then execute), plus the
            # opening call and the final answer.
            state = graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                {"recursion_limit": 2 * self.max_hops + 4},
            )
            text = _final_answer(state["messages"])
            # It answered, but only after being told the budget was gone. That
            # is a different result from answering while it still had searches
            # in hand, and the two used to be indistinguishable in the report.
            if refused:
                stop = "hop_cap"
        except GraphRecursionError:
            # A loop that will not terminate is a result about loop engineering,
            # not a harness failure. Record it and grade it like any other miss.
            text = "I don't know."
            note = "hit the recursion limit without answering"
            stop = "recursion_limit"

        return retriever.finish(text, note=note, stop=stop)


def _final_answer(messages: list) -> str:
    """The last assistant turn that is prose rather than a tool request."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = message.content
            if isinstance(content, list):  # some providers return content blocks
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if str(content).strip():
                return str(content).strip()
    return "I don't know."
