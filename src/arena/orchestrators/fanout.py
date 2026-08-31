"""Breadth without feedback: write every query up front, retrieve them all.

    START -> rewrite -> retrieve x N (all at once) -> answer -> END

This is `graph` with the feedback edge cut, and it exists to answer a question
the loop-versus-graph table cannot: when an iterative orchestrator beats
single-shot, is it because the agent *adapted* to what came back, or merely
because it searched more times?

Held identical to `graph` on purpose:

  * the same opening query -- the question, verbatim;
  * the same total retrieval budget, `max_hops`;
  * the same instruction to phrase later queries the way stored text would
    put it rather than the way the question does.

The one difference is information. `graph` writes each new query after seeing
what the last one returned; this writes them all before seeing anything. So a
tie says the reformulation never needed the evidence, and the per-cycle
assessment call was pure overhead.

It is also the cheap end of the axis on control flow: ONE `orchestrate` call
regardless of hop count, against the graph's one per cycle. If it ties on score
it wins outright on cost, and that is a result worth being able to print.

No LangChain, no LangGraph. A single fan-out with no cycle in it is a list
comprehension, and dressing it as a framework graph would obscure the fact that
this is the control -- it also keeps the ablation runnable on the base install,
without the orchestration extra.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .. import agent
from ..memory import Memory
from ..orchestration import Orchestrator, Retriever, register
from ..types import Attempt, Recall


class Rewrites(BaseModel):
    """The whole search plan, decided before any of it runs."""

    # Required on purpose, and the distinction from a default is not pedantic.
    # A model that echoes the schema back instead of filling it in -- small
    # local ones do, nesting the real answers under `properties` -- returns an
    # object with no `queries` key at all. Under `default_factory=list` that
    # validated as "no rewrites wanted": the fan-out narrowed to the bare
    # question while still paying for the call that was supposed to widen it,
    # and the table printed `+0.00 score lift` against a positive cost delta.
    # That is this harness's headline finding shape, so the one failure worth
    # ruling out is the one that forges it. Required instead, so a missing key
    # fails validation, takes the provider's repair round trip, and is counted
    # in `json_repairs` if it happens.
    #
    # An explicitly empty list still validates, and still means what it says:
    # the model was asked and had nothing to add, which narrows the fan to the
    # question alone. That is a thin plan, not a broken parse, and the two are
    # only distinguishable because this field has no default.
    queries: list[str] = Field(
        ...,
        description="Alternative search phrasings, most promising first.",
    )


REWRITE = """\
You are writing search queries against a store of things one person has said \
over many past conversations.

QUESTION
--------
{question}

The question itself has already been searched for, verbatim. Write {n} MORE \
queries that would find the answer if that first search missed.

Make them genuinely different from each other and from the question -- vary the \
vocabulary, not the word order. Phrase each one the way the stored text would \
put it rather than the way the question does: the person wrote about their own \
life as it happened, so they said "I started at the new place" where the \
question says "when did you change jobs".

You cannot see any results and will not get another turn, so spend the queries \
on different guesses rather than on refining one."""


@register("fanout")
class Fanout(Orchestrator):
    """Every query written up front, retrieved in one round, answered once."""

    blurb = "Breadth, no feedback -- all queries written before any of them run."

    def run(self, store: Memory, question: str) -> Attempt:
        retriever = Retriever(store)
        # The question itself always goes first, which makes this orchestrator's
        # evidence a superset of what `single_shot` saw. Any difference between
        # them is then attributable to the extra queries and nothing else.
        queries = [question]

        wanted = self.max_hops - 1
        if wanted > 0:
            plan = self.llm.parse(
                REWRITE.format(question=question, n=wanted),
                Rewrites,
                phase="orchestrate",
                max_tokens=400,
            )
            queries.extend(_distinct(plan.queries, seen=queries, limit=wanted))
        # With `--max-hops 1` there is no room for a rewrite, so none is paid
        # for: the orchestrator degenerates to single-shot plus zero overhead
        # rather than to single-shot plus a wasted call.

        evidence = []
        for position, query in enumerate(queries):
            body = retriever.fetch(query).context or "(no matching memory found)"
            evidence.append(f"--- search {position + 1}: {query!r} ---\n{body}")

        text = agent.answer(
            self.llm, Recall(context="\n\n".join(evidence)), question
        )
        # `answered`, not `hop_cap`: the budget was spent by design, not
        # exhausted by a policy that wanted more. Reporting it as capped would
        # put every fanout row in the "re-run with a longer leash" column.
        return retriever.finish(text, stop="answered")


def _distinct(proposed: list[str], *, seen: list[str], limit: int) -> list[str]:
    """Drop blanks and repeats, case- and whitespace-insensitively.

    A rewrite that comes back as the question again would spend a hop to
    retrieve something already in hand, and would quietly make the fan-out
    narrower than the budget it is being charged for.
    """
    taken = {q.strip().casefold() for q in seen}
    kept: list[str] = []
    for query in proposed:
        cleaned = query.strip()
        if not cleaned or cleaned.casefold() in taken:
            continue
        taken.add(cleaned.casefold())
        kept.append(cleaned)
        if len(kept) == limit:
            break
    return kept
