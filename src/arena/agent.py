"""The agent under test. Identical for every backend -- that is the point.

The only thing that varies across the matrix is what `Memory.recall` put in
front of it, so any difference in the scores is attributable to the memory
architecture rather than to prompt differences.

The system prompt forces a closed-book answer. Without that instruction the
model happily answers plausible-sounding personal questions from its own priors,
and a backend that retrieved nothing scores as if it had retrieved correctly --
which silently inflates every number in the report.

`RULES` is shared with the orchestrators for the same reason. Once the agent can
also run as a tool loop or a state graph, the closed-book constraint has to be
worded identically in all three, or the orchestration axis measures prompt drift
rather than orchestration.
"""

from __future__ import annotations

from .llm import LLM
from .types import Recall

RULES = """\
Rules:
- Never use outside knowledge or guess. The {source} is the only valid source.
- If the {source} does not contain the answer, reply exactly: I don't know.
- If the {source} marks a fact as superseded or no longer true, answer with the \
current fact, not the retired one.
- Answer in one short sentence. No preamble, no hedging, no restating the \
question."""

SYSTEM = (
    """\
You are answering questions about a person using ONLY the memory excerpt \
provided in the message.

"""
    + RULES.format(source="excerpt")
)

#: The loop orchestrator's evidence arrives as tool results rather than as a
#: pre-assembled excerpt, so the framing differs -- the constraints do not.
TOOL_SYSTEM = (
    """\
You are answering questions about a person using ONLY what the `recall` tool \
returns.

You know nothing about this person until you search, so call `recall` first. If \
the results do not answer the question, search again with different wording, or \
for the piece that is missing -- then answer.

"""
    + RULES.format(source="retrieved evidence")
)

TEMPLATE = """\
MEMORY EXCERPT
--------------
{context}

QUESTION
--------
{question}"""


def answer(llm: LLM, recall: Recall, question: str) -> str:
    return llm.complete(
        TEMPLATE.format(context=recall.context, question=question),
        phase="answer",
        system=SYSTEM,
        max_tokens=300,
    )
