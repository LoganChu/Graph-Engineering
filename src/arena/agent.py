"""The agent under test. Identical for every backend -- that is the point.

The only thing that varies across the matrix is what `Memory.recall` put in
front of it, so any difference in the scores is attributable to the memory
architecture rather than to prompt differences.

The system prompt forces a closed-book answer. Without that instruction the
model happily answers plausible-sounding personal questions from its own priors,
and a backend that retrieved nothing scores as if it had retrieved correctly --
which silently inflates every number in the report.
"""

from __future__ import annotations

from .llm import LLM
from .types import Recall

SYSTEM = """\
You are answering questions about a person using ONLY the memory excerpt \
provided in the message.

Rules:
- Never use outside knowledge or guess. The excerpt is the only valid source.
- If the excerpt does not contain the answer, reply exactly: I don't know.
- If the excerpt marks a fact as superseded or no longer true, answer with the \
current fact, not the retired one.
- Answer in one short sentence. No preamble, no hedging, no restating the \
question."""

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
