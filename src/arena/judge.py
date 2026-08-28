"""Grading. Two gates, cheapest first.

A deterministic guard runs before the model ever sees the answer: probes may
declare strings that must NOT appear (typically the superseded value in a
contradiction probe). A backend that answers "Durham" after the user moved to
New York fails on the string check regardless of how confident it sounded, and
no judge call is spent.

Everything else goes to a rubric judge. The judge is the single largest source
of measurement error in a harness like this, so `calibrate()` exists to report
agreement against hand-labeled examples -- report that number alongside the
accuracy figures or the accuracy figures do not mean anything.
"""

from __future__ import annotations

from .llm import LLM
from .types import Probe, Verdict

#: Partial credit, shared with the report so both sides of the harness agree on
#: what a grade is worth. `PassK` deliberately ignores this and counts only
#: `correct` -- reliability is a stricter question than accuracy.
POINTS: dict[str, float] = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

RUBRIC = """\
Grade an assistant's answer to a memory-recall question.

QUESTION: {question}
REFERENCE ANSWER: {expected}
ASSISTANT'S ANSWER: {actual}

Grade against the reference answer only:
- correct  -- conveys the same fact as the reference, even if worded differently
              or with extra detail.
- partial  -- right direction but incomplete, ambiguous, or hedged to the point
              that a user would still be unsure.
- incorrect -- a different fact, a refusal, "I don't know", or an answer that
              contradicts the reference.

"I don't know" is INCORRECT when the reference contains a real fact. It is
CORRECT only when the reference itself says the information is unknown or was
never stated."""


def grade(llm: LLM, probe: Probe, actual: str) -> tuple[str, str, bool]:
    """Return (grade, reason, hard_fail)."""
    lowered = actual.lower()
    for banned in probe.must_not_contain:
        if banned.lower() in lowered:
            return (
                "incorrect",
                f"answer contained the superseded value {banned!r}",
                True,
            )

    verdict = llm.parse(
        RUBRIC.format(
            question=probe.question, expected=probe.expected, actual=actual
        ),
        Verdict,
        phase="judge",
    )
    return verdict.grade, verdict.reason, False


def score(grades: list[str]) -> float:
    """Partial credit: correct = 1.0, partial = 0.5, incorrect = 0."""
    if not grades:
        return 0.0
    return sum(POINTS.get(g, 0.0) for g in grades) / len(grades)


def calibrate(llm: LLM, labeled: list[tuple[Probe, str, str]]) -> dict:
    """Agreement between the judge and human labels.

    `labeled` is a list of (probe, answer_text, human_grade). Run this on a
    sample before trusting any headline number.
    """
    agree = 0
    disagreements: list[dict] = []
    for probe, actual, human in labeled:
        got, reason, _ = grade(llm, probe, actual)
        if got == human:
            agree += 1
        else:
            disagreements.append(
                {
                    "probe": probe.probe_id,
                    "answer": actual,
                    "human": human,
                    "judge": got,
                    "judge_reason": reason,
                }
            )
    total = len(labeled) or 1
    return {
        "n": len(labeled),
        "agreement": agree / total,
        "disagreements": disagreements,
    }
