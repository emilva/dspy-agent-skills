"""Multi-step arithmetic word-problem pipeline.

Showcase: how a tuned ChainOfThought instruction (via GEPA) unlocks accuracy
on problems where the baseline model skips steps or mishandles units.

Each data point ships a `trap` hint explaining the common mistake; the metric
weaves this into its feedback so GEPA's reflection LM learns structural fixes
rather than memorizing individual answers.
"""

from __future__ import annotations

import re
from typing import Iterable


def build_program():
    import dspy

    class SolveProblem(dspy.Signature):
        """Solve a grade-school arithmetic word problem. Work step by step,
        then give a single numeric final answer. The `answer` field must be
        ONLY the number (no units, no words, no currency symbols)."""

        problem: str = dspy.InputField()
        answer: str = dspy.OutputField(
            desc="final numeric answer only, e.g. '42' or '37.5'"
        )

    class MathReasoner(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.ChainOfThought(SolveProblem)

        def forward(self, problem: str):
            return self.solve(problem=problem)

    return MathReasoner()


def make_examples(records: Iterable[dict]):
    import dspy

    return [
        dspy.Example(
            problem=r["problem"], answer=r["answer"], trap=r["trap"]
        ).with_inputs("problem")
        for r in records
    ]


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_number(text: str) -> float | None:
    """Return the final numeric token in `text`, or None if none found."""
    if text is None:
        return None
    nums = _NUM_RE.findall(str(text).replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _approx_equal(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol + 1e-6 * max(abs(a), abs(b))


def rich_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Exact-match numeric metric with trap-aware rich feedback.

    Returns ``dspy.Prediction(score: float, feedback: str)`` — this is the
    format GEPA expects (see ``dspy.GEPA`` docstring). ``dspy.Prediction``
    supports ``__float__``/``__add__`` so it plays nicely with
    ``dspy.Evaluate``'s running-total progress display.
    """
    import dspy

    pred_ans = _parse_number(getattr(pred, "answer", None))
    gold_ans = float(gold.answer)
    reasoning = str(getattr(pred, "reasoning", "") or "")

    if pred_ans is None:
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"FORMAT: could not extract a numeric answer from the output. "
                f"The final `answer` field must be ONLY the number (e.g. '42'), "
                f"with no units or commentary. Expected: {gold_ans:g}. Hint: {gold.trap}"
            ),
        )

    correct = _approx_equal(pred_ans, gold_ans)
    if correct:
        return dspy.Prediction(
            score=1.0,
            feedback=f"PASS: correct answer ({pred_ans:g}).",
        )

    # Scale the miss: a near-miss (within 10% relative) gets partial credit
    # so the gradient isn't all-or-nothing and GEPA can notice small fixes.
    denom = max(abs(gold_ans), 1.0)
    rel_err = abs(pred_ans - gold_ans) / denom
    partial = 0.2 if rel_err < 0.1 else 0.0

    # Diagnose: did the reasoning at least mention the right numbers?
    steps_look_sane = all(
        str(int(v)) in reasoning or f"{v:.1f}" in reasoning
        for v in _extract_salient_numbers(gold.problem)
    )
    diag = []
    if not steps_look_sane:
        diag.append("reasoning trace missed a salient number from the problem")
    if rel_err < 0.1:
        diag.append("off by a small arithmetic slip")
    elif rel_err > 2.0:
        diag.append("likely misread the problem structure")

    feedback = (
        f"WRONG: predicted {pred_ans:g}, expected {gold_ans:g}"
        f" (relative error {rel_err:.2f}). "
        + ("Observations: " + "; ".join(diag) + ". " if diag else "")
        + f"HINT for this problem: {gold.trap}"
    )
    return dspy.Prediction(score=partial, feedback=feedback)


def _extract_salient_numbers(problem: str) -> list[float]:
    return [float(n) for n in _NUM_RE.findall(problem.replace(",", ""))]
