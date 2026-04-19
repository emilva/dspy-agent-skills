"""RAG QA pipeline: BM25 retrieve over a small corpus, then a ChainOfThought
synthesizer that must answer AND cite source doc IDs.

Kept deliberately small and explicit so GEPA's improvements are legible.
"""

from __future__ import annotations

import re
from typing import Iterable

import dspy


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9°]+", text.lower())


class BM25Retriever:
    """Minimal self-contained BM25 retriever over an in-memory doc list."""

    def __init__(self, docs: list[dict], k: int = 3):
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self.k = k
        self._tokens = [_tokenize(f"{d['title']} {d['text']}") for d in docs]
        self._bm25 = BM25Okapi(self._tokens)

    def retrieve(self, query: str) -> list[dict]:
        scores = self._bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.k]
        return [self.docs[i] for i in top]


class AnswerWithCitation(dspy.Signature):
    """Given retrieved passages, answer the question concisely and cite the
    passage IDs used as evidence. The cited IDs must come from the provided
    context and must support the answer."""

    context: str = dspy.InputField(desc="retrieved passages labeled with [id] markers")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="short, factual answer — no prose padding")
    citations: list[str] = dspy.OutputField(
        desc="doc IDs from the context that support the answer"
    )


class RagQA(dspy.Module):
    """RAG module: BM25 retrieval feeds a ChainOfThought synthesizer.

    The retriever is attached as an attribute rather than captured via a closure
    so the module/signature remain picklable (GEPA checkpoints state).
    """

    def __init__(self, retriever: "BM25Retriever"):
        super().__init__()
        self.synth = dspy.ChainOfThought(AnswerWithCitation)
        self._retriever = retriever

    def forward(self, question: str):
        hits = self._retriever.retrieve(question)
        ctx = "\n\n".join(f"[{h['id']}] {h['title']}: {h['text']}" for h in hits)
        return self.synth(context=ctx, question=question)


def build_program(retriever: BM25Retriever):
    return RagQA(retriever)


def make_examples(records: Iterable[dict]):
    """Convert JSONL records into dspy.Example objects with inputs marked."""
    import dspy

    return [
        dspy.Example(
            question=r["question"], answer=r["answer"], cite=r["cite"]
        ).with_inputs("question")
        for r in records
    ]


def rich_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Multi-axis metric: correctness (0.55) + citation validity (0.30) + conciseness (0.15).

    Returns a ``dspy.Prediction(score=float, feedback=str)`` — the form GEPA's
    DSPy adapter expects. A plain dict breaks ``dspy.Evaluate`` because its
    parallelizer sums scored outputs internally (dicts can't be summed).
    """

    pred_answer = str(getattr(pred, "answer", "")).strip()
    pred_cites = list(getattr(pred, "citations", []) or [])

    gold_answer = gold.answer.strip().lower()
    gold_cites = set(gold.cite)

    pa_lower = pred_answer.lower()
    # Fuzzy correctness: full-answer substring OR all significant tokens present.
    significant = [t for t in _tokenize(gold.answer) if len(t) >= 2]
    token_hits = sum(1 for t in significant if t in pa_lower) / max(1, len(significant))
    correct_substr = gold_answer in pa_lower
    correctness = 1.0 if correct_substr else token_hits  # 0..1

    # Citation validity.
    if not pred_cites:
        citation_score = 0.0
    else:
        hit = any(c in gold_cites for c in pred_cites)
        extra = sum(1 for c in pred_cites if c not in gold_cites)
        citation_score = 1.0 if hit else 0.0
        # Penalise over-citation (dilutes grounding): -0.15 per wrong extra, floored at 0.
        citation_score = max(0.0, citation_score - 0.15 * extra)

    # Conciseness: 3–25 words is ideal; too short or too long costs.
    n_words = len(pred_answer.split())
    if 3 <= n_words <= 25:
        concise = 1.0
    elif n_words < 3:
        concise = 0.3
    else:
        concise = max(0.0, 1.0 - (n_words - 25) / 50)

    score = 0.55 * correctness + 0.30 * citation_score + 0.15 * concise

    bits = []
    if correctness < 1.0:
        bits.append(
            f"ANSWER: predicted {pred_answer!r} but expected {gold.answer!r}. "
            f"Likely cause: the model paraphrased away key facts — quote specific values from the passage verbatim."
        )
    if citation_score < 1.0:
        if not pred_cites:
            bits.append(
                f"CITATION: none produced. You MUST cite doc IDs — the correct one is {sorted(gold_cites)}."
            )
        elif not any(c in gold_cites for c in pred_cites):
            bits.append(
                f"CITATION: cited {pred_cites} but the evidence was in {sorted(gold_cites)}. "
                f"Cite the passage whose text literally contains the answer."
            )
        else:
            bits.append(
                f"CITATION: {pred_cites} includes extras not in {sorted(gold_cites)}. "
                f"Cite only the minimum evidence — extra IDs dilute grounding."
            )
    if concise < 1.0:
        if n_words > 25:
            bits.append(
                f"CONCISENESS: {n_words} words is too long. Give a 3–25 word factual answer, no explanatory prose."
            )
        else:
            bits.append(
                f"CONCISENESS: answer too terse ({n_words} words) — include the specific fact."
            )
    if not bits:
        bits.append("PASS: correct answer, correct citation, appropriate length.")

    import dspy

    return dspy.Prediction(score=score, feedback=" ".join(bits))
