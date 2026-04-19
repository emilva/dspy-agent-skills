# Example 01 — RAG Q&A with Citations

Retrieve, answer, cite. The prototypical DSPy use case. A **ChainOfThought** synthesizer reads BM25-retrieved passages and returns both a concise answer **and** the doc IDs that support it. GEPA optimizes the synthesizer's instruction to tighten correctness, citation grounding, and conciseness simultaneously.

## Committed results

| Metric | Baseline | Optimized | Δ |
|---|---:|---:|---:|
| Overall (weighted multi-axis) | 81.15 | **100.00** | **+18.85** |

- **Task LM**: `openrouter/z-ai/glm-4.5-air:free` (GLM 4.5 Air, 32B, $0)
- **Reflection LM**: `openrouter/nvidia/nemotron-3-super-120b-a12b:free` ($0)
- **GEPA mode**: `auto="light"`, seed=0
- **Trainset**: 15 · **valset**: 10
- **Runtime**: ~28 min (free-tier rate limits)
- **Artifact**: `optimized_program.json` (2.4 KB, portable state)

GEPA accepted 1 mutation on iteration 1 — a reworded instruction that scored perfect on the full valset — then correctly no-opped for the remaining budget because no further improvement was possible.

## Task

Corpus: 12 short solar-system articles (`data/docs.jsonl`). Questions like *"What is the orbital period of Mars?"* / *"How many moons does Jupiter have?"* / *"Who discovered Io and in what year?"* — each with a single authoritative source doc.

```
BM25Retriever(k=3) → dspy.ChainOfThought("context, question -> answer, citations: list[str]")
```

## Metric (`pipeline.py:rich_metric`)

Weighted 3-axis metric returning `dspy.Prediction(score, feedback)`:

| Axis | Weight | Checks |
|---|---:|---|
| Correctness | 0.55 | Fuzzy answer match (substring or ≥ all-token overlap) |
| Citation validity | 0.30 | At least one cited doc ID is in the gold set; extras are penalized |
| Conciseness | 0.15 | 3–25 word answer; penalties for too-short or too-long |

Feedback text is specific per failure axis so GEPA's reflection LM can target the right fix.

## Run it

```bash
# From repo root:
cp .env.example .env    # then edit to add OPENROUTER_API_KEY

# Smoke test (no LM calls)
uv run --with dspy --with python-dotenv --with rank-bm25 python examples/01-rag-qa/run.py --dry-run

# Baseline only
uv run --with dspy --with python-dotenv --with rank-bm25 python examples/01-rag-qa/run.py --baseline

# Full GEPA optimization (~20-40 min on free tier)
uv run --with dspy --with python-dotenv --with rank-bm25 python examples/01-rag-qa/run.py --optimize --auto light --seed 0

# Score a saved program
uv run --with dspy --with python-dotenv --with rank-bm25 python examples/01-rag-qa/run.py --eval optimized_program.json
```

## Why GLM 4.5 Air works here

GLM 4.5 Air nails the factual retrieval but routinely skips citations when the question could be answered from general knowledge. That's the gap GEPA closes — the reflection LM sees "citation score = 0.0, expected ['mars']" and mutates the instruction to enforce grounding. Baseline was already 0.81, so the 18.85-point lift comes from pushing correctness + citation + conciseness to perfect simultaneously.

## Reproducibility

`seed=0`, `auto="light"`, both models on OpenRouter free tier → **$0** reproduction. Free-tier 429s slow the run but don't affect the final score (litellm retries with backoff). GEPA's fitness signal uses cached full-valset evaluations, so interrupted runs can resume from `gepa_logs/gepa_state.bin`.
