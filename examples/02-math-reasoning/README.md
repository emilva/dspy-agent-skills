# Example 02 — Multi-Step Math Reasoning

Grade-school word problems with compound percentages, work-rate, weighted averages, and distractors. A **ChainOfThought** solver must produce the correct final numeric answer and nothing else. GEPA tunes the instruction to unlock systematic reasoning — step enumeration, unit tracking, avoiding the classic traps encoded in the dataset.

## Committed results

| Metric | Baseline | Optimized | Δ |
|---|---:|---:|---:|
| Exact-match (numeric) | 45.00 | **70.00** | **+25.00** |

- **Task LM**: `openrouter/liquid/lfm-2.5-1.2b-instruct:free` (Liquid LFM 2.5, 1.2B, $0)
- **Reflection LM**: `openrouter/nvidia/nemotron-3-super-120b-a12b:free` ($0)
- **GEPA mode**: `auto="light"`, seed=0
- **Trainset**: 34 · **valset**: 12
- **Runtime**: ~23 min on free tier
- **Artifact**: `optimized_program.json`

GEPA ran 23 iterations, **accepted 5 mutations** (iters 2, 3, 4, 6, 10), landing candidate 4 (valset score 0.70) as the winner.

## Task

```
dspy.ChainOfThought("problem -> answer (numeric only)")
```

34 train + 12 val problems (`data/train.jsonl`, `data/val.jsonl`). Each carries a `trap` hint — a one-line description of the typical mistake — which the metric weaves into its feedback so the reflection LM learns *structural* fixes rather than memorizing answers.

Example problem + trap:
```json
{
  "problem": "A $200 item is marked up 25%, then discounted 20% on the marked-up price. What is the final price in dollars?",
  "answer": 200,
  "trap": "200*1.25=250, 250*0.8=200. The sequence matters; do NOT add the percentages."
}
```

## Metric (`pipeline.py:rich_metric`)

Exact-match on the parsed numeric answer, with 0.2 partial credit for near-misses within 10% relative error. Returns `dspy.Prediction(score, feedback)` where the feedback includes the problem's trap hint when wrong. GEPA's reflection LM uses these hints to generalize — the instruction it lands teaches the solver to be explicit about operation order and re-check arithmetic.

## Run it

```bash
# Smoke test
uv run --with dspy --with python-dotenv python examples/02-math-reasoning/run.py --dry-run

# Baseline
DSPY_TASK_MODEL=openrouter/liquid/lfm-2.5-1.2b-instruct:free \
  uv run --with dspy --with python-dotenv python examples/02-math-reasoning/run.py --baseline

# Full GEPA run (~20-50 min on free tier)
DSPY_TASK_MODEL=openrouter/liquid/lfm-2.5-1.2b-instruct:free \
  uv run --with dspy --with python-dotenv \
  python examples/02-math-reasoning/run.py --optimize --auto light --seed 0
```

## Why Liquid 1.2B is the right task LM

This example is a deliberate **weaker-task-LM showcase**. On stronger models (GLM 4.5 Air, Ministral 8B, Nemotron Nano 9B) the trainset baseline is already 0.83–0.93, leaving GEPA with so few failures that every minibatch is all-perfect and the reflection LM is never called. The 1.2B Liquid model creates real headroom — it fails on enough compound-operation problems that GEPA can find structural instruction improvements worth +25 points.

If you swap in a bigger task LM, don't be surprised by a saturated baseline with +0.00 improvement — that's a feature of the task/model combo, not a GEPA bug.

## Reproducibility

`seed=0`, `auto="light"`, Liquid 1.2B + Nemotron-3 120B both free. **$0** reproduction. Run.py sets `num_threads=1` and `num_retries=12` by default to be polite with the 20 req/min free-tier cap. Even with hardening, expect occasional 429s — they retry automatically with backoff.
