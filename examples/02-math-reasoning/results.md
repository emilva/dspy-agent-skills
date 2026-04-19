# Example 02 — Math Reasoning Results

| Metric | Baseline | Optimized | Δ |
|---|---:|---:|---:|
| Exact-match (numeric) | 45.000 | 70.000 | **+25.000** |

- Task LM: `openrouter/liquid/lfm-2.5-1.2b-instruct:free`
- Reflection LM: `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- GEPA mode: `auto="light"`, seed=0
- Trainset: 34 · valset: 12
- Times: baseline 0.2s, optimize 1399.9s, optimized-eval 0.1s

Metric: exact match on final numeric answer (partial credit 0.2 for near-misses
within 10% relative error). Feedback includes per-problem `trap` hints so the
reflection LM can learn structural fixes rather than memorize answers.
See `pipeline.py:rich_metric`.
