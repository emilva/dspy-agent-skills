# Example 03 — Invoice Extraction Results

| Metric | Baseline | Optimized | Δ |
|---|---:|---:|---:|
| Weighted multi-axis | 0.833 | 0.931 | **+0.098** |

- Task LM: `openrouter/liquid/lfm-2.5-1.2b-instruct:free`
- Reflection LM: `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- GEPA mode: `auto="light"`, seed=0
- Trainset: 20 · valset: 9
- Accepted mutations: 5 (candidate pool size 6)
- Best candidate: idx 5

Axes & weights: schema validity (0.20) + vendor match (0.15) + date match (0.15)
+ line-item F1 (0.35) + total match (0.15). See `pipeline.py:rich_metric`.

Note: OpenRouter free-tier daily quota (2000 req/day) was exhausted at iteration 24
of ~25. Scores shown are GEPA's tracked full-valset aggregates per candidate; the
final end-of-run optimized re-eval couldn't complete. baseline/optimized are the
full-valset scores GEPA recorded at iter 0 (base) and at candidate 5 acceptance.
