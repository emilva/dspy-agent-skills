# Example 01 — RAG QA Results

| Metric | Baseline | Optimized | Δ |
|---|---:|---:|---:|
| Overall | 81.150 | 100.000 | **+18.850** |

- Task LM: `openrouter/z-ai/glm-4.5-air:free`
- Reflection LM: `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
- GEPA mode: `auto="light"`, seed=0
- Trainset: 15 · valset: 10
- Times: baseline 0.2s, optimize 1695.4s, optimized-eval 0.0s

Metric axes: correctness (0.55) + citation validity (0.30) + conciseness (0.15).
See `pipeline.py:rich_metric` for the exact scoring.
