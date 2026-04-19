# Example 03 — Typed Invoice Extraction

Extract a **Pydantic-typed** `InvoiceRecord` from unstructured invoice text — vendor, date, line items (description/quantity/unit_price), and total. The example exercises typed DSPy outputs and a multi-axis metric that rewards schema validity, field correctness, and arithmetic consistency simultaneously.

## Committed results

| Metric | Baseline | Optimized | Δ |
|---|---:|---:|---:|
| Weighted multi-axis | 0.833 | **0.931** | **+0.098** |

- **Task LM**: `openrouter/liquid/lfm-2.5-1.2b-instruct:free` (Liquid LFM 2.5, 1.2B, $0)
- **Reflection LM**: `openrouter/nvidia/nemotron-3-super-120b-a12b:free` ($0)
- **GEPA mode**: `auto="light"`, seed=0
- **Trainset**: 20 · **valset**: 9
- **Runtime**: ~36 min on free tier (exhausted OpenRouter's 2000 req/day quota at iter 24 of 25)
- **Artifact**: `optimized_program.json`

GEPA accepted **5 mutations** (candidate pool 6). Candidate 5 at iteration 22 landed the best full-valset aggregate of 0.931.

> **Note on the final re-eval**: OpenRouter's free-tier daily quota exhausted late in the run, so the very-last end-of-run re-evaluation couldn't execute on fresh LM calls. The baseline and optimized scores come from GEPA's own full-valset evaluations cached in `gepa_state.bin`, which are the same values GEPA uses for candidate selection — they're reliable; they just weren't re-computed after quota reset.

## Task

```
dspy.ChainOfThought("invoice_text -> record: InvoiceRecord")
```

where `InvoiceRecord` is a Pydantic model:

```python
class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float

class InvoiceRecord(BaseModel):
    vendor: str         # seller — not buyer or shipper
    date: str           # YYYY-MM-DD
    line_items: list[LineItem]
    total: float        # final amount due (post-tax/shipping/discount)
```

Dataset (`data/{train,val}.jsonl`) includes genuinely tricky layouts: discount/rebate rows that reduce the total, seller/bill-to/shipper ambiguity in headers, varied date formats (DD-MM-YYYY, "March 8, 2024", "22 September 2024"), freight/handling lines that are NOT line items, and tax-inclusive vs pre-tax totals.

## Metric (`pipeline.py:rich_metric`)

Five-axis weighted score returning `dspy.Prediction(score, feedback)`:

| Axis | Weight | Checks |
|---|---:|---|
| Schema validity | 0.20 | Output parses as `InvoiceRecord` |
| Vendor match | 0.15 | Normalized equality (prefix-tolerant) |
| Date match | 0.15 | Exact YYYY-MM-DD |
| Line-item F1 | 0.35 | Set-F1 over (fuzzy_description, qty, unit_price) triples |
| Total match | 0.15 | Absolute delta ≤ $0.50 |

Feedback names each failing axis with specifics (e.g., *"DATE: predicted '03-05-2024', expected '2024-05-03' — YYYY-MM-DD format"*), which is what lets GEPA's reflection LM learn per-axis fixes.

## Run it

```bash
# Smoke test
uv run --with dspy --with python-dotenv --with pydantic python examples/03-invoice-extraction/run.py --dry-run

# Baseline
DSPY_TASK_MODEL=openrouter/liquid/lfm-2.5-1.2b-instruct:free \
  uv run --with dspy --with python-dotenv --with pydantic \
  python examples/03-invoice-extraction/run.py --baseline

# Full GEPA run (~30-60 min on free tier)
DSPY_TASK_MODEL=openrouter/liquid/lfm-2.5-1.2b-instruct:free \
  uv run --with dspy --with python-dotenv --with pydantic \
  python examples/03-invoice-extraction/run.py --optimize --auto light --seed 0
```

## Why Liquid 1.2B

Prior validation runs on GLM 4.5 Air (32B), Ministral 8B, and Nemotron Nano 9B all saturated at baseline ≥ 0.98 — modern open models in the 8B+ range trivially handle typed invoice extraction. To showcase GEPA headroom, this example uses the smaller 1.2B Liquid model, which struggles with subtleties (seller-vs-shipper, discount rows, date formats) and gives the reflection LM enough errors to target. With bigger task LMs, expect near-saturated baselines and minimal GEPA movement — that's the task's nature, not a GEPA limitation.

## Gotchas (patched)

Two non-obvious issues surfaced during validation; both are fixed in the committed code and worth flagging:

1. **Pydantic output signatures + `from __future__ import annotations`** → DSPy's signature builder can't resolve the `ForwardRef('InvoiceRecord')` string it receives. `pipeline.py` deliberately does NOT use the future-annotations import so DSPy sees concrete types at class-body evaluation time.
2. **GEPA state pickling with typed signatures** → stock `pickle` can't serialize the dynamic `StringSignature` subclass that DSPy generates for Pydantic outputs. Fix: `dspy.GEPA(..., gepa_kwargs={"use_cloudpickle": True})`.

## Reproducibility

`seed=0`, `auto="light"`, free-tier models → **$0** reproduction in principle, but the full run hits OpenRouter's 2000 req/day cap so a fresh reproduction needs either waiting for the daily quota reset or a paid-tier fallback (e.g., `openrouter/mistralai/ministral-3b-2512` at $0.10/M tokens). GEPA checkpoints to `gepa_logs/gepa_state.bin`, so you can resume an interrupted run.
