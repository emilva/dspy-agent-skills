"""Invoice extraction pipeline — Pydantic-typed DSPy output.

Showcase: a typed output schema backed by Pydantic lets the model produce
structured JSON-style records; the multi-axis metric rewards schema validity,
field-level F1, and arithmetic consistency (total check).

Note: this module deliberately does NOT use ``from __future__ import annotations``.
DSPy's ``make_signature`` reads field types at class-body evaluation time; with
PEP 563 deferred annotations, a typed ``record: InvoiceRecord = dspy.OutputField()``
becomes a ``ForwardRef`` string that DSPy can't resolve. Concrete types at
class-creation time sidestep this entirely.
"""

import re
from typing import Iterable

import dspy
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(
        description="human-readable item name, e.g. 'LED Panel 40W'"
    )
    quantity: int = Field(description="integer units purchased")
    unit_price: float = Field(
        description="price per single unit in the same currency as the total"
    )


class InvoiceRecord(BaseModel):
    vendor: str = Field(description="vendor/seller name as it appears on the invoice")
    date: str = Field(description="invoice date in YYYY-MM-DD format")
    line_items: list[LineItem] = Field(
        description="one entry per distinct line on the invoice"
    )
    total: float = Field(description="final amount due (after tax/shipping if present)")


class ExtractInvoice(dspy.Signature):
    """Extract a structured InvoiceRecord from raw invoice text.

    Rules:
      - `vendor` must be the exact seller name from the invoice header.
      - `date` must be reformatted to YYYY-MM-DD regardless of source format.
      - Each line item = one row on the invoice with its description, integer
        quantity, and per-unit price.
      - `total` is the FINAL amount due (including tax/shipping when present).
      - Do not invent fields. Every value must appear in the source text."""

    invoice_text: str = dspy.InputField()
    record: InvoiceRecord = dspy.OutputField()


class InvoiceExtractor(dspy.Module):
    """Module wrapper. Kept at module scope (not nested in a factory) so GEPA's
    checkpointing can pickle it."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractInvoice)

    def forward(self, invoice_text: str):
        return self.extract(invoice_text=invoice_text)


def build_program():
    return InvoiceExtractor()


def make_examples(records: Iterable[dict]):
    return [
        dspy.Example(
            invoice_text=r["text"],
            vendor=r["vendor"],
            date=r["date"],
            line_items=r["line_items"],
            total=r["total"],
        ).with_inputs("invoice_text")
        for r in records
    ]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _fuzzy_desc_match(pred_desc: str, gold_desc: str) -> bool:
    p, g = _normalize(pred_desc), _normalize(gold_desc)
    if not p or not g:
        return False
    if p == g:
        return True
    g_tokens = {t for t in re.findall(r"[a-z0-9]+", g) if len(t) >= 2}
    if not g_tokens:
        return p in g or g in p
    hits = sum(1 for t in g_tokens if t in p)
    return hits / len(g_tokens) >= 0.6


def _field(record, name):
    if record is None:
        return None
    if hasattr(record, name):
        return getattr(record, name)
    if isinstance(record, dict):
        return record.get(name)
    return None


def _coerce_record(record) -> "InvoiceRecord | None":
    if isinstance(record, InvoiceRecord):
        return record
    if isinstance(record, dict):
        try:
            return InvoiceRecord(**record)
        except Exception:
            return None
    return None


def _normalized_line_items(lst):
    out = []
    for it in lst or []:
        if isinstance(it, LineItem):
            out.append((it.description, int(it.quantity), float(it.unit_price)))
        elif isinstance(it, dict):
            try:
                out.append(
                    (
                        str(it["description"]),
                        int(it["quantity"]),
                        float(it["unit_price"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return out


def rich_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Weighted multi-axis metric returning ``dspy.Prediction(score, feedback)``.

    Axes (weights):
      - schema_ok (0.20): output parses into InvoiceRecord
      - vendor_match (0.15): normalized vendor equality (prefix-tolerant)
      - date_match (0.15): exact YYYY-MM-DD
      - line_items_f1 (0.35): set-F1 over (fuzzy_description, qty, unit_price)
      - total_match (0.15): absolute delta ≤ $0.50
    """
    raw_record = getattr(pred, "record", None)
    record = _coerce_record(raw_record)
    schema_ok = record is not None

    if not schema_ok:
        return dspy.Prediction(
            score=0.0,
            feedback=(
                f"SCHEMA: output did not parse as InvoiceRecord. Got type "
                f"{type(raw_record).__name__}. Expected a Pydantic record with "
                f"fields {{vendor, date, line_items, total}}. Each line_items entry "
                f"must have description (str), quantity (int), and unit_price (float)."
            ),
        )

    pred_vendor = _normalize(_field(record, "vendor") or "")
    gold_vendor = _normalize(gold.vendor)
    vendor_match = pred_vendor == gold_vendor
    if not vendor_match and pred_vendor and gold_vendor:
        vendor_match = pred_vendor.startswith(
            gold_vendor[:-1]
        ) or gold_vendor.startswith(pred_vendor)

    pred_date = (_field(record, "date") or "").strip()
    gold_date = gold.date
    date_match = pred_date == gold_date

    pred_triples = _normalized_line_items(_field(record, "line_items"))
    gold_triples = _normalized_line_items(gold.line_items)

    matched = 0
    unmatched_gold = list(gold_triples)
    for p_desc, p_qty, p_price in pred_triples:
        for j, (g_desc, g_qty, g_price) in enumerate(unmatched_gold):
            if (
                _fuzzy_desc_match(p_desc, g_desc)
                and p_qty == g_qty
                and abs(p_price - g_price) < 0.01
            ):
                matched += 1
                unmatched_gold.pop(j)
                break

    precision = matched / len(pred_triples) if pred_triples else 0.0
    recall = matched / len(gold_triples) if gold_triples else 0.0
    line_items_f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    pred_total = float(_field(record, "total") or 0.0)
    total_match = abs(pred_total - float(gold.total)) <= 0.50

    axes = {
        "schema": 1.0,
        "vendor": 1.0 if vendor_match else 0.0,
        "date": 1.0 if date_match else 0.0,
        "line_items_f1": line_items_f1,
        "total": 1.0 if total_match else 0.0,
    }
    weights = {
        "schema": 0.20,
        "vendor": 0.15,
        "date": 0.15,
        "line_items_f1": 0.35,
        "total": 0.15,
    }
    score = sum(axes[k] * weights[k] for k in axes)

    bits = []
    if not vendor_match:
        bits.append(
            f"VENDOR: predicted {pred_vendor!r}, expected {gold_vendor!r} "
            f"(normalize whitespace/case; copy the header verbatim)."
        )
    if not date_match:
        bits.append(
            f"DATE: predicted {pred_date!r}, expected {gold_date!r} (YYYY-MM-DD format)."
        )
    if line_items_f1 < 1.0:
        bits.append(
            f"LINE_ITEMS: F1={line_items_f1:.2f} (precision={precision:.2f}, recall={recall:.2f}, "
            f"matched {matched}/{len(gold_triples)} gold). Each line = one distinct item with integer "
            f"quantity and per-unit price — do not collapse rows or split one row into two."
        )
    if not total_match:
        bits.append(
            f"TOTAL: predicted {pred_total:.2f}, expected {gold.total:.2f} "
            f"(include tax/shipping if present in the source)."
        )
    if not bits:
        bits.append("PASS: all axes correct.")

    return dspy.Prediction(score=score, feedback=" ".join(bits))
