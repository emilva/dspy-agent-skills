---
title: "Inside the examples: how GEPA lifted a 1.2B model by 25 points"
subtitle: "Three end-to-end runs on free OpenRouter models. Real baselines. Real optimized scores. The counterintuitive reason why a smaller task LM exercises GEPA better than a bigger one."
platforms: [x.com, linkedin, code-and-context]
target_audience: AI and LLM enthusiasts who want to see GEPA work on a real task
estimated_read_time: 9 min
---

![alt: three horizontal bar charts stacked, each showing a baseline vs optimized score with a labeled delta](placeholder-hero-results.png)
<!-- IMAGE PROMPT (Nano Banana): "A clean editorial data visualization showing three stacked horizontal bar-chart panels on an off-white background. Each panel has two bars: a shorter gray 'Baseline' bar and a longer green 'Optimized' bar, with a delta annotation like '+18.85' in bold beside the pair. Panel 1 is labeled 'RAG QA · GLM 4.5 Air', bars show 81.15 → 100.00. Panel 2 is labeled 'Math reasoning · Liquid 1.2B', bars show 45.00 → 70.00. Panel 3 is labeled 'Invoice extraction · Liquid 1.2B', bars show 0.833 → 0.931. Sans-serif labels, subtle hand-sketched annotations, no gridlines, thin horizontal guide line between panels. Editorial tech-magazine feel, 16:9." -->

## The committed numbers

Three examples. All three runs used `auto="light"` and `seed=0`, free OpenRouter models only, a single reflection LM (Nemotron-3-super-120B-a12b). Total spend to reproduce: **zero dollars** if you stay inside OpenRouter's free tier.

| Example | Task LM | Baseline | Optimized | Δ | Mutations accepted |
|---|---|---:|---:|---:|---:|
| [01 RAG QA with citations](https://github.com/intertwine/dspy-agent-skills/tree/main/examples/01-rag-qa) | GLM 4.5 Air (32B) | 81.15 | **100.00** | +18.85 | 1 |
| [02 multi-step math reasoning](https://github.com/intertwine/dspy-agent-skills/tree/main/examples/02-math-reasoning) | Liquid LFM 2.5 (1.2B) | 45.00 | **70.00** | +25.00 | 5 |
| [03 typed invoice extraction](https://github.com/intertwine/dspy-agent-skills/tree/main/examples/03-invoice-extraction) | Liquid LFM 2.5 (1.2B) | 0.833 | **0.931** | +0.098 | 5 |

I'll walk each one top-to-bottom, then end with the saturation lesson that forced me to pick a 1.2B task LM for two of them.

## Example 01: RAG QA with citations

**The task.** Twelve short articles about the solar system (one per planet plus a few moons). Fifteen questions that each have a single authoritative source. The program retrieves top-3 passages via BM25, then passes them to a `dspy.ChainOfThought` that must emit both an answer *and* the doc IDs it used as evidence.

```python
class AnswerWithCitation(dspy.Signature):
    """Given retrieved passages, answer the question concisely and cite the
    passage IDs used as evidence."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="short, factual answer — no prose padding")
    citations: list[str] = dspy.OutputField(desc="doc IDs from the context that support the answer")
```

**The metric.** Three weighted axes:

- Correctness (weight 0.55): fuzzy substring or token-overlap match against gold.
- Citation validity (0.30): at least one cited doc ID is in the gold set; extras are penalized.
- Conciseness (0.15): 3–25 words, penalty either side.

The metric returns `dspy.Prediction(score, feedback)` where `feedback` names the failing axis with specifics. That's what lets the reflection LM learn what to fix.

**What GEPA did.** Iteration 1 produced an instruction that scored perfect on the full valset. GEPA spent the rest of the budget correctly refusing to mutate. Every subsequent minibatch was all-correct, so there was no signal. The final `optimized_program.json` is a 2.4 KB state file you can load in a separate Python process:

```python
import dspy
dspy.configure(lm=dspy.LM("openrouter/z-ai/glm-4.5-air:free", ...))
program = RagQA(retriever=BM25Retriever(docs))
program.load("optimized_program.json")
print(program(question="What is the orbital period of Mars?").answer)
# → '687 Earth days'
```

The improvement (+18.85 points) came from GEPA teaching the synthesizer to consistently emit citations even for questions where the model "knew" the answer from pretraining. That's a format problem, not a knowledge problem. Exactly the kind of thing prompt optimization was built for.

## Example 02: multi-step math reasoning

**The task.** Thirty-four grade-school word problems with compound percentages, work-rate problems, weighted averages, and distractor-laden setups. The program is a plain `dspy.ChainOfThought("problem -> answer")`. No tools, no external data, just reasoning.

Each data point ships a `trap` field: a one-line description of the common mistake. The metric uses it in feedback so the reflection LM learns *structural fixes* (step enumeration, unit tracking) rather than memorizing individual answers.

```json
{
  "problem": "A $200 item is marked up 25%, then discounted 20% on the marked-up price. What is the final price?",
  "answer": 200,
  "trap": "200*1.25=250, 250*0.8=200. The sequence matters; do NOT add the percentages."
}
```

**Why a 1.2B model.** This is where the saturation story starts. I ran the example first with GLM 4.5 Air (32B). Baseline: 83.33%. Ran it again with Ministral 8B. Baseline: 93.33%. Both runs accepted zero mutations. Every GEPA minibatch was all-correct, so the reflection LM was never called.

The issue isn't GEPA. Modern 8B-plus open-weight models have effectively saturated grade-school word problems. To actually see GEPA improve a program, I needed a task LM that fails on enough examples for there to be signal. [Liquid LFM 2.5 (1.2B)](https://openrouter.ai/liquid/lfm-2.5-1.2b-instruct:free) fit the bill: baseline 45.00%.

**What GEPA did.** Twenty-three iterations. Five accepted mutations (at iters 2, 3, 4, 6, 10). Candidate 4 hit the final valset score of 70.00%: a 25-point improvement entirely from rewriting the ChainOfThought instruction. The final instruction prescribes explicit step numbering, unit tracking, and a verification pass before committing the final number.

Total time: ~23 minutes on the free tier with `num_threads=1`.

![alt: GEPA candidate-frontier diagram showing five accepted mutations climbing from baseline to optimized](placeholder-gepa-frontier.png)
<!-- IMAGE PROMPT (Nano Banana): "A hand-sketched scientific-notebook style diagram titled 'GEPA candidate trajectory'. The x-axis is iteration (1 through 23). The y-axis is valset score (0.45 through 0.70). A thin gray dotted line shows the baseline at 0.45. Five distinct candidate markers (small colored circles labeled 'C1', 'C2', 'C3', 'C4', 'C5') are placed at iterations 2, 3, 4, 6, 10 respectively, climbing up to 0.70. A red star on C4 is labeled 'winner'. Subtle graph-paper grid, annotations in handwritten-style type. Off-white background, muted indigo and coral accents. 16:9." -->

## Example 03: typed invoice extraction

**The task.** Pull a typed `InvoiceRecord` out of unstructured invoice text: vendor, date (normalized to YYYY-MM-DD regardless of source format), line items with description/quantity/unit_price, and final total (after tax/shipping/discounts). The output is a Pydantic model:

```python
class InvoiceRecord(BaseModel):
    vendor: str
    date: str
    line_items: list[LineItem]
    total: float
```

The program is a `dspy.ChainOfThought("invoice_text -> record: InvoiceRecord")`. DSPy handles the Pydantic serialization round-trip automatically.

**The metric.** Five axes: schema validity (0.20), vendor match (0.15), date match (0.15), line-item F1 (0.35), total match (0.15). Feedback calls out each failing axis with specifics. F1 is computed on the set of `(fuzzy_description, quantity, unit_price)` triples, so collapsing two rows into one or splitting one row into two gets penalized as you'd expect.

**The data traps.** To prevent the same saturation I hit on ex02, I wrote invoices with deliberate gotchas:

- Dates in DD-MM-YYYY, "March 8, 2024", and "22 September 2024" formats.
- Headers with both a **seller** (vendor) and a distinct **bill-to** / **shipper** entity. The gold extraction always wants the seller.
- Discount and rebate rows that reduce the total but are *not* line items.
- Freight and handling lines that are *also* not line items.
- Tax-inclusive vs. pre-tax totals where the model has to pick the right "final" number.

**What GEPA did.** Five accepted mutations, final valset score 0.931 (baseline 0.833, +0.098). The winning candidate's instruction explicitly tells the extractor to pick the FROM: block's entity as `vendor`, to exclude freight/handling from `line_items`, and to use the most terminal total in the text.

Total time: ~36 minutes. This run exhausted OpenRouter's 2000 req/day free cap right before the final re-eval. The committed scores are GEPA's own cached full-valset evaluations (used for candidate selection), which is what you'd want: the same number GEPA optimized against.

## The saturation lesson

Two out of three examples use a **1.2B-parameter** task LM. That's deliberate.

GEPA shines when there's signal to work with. If the baseline is already 0.95 on a task, every minibatch GEPA samples is all-correct, the reflection LM is never called, and the optimizer correctly no-ops. This is **not a bug**. It's GEPA's designed behavior: don't propose changes when there's nothing to improve.

But for a *demo*, a no-op is a boring story. So I went looking for a task-model/task combination that produced enough failure signal for GEPA to actually exercise. Below 1B parameters (Gemma 3 4B, for example), OpenRouter's free tier either doesn't accept system prompts or hits structural format-output limits. Above 8B, grade-school math and typed extraction are already solved. 1.2B is the sweet spot for this particular demo.

The practical implication: **if you run the pack and see baseline == optimized, check the difficulty of your data relative to your task LM**. Either harden the data, weaken the task LM, or accept that the task is solved and spend your optimization budget somewhere that matters. [Arize's independent benchmark](https://arize.com/blog/gepa-vs-prompt-learning-benchmarking-different-prompt-optimization-approaches/) of GEPA against other optimizers makes the same point with different wording: optimizer choice matters less than metric quality and task difficulty.

## The $0 reproducibility promise

Every committed number in the table above came from a single deterministic run (`seed=0`). The model choices are captured in each example's `results.json`. The optimized programs are committed as `optimized_program.json` alongside the source, so you can load and re-score them against fresh data without re-running GEPA.

To reproduce:

```bash
git clone https://github.com/intertwine/dspy-agent-skills
cd dspy-agent-skills
cp .env.example .env  # fill in OPENROUTER_API_KEY

cd examples/02-math-reasoning
DSPY_TASK_MODEL=openrouter/liquid/lfm-2.5-1.2b-instruct:free \
  uv run --with dspy --with python-dotenv \
  python run.py --optimize --auto light --seed 0
```

About 20–45 minutes depending on free-tier congestion. You should land within a few points of the committed number (single-seed runs aren't perfectly deterministic because free-tier model providers occasionally adjust behind the endpoint).

## What the examples aren't

Three reproducible demos, not a benchmark suite. Two dozen to three dozen examples in each train/val set is small by academic standards. The point is to show the *shape* of a GEPA loop end-to-end on a single machine in under an hour, not to claim state-of-the-art on any particular task. If you want rigorous numbers, the [GEPA paper](https://arxiv.org/abs/2507.19457) is the source; the [DSPy tutorials](https://dspy.ai/tutorials/gepa_aime/) show the AIME benchmark run properly.

## Next

[Article 4](04-future-use-cases.md) sketches the composition patterns I'm most excited about: RLM inside a GEPA-optimized module, multi-agent workflows where one program optimizes another, and production deployment patterns that take a saved program into a real service.
