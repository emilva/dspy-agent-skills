# Changelog

## v2.2.0 — 2026-04-19

First published release. Synthesis and correction of the initial `PLAN.md` draft into a spec-compliant pack that installs cleanly in Claude Code and Codex CLI.

### Skills

- `dspy-fundamentals` — Signatures, Modules, Predict/ChainOfThought/ReAct/ProgramOfThought, save/load
- `dspy-evaluation-harness` — rich-feedback metrics, `dspy.Evaluate`, multi-axis scoring
- `dspy-gepa-optimizer` — full `dspy.GEPA` API (all 22 constructor params)
- `dspy-rlm-module` — `dspy.RLM` long-context / recursive REPL usage
- `dspy-advanced-workflow` — orchestrated end-to-end pipeline

### Corrections to the original PLAN.md draft

| Draft | Correction | Source |
|---|---|---|
| `from dspy.optimizers import GEPA` | `import dspy; dspy.GEPA(...)` (or `from dspy.teleprompt import GEPA`) | https://dspy.ai/api/optimizers/GEPA/overview/ |
| Used `dspy.TypedPredictor` + Pydantic | Use `dspy.Predict` with Pydantic-typed fields (TypedPredictor superseded) | https://dspy.ai/api/modules/Predict/ |
| `dspy.configure(lm=dspy.LM("openai/gpt-5"))` — speculative | Kept `openai/gpt-4o` as default; `DSPY_MODEL` env override | https://dspy.ai/api/models/LM/ |
| GEPA constructor params incomplete (~6 listed) | All 22 params documented with defaults | https://dspy.ai/api/optimizers/GEPA/overview/ |
| `dspy.RLM` args incomplete | Added `max_llm_calls`, `max_output_chars`, `interpreter`; noted Deno requirement | https://dspy.ai/api/modules/RLM/ |
| `dspy.Evaluate(return_all_scores=...)` | `num_threads`, `display_table`, `provide_traceback`, `save_as_csv/json` (the real kwargs) | https://dspy.ai/api/evaluation/Evaluate/ |
| SKILL.md frontmatter used `triggers`, `version`, `dspy-compatibility` | Removed — Claude Code ignores them; version lives in plugin.json. Use `description` + `when_to_use` for auto-invocation | https://code.claude.com/docs/en/skills.md |
| Relied on `npx skillfish add ...` installer | Replaced with official `/plugin marketplace add` path + `scripts/install.sh` for dual-target | https://code.claude.com/docs/en/plugin-marketplaces.md, https://developers.openai.com/codex/skills |
| Single-format distribution | Added `.claude-plugin/{plugin.json, marketplace.json}` + Codex `~/.agents/skills/` support | — |

### Validation / discovered issues

- GEPA asserts `reflection_lm is not None` at **construction time**, not compile — documented as a pitfall in the GEPA skill, and dry-run examples now pass a stub `dspy.LM(...)`.
- 34 tests now cover: SKILL.md frontmatter (spec fields only, kebab-case names, length limits, filename case), plugin/marketplace JSON schemas, and example Python AST parsing.
- All four example scripts execute offline via `--dry-run` against real DSPy 3.1.x.

### Distribution

- Claude Code marketplace manifest (`.claude-plugin/marketplace.json`)
- Claude Code plugin manifest (`.claude-plugin/plugin.json`)
- `scripts/install.sh` for direct install into `~/.claude/skills/` and `~/.agents/skills/` (symlink or copy, idempotent, `--uninstall` supported)

### End-to-end examples

Three validated showcases under `examples/`, each with committed baseline vs. GEPA-optimized numbers:

- `examples/01-rag-qa/` — RAG with citations. GLM 4.5 Air (32B): **81.15 → 100.00 (+18.85)**, 1 mutation accepted.
- `examples/02-math-reasoning/` — multi-step arithmetic. Liquid LFM 2.5 (1.2B): **45.00 → 70.00 (+25.00)**, 5 mutations accepted.
- `examples/03-invoice-extraction/` — Pydantic-typed invoice records. Liquid LFM 2.5 (1.2B): **0.833 → 0.931 (+0.098)**, 5 mutations accepted.

Each example ships `pipeline.py` (module + Signature + metric), `run.py` (CLI: `--dry-run` / `--baseline` / `--optimize` / `--eval`), data JSONL, and `results.{json,md}` from the author's run. Default models are all free on OpenRouter.

### Lessons discovered during validation (baked into the skills)

1. **GEPA metric return must be `dspy.Prediction(score, feedback)`, not a dict.** DSPy's parallel evaluator cannot sum dict-typed outputs (`TypeError: int + dict`). Every example's metric uses the Prediction shape.
2. **GEPA requires `reflection_lm` at construction time**, not at `.compile()`. A cheap stub `dspy.LM(...)` is sufficient for dry-runs (construction is a no-op network-wise).
3. **GEPA state pickling needs cloudpickle** when signatures/modules are dynamic (e.g., typed Pydantic outputs). Pass `gepa_kwargs={"use_cloudpickle": True}`.
4. **`from __future__ import annotations` breaks Pydantic-typed DSPy signatures** — DSPy receives a `ForwardRef` string instead of the actual type. Ex03's pipeline deliberately omits the future import.
5. **`reflection_minibatch_size` matters more than you'd expect.** With small minibatches and high baseline accuracy, GEPA keeps sampling all-correct subsets and the reflection LM is never called. Raise to 6–8 when baseline > 0.7.
6. **Modern 8B+ open models saturate simple extraction and grade-school math**. The math and invoice examples use Liquid 1.2B to create real headroom; stronger models produce baseline ≥ 0.95 and GEPA correctly no-ops.

### Post-release review fixes (Codex audit)

- Replaced `result.overall_score` with the real attribute `result.score` in every skill (`dspy.EvaluationResult` does not have `.overall_score`).
- Skill docs and examples now return `dspy.Prediction(score, feedback)` everywhere — the earlier dict-returning examples would crash `dspy.Evaluate`'s parallel aggregator.
- Added `skills/dspy-rlm-module/example_rlm.py` (every skill now has a runnable `example_*.py`, matching the claim in `docs/usage.md`).
- Fixed `dspy.RLM` `max_output_chars` default: was documented as `10_000`, real default is `100_000`.
- Added `wandb_api_key` and `wandb_init_kwargs` to the GEPA constructor listing.
- Corrected `examples/README.md`'s reproduction instructions (removed references to non-existent `runs/latest/results.json` and `--bench --seeds 5` flag).
- New regression tests (`tests/test_skill_correctness.py`) now fail on: any `.overall_score` in skill docs, dict-returning metrics in skill examples, or a skill missing `example_*.py`.
