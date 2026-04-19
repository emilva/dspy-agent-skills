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
