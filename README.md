# DSPy Agent Skills

**Production-grade DSPy 3.1.x skills for coding agents.** A synthesized, spec-compliant pack of five agent skills that turns Claude Code, Codex CLI, and any other [agentskills.io](https://agentskills.io)-compatible agent into a DSPy expert.

- ✅ Validated against DSPy 3.1.3 (the real API, not inferred from older docs)
- ✅ Single source of truth for both **Claude Code** and **Codex CLI**
- ✅ Progressive disclosure (short `SKILL.md` + deep `reference.md`)
- ✅ Runnable `example_*.py` scripts with offline `--dry-run`
- ✅ Plugin manifest + marketplace manifest for one-click install
- ✅ 34 validation tests (frontmatter spec, JSON schema, Python AST)

## What's inside

| Skill | When it auto-invokes |
|---|---|
| [`dspy-fundamentals`](skills/dspy-fundamentals/SKILL.md) | Any new DSPy code: Signatures, Modules, Predict/ChainOfThought/ReAct, save/load |
| [`dspy-evaluation-harness`](skills/dspy-evaluation-harness/SKILL.md) | Writing metrics, splitting dev/val sets, calling `dspy.Evaluate` |
| [`dspy-gepa-optimizer`](skills/dspy-gepa-optimizer/SKILL.md) | Optimizing/compiling DSPy programs with `dspy.GEPA` |
| [`dspy-rlm-module`](skills/dspy-rlm-module/SKILL.md) | Long context, codebase QA, recursive exploration via `dspy.RLM` |
| [`dspy-advanced-workflow`](skills/dspy-advanced-workflow/SKILL.md) | End-to-end builds — orchestrates the other four |

## Install

### Claude Code (via marketplace)

```text
/plugin marketplace add intertwine/dspy-agent-skills
/plugin install dspy-agent-skills@dspy-agent-skills
```

### Claude Code + Codex (one command)

```bash
git clone https://github.com/intertwine/dspy-agent-skills
cd dspy-agent-skills
./scripts/install.sh           # symlinks into ~/.claude/skills/ and ~/.agents/skills/
```

Flags: `--claude-only`, `--codex-only`, `--copy` (copy instead of symlink), `--uninstall`, `--dry-run`.

### Manual

Drop `skills/*` into `~/.claude/skills/` (Claude Code) or `~/.agents/skills/` (Codex CLI). See [docs/installation.md](docs/installation.md) for all options.

## Five-second demo

In your agent, say:

> "Build a DSPy sentiment classifier, optimize it with GEPA, and save the artifact."

The agent auto-loads `dspy-advanced-workflow`, which chains the other skills and outputs a full baseline → GEPA → export pipeline. No further prompting needed.

## Why this pack exists

Earlier drafts had several plausible-but-wrong details — wrong `GEPA` import path, deprecated `TypedPredictor`, frontmatter fields that Claude Code ignores (`triggers`, `version`, `dspy-compatibility`). This pack is the corrected, source-verified successor. See [docs/CHANGELOG.md](docs/CHANGELOG.md) for the full diff.

Every API claim is grounded in:

- https://dspy.ai/ (official docs, DSPy 3.1.x)
- https://code.claude.com/docs/en/skills.md (Claude Code skill spec)
- https://developers.openai.com/codex/skills (Codex skill spec)

## Development

```bash
# Run validation suite
uv run --with pytest python -m pytest tests/ -v

# Smoke-test every example offline (no API key needed)
for f in skills/*/example_*.py; do uv run --with dspy python "$f" --dry-run; done

# Live GEPA run (requires OPENAI_API_KEY)
cd skills/dspy-advanced-workflow
OPENAI_API_KEY=... uv run --with dspy python example_pipeline.py --auto light
```

## Compatibility

- **DSPy**: 3.1.x (tested against 3.1.3)
- **Claude Code**: current (skill spec as of 2026-04-17)
- **Codex CLI**: current Agent Skills format
- **Python**: 3.10+
- **Deno**: required only for `dspy.RLM` examples (Pyodide sandbox)

## Layout

```
dspy-agent-skills/
├── .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── skills/
│   ├── dspy-fundamentals/{SKILL.md, reference.md, example_qa.py}
│   ├── dspy-evaluation-harness/{SKILL.md, reference.md, example_metric.py}
│   ├── dspy-gepa-optimizer/{SKILL.md, reference.md, example_gepa.py}
│   ├── dspy-rlm-module/{SKILL.md, reference.md}
│   └── dspy-advanced-workflow/{SKILL.md, example_pipeline.py}
├── scripts/install.sh           # dual-target installer
├── tests/                       # spec validators
├── docs/{installation,usage,CHANGELOG}.md
├── README.md  LICENSE  .gitignore
```

## Version

**v2.2.0** — April 19, 2026  •  Targets DSPy 3.1.x

## License

MIT — see [LICENSE](LICENSE).

## Credits

Draft contributors: Bryan Young ([@intertwine](https://github.com/intertwine)) with Grok (xAI).
Validation, spec-alignment, and dual-agent packaging: Claude Opus 4.7, April 2026.
