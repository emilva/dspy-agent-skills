"""Regression guards for skill doc correctness.

These rules exist because we shipped subtly wrong teaching material once
and caught it in external review. Each guard maps to a specific pitfall:

1. `.overall_score` — wrong attribute. DSPy 3.1.x's `EvaluationResult` uses
   `.score`. An agent that learns `.overall_score` will write code that
   raises `AttributeError` at runtime.
2. Dict-returning metrics — `dspy.Evaluate`'s parallel executor aggregates
   per-example outputs via `sum()`. A dict metric crashes with
   `TypeError: unsupported operand type(s) for +: 'int' and 'dict'`. Metrics
   must return `dspy.Prediction(score=..., feedback=...)`.
3. Every skill must ship a runnable `example_*.py` — `docs/usage.md` makes
   that claim, and the dry-run smoke-test loop depends on it.

Regex rules ignore blocks that explicitly mark themselves as anti-patterns
(recognised by a `# BAD` / `# WRONG` marker or a containing heading).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / "skills"
EXAMPLE_PY_PATTERN = "skills/*/example_*.py"


def _iter_skill_dirs() -> list[Path]:
    return sorted(p for p in SKILLS.iterdir() if p.is_dir())


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_no_overall_score_in_skills():
    """`result.overall_score` is the wrong attribute; DSPy returns `.score`."""
    offenders: list[str] = []
    for path in SKILLS.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".md", ".py"}:
            continue
        text = _read(path)
        for i, line in enumerate(text.splitlines(), 1):
            if "overall_score" in line:
                offenders.append(f"{path.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, (
        "`.overall_score` appears in skill docs/examples. DSPy 3.1.x uses "
        "`result.score`. Offending lines:\n  " + "\n  ".join(offenders)
    )


# A dict metric return is a real footgun: it looks plausible (matches the GEPA
# feedback shape conceptually) but crashes dspy.Evaluate. The "anti-pattern"
# mentions we intentionally write must include an explicit flag word so this
# check can ignore them.
_DICT_RETURN_RE = re.compile(r'return\s*\{\s*"score"\s*:')
_ANTIPATTERN_MARKERS = (
    "crashes",
    "anti-pattern",
    "WRONG",
    "BAD",
    "do not use",
    "don't use",
)


def _is_antipattern_context(line: str) -> bool:
    lower = line.lower()
    return any(m.lower() in lower for m in _ANTIPATTERN_MARKERS)


@pytest.mark.parametrize(
    "path",
    sorted(REPO.glob(EXAMPLE_PY_PATTERN)),
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_no_dict_metric_return_in_example(path: Path):
    """Skill example scripts must not return dict metrics (crashes Evaluate)."""
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if _DICT_RETURN_RE.search(line) and not _is_antipattern_context(line):
            pytest.fail(
                f"{path.relative_to(REPO)}:{i}: dict-returning metric detected:\n"
                f"  {line.strip()}\n"
                f"Use `dspy.Prediction(score=..., feedback=...)` instead — dicts crash "
                f"dspy.Evaluate's parallel aggregator with `TypeError: int + dict`."
            )


def test_dict_metric_guidance_in_skill_md_only_as_antipattern():
    """Skill MD files may mention dict metrics only as anti-patterns."""
    offenders: list[str] = []
    for md in SKILLS.rglob("SKILL.md"):
        text = _read(md)
        for i, line in enumerate(text.splitlines(), 1):
            if _DICT_RETURN_RE.search(line) and not _is_antipattern_context(line):
                offenders.append(f"{md.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, (
        "Dict-returning metric guidance found without anti-pattern framing:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("skill_dir", _iter_skill_dirs(), ids=lambda p: p.name)
def test_every_skill_has_example(skill_dir: Path):
    """Every skill directory must ship a runnable example_*.py (claim made in docs/usage.md)."""
    examples = list(skill_dir.glob("example_*.py"))
    assert examples, (
        f"{skill_dir.relative_to(REPO)}: no `example_*.py` found. Every skill "
        f"must ship a runnable smoke test (see `docs/usage.md`)."
    )
    for ex in examples:
        assert "--dry-run" in ex.read_text(), (
            f"{ex.relative_to(REPO)}: example must support `--dry-run` so it "
            f"can be smoke-tested offline."
        )
