"""Shared helpers for all runnable DSPy examples."""

from .config import (
    get_task_lm,
    get_reflection_lm,
    configure_dspy,
    DEFAULT_TASK_MODEL,
    DEFAULT_REFLECTION_MODEL,
)

__all__ = [
    "get_task_lm",
    "get_reflection_lm",
    "configure_dspy",
    "DEFAULT_TASK_MODEL",
    "DEFAULT_REFLECTION_MODEL",
]
