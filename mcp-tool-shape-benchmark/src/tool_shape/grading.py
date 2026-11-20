"""Exact, code-only grading of a submitted answer against a task's ground truth."""

from __future__ import annotations

from typing import Any

from tool_shape.tasks import Task


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return value


def grade(task: Task, answer: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """(all fields correct, names of the wrong or missing fields). No answer fails every field."""
    if answer is None:
        return False, list(task.fields)
    wrong = [name for name, want in task.expected.items() if _norm(answer.get(name)) != _norm(want)]
    return not wrong, wrong
