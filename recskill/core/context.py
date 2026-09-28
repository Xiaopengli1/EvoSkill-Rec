from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SkillContext(dict):
    """Dict-like container passed between sequential skills."""

    def __init__(self, initial: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        if initial is not None:
            self.update(initial)
        if kwargs:
            self.update(kwargs)

    def get_required(self, key: str) -> Any:
        """Return a required value or raise a clear missing-key error."""
        if key not in self:
            available = ", ".join(sorted(self.keys())) or "<empty>"
            raise KeyError(f"SkillContext missing required key '{key}'. Available keys: {available}")
        return self[key]

    def put(self, key: str, value: Any) -> "SkillContext":
        """Store a value and return this context for fluent skill code."""
        self[key] = value
        return self
