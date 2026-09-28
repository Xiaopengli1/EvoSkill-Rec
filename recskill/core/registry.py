from __future__ import annotations

from typing import Callable, TypeVar

from .base import BaseSkill

SkillT = TypeVar("SkillT", bound=type[BaseSkill])

SKILL_REGISTRY: dict[str, type[BaseSkill]] = {}


def register_skill(name: str) -> Callable[[SkillT], SkillT]:
    """Register a BaseSkill subclass under a stable registry name."""

    def decorator(cls: SkillT) -> SkillT:
        if name in SKILL_REGISTRY and SKILL_REGISTRY[name] is not cls:
            raise KeyError(f"Skill '{name}' is already registered by {SKILL_REGISTRY[name].__name__}")
        SKILL_REGISTRY[name] = cls
        return cls

    return decorator


def build_skill(name: str, **kwargs) -> BaseSkill:
    """Instantiate a registered skill."""
    if name not in SKILL_REGISTRY:
        available = ", ".join(list_skills()) or "<none>"
        raise KeyError(f"Unknown skill '{name}'. Available skills: {available}")
    return SKILL_REGISTRY[name](**kwargs)


def list_skills() -> list[str]:
    """Return registered skill names in deterministic order."""
    return sorted(SKILL_REGISTRY)
