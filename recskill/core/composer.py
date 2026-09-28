from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import yaml

from .context import SkillContext
from .registry import build_skill

_PLACEHOLDER_RE = re.compile(r"^\$\{([^}]+)\}$")


class SkillGraphModel(torch.nn.Module):
    """Sequential skill graph model built from a genome YAML file."""

    def __init__(self, skills: list[torch.nn.Module], genome: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.skills = torch.nn.ModuleList(skills)
        self.genome = genome or {}

    def forward(self, batch) -> SkillContext:
        ctx = SkillContext()
        if not isinstance(batch, dict):
            raise TypeError(f"SkillGraphModel expects batch to be a dict, got {type(batch).__name__}")
        for key, value in batch.items():
            ctx.put(key, value)
        for skill in self.skills:
            ctx = skill(ctx)
            if not isinstance(ctx, SkillContext):
                raise TypeError(f"{skill.__class__.__name__}.forward must return SkillContext")
        return ctx


def build_model_from_genome(genome_path, runtime_params: dict[str, Any] | None = None) -> SkillGraphModel:
    """Build a sequential SkillGraphModel from a genome YAML file."""
    runtime_params = runtime_params or {}
    path = Path(genome_path)
    with path.open("r", encoding="utf-8") as f:
        genome = yaml.safe_load(f) or {}

    skill_modules: list[torch.nn.Module] = []
    for step in genome.get("skills", []):
        skill_name = step["skill"]
        raw_params = step.get("params", {}) or {}
        params = _resolve_placeholders(raw_params, runtime_params)
        module = build_skill(skill_name, **params)
        module.skill_id = step.get("id", skill_name)
        skill_modules.append(module)

    return SkillGraphModel(skill_modules, genome=genome)


def _resolve_placeholders(value: Any, runtime_params: dict[str, Any]) -> Any:
    if isinstance(value, str):
        match = _PLACEHOLDER_RE.match(value)
        if match:
            key = match.group(1)
            if key not in runtime_params:
                raise KeyError(f"Missing runtime param '{key}' required by genome")
            return runtime_params[key]
        return value
    if isinstance(value, list):
        return [_resolve_placeholders(item, runtime_params) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_placeholders(item, runtime_params) for item in value)
    if isinstance(value, dict):
        return {key: _resolve_placeholders(item, runtime_params) for key, item in value.items()}
    return value
