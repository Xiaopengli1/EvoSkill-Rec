from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TensorSpec:
    """Lightweight tensor contract for a skill input or output."""

    name: str
    shape: str
    dtype: str = "float32"
    description: str = ""


@dataclass(frozen=True)
class SkillSpec:
    """Metadata describing one reusable recommendation-model skill."""

    name: str
    category: str
    description: str
    input_specs: list[TensorSpec]
    output_specs: list[TensorSpec]
    task_types: list[str]
    inductive_bias: list[str]
    failure_signatures: list[str]
    hyperparameters: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    paper_origin: str | None = None
    implementation: str | None = None
