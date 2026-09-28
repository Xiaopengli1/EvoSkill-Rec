from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("additive_fusion")
class AdditiveFusionSkill(BaseSkill):
    """Sum several compatible tensors into one context value."""

    skill_spec = SkillSpec(
        name="additive_fusion",
        category="utility",
        description="Add multiple branch tensors with matching or broadcast-compatible shapes.",
        input_specs=[TensorSpec("input_keys", "list[tensor]")],
        output_specs=[TensorSpec("output_key", "same as broadcasted inputs")],
        task_types=["ctr", "ranking", "matching", "multitask", "generative"],
        inductive_bias=["residual_or_logit_addition"],
        failure_signatures=["empty_inputs", "shape_not_broadcastable"],
        hyperparameters={"input_keys": "list[str]", "output_key": "str"},
    )

    def __init__(self, input_keys: list[str], output_key: str = "fused_sum") -> None:
        super().__init__()
        if not input_keys:
            raise ValueError("input_keys must not be empty")
        self.input_keys = list(input_keys)
        self.output_key = output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        values = [ctx.get_required(key) for key in self.input_keys]
        total = torch.zeros_like(values[0])
        for value in values:
            total = total + value
        return ctx.put(self.output_key, total)
