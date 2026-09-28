from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("concat_fusion")
class ConcatFusionSkill(BaseSkill):
    """Concatenate branch representations for a downstream head."""

    skill_spec = SkillSpec(
        name="concat_fusion",
        category="utility",
        description="Concatenate tensors from multiple context keys along the last dimension.",
        input_specs=[TensorSpec("input_keys", "list[[batch_size, dim_i]]")],
        output_specs=[TensorSpec("fused_representation", "[batch_size, sum(dim_i)]")],
        task_types=["ctr", "ranking", "multitask"],
        inductive_bias=["late_feature_fusion"],
        failure_signatures=["missing_input_key", "batch_size_mismatch"],
        hyperparameters={"input_keys": "list[str]", "output_key": "str"},
        implementation="torch.cat",
    )

    def __init__(self, input_keys: list[str], output_key: str = "fused_representation", dim: int = -1) -> None:
        super().__init__()
        if not input_keys:
            raise ValueError("input_keys must not be empty")
        self.input_keys = list(input_keys)
        self.output_key = output_key
        self.dim = dim

    def forward(self, ctx: SkillContext) -> SkillContext:
        tensors = []
        for key in self.input_keys:
            value = ctx.get_required(key)
            if value.dim() == 1:
                value = value.unsqueeze(-1)
            tensors.append(value)
        return ctx.put(self.output_key, torch.cat(tensors, dim=self.dim))
