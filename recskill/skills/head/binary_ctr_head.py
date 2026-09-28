from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("binary_ctr_head")
class BinaryCTRHeadSkill(BaseSkill):
    """Binary CTR logit head for fused ranking representations."""

    skill_spec = SkillSpec(
        name="binary_ctr_head",
        category="head",
        description="Project a representation to a single raw CTR logit.",
        input_specs=[TensorSpec("input_key", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("logits", "[batch_size, 1]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["linear_logit_projection"],
        failure_signatures=["input_dim_mismatch"],
        hyperparameters={"input_key": "str", "input_dim": "int", "output_key": "str"},
        implementation="torch.nn.Linear",
    )

    def __init__(self, input_key: str, input_dim: int, output_key: str = "logits") -> None:
        super().__init__()
        self.input_key = input_key
        self.input_dim = input_dim
        self.output_key = output_key
        self.proj = torch.nn.Linear(input_dim, 1)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2:
            raise ValueError(f"{self.input_key} must have shape [batch_size, input_dim]")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_key} dim {self.input_dim}, got {x.size(-1)}")
        return ctx.put(self.output_key, self.proj(x))
