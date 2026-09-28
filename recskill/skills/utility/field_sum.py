from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("field_sum")
class FieldSumSkill(BaseSkill):
    """Sum all non-batch dimensions into a scalar branch logit."""

    skill_spec = SkillSpec(
        name="field_sum",
        category="utility",
        description="Reduce field embeddings or interaction tensors to one scalar per example.",
        input_specs=[TensorSpec("input_key", "[batch_size, ...]")],
        output_specs=[TensorSpec("output_key", "[batch_size, 1]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["first_order_embedding_sum"],
        failure_signatures=["rank_too_small"],
        hyperparameters={"input_key": "str", "output_key": "str"},
    )

    def __init__(self, input_key: str, output_key: str = "field_sum") -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() < 2:
            raise ValueError(f"{self.input_key} must have at least batch and feature dimensions")
        dims = tuple(range(1, x.dim()))
        return ctx.put(self.output_key, x.sum(dim=dims, keepdim=False).unsqueeze(-1))
