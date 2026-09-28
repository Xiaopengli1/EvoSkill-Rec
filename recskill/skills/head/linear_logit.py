from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import LR


@register_skill("linear_logit")
class LinearLogitSkill(BaseSkill):
    """Apply a linear logit projection to a 2D representation."""

    skill_spec = SkillSpec(
        name="linear_logit",
        category="head",
        description="Project a dense representation to one scalar logit.",
        input_specs=[TensorSpec("input_key", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("output_key", "[batch_size, 1]")],
        task_types=["ctr", "ranking", "matching"],
        inductive_bias=["first_order_linear_terms"],
        failure_signatures=["input_dim_mismatch"],
        hyperparameters={"input_key": "str", "output_key": "str", "input_dim": "int"},
        implementation="torch_rechub.basic.layers.LR",
    )

    def __init__(self, input_key: str, output_key: str, input_dim: int, sigmoid: bool = False) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.input_dim = input_dim
        self.linear = LR(input_dim, sigmoid=sigmoid)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        return ctx.put(self.output_key, self.linear(x))
