from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import CrossNetV2


@register_skill("crossnet_v2")
class CrossNetV2Skill(BaseSkill):
    """Wrap DCNv2's full-rank cross network."""

    skill_spec = SkillSpec(
        name="crossnet_v2",
        category="interaction",
        description="Apply DCNv2 full-rank explicit feature crossing to a flat representation.",
        input_specs=[TensorSpec("flat_embeddings", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("cross_output", "[batch_size, input_dim]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["explicit_high_order_crosses"],
        failure_signatures=["input_dim_mismatch"],
        hyperparameters={"input_dim": "int", "num_layers": "int"},
        paper_origin="DCN V2",
        implementation="torch_rechub.basic.layers.CrossNetV2",
    )

    def __init__(self, input_dim: int, num_layers: int, input_key: str = "flat_embeddings", output_key: str = "cross_output") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_key = input_key
        self.output_key = output_key
        self.cross = CrossNetV2(input_dim, num_layers)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        return ctx.put(self.output_key, self.cross(x))
