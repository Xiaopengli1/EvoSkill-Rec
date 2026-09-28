from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import CrossNetwork


@register_skill("crossnet_v1")
class CrossNetV1Skill(BaseSkill):
    """Wraps torch_rechub.basic.layers.CrossNetwork from the DCN model."""

    skill_spec = SkillSpec(
        name="crossnet_v1",
        category="interaction",
        description="Apply DCN v1 explicit cross layers to a flattened representation.",
        input_specs=[TensorSpec("flat_embeddings", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("cross_output", "[batch_size, input_dim]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["bounded_degree_explicit_crosses"],
        failure_signatures=["input_dim_mismatch", "rank_not_2"],
        hyperparameters={"input_dim": "int", "num_layers": "int"},
        paper_origin="Deep & Cross Network for Ad Click Predictions",
        implementation="torch_rechub.basic.layers.CrossNetwork",
    )

    def __init__(self, input_dim: int, num_layers: int = 2, input_key: str = "flat_embeddings", output_key: str = "cross_output") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_key = input_key
        self.output_key = output_key
        self.cross = CrossNetwork(input_dim, num_layers)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2:
            raise ValueError(f"{self.input_key} must have shape [batch_size, input_dim]")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_key} dim {self.input_dim}, got {x.size(-1)}")
        return ctx.put(self.output_key, self.cross(x))
