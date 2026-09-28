from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import CrossNetMix


@register_skill("crossnet_mix")
class CrossNetMixSkill(BaseSkill):
    """Wrap DCNv2's low-rank mixture cross network."""

    skill_spec = SkillSpec(
        name="crossnet_mix",
        category="interaction",
        description="Apply DCNv2 low-rank mixture explicit feature crossing to a flat representation.",
        input_specs=[TensorSpec("flat_embeddings", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("cross_output", "[batch_size, input_dim]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["low_rank_expert_crosses"],
        failure_signatures=["input_dim_mismatch"],
        hyperparameters={"input_dim": "int", "num_layers": "int", "low_rank": "int", "num_experts": "int"},
        paper_origin="DCN V2",
        implementation="torch_rechub.basic.layers.CrossNetMix",
    )

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        low_rank: int = 32,
        num_experts: int = 4,
        input_key: str = "flat_embeddings",
        output_key: str = "cross_output",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_key = input_key
        self.output_key = output_key
        self.cross = CrossNetMix(input_dim, num_layers=num_layers, low_rank=low_rank, num_experts=num_experts)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        out = self.cross(x)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return ctx.put(self.output_key, out)
