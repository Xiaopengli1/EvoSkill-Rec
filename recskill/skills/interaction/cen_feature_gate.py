from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import CEN


@register_skill("cen_feature_gate")
class CENFeatureGateSkill(BaseSkill):
    """Apply FAT-DeepFFM compose-excitation gating to FFM interactions."""

    skill_spec = SkillSpec(
        name="cen_feature_gate",
        category="interaction",
        description="Rescale pairwise FFM interaction embeddings with CEN field attention.",
        input_specs=[TensorSpec("ffm_interactions", "[batch_size, num_pairs, embedding_dim]")],
        output_specs=[TensorSpec("gated_ffm_flat", "[batch_size, num_pairs * embedding_dim]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["field_cross_attention"],
        failure_signatures=["rank_not_3", "shape_mismatch"],
        hyperparameters={"embedding_dim": "int", "num_field_crosses": "int", "reduction_ratio": "int"},
        paper_origin="FAT-DeepFFM",
        implementation="torch_rechub.basic.layers.CEN",
    )

    def __init__(
        self,
        embedding_dim: int,
        num_field_crosses: int,
        reduction_ratio: int,
        input_key: str = "ffm_interactions",
        output_key: str = "gated_ffm_flat",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_field_crosses = num_field_crosses
        self.input_key = input_key
        self.output_key = output_key
        self.cen = CEN(embedding_dim, num_field_crosses, reduction_ratio)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(1) != self.num_field_crosses or x.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.num_field_crosses}, {self.embedding_dim}]")
        return ctx.put(self.output_key, self.cen(x))
