from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import SENETLayer


@register_skill("senet_feature_gate")
class SENETFeatureGateSkill(BaseSkill):
    """Apply SENet-style field gating to field embeddings."""

    skill_spec = SkillSpec(
        name="senet_feature_gate",
        category="interaction",
        description="Reweight feature-field embeddings with a squeeze-and-excitation gate.",
        input_specs=[TensorSpec("field_embeddings", "[batch_size, num_fields, embedding_dim]")],
        output_specs=[TensorSpec("gated_embeddings", "[batch_size, num_fields, embedding_dim]")],
        task_types=["ctr", "ranking", "matching"],
        inductive_bias=["feature_importance_gating"],
        failure_signatures=["rank_not_3", "field_count_mismatch"],
        hyperparameters={"num_fields": "int", "reduction_ratio": "int"},
        implementation="torch_rechub.basic.layers.SENETLayer",
    )

    def __init__(self, num_fields: int, reduction_ratio: int = 3, input_key: str = "field_embeddings", output_key: str = "gated_embeddings") -> None:
        super().__init__()
        self.num_fields = num_fields
        self.input_key = input_key
        self.output_key = output_key
        self.gate = SENETLayer(num_fields, reduction_ratio=reduction_ratio)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(1) != self.num_fields:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.num_fields}, embedding_dim]")
        return ctx.put(self.output_key, self.gate(x))
