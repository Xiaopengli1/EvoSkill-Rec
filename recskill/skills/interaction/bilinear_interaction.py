from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import BiLinearInteractionLayer


@register_skill("bilinear_interaction")
class BilinearInteractionSkill(BaseSkill):
    """Compute pairwise bilinear feature interactions."""

    skill_spec = SkillSpec(
        name="bilinear_interaction",
        category="interaction",
        description="Produce pairwise bilinear interaction embeddings between feature fields.",
        input_specs=[TensorSpec("field_embeddings", "[batch_size, num_fields, embedding_dim]")],
        output_specs=[TensorSpec("bilinear_interactions", "[batch_size, num_pairs, embedding_dim]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["pairwise_bilinear_crosses"],
        failure_signatures=["rank_not_3", "field_count_mismatch"],
        hyperparameters={"embedding_dim": "int", "num_fields": "int", "bilinear_type": "str"},
        implementation="torch_rechub.basic.layers.BiLinearInteractionLayer",
    )

    def __init__(
        self,
        embedding_dim: int,
        num_fields: int,
        bilinear_type: str = "field_interaction",
        input_key: str = "field_embeddings",
        output_key: str = "bilinear_interactions",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.input_key = input_key
        self.output_key = output_key
        self.layer = BiLinearInteractionLayer(embedding_dim, num_fields, bilinear_type=bilinear_type)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(1) != self.num_fields or x.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.num_fields}, {self.embedding_dim}]")
        return ctx.put(self.output_key, self.layer(x))
