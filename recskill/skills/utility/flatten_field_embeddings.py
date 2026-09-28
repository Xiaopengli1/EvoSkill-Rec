from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("flatten_field_embeddings")
class FlattenFieldEmbeddingsSkill(BaseSkill):
    """Flatten field embeddings as DeepFM does before its deep MLP branch."""

    skill_spec = SkillSpec(
        name="flatten_field_embeddings",
        category="utility",
        description="Flatten [B, F, D] field embeddings into [B, F * D].",
        input_specs=[TensorSpec("field_embeddings", "[batch_size, num_fields, embedding_dim]")],
        output_specs=[TensorSpec("flat_embeddings", "[batch_size, num_fields * embedding_dim]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["field_order_preserving_flatten"],
        failure_signatures=["rank_not_3"],
        implementation="torch.Tensor.flatten",
    )

    def __init__(self, input_key: str = "field_embeddings", output_key: str = "flat_embeddings") -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        field_embeddings = ctx.get_required(self.input_key)
        if field_embeddings.dim() != 3:
            raise ValueError(f"{self.input_key} must have shape [batch_size, num_fields, embedding_dim]")
        return ctx.put(self.output_key, field_embeddings.flatten(start_dim=1))
