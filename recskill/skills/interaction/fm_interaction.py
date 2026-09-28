from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import FM


@register_skill("fm_interaction")
class FMInteractionSkill(BaseSkill):
    """Wraps torch_rechub.basic.layers.FM from DeepFM's second-order branch."""

    skill_spec = SkillSpec(
        name="fm_interaction",
        category="interaction",
        description="Compute second-order factorization-machine interactions over field embeddings.",
        input_specs=[TensorSpec("field_embeddings", "[batch_size, num_fields, embedding_dim]")],
        output_specs=[TensorSpec("fm_output", "[batch_size, 1]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["second_order_feature_interactions"],
        failure_signatures=["rank_not_3"],
        hyperparameters={"reduce_sum": "bool"},
        paper_origin="DeepFM / Factorization Machines",
        implementation="torch_rechub.basic.layers.FM",
    )

    def __init__(self, input_key: str = "field_embeddings", output_key: str = "fm_output", reduce_sum: bool = True) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.fm = FM(reduce_sum=reduce_sum)

    def forward(self, ctx: SkillContext) -> SkillContext:
        field_embeddings = ctx.get_required(self.input_key)
        if field_embeddings.dim() != 3:
            raise ValueError(f"{self.input_key} must have shape [batch_size, num_fields, embedding_dim]")
        return ctx.put(self.output_key, self.fm(field_embeddings))
