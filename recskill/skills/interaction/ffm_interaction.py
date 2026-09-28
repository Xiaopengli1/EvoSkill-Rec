from __future__ import annotations

from itertools import combinations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("ffm_interaction")
class FFMInteractionSkill(BaseSkill):
    """Shape-compatible field-aware pairwise interaction over field embeddings."""

    skill_spec = SkillSpec(
        name="ffm_interaction",
        category="interaction",
        description="Compute field-aware-style pairwise feature crosses from field embeddings.",
        input_specs=[TensorSpec("field_embeddings", "[batch_size, num_fields, embedding_dim]")],
        output_specs=[TensorSpec("ffm_interactions", "[batch_size, num_pairs, embedding_dim or 1]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["field_aware_pairwise_crosses"],
        failure_signatures=["rank_not_3", "field_count_mismatch"],
        hyperparameters={"num_fields": "int", "reduce_sum": "bool"},
        paper_origin="Field-aware Factorization Machines",
    )

    def __init__(self, num_fields: int, input_key: str = "field_embeddings", output_key: str = "ffm_interactions", reduce_sum: bool = False) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.input_key = input_key
        self.output_key = output_key
        self.reduce_sum = reduce_sum

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(1) != self.num_fields:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.num_fields}, embedding_dim]")
        crosses = [x[:, i, :] * x[:, j, :] for i, j in combinations(range(self.num_fields), 2)]
        out = torch.stack(crosses, dim=1)
        if self.reduce_sum:
            out = out.sum(dim=-1, keepdim=True)
        return ctx.put(self.output_key, out)
