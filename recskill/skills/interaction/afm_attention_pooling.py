from __future__ import annotations

from itertools import combinations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("afm_attention_pooling")
class AFMAttentionPoolingSkill(BaseSkill):
    """Attention pooling over pairwise feature interactions."""

    skill_spec = SkillSpec(
        name="afm_attention_pooling",
        category="interaction",
        description="Pool pairwise feature interactions with an AFM-style attention network.",
        input_specs=[TensorSpec("field_embeddings", "[batch_size, num_fields, embedding_dim]")],
        output_specs=[TensorSpec("afm_output", "[batch_size, embedding_dim]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["attention_weighted_pairwise_crosses"],
        failure_signatures=["rank_not_3", "field_count_mismatch"],
        hyperparameters={"embedding_dim": "int", "num_fields": "int", "attention_dim": "int"},
        paper_origin="Attentional Factorization Machines",
    )

    def __init__(
        self,
        embedding_dim: int,
        num_fields: int,
        attention_dim: int = 64,
        input_key: str = "field_embeddings",
        output_key: str = "afm_output",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.input_key = input_key
        self.output_key = output_key
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, attention_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(1) != self.num_fields or x.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.num_fields}, {self.embedding_dim}]")
        pairs = [x[:, i, :] * x[:, j, :] for i, j in combinations(range(self.num_fields), 2)]
        pair_tensor = torch.stack(pairs, dim=1)
        weights = torch.softmax(self.attention(pair_tensor), dim=1)
        return ctx.put(self.output_key, torch.sum(weights * pair_tensor, dim=1))
