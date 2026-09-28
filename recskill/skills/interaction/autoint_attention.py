from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import InteractingLayer


@register_skill("autoint_attention")
class AutoIntAttentionSkill(BaseSkill):
    """Stack AutoInt interacting layers over field embeddings."""

    skill_spec = SkillSpec(
        name="autoint_attention",
        category="interaction",
        description="Apply multi-head self-attention interaction layers over feature fields.",
        input_specs=[TensorSpec("field_embeddings", "[batch_size, num_fields, embedding_dim]")],
        output_specs=[TensorSpec("attention_embeddings", "[batch_size, num_fields, embedding_dim]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["self_attention_feature_interactions"],
        failure_signatures=["rank_not_3", "embedding_dim_not_divisible_by_heads"],
        hyperparameters={"embedding_dim": "int", "num_layers": "int", "num_heads": "int"},
        paper_origin="AutoInt",
        implementation="torch_rechub.basic.layers.InteractingLayer",
    )

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int = 3,
        num_heads: int = 2,
        dropout: float = 0.0,
        residual: bool = True,
        input_key: str = "field_embeddings",
        output_key: str = "attention_embeddings",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_key = input_key
        self.output_key = output_key
        self.layers = torch.nn.ModuleList([InteractingLayer(embedding_dim, num_heads=num_heads, dropout=dropout, residual=residual) for _ in range(num_layers)])

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, num_fields, {self.embedding_dim}]")
        for layer in self.layers:
            x = layer(x)
        return ctx.put(self.output_key, x)
