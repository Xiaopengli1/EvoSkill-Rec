from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.models.ranking.din import ActivationUnit


@register_skill("din_target_attention")
class DINTargetAttentionSkill(BaseSkill):
    """Wrap DIN's ActivationUnit target attention over history fields."""

    skill_spec = SkillSpec(
        name="din_target_attention",
        category="sequence",
        description="Pool behavior history embeddings using target-aware DIN attention.",
        input_specs=[
            TensorSpec("history_embeddings", "[batch_size, num_fields, seq_len, embedding_dim]"),
            TensorSpec("target_embeddings", "[batch_size, num_fields, embedding_dim]"),
        ],
        output_specs=[TensorSpec("attention_pooling", "[batch_size, num_fields, embedding_dim]")],
        task_types=["ctr", "ranking", "sequence"],
        inductive_bias=["target_conditioned_interest_pooling"],
        failure_signatures=["history_target_field_mismatch", "embedding_dim_mismatch"],
        hyperparameters={"embedding_dim": "int", "num_fields": "int", "dims": "list[int]", "activation": "str"},
        paper_origin="Deep Interest Network for Click-Through Rate Prediction",
        implementation="torch_rechub.models.ranking.din.ActivationUnit",
    )

    def __init__(
        self,
        embedding_dim: int,
        num_fields: int = 1,
        history_key: str = "history_embeddings",
        target_key: str = "target_embeddings",
        output_key: str = "attention_pooling",
        dims: list[int] | None = None,
        activation: str = "dice",
        use_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.history_key = history_key
        self.target_key = target_key
        self.output_key = output_key
        self.attention_layers = torch.nn.ModuleList(
            [ActivationUnit(embedding_dim, dims=dims, activation=activation, use_softmax=use_softmax) for _ in range(num_fields)]
        )

    def forward(self, ctx: SkillContext) -> SkillContext:
        history = ctx.get_required(self.history_key)
        target = ctx.get_required(self.target_key)
        if history.dim() != 4:
            raise ValueError(f"{self.history_key} must have shape [batch_size, num_fields, seq_len, embedding_dim]")
        if target.dim() != 3:
            raise ValueError(f"{self.target_key} must have shape [batch_size, num_fields, embedding_dim]")
        if history.size(1) != self.num_fields or target.size(1) != self.num_fields:
            raise ValueError("history and target field counts must match num_fields")
        if history.size(-1) != self.embedding_dim or target.size(-1) != self.embedding_dim:
            raise ValueError("history and target embedding dimensions must match embedding_dim")
        pooled = [
            attention(history[:, idx, :, :], target[:, idx, :]).unsqueeze(1)
            for idx, attention in enumerate(self.attention_layers)
        ]
        return ctx.put(self.output_key, torch.cat(pooled, dim=1))
