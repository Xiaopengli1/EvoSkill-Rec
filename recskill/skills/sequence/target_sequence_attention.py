from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("target_sequence_attention")
class TargetSequenceAttentionSkill(BaseSkill):
    """Dot-product attention from a target item to a history sequence."""

    skill_spec = SkillSpec(
        name="target_sequence_attention",
        category="sequence",
        description="Pool a history sequence by attention against a target embedding.",
        input_specs=[
            TensorSpec("history_embeddings", "[batch_size, seq_len, embedding_dim]"),
            TensorSpec("target_embedding", "[batch_size, embedding_dim]"),
        ],
        output_specs=[TensorSpec("attention_pooling", "[batch_size, embedding_dim]")],
        task_types=["ranking", "matching", "sequence"],
        inductive_bias=["target_conditioned_history_pooling"],
        failure_signatures=["embedding_dim_mismatch", "mask_shape_mismatch"],
        hyperparameters={"embedding_dim": "int"},
        paper_origin="DIN / DIEN / STAMP-style target attention",
    )

    def __init__(
        self,
        embedding_dim: int,
        history_key: str = "history_embeddings",
        target_key: str = "target_embedding",
        ids_key: str | None = None,
        output_key: str = "attention_pooling",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.history_key = history_key
        self.target_key = target_key
        self.ids_key = ids_key
        self.output_key = output_key
        self.padding_idx = padding_idx
        self.proj = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, ctx: SkillContext) -> SkillContext:
        history = ctx.get_required(self.history_key)
        target = ctx.get_required(self.target_key)
        if target.dim() == 3 and target.size(1) == 1:
            target = target.squeeze(1)
        if history.dim() != 3 or target.dim() != 2 or history.size(-1) != self.embedding_dim or target.size(-1) != self.embedding_dim:
            raise ValueError("history and target embedding dimensions must match")
        scores = torch.sum(self.proj(history) * target.unsqueeze(1), dim=-1)
        if self.ids_key is not None and self.ids_key in ctx:
            ids = ctx.get_required(self.ids_key).long()
            if ids.shape != history.shape[:2]:
                raise ValueError(f"{self.ids_key} must match history batch and seq_len")
            scores = scores.masked_fill(ids == self.padding_idx, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return ctx.put(self.output_key, torch.sum(weights * history, dim=1))
