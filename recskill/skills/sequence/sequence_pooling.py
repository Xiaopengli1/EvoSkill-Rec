from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("sequence_pooling")
class SequencePoolingSkill(BaseSkill):
    """Pool sequence embeddings into one fixed-size representation."""

    skill_spec = SkillSpec(
        name="sequence_pooling",
        category="sequence",
        description="Pool sequence embeddings with mean, sum, or last-valid pooling.",
        input_specs=[TensorSpec("sequence_embeddings", "[batch_size, seq_len, embedding_dim]")],
        output_specs=[TensorSpec("sequence_summary", "[batch_size, embedding_dim]")],
        task_types=["ranking", "matching", "sequence", "generative"],
        inductive_bias=["order_aware_or_masked_pooling"],
        failure_signatures=["rank_not_3", "id_shape_mismatch"],
        hyperparameters={"mode": "mean|sum|last", "padding_idx": "int"},
    )

    def __init__(
        self,
        input_key: str = "seq_embeddings",
        ids_key: str | None = "seq_features",
        output_key: str = "sequence_summary",
        mode: str = "mean",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if mode not in {"mean", "sum", "last"}:
            raise ValueError("mode must be one of: mean, sum, last")
        self.input_key = input_key
        self.ids_key = ids_key
        self.output_key = output_key
        self.mode = mode
        self.padding_idx = padding_idx

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3:
            raise ValueError(f"{self.input_key} must have shape [batch_size, seq_len, embedding_dim]")
        mask = None
        if self.ids_key is not None and self.ids_key in ctx:
            ids = ctx.get_required(self.ids_key).long()
            if ids.shape != x.shape[:2]:
                raise ValueError(f"{self.ids_key} must match first two dims of {self.input_key}")
            mask = ids != self.padding_idx

        if self.mode == "last":
            if mask is None:
                return ctx.put(self.output_key, x[:, -1, :])
            lengths = mask.long().sum(dim=1).clamp_min(1) - 1
            batch_idx = torch.arange(x.size(0), device=x.device)
            return ctx.put(self.output_key, x[batch_idx, lengths, :])

        if mask is None:
            pooled = x.sum(dim=1)
            if self.mode == "mean":
                pooled = pooled / max(x.size(1), 1)
            return ctx.put(self.output_key, pooled)

        masked = x * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1)
        if self.mode == "mean":
            pooled = pooled / mask.long().sum(dim=1).clamp_min(1).unsqueeze(-1)
        return ctx.put(self.output_key, pooled)
