from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("target_append_sequence")
class TargetAppendSequenceSkill(BaseSkill):
    """Fuse history fields per timestep and append target fields as the last token."""

    skill_spec = SkillSpec(
        name="target_append_sequence",
        category="sequence",
        description="Build a BST-style sequence by appending target embeddings after history embeddings.",
        input_specs=[
            TensorSpec("history_embeddings", "[batch_size, num_fields, seq_len, embedding_dim]"),
            TensorSpec("target_embeddings", "[batch_size, num_fields, embedding_dim]"),
        ],
        output_specs=[
            TensorSpec("sequence_embeddings", "[batch_size, seq_len + 1, num_fields * embedding_dim]"),
            TensorSpec("sequence_ids", "[batch_size, seq_len + 1]", "int64"),
        ],
        task_types=["ctr", "ranking", "sequence"],
        inductive_bias=["target_as_query_token"],
        failure_signatures=["history_target_field_mismatch"],
        hyperparameters={"num_fields": "int", "embedding_dim": "int"},
        paper_origin="BST",
    )

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        history_key: str = "history_embeddings",
        target_key: str = "target_embeddings",
        ids_key: str | None = "history_features",
        output_key: str = "sequence_embeddings",
        output_ids_key: str = "sequence_ids",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.history_key = history_key
        self.target_key = target_key
        self.ids_key = ids_key
        self.output_key = output_key
        self.output_ids_key = output_ids_key
        self.padding_idx = padding_idx

    def forward(self, ctx: SkillContext) -> SkillContext:
        history = ctx.get_required(self.history_key)
        target = ctx.get_required(self.target_key)
        if history.dim() != 4 or history.size(1) != self.num_fields or history.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.history_key} must have shape [batch_size, {self.num_fields}, seq_len, {self.embedding_dim}]")
        if target.dim() != 3 or target.size(1) != self.num_fields or target.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.target_key} must have shape [batch_size, {self.num_fields}, {self.embedding_dim}]")
        batch_size, _, seq_len, _ = history.shape
        hist_tokens = history.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_fields * self.embedding_dim)
        target_token = target.reshape(batch_size, self.num_fields * self.embedding_dim).unsqueeze(1)
        sequence = torch.cat([hist_tokens, target_token], dim=1)

        if self.ids_key is not None and self.ids_key in ctx:
            ids = ctx.get_required(self.ids_key).long()
            if ids.shape != history.shape[:3]:
                raise ValueError(f"{self.ids_key} must have shape [batch_size, {self.num_fields}, seq_len]")
            valid = (ids != self.padding_idx).any(dim=1).long()
        else:
            valid = torch.ones(batch_size, seq_len, dtype=torch.long, device=history.device)
        target_valid = torch.ones(batch_size, 1, dtype=torch.long, device=history.device)
        ctx.put(self.output_ids_key, torch.cat([valid, target_valid], dim=1))
        return ctx.put(self.output_key, sequence)
