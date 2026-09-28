from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("dien_interest_evolution")
class DIENInterestEvolutionSkill(BaseSkill):
    """Shape-compatible DIEN interest extraction and target-aware evolution."""

    skill_spec = SkillSpec(
        name="dien_interest_evolution",
        category="sequence",
        description="Encode behavior histories with GRUs and pool evolved interests against target embeddings.",
        input_specs=[
            TensorSpec("history_embeddings", "[batch_size, num_fields, seq_len, embedding_dim]"),
            TensorSpec("target_embeddings", "[batch_size, num_fields, embedding_dim]"),
        ],
        output_specs=[
            TensorSpec("interest_evolving", "[batch_size, num_fields, embedding_dim]"),
            TensorSpec("aux_loss", "[]"),
        ],
        task_types=["ctr", "ranking", "sequence"],
        inductive_bias=["recurrent_interest_evolution", "target_conditioned_attention"],
        failure_signatures=["history_target_field_mismatch"],
        hyperparameters={"embedding_dim": "int", "num_fields": "int"},
        paper_origin="DIEN",
    )

    def __init__(
        self,
        embedding_dim: int,
        num_fields: int = 1,
        history_key: str = "history_embeddings",
        target_key: str = "target_embeddings",
        ids_key: str | None = "history_features",
        output_key: str = "interest_evolving",
        aux_loss_key: str = "aux_loss",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.history_key = history_key
        self.target_key = target_key
        self.ids_key = ids_key
        self.output_key = output_key
        self.aux_loss_key = aux_loss_key
        self.padding_idx = padding_idx
        self.grus = torch.nn.ModuleList([torch.nn.GRU(embedding_dim, embedding_dim, batch_first=True) for _ in range(num_fields)])
        self.attention = torch.nn.ModuleList([torch.nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(num_fields)])

    def forward(self, ctx: SkillContext) -> SkillContext:
        history = ctx.get_required(self.history_key)
        target = ctx.get_required(self.target_key)
        if history.dim() != 4 or history.size(1) != self.num_fields or history.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.history_key} must have shape [batch_size, {self.num_fields}, seq_len, {self.embedding_dim}]")
        if target.dim() != 3 or target.size(1) != self.num_fields or target.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.target_key} must have shape [batch_size, {self.num_fields}, {self.embedding_dim}]")
        ids = None
        if self.ids_key is not None and self.ids_key in ctx:
            ids = ctx.get_required(self.ids_key).long()
            if ids.shape != history.shape[:3]:
                raise ValueError(f"{self.ids_key} must have shape [batch_size, {self.num_fields}, seq_len]")

        evolved = []
        for idx, (gru, attention) in enumerate(zip(self.grus, self.attention)):
            seq = history[:, idx, :, :]
            out, _ = gru(seq)
            scores = (attention(out) * target[:, idx, :].unsqueeze(1)).sum(dim=-1)
            valid_rows = None
            if ids is not None:
                mask = ids[:, idx, :] != self.padding_idx
                scores = scores.masked_fill(~mask, -1e9)
                valid_rows = mask.any(dim=1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = torch.sum(weights * out, dim=1)
            if valid_rows is not None:
                pooled = torch.where(valid_rows.unsqueeze(-1), pooled, torch.zeros_like(pooled))
            evolved.append(pooled.unsqueeze(1))

        ctx.put(self.aux_loss_key, history.new_tensor(0.0))
        return ctx.put(self.output_key, torch.cat(evolved, dim=1))
