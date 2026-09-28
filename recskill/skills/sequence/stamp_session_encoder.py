from __future__ import annotations

import torch
import torch.nn.functional as F

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("stamp_session_encoder")
class STAMPSessionEncoderSkill(BaseSkill):
    """Encode a session sequence with STAMP short-term attention and memory priority."""

    skill_spec = SkillSpec(
        name="stamp_session_encoder",
        category="sequence",
        description="Compute STAMP user representations from mean memory, last item, and attention memory.",
        input_specs=[
            TensorSpec("seq_embeddings", "[batch_size, seq_len, embedding_dim]"),
            TensorSpec("seq_features", "[batch_size, seq_len]", "int64"),
        ],
        output_specs=[TensorSpec("user_embedding", "[batch_size, embedding_dim]")],
        task_types=["matching", "sequence"],
        inductive_bias=["short_term_attention_memory"],
        failure_signatures=["embedding_dim_mismatch", "all_padding_sequence"],
        hyperparameters={"embedding_dim": "int"},
        paper_origin="STAMP",
    )

    def __init__(
        self,
        embedding_dim: int,
        input_key: str = "seq_embeddings",
        ids_key: str = "seq_features",
        output_key: str = "user_embedding",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_key = input_key
        self.ids_key = ids_key
        self.output_key = output_key
        self.padding_idx = padding_idx
        self.w_0 = torch.nn.Parameter(torch.zeros(embedding_dim, 1))
        self.w_1_t = torch.nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.w_2_t = torch.nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.w_3_t = torch.nn.Parameter(torch.zeros(embedding_dim, embedding_dim))
        self.b_a = torch.nn.Parameter(torch.zeros(embedding_dim))
        self.f_s = torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Linear(embedding_dim, embedding_dim))
        self.f_t = torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Linear(embedding_dim, embedding_dim))

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        ids = ctx.get_required(self.ids_key).long()
        if x.dim() != 3 or x.size(-1) != self.embedding_dim or ids.shape != x.shape[:2]:
            raise ValueError("sequence embeddings and ids must have matching [batch_size, seq_len] dimensions")
        mask = (ids != self.padding_idx).unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp_min(1)
        masked_x = x * mask
        batch_idx = torch.arange(x.size(0), device=x.device)
        last_idx = (lengths.squeeze(-1) - 1).long()
        x_t = x[batch_idx, last_idx, :].unsqueeze(1)
        m_s = (masked_x.sum(dim=1) / lengths).unsqueeze(1)
        attention_logits = torch.sigmoid(masked_x @ self.w_1_t + x_t @ self.w_2_t + m_s @ self.w_3_t + self.b_a) @ self.w_0
        weights = torch.exp(attention_logits).masked_fill(~mask, 0.0)
        weights = F.normalize(weights, p=1, dim=1)
        m_a = (weights * masked_x).sum(dim=1) + m_s.squeeze(1)
        h_s = self.f_s(m_a)
        h_t = self.f_t(x_t).squeeze(1)
        return ctx.put(self.output_key, h_s * h_t)
