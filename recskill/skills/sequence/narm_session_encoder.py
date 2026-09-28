from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("narm_session_encoder")
class NARMSessionEncoderSkill(BaseSkill):
    """Encode a session sequence with NARM global-local attention."""

    skill_spec = SkillSpec(
        name="narm_session_encoder",
        category="sequence",
        description="Build NARM session representations from GRU hidden states and local attention.",
        input_specs=[
            TensorSpec("seq_embeddings", "[batch_size, seq_len, embedding_dim]"),
            TensorSpec("seq_features", "[batch_size, seq_len]", "int64"),
        ],
        output_specs=[TensorSpec("user_embedding", "[batch_size, embedding_dim]")],
        task_types=["matching", "sequence"],
        inductive_bias=["session_global_local_attention"],
        failure_signatures=["embedding_dim_mismatch", "all_padding_sequence"],
        hyperparameters={"embedding_dim": "int", "hidden_dim": "int"},
        paper_origin="NARM",
    )

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        input_key: str = "seq_embeddings",
        ids_key: str = "seq_features",
        output_key: str = "user_embedding",
        dropout: float = 0.0,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_key = input_key
        self.ids_key = ids_key
        self.output_key = output_key
        self.padding_idx = padding_idx
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.a_1 = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.a_2 = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.v = torch.nn.Parameter(torch.randn(hidden_dim, 1))
        self.proj = torch.nn.Linear(hidden_dim * 2, embedding_dim, bias=False)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        ids = ctx.get_required(self.ids_key).long()
        if x.dim() != 3 or x.size(-1) != self.embedding_dim or ids.shape != x.shape[:2]:
            raise ValueError("sequence embeddings and ids must have matching [batch_size, seq_len] dimensions")
        mask = ids != self.padding_idx
        out, h_t = self.gru(self.dropout(x))
        c_g = h_t[-1]
        q = torch.sigmoid(torch.matmul(c_g.unsqueeze(1), self.a_1.T) + torch.matmul(out, self.a_2.T))
        alpha = torch.exp(torch.matmul(q, self.v)).masked_fill(~mask.unsqueeze(-1), 0.0)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
        c_l = (alpha * out).sum(dim=1)
        return ctx.put(self.output_key, self.proj(torch.cat([c_g, c_l], dim=-1)))
