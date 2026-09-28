from __future__ import annotations

import torch
import torch.nn.functional as F

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("sine_interest_encoder")
class SINEInterestEncoderSkill(BaseSkill):
    """Sparse-interest encoder for SINE-style sequential matching."""

    skill_spec = SkillSpec(
        name="sine_interest_encoder",
        category="sequence",
        description="Extract sparse conceptual interests and aggregate them into one next-intention user representation.",
        input_specs=[
            TensorSpec("seq_embeddings", "[batch_size, seq_len, embedding_dim]"),
            TensorSpec("seq_features", "[batch_size, seq_len]", "int64"),
        ],
        output_specs=[TensorSpec("user_embedding", "[batch_size, embedding_dim]")],
        task_types=["matching", "sequence"],
        inductive_bias=["sparse_concept_interest", "adaptive_intention_aggregation"],
        failure_signatures=["seq_len_mismatch", "all_padding_sequence"],
        hyperparameters={"embedding_dim": "int", "hidden_dim": "int", "num_concept": "int", "num_intention": "int"},
        paper_origin="SINE",
    )

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_concept: int,
        num_intention: int,
        seq_len: int,
        num_heads: int = 1,
        temperature: float = 1.0,
        input_key: str = "seq_embeddings",
        ids_key: str = "seq_features",
        output_key: str = "user_embedding",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_concept = num_concept
        self.num_intention = num_intention
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.temperature = temperature
        self.input_key = input_key
        self.ids_key = ids_key
        self.output_key = output_key
        self.padding_idx = padding_idx
        self.concept_embedding = torch.nn.Embedding(num_concept, embedding_dim)
        self.position_embedding = torch.nn.Embedding(seq_len, embedding_dim)
        self.w_1 = torch.nn.Parameter(torch.rand(embedding_dim, hidden_dim))
        self.w_2 = torch.nn.Parameter(torch.rand(hidden_dim, num_heads))
        self.w_3 = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim))
        self.w_k1 = torch.nn.Parameter(torch.rand(embedding_dim, hidden_dim))
        self.w_k2 = torch.nn.Parameter(torch.rand(hidden_dim, num_intention))
        self.w_4 = torch.nn.Parameter(torch.rand(embedding_dim, hidden_dim))
        self.w_5 = torch.nn.Parameter(torch.rand(hidden_dim, num_heads))

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        ids = ctx.get_required(self.ids_key).long()
        if x.dim() != 3 or x.size(1) != self.seq_len or x.size(-1) != self.embedding_dim or ids.shape != x.shape[:2]:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.seq_len}, {self.embedding_dim}]")
        mask = (ids != self.padding_idx).float()
        pos = torch.arange(self.seq_len, device=x.device).unsqueeze(0)
        x_u = x + self.position_embedding(pos)

        h_1 = torch.einsum("bse,ed->bsd", x_u, self.w_1).tanh()
        a_hist = F.softmax(torch.einsum("bsd,dh->bsh", h_1, self.w_2) - 1e9 * (1 - mask.unsqueeze(-1)), dim=1)
        z_u = torch.einsum("bse,bsh->be", x_u, a_hist)
        s_u = torch.einsum("be,te->bt", z_u, self.concept_embedding.weight)
        top_k = torch.topk(s_u, self.num_intention)
        c_u = torch.einsum("bk,bke->bke", torch.sigmoid(top_k.values), self.concept_embedding(top_k.indices))
        p_u = F.softmax(
            torch.einsum("bse,bke->bks", F.normalize(x_u @ self.w_3, p=2, dim=-1), F.normalize(c_u, p=2, dim=-1)),
            dim=1,
        )
        h_2 = torch.einsum("bse,ed->bsd", x_u, self.w_k1).tanh()
        a_concept = F.softmax(torch.einsum("bsd,dk->bsk", h_2, self.w_k2) - 1e9 * (1 - mask.unsqueeze(-1)), dim=1)
        phi_u = torch.einsum("bks,bse->bke", p_u * a_concept.permute(0, 2, 1), x_u)
        x_u_hat = torch.einsum("bks,bke->bse", p_u, c_u)
        h_3 = torch.einsum("bse,ed->bsd", x_u_hat, self.w_4).tanh()
        apt_scores = torch.einsum("bsd,dh->bsh", h_3, self.w_5).reshape(-1, self.seq_len) - 1e9 * (1 - mask)
        c_u_apt = F.normalize(torch.einsum("bs,bse->be", F.softmax(apt_scores, dim=1), x_u_hat), p=2, dim=-1)
        e_u = F.softmax(torch.einsum("be,bke->bk", c_u_apt, phi_u) / self.temperature, dim=1)
        return ctx.put(self.output_key, torch.einsum("bk,bke->be", e_u, phi_u))
