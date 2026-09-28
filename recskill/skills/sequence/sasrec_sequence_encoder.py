from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.models.matching.sasrec import PointWiseFeedForward


@register_skill("sasrec_sequence_encoder")
class SASRecSequenceEncoderSkill(BaseSkill):
    """SASRec self-attention sequence encoder.

    This mirrors the encoder stack from ``torch_rechub.models.matching.sasrec``
    while keeping tensors device-aware for genome tests.
    """

    skill_spec = SkillSpec(
        name="sasrec_sequence_encoder",
        category="sequence",
        description="Encode an item history sequence with SASRec causal self-attention.",
        input_specs=[
            TensorSpec("seq_features", "[batch_size, seq_len]", "int64"),
            TensorSpec("seq_embeddings", "[batch_size, seq_len, embedding_dim]"),
        ],
        output_specs=[TensorSpec("sequence_output", "[batch_size, seq_len, embedding_dim]")],
        task_types=["matching", "sequence"],
        inductive_bias=["causal_self_attention", "position_encoding"],
        failure_signatures=["seq_len_exceeds_max_len", "embedding_dim_mismatch"],
        hyperparameters={"embedding_dim": "int", "max_len": "int", "num_blocks": "int", "num_heads": "int"},
        paper_origin="Self-Attentive Sequential Recommendation",
        implementation="torch_rechub.models.matching.sasrec.PointWiseFeedForward and torch.nn.MultiheadAttention",
    )

    def __init__(
        self,
        embedding_dim: int,
        max_len: int = 50,
        dropout_rate: float = 0.5,
        num_blocks: int = 2,
        num_heads: int = 1,
        ids_key: str = "seq_features",
        input_key: str = "seq_embeddings",
        output_key: str = "sequence_output",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.ids_key = ids_key
        self.input_key = input_key
        self.output_key = output_key
        self.padding_idx = padding_idx
        self.position_emb = torch.nn.Embedding(max_len, embedding_dim)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList([torch.nn.LayerNorm(embedding_dim, eps=1e-8) for _ in range(num_blocks)])
        self.attention_layers = torch.nn.ModuleList([torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.forward_layernorms = torch.nn.ModuleList([torch.nn.LayerNorm(embedding_dim, eps=1e-8) for _ in range(num_blocks)])
        self.forward_layers = torch.nn.ModuleList([PointWiseFeedForward(embedding_dim, dropout_rate) for _ in range(num_blocks)])
        self.last_layernorm = torch.nn.LayerNorm(embedding_dim, eps=1e-8)

    def forward(self, ctx: SkillContext) -> SkillContext:
        seq_ids = ctx.get_required(self.ids_key).long()
        seq_embeddings = ctx.get_required(self.input_key)
        if seq_ids.dim() != 2:
            raise ValueError(f"{self.ids_key} must have shape [batch_size, seq_len]")
        if seq_embeddings.dim() != 3:
            raise ValueError(f"{self.input_key} must have shape [batch_size, seq_len, embedding_dim]")
        batch_size, seq_len = seq_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_len {self.max_len}")
        if seq_embeddings.shape[:2] != (batch_size, seq_len) or seq_embeddings.size(-1) != self.embedding_dim:
            raise ValueError("sequence embedding shape must match ids and embedding_dim")

        x = seq_embeddings * (self.embedding_dim ** 0.5)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_emb(positions)
        x = self.emb_dropout(x)

        timeline_mask = seq_ids == self.padding_idx
        x = x * (~timeline_mask).unsqueeze(-1)
        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))

        for attention_layernorm, attention_layer, forward_layernorm, forward_layer in zip(
            self.attention_layernorms,
            self.attention_layers,
            self.forward_layernorms,
            self.forward_layers,
        ):
            x_t = torch.transpose(x, 0, 1)
            q = attention_layernorm(x_t)
            mha_outputs, _ = attention_layer(q, x_t, x_t, attn_mask=attention_mask)
            x = torch.transpose(q + mha_outputs, 0, 1)
            x = forward_layernorm(x)
            x = forward_layer(x)
            x = x * (~timeline_mask).unsqueeze(-1)

        return ctx.put(self.output_key, self.last_layernorm(x))
