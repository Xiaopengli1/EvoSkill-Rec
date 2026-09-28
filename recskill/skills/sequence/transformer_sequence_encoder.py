from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("transformer_sequence_encoder")
class TransformerSequenceEncoderSkill(BaseSkill):
    """Lightweight causal Transformer encoder for sequence recommendation genomes."""

    skill_spec = SkillSpec(
        name="transformer_sequence_encoder",
        category="sequence",
        description="Encode item sequences with a causal Transformer encoder stack.",
        input_specs=[TensorSpec("seq_embeddings", "[batch_size, seq_len, d_model]")],
        output_specs=[TensorSpec("sequence_output", "[batch_size, seq_len, d_model]")],
        task_types=["generative", "sequence", "matching"],
        inductive_bias=["causal_self_attention"],
        failure_signatures=["d_model_head_mismatch", "seq_len_exceeds_max"],
        hyperparameters={"d_model": "int", "n_heads": "int", "n_layers": "int"},
        paper_origin="Transformer sequence recommendation / HLLM / TIGER adapter",
    )

    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        input_key: str = "seq_embeddings",
        ids_key: str | None = "seq_features",
        output_key: str = "sequence_output",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.input_key = input_key
        self.ids_key = ids_key
        self.output_key = output_key
        self.padding_idx = padding_idx
        self.position = torch.nn.Embedding(max_seq_len, d_model)
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=max(4 * d_model, 16), dropout=dropout, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(-1) != self.d_model or x.size(1) > self.max_seq_len:
            raise ValueError(f"{self.input_key} must have shape [batch_size, seq_len <= {self.max_seq_len}, {self.d_model}]")
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.position(pos)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        key_padding_mask = None
        if self.ids_key is not None and self.ids_key in ctx:
            key_padding_mask = ctx.get_required(self.ids_key).long() == self.padding_idx
        return ctx.put(self.output_key, self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask))
