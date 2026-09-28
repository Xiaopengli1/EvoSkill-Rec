from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("gru_sequence_encoder")
class GRUSequenceEncoderSkill(BaseSkill):
    """Encode sequence embeddings with a GRU and expose sequence and summary states."""

    skill_spec = SkillSpec(
        name="gru_sequence_encoder",
        category="sequence",
        description="Encode item history embeddings with a GRU sequence encoder.",
        input_specs=[TensorSpec("seq_embeddings", "[batch_size, seq_len, embedding_dim]")],
        output_specs=[
            TensorSpec("sequence_output", "[batch_size, seq_len, hidden_dim]"),
            TensorSpec("sequence_summary", "[batch_size, hidden_dim]"),
        ],
        task_types=["matching", "ranking", "sequence"],
        inductive_bias=["recurrent_interest_evolution"],
        failure_signatures=["embedding_dim_mismatch"],
        hyperparameters={"embedding_dim": "int", "hidden_dim": "int", "num_layers": "int"},
        paper_origin="GRU4Rec / DIEN / NARM",
    )

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        input_key: str = "seq_embeddings",
        output_key: str = "sequence_output",
        summary_key: str = "sequence_summary",
        bidirectional: bool = False,
        num_layers: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_key = input_key
        self.output_key = output_key
        self.summary_key = summary_key
        self.gru = torch.nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            bias=bias,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, seq_len, {self.embedding_dim}]")
        out, hidden = self.gru(x)
        if self.gru.bidirectional:
            summary = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            summary = hidden[-1]
        ctx.put(self.output_key, out)
        return ctx.put(self.summary_key, summary)
