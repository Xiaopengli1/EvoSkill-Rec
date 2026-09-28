from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import HSTUBlock


@register_skill("hstu_sequence_encoder")
class HSTUSequenceEncoderSkill(BaseSkill):
    """Encode token embeddings with an HSTU block."""

    skill_spec = SkillSpec(
        name="hstu_sequence_encoder",
        category="sequence",
        description="Apply stacked HSTU layers to sequential item embeddings.",
        input_specs=[TensorSpec("seq_embeddings", "[batch_size, seq_len, d_model]")],
        output_specs=[TensorSpec("sequence_output", "[batch_size, seq_len, d_model]")],
        task_types=["generative", "sequence", "matching"],
        inductive_bias=["causal_sequential_transduction"],
        failure_signatures=["d_model_head_mismatch", "seq_len_exceeds_max"],
        hyperparameters={"d_model": "int", "n_heads": "int", "n_layers": "int"},
        paper_origin="HSTU",
        implementation="torch_rechub.basic.layers.HSTUBlock",
    )

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        n_layers: int = 2,
        dqk: int | None = None,
        dv: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        input_key: str = "seq_embeddings",
        ids_key: str | None = "seq_features",
        time_diffs_key: str | None = "time_diffs",
        output_key: str = "sequence_output",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.input_key = input_key
        self.ids_key = ids_key
        self.time_diffs_key = time_diffs_key
        self.output_key = output_key
        self.padding_idx = padding_idx
        self.encoder = HSTUBlock(d_model=d_model, n_heads=n_heads, n_layers=n_layers, dqk=dqk or d_model, dv=dv or d_model, dropout=dropout, max_seq_len=max_seq_len)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(-1) != self.d_model or x.size(1) > self.max_seq_len:
            raise ValueError(f"{self.input_key} must have shape [batch_size, seq_len <= {self.max_seq_len}, {self.d_model}]")
        padding_mask = None
        if self.ids_key is not None and self.ids_key in ctx:
            padding_mask = ctx.get_required(self.ids_key).long() != self.padding_idx
        time_diffs = ctx.get(self.time_diffs_key) if self.time_diffs_key is not None else None
        return ctx.put(self.output_key, self.encoder(x, padding_mask=padding_mask, time_diffs=time_diffs))
