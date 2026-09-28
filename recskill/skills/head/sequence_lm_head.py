from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("sequence_lm_head")
class SequenceLMHeadSkill(BaseSkill):
    """Project sequence hidden states to vocabulary logits."""

    skill_spec = SkillSpec(
        name="sequence_lm_head",
        category="head",
        description="Produce item or semantic-token logits from sequence hidden states.",
        input_specs=[TensorSpec("sequence_output", "[batch_size, seq_len, hidden_dim]")],
        output_specs=[TensorSpec("logits", "[batch_size, seq_len, vocab_size] or [batch_size, vocab_size]")],
        task_types=["generative", "sequence", "matching"],
        inductive_bias=["next_item_language_modeling"],
        failure_signatures=["hidden_dim_mismatch"],
        hyperparameters={"hidden_dim": "int", "vocab_size": "int", "mode": "all|last"},
    )

    def __init__(self, hidden_dim: int, vocab_size: int, input_key: str = "sequence_output", output_key: str = "logits", mode: str = "all") -> None:
        super().__init__()
        if mode not in {"all", "last"}:
            raise ValueError("mode must be all or last")
        self.hidden_dim = hidden_dim
        self.input_key = input_key
        self.output_key = output_key
        self.mode = mode
        self.proj = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(-1) != self.hidden_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, seq_len, {self.hidden_dim}]")
        logits = self.proj(x)
        if self.mode == "last":
            logits = logits[:, -1, :]
        return ctx.put(self.output_key, logits)
