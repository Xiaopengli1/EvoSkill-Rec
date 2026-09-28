from __future__ import annotations

import torch
import torch.nn.functional as F

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("tied_embedding_lm_head")
class TiedEmbeddingLMHeadSkill(BaseSkill):
    """Project hidden states with a tied item/token embedding matrix."""

    skill_spec = SkillSpec(
        name="tied_embedding_lm_head",
        category="head",
        description="Use token embedding weights as the output projection, matching HSTU tied embedding scoring.",
        input_specs=[TensorSpec("sequence_output", "[batch_size, seq_len, d_model]")],
        output_specs=[TensorSpec("logits", "[batch_size, seq_len, vocab_size]")],
        task_types=["generative", "sequence"],
        inductive_bias=["tied_input_output_embeddings"],
        failure_signatures=["embedding_dim_mismatch"],
        hyperparameters={"vocab_size": "int", "d_model": "int"},
        implementation="torch_rechub.models.generative.hstu tied output projection",
    )

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        input_key: str = "sequence_output",
        embedding_weight_key: str | None = "token_embedding_weight",
        output_key: str = "logits",
        use_output_bias: bool = True,
        score_norm: str = "none",
        temperature: float = 1.0,
        l2_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if score_norm not in ("none", "l2"):
            raise ValueError("score_norm must be 'none' or 'l2'")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.input_key = input_key
        self.embedding_weight_key = embedding_weight_key
        self.output_key = output_key
        self.score_norm = score_norm
        self.temperature = temperature
        self.l2_norm_eps = l2_norm_eps
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.output_bias = torch.nn.Parameter(torch.zeros(vocab_size)) if use_output_bias else None

    def forward(self, ctx: SkillContext) -> SkillContext:
        hidden = ctx.get_required(self.input_key)
        weight = ctx.get(self.embedding_weight_key) if self.embedding_weight_key is not None else None
        if weight is None:
            weight = self.token_embedding.weight
        if hidden.size(-1) != self.d_model or weight.shape != (self.vocab_size, self.d_model):
            raise ValueError("hidden state and tied embedding weight dimensions do not match")
        if self.score_norm == "l2":
            hidden = F.normalize(hidden, p=2, dim=-1, eps=self.l2_norm_eps)
            weight = F.normalize(weight, p=2, dim=-1, eps=self.l2_norm_eps)
        logits = F.linear(hidden, weight, self.output_bias)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return ctx.put(self.output_key, logits)
