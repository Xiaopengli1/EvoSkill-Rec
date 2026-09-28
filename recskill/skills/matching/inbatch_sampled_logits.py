from __future__ import annotations

import torch
import torch.nn.functional as F

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("inbatch_sampled_logits")
class InBatchSampledLogitsSkill(BaseSkill):
    """Build YoutubeSBC-style in-batch sampled logits with sampling-bias correction."""

    skill_spec = SkillSpec(
        name="inbatch_sampled_logits",
        category="matching",
        description="Score in-batch positive and wrapped negative items, then subtract log sampling weights.",
        input_specs=[
            TensorSpec("user_embedding", "[batch_size, embedding_dim]"),
            TensorSpec("item_embedding", "[batch_size, embedding_dim]"),
            TensorSpec("sample_weight", "[batch_size]"),
        ],
        output_specs=[TensorSpec("scores", "[batch_size, 1 + n_neg]")],
        task_types=["matching"],
        inductive_bias=["in_batch_negatives", "sampling_bias_correction"],
        failure_signatures=["batch_size_mismatch", "non_positive_sample_weight"],
        hyperparameters={"n_neg": "int", "temperature": "float"},
        paper_origin="Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations",
    )

    def __init__(
        self,
        user_key: str = "user_embedding",
        item_key: str = "item_embedding",
        sample_weight_key: str = "sample_weight",
        output_key: str = "scores",
        n_neg: int = 3,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.user_key = user_key
        self.item_key = item_key
        self.sample_weight_key = sample_weight_key
        self.output_key = output_key
        self.n_neg = n_neg
        self.temperature = temperature

    def forward(self, ctx: SkillContext) -> SkillContext:
        user = ctx.get_required(self.user_key)
        item = ctx.get_required(self.item_key)
        sample_weight = ctx.get_required(self.sample_weight_key).float()
        if user.dim() != 2 or item.dim() != 2 or user.shape != item.shape:
            raise ValueError("user and item embeddings must have the same [batch_size, embedding_dim] shape")
        if sample_weight.dim() > 1:
            sample_weight = sample_weight.reshape(sample_weight.size(0), -1)[:, 0]
        if sample_weight.size(0) != user.size(0):
            raise ValueError("sample_weight batch size must match embeddings")
        logits = F.cosine_similarity(user.unsqueeze(1), item.unsqueeze(0), dim=-1)
        logits = logits - torch.log(sample_weight.clamp_min(1e-12)).unsqueeze(0)
        batch_size = user.size(0)
        row_indices = torch.arange(batch_size, device=user.device).repeat_interleave(self.n_neg + 1)
        offsets = torch.arange(self.n_neg + 1, device=user.device).repeat(batch_size)
        col_indices = (row_indices + offsets) % batch_size
        sampled = logits[row_indices, col_indices].view(batch_size, self.n_neg + 1)
        return ctx.put(self.output_key, sampled / self.temperature)
