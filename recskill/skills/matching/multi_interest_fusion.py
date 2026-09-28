from __future__ import annotations

import torch
import torch.nn.functional as F

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("multi_interest_fusion")
class MultiInterestFusionSkill(BaseSkill):
    """Fuse static user context with extracted multi-interest vectors."""

    skill_spec = SkillSpec(
        name="multi_interest_fusion",
        category="matching",
        description="Concatenate repeated user-context features with each interest vector and project to retrieval embeddings.",
        input_specs=[
            TensorSpec("user_context", "[batch_size, user_dim]"),
            TensorSpec("multi_interest_embeddings", "[batch_size, interest_num, embedding_dim]"),
        ],
        output_specs=[TensorSpec("user_embedding", "[batch_size, interest_num, output_dim]")],
        task_types=["matching", "sequence"],
        inductive_bias=["multi_interest_user_context_fusion"],
        failure_signatures=["input_dim_mismatch", "interest_count_mismatch"],
        hyperparameters={"user_dim": "int", "embedding_dim": "int", "output_dim": "int"},
        paper_origin="MIND / ComiRec",
    )

    def __init__(
        self,
        user_dim: int,
        embedding_dim: int,
        output_dim: int,
        interest_num: int,
        user_key: str = "user_context",
        interest_key: str = "multi_interest_embeddings",
        output_key: str = "user_embedding",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.user_dim = user_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.interest_num = interest_num
        self.user_key = user_key
        self.interest_key = interest_key
        self.output_key = output_key
        self.normalize = normalize
        self.proj = torch.nn.Linear(user_dim + embedding_dim, output_dim, bias=False)

    def forward(self, ctx: SkillContext) -> SkillContext:
        user = ctx.get_required(self.user_key)
        interests = ctx.get_required(self.interest_key)
        if user.dim() != 2 or user.size(-1) != self.user_dim:
            raise ValueError(f"{self.user_key} must have shape [batch_size, {self.user_dim}]")
        if interests.dim() != 3 or interests.size(1) != self.interest_num or interests.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.interest_key} must have shape [batch_size, {self.interest_num}, {self.embedding_dim}]")
        expanded_user = user.unsqueeze(1).expand(-1, self.interest_num, -1)
        out = self.proj(torch.cat([expanded_user, interests], dim=-1))
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return ctx.put(self.output_key, out)
