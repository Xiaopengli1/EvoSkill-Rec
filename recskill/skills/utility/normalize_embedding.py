from __future__ import annotations

import torch.nn.functional as F

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("normalize_embedding")
class NormalizeEmbeddingSkill(BaseSkill):
    """L2-normalize dense embeddings along the last dimension."""

    skill_spec = SkillSpec(
        name="normalize_embedding",
        category="utility",
        description="Apply p-norm normalization to user or item embeddings along the last dimension.",
        input_specs=[TensorSpec("input_key", "[batch_size, ..., embedding_dim]")],
        output_specs=[TensorSpec("output_key", "[batch_size, ..., embedding_dim]")],
        task_types=["matching", "sequence", "ctr", "ranking"],
        inductive_bias=["cosine_similarity_space"],
        failure_signatures=["zero_norm_embedding", "rank_too_small"],
        hyperparameters={"input_key": "str", "output_key": "str", "p": "float"},
        implementation="torch.nn.functional.normalize",
    )

    def __init__(self, input_key: str, output_key: str, p: float = 2.0, eps: float = 1e-12) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.p = p
        self.eps = eps

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() < 2:
            raise ValueError(f"{self.input_key} must have at least rank 2")
        return ctx.put(self.output_key, F.normalize(x, p=self.p, dim=-1, eps=self.eps))
