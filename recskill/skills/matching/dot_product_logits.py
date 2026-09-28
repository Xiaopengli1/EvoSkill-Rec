from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("dot_product_logits")
class DotProductLogitsSkill(BaseSkill):
    """Compute dot-product retrieval logits for user and item embeddings."""

    skill_spec = SkillSpec(
        name="dot_product_logits",
        category="matching",
        description="Score positive and optional negative items with dot-product retrieval logits.",
        input_specs=[
            TensorSpec("user_embedding", "[batch_size, embedding_dim] or [batch_size, k, embedding_dim]"),
            TensorSpec("item_embedding", "[batch_size, embedding_dim]"),
        ],
        output_specs=[TensorSpec("pos_logits", "[batch_size]")],
        task_types=["matching", "sequence"],
        inductive_bias=["two_tower_inner_product"],
        failure_signatures=["embedding_dim_mismatch"],
        hyperparameters={"temperature": "float"},
        paper_origin="DSSM / YoutubeDNN / sequence matching",
    )

    def __init__(
        self,
        user_key: str = "user_embedding",
        item_key: str = "item_embedding",
        neg_item_key: str | None = None,
        pos_output_key: str = "pos_logits",
        neg_output_key: str = "neg_logits",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.user_key = user_key
        self.item_key = item_key
        self.neg_item_key = neg_item_key
        self.pos_output_key = pos_output_key
        self.neg_output_key = neg_output_key
        self.temperature = temperature

    def forward(self, ctx: SkillContext) -> SkillContext:
        user = ctx.get_required(self.user_key)
        item = ctx.get_required(self.item_key)
        ctx.put(self.pos_output_key, self._score(user, item) / self.temperature)
        if self.neg_item_key is not None and self.neg_item_key in ctx:
            ctx.put(self.neg_output_key, self._score(user, ctx.get_required(self.neg_item_key)) / self.temperature)
        return ctx

    @staticmethod
    def _score(user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        if user.dim() == 2 and item.dim() == 2:
            if user.size(-1) != item.size(-1):
                raise ValueError("user and item embedding dims must match")
            return torch.sum(user * item, dim=-1)
        if user.dim() == 3 and item.dim() == 2:
            if user.size(-1) != item.size(-1):
                raise ValueError("user and item embedding dims must match")
            return torch.max(torch.sum(user * item.unsqueeze(1), dim=-1), dim=1).values
        if user.dim() == 2 and item.dim() == 3:
            if user.size(-1) != item.size(-1):
                raise ValueError("user and item embedding dims must match")
            return torch.max(torch.sum(user.unsqueeze(1) * item, dim=-1), dim=1).values
        raise ValueError("dot_product_logits supports [B,D] or one side as [B,K,D]")
