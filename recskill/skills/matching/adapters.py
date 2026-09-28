from __future__ import annotations

import torch
import torch.nn.functional as F

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("mode_user_item_adapter")
class ModeUserItemAdapterSkill(BaseSkill):
    """Route matching model outputs according to Torch-RecHub user/item mode."""

    skill_spec = SkillSpec(
        name="mode_user_item_adapter",
        category="matching",
        description="Select user representation, item representation, or score path from two-tower outputs.",
        input_specs=[
            TensorSpec("user_embedding", "[batch_size, embedding_dim]"),
            TensorSpec("item_embedding", "[batch_size, embedding_dim]"),
        ],
        output_specs=[TensorSpec("mode_output", "[batch_size, embedding_dim]")],
        task_types=["matching"],
        inductive_bias=["two_tower_serving_modes"],
        failure_signatures=["unsupported_mode"],
        implementation="Torch-RecHub matching model mode=user/item adapter",
    )

    def __init__(
        self,
        mode: str | None = None,
        mode_key: str = "mode",
        user_key: str = "user_embedding",
        item_key: str = "item_embedding",
        output_key: str = "mode_output",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.mode_key = mode_key
        self.user_key = user_key
        self.item_key = item_key
        self.output_key = output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        mode = self.mode or ctx.get(self.mode_key)
        if mode == "user":
            return ctx.put(self.output_key, ctx.get_required(self.user_key))
        if mode == "item":
            return ctx.put(self.output_key, ctx.get_required(self.item_key))
        if mode in (None, "score", "both"):
            return ctx.put(self.output_key, (ctx.get_required(self.user_key), ctx.get_required(self.item_key)))
        raise ValueError(f"unsupported matching mode: {mode}")


@register_skill("all_item_scoring_adapter")
class AllItemScoringAdapterSkill(BaseSkill):
    """Score a query representation against every item embedding."""

    skill_spec = SkillSpec(
        name="all_item_scoring_adapter",
        category="matching",
        description="Compute full-catalog item scores for session/user representations.",
        input_specs=[
            TensorSpec("query_embedding", "[batch_size, embedding_dim]"),
            TensorSpec("item_embeddings", "[num_items, embedding_dim]"),
        ],
        output_specs=[TensorSpec("all_item_scores", "[batch_size, num_items]")],
        task_types=["matching", "sequence"],
        inductive_bias=["full_catalog_retrieval_scoring"],
        failure_signatures=["embedding_dim_mismatch"],
        implementation="NARM/STAMP all-item scoring adapter",
    )

    def __init__(
        self,
        query_key: str = "query_embedding",
        item_key: str = "item_embeddings",
        output_key: str = "all_item_scores",
        normalize: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.query_key = query_key
        self.item_key = item_key
        self.output_key = output_key
        self.normalize = normalize
        self.temperature = temperature

    def forward(self, ctx: SkillContext) -> SkillContext:
        query = ctx.get_required(self.query_key)
        items = ctx.get_required(self.item_key)
        if self.normalize:
            query = F.normalize(query, dim=-1)
            items = F.normalize(items, dim=-1)
        return ctx.put(self.output_key, torch.matmul(query, items.t()) / self.temperature)


@register_skill("in_batch_negative_head")
class InBatchNegativeHeadSkill(BaseSkill):
    """Build positive and in-batch negative retrieval logits."""

    skill_spec = SkillSpec(
        name="in_batch_negative_head",
        category="matching",
        description="Construct retrieval logits from query and item embeddings using other batch rows as negatives.",
        input_specs=[
            TensorSpec("query_embedding", "[batch_size, embedding_dim]"),
            TensorSpec("item_embedding", "[batch_size, embedding_dim]"),
        ],
        output_specs=[TensorSpec("sampled_logits", "[batch_size, batch_size]")],
        task_types=["matching", "sequence"],
        inductive_bias=["in_batch_negative_sampling"],
        failure_signatures=["batch_size_mismatch"],
        implementation="SASRec/YoutubeSBC in-batch negative head adapter",
    )

    def __init__(
        self,
        query_key: str = "query_embedding",
        item_key: str = "item_embedding",
        output_key: str = "sampled_logits",
        normalize: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.query_key = query_key
        self.item_key = item_key
        self.output_key = output_key
        self.normalize = normalize
        self.temperature = temperature

    def forward(self, ctx: SkillContext) -> SkillContext:
        query = ctx.get_required(self.query_key)
        item = ctx.get_required(self.item_key)
        if query.shape != item.shape:
            raise ValueError(f"{self.query_key} and {self.item_key} must have the same shape")
        if self.normalize:
            query = F.normalize(query, dim=-1)
            item = F.normalize(item, dim=-1)
        return ctx.put(self.output_key, torch.matmul(query, item.t()) / self.temperature)
