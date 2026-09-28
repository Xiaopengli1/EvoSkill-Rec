from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("candidate_dot_logits")
class CandidateDotLogitsSkill(BaseSkill):
    """Score one or more candidate item embeddings with a user embedding."""

    skill_spec = SkillSpec(
        name="candidate_dot_logits",
        category="matching",
        description="Compute dot-product logits for positive and sampled negative candidate items.",
        input_specs=[
            TensorSpec("user_embedding", "[batch_size, embedding_dim] or [batch_size, interest_num, embedding_dim]"),
            TensorSpec("candidate_embeddings", "[batch_size, num_candidates, embedding_dim]"),
        ],
        output_specs=[TensorSpec("scores", "[batch_size, num_candidates]")],
        task_types=["matching", "sequence"],
        inductive_bias=["retrieval_inner_product", "best_interest_selection"],
        failure_signatures=["embedding_dim_mismatch", "rank_mismatch"],
        hyperparameters={"temperature": "float", "squeeze_single": "bool"},
        paper_origin="YoutubeDNN / GRU4Rec / MIND / ComiRec",
    )

    def __init__(
        self,
        user_key: str = "user_embedding",
        candidate_key: str = "candidate_embeddings",
        output_key: str = "scores",
        temperature: float = 1.0,
        squeeze_single: bool = False,
        pos_output_key: str | None = None,
        neg_output_key: str | None = None,
    ) -> None:
        super().__init__()
        self.user_key = user_key
        self.candidate_key = candidate_key
        self.output_key = output_key
        self.temperature = temperature
        self.squeeze_single = squeeze_single
        self.pos_output_key = pos_output_key
        self.neg_output_key = neg_output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        user = ctx.get_required(self.user_key)
        candidates = ctx.get_required(self.candidate_key)
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(1)
        if candidates.dim() != 3:
            raise ValueError(f"{self.candidate_key} must have shape [batch_size, num_candidates, embedding_dim]")
        if user.dim() == 3 and user.size(1) == 1:
            user = user.squeeze(1)
        if user.dim() == 2:
            if user.size(-1) != candidates.size(-1):
                raise ValueError("user and candidate embedding dims must match")
            scores = torch.sum(user.unsqueeze(1) * candidates, dim=-1)
        elif user.dim() == 3:
            if user.size(-1) != candidates.size(-1):
                raise ValueError("user and candidate embedding dims must match")
            pos = candidates[:, 0, :]
            interest_scores = torch.sum(user * pos.unsqueeze(1), dim=-1)
            best_idx = torch.argmax(interest_scores, dim=1)
            batch_idx = torch.arange(user.size(0), device=user.device)
            selected_user = user[batch_idx, best_idx, :]
            scores = torch.sum(selected_user.unsqueeze(1) * candidates, dim=-1)
        else:
            raise ValueError(f"{self.user_key} must have rank 2 or 3")
        scores = scores / self.temperature
        if self.squeeze_single and scores.size(1) == 1:
            scores = scores.squeeze(1)
        ctx.put(self.output_key, scores)
        if self.pos_output_key is not None:
            ctx.put(self.pos_output_key, scores if scores.dim() == 1 else scores[:, 0])
        if self.neg_output_key is not None and scores.dim() == 2 and scores.size(1) > 1:
            ctx.put(self.neg_output_key, scores[:, 1:])
        return ctx
