from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import CapsuleNetwork, MultiInterestSA


@register_skill("multi_interest_extractor")
class MultiInterestExtractorSkill(BaseSkill):
    """Extract several user-interest vectors from a behavior sequence."""

    skill_spec = SkillSpec(
        name="multi_interest_extractor",
        category="sequence",
        description="Extract multiple interest vectors with self-attention or capsule routing.",
        input_specs=[TensorSpec("seq_embeddings", "[batch_size, seq_len, embedding_dim]")],
        output_specs=[TensorSpec("multi_interest_embeddings", "[batch_size, interest_num, embedding_dim]")],
        task_types=["matching", "sequence"],
        inductive_bias=["multi_interest_user_modeling"],
        failure_signatures=["seq_len_mismatch", "mask_shape_mismatch"],
        hyperparameters={"mode": "self_attention|capsule", "interest_num": "int"},
        paper_origin="MIND / ComiRec",
    )

    def __init__(
        self,
        embedding_dim: int,
        seq_len: int,
        interest_num: int = 4,
        mode: str = "self_attention",
        input_key: str = "seq_embeddings",
        ids_key: str | None = "seq_features",
        output_key: str = "multi_interest_embeddings",
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if mode not in {"self_attention", "capsule", "mind_capsule"}:
            raise ValueError("mode must be self_attention, capsule, or mind_capsule")
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.mode = mode
        self.input_key = input_key
        self.ids_key = ids_key
        self.output_key = output_key
        self.padding_idx = padding_idx
        if mode == "self_attention":
            self.extractor = MultiInterestSA(embedding_dim, interest_num=interest_num)
        else:
            bilinear_type = 0 if mode == "mind_capsule" else 2
            self.extractor = CapsuleNetwork(embedding_dim, seq_len=seq_len, bilinear_type=bilinear_type, interest_num=interest_num)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(1) != self.seq_len or x.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.seq_len}, {self.embedding_dim}]")
        mask = None
        if self.ids_key is not None and self.ids_key in ctx:
            ids = ctx.get_required(self.ids_key).long()
            if ids.shape != x.shape[:2]:
                raise ValueError(f"{self.ids_key} must match sequence shape")
            mask = ids != self.padding_idx
        if self.mode == "self_attention":
            sa_mask = None if mask is None else mask.unsqueeze(-1)
            out = self.extractor(x, sa_mask)
        else:
            cap_mask = x.new_ones(x.shape[:2]) if mask is None else mask.float()
            out = self.extractor(x, cap_mask)
        return ctx.put(self.output_key, out)
