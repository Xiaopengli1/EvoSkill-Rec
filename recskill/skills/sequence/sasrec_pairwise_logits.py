from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("sasrec_pairwise_logits")
class SASRecPairwiseLogitsSkill(BaseSkill):
    """Compute SASRec positive and negative sequence logits."""

    skill_spec = SkillSpec(
        name="sasrec_pairwise_logits",
        category="head",
        description="Dot SASRec sequence outputs with positive and negative item embeddings.",
        input_specs=[
            TensorSpec("sequence_output", "[batch_size, seq_len, embedding_dim]"),
            TensorSpec("pos_embeddings", "[batch_size, seq_len, embedding_dim]"),
            TensorSpec("neg_embeddings", "[batch_size, seq_len, embedding_dim]"),
        ],
        output_specs=[
            TensorSpec("pos_logits", "[batch_size, seq_len]"),
            TensorSpec("neg_logits", "[batch_size, seq_len]"),
        ],
        task_types=["matching", "sequence"],
        inductive_bias=["pointwise_item_dot_product"],
        failure_signatures=["embedding_shape_mismatch"],
        implementation="torch.sum(sequence_output * item_embedding, dim=-1)",
    )

    def __init__(
        self,
        sequence_key: str = "sequence_output",
        pos_key: str = "pos_embeddings",
        neg_key: str = "neg_embeddings",
        pos_output_key: str = "pos_logits",
        neg_output_key: str = "neg_logits",
    ) -> None:
        super().__init__()
        self.sequence_key = sequence_key
        self.pos_key = pos_key
        self.neg_key = neg_key
        self.pos_output_key = pos_output_key
        self.neg_output_key = neg_output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        sequence_output = ctx.get_required(self.sequence_key)
        pos_embeddings = ctx.get_required(self.pos_key)
        neg_embeddings = ctx.get_required(self.neg_key)
        if sequence_output.shape != pos_embeddings.shape or sequence_output.shape != neg_embeddings.shape:
            raise ValueError("sequence, positive, and negative embeddings must have the same shape")
        ctx.put(self.pos_output_key, (sequence_output * pos_embeddings).sum(dim=-1))
        ctx.put(self.neg_output_key, (sequence_output * neg_embeddings).sum(dim=-1))
        return ctx
