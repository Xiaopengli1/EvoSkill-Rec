from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("sequence_field_embedding")
class SequenceFieldEmbeddingSkill(BaseSkill):
    """Embed sequence feature fields into ``[B, F, L, D]`` tensors.

    Source models: DIN embeds history ``SequenceFeature`` fields with
    ``EmbeddingLayer(..., pooling='concat')`` before target attention.
    """

    skill_spec = SkillSpec(
        name="sequence_field_embedding",
        category="embedding",
        description="Embed integer sequence feature fields into dense sequence tensors.",
        input_specs=[TensorSpec("sequence_features", "[batch_size, num_fields, seq_len]", "int64")],
        output_specs=[TensorSpec("sequence_embeddings", "[batch_size, num_fields, seq_len, embedding_dim]")],
        task_types=["ctr", "ranking", "sequence"],
        inductive_bias=["categorical_sequence_lookup"],
        failure_signatures=["out_of_range_ids", "feature_count_mismatch"],
        hyperparameters={"vocab_sizes": "list[int]", "embedding_dim": "int"},
        implementation="torch.nn.Embedding",
    )

    def __init__(
        self,
        vocab_sizes,
        embedding_dim: int,
        input_key: str = "sequence_features",
        output_key: str = "sequence_embeddings",
        padding_idx: int | None = 0,
    ) -> None:
        super().__init__()
        if isinstance(vocab_sizes, int):
            vocab_sizes = [vocab_sizes]
        self.vocab_sizes = list(vocab_sizes)
        self.embedding_dim = embedding_dim
        self.input_key = input_key
        self.output_key = output_key
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx) for vocab_size in self.vocab_sizes]
        )

    def forward(self, ctx: SkillContext) -> SkillContext:
        sequence_features = ctx.get_required(self.input_key).long()
        if sequence_features.dim() == 2 and len(self.embeddings) == 1:
            sequence_features = sequence_features.unsqueeze(1)
        if sequence_features.dim() != 3:
            raise ValueError(f"{self.input_key} must have shape [batch_size, num_fields, seq_len]")
        if sequence_features.size(1) != len(self.embeddings):
            raise ValueError(f"Expected {len(self.embeddings)} sequence fields, got {sequence_features.size(1)}")
        embedded = [embedding(sequence_features[:, idx, :]).unsqueeze(1) for idx, embedding in enumerate(self.embeddings)]
        return ctx.put(self.output_key, torch.cat(embedded, dim=1))
