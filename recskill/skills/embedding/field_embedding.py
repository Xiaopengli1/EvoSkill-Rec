from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.features import SparseFeature
from torch_rechub.basic.layers import EmbeddingLayer


@register_skill("field_embedding")
class FieldEmbeddingSkill(BaseSkill):
    """Wraps torch_rechub.basic.layers.EmbeddingLayer for tensor field ids.

    Source model: torch_rechub.models.ranking.deepfm.DeepFM uses EmbeddingLayer
    to produce ``[batch_size, num_fields, embedding_dim]`` sparse embeddings for
    the FM branch.
    """

    skill_spec = SkillSpec(
        name="field_embedding",
        category="embedding",
        description="Embed integer sparse feature fields into a dense field embedding tensor.",
        input_specs=[TensorSpec("sparse_features", "[batch_size, num_fields]", "int64")],
        output_specs=[TensorSpec("field_embeddings", "[batch_size, num_fields, embedding_dim]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["categorical_lookup", "shared_embedding_dimension"],
        failure_signatures=["out_of_range_ids", "feature_count_mismatch"],
        hyperparameters={"vocab_sizes": "list[int]", "embedding_dim": "int"},
        implementation="torch_rechub.basic.layers.EmbeddingLayer",
    )

    def __init__(
        self,
        vocab_sizes,
        embedding_dim: int,
        input_key: str = "sparse_features",
        output_key: str = "field_embeddings",
        feature_names: list[str] | None = None,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(vocab_sizes, int):
            vocab_sizes = [vocab_sizes]
        self.vocab_sizes = list(vocab_sizes)
        self.embedding_dim = embedding_dim
        self.input_key = input_key
        self.output_key = output_key
        if feature_names is None:
            feature_names = [f"field_{idx}" for idx in range(len(self.vocab_sizes))]
        if len(feature_names) != len(self.vocab_sizes):
            raise ValueError("feature_names length must match vocab_sizes length")
        self.feature_names = list(feature_names)
        self.features = [
            SparseFeature(name, vocab_size=vocab_size, embed_dim=embedding_dim, padding_idx=padding_idx)
            for name, vocab_size in zip(self.feature_names, self.vocab_sizes)
        ]
        self.embedding = EmbeddingLayer(self.features)

    def forward(self, ctx: SkillContext) -> SkillContext:
        sparse_features = ctx.get_required(self.input_key).long()
        if sparse_features.dim() == 1 and len(self.features) == 1:
            sparse_features = sparse_features.unsqueeze(1)
        if sparse_features.dim() != 2:
            raise ValueError(f"{self.input_key} must have shape [batch_size, num_fields]")
        if sparse_features.size(1) != len(self.features):
            raise ValueError(f"Expected {len(self.features)} fields, got {sparse_features.size(1)}")
        feature_dict = {
            name: sparse_features[:, idx]
            for idx, name in enumerate(self.feature_names)
        }
        return ctx.put(self.output_key, self.embedding(feature_dict, self.features, squeeze_dim=False))
