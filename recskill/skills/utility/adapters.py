from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("named_feature_dict_adapter")
class NamedFeatureDictAdapterSkill(BaseSkill):
    """Convert Torch-RecHub named feature dictionaries into ordered tensors."""

    skill_spec = SkillSpec(
        name="named_feature_dict_adapter",
        category="utility",
        description="Extract named feature tensors from a dict/context and stack them in a stable field order.",
        input_specs=[TensorSpec("features", "dict[str, tensor]")],
        output_specs=[TensorSpec("field_ids", "[batch_size, num_fields]")],
        task_types=["ctr", "ranking", "matching", "multitask", "sequence", "generative"],
        inductive_bias=["torch_rechub_named_feature_contract"],
        failure_signatures=["missing_feature_name", "feature_shape_mismatch"],
        hyperparameters={"feature_names": "list[str]"},
        implementation="torch_rechub model forward named-feature input adapter",
    )

    def __init__(
        self,
        feature_names: Sequence[str],
        input_key: str = "features",
        output_key: str = "field_ids",
        stack_dim: int = 1,
        dtype: str = "long",
        squeeze_last: bool = True,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.input_key = input_key
        self.output_key = output_key
        self.stack_dim = stack_dim
        self.dtype = dtype
        self.squeeze_last = squeeze_last

    def forward(self, ctx: SkillContext) -> SkillContext:
        source = ctx.get_required(self.input_key) if self.input_key in ctx else ctx
        if not isinstance(source, Mapping):
            raise TypeError(f"{self.input_key} must be a mapping of feature names to tensors")
        values = []
        for name in self.feature_names:
            if name not in source:
                available = ", ".join(sorted(str(key) for key in source.keys()))
                raise KeyError(f"missing feature '{name}'. Available features: {available}")
            value = source[name]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            if self.squeeze_last and value.dim() > 1 and value.size(-1) == 1:
                value = value.squeeze(-1)
            values.append(value)
        out = torch.stack(values, dim=self.stack_dim)
        if self.dtype == "long":
            out = out.long()
        elif self.dtype == "float":
            out = out.float()
        return ctx.put(self.output_key, out)


@register_skill("dense_feature_path")
class DenseFeaturePathSkill(BaseSkill):
    """Embed dense scalar fields and optionally append them to field embeddings."""

    skill_spec = SkillSpec(
        name="dense_feature_path",
        category="utility",
        description="Project dense numeric fields into embedding slots compatible with sparse field embeddings.",
        input_specs=[TensorSpec("dense_values", "[batch_size, num_dense]")],
        output_specs=[TensorSpec("dense_embeddings", "[batch_size, num_dense, embedding_dim]")],
        task_types=["ctr", "ranking", "multitask"],
        inductive_bias=["dense_feature_adapter"],
        failure_signatures=["dense_feature_dim_mismatch"],
        hyperparameters={"num_dense": "int", "embedding_dim": "int"},
        implementation="AutoInt/BST dense feature branch adapter",
    )

    def __init__(
        self,
        num_dense: int,
        embedding_dim: int,
        input_key: str = "dense_values",
        output_key: str = "dense_embeddings",
        field_embeddings_key: str | None = "field_embeddings",
        merged_output_key: str = "field_embeddings",
    ) -> None:
        super().__init__()
        self.num_dense = num_dense
        self.embedding_dim = embedding_dim
        self.input_key = input_key
        self.output_key = output_key
        self.field_embeddings_key = field_embeddings_key
        self.merged_output_key = merged_output_key
        self.projections = torch.nn.ModuleList([torch.nn.Linear(1, embedding_dim) for _ in range(num_dense)])

    def forward(self, ctx: SkillContext) -> SkillContext:
        dense = ctx.get_required(self.input_key).float()
        if dense.dim() != 2 or dense.size(1) != self.num_dense:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.num_dense}]")
        embeds = [proj(dense[:, idx:idx + 1]).unsqueeze(1) for idx, proj in enumerate(self.projections)]
        dense_embeddings = torch.cat(embeds, dim=1)
        ctx.put(self.output_key, dense_embeddings)
        if self.field_embeddings_key is not None and self.field_embeddings_key in ctx:
            fields = ctx.get_required(self.field_embeddings_key)
            if fields.dim() != 3 or fields.size(-1) != self.embedding_dim:
                raise ValueError(f"{self.field_embeddings_key} must have shape [batch_size, num_fields, {self.embedding_dim}]")
            ctx.put(self.merged_output_key, torch.cat([fields, dense_embeddings], dim=1))
        return ctx


@register_skill("field_aware_embedding_adapter")
class FieldAwareEmbeddingAdapterSkill(BaseSkill):
    """Create FFM-style field-aware embedding tensors."""

    skill_spec = SkillSpec(
        name="field_aware_embedding_adapter",
        category="embedding",
        description="Map field ids to [source_field, target_field] embeddings for FFM interactions.",
        input_specs=[TensorSpec("field_ids", "[batch_size, num_fields]")],
        output_specs=[TensorSpec("field_aware_embeddings", "[batch_size, num_fields, num_fields, embedding_dim]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["field_aware_factorization"],
        failure_signatures=["field_count_mismatch", "vocab_size_mismatch"],
        hyperparameters={"vocab_sizes": "list[int]", "num_fields": "int", "embedding_dim": "int"},
        implementation="torch_rechub.basic.layers.EmbeddingLayer field-aware FFM adapter",
    )

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        num_fields: int,
        embedding_dim: int,
        input_key: str = "field_ids",
        output_key: str = "field_aware_embeddings",
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if len(vocab_sizes) != num_fields:
            raise ValueError("vocab_sizes length must equal num_fields")
        self.vocab_sizes = list(vocab_sizes)
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.input_key = input_key
        self.output_key = output_key
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(vocab_size, num_fields * embedding_dim, padding_idx=padding_idx) for vocab_size in self.vocab_sizes]
        )

    def forward(self, ctx: SkillContext) -> SkillContext:
        ids = ctx.get_required(self.input_key).long()
        if ids.dim() != 2 or ids.size(1) != self.num_fields:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.num_fields}]")
        fields = []
        for idx, embedding in enumerate(self.embeddings):
            fields.append(embedding(ids[:, idx]).view(ids.size(0), self.num_fields, self.embedding_dim).unsqueeze(1))
        return ctx.put(self.output_key, torch.cat(fields, dim=1))
