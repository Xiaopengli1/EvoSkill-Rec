from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("shared_sequence_embedding")
class SharedSequenceEmbeddingSkill(BaseSkill):
    """Embed multiple sequence id tensors with one shared item table.

    Source model: SASRec uses shared item embeddings for the historical,
    positive, and negative sequence features.
    """

    skill_spec = SkillSpec(
        name="shared_sequence_embedding",
        category="embedding",
        description="Embed several sequence id tensors with a shared embedding table.",
        input_specs=[TensorSpec("input_keys", "list[[batch_size, seq_len]]", "int64")],
        output_specs=[TensorSpec("output_keys", "list[[batch_size, seq_len, embedding_dim]]")],
        task_types=["generative", "matching", "sequence"],
        inductive_bias=["shared_item_embedding_space"],
        failure_signatures=["out_of_range_ids", "seq_len_mismatch"],
        hyperparameters={"vocab_size": "int", "embedding_dim": "int", "input_keys": "list[str]", "output_keys": "list[str]"},
        implementation="torch.nn.Embedding",
    )

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        input_keys: list[str],
        output_keys: list[str] | None = None,
        padding_idx: int | None = 0,
    ) -> None:
        super().__init__()
        if not input_keys:
            raise ValueError("input_keys must not be empty")
        if output_keys is None:
            output_keys = [f"{key}_embedding" for key in input_keys]
        if len(output_keys) != len(input_keys):
            raise ValueError("output_keys length must match input_keys length")
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, ctx: SkillContext) -> SkillContext:
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            sequence_ids = ctx.get_required(input_key).long()
            if sequence_ids.dim() != 2:
                raise ValueError(f"{input_key} must have shape [batch_size, seq_len]")
            ctx.put(output_key, self.embedding(sequence_ids))
        return ctx
