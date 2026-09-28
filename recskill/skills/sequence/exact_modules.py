from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("sequence_mask")
class SequenceMaskSkill(BaseSkill):
    """Build boolean sequence masks from padded id tensors."""

    skill_spec = SkillSpec(
        name="sequence_mask",
        category="sequence",
        description="Create True-for-valid masks from sequence id tensors using Torch-RecHub padding conventions.",
        input_specs=[TensorSpec("sequence_ids", "[batch_size, ... , seq_len]", "int64")],
        output_specs=[TensorSpec("sequence_mask", "[batch_size, ... , seq_len]", "bool")],
        task_types=["ctr", "ranking", "matching", "sequence", "generative"],
        inductive_bias=["padding_idx_zero"],
        failure_signatures=["missing_sequence_ids"],
        implementation="torch_rechub sequence padding mask",
    )

    def __init__(self, input_key: str = "sequence_ids", output_key: str = "sequence_mask", padding_idx: int = 0) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.padding_idx = padding_idx

    def forward(self, ctx: SkillContext) -> SkillContext:
        ids = ctx.get_required(self.input_key).long()
        return ctx.put(self.output_key, ids.ne(self.padding_idx))


@register_skill("full_augru_evolution")
class FullAUGRUEvolutionSkill(BaseSkill):
    """Wrap the exact DIEN AUGRU modules from Torch-RecHub."""

    skill_spec = SkillSpec(
        name="full_augru_evolution",
        category="sequence",
        description="Apply DIEN's attentional update GRU to extracted interests and target embeddings.",
        input_specs=[
            TensorSpec("interest_sequence", "[batch_size, num_fields, seq_len, embedding_dim]"),
            TensorSpec("target_embeddings", "[batch_size, num_fields, embedding_dim]"),
            TensorSpec("sequence_mask", "[batch_size, num_fields, seq_len]", "bool"),
        ],
        output_specs=[
            TensorSpec("interest_evolving", "[batch_size, num_fields, embedding_dim]"),
            TensorSpec("interest_evolution_sequence", "[batch_size, num_fields, seq_len, embedding_dim]"),
        ],
        task_types=["ctr", "ranking", "sequence"],
        inductive_bias=["augru_target_conditioned_update"],
        failure_signatures=["history_target_field_mismatch"],
        hyperparameters={"embedding_dim": "int", "num_fields": "int"},
        implementation="torch_rechub.models.ranking.dien.AUGRU",
    )

    def __init__(
        self,
        embedding_dim: int,
        num_fields: int = 1,
        sequence_key: str = "interest_sequence",
        target_key: str = "target_embeddings",
        mask_key: str | None = "sequence_mask",
        output_key: str = "interest_evolving",
        sequence_output_key: str = "interest_evolution_sequence",
    ) -> None:
        super().__init__()
        from torch_rechub.models.ranking.dien import AUGRU

        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.sequence_key = sequence_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.sequence_output_key = sequence_output_key
        self.layers = torch.nn.ModuleList([AUGRU(embedding_dim) for _ in range(num_fields)])

    def forward(self, ctx: SkillContext) -> SkillContext:
        seq = ctx.get_required(self.sequence_key)
        target = ctx.get_required(self.target_key)
        if seq.dim() != 4 or seq.size(1) != self.num_fields or seq.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.sequence_key} must have shape [batch_size, {self.num_fields}, seq_len, {self.embedding_dim}]")
        if target.dim() != 3 or target.size(1) != self.num_fields or target.size(-1) != self.embedding_dim:
            raise ValueError(f"{self.target_key} must have shape [batch_size, {self.num_fields}, {self.embedding_dim}]")
        mask = ctx.get_required(self.mask_key).bool() if self.mask_key is not None and self.mask_key in ctx else None
        finals = []
        sequences = []
        for idx, layer in enumerate(self.layers):
            field_mask = None if mask is None else mask[:, idx, :]
            out, h = layer(seq[:, idx, :, :], target[:, idx, :], field_mask)
            sequences.append(out.unsqueeze(1))
            finals.append(h.unsqueeze(1))
        ctx.put(self.sequence_output_key, torch.cat(sequences, dim=1))
        return ctx.put(self.output_key, torch.cat(finals, dim=1))


@register_skill("dien_auxiliary_interest_loss")
class DIENAuxiliaryInterestLossSkill(BaseSkill):
    """Compute DIEN next-step positive/negative auxiliary loss."""

    skill_spec = SkillSpec(
        name="dien_auxiliary_interest_loss",
        category="loss",
        description="Use DIEN's auxiliary next-behavior supervision over GRU interest states.",
        input_specs=[
            TensorSpec("interest_sequence", "[batch_size, seq_len, embedding_dim]"),
            TensorSpec("positive_sequence_embeddings", "[batch_size, seq_len, embedding_dim]"),
            TensorSpec("negative_sequence_embeddings", "[batch_size, seq_len, embedding_dim]"),
        ],
        output_specs=[TensorSpec("aux_loss", "[]")],
        task_types=["ctr", "ranking", "sequence"],
        inductive_bias=["next_behavior_auxiliary_supervision"],
        failure_signatures=["all_padding_auxiliary_batch"],
        implementation="torch_rechub.models.ranking.dien.DIEN.auxiliary",
    )

    def __init__(
        self,
        interest_key: str = "interest_sequence",
        positive_key: str = "positive_sequence_embeddings",
        negative_key: str = "negative_sequence_embeddings",
        mask_key: str | None = "sequence_mask",
        output_key: str = "aux_loss",
    ) -> None:
        super().__init__()
        self.interest_key = interest_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.mask_key = mask_key
        self.output_key = output_key
        self.loss = torch.nn.BCELoss()

    def forward(self, ctx: SkillContext) -> SkillContext:
        outs = ctx.get_required(self.interest_key)
        pos_emb = ctx.get_required(self.positive_key)
        neg_emb = ctx.get_required(self.negative_key)
        mask = ctx.get_required(self.mask_key).bool() if self.mask_key is not None and self.mask_key in ctx else None
        h = outs[:, :-1]
        pos = pos_emb[:, 1:]
        neg = neg_emb[:, 1:]
        if mask is not None:
            valid = mask[:, :-1] & mask[:, 1:]
        else:
            valid = torch.ones(h.size(0), h.size(1), dtype=torch.bool, device=h.device)
        h = h[valid]
        pos = pos[valid]
        neg = neg[valid]
        if h.size(0) == 0:
            return ctx.put(self.output_key, outs.new_tensor(0.0))
        pos_score = torch.sigmoid((h * pos).sum(-1, keepdim=True))
        neg_score = torch.sigmoid((h * neg).sum(-1, keepdim=True))
        return ctx.put(self.output_key, self.loss(pos_score, torch.ones_like(pos_score)) + self.loss(neg_score, torch.zeros_like(neg_score)))


@register_skill("exact_leakyrelu_transformer_encoder")
class ExactLeakyReLUTransformerEncoderSkill(BaseSkill):
    """BST-style TransformerEncoderLayer with LeakyReLU activation."""

    skill_spec = SkillSpec(
        name="exact_leakyrelu_transformer_encoder",
        category="sequence",
        description="Encode BST sequence tokens with nn.TransformerEncoderLayer using LeakyReLU activation.",
        input_specs=[TensorSpec("sequence_embeddings", "[batch_size, seq_len, d_model]")],
        output_specs=[TensorSpec("sequence_output", "[batch_size, seq_len, d_model]")],
        task_types=["ctr", "ranking", "sequence"],
        inductive_bias=["bst_transformer_self_attention"],
        failure_signatures=["d_model_head_mismatch"],
        hyperparameters={"d_model": "int", "nhead": "int", "num_layers": "int"},
        implementation="torch_rechub.models.ranking.bst TransformerEncoderLayer(activation=LeakyReLU)",
    )

    def __init__(
        self,
        d_model: int,
        nhead: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_key: str = "sequence_embeddings",
        mask_key: str | None = None,
        output_key: str = "sequence_output",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_key = input_key
        self.mask_key = mask_key
        self.output_key = output_key
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, activation=torch.nn.LeakyReLU(), batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"{self.input_key} must have shape [batch_size, seq_len, {self.d_model}]")
        key_padding_mask = None
        if self.mask_key is not None and self.mask_key in ctx:
            key_padding_mask = ~ctx.get_required(self.mask_key).bool()
        return ctx.put(self.output_key, self.encoder(x, src_key_padding_mask=key_padding_mask))


@register_skill("hllm_transformer_block")
class HLLMTransformerBlockSkill(BaseSkill):
    """Wrap the exact HLLMTransformerBlock implementation."""

    skill_spec = SkillSpec(
        name="hllm_transformer_block",
        category="sequence",
        description="Apply Torch-RecHub HLLM Transformer blocks with causal self-attention and FFN.",
        input_specs=[TensorSpec("sequence_embeddings", "[batch_size, seq_len, d_model]")],
        output_specs=[TensorSpec("sequence_output", "[batch_size, seq_len, d_model]")],
        task_types=["generative", "sequence", "matching"],
        inductive_bias=["hllm_causal_user_llm"],
        failure_signatures=["d_model_head_mismatch"],
        hyperparameters={"d_model": "int", "n_heads": "int", "n_layers": "int"},
        implementation="torch_rechub.models.generative.hllm.HLLMTransformerBlock",
    )

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 1,
        dropout: float = 0.1,
        input_key: str = "sequence_embeddings",
        rel_pos_bias_key: str | None = "rel_pos_bias",
        output_key: str = "sequence_output",
    ) -> None:
        super().__init__()
        from torch_rechub.models.generative.hllm import HLLMTransformerBlock

        self.input_key = input_key
        self.rel_pos_bias_key = rel_pos_bias_key
        self.output_key = output_key
        self.layers = torch.nn.ModuleList([HLLMTransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        rel_pos_bias = ctx.get(self.rel_pos_bias_key) if self.rel_pos_bias_key is not None else None
        for layer in self.layers:
            x = layer(x, rel_pos_bias=rel_pos_bias)
        return ctx.put(self.output_key, x)


@register_skill("relative_position_bias")
class RelativePositionBiasSkill(BaseSkill):
    """Reusable HLLM relative-position bias module."""

    skill_spec = SkillSpec(
        name="relative_position_bias",
        category="sequence",
        description="Compute Torch-RecHub legacy relative-position bias for HLLM attention.",
        input_specs=[TensorSpec("seq_len", "int")],
        output_specs=[TensorSpec("rel_pos_bias", "[1, n_heads, seq_len, seq_len]")],
        task_types=["generative", "sequence", "matching"],
        inductive_bias=["relative_position_attention_bias"],
        failure_signatures=["seq_len_exceeds_max"],
        hyperparameters={"n_heads": "int", "max_seq_len": "int"},
        implementation="torch_rechub.utils.hstu_utils.RelPosBias",
    )

    def __init__(self, n_heads: int, max_seq_len: int, num_buckets: int = 32, seq_len_key: str = "seq_len", output_key: str = "rel_pos_bias") -> None:
        super().__init__()
        from torch_rechub.utils.hstu_utils import RelPosBias

        self.seq_len_key = seq_len_key
        self.output_key = output_key
        self.max_seq_len = max_seq_len
        self.bias = RelPosBias(n_heads, max_seq_len, num_buckets=num_buckets)

    def forward(self, ctx: SkillContext) -> SkillContext:
        seq_len = int(ctx.get(self.seq_len_key, 0) or ctx.get_required("sequence_embeddings").size(1))
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")
        return ctx.put(self.output_key, self.bias(seq_len))


@register_skill("time_bucket_embedding")
class TimeBucketEmbeddingSkill(BaseSkill):
    """Bucket raw time differences and embed them like HSTU/HLLM."""

    skill_spec = SkillSpec(
        name="time_bucket_embedding",
        category="sequence",
        description="Map time differences to sqrt/log buckets and return trainable time embeddings.",
        input_specs=[TensorSpec("time_diffs", "[batch_size, seq_len]")],
        output_specs=[
            TensorSpec("time_buckets", "[batch_size, seq_len]", "int64"),
            TensorSpec("time_embeddings", "[batch_size, seq_len, d_model]"),
        ],
        task_types=["generative", "sequence"],
        inductive_bias=["bucketed_temporal_signal"],
        failure_signatures=["unsupported_time_bucket_fn"],
        hyperparameters={"num_time_buckets": "int", "d_model": "int"},
        implementation="torch_rechub.models.generative.hstu.HSTUModel._time_diff_to_bucket / hllm",
    )

    def __init__(
        self,
        num_time_buckets: int,
        d_model: int,
        time_bucket_fn: str = "sqrt",
        time_bucket_divisor: float = 1.0,
        time_bucket_unit: str = "minutes",
        padding_idx: int | None = None,
        input_key: str = "time_diffs",
        buckets_key: str = "time_buckets",
        output_key: str = "time_embeddings",
    ) -> None:
        super().__init__()
        self.num_time_buckets = num_time_buckets
        self.time_bucket_fn = time_bucket_fn
        self.time_bucket_divisor = time_bucket_divisor
        self.time_bucket_unit = time_bucket_unit
        self.input_key = input_key
        self.buckets_key = buckets_key
        self.output_key = output_key
        self.embedding = torch.nn.Embedding(num_time_buckets, d_model, padding_idx=padding_idx)

    def forward(self, ctx: SkillContext) -> SkillContext:
        time_diffs = ctx.get_required(self.input_key).float()
        if self.time_bucket_unit == "minutes":
            time_diffs = time_diffs / 60.0
        time_diffs = torch.clamp(time_diffs, min=1e-6)
        if self.time_bucket_fn == "sqrt":
            buckets = torch.sqrt(time_diffs)
        elif self.time_bucket_fn == "log":
            buckets = torch.log(time_diffs)
        else:
            raise ValueError(f"Unsupported time_bucket_fn: {self.time_bucket_fn}")
        buckets = (buckets / self.time_bucket_divisor).clamp(min=0, max=self.num_time_buckets - 1).long()
        ctx.put(self.buckets_key, buckets)
        return ctx.put(self.output_key, self.embedding(buckets))
