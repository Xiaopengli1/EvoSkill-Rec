from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("precomputed_item_embedding_adapter")
class PrecomputedItemEmbeddingAdapterSkill(BaseSkill):
    """Load and normalize HLLM precomputed item embeddings."""

    skill_spec = SkillSpec(
        name="precomputed_item_embedding_adapter",
        category="generative",
        description="Use frozen precomputed item embeddings as HLLM sequence inputs and scoring table.",
        input_specs=[TensorSpec("seq_tokens", "[batch_size, seq_len]", "int64")],
        output_specs=[
            TensorSpec("sequence_embeddings", "[batch_size, seq_len, d_model]"),
            TensorSpec("item_embeddings", "[vocab_size, d_model]"),
        ],
        task_types=["generative", "matching", "sequence"],
        inductive_bias=["frozen_item_llm_embeddings"],
        failure_signatures=["item_embedding_shape_mismatch"],
        implementation="torch_rechub.models.generative.hllm.HLLMModel item_embeddings path",
    )

    def __init__(
        self,
        item_embeddings: torch.Tensor | str | None = None,
        vocab_size: int | None = None,
        d_model: int | None = None,
        input_key: str = "seq_tokens",
        table_key: str = "item_embeddings",
        output_key: str = "sequence_embeddings",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.table_key = table_key
        self.output_key = output_key
        self.normalize = normalize
        if isinstance(item_embeddings, str):
            item_embeddings = torch.load(item_embeddings)
        if item_embeddings is not None:
            item_embeddings = item_embeddings.float()
            if normalize:
                item_embeddings = F.normalize(item_embeddings, dim=-1, eps=1e-8)
            self.register_buffer("item_embeddings", item_embeddings)
        else:
            if vocab_size is None or d_model is None:
                self.item_embeddings = None
            else:
                self.register_buffer("item_embeddings", torch.empty(vocab_size, d_model))

    def forward(self, ctx: SkillContext) -> SkillContext:
        table = getattr(self, "item_embeddings", None)
        if table is None or table.numel() == 0:
            table = ctx.get_required(self.table_key).float()
            if self.normalize:
                table = F.normalize(table, dim=-1, eps=1e-8)
        seq_tokens = ctx.get_required(self.input_key).long()
        ctx.put(self.table_key, table)
        return ctx.put(self.output_key, table[seq_tokens])


@register_skill("normalized_item_dot_head")
class NormalizedItemDotHeadSkill(BaseSkill):
    """HLLM normalized item dot-product scoring head."""

    skill_spec = SkillSpec(
        name="normalized_item_dot_head",
        category="generative",
        description="Normalize sequence/user states and score them against item embeddings with temperature.",
        input_specs=[
            TensorSpec("sequence_output", "[batch_size, seq_len, d_model]"),
            TensorSpec("item_embeddings", "[vocab_size, d_model]"),
        ],
        output_specs=[TensorSpec("logits", "[batch_size, seq_len, vocab_size]")],
        task_types=["generative", "matching", "sequence"],
        inductive_bias=["cosine_item_retrieval"],
        failure_signatures=["embedding_dim_mismatch"],
        hyperparameters={"temperature": "float"},
        implementation="torch_rechub.models.generative.hllm normalized item dot head",
    )

    def __init__(
        self,
        input_key: str = "sequence_output",
        item_key: str = "item_embeddings",
        output_key: str = "logits",
        temperature: float = 0.07,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.item_key = item_key
        self.output_key = output_key
        self.temperature = temperature
        self.eps = eps

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = F.normalize(ctx.get_required(self.input_key), dim=-1, eps=self.eps)
        items = F.normalize(ctx.get_required(self.item_key).float(), dim=-1, eps=self.eps)
        return ctx.put(self.output_key, torch.matmul(x, items.t()) / self.temperature)


@register_skill("exact_residual_quantizer_stack")
class ExactResidualQuantizerStackSkill(BaseSkill):
    """Wrap RQVAE's exact ResidualVectorQuantizer stack."""

    skill_spec = SkillSpec(
        name="exact_residual_quantizer_stack",
        category="generative",
        description="Apply Torch-RecHub RQVAE residual vector quantization with optional Sinkhorn assignment.",
        input_specs=[TensorSpec("latent", "[batch_size, e_dim]")],
        output_specs=[
            TensorSpec("quantized", "[batch_size, e_dim]"),
            TensorSpec("rq_loss", "[]"),
            TensorSpec("code_indices", "[batch_size, num_quantizers]", "int64"),
        ],
        task_types=["generative", "matching"],
        inductive_bias=["residual_vector_quantization"],
        failure_signatures=["codebook_collapse", "sinkhorn_nan"],
        hyperparameters={"n_e_list": "list[int]", "e_dim": "int"},
        implementation="torch_rechub.models.generative.rqvae.ResidualVectorQuantizer",
    )

    def __init__(
        self,
        n_e_list: Sequence[int],
        e_dim: int,
        sk_epsilons: Sequence[float] | None = None,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_iters: int = 100,
        input_key: str = "latent",
        output_key: str = "quantized",
        loss_key: str = "rq_loss",
        indices_key: str = "code_indices",
        use_sk: bool = True,
    ) -> None:
        super().__init__()
        from torch_rechub.models.generative.rqvae import ResidualVectorQuantizer

        self.input_key = input_key
        self.output_key = output_key
        self.loss_key = loss_key
        self.indices_key = indices_key
        self.use_sk = use_sk
        sk_epsilons = list(sk_epsilons) if sk_epsilons is not None else [0.003] * len(n_e_list)
        self.quantizer = ResidualVectorQuantizer(list(n_e_list), e_dim, sk_epsilons, beta=beta, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters, sk_iters=sk_iters)

    def forward(self, ctx: SkillContext) -> SkillContext:
        quantized, loss, indices = self.quantizer(ctx.get_required(self.input_key), use_sk=self.use_sk)
        ctx.put(self.output_key, quantized)
        ctx.put(self.loss_key, loss)
        return ctx.put(self.indices_key, indices)


@register_skill("kmeans_sinkhorn_initialization")
class KMeansSinkhornInitializationSkill(BaseSkill):
    """Expose RQVAE kmeans and Sinkhorn assignment utilities as reusable skills."""

    skill_spec = SkillSpec(
        name="kmeans_sinkhorn_initialization",
        category="generative",
        description="Run Torch-RecHub RQVAE kmeans initialization or Sinkhorn soft assignment.",
        input_specs=[TensorSpec("samples", "[num_samples, dim]")],
        output_specs=[TensorSpec("centers", "[num_clusters, dim]")],
        task_types=["generative", "matching"],
        inductive_bias=["codebook_initialization", "balanced_assignment"],
        failure_signatures=["sklearn_unavailable", "sinkhorn_nan"],
        implementation="torch_rechub.models.generative.rqvae.kmeans / sinkhorn_algorithm",
    )

    def __init__(
        self,
        mode: str = "kmeans",
        num_clusters: int | None = None,
        num_iters: int = 10,
        epsilon: float = 0.003,
        sinkhorn_iterations: int = 100,
        input_key: str = "samples",
        distances_key: str = "distances",
        output_key: str = "centers",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_clusters = num_clusters
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.input_key = input_key
        self.distances_key = distances_key
        self.output_key = output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        from torch_rechub.models.generative.rqvae import kmeans, sinkhorn_algorithm

        if self.mode == "kmeans":
            if self.num_clusters is None:
                raise ValueError("num_clusters is required for kmeans mode")
            return ctx.put(self.output_key, kmeans(ctx.get_required(self.input_key), self.num_clusters, self.num_iters))
        if self.mode == "sinkhorn":
            return ctx.put(self.output_key, sinkhorn_algorithm(ctx.get_required(self.distances_key), self.epsilon, self.sinkhorn_iterations))
        raise ValueError(f"unsupported mode: {self.mode}")


@register_skill("semantic_id_export_adapter")
class SemanticIDExportAdapterSkill(BaseSkill):
    """Convert residual quantizer code indices to semantic ID tokens."""

    skill_spec = SkillSpec(
        name="semantic_id_export_adapter",
        category="generative",
        description="Export RQVAE residual code indices as TIGER-style semantic ID token strings.",
        input_specs=[TensorSpec("code_indices", "[batch_size, num_quantizers]", "int64")],
        output_specs=[TensorSpec("semantic_ids", "list[list[str]]")],
        task_types=["generative", "matching"],
        inductive_bias=["semantic_id_tokenization"],
        failure_signatures=["prefix_count_mismatch"],
        implementation="torch_rechub.models.generative.rqvae.RQVAEModel.generate_semantic_ids adapter",
    )

    def __init__(
        self,
        prefixes: Sequence[str] | None = None,
        input_key: str = "code_indices",
        output_key: str = "semantic_ids",
        string_output_key: str = "semantic_id_strings",
    ) -> None:
        super().__init__()
        self.prefixes = list(prefixes or ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"])
        self.input_key = input_key
        self.output_key = output_key
        self.string_output_key = string_output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        indices = ctx.get_required(self.input_key).detach().cpu().long()
        if indices.size(-1) > len(self.prefixes):
            raise ValueError("prefix count must be no less than number of quantizers")
        semantic_ids = [[self.prefixes[idx].format(int(value)) for idx, value in enumerate(row)] for row in indices.view(-1, indices.size(-1))]
        ctx.put(self.output_key, semantic_ids)
        return ctx.put(self.string_output_key, [str(row) for row in semantic_ids])


@register_skill("exact_t5_encoder_decoder")
class ExactT5EncoderDecoderSkill(BaseSkill):
    """Wrap TIGER's T5 encoder-decoder forward path."""

    skill_spec = SkillSpec(
        name="exact_t5_encoder_decoder",
        category="generative",
        description="Run Torch-RecHub TIGERModel, a T5ForConditionalGeneration subclass with ranking loss.",
        input_specs=[TensorSpec("input_ids", "[batch_size, seq_len]", "int64")],
        output_specs=[TensorSpec("logits", "[batch_size, target_len, vocab_size]")],
        task_types=["generative"],
        inductive_bias=["seq2seq_semantic_id_generation"],
        failure_signatures=["transformers_unavailable"],
        implementation="torch_rechub.models.generative.tiger.TIGERModel",
    )

    def __init__(self, config: dict | None = None, output_key: str = "tiger_output", logits_key: str = "logits") -> None:
        super().__init__()
        from transformers.models.t5.configuration_t5 import T5Config
        from torch_rechub.models.generative.tiger import TIGERModel

        self.output_key = output_key
        self.logits_key = logits_key
        self.model = TIGERModel(T5Config(**(config or {})))

    def forward(self, ctx: SkillContext) -> SkillContext:
        kwargs = {
            key: ctx[key]
            for key in (
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
                "labels",
                "inputs_embeds",
                "decoder_inputs_embeds",
            )
            if key in ctx
        }
        output = self.model(**kwargs)
        ctx.put(self.output_key, output)
        return ctx.put(self.logits_key, output.logits)


@register_skill("trie_constrained_decoding")
class TrieConstrainedDecodingSkill(BaseSkill):
    """Reusable prefix-trie constrained decoding helper for TIGER semantic IDs."""

    skill_spec = SkillSpec(
        name="trie_constrained_decoding",
        category="generative",
        description="Return allowed next tokens from a prefix trie during semantic ID generation.",
        input_specs=[TensorSpec("prefix_ids", "[prefix_len]", "int64")],
        output_specs=[TensorSpec("allowed_token_ids", "[num_allowed]", "int64")],
        task_types=["generative"],
        inductive_bias=["semantic_id_trie_constraint"],
        failure_signatures=["dead_prefix"],
        implementation="TIGER trie constrained decoding adapter",
    )

    def __init__(
        self,
        trie: dict | None = None,
        eos_token_id: int | None = None,
        input_key: str = "prefix_ids",
        output_key: str = "allowed_token_ids",
    ) -> None:
        super().__init__()
        self.trie = trie or {}
        self.eos_token_id = eos_token_id
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        prefix = ctx.get_required(self.input_key)
        if torch.is_tensor(prefix):
            prefix = prefix.detach().cpu().long().tolist()
        node = self.trie
        for token in prefix:
            token = str(int(token))
            if token not in node:
                allowed = [] if self.eos_token_id is None else [self.eos_token_id]
                return ctx.put(self.output_key, torch.tensor(allowed, dtype=torch.long))
            node = node[token]
        allowed = [int(token) for token in node.keys()]
        if not allowed and self.eos_token_id is not None:
            allowed = [self.eos_token_id]
        return ctx.put(self.output_key, torch.tensor(allowed, dtype=torch.long))
