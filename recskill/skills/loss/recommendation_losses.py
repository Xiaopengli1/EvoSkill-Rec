from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("sampled_softmax_loss")
class SampledSoftmaxLossSkill(BaseSkill):
    """Cross-entropy over one positive plus sampled negatives."""

    skill_spec = SkillSpec(
        name="sampled_softmax_loss",
        category="loss",
        description="Train sampled matching logits where column 0 is the positive item.",
        input_specs=[TensorSpec("sampled_logits", "[batch_size, 1 + n_neg]")],
        output_specs=[TensorSpec("loss", "[]")],
        task_types=["matching", "sequence"],
        inductive_bias=["sampled_negative_retrieval"],
        failure_signatures=["sampled_logits_rank_mismatch"],
        implementation="YoutubeDNN/GRU4Rec/MIND/ComiRec sampled softmax training objective",
    )

    def __init__(self, logits_key: str = "sampled_logits", target_key: str | None = None, output_key: str = "loss", temperature: float = 1.0) -> None:
        super().__init__()
        self.logits_key = logits_key
        self.target_key = target_key
        self.output_key = output_key
        self.temperature = temperature

    def forward(self, ctx: SkillContext) -> SkillContext:
        logits = ctx.get_required(self.logits_key)
        if logits.dim() != 2:
            raise ValueError(f"{self.logits_key} must have shape [batch_size, 1 + n_neg]")
        target = ctx.get(self.target_key) if self.target_key is not None else None
        if target is None:
            target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return ctx.put(self.output_key, F.cross_entropy(logits / self.temperature, target.long()))


@register_skill("hinge_loss")
class HingeLossSkill(BaseSkill):
    """Torch-RecHub FaceBookDSSM hinge loss wrapper."""

    skill_spec = SkillSpec(
        name="hinge_loss",
        category="loss",
        description="Compute max-margin loss from positive and negative retrieval scores.",
        input_specs=[
            TensorSpec("pos_score", "[batch_size]"),
            TensorSpec("neg_score", "[batch_size, n_neg]"),
        ],
        output_specs=[TensorSpec("loss", "[]")],
        task_types=["matching"],
        inductive_bias=["margin_ranking"],
        failure_signatures=["score_shape_mismatch"],
        hyperparameters={"margin": "float"},
        implementation="torch_rechub.basic.loss_func.HingeLoss",
    )

    def __init__(self, pos_key: str = "pos_score", neg_key: str = "neg_score", output_key: str = "loss", margin: float = 0.8, num_items: int | None = None) -> None:
        super().__init__()
        from torch_rechub.basic.loss_func import HingeLoss

        self.pos_key = pos_key
        self.neg_key = neg_key
        self.output_key = output_key
        self.loss = HingeLoss(margin=margin, num_items=num_items)

    def forward(self, ctx: SkillContext) -> SkillContext:
        return ctx.put(self.output_key, self.loss(ctx.get_required(self.pos_key), ctx.get_required(self.neg_key)))


@register_skill("in_batch_nce_loss")
class InBatchNCELossSkill(BaseSkill):
    """Torch-RecHub in-batch NCE loss over an item table."""

    skill_spec = SkillSpec(
        name="in_batch_nce_loss",
        category="loss",
        description="Compute in-batch NCE loss from query embeddings, item embeddings, and target ids.",
        input_specs=[
            TensorSpec("query_embeddings", "[batch_size, embedding_dim]"),
            TensorSpec("item_embeddings", "[num_items, embedding_dim]"),
            TensorSpec("targets", "[batch_size]", "int64"),
        ],
        output_specs=[TensorSpec("loss", "[]")],
        task_types=["matching", "sequence", "generative"],
        inductive_bias=["in_batch_negative_sampling"],
        failure_signatures=["target_index_mismatch"],
        implementation="torch_rechub.basic.loss_func.InBatchNCELoss",
    )

    def __init__(
        self,
        query_key: str = "query_embeddings",
        item_key: str = "item_embeddings",
        target_key: str = "targets",
        output_key: str = "loss",
        temperature: float = 0.1,
        ignore_index: int = 0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        from torch_rechub.basic.loss_func import InBatchNCELoss

        self.query_key = query_key
        self.item_key = item_key
        self.target_key = target_key
        self.output_key = output_key
        self.loss = InBatchNCELoss(temperature=temperature, ignore_index=ignore_index, reduction=reduction)

    def forward(self, ctx: SkillContext) -> SkillContext:
        loss = self.loss(ctx.get_required(self.query_key), ctx.get_required(self.item_key), ctx.get_required(self.target_key).long())
        return ctx.put(self.output_key, loss)


@register_skill("multitask_loss")
class MultiTaskLossSkill(BaseSkill):
    """Aggregate per-task classification/regression losses."""

    skill_spec = SkillSpec(
        name="multitask_loss",
        category="loss",
        description="Compute weighted multitask losses for SharedBottom/MMOE/PLE/ESMM/AITM outputs.",
        input_specs=[
            TensorSpec("task_outputs", "[batch_size, n_task]"),
            TensorSpec("task_labels", "[batch_size, n_task]"),
        ],
        output_specs=[TensorSpec("loss", "[]")],
        task_types=["multitask"],
        inductive_bias=["task_weighted_supervision"],
        failure_signatures=["task_count_mismatch"],
        implementation="Torch-RecHub multitask model training objective adapter",
    )

    def __init__(
        self,
        task_types: Sequence[str],
        weights: Sequence[float] | None = None,
        predictions_key: str = "task_outputs",
        labels_key: str = "task_labels",
        output_key: str = "loss",
    ) -> None:
        super().__init__()
        self.task_types = list(task_types)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.task_types)
        self.predictions_key = predictions_key
        self.labels_key = labels_key
        self.output_key = output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        preds = ctx.get_required(self.predictions_key).float()
        labels = ctx.get_required(self.labels_key).float()
        if preds.shape != labels.shape or preds.size(1) != len(self.task_types):
            raise ValueError("task predictions and labels must have shape [batch_size, n_task]")
        losses = []
        for idx, task_type in enumerate(self.task_types):
            if task_type in {"binary", "classification", "binary_classification"}:
                loss = F.binary_cross_entropy(preds[:, idx], labels[:, idx])
            elif task_type == "regression":
                loss = F.mse_loss(preds[:, idx], labels[:, idx])
            else:
                raise ValueError(f"unsupported task_type: {task_type}")
            losses.append(self.weights[idx] * loss)
        return ctx.put(self.output_key, torch.stack(losses).sum())


@register_skill("sequence_cross_entropy_loss")
class SequenceCrossEntropyLossSkill(BaseSkill):
    """Token-level cross entropy for generative recommendation."""

    skill_spec = SkillSpec(
        name="sequence_cross_entropy_loss",
        category="loss",
        description="Compute sequence language-model cross entropy over item-token logits.",
        input_specs=[
            TensorSpec("logits", "[batch_size, seq_len, vocab_size]"),
            TensorSpec("labels", "[batch_size, seq_len]", "int64"),
        ],
        output_specs=[TensorSpec("loss", "[]")],
        task_types=["generative", "sequence"],
        inductive_bias=["autoregressive_item_prediction"],
        failure_signatures=["vocab_label_shape_mismatch"],
        implementation="torch.nn.CrossEntropyLoss for HSTU/HLLM/TIGER",
    )

    def __init__(self, logits_key: str = "logits", labels_key: str = "labels", output_key: str = "loss", ignore_index: int = -100, temperature: float = 1.0) -> None:
        super().__init__()
        self.logits_key = logits_key
        self.labels_key = labels_key
        self.output_key = output_key
        self.ignore_index = ignore_index
        self.temperature = temperature

    def forward(self, ctx: SkillContext) -> SkillContext:
        logits = ctx.get_required(self.logits_key)
        labels = ctx.get_required(self.labels_key).long().to(logits.device)
        loss = F.cross_entropy((logits / self.temperature).view(-1, logits.size(-1)), labels.view(-1), ignore_index=self.ignore_index)
        return ctx.put(self.output_key, loss)


@register_skill("ranking_loss_temperature")
class RankingLossTemperatureSkill(BaseSkill):
    """TIGER ranking loss with configurable logit temperature."""

    skill_spec = SkillSpec(
        name="ranking_loss_temperature",
        category="loss",
        description="Scale TIGER logits by temperature before sequence cross entropy.",
        input_specs=[
            TensorSpec("logits", "[batch_size, seq_len, vocab_size]"),
            TensorSpec("labels", "[batch_size, seq_len]", "int64"),
        ],
        output_specs=[TensorSpec("loss", "[]")],
        task_types=["generative"],
        inductive_bias=["temperature_scaled_ranking_loss"],
        failure_signatures=["temperature_not_positive"],
        hyperparameters={"temperature": "float"},
        implementation="torch_rechub.models.generative.tiger.TIGERModel.ranking_loss",
    )

    def __init__(self, logits_key: str = "logits", labels_key: str = "labels", output_key: str = "loss", temperature: float = 1.0, ignore_index: int = -100) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.logits_key = logits_key
        self.labels_key = labels_key
        self.output_key = output_key
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(self, ctx: SkillContext) -> SkillContext:
        logits = ctx.get_required(self.logits_key)
        labels = ctx.get_required(self.labels_key).long().to(logits.device)
        loss = F.cross_entropy((logits / self.temperature).view(-1, logits.size(-1)), labels.view(-1), ignore_index=self.ignore_index)
        return ctx.put(self.output_key, loss)


@register_skill("covariance_regularizer")
class CovarianceRegularizerSkill(BaseSkill):
    """SINE concept-embedding covariance regularizer."""

    skill_spec = SkillSpec(
        name="covariance_regularizer",
        category="loss",
        description="Penalize off-diagonal covariance in concept embeddings.",
        input_specs=[TensorSpec("concept_embeddings", "[num_concepts, embedding_dim]")],
        output_specs=[TensorSpec("covariance_regularizer", "[]")],
        task_types=["matching", "sequence"],
        inductive_bias=["decorrelated_concepts"],
        failure_signatures=["rank_not_2"],
        implementation="torch_rechub.models.matching.sine covariance regularizer",
    )

    def __init__(self, input_key: str = "concept_embeddings", output_key: str = "covariance_regularizer", correction: int = 0) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.correction = correction

    def forward(self, ctx: SkillContext) -> SkillContext:
        embeddings = ctx.get_required(self.input_key).float()
        if embeddings.dim() != 2:
            raise ValueError(f"{self.input_key} must have shape [num_concepts, embedding_dim]")
        cov = torch.cov(embeddings, correction=self.correction)
        penalty = (torch.norm(cov, p="fro") ** 2 - torch.norm(torch.diag(cov), p="fro") ** 2) / 2
        return ctx.put(self.output_key, penalty)
