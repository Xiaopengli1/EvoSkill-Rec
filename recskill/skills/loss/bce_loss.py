from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("bce_loss")
class BCELossSkill(BaseSkill):
    """Binary cross-entropy loss skill for CTR logits and labels."""

    skill_spec = SkillSpec(
        name="bce_loss",
        category="loss",
        description="Compute binary cross-entropy loss from logits or probabilities and labels.",
        input_specs=[TensorSpec("logits", "[batch_size, 1]"), TensorSpec("labels", "[batch_size]")],
        output_specs=[TensorSpec("loss", "[]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["binary_classification_likelihood"],
        failure_signatures=["label_shape_mismatch", "nan_logits"],
        hyperparameters={"from_logits": "bool", "logits_key": "str", "labels_key": "str", "output_key": "str"},
        implementation="torch.nn.BCEWithLogitsLoss",
    )

    def __init__(
        self,
        logits_key: str = "logits",
        labels_key: str = "labels",
        output_key: str = "loss",
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.logits_key = logits_key
        self.labels_key = labels_key
        self.output_key = output_key
        self.loss_fn = torch.nn.BCEWithLogitsLoss() if from_logits else torch.nn.BCELoss()

    def forward(self, ctx: SkillContext) -> SkillContext:
        logits = ctx.get_required(self.logits_key)
        labels = ctx.get_required(self.labels_key).float()
        if labels.shape != logits.shape:
            labels = labels.reshape_as(logits)
        return ctx.put(self.output_key, self.loss_fn(logits, labels))
