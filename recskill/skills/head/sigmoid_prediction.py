from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("sigmoid_prediction")
class SigmoidPredictionSkill(BaseSkill):
    """Convert binary CTR logits to the squeezed probability shape used by ranking models."""

    skill_spec = SkillSpec(
        name="sigmoid_prediction",
        category="head",
        description="Apply sigmoid to logits and squeeze the last singleton dimension.",
        input_specs=[TensorSpec("logits", "[batch_size, 1]")],
        output_specs=[TensorSpec("prediction", "[batch_size]")],
        task_types=["ctr", "ranking"],
        inductive_bias=["binary_probability_output"],
        failure_signatures=["logits_rank_mismatch"],
        hyperparameters={"input_key": "str", "output_key": "str"},
        implementation="torch.sigmoid",
    )

    def __init__(self, input_key: str = "logits", output_key: str = "prediction") -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        logits = ctx.get_required(self.input_key)
        prediction = torch.sigmoid(logits)
        if prediction.dim() > 1 and prediction.size(-1) == 1:
            prediction = prediction.squeeze(-1)
        return ctx.put(self.output_key, prediction)
