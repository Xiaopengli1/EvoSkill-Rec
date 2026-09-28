from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("aitm_transfer")
class AITMTransferSkill(BaseSkill):
    """Apply a lightweight sequential transfer layer across task representations."""

    skill_spec = SkillSpec(
        name="aitm_transfer",
        category="multitask",
        description="Transfer information from earlier task representations to later tasks.",
        input_specs=[TensorSpec("task_representations", "[batch_size, n_task, input_dim]")],
        output_specs=[TensorSpec("task_representations", "[batch_size, n_task, input_dim]")],
        task_types=["multitask"],
        inductive_bias=["ordered_task_transfer"],
        failure_signatures=["rank_not_3", "task_count_mismatch"],
        hyperparameters={"input_dim": "int", "n_task": "int"},
        paper_origin="Adaptive Information Transfer Multi-task Learning",
    )

    def __init__(self, input_dim: int, n_task: int, input_key: str = "task_representations", output_key: str = "task_representations") -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_task = n_task
        self.input_key = input_key
        self.output_key = output_key
        self.transfer = torch.nn.ModuleList([torch.nn.Linear(input_dim * 2, input_dim) for _ in range(max(n_task - 1, 0))])

    def forward(self, ctx: SkillContext) -> SkillContext:
        reps = ctx.get_required(self.input_key)
        if reps.dim() != 3 or reps.size(1) != self.n_task or reps.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.n_task}, {self.input_dim}]")
        outputs = [reps[:, 0, :]]
        for idx in range(1, self.n_task):
            mixed = torch.tanh(self.transfer[idx - 1](torch.cat([outputs[-1], reps[:, idx, :]], dim=-1)))
            outputs.append(mixed)
        return ctx.put(self.output_key, torch.stack(outputs, dim=1))
