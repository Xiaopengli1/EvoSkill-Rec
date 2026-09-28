from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import MLP


@register_skill("task_specific_towers")
class TaskSpecificTowersSkill(BaseSkill):
    """Apply separate bottom towers to one shared input, one per task."""

    skill_spec = SkillSpec(
        name="task_specific_towers",
        category="multitask",
        description="Create task representations with separate bottom MLPs from the same input.",
        input_specs=[TensorSpec("shared_input", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("task_representations", "[batch_size, n_task, hidden_dim]")],
        task_types=["multitask"],
        inductive_bias=["task_specific_bottoms"],
        failure_signatures=["input_dim_mismatch"],
        hyperparameters={"input_dim": "int", "n_task": "int", "hidden_dims": "list[int]"},
        paper_origin="AITM",
    )

    def __init__(
        self,
        input_dim: int,
        n_task: int,
        hidden_dims: list[int],
        input_key: str = "shared_input",
        output_key: str = "task_representations",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        self.input_dim = input_dim
        self.n_task = n_task
        self.input_key = input_key
        self.output_key = output_key
        self.towers = torch.nn.ModuleList(
            [MLP(input_dim, dims=hidden_dims, output_layer=False, activation=activation, dropout=dropout) for _ in range(n_task)]
        )

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        return ctx.put(self.output_key, torch.stack([tower(x) for tower in self.towers], dim=1))
