from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import MLP, PredictionLayer


@register_skill("task_tower")
class TaskTowerSkill(BaseSkill):
    """Apply one MLP tower and prediction layer per task."""

    skill_spec = SkillSpec(
        name="task_tower",
        category="multitask",
        description="Project task-specific representations to task outputs.",
        input_specs=[TensorSpec("task_representations", "[batch_size, n_task, input_dim]")],
        output_specs=[TensorSpec("task_outputs", "[batch_size, n_task]")],
        task_types=["multitask"],
        inductive_bias=["task_specific_heads"],
        failure_signatures=["task_count_mismatch", "input_dim_mismatch"],
        hyperparameters={"input_dim": "int", "task_types": "list[str]", "tower_hidden_dims": "list[list[int]]"},
        implementation="torch_rechub.basic.layers.MLP and PredictionLayer",
    )

    def __init__(
        self,
        input_dim: int,
        task_types: list[str],
        tower_hidden_dims,
        input_key: str = "task_representations",
        output_key: str = "task_outputs",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.task_types = list(task_types)
        self.n_task = len(self.task_types)
        self.input_key = input_key
        self.output_key = output_key
        dims_list = self._normalize_tower_dims(tower_hidden_dims, self.n_task)
        self.towers = torch.nn.ModuleList(
            [MLP(input_dim, output_layer=True, dims=dims, activation=activation, dropout=dropout) for dims in dims_list]
        )
        self.predict_layers = torch.nn.ModuleList([PredictionLayer(task_type) for task_type in self.task_types])

    def forward(self, ctx: SkillContext) -> SkillContext:
        task_representations = ctx.get_required(self.input_key)
        if task_representations.dim() != 3:
            raise ValueError(f"{self.input_key} must have shape [batch_size, n_task, input_dim]")
        if task_representations.size(1) != self.n_task or task_representations.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have n_task={self.n_task} and input_dim={self.input_dim}")
        outputs = []
        for task_idx, (tower, predict_layer) in enumerate(zip(self.towers, self.predict_layers)):
            tower_out = tower(task_representations[:, task_idx, :])
            outputs.append(predict_layer(tower_out))
        return ctx.put(self.output_key, torch.cat(outputs, dim=1))

    @staticmethod
    def _normalize_tower_dims(tower_hidden_dims, n_task: int) -> list[list[int]]:
        if tower_hidden_dims is None:
            return [[] for _ in range(n_task)]
        if len(tower_hidden_dims) == 0:
            return [[] for _ in range(n_task)]
        if all(isinstance(dim, int) for dim in tower_hidden_dims):
            return [list(tower_hidden_dims) for _ in range(n_task)]
        if len(tower_hidden_dims) != n_task:
            raise ValueError("tower_hidden_dims must be a list[int] or one list per task")
        return [list(dims) for dims in tower_hidden_dims]
