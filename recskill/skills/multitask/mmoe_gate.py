from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import MLP


@register_skill("mmoe_gate")
class MMOEGateSkill(BaseSkill):
    """MMoE experts plus task-specific gates.

    Source model: ``torch_rechub.models.multi_task.mmoe.MMOE``.
    """

    skill_spec = SkillSpec(
        name="mmoe_gate",
        category="multitask",
        description="Mix shared expert MLP outputs with one softmax gate per task.",
        input_specs=[TensorSpec("shared_input", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("task_representations", "[batch_size, n_task, expert_dim]")],
        task_types=["multitask"],
        inductive_bias=["shared_experts", "task_specific_gates"],
        failure_signatures=["input_dim_mismatch", "n_task_mismatch"],
        hyperparameters={"input_dim": "int", "n_expert": "int", "n_task": "int", "expert_hidden_dims": "list[int]"},
        paper_origin="Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts",
        implementation="torch_rechub.basic.layers.MLP",
    )

    def __init__(
        self,
        input_dim: int,
        n_expert: int,
        n_task: int,
        expert_hidden_dims: list[int],
        input_key: str = "shared_input",
        output_key: str = "task_representations",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not expert_hidden_dims:
            raise ValueError("expert_hidden_dims must not be empty")
        self.input_dim = input_dim
        self.n_expert = n_expert
        self.n_task = n_task
        self.input_key = input_key
        self.output_key = output_key
        self.experts = torch.nn.ModuleList(
            [MLP(input_dim, output_layer=False, dims=expert_hidden_dims, activation=activation, dropout=dropout) for _ in range(n_expert)]
        )
        self.gates = torch.nn.ModuleList(
            [MLP(input_dim, output_layer=False, dims=[n_expert], activation="softmax", dropout=0.0) for _ in range(n_task)]
        )

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        expert_outs = torch.cat([expert(x).unsqueeze(1) for expert in self.experts], dim=1)
        task_representations = []
        for gate in self.gates:
            gate_out = gate(x).unsqueeze(-1)
            task_representations.append(torch.sum(gate_out * expert_outs, dim=1).unsqueeze(1))
        return ctx.put(self.output_key, torch.cat(task_representations, dim=1))
