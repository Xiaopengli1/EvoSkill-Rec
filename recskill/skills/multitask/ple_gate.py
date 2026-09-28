from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import MLP


@register_skill("ple_gate")
class PLEGateSkill(BaseSkill):
    """Simplified PLE-style expert mixing for shape-compatible genomes."""

    skill_spec = SkillSpec(
        name="ple_gate",
        category="multitask",
        description="Mix shared and task-specific expert outputs into task representations.",
        input_specs=[TensorSpec("shared_input", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("task_representations", "[batch_size, n_task, expert_dim]")],
        task_types=["multitask"],
        inductive_bias=["progressive_expert_sharing", "task_specific_gates"],
        failure_signatures=["input_dim_mismatch"],
        hyperparameters={"input_dim": "int", "n_task": "int", "expert_hidden_dims": "list[int]"},
        paper_origin="Progressive Layered Extraction",
    )

    def __init__(
        self,
        input_dim: int,
        n_task: int,
        n_expert_shared: int,
        n_expert_specific: int,
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
        self.n_task = n_task
        self.input_key = input_key
        self.output_key = output_key
        self.expert_dim = expert_hidden_dims[-1]
        self.shared_experts = torch.nn.ModuleList([MLP(input_dim, dims=expert_hidden_dims, output_layer=False, activation=activation, dropout=dropout) for _ in range(n_expert_shared)])
        self.task_experts = torch.nn.ModuleList(
            torch.nn.ModuleList([MLP(input_dim, dims=expert_hidden_dims, output_layer=False, activation=activation, dropout=dropout) for _ in range(n_expert_specific)])
            for _ in range(n_task)
        )
        self.gates = torch.nn.ModuleList([torch.nn.Linear(input_dim, n_expert_shared + n_expert_specific) for _ in range(n_task)])

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        shared = [expert(x) for expert in self.shared_experts]
        outputs = []
        for task_idx in range(self.n_task):
            experts = [expert(x) for expert in self.task_experts[task_idx]] + shared
            expert_tensor = torch.stack(experts, dim=1)
            gate = torch.softmax(self.gates[task_idx](x), dim=-1).unsqueeze(-1)
            outputs.append(torch.sum(gate * expert_tensor, dim=1).unsqueeze(1))
        return ctx.put(self.output_key, torch.cat(outputs, dim=1))
