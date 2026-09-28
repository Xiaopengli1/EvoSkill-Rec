from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import MLP


@register_skill("multi_level_cgc")
class MultiLevelCGCSkill(BaseSkill):
    """Wrap PLE's exact multi-level CGC extraction stack."""

    skill_spec = SkillSpec(
        name="multi_level_cgc",
        category="multitask",
        description="Apply the original PLE Customized Gate Control layers across task and shared experts.",
        input_specs=[TensorSpec("shared_input", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("task_representations", "[batch_size, n_task, expert_dim]")],
        task_types=["multitask"],
        inductive_bias=["progressive_layered_extraction", "shared_and_task_specific_experts"],
        failure_signatures=["input_dim_mismatch"],
        hyperparameters={"n_level": "int", "n_task": "int", "n_expert_specific": "int", "n_expert_shared": "int"},
        implementation="torch_rechub.models.multi_task.ple.CGC",
    )

    def __init__(
        self,
        input_dim: int,
        n_task: int,
        n_level: int,
        n_expert_specific: int,
        n_expert_shared: int,
        expert_params: dict,
        input_key: str = "shared_input",
        output_key: str = "task_representations",
        list_output_key: str = "ple_outputs",
    ) -> None:
        super().__init__()
        from torch_rechub.models.multi_task.ple import CGC

        self.input_dim = input_dim
        self.n_task = n_task
        self.n_level = n_level
        self.input_key = input_key
        self.output_key = output_key
        self.list_output_key = list_output_key
        self.layers = torch.nn.ModuleList(
            [CGC(i + 1, n_level, n_task, n_expert_specific, n_expert_shared, input_dim, expert_params) for i in range(n_level)]
        )

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        ple_inputs = [x] * (self.n_task + 1)
        ple_outs = ple_inputs
        for layer in self.layers:
            ple_outs = layer(ple_inputs)
            ple_inputs = ple_outs
        ctx.put(self.list_output_key, ple_outs)
        return ctx.put(self.output_key, torch.stack(ple_outs[:self.n_task], dim=1))


@register_skill("exact_attention_transfer")
class ExactAttentionTransferSkill(BaseSkill):
    """Use AITM's original attention transfer layer between ordered tasks."""

    skill_spec = SkillSpec(
        name="exact_attention_transfer",
        category="multitask",
        description="Transfer information from earlier task towers to later towers with AITM AttentionLayer.",
        input_specs=[TensorSpec("task_representations", "[batch_size, n_task, input_dim]")],
        output_specs=[TensorSpec("task_representations", "[batch_size, n_task, input_dim]")],
        task_types=["multitask"],
        inductive_bias=["ordered_task_dependency", "attention_transfer"],
        failure_signatures=["task_count_mismatch"],
        hyperparameters={"input_dim": "int", "n_task": "int"},
        implementation="torch_rechub.models.multi_task.aitm.AttentionLayer",
    )

    def __init__(
        self,
        input_dim: int,
        n_task: int,
        input_key: str = "task_representations",
        output_key: str = "task_representations",
        use_info_gate: bool = True,
    ) -> None:
        super().__init__()
        from torch_rechub.models.multi_task.aitm import AttentionLayer

        self.input_dim = input_dim
        self.n_task = n_task
        self.input_key = input_key
        self.output_key = output_key
        self.info_gates = torch.nn.ModuleList(
            [MLP(input_dim, output_layer=False, dims=[input_dim]) for _ in range(max(n_task - 1, 0))]
        ) if use_info_gate else None
        self.aits = torch.nn.ModuleList([AttentionLayer(input_dim) for _ in range(max(n_task - 1, 0))])

    def forward(self, ctx: SkillContext) -> SkillContext:
        reps = ctx.get_required(self.input_key)
        if reps.dim() != 3 or reps.size(1) != self.n_task or reps.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.n_task}, {self.input_dim}]")
        outputs = [reps[:, 0, :]]
        for idx in range(1, self.n_task):
            previous = outputs[idx - 1]
            info = self.info_gates[idx - 1](previous) if self.info_gates is not None else previous
            ait_input = torch.cat([reps[:, idx, :].unsqueeze(1), info.unsqueeze(1)], dim=1)
            outputs.append(self.aits[idx - 1](ait_input))
        return ctx.put(self.output_key, torch.stack(outputs, dim=1))
