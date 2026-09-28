from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("task_replication")
class TaskReplicationSkill(BaseSkill):
    """Replicate one shared representation for several task towers."""

    skill_spec = SkillSpec(
        name="task_replication",
        category="multitask",
        description="Expand a shared representation into one identical representation per task.",
        input_specs=[TensorSpec("shared_representation", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("task_representations", "[batch_size, n_task, input_dim]")],
        task_types=["multitask"],
        inductive_bias=["hard_parameter_sharing"],
        failure_signatures=["rank_not_2"],
        hyperparameters={"n_task": "int"},
    )

    def __init__(self, n_task: int, input_key: str = "shared_representation", output_key: str = "task_representations") -> None:
        super().__init__()
        self.n_task = n_task
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2:
            raise ValueError(f"{self.input_key} must have shape [batch_size, input_dim]")
        return ctx.put(self.output_key, x.unsqueeze(1).expand(-1, self.n_task, -1).contiguous())
