from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import MLP


@register_skill("shared_bottom_tower")
class SharedBottomTowerSkill(BaseSkill):
    """Shared-bottom MLP for multi-task models."""

    skill_spec = SkillSpec(
        name="shared_bottom_tower",
        category="multitask",
        description="Project a shared input representation through one bottom tower.",
        input_specs=[TensorSpec("shared_input", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("shared_representation", "[batch_size, hidden_dim]")],
        task_types=["multitask"],
        inductive_bias=["hard_parameter_sharing"],
        failure_signatures=["input_dim_mismatch"],
        hyperparameters={"input_dim": "int", "hidden_dims": "list[int]"},
        paper_origin="SharedBottom",
        implementation="torch_rechub.basic.layers.MLP",
    )

    def __init__(self, input_dim: int, hidden_dims: list[int], input_key: str = "shared_input", output_key: str = "shared_representation", activation: str = "relu", dropout: float = 0.0) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        self.input_dim = input_dim
        self.input_key = input_key
        self.output_key = output_key
        self.mlp = MLP(input_dim, dims=hidden_dims, output_layer=False, activation=activation, dropout=dropout)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        return ctx.put(self.output_key, self.mlp(x))
