from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import MLP


@register_skill("mlp_tower")
class MLPTowerSkill(BaseSkill):
    """Wraps torch_rechub.basic.layers.MLP for reusable deep towers."""

    skill_spec = SkillSpec(
        name="mlp_tower",
        category="tower",
        description="Apply an MLP tower to a 2D representation tensor.",
        input_specs=[TensorSpec("input_key", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("output_key", "[batch_size, hidden_dims[-1]]")],
        task_types=["ctr", "ranking", "matching", "multitask"],
        inductive_bias=["nonlinear_feature_mixing"],
        failure_signatures=["input_dim_mismatch", "batchnorm_singleton_batch"],
        hyperparameters={
            "input_key": "str",
            "output_key": "str",
            "input_dim": "int",
            "hidden_dims": "list[int]",
            "dropout": "float",
            "activation": "str",
        },
        implementation="torch_rechub.basic.layers.MLP",
    )

    def __init__(
        self,
        input_key: str,
        output_key: str,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        output_layer: bool = False,
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.input_dim = input_dim
        self.mlp = MLP(input_dim, dims=hidden_dims or [], dropout=dropout, activation=activation, output_layer=output_layer)

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2:
            raise ValueError(f"{self.input_key} must have shape [batch_size, input_dim]")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected {self.input_key} dim {self.input_dim}, got {x.size(-1)}")
        return ctx.put(self.output_key, self.mlp(x))
