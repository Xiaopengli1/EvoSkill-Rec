from __future__ import annotations

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import MLP


@register_skill("shared_mlp_tower")
class SharedMLPTowerSkill(BaseSkill):
    """Apply one MLP tower instance to multiple 2D context tensors."""

    skill_spec = SkillSpec(
        name="shared_mlp_tower",
        category="tower",
        description="Reuse one MLP tower across several inputs, such as positive and negative item branches.",
        input_specs=[TensorSpec("input_keys", "list[[batch_size, input_dim]]")],
        output_specs=[TensorSpec("output_keys", "list[[batch_size, hidden_dim]]")],
        task_types=["matching", "ctr", "ranking"],
        inductive_bias=["shared_tower_weights"],
        failure_signatures=["input_dim_mismatch", "key_count_mismatch"],
        hyperparameters={"input_dim": "int", "hidden_dims": "list[int]"},
        implementation="torch_rechub.basic.layers.MLP",
    )

    def __init__(
        self,
        input_keys: list[str],
        output_keys: list[str],
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        output_layer: bool = False,
    ) -> None:
        super().__init__()
        if not input_keys or len(input_keys) != len(output_keys):
            raise ValueError("input_keys and output_keys must be non-empty lists of the same length")
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        self.input_dim = input_dim
        self.mlp = MLP(input_dim, dims=hidden_dims or [], dropout=dropout, activation=activation, output_layer=output_layer)

    def forward(self, ctx: SkillContext) -> SkillContext:
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            x = ctx.get_required(input_key)
            if x.dim() != 2 or x.size(-1) != self.input_dim:
                raise ValueError(f"{input_key} must have shape [batch_size, {self.input_dim}]")
            ctx.put(output_key, self.mlp(x))
        return ctx
