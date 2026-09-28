from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import CrossLayer, MLP
from torch_rechub.models.ranking.edcn import BridgeModule, RegulationModule


@register_skill("edcn_bridge_stack")
class EDCNBridgeStackSkill(BaseSkill):
    """EDCN regulation, cross/deep iteration, and bridge sharing stack."""

    skill_spec = SkillSpec(
        name="edcn_bridge_stack",
        category="interaction",
        description="Apply EDCN regulation modules, cross layers, deep layers, and bridge modules.",
        input_specs=[TensorSpec("flat_embeddings", "[batch_size, input_dim]")],
        output_specs=[
            TensorSpec("cross_output", "[batch_size, input_dim]"),
            TensorSpec("deep_output", "[batch_size, input_dim]"),
            TensorSpec("bridge_output", "[batch_size, input_dim]"),
        ],
        task_types=["ctr", "ranking"],
        inductive_bias=["explicit_implicit_interaction_sharing"],
        failure_signatures=["input_dim_mismatch", "bridge_dim_mismatch"],
        hyperparameters={"input_dim": "int", "n_cross_layers": "int", "bridge_type": "str"},
        paper_origin="EDCN",
        implementation="torch_rechub.models.ranking.edcn",
    )

    def __init__(
        self,
        input_dim: int,
        num_fields: int,
        field_dims: list[int],
        n_cross_layers: int,
        hidden_dims: list[int] | None = None,
        bridge_type: str = "hadamard_product",
        use_regulation_module: bool = True,
        temperature: float = 1.0,
        input_key: str = "flat_embeddings",
        cross_output_key: str = "cross_output",
        deep_output_key: str = "deep_output",
        bridge_output_key: str = "bridge_output",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [input_dim, input_dim]
        if hidden_dims[-1] != input_dim:
            raise ValueError("EDCN bridge stack requires hidden_dims[-1] == input_dim")
        if len(field_dims) != num_fields or sum(field_dims) != input_dim:
            raise ValueError("field_dims must match num_fields and sum to input_dim")
        self.input_dim = input_dim
        self.input_key = input_key
        self.cross_output_key = cross_output_key
        self.deep_output_key = deep_output_key
        self.bridge_output_key = bridge_output_key
        self.cross_layers = torch.nn.ModuleList([CrossLayer(input_dim) for _ in range(n_cross_layers)])
        self.bridge_modules = torch.nn.ModuleList([BridgeModule(input_dim, bridge_type) for _ in range(n_cross_layers)])
        self.regulation_modules = torch.nn.ModuleList(
            [RegulationModule(num_fields, field_dims, tau=temperature, use_regulation=use_regulation_module) for _ in range(n_cross_layers)]
        )
        self.mlps = torch.nn.ModuleList(
            [MLP(input_dim, dims=hidden_dims, output_layer=False, activation=activation, dropout=dropout) for _ in range(n_cross_layers)]
        )

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        cross_i, deep_i = self.regulation_modules[0](x)
        cross_0 = cross_i
        bridge_i = x
        for idx, (cross_layer, bridge_module, regulation_module, mlp) in enumerate(
            zip(self.cross_layers, self.bridge_modules, self.regulation_modules, self.mlps)
        ):
            if idx > 0:
                cross_i, deep_i = regulation_module(bridge_i)
            cross_i = cross_i + cross_layer(cross_0, cross_i)
            deep_i = mlp(deep_i)
            bridge_i = bridge_module(cross_i, deep_i)
        ctx.put(self.cross_output_key, cross_i)
        ctx.put(self.deep_output_key, deep_i)
        return ctx.put(self.bridge_output_key, bridge_i)
