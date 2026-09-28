from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.activation import activation_layer
from torch_rechub.basic.layers import GateNU, MLP, Pruner


@register_skill("gate_nu_feature_gate")
class GateNUFeatureGateSkill(BaseSkill):
    """Gate an agnostic representation with scenario-aware GateNU weights."""

    skill_spec = SkillSpec(
        name="gate_nu_feature_gate",
        category="scenario",
        description="Apply an EPNet-style GateNU feature gate conditioned on scenario and agnostic representations.",
        input_specs=[
            TensorSpec("scenario_context", "[batch_size, scenario_dim]"),
            TensorSpec("feature_representation", "[batch_size, feature_dim]"),
        ],
        output_specs=[
            TensorSpec("gated_features", "[batch_size, feature_dim]"),
            TensorSpec("gate_values", "[batch_size, feature_dim]"),
        ],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["scenario_conditioned_feature_gating"],
        failure_signatures=["scenario_dim_mismatch", "feature_dim_mismatch"],
        hyperparameters={"scenario_dim": "int", "feature_dim": "int", "gamma": "float"},
        implementation="torch_rechub.basic.layers.GateNU",
    )

    def __init__(
        self,
        scenario_dim: int,
        feature_dim: int,
        scenario_key: str = "scenario_context",
        feature_key: str = "feature_representation",
        output_key: str = "gated_features",
        gate_output_key: str = "gate_values",
        hidden_dim: int | None = None,
        gamma: float = 2.0,
        detach_feature_for_gate: bool = True,
    ) -> None:
        super().__init__()
        self.scenario_dim = scenario_dim
        self.feature_dim = feature_dim
        self.scenario_key = scenario_key
        self.feature_key = feature_key
        self.output_key = output_key
        self.gate_output_key = gate_output_key
        self.detach_feature_for_gate = detach_feature_for_gate
        self.gate = GateNU(scenario_dim + feature_dim, feature_dim, hidden_dim=hidden_dim, gamma=gamma)

    def forward(self, ctx: SkillContext) -> SkillContext:
        scenario = ctx.get_required(self.scenario_key)
        features = ctx.get_required(self.feature_key)
        if scenario.dim() != 2 or scenario.size(-1) != self.scenario_dim:
            raise ValueError(f"{self.scenario_key} must have shape [batch_size, {self.scenario_dim}]")
        if features.dim() != 2 or features.size(-1) != self.feature_dim:
            raise ValueError(f"{self.feature_key} must have shape [batch_size, {self.feature_dim}]")
        gate_features = features.detach() if self.detach_feature_for_gate else features
        gate_values = self.gate(torch.cat((scenario, gate_features), dim=1))
        ctx.put(self.gate_output_key, gate_values)
        return ctx.put(self.output_key, features * gate_values)


class _PPTowerBlock(torch.nn.Module):
    def __init__(self, input_dim: int, fcn_dims: list[int]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dims = [input_dim] + list(fcn_dims)
        self.gate_layers = torch.nn.ModuleList()
        self.mlp_layers = torch.nn.ModuleList()
        for idx in range(len(self.dims) - 1):
            self.mlp_layers.append(MLP(input_dim=self.dims[idx], dims=[self.dims[idx + 1]], output_layer=False))
            self.gate_layers.append(GateNU(self.dims[0], self.dims[idx + 1]))
        self.final_layer = torch.nn.Linear(self.dims[-1], 1)
        self.sigmoid = activation_layer("sigmoid")

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        hidden = gate_input
        for mlp_layer, gate_layer in zip(self.mlp_layers, self.gate_layers):
            hidden = mlp_layer(hidden) * gate_layer(gate_input)
        return self.sigmoid(self.final_layer(hidden))


@register_skill("ppnet_domain_towers")
class PPNetDomainTowersSkill(BaseSkill):
    """Apply PPNet-style gated towers for every domain."""

    skill_spec = SkillSpec(
        name="ppnet_domain_towers",
        category="scenario",
        description="Compute one PPNet personalized tower output per domain from a shared gate input.",
        input_specs=[TensorSpec("gate_input", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("domain_outputs", "[batch_size, domain_num]")],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["domain_specific_towers", "personalized_gate_units"],
        failure_signatures=["input_dim_mismatch", "domain_count_mismatch"],
        hyperparameters={"input_dim": "int", "domain_num": "int", "fcn_dims": "list[int]"},
        implementation="torch_rechub.models.multi_domain.ppnet.PPTowerBlock",
    )

    def __init__(
        self,
        input_dim: int,
        domain_num: int,
        fcn_dims: list[int],
        input_key: str = "gate_input",
        output_key: str = "domain_outputs",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.domain_num = domain_num
        self.input_key = input_key
        self.output_key = output_key
        self.towers = torch.nn.ModuleList([_PPTowerBlock(input_dim, fcn_dims) for _ in range(domain_num)])

    def forward(self, ctx: SkillContext) -> SkillContext:
        gate_input = ctx.get_required(self.input_key)
        if gate_input.dim() != 2 or gate_input.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        outputs = torch.cat([tower(gate_input) for tower in self.towers], dim=1)
        return ctx.put(self.output_key, outputs)


@register_skill("adasparse_pruned_tower")
class AdaSparsePrunedTowerSkill(BaseSkill):
    """AdaSparse-style scenario-conditioned pruning tower."""

    skill_spec = SkillSpec(
        name="adasparse_pruned_tower",
        category="scenario",
        description="Apply scenario-conditioned AdaSparse pruning before and between MLP layers.",
        input_specs=[
            TensorSpec("scenario_context", "[batch_size, scenario_dim]"),
            TensorSpec("agnostic_features", "[batch_size, agnostic_dim]"),
        ],
        output_specs=[
            TensorSpec("logits", "[batch_size, 1]"),
            TensorSpec("prediction", "[batch_size]"),
        ],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["adaptive_sparse_feature_pruning", "scenario_conditioning"],
        failure_signatures=["scenario_dim_mismatch", "agnostic_dim_mismatch"],
        hyperparameters={"scenario_dim": "int", "agnostic_dim": "int", "hidden_dims": "list[int]"},
        implementation="torch_rechub.basic.layers.Pruner",
    )

    def __init__(
        self,
        scenario_dim: int,
        agnostic_dim: int,
        hidden_dims: list[int] | None = None,
        scenario_key: str = "scenario_context",
        agnostic_key: str = "agnostic_features",
        logits_key: str = "logits",
        output_key: str = "prediction",
        form: str = "Fusion",
        epsilon: float = 1e-2,
        beta: float = 2.0,
        alpha: float = 1.0,
        delta_alpha: float = 1e-4,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scenario_dim = scenario_dim
        self.agnostic_dim = agnostic_dim
        self.scenario_key = scenario_key
        self.agnostic_key = agnostic_key
        self.logits_key = logits_key
        self.output_key = output_key
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("delta_alpha", torch.tensor(float(delta_alpha)))
        hidden_dims = list(hidden_dims or [])
        self.pruner_layers = torch.nn.ModuleList([Pruner(scenario_dim, agnostic_dim, form=form, epsilon=epsilon, beta=beta)])
        self.mlp_layers = torch.nn.ModuleList()
        input_dim = scenario_dim + agnostic_dim
        for hidden_dim in hidden_dims:
            self.mlp_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.BatchNorm1d(hidden_dim),
                    activation_layer(activation),
                    torch.nn.Dropout(p=dropout),
                )
            )
            input_dim = hidden_dim
            self.pruner_layers.append(Pruner(scenario_dim, input_dim, form=form, epsilon=epsilon, beta=beta))
        self.final_layer = torch.nn.Linear(input_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ctx: SkillContext) -> SkillContext:
        with torch.no_grad():
            self.alpha += self.delta_alpha
        scenario = ctx.get_required(self.scenario_key)
        agnostic = ctx.get_required(self.agnostic_key)
        if scenario.dim() != 2 or scenario.size(-1) != self.scenario_dim:
            raise ValueError(f"{self.scenario_key} must have shape [batch_size, {self.scenario_dim}]")
        if agnostic.dim() != 2 or agnostic.size(-1) != self.agnostic_dim:
            raise ValueError(f"{self.agnostic_key} must have shape [batch_size, {self.agnostic_dim}]")
        hidden = self.pruner_layers[0](scenario, agnostic, self.alpha) * agnostic
        hidden = torch.cat((scenario, hidden), dim=1)
        for idx, layer in enumerate(self.mlp_layers):
            hidden = layer(hidden)
            hidden = self.pruner_layers[idx + 1](scenario, hidden, self.alpha) * hidden
        logits = self.final_layer(hidden)
        ctx.put(self.logits_key, logits)
        return ctx.put(self.output_key, self.sigmoid(logits).squeeze(-1))
