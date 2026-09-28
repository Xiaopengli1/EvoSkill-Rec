from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.layers import MLP


class _MlpN(torch.nn.Module):
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        layers = []
        for idx in range(len(dims) - 1):
            layers.extend([torch.nn.Linear(dims[idx], dims[idx + 1]), torch.nn.LayerNorm(dims[idx + 1]), torch.nn.ReLU()])
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _ScalarWeight(torch.nn.Module):
    def __init__(self, initial: float) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(initial)))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.value)


@register_skill("m2m_meta_tower")
class M2MMetaTowerSkill(BaseSkill):
    """M2M-style meta attention and meta tower conditioned on a domain embedding."""

    skill_spec = SkillSpec(
        name="m2m_meta_tower",
        category="scenario",
        description="Apply M2M meta attention and meta tower modules over expert outputs conditioned on scenario embeddings.",
        input_specs=[
            TensorSpec("flat_embeddings", "[batch_size, input_dim]"),
            TensorSpec("domain_context", "[batch_size, domain_dim]"),
        ],
        output_specs=[TensorSpec("prediction", "[batch_size]")],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["meta_attention", "scenario_conditioned_tower"],
        failure_signatures=["input_dim_mismatch", "domain_dim_mismatch"],
        hyperparameters={"input_dim": "int", "domain_dim": "int", "num_experts": "int"},
        implementation="torch_rechub.models.multi_domain.m2m.M2M",
    )

    def __init__(
        self,
        input_dim: int,
        domain_dim: int,
        num_experts: int = 4,
        expert_output_size: int = 16,
        transformer_dims: dict | None = None,
        input_key: str = "flat_embeddings",
        domain_key: str = "domain_context",
        output_key: str = "prediction",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.domain_dim = domain_dim
        self.num_experts = num_experts
        self.expert_output_size = expert_output_size
        self.input_key = input_key
        self.domain_key = domain_key
        self.output_key = output_key
        transformer_dims = dict(transformer_dims or {"num_encoder_layers": 2, "num_decoder_layers": 2, "dim_feedforward": 16})
        self.transformer = torch.nn.Transformer(d_model=input_dim, nhead=4, **transformer_dims)
        self.experts = torch.nn.ModuleList(
            [MLP(input_dim, output_layer=False, dims=[expert_output_size], activation="leakyrelu") for _ in range(num_experts)]
        )
        self.task_mlp = MLP(domain_dim, output_layer=False, dims=[expert_output_size], activation="leakyrelu")
        self.scenario_mlp = MLP(domain_dim, output_layer=False, dims=[expert_output_size], activation="leakyrelu")
        self.vw_mlp = MLP(expert_output_size, output_layer=False, dims=[4 * expert_output_size * expert_output_size], activation="leakyrelu")
        self.vb_mlp = MLP(expert_output_size, output_layer=False, dims=[2 * expert_output_size], activation="leakyrelu")
        self.v = torch.nn.Parameter(torch.ones(2 * expert_output_size, 1))
        self.meta_tower_w_mlp = MLP(expert_output_size, output_layer=False, dims=[expert_output_size * expert_output_size], activation="leakyrelu")
        self.meta_tower_b_mlp = MLP(expert_output_size, output_layer=False, dims=[expert_output_size], activation="leakyrelu")
        self.relu = torch.nn.LeakyReLU(0.1)
        self.sigmoid = torch.nn.Sigmoid()
        self.output_mlp = MLP(expert_output_size, dims=[64, 32])

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        domain_context = ctx.get_required(self.domain_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        if domain_context.dim() != 2 or domain_context.size(-1) != self.domain_dim:
            raise ValueError(f"{self.domain_key} must have shape [batch_size, {self.domain_dim}]")
        batch_size = x.size(0)
        transformer_out = self.transformer(x, x)
        scenario_out = self.scenario_mlp(domain_context)
        task_output = self.task_mlp(domain_context)
        expert_output = torch.stack([expert(transformer_out) for expert in self.experts], dim=1)
        meta_input = torch.cat([expert_output, task_output.unsqueeze(1).repeat(1, self.num_experts, 1)], dim=2)
        meta_weight = self.vw_mlp(scenario_out).reshape(batch_size, 2 * self.expert_output_size, 2 * self.expert_output_size)
        meta_bias = self.vb_mlp(scenario_out).unsqueeze(1)
        meta_output = self.relu(torch.matmul(meta_input, meta_weight) + meta_bias)
        meta_output = torch.matmul(meta_output, self.v.unsqueeze(0)).squeeze(2)
        alpha = torch.softmax(meta_output, dim=1).unsqueeze(2)
        routed = torch.sum(alpha * expert_output, dim=1)
        tower_weight = self.meta_tower_w_mlp(scenario_out).reshape(batch_size, self.expert_output_size, self.expert_output_size)
        tower_bias = self.meta_tower_b_mlp(scenario_out).reshape(batch_size, self.expert_output_size)
        output = self.relu(torch.matmul(routed.unsqueeze(1), tower_weight).squeeze(1) + tower_bias + routed)
        return ctx.put(self.output_key, self.sigmoid(self.output_mlp(output)).squeeze(1))


@register_skill("hamur_domain_adapter_tower")
class HamurDomainAdapterTowerSkill(BaseSkill):
    """HAMUR-style domain tower with hypernetwork-generated adapter cells."""

    skill_spec = SkillSpec(
        name="hamur_domain_adapter_tower",
        category="scenario",
        description="Compute per-domain HAMUR adapter tower outputs and select by domain id.",
        input_specs=[
            TensorSpec("flat_embeddings", "[batch_size, input_dim]"),
            TensorSpec("domain_id", "[batch_size]", "int64"),
        ],
        output_specs=[TensorSpec("prediction", "[batch_size]")],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["hypernetwork_adapter", "domain_specific_towers"],
        failure_signatures=["input_dim_mismatch", "domain_id_out_of_range"],
        hyperparameters={"input_dim": "int", "domain_num": "int", "fcn_dims": "list[int]"},
        implementation="torch_rechub.models.multi_domain.hamur.HamurSmall/HamurLarge",
    )

    def __init__(
        self,
        input_dim: int,
        domain_num: int,
        fcn_dims: list[int],
        hyper_dims: list[int],
        k: int,
        input_key: str = "flat_embeddings",
        domain_key: str = "domain_id",
        output_key: str = "prediction",
        adapter_dim: int = 32,
    ) -> None:
        super().__init__()
        if not fcn_dims:
            raise ValueError("fcn_dims must not be empty")
        self.input_dim = input_dim
        self.domain_num = domain_num
        self.input_key = input_key
        self.domain_key = domain_key
        self.output_key = output_key
        self.k = k
        self.adapter_dim = adapter_dim
        dims = [input_dim] + list(fcn_dims)
        self.domain_towers = torch.nn.ModuleList()
        for _ in range(domain_num):
            layers = []
            for idx in range(len(dims) - 1):
                layers.extend([torch.nn.Linear(dims[idx], dims[idx + 1]), torch.nn.BatchNorm1d(dims[idx + 1]), torch.nn.ReLU()])
            self.domain_towers.append(torch.nn.Sequential(*layers))
        hyper_layers = []
        hyper_input_dim = input_dim
        for hidden_dim in list(hyper_dims) + [k * k]:
            hyper_layers.extend([torch.nn.Linear(hyper_input_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(p=0.0)])
            hyper_input_dim = hidden_dim
        self.hyper_net = torch.nn.Sequential(*hyper_layers)
        hidden_dim = dims[-1]
        self.u_down = torch.nn.Parameter(torch.ones(hidden_dim, k))
        self.v_down = torch.nn.Parameter(torch.ones(k, adapter_dim))
        self.u_up = torch.nn.Parameter(torch.ones(adapter_dim, k))
        self.v_up = torch.nn.Parameter(torch.ones(k, hidden_dim))
        self.adapter_bias_down = torch.nn.Parameter(torch.zeros(adapter_dim))
        self.adapter_bias_up = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.gamma = torch.nn.Parameter(torch.ones(hidden_dim))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.eps = 1e-5
        self.final_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, 1) for _ in range(domain_num)])
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        domain_id = ctx.get_required(self.domain_key).long()
        if domain_id.dim() == 2 and domain_id.size(1) == 1:
            domain_id = domain_id.squeeze(1)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        if domain_id.dim() != 1 or domain_id.size(0) != x.size(0):
            raise ValueError(f"{self.domain_key} must have shape [batch_size]")
        if domain_id.numel() > 0 and (torch.any(domain_id < 0) or torch.any(domain_id >= self.domain_num)):
            raise ValueError(f"{self.domain_key} values must be in [0, {self.domain_num})")
        hyper = self.hyper_net(x).reshape(-1, self.k, self.k)
        outputs = []
        for tower, final_layer in zip(self.domain_towers, self.final_layers):
            hidden = tower(x)
            w_down = torch.einsum("mi,bij,jn->bmn", self.u_down, hyper, self.v_down)
            adapter = torch.einsum("bf,bfj->bj", hidden, w_down) + self.adapter_bias_down
            adapter = self.sigmoid(adapter)
            w_up = torch.einsum("mi,bij,jn->bmn", self.u_up, hyper, self.v_up)
            adapter = torch.einsum("bf,bfj->bj", adapter, w_up) + self.adapter_bias_up
            mean = adapter.mean(dim=0)
            var = adapter.var(dim=0)
            adapter = self.gamma * ((adapter - mean) / torch.sqrt(var + self.eps)) + self.bias
            hidden = hidden + adapter
            outputs.append(self.sigmoid(final_layer(hidden)))
        domain_outputs = torch.cat(outputs, dim=1)
        prediction = domain_outputs.gather(1, domain_id.unsqueeze(1)).squeeze(1)
        return ctx.put(self.output_key, prediction)


@register_skill("m3oe_expert_fusion")
class M3oEExpertFusionSkill(BaseSkill):
    """M3oE-style STAR front-end with shared and domain experts."""

    skill_spec = SkillSpec(
        name="m3oe_expert_fusion",
        category="scenario",
        description="Apply M3oE STAR-style domain fusion followed by shared/domain expert fusion.",
        input_specs=[
            TensorSpec("flat_embeddings", "[batch_size, input_dim]"),
            TensorSpec("domain_id", "[batch_size]", "int64"),
        ],
        output_specs=[TensorSpec("prediction", "[batch_size]")],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["star_frontend", "domain_expert_fusion", "mixture_of_experts"],
        failure_signatures=["input_dim_mismatch", "domain_id_out_of_range"],
        hyperparameters={"input_dim": "int", "domain_num": "int", "fcn_dims": "list[int]"},
        implementation="torch_rechub.models.multi_domain.m3oe.M3oE",
    )

    def __init__(
        self,
        input_dim: int,
        domain_num: int,
        fcn_dims: list[int],
        expert_num: int,
        exp_d: float,
        bal_d: float,
        input_key: str = "flat_embeddings",
        domain_key: str = "domain_id",
        output_key: str = "prediction",
    ) -> None:
        super().__init__()
        dims = [input_dim] + list(fcn_dims)
        if len(dims) <= 3:
            raise ValueError("fcn_dims must provide at least three STAR/expert dimensions")
        if dims[2] != dims[3]:
            raise ValueError("M3oE genome expects fcn_dims[1] == fcn_dims[2] for shape alignment")
        self.input_dim = input_dim
        self.domain_num = domain_num
        self.expert_num = expert_num
        self.input_key = input_key
        self.domain_key = domain_key
        self.output_key = output_key
        self.star_dim = dims[:3]
        self.expert_dims = dims[3:]
        self.weight_exp_d = _ScalarWeight(exp_d)
        self.weight_bal_d = _ScalarWeight(bal_d)
        self.skip_conn = _MlpN([self.star_dim[0], self.star_dim[2]])
        self.shared_weight = torch.nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = torch.nn.Parameter(torch.zeros(self.star_dim[1]))
        self.slot_weight = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for _ in range(domain_num)])
        self.slot_bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(self.star_dim[1])) for _ in range(domain_num)])
        self.star_mlp = _MlpN([self.star_dim[1], self.star_dim[2]])
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for weight in self.slot_weight:
            torch.nn.init.xavier_uniform_(weight.data)
        self.expert = torch.nn.ModuleList([_MlpN(self.expert_dims) for _ in range(expert_num)])
        self.domain_expert = torch.nn.ModuleList([_MlpN(self.expert_dims) for _ in range(domain_num)])
        self.gate = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(self.expert_dims[0], expert_num), torch.nn.Softmax(dim=1)) for _ in range(domain_num)]
        )
        self.tower = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.expert_dims[-1], self.expert_dims[-1]),
                    torch.nn.LayerNorm(self.expert_dims[-1]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.expert_dims[-1], 1),
                )
                for _ in range(domain_num)
            ]
        )

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        domain_id = ctx.get_required(self.domain_key).long()
        if domain_id.dim() == 2 and domain_id.size(1) == 1:
            domain_id = domain_id.squeeze(1)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        if domain_id.dim() != 1 or domain_id.size(0) != x.size(0):
            raise ValueError(f"{self.domain_key} must have shape [batch_size]")
        if domain_id.numel() > 0 and (torch.any(domain_id < 0) or torch.any(domain_id >= self.domain_num)):
            raise ValueError(f"{self.domain_key} values must be in [0, {self.domain_num})")
        masks = [domain_id == domain_idx for domain_idx in range(self.domain_num)]
        skip = self.skip_conn(x)
        emb = torch.zeros((x.shape[0], self.star_dim[1]), dtype=x.dtype, device=x.device)
        for domain_idx, (weight, bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            domain_output = torch.matmul(x, weight * self.shared_weight) + bias + self.shared_bias
            emb = torch.where(masks[domain_idx].unsqueeze(1), domain_output, emb)
        emb = self.star_mlp(emb) + skip
        gate_value = [gate(emb.detach()).unsqueeze(1) for gate in self.gate]
        shared_experts = torch.cat([expert(emb).unsqueeze(1) for expert in self.expert], dim=1)
        domain_experts = torch.cat([expert(emb).unsqueeze(1) for expert in self.domain_expert], dim=1)
        balance = self.weight_bal_d()
        weighted_domain = []
        for domain_idx in range(self.domain_num):
            value = balance * domain_experts[:, domain_idx, :]
            for other_idx in range(self.domain_num):
                if other_idx != domain_idx:
                    value = value + (1 - balance) / (self.domain_num - 1) * domain_experts[:, other_idx, :]
            weighted_domain.append(value)
        fused = [
            torch.bmm(gate_value[domain_idx], shared_experts).squeeze(1) + self.weight_exp_d() * weighted_domain[domain_idx]
            for domain_idx in range(self.domain_num)
        ]
        outputs = [torch.sigmoid(self.tower[domain_idx](fused[domain_idx]).squeeze(1)) for domain_idx in range(self.domain_num)]
        result = torch.zeros_like(outputs[0])
        for domain_idx in range(self.domain_num):
            result = torch.where(masks[domain_idx], outputs[domain_idx], result)
        return ctx.put(self.output_key, result)
