from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill
from torch_rechub.basic.activation import activation_layer
from torch_rechub.basic.layers import MLP


@register_skill("star_domain_fcn")
class StarDomainFCNSkill(BaseSkill):
    """STAR-style shared-by-domain factorized FCN for all domains."""

    skill_spec = SkillSpec(
        name="star_domain_fcn",
        category="scenario",
        description="Compute STAR domain outputs with shared weights multiplied by domain-specific weights.",
        input_specs=[TensorSpec("flat_embeddings", "[batch_size, input_dim]")],
        output_specs=[TensorSpec("domain_outputs", "[batch_size, domain_num]")],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["shared_domain_weight_factorization", "domain_specific_normalization"],
        failure_signatures=["input_dim_mismatch"],
        hyperparameters={"input_dim": "int", "domain_num": "int", "fcn_dims": "list[int]"},
        implementation="torch_rechub.models.multi_domain.star.Star",
    )

    def __init__(
        self,
        input_dim: int,
        domain_num: int,
        fcn_dims: list[int],
        aux_dims: list[int],
        input_key: str = "flat_embeddings",
        output_key: str = "domain_outputs",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.domain_num = domain_num
        self.input_key = input_key
        self.output_key = output_key
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [input_dim] + list(fcn_dims) + [1]
        self.dn_share_gamma = Parameter(torch.ones(input_dim))
        self.dn_share_bias = Parameter(torch.zeros(input_dim))
        self.eps = eps
        self.auxnet = MLP(input_dim, dims=aux_dims)
        self.relu = activation_layer("relu")
        self.sigmoid = activation_layer("sigmoid")
        self.share_parm_w = torch.nn.ParameterList()
        self.share_parm_b = torch.nn.ParameterList()
        for idx in range(self.layer_num):
            self.share_parm_w.append(Parameter(torch.empty((self.fcn_dim[idx], self.fcn_dim[idx + 1])), requires_grad=True))
            self.share_parm_b.append(Parameter(torch.empty(self.fcn_dim[idx + 1]), requires_grad=True))
        self.domain_specific_dn_gamma = torch.nn.ParameterList()
        self.domain_specific_dn_bias = torch.nn.ParameterList()
        self.domain_specific_w = torch.nn.ModuleList()
        self.domain_specific_b = torch.nn.ModuleList()
        self.domain_specific_bn = torch.nn.ModuleList()
        for _ in range(domain_num):
            self.domain_specific_dn_gamma.append(Parameter(torch.ones(input_dim)))
            self.domain_specific_dn_bias.append(Parameter(torch.zeros(input_dim)))
            layer_weights = torch.nn.ParameterList()
            layer_biases = torch.nn.ParameterList()
            layer_bns = torch.nn.ModuleList()
            for idx in range(self.layer_num):
                layer_weights.append(Parameter(torch.empty((self.fcn_dim[idx], self.fcn_dim[idx + 1])), requires_grad=True))
                layer_biases.append(Parameter(torch.empty(self.fcn_dim[idx + 1]), requires_grad=True))
                layer_bns.append(torch.nn.BatchNorm1d(self.fcn_dim[idx + 1]))
            self.domain_specific_w.append(layer_weights)
            self.domain_specific_b.append(layer_biases)
            self.domain_specific_bn.append(layer_bns)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for idx in range(len(self.share_parm_w)):
            init.kaiming_uniform_(self.share_parm_w[idx])
            init.uniform_(self.share_parm_b[idx], 0, 1)
        for domain_idx in range(len(self.domain_specific_w)):
            for layer_idx in range(len(self.domain_specific_w[domain_idx])):
                init.kaiming_uniform_(self.domain_specific_w[domain_idx][layer_idx])
                init.uniform_(self.domain_specific_b[domain_idx][layer_idx], 0, 1)

    def forward(self, ctx: SkillContext) -> SkillContext:
        emb = ctx.get_required(self.input_key)
        if emb.dim() != 2 or emb.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        aux_out = self.auxnet(emb)
        outputs = []
        mean = emb.mean(dim=0)
        var = ((emb - mean) ** 2).mean(dim=0)
        normalized = (emb - mean) / torch.sqrt(var + self.eps)
        for domain_idx in range(self.domain_num):
            domain_input = (
                (self.dn_share_gamma * self.domain_specific_dn_gamma[domain_idx]) * normalized
                + self.dn_share_bias
                + self.domain_specific_dn_bias[domain_idx]
            )
            for layer_idx in range(self.layer_num):
                weight = self.share_parm_w[layer_idx] * self.domain_specific_w[domain_idx][layer_idx]
                bias = self.share_parm_b[layer_idx] + self.domain_specific_b[domain_idx][layer_idx]
                domain_input = domain_input @ weight + bias
                domain_input = self.domain_specific_bn[domain_idx][layer_idx](domain_input)
                domain_input = self.relu(domain_input)
            outputs.append(self.sigmoid(domain_input + aux_out))
        return ctx.put(self.output_key, torch.cat(outputs, dim=1))


class _DebiasExpertNet(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 16) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.bn(x))


@register_skill("sarnet_expert_mixer")
class SARNETExpertMixerSkill(BaseSkill):
    """SARNET-style shared and domain-specific debias expert mixer."""

    skill_spec = SkillSpec(
        name="sarnet_expert_mixer",
        category="scenario",
        description="Mix SARNET shared and domain-specific debias experts using the active domain id.",
        input_specs=[
            TensorSpec("flat_embeddings", "[batch_size, input_dim]"),
            TensorSpec("domain_id", "[batch_size]", "int64"),
        ],
        output_specs=[TensorSpec("prediction", "[batch_size]")],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["shared_domain_experts", "domain_specific_debiasing"],
        failure_signatures=["input_dim_mismatch", "domain_id_out_of_range"],
        hyperparameters={"input_dim": "int", "domain_num": "int"},
        implementation="torch_rechub.models.multi_domain.sarnet.Sarnet",
    )

    def __init__(
        self,
        input_dim: int,
        domain_num: int,
        domain_shared_expert_num: int = 8,
        domain_specific_expert_num: int = 2,
        expert_dim: int = 16,
        input_key: str = "flat_embeddings",
        domain_key: str = "domain_id",
        output_key: str = "prediction",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.domain_num = domain_num
        self.domain_shared_expert_num = domain_shared_expert_num
        self.domain_specific_expert_num = domain_specific_expert_num
        self.input_key = input_key
        self.domain_key = domain_key
        self.output_key = output_key
        self.domain_weight = torch.nn.ParameterList([Parameter(torch.empty(1, input_dim), requires_grad=True) for _ in range(domain_num)])
        self.domain_bias = torch.nn.ParameterList([Parameter(torch.empty(input_dim), requires_grad=True) for _ in range(domain_num)])
        self.shared_expert = torch.nn.ModuleList([_DebiasExpertNet(input_dim, expert_dim) for _ in range(domain_shared_expert_num)])
        self.domain_specific_expert = torch.nn.ModuleList(
            [torch.nn.ModuleList([_DebiasExpertNet(input_dim, expert_dim) for _ in range(domain_specific_expert_num)]) for _ in range(domain_num)]
        )
        self.gate_net = torch.nn.Linear(input_dim, domain_shared_expert_num + domain_specific_expert_num)
        self.final_mlp = MLP(input_dim=expert_dim, output_layer=True, dims=[32, 32])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for idx in range(len(self.domain_weight)):
            init.xavier_uniform_(self.domain_weight[idx])
            init.uniform_(self.domain_bias[idx], 0, 1)

    def forward(self, ctx: SkillContext) -> SkillContext:
        emb = ctx.get_required(self.input_key)
        domain_id = ctx.get_required(self.domain_key).long()
        if domain_id.dim() == 2 and domain_id.size(1) == 1:
            domain_id = domain_id.squeeze(1)
        if emb.dim() != 2 or emb.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        if domain_id.dim() != 1 or domain_id.size(0) != emb.size(0):
            raise ValueError(f"{self.domain_key} must have shape [batch_size]")
        if domain_id.numel() > 0 and (torch.any(domain_id < 0) or torch.any(domain_id >= self.domain_num)):
            raise ValueError(f"{self.domain_key} values must be in [0, {self.domain_num})")
        domain_inputs = []
        specific_outputs = []
        for domain_idx in range(self.domain_num):
            domain_input = emb * self.domain_weight[domain_idx] + self.domain_bias[domain_idx]
            domain_inputs.append(domain_input)
            specific_outputs.append(
                torch.stack([expert(domain_input) for expert in self.domain_specific_expert[domain_idx]], dim=1)
            )
        domain_mask = F.one_hot(domain_id, num_classes=self.domain_num).to(dtype=emb.dtype, device=emb.device)
        shared_emb = torch.sum(torch.stack(domain_inputs, dim=1) * domain_mask.unsqueeze(-1), dim=1)
        specific_tensor = torch.stack(specific_outputs, dim=1)
        specific_emb = torch.sum(specific_tensor * domain_mask.view(domain_id.size(0), self.domain_num, 1, 1), dim=1)
        shared_expert_out = torch.stack([expert(shared_emb) for expert in self.shared_expert], dim=1)
        expert_out = torch.cat([shared_expert_out, specific_emb], dim=1)
        gate_value = torch.softmax(self.gate_net(shared_emb), dim=-1)
        expert_out = torch.sum(expert_out * gate_value.unsqueeze(-1), dim=1)
        prediction = torch.sigmoid(self.final_mlp(expert_out)).squeeze(-1)
        return ctx.put(self.output_key, prediction)


@register_skill("adaptdhm_cluster_tower")
class AdaptDHMClusterTowerSkill(BaseSkill):
    """AdaptDHM-style latent cluster router with shared/domain weight products."""

    skill_spec = SkillSpec(
        name="adaptdhm_cluster_tower",
        category="scenario",
        description="Route examples to latent clusters and select cluster-specific product-weight FCN outputs.",
        input_specs=[TensorSpec("flat_embeddings", "[batch_size, input_dim]")],
        output_specs=[
            TensorSpec("prediction", "[batch_size]"),
            TensorSpec("cluster_id", "[batch_size]", "int64"),
        ],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["latent_domain_clustering", "shared_specific_weight_products"],
        failure_signatures=["input_dim_mismatch", "cluster_count_mismatch"],
        hyperparameters={"input_dim": "int", "cluster_num": "int", "fcn_dims": "list[int]"},
        implementation="torch_rechub.models.multi_domain.adaptdhm.AdaptDHM",
    )

    def __init__(
        self,
        input_dim: int,
        fcn_dims: list[int],
        cluster_num: int,
        beta: float,
        input_key: str = "flat_embeddings",
        output_key: str = "prediction",
        cluster_key: str = "cluster_id",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.cluster_num = cluster_num
        self.beta = beta
        self.input_key = input_key
        self.output_key = output_key
        self.cluster_key = cluster_key
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dims = [input_dim] + list(fcn_dims) + [1]
        self.register_buffer("center", F.normalize(torch.randn(cluster_num, input_dim), p=2, dim=1))
        self.domain_w = torch.nn.ModuleList()
        self.domain_b = torch.nn.ModuleList()
        for _ in range(cluster_num + 1):
            weights = torch.nn.ParameterList()
            biases = torch.nn.ParameterList()
            for idx in range(self.layer_num):
                weight = Parameter(torch.empty((self.fcn_dims[idx], self.fcn_dims[idx + 1])), requires_grad=True)
                bias = Parameter(torch.empty(self.fcn_dims[idx + 1]), requires_grad=True)
                init.xavier_uniform_(weight, gain=init.calculate_gain("relu"))
                init.normal_(bias, 0, 1e-7)
                weights.append(weight)
                biases.append(bias)
            self.domain_w.append(weights)
            self.domain_b.append(biases)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ctx: SkillContext) -> SkillContext:
        x = ctx.get_required(self.input_key)
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"{self.input_key} must have shape [batch_size, {self.input_dim}]")
        cluster_id = self._cal_router(x).squeeze(1)
        cluster_outputs = []
        for domain_idx in range(1, self.cluster_num + 1):
            present = x
            for layer_idx in range(self.layer_num - 1):
                present = torch.matmul(present, self.domain_w[0][layer_idx] * self.domain_w[domain_idx][layer_idx])
                present = self.relu(present)
            present = torch.matmul(present, self.domain_w[0][self.layer_num - 1] * self.domain_w[domain_idx][self.layer_num - 1])
            cluster_outputs.append(self.sigmoid(present))
        outputs = torch.cat(cluster_outputs, dim=1)
        prediction = torch.gather(outputs, 1, cluster_id.unsqueeze(1)).squeeze(1)
        ctx.put(self.cluster_key, cluster_id)
        return ctx.put(self.output_key, prediction)

    def _cal_router(self, inputs: torch.Tensor, times: int = 3) -> torch.Tensor:
        x = inputs.detach()
        with torch.no_grad():
            repeated = x.unsqueeze(1).repeat(1, self.center.shape[0], 1)
            routing = None
            for _ in range(times):
                scores = (repeated * self.center.unsqueeze(0)).sum(dim=2)
                routing = F.softmax(scores, dim=1)
                if not self.training:
                    break
                cluster_center = (routing.unsqueeze(2) * repeated).sum(dim=0)
                self.center = F.normalize(self.beta * self.center + (1 - self.beta) * cluster_center, p=2, dim=1)
            if self.training:
                scores = (repeated * self.center.unsqueeze(0)).sum(dim=2)
                routing = F.softmax(scores, dim=1)
            return torch.argmax(routing, dim=1).unsqueeze(1)
