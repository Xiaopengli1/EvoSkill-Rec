from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("domain_indicator_adapter")
class DomainIndicatorAdapterSkill(BaseSkill):
    """Normalize a multi-domain batch indicator into a reusable domain id key."""

    skill_spec = SkillSpec(
        name="domain_indicator_adapter",
        category="scenario",
        description="Read a domain indicator tensor and expose it as a 1D domain id tensor.",
        input_specs=[TensorSpec("domain_indicator", "[batch_size]", "int64")],
        output_specs=[TensorSpec("domain_id", "[batch_size]", "int64")],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["scenario_conditioning", "samplewise_domain_routing"],
        failure_signatures=["domain_id_rank_mismatch", "domain_id_out_of_range"],
        hyperparameters={"input_key": "str", "output_key": "str", "num_domains": "int | None"},
    )

    def __init__(
        self,
        input_key: str = "domain_indicator",
        output_key: str = "domain_id",
        num_domains: int | None = None,
        detach: bool = True,
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.num_domains = num_domains
        self.detach = detach

    def forward(self, ctx: SkillContext) -> SkillContext:
        domain_id = ctx.get_required(self.input_key).long()
        if domain_id.dim() == 2 and domain_id.size(1) == 1:
            domain_id = domain_id.squeeze(1)
        if domain_id.dim() != 1:
            raise ValueError(f"{self.input_key} must have shape [batch_size] or [batch_size, 1]")
        if self.num_domains is not None and domain_id.numel() > 0:
            if torch.any(domain_id < 0) or torch.any(domain_id >= self.num_domains):
                raise ValueError(f"{self.input_key} values must be in [0, {self.num_domains})")
        if self.detach:
            domain_id = domain_id.detach()
        return ctx.put(self.output_key, domain_id)


@register_skill("domain_select")
class DomainSelectSkill(BaseSkill):
    """Select per-sample outputs from a domain axis using domain ids."""

    skill_spec = SkillSpec(
        name="domain_select",
        category="scenario",
        description="Gather each sample's active domain output from a [batch_size, domain_num, ...] tensor.",
        input_specs=[
            TensorSpec("domain_outputs", "[batch_size, domain_num, ...]"),
            TensorSpec("domain_id", "[batch_size]", "int64"),
        ],
        output_specs=[TensorSpec("prediction", "[batch_size, ...]")],
        task_types=["multi_domain", "ctr", "ranking"],
        inductive_bias=["scenario_conditioning", "samplewise_domain_routing"],
        failure_signatures=["domain_axis_missing", "domain_id_out_of_range", "batch_size_mismatch"],
        hyperparameters={"input_key": "str", "domain_key": "str", "output_key": "str"},
    )

    def __init__(
        self,
        input_key: str = "domain_outputs",
        domain_key: str = "domain_id",
        output_key: str = "prediction",
        squeeze_last_singleton: bool = True,
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.domain_key = domain_key
        self.output_key = output_key
        self.squeeze_last_singleton = squeeze_last_singleton

    def forward(self, ctx: SkillContext) -> SkillContext:
        values = ctx.get_required(self.input_key)
        domain_id = ctx.get_required(self.domain_key).long()
        if domain_id.dim() == 2 and domain_id.size(1) == 1:
            domain_id = domain_id.squeeze(1)
        if domain_id.dim() != 1:
            raise ValueError(f"{self.domain_key} must have shape [batch_size] or [batch_size, 1]")
        if values.dim() < 2:
            raise ValueError(f"{self.input_key} must include a domain axis at dimension 1")
        if values.size(0) != domain_id.size(0):
            raise ValueError(f"{self.input_key} and {self.domain_key} batch sizes must match")
        if domain_id.numel() > 0 and (torch.any(domain_id < 0) or torch.any(domain_id >= values.size(1))):
            raise ValueError(f"{self.domain_key} values must be in [0, {values.size(1)})")

        index_shape = [domain_id.size(0), 1] + [1] * (values.dim() - 2)
        gather_index = domain_id.view(index_shape).expand(-1, 1, *values.shape[2:])
        selected = values.gather(1, gather_index).squeeze(1)
        if self.squeeze_last_singleton and selected.dim() >= 2 and selected.size(-1) == 1:
            selected = selected.squeeze(-1)
        return ctx.put(self.output_key, selected)
