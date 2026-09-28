from __future__ import annotations

import torch

from recskill.core import BaseSkill, SkillContext, SkillSpec, TensorSpec, register_skill


@register_skill("esmm_chain_head")
class ESMMChainHeadSkill(BaseSkill):
    """Compose CTR, CVR, and CTCVR probabilities for ESMM."""

    skill_spec = SkillSpec(
        name="esmm_chain_head",
        category="multitask",
        description="Build ESMM task outputs where CTCVR is CTR multiplied by CVR.",
        input_specs=[TensorSpec("ctr", "[batch_size, 1]"), TensorSpec("cvr", "[batch_size, 1]")],
        output_specs=[TensorSpec("task_outputs", "[batch_size, 3]")],
        task_types=["multitask"],
        inductive_bias=["conversion_chain_rule"],
        failure_signatures=["task_shape_mismatch"],
        hyperparameters={"ctr_key": "str", "cvr_key": "str"},
        paper_origin="Entire Space Multi-Task Model",
    )

    def __init__(
        self,
        ctr_key: str = "ctr_output",
        cvr_key: str = "cvr_output",
        output_key: str = "task_outputs",
        inputs_are_logits: bool = False,
        output_order: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.ctr_key = ctr_key
        self.cvr_key = cvr_key
        self.output_key = output_key
        self.inputs_are_logits = inputs_are_logits
        self.output_order = output_order or ["ctr", "cvr", "ctcvr"]
        allowed = {"ctr", "cvr", "ctcvr"}
        if set(self.output_order) != allowed or len(self.output_order) != 3:
            raise ValueError("output_order must contain ctr, cvr, and ctcvr exactly once")

    def forward(self, ctx: SkillContext) -> SkillContext:
        ctr = ctx.get_required(self.ctr_key)
        cvr = ctx.get_required(self.cvr_key)
        if self.inputs_are_logits:
            ctr = torch.sigmoid(ctr)
            cvr = torch.sigmoid(cvr)
        if ctr.dim() == 1:
            ctr = ctr.unsqueeze(-1)
        if cvr.dim() == 1:
            cvr = cvr.unsqueeze(-1)
        ctcvr = ctr * cvr
        values = {"ctr": ctr, "cvr": cvr, "ctcvr": ctcvr}
        return ctx.put(self.output_key, torch.cat([values[name] for name in self.output_order], dim=1))
