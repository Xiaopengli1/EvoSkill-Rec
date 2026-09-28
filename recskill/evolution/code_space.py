from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .evolution_memory import EvolutionMemory
from .genome import SkillGenome
from .open_ended import MACRO_ALLOWED_WIRING, OpenEndedCodingBranch, OpenEndedProposal, normalize_open_ended_structural_scope


MACRO_TRANSFORMATION_CATALOG: list[dict[str, Any]] = [
    {
        "kind": "interaction_replace",
        "wiring": "replace_node",
        "description": "Replace an existing interaction node while preserving its output key contract.",
        "diagnostic_cues": ["interaction_path_is_limited_to_fm_pairwise_terms", "no_cross_network_path_detected"],
        "preconditions": ["interaction_family_transform"],
    },
    {
        "kind": "fusion_replace",
        "wiring": "replace_fusion",
        "description": "Generate a new terminal fusion module that consumes branch logits and reroutes prediction/loss.",
        "diagnostic_cues": ["fixed_additive_fusion_limits_sample_adaptive_branch_weighting"],
        "preconditions": ["fusion_family_transform"],
    },
    {
        "kind": "routing_introduce",
        "wiring": "branch_to_fusion",
        "description": "Introduce a new routed/expert branch and attach its logit to the active terminal fusion.",
        "diagnostic_cues": ["no_explicit_routing_or_expert_selection_path", "fixed_additive_fusion_limits_sample_adaptive_branch_weighting"],
        "preconditions": ["interaction_family_transform"],
    },
    {
        "kind": "tower_replace",
        "wiring": "replace_node",
        "description": "Replace a representation or tower node with a generated architecture-family alternative that preserves downstream output keys.",
        "diagnostic_cues": ["deep_tower_is_plain_mlp", "no_explicit_routing_or_expert_selection_path"],
        "preconditions": ["tower_family_transform"],
    },
    {
        "kind": "embedding_reparameterize",
        "wiring": "insert_between",
        "description": "Insert a generated field-aware embedding reparameterization before interaction/tower consumers.",
        "diagnostic_cues": ["raw_field_embeddings_feed_multiple_paths_without_field_reweighting"],
        "preconditions": ["embedding_family_transform"],
    },
    {
        "kind": "representation_insert",
        "wiring": "insert_between",
        "description": "Insert a generated representation transformer on an existing edge.",
        "diagnostic_cues": ["no_sequence_or_interest_evolution_path_detected", "no_positional_encoding_path_detected"],
        "preconditions": ["interaction_family_transform"],
    },
]

NON_MACRO_CODE_SPACE_WIRINGS = {
    "add_to_fusion",
    "append_to_fusion",
    "reroute_logits",
    "replace_logits",
    "calibrate_logits",
    "post_logit",
}


@dataclass
class CodeSpaceConfig:
    provider: str = "fallback"
    command: str | None = None
    planner_command: str | None = None
    synthesizer_command: str | None = None
    sketch_file: str | None = None
    max_retries: int = 1
    proposals_per_round: int = 2
    sketches_per_round: int = 8
    prompt_path: str | None = None
    sketch_prompt_path: str | None = None
    synthesis_prompt_path: str | None = None
    diversity_lanes: list[str] = field(
        default_factory=lambda: ["interaction", "routing", "fusion", "embedding", "sequence", "regularization"]
    )
    structural_scopes: list[str] = field(default_factory=lambda: ["macro"])
    scope_mix_strategy: str = "balanced"
    scope_providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    macro_requires_live_provider: bool = True
    allow_fallback_templates: bool = True
    reuse_promoted_generated_skills: bool = False
    reuse_promoted_code_skills: bool = False
    require_active_provider: bool = False
    promotion_policy: str = "survivor"
    staging_root: str | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None, *, fallback_budget: int = 2) -> "CodeSpaceConfig":
        raw = raw or {}
        return cls(
            provider=str(raw.get("provider", "fallback")),
            command=raw.get("command"),
            planner_command=raw.get("planner_command"),
            synthesizer_command=raw.get("synthesizer_command"),
            sketch_file=raw.get("sketch_file"),
            max_retries=int(raw.get("max_retries", 1)),
            proposals_per_round=int(raw.get("proposals_per_round", fallback_budget)),
            sketches_per_round=int(
                raw.get("sketches_per_round", max(fallback_budget * 4, fallback_budget))
            ),
            prompt_path=raw.get("prompt_path"),
            sketch_prompt_path=raw.get("sketch_prompt_path"),
            synthesis_prompt_path=raw.get("synthesis_prompt_path"),
            diversity_lanes=list(
                raw.get("diversity_lanes")
                or ["interaction", "routing", "fusion", "embedding", "sequence", "regularization"]
            ),
            structural_scopes=_normalize_structural_scopes(raw.get("structural_scopes") or raw.get("allowed_structural_scopes")),
            scope_mix_strategy=str(raw.get("scope_mix_strategy", "balanced")),
            scope_providers={
                str(scope): dict(scope_raw or {})
                for scope, scope_raw in dict(raw.get("scope_providers") or {}).items()
            },
            macro_requires_live_provider=bool(raw.get("macro_requires_live_provider", True)),
            allow_fallback_templates=bool(raw.get("allow_fallback_templates", True)),
            reuse_promoted_generated_skills=bool(
                raw.get("reuse_promoted_generated_skills", raw.get("reuse_promoted_code_skills", False))
            ),
            reuse_promoted_code_skills=bool(raw.get("reuse_promoted_code_skills", False)),
            require_active_provider=bool(raw.get("require_active_provider", False)),
            promotion_policy=str(raw.get("promotion_policy", "survivor")),
            staging_root=raw.get("staging_root"),
        )


def _normalize_structural_scopes(values: Any = None) -> list[str]:
    if values is None or values == "":
        values = ["macro"]
    if isinstance(values, str):
        values = [item.strip() for item in values.split(",")]
    scopes: list[str] = []
    for value in values:
        if value is None or str(value).strip() == "":
            continue
        scope = normalize_open_ended_structural_scope(value, allow_empty=False)
        if scope not in scopes:
            scopes.append(scope)
    return scopes or ["macro"]


def _normalize_wiring_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("op", "operation", "mode", "type", "wiring"):
            nested = value.get(key)
            if nested:
                return _normalize_wiring_value(nested)
        return ""
    return str(value or "").lower().replace("-", "_")


@dataclass
class OpenEndedCandidateBuildResult:
    proposal: OpenEndedProposal
    ingestion: Any

    @property
    def success(self) -> bool:
        return bool(getattr(self.ingestion, "success", False) and getattr(self.ingestion, "genome", None) is not None)

    def diagnostic(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal.proposal_id,
            "skill_id": getattr(self.ingestion, "skill_id", None) or self.proposal.skill_id,
            "success": bool(getattr(self.ingestion, "success", False)),
            "message": getattr(self.ingestion, "message", ""),
            "code_path": getattr(self.ingestion, "code_path", None),
            "skill_card_path": getattr(self.ingestion, "skill_card_path", None),
            "validation_results": getattr(self.ingestion, "validation_results", {}),
        }


@dataclass
class ArchitectureSketch:
    sketch_id: str
    innovation_lane: str = "interaction"
    target_failure_mode: str = ""
    hypothesis: str = ""
    affected_genome_nodes: list[str] = field(default_factory=list)
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    wiring: str = ""
    core_operator: dict[str, Any] | str | None = None
    complexity_budget: dict[str, Any] = field(default_factory=dict)
    expected_risks: list[str] = field(default_factory=list)
    ablation_plan: dict[str, Any] | str | None = None
    structural_scope: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchitectureSketch":
        sketch_id = data.get("sketch_id") or data.get("id") or data.get("name")
        if not sketch_id:
            raise ValueError("Architecture sketch is missing sketch_id")
        metadata = dict(data.get("metadata") or {})
        structural_scope = normalize_open_ended_structural_scope(
            data.get("structural_scope") or metadata.get("structural_scope") or metadata.get("scope") or "",
            allow_empty=True,
        )
        if structural_scope:
            metadata["structural_scope"] = structural_scope
        return cls(
            sketch_id=str(sketch_id),
            innovation_lane=str(data.get("innovation_lane") or data.get("lane") or "interaction"),
            target_failure_mode=str(data.get("target_failure_mode") or data.get("failure_mode") or ""),
            hypothesis=str(data.get("hypothesis") or data.get("architecture_hypothesis") or ""),
            affected_genome_nodes=list(data.get("affected_genome_nodes") or data.get("affected_nodes") or []),
            input_keys=list(data.get("input_keys") or data.get("inputs") or []),
            output_keys=list(data.get("output_keys") or data.get("outputs") or []),
            wiring=_normalize_wiring_value(data.get("wiring") or data.get("integration") or ""),
            core_operator=data.get("core_operator"),
            complexity_budget=dict(data.get("complexity_budget") or {}),
            expected_risks=list(data.get("expected_risks") or data.get("risks") or []),
            ablation_plan=data.get("ablation_plan"),
            structural_scope=structural_scope,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sketch_id": self.sketch_id,
            "innovation_lane": self.innovation_lane,
            "target_failure_mode": self.target_failure_mode,
            "hypothesis": self.hypothesis,
            "affected_genome_nodes": self.affected_genome_nodes,
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "wiring": self.wiring,
            "core_operator": self.core_operator,
            "complexity_budget": self.complexity_budget,
            "expected_risks": self.expected_risks,
            "ablation_plan": self.ablation_plan,
            "structural_scope": self.structural_scope,
            "metadata": self.metadata,
        }


class CodeSpaceProposalProvider:
    def propose(
        self,
        *,
        parent: SkillGenome,
        diagnosis_report: dict[str, Any],
        evolution_memory: EvolutionMemory | None,
        budget: int,
        round_idx: int,
    ) -> list[OpenEndedProposal]:
        raise NotImplementedError


class CommandProposalProvider(CodeSpaceProposalProvider):
    """Call a local command that reads a JSON prompt on stdin and emits proposal JSON."""

    def __init__(
        self,
        command: str,
        *,
        prompt_path: str | Path | None = None,
        max_retries: int = 1,
        structural_scopes: list[str] | None = None,
        scope_mix_strategy: str = "balanced",
    ) -> None:
        self.command = command
        self.prompt_path = Path(prompt_path) if prompt_path else None
        self.max_retries = max(1, int(max_retries))
        self.structural_scopes = _normalize_structural_scopes(structural_scopes)
        self.scope_mix_strategy = str(scope_mix_strategy or "balanced").lower().replace("-", "_")

    def propose(
        self,
        *,
        parent: SkillGenome,
        diagnosis_report: dict[str, Any],
        evolution_memory: EvolutionMemory | None,
        budget: int,
        round_idx: int,
    ) -> list[OpenEndedProposal]:
        diagnosis_report = {**dict(diagnosis_report or {}), "structural_scopes": self.structural_scopes}
        prompt = build_open_ended_prompt(
            parent=parent,
            diagnosis_report=diagnosis_report,
            evolution_memory=evolution_memory,
            budget=budget,
            round_idx=round_idx,
        )
        if self.prompt_path:
            self.prompt_path.parent.mkdir(parents=True, exist_ok=True)
            self.prompt_path.write_text(json.dumps(prompt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        last_error = None
        for _ in range(self.max_retries):
            try:
                proc = subprocess.run(
                    shlex.split(self.command),
                    input=json.dumps(prompt, sort_keys=True),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip() or f"command exited {proc.returncode}")
                proposals = _filter_proposals_by_scope(_parse_proposals(proc.stdout), self.structural_scopes)
                if self.scope_mix_strategy == "balanced":
                    proposals = _take_balanced_by_scope(proposals, start=0, budget=budget, scopes=self.structural_scopes)
                return proposals[:budget]
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Code-space command provider failed: {last_error}")


class CreativeCommandProposalProvider(CodeSpaceProposalProvider):
    """Two-stage creative provider: architecture sketches first, code proposals second."""

    def __init__(
        self,
        *,
        planner_command: str | None = None,
        synthesizer_command: str | None = None,
        sketches_per_round: int = 8,
        proposals_per_round: int = 2,
        diversity_lanes: list[str] | None = None,
        structural_scopes: list[str] | None = None,
        scope_mix_strategy: str = "balanced",
        sketch_prompt_path: str | Path | None = None,
        synthesis_prompt_path: str | Path | None = None,
        max_retries: int = 1,
    ) -> None:
        if not synthesizer_command:
            raise ValueError("creative_command requires synthesizer_command")
        self.planner_command = planner_command
        self.synthesizer_command = synthesizer_command
        self.sketches_per_round = max(1, int(sketches_per_round))
        self.proposals_per_round = max(1, int(proposals_per_round))
        self.diversity_lanes = list(diversity_lanes or [])
        self.structural_scopes = _normalize_structural_scopes(structural_scopes)
        self.scope_mix_strategy = str(scope_mix_strategy or "balanced").lower().replace("-", "_")
        self.sketch_prompt_path = Path(sketch_prompt_path) if sketch_prompt_path else None
        self.synthesis_prompt_path = Path(synthesis_prompt_path) if synthesis_prompt_path else None
        self.max_retries = max(1, int(max_retries))

    def propose(
        self,
        *,
        parent: SkillGenome,
        diagnosis_report: dict[str, Any],
        evolution_memory: EvolutionMemory | None,
        budget: int,
        round_idx: int,
    ) -> list[OpenEndedProposal]:
        if budget <= 0:
            return []
        design_brief = build_design_brief(
            parent=parent,
            diagnosis_report=diagnosis_report,
            evolution_memory=evolution_memory,
            budget=budget,
            round_idx=round_idx,
            diversity_lanes=self.diversity_lanes,
            structural_scopes=self.structural_scopes,
        )
        sketch_prompt = build_architecture_sketch_prompt(
            design_brief=design_brief,
            sketch_budget=max(self.sketches_per_round, budget),
        )
        if self.sketch_prompt_path:
            _write_prompt(self.sketch_prompt_path, sketch_prompt)
        sketches = self._plan_sketches(sketch_prompt)
        sketches = _filter_sketches_by_scope(sketches, self.structural_scopes)
        selected = select_architecture_sketches(
            sketches,
            parent=parent,
            budget=max(budget, self.proposals_per_round),
            diversity_lanes=self.diversity_lanes,
            structural_scopes=self.structural_scopes,
            scope_mix_strategy=self.scope_mix_strategy,
        )
        if not selected:
            return []
        synthesis_prompt = build_code_synthesis_prompt(
            design_brief=design_brief,
            sketches=selected,
            proposal_budget=budget,
        )
        if self.synthesis_prompt_path:
            _write_prompt(self.synthesis_prompt_path, synthesis_prompt)
        proposals = _run_json_command(
            self.synthesizer_command,
            synthesis_prompt,
            parser=_parse_proposals,
            max_retries=self.max_retries,
            error_prefix="Creative synthesizer command failed",
        )
        proposals = _attach_sketch_metadata_to_proposals(proposals, selected)
        proposals = _filter_proposals_by_scope(proposals, self.structural_scopes)
        if self.scope_mix_strategy == "balanced":
            proposals = _take_balanced_by_scope(proposals, start=0, budget=budget, scopes=self.structural_scopes)
        return proposals[:budget]

    def _plan_sketches(self, sketch_prompt: dict[str, Any]) -> list[ArchitectureSketch]:
        if self.planner_command:
            return _run_json_command(
                self.planner_command,
                sketch_prompt,
                parser=_parse_architecture_sketches,
                max_retries=self.max_retries,
                error_prefix="Creative planner command failed",
            )
        raise ValueError("creative_command requires planner_command")


class DisabledProposalProvider(CodeSpaceProposalProvider):
    def propose(
        self,
        *,
        parent: SkillGenome,
        diagnosis_report: dict[str, Any],
        evolution_memory: EvolutionMemory | None,
        budget: int,
        round_idx: int,
    ) -> list[OpenEndedProposal]:
        return []


class ScopeSplitProposalProvider(CodeSpaceProposalProvider):
    """Route code-space generation through configured structural-scope providers."""

    def __init__(
        self,
        *,
        scope_providers: dict[str, CodeSpaceProposalProvider],
        structural_scopes: list[str],
        scope_mix_strategy: str = "balanced",
        scope_provider_status: dict[str, dict[str, Any]] | None = None,
        require_active_provider: bool = False,
    ) -> None:
        self.scope_providers = dict(scope_providers)
        self.structural_scopes = _normalize_structural_scopes(structural_scopes)
        self.scope_mix_strategy = str(scope_mix_strategy or "balanced").lower().replace("-", "_")
        self.scope_provider_status = dict(scope_provider_status or {})
        self.require_active_provider = bool(require_active_provider)

    def propose(
        self,
        *,
        parent: SkillGenome,
        diagnosis_report: dict[str, Any],
        evolution_memory: EvolutionMemory | None,
        budget: int,
        round_idx: int,
    ) -> list[OpenEndedProposal]:
        if budget <= 0:
            return []
        if not self.scope_providers:
            if self.require_active_provider:
                raise RuntimeError(f"No active code-space providers for scopes {self.structural_scopes}: {self.scope_provider_status}")
            return []
        quotas = _scope_quotas(self.structural_scopes, budget=budget)
        proposals: list[OpenEndedProposal] = []
        for scope in self.structural_scopes:
            quota = quotas.get(scope, 0)
            if quota <= 0 or scope not in self.scope_providers:
                if quota > 0 and self.require_active_provider:
                    status = self.scope_provider_status.get(scope, {})
                    raise RuntimeError(f"Required code-space provider for scope '{scope}' is inactive: {status}")
                continue
            provider = self.scope_providers[scope]
            scope_report = {**dict(diagnosis_report or {}), "structural_scopes": [scope], "target_structural_scope": scope}
            scoped = provider.propose(
                parent=parent,
                diagnosis_report=scope_report,
                evolution_memory=evolution_memory,
                budget=quota,
                round_idx=round_idx,
            )
            proposals.extend(_filter_proposals_by_scope(scoped, [scope])[:quota])
        if self.scope_mix_strategy == "balanced":
            proposals = _interleave_by_scope(proposals, self.structural_scopes)
        return proposals[:budget]


def _scope_quotas(scopes: list[str], *, budget: int) -> dict[str, int]:
    if not scopes or budget <= 0:
        return {}
    quotas = {scope: budget // len(scopes) for scope in scopes}
    for scope in scopes[: budget % len(scopes)]:
        quotas[scope] += 1
    return quotas


def _interleave_by_scope(proposals: list[OpenEndedProposal], scopes: list[str]) -> list[OpenEndedProposal]:
    grouped = _group_by_scope(proposals)
    ordered: list[OpenEndedProposal] = []
    seen_ids: set[str] = set()
    while len(ordered) < len(proposals):
        progressed = False
        for scope in scopes:
            group = grouped.get(scope) or []
            idx = sum(1 for item in ordered if item.structural_scope == scope)
            if idx >= len(group):
                continue
            proposal = group[idx]
            if proposal.proposal_id in seen_ids:
                continue
            ordered.append(proposal)
            seen_ids.add(proposal.proposal_id)
            progressed = True
        if not progressed:
            break
    for proposal in proposals:
        if proposal.proposal_id not in seen_ids:
            ordered.append(proposal)
            seen_ids.add(proposal.proposal_id)
    return ordered


def _build_scope_providers(config: CodeSpaceConfig) -> dict[str, CodeSpaceProposalProvider]:
    providers: dict[str, CodeSpaceProposalProvider] = {}
    for scope in config.structural_scopes:
        raw = dict(config.scope_providers.get(scope) or {})
        skip_reason = _scope_provider_skip_reason(scope, raw, config)
        if skip_reason:
            continue
        provider_name = str(raw.get("provider", "")).lower().replace("-", "_")
        scope_config = CodeSpaceConfig.from_raw(
            {
                **raw,
                "structural_scopes": [scope],
                "scope_mix_strategy": raw.get("scope_mix_strategy", "sequential"),
                "allow_fallback_templates": False,
                "macro_requires_live_provider": config.macro_requires_live_provider,
            },
            fallback_budget=config.proposals_per_round,
        )
        providers[scope] = build_code_space_provider(scope_config)
    return providers


def code_space_provider_diagnostics(config: CodeSpaceConfig) -> dict[str, Any]:
    provider = config.provider.lower().replace("-", "_")
    diagnostics: dict[str, Any] = {
        "provider": config.provider,
        "structural_scopes": list(config.structural_scopes),
        "macro_requires_live_provider": config.macro_requires_live_provider,
        "require_active_provider": config.require_active_provider,
        "reuse_promoted_generated_skills": config.reuse_promoted_generated_skills,
        "reuse_promoted_code_skills": config.reuse_promoted_code_skills,
    }
    if provider in {"scope_split", "split_by_scope", "scoped"}:
        diagnostics["scope_providers"] = {
            scope: _scope_provider_status(scope, dict(config.scope_providers.get(scope) or {}), config)
            for scope in config.structural_scopes
        }
    return diagnostics


def _scope_provider_status(scope: str, raw: dict[str, Any], config: CodeSpaceConfig) -> dict[str, Any]:
    provider_name = str(raw.get("provider", "")).lower().replace("-", "_")
    reason = _scope_provider_skip_reason(scope, raw, config)
    return {
        "provider": provider_name or "",
        "active": not bool(reason),
        "reason": reason or "active",
    }


def _scope_provider_skip_reason(scope: str, raw: dict[str, Any], config: CodeSpaceConfig) -> str:
    if not raw:
        return "not_configured"
    provider_name = str(raw.get("provider", "")).lower().replace("-", "_")
    if not provider_name:
        return "missing_provider"
    if provider_name in {"proposal_file", "file", "offline"}:
        return "proposal_file_provider_removed"
    if scope == "macro" and raw.get("sketch_file"):
        return "sketch_file_provider_removed"
    if scope == "macro" and config.macro_requires_live_provider:
        if provider_name in {"creative_command", "creative", "two_stage_command", "architecture_command"}:
            if not (raw.get("planner_command") or raw.get("command")):
                return "macro_live_planner_command_required"
            if not raw.get("synthesizer_command"):
                return "macro_live_synthesizer_command_required"
        if provider_name in {"command", "llm_command", "llm"} and not raw.get("command"):
            return "macro_live_command_required"
    elif provider_name in {"command", "llm_command", "llm"} and not raw.get("command"):
        return "missing_command"
    if provider_name in {"creative_command", "creative", "two_stage_command", "architecture_command"}:
        has_planner = bool(raw.get("planner_command") or raw.get("command"))
        has_synthesizer = bool(raw.get("synthesizer_command"))
        if not has_planner:
            return "missing_planner_command"
        if not has_synthesizer:
            return "missing_synthesizer_command"
    return ""


def build_code_space_provider(config: CodeSpaceConfig) -> CodeSpaceProposalProvider:
    provider = config.provider.lower().replace("-", "_")
    if provider in {"scope_split", "split_by_scope", "scoped"}:
        scope_providers = _build_scope_providers(config)
        return ScopeSplitProposalProvider(
            scope_providers=scope_providers,
            structural_scopes=config.structural_scopes,
            scope_mix_strategy=config.scope_mix_strategy,
            scope_provider_status=code_space_provider_diagnostics(config).get("scope_providers"),
            require_active_provider=config.require_active_provider,
        )
    if provider in {"proposal_file", "file", "offline"}:
        raise ValueError(
            "code-space no longer supports proposal_file/offline replay; "
            "use provider='creative_command' or provider='command' so proposals are generated from the current evolution state."
        )
    if provider in {"command", "llm_command", "llm"}:
        if not config.command:
            raise ValueError("code_space.command is required for command provider")
        return CommandProposalProvider(
            config.command,
            prompt_path=config.prompt_path,
            max_retries=config.max_retries,
            structural_scopes=config.structural_scopes,
            scope_mix_strategy=config.scope_mix_strategy,
        )
    if provider in {"creative_command", "creative", "two_stage_command", "architecture_command"}:
        planner_command = config.planner_command or config.command
        synthesizer_command = config.synthesizer_command
        if "macro" in _normalize_structural_scopes(config.structural_scopes) and config.sketch_file:
            raise ValueError("macro code-space requires live planner_command; sketch_file/offline sketches are not supported")
        if not planner_command:
            raise ValueError("code_space.planner_command is required for creative_command provider")
        if not synthesizer_command:
            raise ValueError("code_space.synthesizer_command is required for creative_command provider")
        return CreativeCommandProposalProvider(
            planner_command=planner_command,
            synthesizer_command=synthesizer_command,
            sketches_per_round=config.sketches_per_round,
            proposals_per_round=config.proposals_per_round,
            diversity_lanes=config.diversity_lanes,
            structural_scopes=config.structural_scopes,
            scope_mix_strategy=config.scope_mix_strategy,
            sketch_prompt_path=config.sketch_prompt_path or config.prompt_path,
            synthesis_prompt_path=config.synthesis_prompt_path,
            max_retries=config.max_retries,
        )
    if provider in {"disabled", "none"}:
        return DisabledProposalProvider()
    if provider == "fallback":
        return DisabledProposalProvider()
    raise ValueError(f"Unknown code-space provider: {config.provider}")


def ingest_open_ended_proposals(
    *,
    proposals: list[OpenEndedProposal],
    parent: SkillGenome,
    output_dir: str | Path,
    memory: EvolutionMemory | None,
    repo_root: str | Path | None = None,
    staging_root: str | Path | None = None,
) -> list[OpenEndedCandidateBuildResult]:
    return [item for item in inspect_open_ended_proposal_ingestions(
        proposals=proposals,
        parent=parent,
        output_dir=output_dir,
        memory=memory,
        repo_root=repo_root,
        staging_root=staging_root,
    ) if item.success]


def inspect_open_ended_proposal_ingestions(
    *,
    proposals: list[OpenEndedProposal],
    parent: SkillGenome,
    output_dir: str | Path,
    memory: EvolutionMemory | None,
    repo_root: str | Path | None = None,
    staging_root: str | Path | None = None,
) -> list[OpenEndedCandidateBuildResult]:
    output_dir = Path(output_dir)
    staging_root = Path(staging_root) if staging_root is not None else output_dir / "generated_skill_staging"
    branch = OpenEndedCodingBranch(repo_root=repo_root, generated_root=staging_root, memory=memory)
    results: list[OpenEndedCandidateBuildResult] = []
    for proposal in proposals:
        ingestion = branch.ingest(proposal, parent)
        results.append(OpenEndedCandidateBuildResult(proposal=proposal, ingestion=ingestion))
    return results


def build_open_ended_prompt(
    *,
    parent: SkillGenome,
    diagnosis_report: dict[str, Any],
    evolution_memory: EvolutionMemory | None,
    budget: int,
    round_idx: int,
) -> dict[str, Any]:
    records = evolution_memory.all_records()[-20:] if evolution_memory else []
    architecture_profile = _genome_architecture_profile(parent)
    structural_scopes = _normalize_structural_scopes(diagnosis_report.get("structural_scopes") if diagnosis_report else None)
    return {
        "instruction": (
            "Generate safe local PyTorch OpenEndedProposal JSON objects for CTR model evolution from the retained parent genome. "
            "Return only JSON with a top-level proposals list. Code must not use file IO, network, subprocess, eval, or exec. "
            "Generated CTR skills should be reusable across datasets: use symbolic tensor shapes such as "
            "[batch_size, num_fields, embedding_dim] for field_embeddings and [batch_size, flat_input_dim] for flat_embeddings; "
            "do not hard-code dataset-specific field counts or flat dimensions in signatures or init_params. "
            "This is macro-only code-space evolution: every proposal must be an architecture-level transformation, "
            "must set structural_scope='macro', must include proposal.metadata.macro_judgment, and must make an open-ended "
            "model change that alters topology, interaction family, routing/fusion, embedding reparameterization, or a core "
            "representation pathway. Do not emit calibration-only, scalar-only, or residual-only local edits."
        ),
        "round_idx": round_idx,
        "budget": budget,
        "diagnosis_report": diagnosis_report,
        "available_context_keys": sorted(set(parent.constraints.required_inputs) | parent.produced_keys()),
        "current_architecture_profile": architecture_profile,
        "available_macro_transformations": available_macro_transformations(architecture_profile),
        "macro_proposal_requirements": _macro_proposal_requirements(),
        "parent_genome": parent.to_dict(),
        "recent_memory": [record.to_dict() for record in records],
        "required_schema": open_ended_proposal_schema(),
        "portability_requirements": _generated_skill_portability_requirements(),
        "structural_scope_policy": _structural_scope_policy(structural_scopes),
        "macro_judgment_requirements": _macro_judgment_requirements(),
    }


def open_ended_proposal_schema() -> dict[str, Any]:
    return {
        "proposal_id": "stable_unique_id",
        "proposal_type": "NEW_SKILL_INVENTION|NEW_BRANCH_DESIGN|NEW_EMBEDDING_DESIGN|NEW_FUSION_DESIGN|NEW_INTERACTION_DESIGN|NEW_SEQUENCE_DESIGN|NEW_ROUTING_OR_GATING_DESIGN",
        "structural_scope": "macro",
        "target_failure_mode": "failure being addressed",
        "architecture_hypothesis": "short hypothesis",
        "affected_genome_nodes": ["node_id to connect from"],
        "code": "safe Python module defining one nn.Module or BaseSkill-compatible class",
        "expected_input_signature": [{"name": "input_key", "shape": ["batch_size", 1], "dtype": "float32"}],
        "expected_output_signature": [{"name": "output_key", "shape": ["batch_size", 1], "dtype": "float32"}],
        "class_name": "ClassName",
        "skill_id": "generated_skill_id",
        "task_types": ["ctr"],
        "init_params": {},
        "metadata": {
            "node_id": "new_node_id",
            "structural_scope": "macro",
            "wiring": "replace_fusion|replace_node|insert_between|branch_to_fusion",
            "output_key": "optional generated output key",
            "old_logits_key": "optional replaced logits key for reroute/replace_fusion",
            "target_node_id": "required for replace_node",
            "upstream_node_id": "required for insert_between or optional branch anchor",
            "downstream_node_id": "required for insert_between",
            "edges": [],
            "src_output_key": "optional source key",
            "dst_input_key": "optional destination key",
            "macro_judgment": {
                "current_architecture_family": "required for macro proposals",
                "diagnosed_limitation": "required for macro proposals",
                "target_architecture_transformation": "required for macro proposals",
                "why_local_edit_is_insufficient": "required for macro proposals",
                "parent_evidence": ["specific nodes/edges/keys/metrics from the parent that motivate the change"],
                "proposal_insight": "the non-obvious modeling insight behind this macro change",
                "preconditions": {"met": [], "missing": []},
                "wiring_plan": "required for macro proposals",
                "ablation_plan": "required for macro proposals",
            },
        },
    }


def build_design_brief(
    *,
    parent: SkillGenome,
    diagnosis_report: dict[str, Any],
    evolution_memory: EvolutionMemory | None,
    budget: int,
    round_idx: int,
    diversity_lanes: list[str] | None = None,
    structural_scopes: list[str] | None = None,
) -> dict[str, Any]:
    records = evolution_memory.all_records()[-30:] if evolution_memory else []
    recent = [record.to_dict() for record in records]
    produced_keys = sorted(parent.produced_keys())
    required_inputs = sorted(set(parent.constraints.required_inputs))
    available_keys = sorted(set(required_inputs) | set(produced_keys))
    parent_skill_ids = [node.skill_id for node in parent.nodes]
    architecture_profile = _genome_architecture_profile(parent)
    scopes = _normalize_structural_scopes(structural_scopes)
    return {
        "round_idx": round_idx,
        "proposal_budget": budget,
        "diagnosis_report": diagnosis_report,
        "available_context_keys": available_keys,
        "required_inputs": required_inputs,
        "produced_keys": produced_keys,
        "current_logits_key": _current_logits_key(parent),
        "parent_node_summaries": [
            {
                "node_id": node.node_id,
                "skill_id": node.skill_id,
                "category": node.category,
                "input_keys": node.input_keys,
                "output_keys": node.output_keys,
                "source": node.source,
            }
            for node in parent.nodes
        ],
        "parent_skill_ids": parent_skill_ids,
        "current_architecture_profile": architecture_profile,
        "parent_genome": parent.to_dict(),
        "recent_memory": recent,
        "recent_successful_generated_skills": [
            record
            for record in recent
            if str(record.get("status", "")).lower() in {"validated", "promoted", "success"}
            and str(record.get("mutation_type", "")).lower().startswith(("new_", "code_", "local_"))
        ],
        "recent_failed_generated_skills": [
            record
            for record in recent
            if str(record.get("status", "")).lower() in {"failed", "rejected", "promotion_failed"}
        ],
        "diversity_lanes": list(diversity_lanes or []),
        "structural_scopes": scopes,
        "design_constraints": {
            "task_type": "ctr",
            "safe_code_imports": ["torch", "torch.nn", "torch.nn.functional", "math", "typing", "dataclasses"],
            "forbidden_capabilities": ["file_io", "network", "subprocess", "eval", "exec", "environment_access"],
            "module_interface": "forward(inputs: dict[str, Tensor]) -> dict[str, Tensor]",
            "preferred_wiring_modes": ["replace_node", "insert_between", "branch_to_fusion", "replace_fusion"],
            "allowed_macro_wiring_modes": sorted(MACRO_ALLOWED_WIRING),
            "available_macro_transformations": available_macro_transformations(architecture_profile),
            "allowed_structural_scopes": scopes,
            "structural_scope_definitions": {
                "macro": "real-time architecture-level transformation chosen from the current genome diagnosis, changing a representational mechanism, interaction family, sequence model, positional encoding, routing/expert system, or other core architecture family",
            },
            "macro_generation_policy": (
                "Do not replay a predefined macro template or an offline sketch. Inspect the parent_genome and "
                "current_architecture_profile, identify a concrete architectural limitation, decide whether a macro "
                "change is warranted instead of local refinement, then propose a next architecture-family transformation "
                "with satisfied preconditions, wiring, and ablation rationale. Use only available_context_keys unless "
                "the proposal explicitly introduces and validates the required upstream representation."
            ),
            "macro_only_policy": (
                "All selected sketches and proposals must be structural_scope='macro'. Do not emit calibration-only, "
                "scalar-only, or residual-only local refinements. Proposals should alter topology, interaction family, "
                "fusion/routing mechanism, embedding reparameterization, tower family, or an inserted representation pathway "
                "using available macro wiring modes."
            ),
            "macro_judgment_requirements": _macro_judgment_requirements(),
            "macro_proposal_requirements": _macro_proposal_requirements(),
            "max_reasonable_new_parameters": 100000,
        },
    }


def build_architecture_sketch_prompt(*, design_brief: dict[str, Any], sketch_budget: int) -> dict[str, Any]:
    return {
        "instruction": (
            "Create genuinely new CTR architecture sketches from the retained parent genome before writing code. "
            "Return only JSON with a top-level sketches list. Do not write Python code in this stage."
        ),
        "sketch_budget": sketch_budget,
        "design_brief": design_brief,
        "required_schema": {
            "sketch_id": "stable_unique_id",
            "innovation_lane": "interaction|routing|fusion|sequence|embedding|regularization|adapter",
            "target_failure_mode": "specific failure addressed",
            "hypothesis": "why this structure should help",
            "affected_genome_nodes": ["node_id to connect from"],
            "input_keys": ["existing context keys to consume"],
            "output_keys": ["new context keys to produce"],
            "wiring": "replace_fusion|replace_node|insert_between|branch_to_fusion",
            "core_operator": {"type": "short operator family", "components": ["main differentiable parts"]},
            "complexity_budget": {"max_params": 50000, "latency_risk": "low|medium|high"},
            "expected_risks": ["risk"],
            "ablation_plan": "how to remove or simplify this idea",
            "structural_scope": "macro",
            "metadata": {},
        },
        "selection_guidance": {
            "prefer": [
                "ideas tied to diagnosis_report and specific parent nodes/edges/keys",
                "comprehensive macro proposals that evaluate multiple transformation families before choosing one",
                "macro sketches that name the current architecture family, diagnosed limitation, and target transformation",
                "macro sketches that explain why a local refinement is insufficient",
                "different innovation lanes",
                "only macro scopes",
                "explicit tensor inputs and outputs",
                "small residual or gated initialization that preserves baseline behavior",
            ],
            "avoid": [
                "duplicates of recent failed generated skills",
                "generic logit affine calibration only",
                "tiny local patches that leave the same architecture family intact",
                "predefined macro recipes or offline conversions not justified by the current genome",
                "sequence or positional-encoding transformations when the current genome has no sequence/position inputs",
                "large unbounded MLPs without a structural hypothesis",
                "inputs not present in available_context_keys",
            ],
        },
    }


def build_code_synthesis_prompt(
    *,
    design_brief: dict[str, Any],
    sketches: list[ArchitectureSketch],
    proposal_budget: int,
) -> dict[str, Any]:
    return {
        "instruction": (
            "Convert the selected architecture sketches into safe local PyTorch OpenEndedProposal JSON objects. "
            "Return only JSON with a top-level proposals list. Code must not use file IO, network, subprocess, eval, or exec. "
            "For macro sketches, preserve the architectural intent in metadata.macro_judgment and implement a real topology-level "
            "integration plan, not a local logit calibration patch. Make generated CTR skills portable: expected_input_signature "
            "must use symbolic CTR dimensions and init_params should use ${num_fields}, ${embedding_dim}, and ${flat_input_dim} "
            "when a parameter controls field count, embedding width, or flattened input size."
        ),
        "proposal_budget": proposal_budget,
        "design_brief": design_brief,
        "selected_sketches": [sketch.to_dict() for sketch in sketches],
        "required_schema": open_ended_proposal_schema(),
        "code_requirements": {
            "imports": ["torch", "torch.nn", "torch.nn.functional"],
            "interface": "class extends torch.nn.Module and forward accepts a dict named inputs",
            "output": "forward returns dict[str, Tensor] with exactly the sketched output keys",
            "batch_safe": True,
            "initialization": "prefer residual scale/gate initialized near zero or identity",
            "metadata": "include source_sketch_id, innovation_lane, wiring, and output_key",
            "structural_scope": "must be macro and should match the selected sketch",
            "macro_metadata": "for macro proposals include metadata.macro_judgment and wiring fields required by the selected macro operation",
            "portability": "for CTR field/flat inputs avoid hard-coded dataset shapes; support num_fields=2,3,7 where mathematically valid",
        },
        "portability_requirements": _generated_skill_portability_requirements(),
    }


def _macro_judgment_requirements() -> dict[str, Any]:
    return {
        "required_for_structural_scope_macro": [
            "current_architecture_family",
            "diagnosed_limitation",
            "target_architecture_transformation",
            "why_local_edit_is_insufficient",
            "parent_evidence",
            "proposal_insight",
            "preconditions",
            "wiring_plan",
            "ablation_plan",
        ],
        "decision_rule": (
            "Choose macro only when the current genome lacks a core representational mechanism or topology that "
            "cannot be addressed by a local calibration, scalar gate, or residual branch."
        ),
        "precondition_rule": (
            "Do not propose sequence-model or positional-encoding transformations unless sequence, mask, "
            "timestamp, history, or position tensors exist in available_context_keys or are introduced by the proposal."
        ),
        "metadata_location": "proposal.metadata.macro_judgment",
    }


def _generated_skill_portability_requirements() -> dict[str, Any]:
    return {
        "goal": "Generated skills should be reusable across CTR datasets with different categorical field counts.",
        "symbolic_input_signatures": {
            "field_embeddings": ["batch_size", "num_fields", "embedding_dim"],
            "flat_embeddings": ["batch_size", "flat_input_dim"],
        },
        "symbolic_init_params": {
            "num_fields": "${num_fields}",
            "embedding_dim": "${embedding_dim}",
            "input_dim": "${flat_input_dim}",
        },
        "code_guidelines": [
            "Derive pair, tuple, or routing indices from runtime num_fields rather than hard-coded dataset schemas.",
            "Avoid constants such as num_fields=7 or input_dim=112 unless metadata explicitly marks the skill dataset_specific.",
            "Gracefully handle small schemas such as num_fields=2 when the architecture has higher-order interactions.",
        ],
        "promotion_rule": "Only symbolic signatures that pass multi-shape portability tests are marked reusable; fixed-shape skills are retained as dataset_specific.",
    }


def _macro_proposal_requirements() -> dict[str, Any]:
    return {
        "macro_only": True,
        "proposal_quality_bar": [
            "Inspect the retained parent genome, not only global failure labels.",
            "Compare plausible macro transformations such as interaction replacement, branch routing, fusion replacement, embedding reparameterization, and tower replacement.",
            "Choose a transformation whose preconditions are met by available parent keys and edges.",
            "Name the exact parent nodes/edges/keys that motivate the proposal.",
            "State why local calibration/residual edits are insufficient.",
            "Provide a concrete wiring mode and the metadata fields required by that mode.",
            "Include ablation and fallback reasoning so failed ideas are informative.",
        ],
        "disallowed_when_macro_only": [
            "LOCAL_CODE_SURGERY",
            "calibration-only logit affine transforms",
            "single scalar gates that leave the topology unchanged",
            "residual-only branches without a new architecture family or routing mechanism",
            "offline macro recipes unrelated to current_architecture_profile",
        ],
    }


def _genome_architecture_profile(genome: SkillGenome) -> dict[str, Any]:
    current_logits = _current_logits_key(genome)
    produced_keys = sorted(genome.produced_keys())
    required_inputs = sorted(set(genome.constraints.required_inputs))
    available_keys = sorted(set(required_inputs) | set(produced_keys))
    nodes = [
        {
            "node_id": node.node_id,
            "skill_id": node.skill_id,
            "category": node.category,
            "input_keys": list(node.input_keys),
            "output_keys": list(node.output_keys),
        }
        for node in genome.nodes
    ]
    fusion_nodes = [node for node in nodes if _node_matches(node, ["fusion"])]
    interaction_nodes = [
        node
        for node in nodes
        if _node_matches(node, ["interaction", "fm", "cross", "bilinear", "autoint", "attention", "transformer"])
    ]
    sequence_nodes = [
        node
        for node in nodes
        if _node_matches(node, ["sequence", "seq", "history", "gru", "dien", "din", "augru", "interest"])
        or any(_key_matches(key, ["sequence", "seq", "history", "hist", "mask", "position", "timestamp"]) for key in node["input_keys"] + node["output_keys"])
    ]
    routing_nodes = [node for node in nodes if _node_matches(node, ["router", "routing", "expert", "gate", "moe"])]
    tower_nodes = [
        node
        for node in nodes
        if _node_matches(node, ["tower", "mlp", "dnn", "deep"])
    ]
    embedding_nodes = [
        node
        for node in nodes
        if _node_matches(node, ["embedding", "embed"])
    ]
    positional_nodes = [
        node
        for node in nodes
        if _node_matches(node, ["position", "positional", "rope", "rotary"])
        or any(_key_matches(key, ["position", "positional", "rope", "rotary"]) for key in node["input_keys"] + node["output_keys"])
    ]
    logit_producers = [
        {
            "node_id": node.node_id,
            "skill_id": node.skill_id,
            "output_keys": [key for key in node.output_keys if _is_logit_like_key(key)],
        }
        for node in genome.nodes
        if any(_is_logit_like_key(key) for key in node.output_keys)
    ]
    fusion_summaries = []
    for node in genome.nodes:
        if _node_matches(
            {
                "node_id": node.node_id,
                "skill_id": node.skill_id,
                "category": node.category,
                "input_keys": node.input_keys,
                "output_keys": node.output_keys,
            },
            ["fusion"],
        ):
            fusion_summaries.append(
                {
                    "node_id": node.node_id,
                    "skill_id": node.skill_id,
                    "input_keys": list(node.params.get("input_keys") or node.input_keys),
                    "output_keys": list(node.output_keys),
                    "params": dict(node.params),
                }
            )
    has_sequence_context = bool(sequence_nodes) or any(
        _key_matches(key, ["sequence", "seq", "history", "hist", "mask", "position", "timestamp"])
        for key in available_keys
    )
    has_position_context = bool(positional_nodes) or any(
        _key_matches(key, ["position", "positional", "timestamp", "time_idx"])
        for key in available_keys
    )
    profile = {
        "node_count": len(genome.nodes),
        "edge_count": len(genome.edges),
        "tags": list(genome.metadata.tags),
        "required_inputs": required_inputs,
        "available_context_keys": available_keys,
        "current_logits_key": current_logits,
        "inferred_architecture_family": _infer_architecture_family(genome),
        "fusion_nodes": fusion_summaries,
        "interaction_nodes": interaction_nodes,
        "sequence_nodes": sequence_nodes,
        "routing_nodes": routing_nodes,
        "tower_nodes": tower_nodes,
        "embedding_nodes": embedding_nodes,
        "positional_encoding_nodes": positional_nodes,
        "logit_producers": logit_producers,
        "diagnostic_cues": [],
        "macro_preconditions": {
            "sequence_family_transform": {
                "met": has_sequence_context,
                "reason": "requires sequence/history/mask/position context such as DIN/DIEN-style behavior tensors",
            },
            "positional_encoding_transform": {
                "met": has_position_context,
                "reason": "requires ordered tokens or position indices before transformations such as absolute position encoding to RoPE",
            },
            "fusion_family_transform": {
                "met": any(len(item.get("input_keys") or []) >= 2 for item in fusion_summaries),
                "reason": "requires multiple branch outputs that can be reweighted, routed, replaced, or reparameterized",
            },
            "interaction_family_transform": {
                "met": "field_embeddings" in available_keys or "flat_embeddings" in available_keys,
                "reason": "requires field or flattened embeddings for new interaction operators",
            },
            "tower_family_transform": {
                "met": bool(tower_nodes) and ("field_embeddings" in available_keys or "flat_embeddings" in available_keys),
                "reason": "requires an existing representation tower and its input representation to replace or reparameterize",
            },
            "embedding_family_transform": {
                "met": "field_embeddings" in available_keys,
                "reason": "requires field embeddings before inserting field-aware reparameterization or normalization",
            },
        },
    }
    profile["diagnostic_cues"] = _architecture_diagnostic_cues(profile)
    return profile


def _architecture_diagnostic_cues(profile: dict[str, Any]) -> list[str]:
    cues: list[str] = []
    fusion_nodes = list(profile.get("fusion_nodes") or [])
    if any(str(node.get("skill_id", "")).lower() == "additive_fusion" for node in fusion_nodes):
        cues.append("fixed_additive_fusion_limits_sample_adaptive_branch_weighting")
    interaction_skill_ids = {
        str(node.get("skill_id", "")).lower()
        for node in profile.get("interaction_nodes") or []
    }
    if interaction_skill_ids <= {"fm_interaction"} and interaction_skill_ids:
        cues.append("interaction_path_is_limited_to_fm_pairwise_terms")
    if not profile.get("routing_nodes"):
        cues.append("no_explicit_routing_or_expert_selection_path")
    if not profile.get("sequence_nodes"):
        cues.append("no_sequence_or_interest_evolution_path_detected")
    if not profile.get("positional_encoding_nodes"):
        cues.append("no_positional_encoding_path_detected")
    if not any("cross" in str(node.get("skill_id", "")).lower() for node in profile.get("interaction_nodes") or []):
        cues.append("no_cross_network_path_detected")
    if any(str(node.get("skill_id", "")).lower() in {"mlp_tower", "shared_mlp_tower"} for node in profile.get("tower_nodes") or []):
        cues.append("deep_tower_is_plain_mlp")
    embedding_consumers: dict[str, int] = {}
    for node in profile.get("interaction_nodes") or []:
        for key in node.get("input_keys") or []:
            if str(key) == "field_embeddings":
                embedding_consumers[key] = embedding_consumers.get(key, 0) + 1
    for node in profile.get("tower_nodes") or []:
        for key in node.get("input_keys") or []:
            if str(key) == "field_embeddings":
                embedding_consumers[key] = embedding_consumers.get(key, 0) + 1
    if embedding_consumers.get("field_embeddings", 0) >= 2 and not profile.get("routing_nodes"):
        cues.append("raw_field_embeddings_feed_multiple_paths_without_field_reweighting")
    return cues


def available_macro_transformations(profile: dict[str, Any]) -> list[dict[str, Any]]:
    cues = set(profile.get("diagnostic_cues") or [])
    preconditions = dict(profile.get("macro_preconditions") or {})
    annotated = []
    for entry in MACRO_TRANSFORMATION_CATALOG:
        required = list(entry.get("preconditions") or [])
        met = []
        missing = []
        for name in required:
            status = preconditions.get(name) or {}
            if bool(status.get("met")):
                met.append(name)
            else:
                missing.append(name)
        cue_overlap = sorted(cues & set(entry.get("diagnostic_cues") or []))
        annotated.append(
            {
                **entry,
                "available": bool(required) and not missing,
                "cue_overlap": cue_overlap,
                "preconditions_status": {"met": met, "missing": missing},
            }
        )
    return annotated


def _infer_architecture_family(genome: SkillGenome) -> str:
    skill_ids = {node.skill_id.lower() for node in genome.nodes}
    categories = {node.category.lower() for node in genome.nodes}
    if {"field_embedding", "fm_interaction", "mlp_tower", "additive_fusion"} <= skill_ids:
        return "deepfm_additive_linear_fm_mlp"
    if any("dien" in skill for skill in skill_ids) or any("augru" in skill for skill in skill_ids):
        return "dien_like_interest_evolution"
    if any("din" in skill for skill in skill_ids):
        return "din_like_target_attention"
    if any("autoint" in skill or "transformer" in skill for skill in skill_ids):
        return "self_attention_interaction"
    if "fusion" in categories:
        return "multi_branch_ctr_fusion"
    return "generic_ctr_genome"


def _node_matches(node: dict[str, Any], tokens: list[str]) -> bool:
    text = " ".join(
        [
            str(node.get("node_id", "")),
            str(node.get("skill_id", "")),
            str(node.get("category", "")),
        ]
    ).lower()
    return any(token in text for token in tokens)


def _key_matches(key: str, tokens: list[str]) -> bool:
    key = str(key).lower()
    return any(token in key for token in tokens)


def _is_logit_like_key(key: str) -> bool:
    key = str(key).lower()
    return "logit" in key or key in {"logits", "fm_output"}


def select_architecture_sketches(
    sketches: list[ArchitectureSketch],
    *,
    parent: SkillGenome,
    budget: int,
    diversity_lanes: list[str] | None = None,
    structural_scopes: list[str] | None = None,
    scope_mix_strategy: str = "balanced",
) -> list[ArchitectureSketch]:
    if budget <= 0:
        return []
    available = set(parent.constraints.required_inputs) | parent.produced_keys()
    parent_skill_ids = {node.skill_id for node in parent.nodes}
    allowed_scopes = _normalize_structural_scopes(structural_scopes)
    seen_fingerprints: set[str] = set()
    scored: list[tuple[int, int, ArchitectureSketch]] = []
    for order, sketch in enumerate(sketches):
        if not sketch.sketch_id:
            continue
        scope = sketch.structural_scope or _infer_structural_scope_for_sketch(sketch)
        scope = normalize_open_ended_structural_scope(scope, allow_empty=False)
        sketch.structural_scope = scope
        sketch.metadata = {**dict(sketch.metadata or {}), "structural_scope": scope}
        if scope not in allowed_scopes:
            continue
        if scope == "macro" and _is_non_macro_code_space_sketch(sketch):
            continue
        input_keys = set(sketch.input_keys)
        if input_keys and not input_keys <= available:
            continue
        if not sketch.output_keys:
            continue
        fingerprint = json.dumps(
            {
                "lane": sketch.innovation_lane,
                "inputs": sorted(sketch.input_keys),
                "outputs": sorted(sketch.output_keys),
                "wiring": sketch.wiring,
                "operator": sketch.core_operator,
            },
            sort_keys=True,
            default=str,
        )
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        score = _score_architecture_sketch(
            sketch,
            available=available,
            parent_skill_ids=parent_skill_ids,
            diversity_lanes=diversity_lanes or [],
        )
        scored.append((score, order, sketch))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered = [item[2] for item in scored]
    if str(scope_mix_strategy or "balanced").lower().replace("-", "_") == "balanced":
        ordered = _take_balanced_sketches_by_scope(
            ordered,
            budget=budget,
            scopes=_normalize_structural_scopes(structural_scopes),
        )
    return _take_diverse_sketches(ordered, budget=budget, diversity_lanes=diversity_lanes or [])


def _score_architecture_sketch(
    sketch: ArchitectureSketch,
    *,
    available: set[str],
    parent_skill_ids: set[str],
    diversity_lanes: list[str],
) -> int:
    score = 0
    if sketch.innovation_lane in diversity_lanes:
        score += 10
    if sketch.hypothesis:
        score += 8
    if sketch.target_failure_mode:
        score += 6
    if sketch.wiring in {"replace_node", "insert_between", "branch_to_fusion", "replace_fusion"}:
        score += 9
    if sketch.structural_scope == "macro":
        score += 8
    if sketch.innovation_lane in {"calibration", "adapter"} and sketch.wiring not in {"replace_node", "insert_between", "branch_to_fusion", "replace_fusion"}:
        score -= 10
    if set(sketch.input_keys) <= available:
        score += 5
    if any("logit" in key.lower() for key in sketch.output_keys):
        score += 3
    operator_text = json.dumps(sketch.core_operator, sort_keys=True, default=str).lower()
    if operator_text and not any(skill.lower() in operator_text for skill in parent_skill_ids):
        score += 4
    try:
        max_params = int((sketch.complexity_budget or {}).get("max_params", 0))
        if 0 < max_params <= 100000:
            score += 3
    except Exception:
        pass
    return score


def _take_diverse_sketches(sketches: list[ArchitectureSketch], *, budget: int, diversity_lanes: list[str]) -> list[ArchitectureSketch]:
    selected: list[ArchitectureSketch] = []
    selected_ids: set[str] = set()
    for lane in diversity_lanes:
        if len(selected) >= budget:
            break
        for sketch in sketches:
            if sketch.sketch_id in selected_ids:
                continue
            if sketch.innovation_lane == lane:
                selected.append(sketch)
                selected_ids.add(sketch.sketch_id)
                break
    for sketch in sketches:
        if len(selected) >= budget:
            break
        if sketch.sketch_id in selected_ids:
            continue
        selected.append(sketch)
        selected_ids.add(sketch.sketch_id)
    return selected


def _filter_sketches_by_scope(sketches: list[ArchitectureSketch], scopes: list[str]) -> list[ArchitectureSketch]:
    allowed = set(_normalize_structural_scopes(scopes))
    filtered = []
    for sketch in sketches:
        scope = sketch.structural_scope or _infer_structural_scope_for_sketch(sketch)
        scope = normalize_open_ended_structural_scope(scope, allow_empty=False)
        sketch.structural_scope = scope
        sketch.metadata = {**dict(sketch.metadata or {}), "structural_scope": scope}
        if scope in allowed:
            if scope == "macro" and _is_non_macro_code_space_sketch(sketch):
                continue
            filtered.append(sketch)
    return filtered


def _filter_proposals_by_scope(proposals: list[OpenEndedProposal], scopes: list[str]) -> list[OpenEndedProposal]:
    allowed = set(_normalize_structural_scopes(scopes))
    filtered = []
    for proposal in proposals:
        scope = proposal.structural_scope or proposal.metadata.get("structural_scope") or _infer_structural_scope_for_proposal(proposal)
        scope = normalize_open_ended_structural_scope(scope, allow_empty=False)
        proposal.structural_scope = scope
        proposal.metadata = {**dict(proposal.metadata or {}), "structural_scope": scope}
        if scope in allowed:
            if scope == "macro" and _is_non_macro_code_space_proposal(proposal):
                continue
            filtered.append(proposal)
    return filtered


def _take_balanced_by_scope(
    proposals: list[OpenEndedProposal],
    *,
    start: int,
    budget: int,
    scopes: list[str],
) -> list[OpenEndedProposal]:
    if budget <= 0:
        return []
    allowed = _normalize_structural_scopes(scopes)
    by_scope = _group_by_scope(proposals)
    active_scopes = [scope for scope in allowed if by_scope.get(scope)]
    if not active_scopes:
        return proposals[:budget]
    offset = max(0, int(start)) // max(1, len(active_scopes))
    selected: list[OpenEndedProposal] = []
    seen_ids: set[str] = set()
    while len(selected) < budget:
        progressed = False
        for scope in active_scopes:
            group = by_scope.get(scope) or []
            scope_count = sum(1 for item in selected if item.structural_scope == scope)
            absolute_idx = offset + scope_count
            if absolute_idx >= len(group):
                continue
            candidate = group[absolute_idx]
            if candidate.proposal_id not in seen_ids:
                selected.append(candidate)
                seen_ids.add(candidate.proposal_id)
                progressed = True
                if len(selected) >= budget:
                    break
        if not progressed:
            break
    if len(selected) < budget:
        for proposal in proposals:
            if proposal.proposal_id in seen_ids:
                continue
            selected.append(proposal)
            seen_ids.add(proposal.proposal_id)
            if len(selected) >= budget:
                break
    return selected


def _take_balanced_sketches_by_scope(sketches: list[ArchitectureSketch], *, budget: int, scopes: list[str]) -> list[ArchitectureSketch]:
    allowed = _normalize_structural_scopes(scopes)
    by_scope: dict[str, list[ArchitectureSketch]] = {scope: [] for scope in allowed}
    for sketch in sketches:
        scope = sketch.structural_scope or _infer_structural_scope_for_sketch(sketch)
        sketch.structural_scope = normalize_open_ended_structural_scope(scope, allow_empty=False)
        by_scope.setdefault(sketch.structural_scope, []).append(sketch)
    selected: list[ArchitectureSketch] = []
    seen_ids: set[str] = set()
    while len(selected) < budget:
        progressed = False
        for scope in allowed:
            group = by_scope.get(scope) or []
            scope_count = sum(1 for item in selected if item.structural_scope == scope)
            if scope_count >= len(group):
                continue
            sketch = group[scope_count]
            if sketch.sketch_id in seen_ids:
                continue
            selected.append(sketch)
            seen_ids.add(sketch.sketch_id)
            progressed = True
            if len(selected) >= budget:
                break
        if not progressed:
            break
    for sketch in sketches:
        if len(selected) >= budget:
            break
        if sketch.structural_scope in allowed and sketch.sketch_id not in seen_ids:
            selected.append(sketch)
            seen_ids.add(sketch.sketch_id)
    return selected


def _group_by_scope(proposals: list[OpenEndedProposal]) -> dict[str, list[OpenEndedProposal]]:
    grouped: dict[str, list[OpenEndedProposal]] = {}
    for proposal in proposals:
        scope = proposal.structural_scope or proposal.metadata.get("structural_scope") or _infer_structural_scope_for_proposal(proposal)
        scope = normalize_open_ended_structural_scope(scope, allow_empty=False)
        proposal.structural_scope = scope
        proposal.metadata = {**dict(proposal.metadata or {}), "structural_scope": scope}
        grouped.setdefault(scope, []).append(proposal)
    return grouped


def _infer_structural_scope_for_sketch(sketch: ArchitectureSketch) -> str:
    return "macro"


def _infer_structural_scope_for_proposal(proposal: OpenEndedProposal) -> str:
    return "macro"


def _is_non_macro_code_space_sketch(sketch: ArchitectureSketch) -> bool:
    wiring = str(sketch.wiring or "").lower().replace("-", "_")
    if wiring in NON_MACRO_CODE_SPACE_WIRINGS:
        return True
    lane = str(sketch.innovation_lane or "").lower().replace("-", "_")
    operator_text = json.dumps(sketch.core_operator or {}, sort_keys=True, default=str).lower()
    return lane == "calibration" and any(token in operator_text for token in ("calibr", "temperature", "affine"))


def _is_non_macro_code_space_proposal(proposal: OpenEndedProposal) -> bool:
    metadata = proposal.metadata or {}
    wiring = str(metadata.get("wiring") or metadata.get("integration") or "").lower().replace("-", "_")
    if wiring in NON_MACRO_CODE_SPACE_WIRINGS:
        return True
    proposal_text = " ".join(
        str(value).lower()
        for value in (
            proposal.proposal_type,
            proposal.target_failure_mode,
            proposal.architecture_hypothesis,
            metadata.get("innovation_lane", ""),
        )
    )
    return "calibration" in proposal_text and wiring not in {"replace_node", "insert_between", "branch_to_fusion", "replace_fusion"}


def _structural_scope_policy(values: Any = None) -> dict[str, Any]:
    scopes = _normalize_structural_scopes(values)
    return {
        "allowed": scopes,
        "definitions": {
            "macro": "runtime architecture-level transformation selected by judging the current genome, such as changing the interaction family, sequence model, positional encoding, routing/expert mechanism, or other core representational pathway",
        },
        "require_each_proposal_to_set_structural_scope": True,
    }


def _write_prompt(path: Path, prompt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_json_command(command: str, prompt: dict[str, Any], *, parser: Any, max_retries: int, error_prefix: str) -> Any:
    last_error = None
    for _ in range(max(1, int(max_retries))):
        try:
            proc = subprocess.run(
                shlex.split(command),
                input=json.dumps(prompt, sort_keys=True),
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or f"command exited {proc.returncode}")
            return parser(proc.stdout)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"{error_prefix}: {last_error}")


def _parse_architecture_sketches(text: str) -> list[ArchitectureSketch]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = yaml.safe_load(stripped)
    return _items_to_architecture_sketches(data)


def _parse_proposals(text: str) -> list[OpenEndedProposal]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = yaml.safe_load(stripped)
    return _items_to_proposals(data)


def _attach_sketch_metadata_to_proposals(proposals: list[OpenEndedProposal], sketches: list[ArchitectureSketch]) -> list[OpenEndedProposal]:
    sketch_by_id = {sketch.sketch_id: sketch for sketch in sketches}
    if not sketch_by_id:
        return proposals
    attached = []
    for idx, proposal in enumerate(proposals):
        metadata = dict(proposal.metadata or {})
        sketch_id = metadata.get("source_sketch_id") or metadata.get("sketch_id")
        sketch = sketch_by_id.get(str(sketch_id)) if sketch_id else None
        if sketch is None and idx < len(sketches):
            sketch = sketches[idx]
        if sketch is not None:
            metadata = {
                **metadata,
                "source_sketch_id": sketch.sketch_id,
                "innovation_lane": sketch.innovation_lane,
                "structural_scope": metadata.get("structural_scope") or proposal.structural_scope or sketch.structural_scope or _infer_structural_scope_for_proposal(proposal),
                "wiring": metadata.get("wiring") or sketch.wiring,
                "output_key": metadata.get("output_key") or (sketch.output_keys[0] if sketch.output_keys else None),
                "creative_sketch": sketch.to_dict(),
            }
            proposal.structural_scope = str(metadata.get("structural_scope") or "")
            proposal.target_failure_mode = proposal.target_failure_mode or sketch.target_failure_mode
            proposal.architecture_hypothesis = proposal.architecture_hypothesis or sketch.hypothesis
            if not proposal.affected_genome_nodes:
                proposal.affected_genome_nodes = list(sketch.affected_genome_nodes)
            if not proposal.expected_risks:
                proposal.expected_risks = list(sketch.expected_risks)
            if proposal.ablation_plan is None:
                proposal.ablation_plan = sketch.ablation_plan
        proposal.metadata = {key: value for key, value in metadata.items() if value is not None}
        if not proposal.structural_scope:
            proposal.structural_scope = _infer_structural_scope_for_proposal(proposal)
            proposal.metadata["structural_scope"] = proposal.structural_scope
        attached.append(proposal)
    return attached


def _current_logits_key(genome: SkillGenome) -> str:
    try:
        loss = genome.get_node("loss")
        if loss.params.get("logits_key"):
            return str(loss.params["logits_key"])
    except Exception:
        pass
    try:
        prediction = genome.get_node("prediction")
        if prediction.params.get("input_key"):
            return str(prediction.params["input_key"])
    except Exception:
        pass
    return "logits"


def _items_to_proposals(data: Any) -> list[OpenEndedProposal]:
    if isinstance(data, dict) and "proposals" in data:
        items = data["proposals"]
    elif isinstance(data, dict) and "open_ended_proposals" in data:
        items = data["open_ended_proposals"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [data]
    else:
        raise TypeError(f"Unsupported proposal payload: {type(data).__name__}")
    return [OpenEndedProposal.from_dict(item) for item in items]


def _items_to_architecture_sketches(data: Any) -> list[ArchitectureSketch]:
    if isinstance(data, dict) and "sketches" in data:
        items = data["sketches"]
    elif isinstance(data, dict) and "architecture_sketches" in data:
        items = data["architecture_sketches"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [data]
    else:
        raise TypeError(f"Unsupported sketch payload: {type(data).__name__}")
    return [ArchitectureSketch.from_dict(item) for item in items]
