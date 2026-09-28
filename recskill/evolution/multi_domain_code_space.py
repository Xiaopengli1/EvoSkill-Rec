"""Code-space support for multi-domain evolution.

The generic open-ended ingestion path validates generated PyTorch modules and
integrates them into a ``SkillGenome`` DAG. Multi-domain prompts should therefore
be open-ended over the parent DAG while still making executable node/edge edits
that the current ingestion engine can compile and train.
"""

from __future__ import annotations

from typing import Any

from .code_space import (
    CodeSpaceConfig,
    CommandProposalProvider,
    CreativeCommandProposalProvider,
    DisabledProposalProvider,
    ScopeSplitProposalProvider,
    _attach_sketch_metadata_to_proposals,
    _filter_proposals_by_scope,
    _filter_sketches_by_scope,
    _normalize_structural_scopes,
    _parse_proposals,
    _run_json_command,
    _scope_provider_skip_reason,
    _take_balanced_by_scope,
    _write_prompt,
    code_space_provider_diagnostics,
    select_architecture_sketches,
)
from .evolution_memory import EvolutionMemory
from .genome import SkillGenome
from .open_ended import OpenEndedProposal


MULTI_DOMAIN_SUPPORTED_WIRING = {"insert_between", "replace_node"}
MULTI_DOMAIN_OPEN_SEARCH_LANES = (
    "embedding",
    "dense_feature_processing",
    "field_interaction",
    "shared_representation",
    "multi_domain_backbone",
    "domain_adapter",
    "domain_expert_routing",
    "domain_tower_head",
    "domain_selection",
    "regularization_signal",
)


def _multi_domain_runtime_dims(genome: SkillGenome) -> dict[str, int]:
    extras = genome.metadata.extras or {}
    try:
        embedding = genome.get_node("field_embedding")
        num_sparse_fields = len(embedding.params.get("vocab_sizes") or [])
        embedding_dim = int(extras.get("embedding_dim") or embedding.params.get("embedding_dim", 16))
    except Exception:
        num_sparse_fields, embedding_dim = 1, int(extras.get("embedding_dim") or 16)
    try:
        num_dense_fields = int(extras.get("num_dense_fields") or genome.get_node("dense_feature_path").params.get("num_dense") or 0)
    except Exception:
        num_dense_fields = int(extras.get("num_dense_fields") or 0)
    num_fields = int(extras.get("num_fields") or (num_sparse_fields + num_dense_fields))
    domain_num = int(extras.get("domain_num") or _domain_count_from_nodes(genome) or 1)
    flat_input_dim = int(extras.get("flat_input_dim") or max(1, num_fields) * max(1, embedding_dim))
    return {
        "num_fields": max(1, num_fields),
        "num_sparse_fields": max(0, num_sparse_fields),
        "num_dense_fields": max(0, num_dense_fields),
        "embedding_dim": max(1, embedding_dim),
        "flat_input_dim": max(1, flat_input_dim),
        "domain_num": max(1, domain_num),
        "domain_representation_dim": max(1, _domain_representation_dim(genome)),
    }


def _domain_count_from_nodes(genome: SkillGenome) -> int:
    for node_id in ("domain_adapter", "star_domain_fcn", "domain_mmoe_gate", "domain_ple_gate", "domain_replication"):
        try:
            node = genome.get_node(node_id)
        except Exception:
            continue
        for key in ("num_domains", "domain_num", "n_task"):
            if node.params.get(key) is not None:
                return int(node.params[key])
    return 0


def _domain_representation_dim(genome: SkillGenome) -> int:
    try:
        return int(genome.get_node("domain_towers").params.get("input_dim") or 1)
    except Exception:
        return 1


def multi_domain_injection_points(genome: SkillGenome) -> list[dict[str, Any]]:
    dims = _multi_domain_runtime_dims(genome)
    node_ids = genome.node_ids()
    points: list[dict[str, Any]] = []

    field_producer = None
    for edge in genome.edges:
        if edge.dst_node_id == "flatten" and edge.dst_input_key == "field_embeddings":
            field_producer = edge.src_node_id
            break
    if field_producer is not None and "flatten" in node_ids:
        points.append(
            {
                "lane": "field_embedding",
                "wiring": "insert_between",
                "upstream_node_id": field_producer,
                "downstream_node_id": "flatten",
                "tensor_key": "field_embeddings",
                "tensor_shape": ["batch_size", "num_fields", "embedding_dim"],
                "contract": (
                    "Consume field_embeddings [batch_size, num_fields, embedding_dim] and produce the same shape. "
                    "Use this for field-aware scenario reweighting or local feature interaction."
                ),
                "portable": True,
            }
        )

    flat_consumer = None
    for edge in genome.edges:
        if edge.src_node_id == "flatten" and edge.src_output_key == "flat_embeddings":
            flat_consumer = edge.dst_node_id
            break
    if flat_consumer is not None:
        points.append(
            {
                "lane": "shared_representation",
                "wiring": "insert_between",
                "upstream_node_id": "flatten",
                "downstream_node_id": flat_consumer,
                "tensor_key": "flat_embeddings",
                "tensor_shape": ["batch_size", "flat_input_dim"],
                "contract": (
                    "Consume flat_embeddings [batch_size, flat_input_dim] and produce the same shape before "
                    "the multi-domain transform."
                ),
                "portable": True,
            }
        )

    rep_producer = None
    for edge in genome.edges:
        if edge.dst_node_id == "domain_towers" and edge.dst_input_key == "domain_representations":
            rep_producer = edge.src_node_id
            break
    if rep_producer is not None:
        points.append(
            {
                "lane": "domain_representation",
                "wiring": "insert_between",
                "upstream_node_id": rep_producer,
                "downstream_node_id": "domain_towers",
                "tensor_key": "domain_representations",
                "tensor_shape": ["batch_size", "domain_num", dims["domain_representation_dim"]],
                "contract": (
                    "Consume domain_representations [batch_size, domain_num, representation_dim] and produce "
                    "the same shape before the domain towers."
                ),
                "portable": False,
            }
        )

    domain_output_producer = None
    for edge in genome.edges:
        if edge.dst_node_id == "domain_select" and edge.dst_input_key == "domain_outputs":
            domain_output_producer = edge.src_node_id
            break
    if domain_output_producer is not None:
        points.append(
            {
                "lane": "domain_outputs",
                "wiring": "insert_between",
                "upstream_node_id": domain_output_producer,
                "downstream_node_id": "domain_select",
                "tensor_key": "domain_outputs",
                "tensor_shape": ["batch_size", "domain_num"],
                "contract": (
                    "Consume domain_outputs [batch_size, domain_num] and produce the same shape before "
                    "domain_select gathers the active scenario output."
                ),
                "portable": False,
            }
        )
    return points


def multi_domain_architecture_profile(genome: SkillGenome) -> dict[str, Any]:
    dims = _multi_domain_runtime_dims(genome)
    nodes = [
        {
            "node_id": node.node_id,
            "skill_id": node.skill_id,
            "category": node.category,
            "input_keys": list(node.input_keys),
            "output_keys": list(node.output_keys),
            "source": node.source,
        }
        for node in genome.nodes
    ]
    return {
        "task_family": "multi_domain_recommendation",
        "node_count": len(genome.nodes),
        "edge_count": len(genome.edges),
        "tags": list(genome.metadata.tags),
        "runtime_dims": dims,
        "nodes": nodes,
        "tensor_flow": [
            "domain_indicator -> domain_indicator_adapter -> domain_id [B]",
            "sparse_features -> field_embedding -> field_embeddings [B, num_sparse_fields, embedding_dim]",
            "optional dense_values -> dense_feature_path -> field_embeddings [B, num_fields, embedding_dim]",
            "field_embeddings -> flatten -> flat_embeddings [B, flat_input_dim]",
            "flat_embeddings -> <multi-domain transform> -> domain_outputs [B, domain_num]",
            "domain_outputs + domain_id -> domain_select -> prediction [B]",
            "prediction + labels -> bce_loss(from_logits=False) -> loss",
        ],
        "injection_points": multi_domain_injection_points(genome),
    }


def multi_domain_proposal_schema() -> dict[str, Any]:
    return {
        "proposal_id": "stable_unique_id",
        "proposal_type": (
            "NEW_SKILL_INVENTION|NEW_EMBEDDING_DESIGN|NEW_INTERACTION_DESIGN|"
            "NEW_BRANCH_DESIGN|NEW_ROUTING_OR_GATING_DESIGN|LOCAL_CODE_SURGERY"
        ),
        "structural_scope": "macro",
        "target_failure_mode": "multi-domain failure being addressed",
        "architecture_hypothesis": "short hypothesis grounded in the parent genome",
        "affected_genome_nodes": ["parent node ids touched by the edit"],
        "code": "safe Python module defining one nn.Module whose forward(inputs: dict) -> dict",
        "expected_input_signature": [
            {
                "name": "available context key consumed by the generated module",
                "shape": ["symbolic or concrete tensor shape"],
                "dtype": "float32|int64",
            }
        ],
        "expected_output_signature": [
            {
                "name": "generated output key; for replace_node preserve the replaced node's output key names",
                "shape": ["downstream-compatible symbolic or concrete tensor shape"],
                "dtype": "float32|int64",
            }
        ],
        "class_name": "ClassName",
        "skill_id": "generated_multi_domain_skill_id",
        "task_types": ["multi_domain"],
        "init_params": {},
        "metadata": {
            "node_id": "new_node_id",
            "structural_scope": "macro",
            "wiring": "insert_between|replace_node",
            "output_key": "primary generated output key",
            "upstream_node_id": "required for insert_between; any existing producer node on the selected edge",
            "downstream_node_id": "required for insert_between; any existing consumer node on the selected edge",
            "target_node_id": "required for replace_node; any existing non-terminal node whose output contract is preserved",
            "integration_lane": "|".join(MULTI_DOMAIN_OPEN_SEARCH_LANES),
            "macro_judgment": {
                "current_architecture_family": "star/shared_bottom/mmoe/ple",
                "diagnosed_limitation": "what the current domain transform cannot do",
                "target_architecture_transformation": "what the generated skill changes",
                "why_local_edit_is_insufficient": "why a scalar or calibration edit is not enough",
                "parent_evidence": ["specific parent nodes/keys that motivate this"],
                "proposal_insight": "multi-domain modeling insight",
                "preconditions": {"met": [], "missing": []},
                "wiring_plan": "exact insert_between/replace_node wiring and boundary tensor contract",
                "ablation_plan": "how to remove or simplify this",
            },
        },
    }


def build_multi_domain_design_brief(
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
    profile = multi_domain_architecture_profile(parent)
    scopes = _normalize_structural_scopes(structural_scopes)
    return {
        "round_idx": round_idx,
        "proposal_budget": budget,
        "task_family": "multi_domain_recommendation",
        "diagnosis_report": diagnosis_report,
        "available_context_keys": sorted(set(parent.constraints.required_inputs) | parent.produced_keys()),
        "required_inputs": sorted(set(parent.constraints.required_inputs)),
        "produced_keys": sorted(parent.produced_keys()),
        "current_architecture_profile": profile,
        "injection_points": profile["injection_points"],
        "integration_examples": profile["injection_points"],
        "open_search_lanes": list(MULTI_DOMAIN_OPEN_SEARCH_LANES),
        "parent_genome": parent.to_dict(),
        "recent_memory": recent,
        "recent_failed_generated_skills": [
            r for r in recent if str(r.get("status", "")).lower() in {"failed", "rejected", "promotion_failed"}
        ],
        "diversity_lanes": list(diversity_lanes or MULTI_DOMAIN_OPEN_SEARCH_LANES),
        "structural_scopes": scopes,
        "design_constraints": {
            "task_type": "multi_domain",
            "hard_safety_constraints": {
                "safe_code_imports": ["torch", "torch.nn", "torch.nn.functional", "math", "typing", "dataclasses"],
                "forbidden_capabilities": ["file_io", "network", "subprocess", "eval", "exec", "environment_access"],
                "module_interface": "forward(inputs: dict[str, Tensor]) -> dict[str, Tensor]",
                "must_compile_and_train": True,
                "must_not_read_labels_except_loss_or_explicit_regularization_inputs": True,
            },
            "supported_wiring_modes": sorted(MULTI_DOMAIN_SUPPORTED_WIRING),
            "allowed_wiring_modes": sorted(MULTI_DOMAIN_SUPPORTED_WIRING),
            "allowed_structural_scopes": scopes,
            "search_space_policy": (
                "Search is open over the parent multi-domain DAG: generated skills may target embedding, dense "
                "feature processing, field interaction, domain adapters, backbone transforms, expert routing, "
                "domain towers, and domain selection. The listed integration_examples are stable examples, not "
                "an exhaustive target list."
            ),
            "boundary_contract_rule": (
                "Generated skills do not have to preserve every internal hidden shape. They must preserve the "
                "observable contract at the integration boundary: replace_node proposals must produce the target "
                "node's output key names with downstream-compatible tensor shapes; insert_between proposals must "
                "produce a tensor compatible with the downstream node input reached by the selected edge."
            ),
            "current_engine_limits": (
                "The current ingestion engine executes generated nn.Module DAG edits via insert_between or "
                "replace_node. It does not yet apply arbitrary repo patches or rewrite the terminal objective "
                "node in this multi-domain path; proposals needing that should be treated as a future engine "
                "extension rather than emitted as an executable candidate."
            ),
            "portability_rule": (
                "Prefer symbolic init_params such as ${num_fields}, ${embedding_dim}, ${flat_input_dim}, "
                "${domain_num}, and representation dimensions when the edit is intended to survive across "
                "Amazon, MovieLens, or other multi-domain schemas. Dataset-specific skills are allowed only "
                "when metadata states that constraint explicitly."
            ),
            "task_types_rule": "Every proposal must set task_types=['multi_domain'].",
            "scope_policy": (
                "Current structural_scope is macro because the shared OpenEndedProposal engine currently exposes "
                "macro DAG edits. Within macro, prefer substantive architecture changes over scalar-only edits."
            ),
        },
    }


def build_multi_domain_sketch_prompt(*, design_brief: dict[str, Any], sketch_budget: int) -> dict[str, Any]:
    return {
        "instruction": (
            "Create open-ended multi-domain recommendation architecture sketches from the retained parent genome. "
            "Return only JSON with a top-level sketches list. The integration_examples are stable examples, not "
            "the whole search space. A sketch may replace any non-terminal parent node or insert on any existing "
            "edge when the downstream boundary contract remains compatible."
        ),
        "sketch_budget": sketch_budget,
        "design_brief": design_brief,
        "required_schema": {
            "sketch_id": "stable_unique_id",
            "innovation_lane": "|".join(MULTI_DOMAIN_OPEN_SEARCH_LANES),
            "target_failure_mode": "specific multi-domain failure addressed",
            "hypothesis": "why this structure helps scenario transfer or specialization",
            "affected_genome_nodes": ["node_id to connect from"],
            "input_keys": ["existing context keys"],
            "output_keys": ["new or preserved context keys with downstream-compatible shapes"],
            "wiring": "insert_between|replace_node",
            "core_operator": {"type": "operator family", "components": ["main differentiable parts"]},
            "complexity_budget": {"max_params": 50000, "latency_risk": "low|medium|high"},
            "expected_risks": ["risk"],
            "ablation_plan": "how to remove or simplify this idea",
            "structural_scope": "macro",
            "metadata": {
                "integration_lane": "|".join(MULTI_DOMAIN_OPEN_SEARCH_LANES),
                "target_node_id": "for replace_node",
                "upstream_node_id": "for insert_between",
                "downstream_node_id": "for insert_between",
            },
        },
    }


def build_multi_domain_synthesis_prompt(
    *,
    design_brief: dict[str, Any],
    sketches: list[Any],
    proposal_budget: int,
) -> dict[str, Any]:
    return {
        "instruction": (
            "Convert selected multi-domain sketches into safe PyTorch OpenEndedProposal JSON. Return only JSON "
            "with a top-level proposals list. Code must define an nn.Module.forward(inputs) returning a dict. "
            "Use task_types=['multi_domain'] and structural_scope='macro'. Use insert_between or replace_node "
            "with explicit node ids and a downstream-compatible tensor contract. Do not use CTR-only fusion/logit "
            "wiring shortcuts unless the parent genome actually exposes that terminal."
        ),
        "proposal_budget": proposal_budget,
        "design_brief": design_brief,
        "selected_sketches": [sketch.to_dict() for sketch in sketches],
        "required_schema": multi_domain_proposal_schema(),
        "code_requirements": {
            "imports": ["torch", "torch.nn", "torch.nn.functional"],
            "interface": "class extends torch.nn.Module and forward accepts a dict named inputs",
            "output": "forward returns dict[str, Tensor] with the sketched output keys and downstream-compatible shapes",
            "batch_safe": True,
            "initialization": "prefer identity or near-zero residual initialization",
            "metadata": "include source_sketch_id, innovation_lane, integration_lane, wiring, output_key, and node ids",
            "structural_scope": "macro",
            "task_types": "['multi_domain']",
        },
    }


class MultiDomainCommandProposalProvider(CommandProposalProvider):
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
        design_brief = build_multi_domain_design_brief(
            parent=parent,
            diagnosis_report={**dict(diagnosis_report or {}), "structural_scopes": self.structural_scopes},
            evolution_memory=evolution_memory,
            budget=budget,
            round_idx=round_idx,
            structural_scopes=self.structural_scopes,
        )
        prompt = {
            "instruction": (
                "Generate safe local PyTorch OpenEndedProposal JSON objects for multi-domain recommendation "
                "evolution. Search open-endedly over the parent DAG while using task_types=['multi_domain'], "
                "macro scope, and executable insert_between or replace_node wiring with compatible boundary tensors."
            ),
            "round_idx": round_idx,
            "budget": budget,
            "design_brief": design_brief,
            "required_schema": multi_domain_proposal_schema(),
        }
        if self.prompt_path:
            _write_prompt(self.prompt_path, prompt)
        proposals = _run_json_command(
            self.command,
            prompt,
            parser=_parse_proposals,
            max_retries=self.max_retries,
            error_prefix="Multi-domain code-space command provider failed",
        )
        proposals = _filter_proposals_by_scope(proposals, self.structural_scopes)
        if self.scope_mix_strategy == "balanced":
            proposals = _take_balanced_by_scope(proposals, start=0, budget=budget, scopes=self.structural_scopes)
        return proposals[:budget]


class MultiDomainCreativeCommandProposalProvider(CreativeCommandProposalProvider):
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
        design_brief = build_multi_domain_design_brief(
            parent=parent,
            diagnosis_report=diagnosis_report,
            evolution_memory=evolution_memory,
            budget=budget,
            round_idx=round_idx,
            diversity_lanes=self.diversity_lanes,
            structural_scopes=self.structural_scopes,
        )
        sketch_prompt = build_multi_domain_sketch_prompt(
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
        synthesis_prompt = build_multi_domain_synthesis_prompt(
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
            error_prefix="Multi-domain creative synthesizer command failed",
        )
        proposals = _attach_sketch_metadata_to_proposals(proposals, selected)
        return [
            proposal
            for proposal in proposals
            if not self.structural_scopes or proposal.structural_scope in self.structural_scopes
        ][:budget]


def build_multi_domain_code_space_provider(config: CodeSpaceConfig):
    provider = config.provider.lower().replace("-", "_")
    if provider in {"scope_split", "split_by_scope", "scoped"}:
        scope_providers = _build_multi_domain_scope_providers(config)
        return ScopeSplitProposalProvider(
            scope_providers=scope_providers,
            structural_scopes=config.structural_scopes,
            scope_mix_strategy=config.scope_mix_strategy,
            scope_provider_status=code_space_provider_diagnostics(config).get("scope_providers"),
            require_active_provider=config.require_active_provider,
        )
    if provider in {"command", "llm_command", "llm"}:
        if not config.command:
            raise ValueError("code_space.command is required for command provider")
        return MultiDomainCommandProposalProvider(
            config.command,
            prompt_path=config.prompt_path,
            max_retries=config.max_retries,
            structural_scopes=config.structural_scopes,
            scope_mix_strategy=config.scope_mix_strategy,
        )
    if provider in {"creative_command", "creative", "two_stage_command", "architecture_command"}:
        planner_command = config.planner_command or config.command
        synthesizer_command = config.synthesizer_command
        if not planner_command:
            raise ValueError("code_space.planner_command is required for creative_command provider")
        if not synthesizer_command:
            raise ValueError("code_space.synthesizer_command is required for creative_command provider")
        return MultiDomainCreativeCommandProposalProvider(
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
    if provider in {"disabled", "none", "fallback"}:
        return DisabledProposalProvider()
    if provider in {"proposal_file", "file", "offline"}:
        raise ValueError("Multi-domain code-space requires a live command provider; proposal_file is not supported.")
    raise ValueError(f"Unknown multi-domain code-space provider: {config.provider}")


def _build_multi_domain_scope_providers(config: CodeSpaceConfig) -> dict[str, Any]:
    providers: dict[str, Any] = {}
    for scope in config.structural_scopes:
        raw = dict(config.scope_providers.get(scope) or {})
        if _scope_provider_skip_reason(scope, raw, config):
            continue
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
        providers[scope] = build_multi_domain_code_space_provider(scope_config)
    return providers
