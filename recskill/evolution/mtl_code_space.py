"""Code-space (LLM-driven open-ended skill generation) for multi-task evolution.

This is the MTL counterpart of the CTR code-space machinery in
:mod:`recskill.evolution.code_space`. The provider classes, proposal ingestion,
static/shape/portability validation, skill-card consolidation, and promotion in
that module are architecture-agnostic and are reused verbatim. Only the *prompts*
and the *design brief* are CTR-specific, so this module supplies MTL versions.

Key differences from CTR code-space:

* The genome terminal is ``task_tower -> multitask_loss`` producing
  ``task_outputs`` [B, n_task], not ``fusion -> prediction/loss`` producing
  ``logits``. There is no logits key and no fusion node.
* Generated skills are wired with the **topology-generic** modes only:
  ``insert_between`` and ``replace_node`` (both defined in ``open_ended.py`` and
  free of any logits assumption). The logit/fusion wiring modes are excluded.
* The natural injection points are the ``field_embeddings`` edge (field-level
  reparameterization before flattening), the ``flat_embeddings`` edge (a shared
  representation reparameterizer), and the ``task_representations`` edge (a
  per-task representation transformer).
* MTL Code-space is macro-only: generated proposals must justify a field,
  representation, routing, expert, or cross-task architecture transformation.

Portability: ``insert_between`` on ``field_embeddings`` or ``flat_embeddings``
lets generated skills use the same symbolic CTR dimensions (``num_fields`` /
``embedding_dim`` / ``flat_input_dim``) that the shared validation/portability
harness understands, because the MTL genome still carries a
``field_embedding`` node and records dense fields in genome metadata.
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


# MTL restricts generated-skill wiring to the two topology-generic modes.
MTL_ALLOWED_WIRING = {"insert_between", "replace_node"}

# Nodes that must not be replaced (they define the MTL task/loss contract).
MTL_PROTECTED_NODES = {"task_tower", "loss", "field_embedding"}


# ---------------------------------------------------------------------------
# Genome profiling
# ---------------------------------------------------------------------------


def _mtl_runtime_dims(genome: SkillGenome) -> dict[str, int]:
    extras = genome.metadata.extras or {}
    try:
        embedding = genome.get_node("field_embedding")
        vocab_sizes = embedding.params.get("vocab_sizes") or []
        num_sparse_fields = len(vocab_sizes) if vocab_sizes else int(embedding.params.get("num_fields", 3))
        embedding_dim = int(extras.get("embedding_dim") or embedding.params.get("embedding_dim", 8))
    except Exception:
        num_sparse_fields, embedding_dim = 3, int(extras.get("embedding_dim") or 8)
    try:
        num_dense_fields = int(extras.get("num_dense_fields") or genome.get_node("dense_feature_path").params.get("num_dense") or 0)
    except Exception:
        num_dense_fields = int(extras.get("num_dense_fields") or 0)
    num_fields = int(extras.get("num_fields") or (num_sparse_fields + num_dense_fields))
    flat_input_dim = int(extras.get("flat_input_dim") or max(1, num_fields) * max(1, embedding_dim))
    n_task = 0
    for node_id in ("task_tower", "loss"):
        try:
            node = genome.get_node(node_id)
            types = node.params.get("task_types")
            if types:
                n_task = len(types)
                break
        except Exception:
            continue
    return {
        "num_fields": max(1, int(num_fields)),
        "num_sparse_fields": max(0, int(num_sparse_fields)),
        "num_dense_fields": max(0, int(num_dense_fields)),
        "embedding_dim": max(1, int(embedding_dim)),
        "flat_input_dim": max(1, int(flat_input_dim)),
        "n_task": n_task or 2,
    }


def _representation_dim(genome: SkillGenome) -> int | None:
    try:
        return int(genome.get_node("task_tower").params.get("input_dim"))
    except Exception:
        return None


def mtl_injection_points(genome: SkillGenome) -> list[dict[str, Any]]:
    """Edges where a generated skill can be inserted, with shape contracts."""
    node_ids = genome.node_ids()
    dims = _mtl_runtime_dims(genome)
    points: list[dict[str, Any]] = []

    # field_embeddings edge: transform sparse+dense field embeddings before flattening.
    field_producer = None
    for edge in genome.edges:
        if edge.dst_node_id == "flatten" and edge.dst_input_key == "field_embeddings":
            field_producer = edge.src_node_id
            break
    if "flatten" in node_ids and field_producer is not None:
        points.append(
            {
                "lane": "field_embedding",
                "wiring": "insert_between",
                "upstream_node_id": field_producer,
                "downstream_node_id": "flatten",
                "tensor_key": "field_embeddings",
                "tensor_shape": ["batch_size", "num_fields", "embedding_dim"],
                "contract": (
                    "Consume field_embeddings [batch_size, num_fields, embedding_dim] and produce a new "
                    "tensor of the SAME shape so flatten still receives the full sparse+dense field stack. "
                    "Good for field-aware reweighting, local feature interaction, or per-field gating. "
                    "Use symbolic init_params ${num_fields} / ${embedding_dim}."
                ),
                "portable": True,
            }
        )

    # Find the consumer of flat_embeddings (the shared transform's entry node).
    flat_consumer = None
    for edge in genome.edges:
        if edge.src_output_key == "flat_embeddings" and edge.src_node_id == "flatten":
            flat_consumer = edge.dst_node_id
            break
    if "flatten" in node_ids and flat_consumer is not None:
        points.append(
            {
                "lane": "shared_representation",
                "wiring": "insert_between",
                "upstream_node_id": "flatten",
                "downstream_node_id": flat_consumer,
                "tensor_key": "flat_embeddings",
                "tensor_shape": ["batch_size", "flat_input_dim"],
                "contract": (
                    "Consume flat_embeddings [batch_size, flat_input_dim] and produce a new "
                    "[batch_size, flat_input_dim] tensor (same width) so the downstream shared "
                    "transform is unaffected. Good for feature-crossing, gating, or self-attention "
                    "reparameterization of the shared input. Use symbolic init_params "
                    "${flat_input_dim} / ${num_fields} / ${embedding_dim}."
                ),
                "portable": True,
            }
        )

    # task_representations edge: transform per-task reps before the task tower.
    rep_producer = None
    for edge in genome.edges:
        if edge.dst_node_id == "task_tower" and edge.dst_input_key == "task_representations":
            rep_producer = edge.src_node_id
            break
    rep_dim = _representation_dim(genome)
    if rep_producer is not None and "task_tower" in node_ids:
        points.append(
            {
                "lane": "task_representation",
                "wiring": "insert_between",
                "upstream_node_id": rep_producer,
                "downstream_node_id": "task_tower",
                "tensor_key": "task_representations",
                "tensor_shape": ["batch_size", "n_task", rep_dim if rep_dim else "representation_dim"],
                "contract": (
                    "Consume task_representations [batch_size, n_task, representation_dim] and produce a "
                    "[batch_size, n_task, representation_dim] tensor of the SAME shape. Good for cross-task "
                    "information transfer (e.g. attention across the n_task axis) or per-task gating. "
                    f"n_task={dims['n_task']}, representation_dim={rep_dim}. These dims are dataset/architecture "
                    "specific: set them explicitly in init_params."
                ),
                "portable": False,
            }
        )

    return points


def mtl_architecture_profile(genome: SkillGenome) -> dict[str, Any]:
    dims = _mtl_runtime_dims(genome)
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
    transform_node = None
    for node in genome.nodes:
        if node.category == "multitask" and "task_representations" in node.output_keys:
            transform_node = node.node_id
            break
    return {
        "task_family": "multi_task_recommendation",
        "node_count": len(genome.nodes),
        "edge_count": len(genome.edges),
        "tags": list(genome.metadata.tags),
        "runtime_dims": dims,
        "representation_dim": _representation_dim(genome),
        "shared_transform_node": transform_node,
        "nodes": nodes,
        "tensor_flow": [
            "sparse_features -> field_embedding -> sparse field embeddings [B, num_sparse_fields, embedding_dim]",
            "dense_values -> dense_feature_path -> dense embeddings [B, num_dense_fields, embedding_dim]",
            "sparse+dense embeddings -> field_embeddings [B, num_fields, embedding_dim]",
            "field_embeddings -> flatten -> flat_embeddings [B, flat_input_dim]",
            "flat_embeddings -> <shared transform> -> task_representations [B, n_task, R]",
            "task_representations -> task_tower -> task_outputs [B, n_task]",
            "task_outputs + task_labels -> multitask_loss -> loss",
        ],
        "injection_points": mtl_injection_points(genome),
    }


def mtl_proposal_schema() -> dict[str, Any]:
    return {
        "proposal_id": "stable_unique_id",
        "proposal_type": "NEW_INTERACTION_DESIGN|NEW_EMBEDDING_DESIGN|NEW_ROUTING_OR_GATING_DESIGN|NEW_SKILL_INVENTION",
        "structural_scope": "macro",
        "target_failure_mode": "multi-task failure being addressed (e.g. negative_transfer_between_tasks, shared_bottom_underfitting)",
        "architecture_hypothesis": "short hypothesis grounded in the parent genome",
        "affected_genome_nodes": ["node_id to connect from"],
        "code": "safe Python module defining one nn.Module whose forward(inputs: dict) -> dict",
        "expected_input_signature": [{"name": "field_embeddings|flat_embeddings|task_representations", "shape": ["shape from selected injection point"], "dtype": "float32"}],
        "expected_output_signature": [{"name": "generated_output_key", "shape": ["same shape as selected injection point"], "dtype": "float32"}],
        "class_name": "ClassName",
        "skill_id": "generated_mtl_skill_id",
        "task_types": ["multitask"],
        "init_params": {},
        "metadata": {
            "node_id": "new_node_id",
            "structural_scope": "macro",
            "wiring": "insert_between|replace_node",
            "output_key": "generated output key (same shape as the intercepted tensor)",
            "upstream_node_id": "required for insert_between: producer of the intercepted tensor",
            "downstream_node_id": "required for insert_between: consumer of the intercepted tensor",
            "target_node_id": "required for replace_node: the shared-transform node to replace (never task_tower/loss/field_embedding)",
            "macro_judgment": {
                "current_architecture_family": "required for macro: e.g. aitm / shared_bottom / mmoe / ple / task_specific",
                "diagnosed_limitation": "what the current shared transform cannot do",
                "target_architecture_transformation": "what the generated skill changes",
                "why_local_edit_is_insufficient": "why a local calibration/edit is not enough",
                "parent_evidence": ["specific parent nodes/keys that motivate this"],
                "proposal_insight": "the non-obvious multi-task modeling insight",
                "preconditions": {"met": [], "missing": []},
                "wiring_plan": "exact insert_between/replace_node wiring",
                "ablation_plan": "how to remove or simplify this",
            },
        },
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_mtl_design_brief(
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
    profile = mtl_architecture_profile(parent)
    scopes = _normalize_structural_scopes(structural_scopes)
    return {
        "round_idx": round_idx,
        "proposal_budget": budget,
        "task_family": "multi_task_recommendation",
        "diagnosis_report": diagnosis_report,
        "available_context_keys": sorted(set(parent.constraints.required_inputs) | parent.produced_keys()),
        "required_inputs": sorted(set(parent.constraints.required_inputs)),
        "produced_keys": sorted(parent.produced_keys()),
        "current_architecture_profile": profile,
        "injection_points": profile["injection_points"],
        "parent_genome": parent.to_dict(),
        "recent_memory": recent,
        "recent_failed_generated_skills": [
            r for r in recent if str(r.get("status", "")).lower() in {"failed", "rejected", "promotion_failed"}
        ],
        "diversity_lanes": list(diversity_lanes or ["field_embedding", "shared_representation", "task_representation", "gating", "interaction"]),
        "structural_scopes": scopes,
        "design_constraints": {
            "task_type": "multitask",
            "safe_code_imports": ["torch", "torch.nn", "torch.nn.functional", "math", "typing", "dataclasses"],
            "forbidden_capabilities": ["file_io", "network", "subprocess", "eval", "exec", "environment_access"],
            "module_interface": "forward(inputs: dict[str, Tensor]) -> dict[str, Tensor]",
            "allowed_wiring_modes": sorted(MTL_ALLOWED_WIRING),
            "allowed_structural_scopes": scopes,
            "structural_scope_definitions": {
                "macro": (
                    "architecture-level transformation of the field, shared, or per-task representation pathway, "
                    "such as new field reparameterization, routing, expert mixing, cross-task transfer, or "
                    "replacement of the shared transform"
                ),
            },
            "protected_nodes": sorted(MTL_PROTECTED_NODES),
            "shape_preservation_rule": (
                "A generated skill must preserve the shape of the tensor it intercepts so the downstream "
                "node still receives what it expects. For field_embeddings keep "
                "[batch_size, num_fields, embedding_dim]; for flat_embeddings keep [batch_size, flat_input_dim]; "
                "for task_representations keep [batch_size, n_task, representation_dim]."
            ),
            "portability_rule": (
                "Prefer the field_embeddings or flat_embeddings injection points and symbolic init_params "
                "${num_fields}, ${embedding_dim}, ${flat_input_dim} so the skill is reusable across datasets. "
                "task_representations skills may use explicit n_task/representation_dim init_params."
            ),
            "task_types_rule": "Every proposal must set task_types=['multitask'] so it satisfies MTL genome constraints.",
            "scope_policy": (
                "All proposals must be structural_scope='macro', include metadata.macro_judgment, and make an "
                "architecture-level change to the shared or per-task representation pathway. Do not emit local "
                "calibration-only, scalar-only, or residual-only edits."
            ),
            "max_reasonable_new_parameters": {"macro": 100000},
        },
    }


def build_mtl_sketch_prompt(*, design_brief: dict[str, Any], sketch_budget: int) -> dict[str, Any]:
    return {
        "instruction": (
            "Create genuinely new multi-task recommendation architecture sketches from the retained parent genome "
            "before writing any code. The model jointly trains multiple binary task heads (income as a CTR-style task, "
            "marital status as a CVR-style task) via a shared representation that is split into per-task representations. "
            "Return only JSON with a top-level sketches list. Do not write Python code in this stage. "
            "Every sketch must target one of design_brief.injection_points and use wiring 'insert_between' or 'replace_node'. "
            "Every sketch must be structural_scope='macro' and describe a representation/routing transformation. "
            "Always preserve tensor shapes."
        ),
        "sketch_budget": sketch_budget,
        "design_brief": design_brief,
        "required_schema": {
            "sketch_id": "stable_unique_id",
            "innovation_lane": "shared_representation|task_representation|gating|interaction|routing",
            "target_failure_mode": "specific multi-task failure addressed",
            "hypothesis": "why this structure should help joint task learning",
            "affected_genome_nodes": ["node_id to connect from"],
            "input_keys": ["existing context keys to consume (field_embeddings, flat_embeddings, or task_representations)"],
            "output_keys": ["new context key, same shape as the intercepted tensor"],
            "wiring": "insert_between|replace_node",
            "core_operator": {"type": "short operator family", "components": ["main differentiable parts"]},
            "complexity_budget": {"max_params": 50000, "latency_risk": "low|medium|high"},
            "expected_risks": ["risk"],
            "ablation_plan": "how to remove or simplify this idea",
            "structural_scope": "macro",
            "metadata": {"injection_point_lane": "field_embedding|shared_representation|task_representation"},
        },
        "selection_guidance": {
            "prefer": [
                "ideas tied to a specific injection_point and its shape contract",
                "cross-task transfer mechanisms (attention across the n_task axis, gated task mixing)",
                "field-aware reweighting or interaction operators on field_embeddings",
                "shared-representation operators (feature crossing, SENet-style gating) on flat_embeddings",
                "shape-preserving transforms that keep the downstream node valid",
                "macro sketches only",
                "different innovation lanes",
            ],
            "avoid": [
                "duplicates of recent failed generated skills",
                "calibration-only or scalar-only edits that do not operate on an MTL representation lane",
                "changing the tensor shape the downstream node consumes",
                "touching protected nodes (task_tower, loss, field_embedding)",
                "inputs not present in available_context_keys",
            ],
        },
    }


def build_mtl_synthesis_prompt(
    *,
    design_brief: dict[str, Any],
    sketches: list[Any],
    proposal_budget: int,
) -> dict[str, Any]:
    return {
        "instruction": (
            "Convert the selected multi-task architecture sketches into safe local PyTorch OpenEndedProposal JSON objects. "
            "Return only JSON with a top-level proposals list. Code must not use file IO, network, subprocess, eval, or exec. "
            "Each proposal must implement an nn.Module whose forward(inputs: dict) returns a dict with exactly the sketched "
            "output key, preserving the intercepted tensor's shape. Set structural_scope to match the selected sketch and "
            "set task_types=['multitask']. Use wiring 'insert_between' "
            "(with metadata.upstream_node_id and metadata.downstream_node_id) or 'replace_node' (with metadata.target_node_id "
            "pointing at the shared-transform node, never task_tower/loss/field_embedding). For field_embeddings skills use "
            "symbolic init_params ${num_fields}, ${embedding_dim}; for flat_embeddings skills use symbolic init_params "
            "${num_fields}, ${embedding_dim}, ${flat_input_dim}; for task_representations skills set n_task and "
            "representation_dim explicitly. Include metadata.macro_judgment for macro proposals."
        ),
        "proposal_budget": proposal_budget,
        "design_brief": design_brief,
        "selected_sketches": [sketch.to_dict() for sketch in sketches],
        "required_schema": mtl_proposal_schema(),
        "code_requirements": {
            "imports": ["torch", "torch.nn", "torch.nn.functional"],
            "interface": "class extends torch.nn.Module and forward accepts a dict named inputs",
            "output": "forward returns dict[str, Tensor] with exactly the sketched output key and identical shape to the input tensor",
            "batch_safe": True,
            "initialization": "prefer residual scale/gate initialized near zero or identity to preserve baseline behavior",
            "metadata": "include source_sketch_id, innovation_lane, wiring, output_key, and the required node ids",
            "structural_scope": "must be macro and should match the selected sketch",
            "macro_metadata": "for macro proposals include metadata.macro_judgment with all required fields",
            "task_types": "['multitask']",
        },
    }


# ---------------------------------------------------------------------------
# Providers (reuse generic command/file machinery, MTL prompts)
# ---------------------------------------------------------------------------


class MTLCommandProposalProvider(CommandProposalProvider):
    """Single-stage command provider that emits MTL proposals via an MTL design brief."""

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
        design_brief = build_mtl_design_brief(
            parent=parent,
            diagnosis_report={**dict(diagnosis_report or {}), "structural_scopes": self.structural_scopes},
            evolution_memory=evolution_memory,
            budget=budget,
            round_idx=round_idx,
            structural_scopes=self.structural_scopes,
        )
        prompt = {
            "instruction": (
                "Generate safe local PyTorch OpenEndedProposal JSON objects for multi-task recommendation-model "
                "evolution from the retained parent genome. Return only JSON with a top-level proposals list. "
                "Code must not use file IO, network, subprocess, eval, or exec. Use wiring 'insert_between' or "
                "'replace_node' only, preserve the intercepted tensor's shape, and respect design_brief.structural_scopes. "
                "Set task_types=['multitask']."
            ),
            "round_idx": round_idx,
            "budget": budget,
            "design_brief": design_brief,
            "required_schema": mtl_proposal_schema(),
        }
        if self.prompt_path:
            _write_prompt(self.prompt_path, prompt)
        proposals = _run_json_command(
            self.command,
            prompt,
            parser=_parse_proposals,
            max_retries=self.max_retries,
            error_prefix="MTL code-space command provider failed",
        )
        proposals = _filter_proposals_by_scope(proposals, self.structural_scopes)
        if self.scope_mix_strategy == "balanced":
            proposals = _take_balanced_by_scope(proposals, start=0, budget=budget, scopes=self.structural_scopes)
        return proposals[:budget]


class MTLCreativeCommandProposalProvider(CreativeCommandProposalProvider):
    """Two-stage (planner -> synthesizer) provider with MTL prompts."""

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
        design_brief = build_mtl_design_brief(
            parent=parent,
            diagnosis_report=diagnosis_report,
            evolution_memory=evolution_memory,
            budget=budget,
            round_idx=round_idx,
            diversity_lanes=self.diversity_lanes,
            structural_scopes=self.structural_scopes,
        )
        sketch_prompt = build_mtl_sketch_prompt(
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
        synthesis_prompt = build_mtl_synthesis_prompt(
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
            error_prefix="MTL creative synthesizer command failed",
        )
        proposals = _attach_sketch_metadata_to_proposals(proposals, selected)
        proposals = [
            proposal
            for proposal in proposals
            if not self.structural_scopes or proposal.structural_scope in self.structural_scopes
        ]
        return proposals[:budget]


def build_mtl_code_space_provider(config: CodeSpaceConfig):
    """Factory mirroring code_space.build_code_space_provider for MTL providers.

    MTL keeps CTR's live provider semantics, including scope-split routing, but
    swaps in MTL-specific prompts.
    """
    provider = config.provider.lower().replace("-", "_")
    if provider in {"scope_split", "split_by_scope", "scoped"}:
        scope_providers = _build_mtl_scope_providers(config)
        return ScopeSplitProposalProvider(
            scope_providers=scope_providers,
            structural_scopes=config.structural_scopes,
            scope_mix_strategy=config.scope_mix_strategy,
            scope_provider_status=code_space_provider_diagnostics(config).get("scope_providers"),
            require_active_provider=config.require_active_provider,
        )
    if provider in {"proposal_file", "file", "offline"}:
        raise ValueError(
            "MTL code-space no longer supports proposal_file/offline replay; "
            "use provider='creative_command' so the LLM judges the current evolution state."
        )
    if provider in {"command", "llm_command", "llm"}:
        if not config.command:
            raise ValueError("code_space.command is required for command provider")
        return MTLCommandProposalProvider(
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
            raise ValueError("MTL macro code-space requires live planner_command; sketch_file/offline sketches are not supported")
        if not planner_command:
            raise ValueError("code_space.planner_command is required for creative_command provider")
        if not synthesizer_command:
            raise ValueError("code_space.synthesizer_command is required for creative_command provider")
        return MTLCreativeCommandProposalProvider(
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
    raise ValueError(f"Unknown MTL code-space provider: {config.provider}")


def _build_mtl_scope_providers(config: CodeSpaceConfig) -> dict[str, Any]:
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
        providers[scope] = build_mtl_code_space_provider(scope_config)
    return providers
