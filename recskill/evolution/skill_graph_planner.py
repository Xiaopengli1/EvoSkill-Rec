from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from .compiler import SkillGenomeCompiler
from .fingerprint import genome_architecture_fingerprint
from .genome import GenomeMutation, SkillEdge, SkillGenome, SkillNode
from .skill_library import SkillCard, SkillLibrary
from .verification import GenomeVerifier


DEFAULT_SKILL_GRAPH_OPERATIONS = ["add_branch", "replace_node", "insert_after", "insert_before"]


@dataclass
class SkillGraphPlannerConfig:
    """Configuration for card-filtered skill graph search."""

    card_top_k: int = 24
    per_category_top_k: int = 6
    beam_width: int = 12
    max_depth: int = 2
    candidate_pool_size: int = 40
    max_actions_per_state: int = 80
    operations: list[str] = field(default_factory=lambda: list(DEFAULT_SKILL_GRAPH_OPERATIONS))
    operation_priors: dict[str, float] = field(
        default_factory=lambda: {"add_branch": 1.0, "replace_node": 0.85, "insert_after": 0.9, "insert_before": 0.9}
    )
    allow_template_fallback: bool = True
    allow_adapter_codegen: bool = False
    validate_compile: bool = True
    random_tie_break: float = 0.001

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "SkillGraphPlannerConfig":
        raw = raw or {}
        return cls(
            card_top_k=int(raw.get("card_top_k", 24)),
            per_category_top_k=int(raw.get("per_category_top_k", 6)),
            beam_width=int(raw.get("beam_width", 12)),
            max_depth=int(raw.get("max_depth", 2)),
            candidate_pool_size=int(raw.get("candidate_pool_size", 40)),
            max_actions_per_state=int(raw.get("max_actions_per_state", 80)),
            operations=list(raw.get("operations") or DEFAULT_SKILL_GRAPH_OPERATIONS),
            operation_priors=dict(raw.get("operation_priors") or {"add_branch": 1.0, "replace_node": 0.85, "insert_after": 0.9, "insert_before": 0.9}),
            allow_template_fallback=bool(raw.get("allow_template_fallback", True)),
            allow_adapter_codegen=bool(raw.get("allow_adapter_codegen", False)),
            validate_compile=bool(raw.get("validate_compile", True)),
            random_tie_break=float(raw.get("random_tie_break", 0.001)),
        )


@dataclass(frozen=True)
class SkillGraphAction:
    operation: str
    skill_id: str
    target_node_id: str = ""
    target_key: str = ""
    score: float = 0.0
    rationale: str = ""


@dataclass
class SkillGraphCandidate:
    genome: SkillGenome
    operation: str
    mutation_type: str
    rationale: str
    score: float
    actions: list[SkillGraphAction] = field(default_factory=list)
    architecture_fingerprint: str = ""


@dataclass
class _PlannerState:
    genome: SkillGenome
    score: float
    actions: list[SkillGraphAction] = field(default_factory=list)
    fingerprints: set[str] = field(default_factory=set)


class SkillGraphSearchPlanner:
    """Search graph edits over relevant skill cards without writing new operator code."""

    def __init__(
        self,
        *,
        skill_library: SkillLibrary | None = None,
        config: SkillGraphPlannerConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.skill_library = skill_library or SkillLibrary.from_repo(include_generated=True)
        self.config = config or SkillGraphPlannerConfig()
        self.rng = random.Random(seed)

    def plan(
        self,
        parent: SkillGenome,
        *,
        failure_modes: list[str] | None = None,
        budget: int = 1,
        round_idx: int = 0,
    ) -> list[SkillGraphCandidate]:
        if budget <= 0:
            return []
        failure_modes = list(failure_modes or [])
        cards = self.select_relevant_cards(parent, failure_modes=failure_modes)
        if not cards:
            return []

        parent_fp = genome_architecture_fingerprint(parent)
        root = _PlannerState(genome=parent, score=0.0, actions=[], fingerprints={parent_fp})
        frontier = [root]
        pool: list[SkillGraphCandidate] = []
        seen_fingerprints = {parent_fp}

        for depth in range(1, max(1, self.config.max_depth) + 1):
            expanded: list[_PlannerState] = []
            for state in frontier:
                for candidate in self._expand_state(state, cards, failure_modes=failure_modes, round_idx=round_idx, depth=depth):
                    if candidate.architecture_fingerprint in state.fingerprints:
                        continue
                    if candidate.architecture_fingerprint in seen_fingerprints and len(candidate.actions) <= 1:
                        continue
                    seen_fingerprints.add(candidate.architecture_fingerprint)
                    pool.append(candidate)
                    expanded.append(
                        _PlannerState(
                            genome=candidate.genome,
                            score=candidate.score,
                            actions=list(candidate.actions),
                            fingerprints=set(state.fingerprints) | {candidate.architecture_fingerprint},
                        )
                    )
            expanded.sort(key=lambda item: (-item.score, _actions_key(item.actions)))
            frontier = expanded[: max(1, self.config.beam_width)]
            if len(pool) >= max(self.config.candidate_pool_size, budget) and depth >= self.config.max_depth:
                break

        pool.sort(key=lambda item: (-item.score, _actions_key(item.actions)))
        selected: list[SkillGraphCandidate] = []
        selected_fps: set[str] = set()
        selected_signatures: set[str] = set()
        for candidate in pool:
            signature = _candidate_diversity_signature(candidate)
            if candidate.architecture_fingerprint in selected_fps:
                continue
            if signature in selected_signatures and len(selected) >= budget // 2:
                continue
            selected.append(candidate)
            selected_fps.add(candidate.architecture_fingerprint)
            selected_signatures.add(signature)
            if len(selected) >= budget:
                break
        if len(selected) < budget:
            for candidate in pool:
                if candidate.architecture_fingerprint in selected_fps:
                    continue
                selected.append(candidate)
                selected_fps.add(candidate.architecture_fingerprint)
                if len(selected) >= budget:
                    break
        promoted_candidates = [
            candidate
            for candidate in pool
            if _candidate_uses_promoted_generated_skill(candidate, self.skill_library)
        ]
        if promoted_candidates and not any(_candidate_uses_promoted_generated_skill(candidate, self.skill_library) for candidate in selected):
            promoted_candidate = promoted_candidates[0]
            if promoted_candidate.architecture_fingerprint not in selected_fps:
                if len(selected) >= budget:
                    selected[-1] = promoted_candidate
                else:
                    selected.append(promoted_candidate)
        return selected[:budget]

    def select_relevant_cards(self, parent: SkillGenome, *, failure_modes: list[str]) -> list[SkillCard]:
        scored: list[tuple[float, str, SkillCard]] = []
        for skill_id in self.skill_library.list_skill_ids():
            try:
                card = self.skill_library.get(skill_id)
            except Exception:
                continue
            if not self._card_can_enter_search(card, parent):
                continue
            score = self._score_card_relevance(card, parent, failure_modes)
            if score <= 0:
                continue
            score += self.rng.random() * self.config.random_tie_break
            scored.append((score, skill_id, card))
        scored.sort(key=lambda item: (-item[0], item[1]))

        selected: list[SkillCard] = []
        category_counts: dict[str, int] = {}
        for _, _, card in scored:
            category = str(card.category or "unknown")
            if category_counts.get(category, 0) >= max(1, self.config.per_category_top_k):
                continue
            selected.append(card)
            category_counts[category] = category_counts.get(category, 0) + 1
            if len(selected) >= max(1, self.config.card_top_k):
                break
        if len(selected) < max(1, self.config.card_top_k):
            seen = {card.skill_id for card in selected}
            for _, _, card in scored:
                if card.skill_id in seen:
                    continue
                selected.append(card)
                seen.add(card.skill_id)
                if len(selected) >= max(1, self.config.card_top_k):
                    break
        return _rebalance_skill_graph_card_selection(selected, scored, target=max(1, self.config.card_top_k))

    def _expand_state(
        self,
        state: _PlannerState,
        cards: list[SkillCard],
        *,
        failure_modes: list[str],
        round_idx: int,
        depth: int,
    ) -> list[SkillGraphCandidate]:
        generated: list[SkillGraphCandidate] = []
        operations = [op for op in self.config.operations if op in DEFAULT_SKILL_GRAPH_OPERATIONS]
        for operation in operations:
            if operation == "add_branch":
                generated.extend(self._enumerate_add_branch(state, cards, failure_modes=failure_modes, round_idx=round_idx, depth=depth))
            elif operation == "replace_node":
                generated.extend(self._enumerate_replace_node(state, cards, failure_modes=failure_modes, round_idx=round_idx, depth=depth))
            elif operation == "insert_after":
                generated.extend(self._enumerate_insert_after(state, cards, failure_modes=failure_modes, round_idx=round_idx, depth=depth))
            elif operation == "insert_before":
                generated.extend(self._enumerate_insert_before(state, cards, failure_modes=failure_modes, round_idx=round_idx, depth=depth))
        generated.sort(key=lambda item: (-item.score, _actions_key(item.actions)))
        return generated[: max(1, self.config.max_actions_per_state)]

    def _enumerate_add_branch(
        self,
        state: _PlannerState,
        cards: list[SkillCard],
        *,
        failure_modes: list[str],
        round_idx: int,
        depth: int,
    ) -> list[SkillGraphCandidate]:
        candidates: list[SkillGraphCandidate] = []
        for card in cards:
            if _is_connector_or_terminal_card(card):
                continue
            for input_keys in _input_key_options(card, state.genome):
                output_suffix = "logit" if _is_promoted_generated_card(card) and _card_outputs_logit(card) else "output"
                output_key = _unique_key(state.genome, f"{_slug(card.skill_id)}_r{round_idx}_d{depth}_{output_suffix}")
                genome = state.genome.clone()
                node_id = _unique_node_id(genome, f"{_slug(card.skill_id)}_r{round_idx}_d{depth}")
                params = _params_for_card(card, genome, input_keys=input_keys, output_key=output_key)
                node = _node_from_card(card, node_id=node_id, params=params, input_keys=input_keys, output_keys=[output_key])
                if _is_promoted_generated_card(card) and _card_outputs_logit(card) and not any(_is_logit_like(key) for key in input_keys):
                    _insert_node_before(genome, node, before_node_id="fusion")
                else:
                    genome.nodes.append(node)
                _add_input_edges(genome, node_id, input_keys)
                if _is_promoted_generated_card(card):
                    _wire_reused_generated_output(genome, node_id=node_id, input_keys=input_keys, output_key=output_key, card=card)
                else:
                    _wire_branch_output_to_ctr_head_or_fusion(genome, node_id, output_key, card, round_idx=round_idx, depth=depth)
                action = SkillGraphAction(
                    operation="add_branch",
                    skill_id=card.skill_id,
                    target_key=",".join(input_keys),
                    score=self._score_action("add_branch", card, state.genome, failure_modes),
                    rationale=f"Add {card.skill_id} branch from {input_keys}",
                )
                candidate = self._candidate_from_genome(state, genome, [action])
                if candidate is not None:
                    candidates.append(candidate)
        return candidates

    def _enumerate_insert_before(
        self,
        state: _PlannerState,
        cards: list[SkillCard],
        *,
        failure_modes: list[str],
        round_idx: int,
        depth: int,
    ) -> list[SkillGraphCandidate]:
        candidates: list[SkillGraphCandidate] = []
        for target in state.genome.nodes:
            if target.node_id in {"field_embedding", "prediction", "loss", "fusion"}:
                continue
            for target_key in target.input_keys:
                producer = _producer_for_key(state.genome, target_key)
                if producer is None and target_key not in state.genome.constraints.required_inputs:
                    continue
                for card in cards:
                    if card.skill_id == target.skill_id or _is_connector_or_terminal_card(card):
                        continue
                    if not _card_can_consume_exact_inputs(card, [target_key]):
                        continue
                    if not _insert_preserves_rank(card):
                        continue
                    genome = state.genome.clone()
                    node_id = _unique_node_id(genome, f"{_slug(card.skill_id)}_before_{target.node_id}_r{round_idx}_d{depth}")
                    output_key = _unique_key(genome, f"{_slug(card.skill_id)}_{target_key}_r{round_idx}_d{depth}")
                    params = _params_for_card(card, genome, input_keys=[target_key], output_key=output_key)
                    node = _node_from_card(card, node_id=node_id, params=params, input_keys=[target_key], output_keys=[output_key])
                    genome.nodes.append(node)
                    if producer is not None:
                        genome.edges.append(SkillEdge(producer, node_id, target_key, target_key))
                    genome.edges = [
                        edge
                        for edge in genome.edges
                        if not (edge.dst_node_id == target.node_id and edge.dst_input_key == target_key)
                    ]
                    genome.edges.append(SkillEdge(node_id, target.node_id, output_key, output_key))
                    rewritten_target = genome.get_node(target.node_id)
                    rewritten_target.input_keys = [output_key if key == target_key else key for key in rewritten_target.input_keys]
                    if rewritten_target.params.get("input_key") == target_key:
                        rewritten_target.params["input_key"] = output_key
                    if isinstance(rewritten_target.params.get("input_keys"), list):
                        rewritten_target.params["input_keys"] = [
                            output_key if key == target_key else key for key in rewritten_target.params["input_keys"]
                        ]
                    action = SkillGraphAction(
                        operation="insert_before",
                        skill_id=card.skill_id,
                        target_node_id=target.node_id,
                        target_key=target_key,
                        score=self._score_action("insert_before", card, state.genome, failure_modes),
                        rationale=f"Insert {card.skill_id} before {target.node_id}.{target_key}",
                    )
                    candidate = self._candidate_from_genome(state, genome, [action])
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    def _enumerate_replace_node(
        self,
        state: _PlannerState,
        cards: list[SkillCard],
        *,
        failure_modes: list[str],
        round_idx: int,
        depth: int,
    ) -> list[SkillGraphCandidate]:
        candidates: list[SkillGraphCandidate] = []
        for target in state.genome.nodes:
            if target.node_id in {"field_embedding", "prediction", "loss", "fusion"}:
                continue
            if not target.input_keys or not target.output_keys:
                continue
            for card in cards:
                if card.skill_id == target.skill_id or _is_connector_or_terminal_card(card):
                    continue
                if not _card_can_consume_exact_inputs(card, target.input_keys):
                    continue
                if not _replacement_category_compatible(card, target):
                    continue
                genome = state.genome.clone()
                replacement = genome.get_node(target.node_id)
                output_key = target.output_keys[0]
                params = _params_for_card(card, genome, input_keys=target.input_keys, output_key=output_key)
                replacement.skill_id = card.skill_id
                replacement.skill_name = card.skill_name
                replacement.category = card.category
                replacement.params = params
                replacement.input_keys = list(target.input_keys)
                replacement.output_keys = list(target.output_keys)
                replacement.task_types = card.task_types or target.task_types
                replacement.source = "generated_skill" if _is_promoted_generated_card(card) else "existing_skill"
                replacement.metadata = {
                    **replacement.metadata,
                    "skill_graph_planner": True,
                    "reused_generated_skill": _is_promoted_generated_card(card),
                    "replaced_skill_id": target.skill_id,
                    "round_idx": round_idx,
                    "depth": depth,
                }
                action = SkillGraphAction(
                    operation="replace_node",
                    skill_id=card.skill_id,
                    target_node_id=target.node_id,
                    target_key=output_key,
                    score=self._score_action("replace_node", card, state.genome, failure_modes),
                    rationale=f"Replace {target.node_id}:{target.skill_id} with {card.skill_id}",
                )
                candidate = self._candidate_from_genome(state, genome, [action])
                if candidate is not None:
                    candidates.append(candidate)
        return candidates

    def _enumerate_insert_after(
        self,
        state: _PlannerState,
        cards: list[SkillCard],
        *,
        failure_modes: list[str],
        round_idx: int,
        depth: int,
    ) -> list[SkillGraphCandidate]:
        candidates: list[SkillGraphCandidate] = []
        for target in state.genome.nodes:
            for target_key in target.output_keys:
                downstream_edges = [edge for edge in state.genome.edges if edge.src_node_id == target.node_id and edge.src_output_key == target_key]
                if not downstream_edges:
                    continue
                for card in cards:
                    if card.skill_id == target.skill_id or _is_connector_or_terminal_card(card):
                        continue
                    if not _card_can_consume_exact_inputs(card, [target_key]):
                        continue
                    if not _insert_preserves_rank(card):
                        continue
                    genome = state.genome.clone()
                    node_id = _unique_node_id(genome, f"{_slug(card.skill_id)}_after_{target.node_id}_r{round_idx}_d{depth}")
                    output_key = _unique_key(genome, f"{_slug(card.skill_id)}_{target_key}_r{round_idx}_d{depth}")
                    params = _params_for_card(card, genome, input_keys=[target_key], output_key=output_key)
                    node = _node_from_card(card, node_id=node_id, params=params, input_keys=[target_key], output_keys=[output_key])
                    genome.nodes.append(node)
                    genome.edges.append(SkillEdge(target.node_id, node_id, target_key, target_key))
                    genome.edges = [
                        edge
                        for edge in genome.edges
                        if not (edge.src_node_id == target.node_id and edge.src_output_key == target_key and edge.dst_node_id != node_id)
                    ]
                    for edge in downstream_edges:
                        genome.edges.append(SkillEdge(node_id, edge.dst_node_id, output_key, edge.dst_input_key, edge.tensor_semantics))
                    action = SkillGraphAction(
                        operation="insert_after",
                        skill_id=card.skill_id,
                        target_node_id=target.node_id,
                        target_key=target_key,
                        score=self._score_action("insert_after", card, state.genome, failure_modes),
                        rationale=f"Insert {card.skill_id} after {target.node_id}.{target_key}",
                    )
                    candidate = self._candidate_from_genome(state, genome, [action])
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    def _candidate_from_genome(
        self,
        state: _PlannerState,
        genome: SkillGenome,
        new_actions: list[SkillGraphAction],
    ) -> SkillGraphCandidate | None:
        actions = list(state.actions) + list(new_actions)
        if _has_repeated_action(actions):
            return None
        if not self._validate_candidate(genome):
            return None
        fp = genome_architecture_fingerprint(genome)
        operation = actions[-1].operation if len(actions) == 1 else "hybridize"
        skill_part = "_".join(_slug(action.skill_id) for action in actions[:2])
        mutation_type = f"skill_graph_{operation}_{skill_part}"
        rationale = "; ".join(action.rationale for action in actions)
        score = state.score + sum(action.score for action in new_actions) - 0.2 * max(0, len(actions) - 1)
        _record_skill_graph_mutation(genome, parent_id=state.genome.metadata.genome_id, mutation_type=mutation_type, actions=actions, score=score)
        return SkillGraphCandidate(
            genome=genome,
            operation=operation,
            mutation_type=mutation_type,
            rationale=rationale,
            score=score,
            actions=actions,
            architecture_fingerprint=fp,
        )

    def _validate_candidate(self, genome: SkillGenome) -> bool:
        try:
            GenomeVerifier(skill_library=self.skill_library).assert_valid(genome)
            if self.config.validate_compile:
                model = SkillGenomeCompiler(skill_library=self.skill_library).compile(genome)
                _synthetic_forward_smoke(model, genome)
            return True
        except Exception:
            return False

    def _card_can_enter_search(self, card: SkillCard, parent: SkillGenome) -> bool:
        if card.skill_id in {node.skill_id for node in parent.nodes} and card.category in {"embedding", "objective", "loss"}:
            return False
        promoted_generated = _is_promoted_generated_card(card)
        if str(card.category).lower() in {"loss", "objective", "training", "generative", "generated"} and not promoted_generated:
            return False
        if promoted_generated and not _skill_card_has_loadable_implementation(card):
            return False
        required_tasks = {str(task).lower() for task in parent.constraints.task_types}
        if required_tasks:
            card_tasks = {str(task).lower() for task in card.task_types}
            retrieval_tasks = {str(task).lower() for task in (card.manifest.get("retrieval") or {}).get("task_types", [])}
            if (card_tasks or retrieval_tasks) and not ((card_tasks | retrieval_tasks) & required_tasks):
                return False
        return bool(_input_key_options(card, parent) or any(_card_can_consume_exact_inputs(card, node.input_keys) for node in parent.nodes if node.input_keys))

    def _score_card_relevance(self, card: SkillCard, parent: SkillGenome, failure_modes: list[str]) -> float:
        text = _card_text(card)
        score = 1.0
        for failure_mode in failure_modes:
            for token in str(failure_mode).lower().replace("_", " ").split():
                if token and token in text:
                    score += 4.0
        parent_skill_ids = {node.skill_id for node in parent.nodes}
        composition = card.manifest.get("composition") or {}
        upstream = {str(item) for item in composition.get("common_upstream", []) or []}
        downstream = {str(item) for item in composition.get("common_downstream", []) or []}
        if upstream & parent_skill_ids:
            score += 5.0
        if downstream & parent_skill_ids:
            score += 3.0
        if _input_key_options(card, parent):
            score += 8.0
        if _is_promoted_generated_card(card):
            score += 8.0
            metrics = card.manifest.get("candidate_metrics") or (card.manifest.get("metadata") or {}).get("candidate_metrics") or {}
            try:
                score += min(4.0, max(0.0, float(metrics.get("validation_best_auc", 0.0)) * 2.0))
            except Exception:
                pass
        if _is_ctr_interaction_underfit_card(card, parent):
            score += 8.0
        if _is_fusion_or_calibration_card(card, parent):
            score += 3.0
        if card.skill_id in parent_skill_ids:
            score -= 4.0
        score -= _complexity_penalty(card)
        return score

    def _score_action(self, operation: str, card: SkillCard, parent: SkillGenome, failure_modes: list[str]) -> float:
        return (
            self._score_card_relevance(card, parent, failure_modes)
            + float(self.config.operation_priors.get(operation, 1.0))
            + self.rng.random() * self.config.random_tie_break
        )


def _node_from_card(
    card: SkillCard,
    *,
    node_id: str,
    params: dict[str, Any],
    input_keys: list[str],
    output_keys: list[str],
) -> SkillNode:
    return SkillNode(
        node_id=node_id,
        skill_id=card.skill_id,
        skill_name=card.skill_name,
        category=card.category,
        params=params,
        input_keys=list(input_keys),
        output_keys=list(output_keys),
        task_types=card.task_types or ["ctr"],
        source="generated_skill" if _is_promoted_generated_card(card) else "existing_skill",
        metadata={"skill_graph_planner": True},
    )


def _rebalance_skill_graph_card_selection(
    selected: list[SkillCard],
    scored: list[tuple[float, str, SkillCard]],
    *,
    target: int,
) -> list[SkillCard]:
    if not selected or target <= 0:
        return selected[:target]
    selected = list(selected[:target])
    selected_ids = {card.skill_id for card in selected}
    non_generated_pool = [card for _, _, card in scored if not _is_promoted_generated_card(card)]
    generated_pool = [card for _, _, card in scored if _is_promoted_generated_card(card)]

    if non_generated_pool:
        min_non_generated = min(len(non_generated_pool), max(1, target // 2))
        while sum(1 for card in selected if not _is_promoted_generated_card(card)) < min_non_generated:
            replacement = next((card for card in non_generated_pool if card.skill_id not in selected_ids), None)
            if replacement is None:
                break
            replace_idx = next(
                (idx for idx in range(len(selected) - 1, -1, -1) if _is_promoted_generated_card(selected[idx])),
                None,
            )
            if replace_idx is None:
                if len(selected) >= target:
                    break
                selected.append(replacement)
            else:
                selected_ids.discard(selected[replace_idx].skill_id)
                selected[replace_idx] = replacement
            selected_ids.add(replacement.skill_id)

    if generated_pool and not any(_is_promoted_generated_card(card) for card in selected):
        replacement = next((card for card in generated_pool if card.skill_id not in selected_ids), None)
        if replacement is not None:
            replace_idx = len(selected) - 1 if len(selected) >= target else None
            if replace_idx is None:
                selected.append(replacement)
            else:
                selected_ids.discard(selected[replace_idx].skill_id)
                selected[replace_idx] = replacement
            selected_ids.add(replacement.skill_id)
    return selected[:target]


def _candidate_uses_promoted_generated_skill(candidate: SkillGraphCandidate, library: SkillLibrary) -> bool:
    for action in candidate.actions:
        try:
            if _is_promoted_generated_card(library.get(action.skill_id)):
                return True
        except Exception:
            continue
    return False


def _params_for_card(card: SkillCard, genome: SkillGenome, *, input_keys: list[str], output_key: str) -> dict[str, Any]:
    params = _example_params(card)
    if len(input_keys) == 1:
        params["input_key"] = input_keys[0]
    elif input_keys:
        params["input_keys"] = list(input_keys)
    params["output_key"] = output_key
    if _is_promoted_generated_card(card):
        return {key: _resolve_param_value(value, genome) for key, value in params.items()}
    if "input_dim" in _param_text(card, params):
        params["input_dim"] = _infer_dim_for_key(genome, input_keys[0] if input_keys else "")
    if _needs_param(card, params, "embedding_dim"):
        params["embedding_dim"] = _embedding_dim(genome)
    if _needs_param(card, params, "num_fields"):
        params["num_fields"] = _num_fields(genome)
    if _needs_param(card, params, "num_layers"):
        params["num_layers"] = int(params.get("num_layers") or 2)
    if _needs_param(card, params, "attention_dim"):
        params["attention_dim"] = int(params.get("attention_dim") or 64)
    if _needs_param(card, params, "num_heads"):
        emb = _embedding_dim(genome)
        params["num_heads"] = int(params.get("num_heads") or (2 if emb % 2 == 0 else 1))
    if _needs_param(card, params, "low_rank"):
        params["low_rank"] = int(params.get("low_rank") or max(1, min(32, _flat_input_dim(genome) // 4)))
    if _needs_param(card, params, "num_experts"):
        params["num_experts"] = int(params.get("num_experts") or 4)
    if _needs_param(card, params, "reduction_ratio"):
        params["reduction_ratio"] = int(params.get("reduction_ratio") or 3)
    if _needs_param(card, params, "bilinear_type"):
        params["bilinear_type"] = str(params.get("bilinear_type") or "field_interaction")
    return {key: _resolve_param_value(value, genome) for key, value in params.items()}


def _wire_branch_output_to_ctr_head_or_fusion(
    genome: SkillGenome,
    producer_node_id: str,
    output_key: str,
    card: SkillCard,
    *,
    round_idx: int,
    depth: int,
) -> None:
    if _is_logit_like(output_key) or _card_outputs_logit(card):
        _append_logit_to_fusion(genome, producer_node_id, output_key)
        return
    rank = _first_output_rank(card)
    head_input_key = output_key
    head_input_dim = _infer_output_dim(card, genome)
    if rank is not None and rank >= 3:
        flat_node_id = _unique_node_id(genome, f"{_slug(card.skill_id)}_flatten_r{round_idx}_d{depth}")
        flat_key = _unique_key(genome, f"{_slug(card.skill_id)}_flat_r{round_idx}_d{depth}")
        genome.nodes.append(
            SkillNode(
                node_id=flat_node_id,
                skill_id="flatten_field_embeddings",
                skill_name="flatten_field_embeddings",
                category="utility",
                params={"input_key": output_key, "output_key": flat_key},
                input_keys=[output_key],
                output_keys=[flat_key],
                task_types=["ctr"],
                metadata={"skill_graph_planner": True, "connector": True},
            )
        )
        genome.edges.append(SkillEdge(producer_node_id, flat_node_id, output_key, output_key))
        head_input_key = flat_key
    head_node_id = _unique_node_id(genome, f"{_slug(card.skill_id)}_head_r{round_idx}_d{depth}")
    logit_key = _unique_key(genome, f"{_slug(card.skill_id)}_logit_r{round_idx}_d{depth}")
    genome.nodes.append(
        SkillNode(
            node_id=head_node_id,
            skill_id="binary_ctr_head",
            skill_name="binary_ctr_head",
            category="head",
            params={"input_key": head_input_key, "input_dim": head_input_dim, "output_key": logit_key},
            input_keys=[head_input_key],
            output_keys=[logit_key],
            task_types=["ctr"],
            metadata={"skill_graph_planner": True, "connector": True},
        )
    )
    genome.edges.append(SkillEdge(producer_node_id if head_input_key == output_key else _producer_for_key(genome, head_input_key) or producer_node_id, head_node_id, head_input_key, head_input_key))
    _append_logit_to_fusion(genome, head_node_id, logit_key)


def _append_logit_to_fusion(genome: SkillGenome, producer_node_id: str, logit_key: str) -> None:
    try:
        fusion = genome.get_node("fusion")
    except Exception:
        return
    fusion.params.setdefault("input_keys", list(fusion.input_keys))
    fusion.params["input_keys"] = list(dict.fromkeys(list(fusion.params["input_keys"]) + [logit_key]))
    fusion.input_keys = list(dict.fromkeys(list(fusion.input_keys) + [logit_key]))
    edge = SkillEdge(producer_node_id, "fusion", logit_key, logit_key)
    if edge not in genome.edges:
        genome.edges.append(edge)


def _insert_node_before(genome: SkillGenome, node: SkillNode, *, before_node_id: str) -> None:
    for idx, existing in enumerate(genome.nodes):
        if existing.node_id == before_node_id:
            genome.nodes.insert(idx, node)
            return
    genome.nodes.append(node)


def _wire_reused_generated_output(
    genome: SkillGenome,
    *,
    node_id: str,
    input_keys: list[str],
    output_key: str,
    card: SkillCard,
) -> None:
    if _is_logit_like(output_key) or _card_outputs_logit(card):
        if any(_is_logit_like(key) for key in input_keys):
            old_key = next((key for key in input_keys if _is_logit_like(key)), _current_logits_key(genome))
            _reroute_prediction_and_loss_logits(genome, old_key=old_key, new_key=output_key, producer_node_id=node_id)
            return
        _append_logit_to_fusion(genome, node_id, output_key)


def _reroute_prediction_and_loss_logits(genome: SkillGenome, *, old_key: str, new_key: str, producer_node_id: str) -> None:
    try:
        prediction = genome.get_node("prediction")
        loss = genome.get_node("loss")
    except Exception:
        _append_logit_to_fusion(genome, producer_node_id, new_key)
        return
    prediction.params["input_key"] = new_key
    prediction.input_keys = [new_key]
    loss.params["logits_key"] = new_key
    non_logit_loss_inputs = [key for key in loss.input_keys if not _is_logit_like(key)]
    loss.input_keys = list(dict.fromkeys([new_key] + non_logit_loss_inputs))
    genome.edges = [
        edge
        for edge in genome.edges
        if not (
            edge.dst_node_id in {"prediction", "loss"}
            and (
                edge.dst_input_key == old_key
                or edge.src_output_key == old_key
                or _is_logit_like(edge.dst_input_key)
                or _is_logit_like(edge.src_output_key)
            )
        )
    ]
    genome.edges.extend(
        [
            SkillEdge(producer_node_id, "prediction", new_key, new_key),
            SkillEdge(producer_node_id, "loss", new_key, new_key),
        ]
    )


def _add_input_edges(genome: SkillGenome, node_id: str, input_keys: list[str]) -> None:
    required_inputs = set(genome.constraints.required_inputs)
    existing = {(edge.dst_node_id, edge.dst_input_key) for edge in genome.edges}
    for input_key in input_keys:
        if input_key in required_inputs or (node_id, input_key) in existing:
            continue
        producer = _producer_for_key(genome, input_key)
        if producer is not None:
            genome.edges.append(SkillEdge(producer, node_id, input_key, input_key))


def _input_key_options(card: SkillCard, genome: SkillGenome) -> list[list[str]]:
    requirements = list(card.input_keys)
    available = _available_keys(genome)
    if not requirements:
        return []
    if len(requirements) > 1 and "input_keys" not in requirements:
        if all(req in available for req in requirements):
            return [requirements]
        return []
    req = requirements[0]
    if req in available and not _is_param_placeholder(req):
        return [[req]]
    if _is_logit_like(req):
        current = _current_logits_key(genome)
        return [[current]] if current in available else []
    if req in {"input_key", "input_keys", "output_key"} or _is_param_placeholder(req):
        candidates = _preferred_input_keys_for_card(card, genome)
        return [[key] for key in candidates if key in available]
    return []


def _preferred_input_keys_for_card(card: SkillCard, genome: SkillGenome) -> list[str]:
    retrieval = card.manifest.get("retrieval") or {}
    modalities = " ".join(str(item).lower() for item in retrieval.get("input_modalities", []) or [])
    text = f"{card.skill_id} {card.category} {modalities} {_card_text(card)}"
    options: list[str] = []
    if "field" in text and "field_embeddings" in _available_keys(genome):
        options.append("field_embeddings")
    if "flat" in text and "flat_embeddings" in _available_keys(genome):
        options.append("flat_embeddings")
    if "logit" in text and _current_logits_key(genome) in _available_keys(genome):
        options.append(_current_logits_key(genome))
    if not options:
        options.extend(key for key in ["flat_embeddings", "field_embeddings", _current_logits_key(genome)] if key in _available_keys(genome))
    return list(dict.fromkeys(options))[:3]


def _card_can_consume_exact_inputs(card: SkillCard, input_keys: list[str]) -> bool:
    if not input_keys:
        return False
    requirements = list(card.input_keys)
    if len(input_keys) == 1:
        key = input_keys[0]
        if key in requirements:
            return True
        if requirements and _is_logit_like(requirements[0]) and _is_logit_like(key):
            return True
        if requirements and _is_param_placeholder(requirements[0]):
            return True
    return requirements == input_keys


def _replacement_category_compatible(card: SkillCard, target: SkillNode) -> bool:
    if card.category == target.category:
        return True
    if target.category in {"interaction", "utility"} and card.category in {"interaction", "utility", "adapter"}:
        return True
    if target.category in {"tower", "head"} and card.category == target.category:
        return True
    return False


def _insert_preserves_rank(card: SkillCard) -> bool:
    in_rank = _first_input_rank(card)
    out_rank = _first_output_rank(card)
    if in_rank is None or out_rank is None:
        return card.skill_id in {"senet_feature_gate", "normalize_embedding"}
    return in_rank == out_rank


def _record_skill_graph_mutation(
    genome: SkillGenome,
    *,
    parent_id: str,
    mutation_type: str,
    actions: list[SkillGraphAction],
    score: float,
) -> None:
    genome.record_mutation(
        GenomeMutation(
            mutation_type=mutation_type,
            parent_genome_id=parent_id,
            child_genome_id=genome.metadata.genome_id,
            description=f"Skill graph planner mutation: {mutation_type}",
            details={
                "actions": [
                    {
                        "operation": action.operation,
                        "skill_id": action.skill_id,
                        "target_node_id": action.target_node_id,
                        "target_key": action.target_key,
                        "score": action.score,
                        "rationale": action.rationale,
                    }
                    for action in actions
                ],
                "score": score,
            },
        )
    )


def _example_params(card: SkillCard) -> dict[str, Any]:
    fragment = (card.manifest.get("composition") or {}).get("example_genome_fragment") or {}
    if isinstance(fragment, str):
        try:
            fragment = yaml.safe_load(fragment) or {}
        except Exception:
            fragment = {}
    if isinstance(fragment, dict):
        return dict(fragment.get("params") or {})
    return {}


def _needs_param(card: SkillCard, params: dict[str, Any], name: str) -> bool:
    if name in params:
        value = params[name]
        return isinstance(value, str) and value.startswith("${")
    blob = _param_text(card, params)
    return name in blob


def _param_text(card: SkillCard, params: dict[str, Any]) -> str:
    return " ".join(
        [
            str(card.manifest.get("inputs") or ""),
            str(card.manifest.get("input_signature") or ""),
            str(card.manifest.get("composition") or ""),
            str(params),
            str((card.manifest.get("source") or {}).get("class_or_function") or ""),
        ]
    )


def _resolve_param_value(value: Any, genome: SkillGenome) -> Any:
    if value == "${fusion_dim}" or value == "${input_dim}":
        return _flat_input_dim(genome)
    if value == "${embedding_dim}":
        return _embedding_dim(genome)
    if value == "${num_fields}":
        return _num_fields(genome)
    return value


def _card_text(card: SkillCard) -> str:
    manifest = card.manifest
    retrieval = manifest.get("retrieval") or {}
    values = [
        card.skill_id,
        card.category,
        manifest.get("description"),
        manifest.get("inductive_bias"),
        manifest.get("failure_signatures"),
        manifest.get("failure_modes_addressed"),
        retrieval.get("summary"),
        retrieval.get("use_when"),
        retrieval.get("architecture_roles"),
        retrieval.get("objectives"),
        retrieval.get("model_families"),
        retrieval.get("aliases"),
    ]
    return _lower_text(values)


def _lower_text(values: list[Any]) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            parts.append(_lower_text(list(value.values())))
        elif isinstance(value, (list, tuple, set)):
            parts.append(_lower_text(list(value)))
        else:
            parts.append(str(value))
    return " ".join(parts).lower()


def _is_ctr_interaction_underfit_card(card: SkillCard, genome: SkillGenome) -> bool:
    text = _card_text(card)
    interaction_skills = {node.skill_id for node in genome.nodes if node.category == "interaction"}
    only_fm = bool(interaction_skills) and interaction_skills <= {"fm_interaction"}
    return only_fm and any(token in text for token in ["cross", "attention", "bilinear", "gate", "interaction"])


def _is_fusion_or_calibration_card(card: SkillCard, genome: SkillGenome) -> bool:
    text = _card_text(card)
    try:
        fusion = genome.get_node("fusion")
    except Exception:
        fusion = None
    return fusion is not None and any(token in text for token in ["fusion", "calibration", "logit", "gate"])


def _complexity_penalty(card: SkillCard) -> float:
    text = _card_text(card)
    penalty = 0.0
    for token in ["transformer", "sequence", "generative", "quantizer", "t5"]:
        if token in text:
            penalty += 2.0
    return penalty


def _is_connector_or_terminal_card(card: SkillCard) -> bool:
    skill_id = card.skill_id
    category = str(card.category).lower()
    return skill_id in {"binary_ctr_head", "sigmoid_prediction", "bce_loss", "additive_fusion"} or category in {"loss", "objective"}


def _is_promoted_generated_card(card: SkillCard) -> bool:
    manifest = card.manifest
    aliases = {str(item).lower() for item in (manifest.get("retrieval") or {}).get("aliases", [])}
    return bool(
        manifest.get("created_from_open_ended_evolution")
        or str(manifest.get("promotion_status", "")).lower() in {"promoted", "validated"}
        or "open_ended_evolution" in aliases
    )


def _skill_card_has_loadable_implementation(card: SkillCard) -> bool:
    implementation_path = card.implementation_path
    if not implementation_path:
        return False
    path = Path(implementation_path)
    if not path.is_absolute():
        if card.path is not None:
            for base in [card.path.parent, card.path.parent.parent]:
                candidate = base / path
                if candidate.exists():
                    return True
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / path
    return path.exists()


def _card_outputs_logit(card: SkillCard) -> bool:
    outputs = list(card.output_keys)
    retrieval = card.manifest.get("retrieval") or {}
    semantics = " ".join(str(item).lower() for item in retrieval.get("output_semantics", []) or [])
    return any(_is_logit_like(key) for key in outputs) or "logit" in semantics


def _is_logit_like(key: str) -> bool:
    lowered = str(key).lower()
    return lowered == "logits" or "logit" in lowered


def _is_param_placeholder(key: str) -> bool:
    return str(key) in {"input_key", "input_keys", "output_key", "output_keys", "logits_key", "labels_key"}


def _available_keys(genome: SkillGenome) -> set[str]:
    return set(genome.constraints.required_inputs) | genome.produced_keys()


def _producer_for_key(genome: SkillGenome, key: str) -> str | None:
    for node in genome.nodes:
        if key in node.output_keys:
            return node.node_id
    return None


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


def _flat_input_dim(genome: SkillGenome) -> int:
    try:
        embedding = genome.get_node("field_embedding")
        vocab_sizes = embedding.params.get("vocab_sizes") or []
        embedding_dim = int(embedding.params.get("embedding_dim", 16))
        if vocab_sizes:
            return len(vocab_sizes) * embedding_dim
    except Exception:
        pass
    return 16


def _embedding_dim(genome: SkillGenome) -> int:
    try:
        return int(genome.get_node("field_embedding").params.get("embedding_dim", 16))
    except Exception:
        return 16


def _num_fields(genome: SkillGenome) -> int:
    try:
        vocab_sizes = genome.get_node("field_embedding").params.get("vocab_sizes") or []
        if vocab_sizes:
            return len(vocab_sizes)
    except Exception:
        pass
    return 1


def _infer_dim_for_key(genome: SkillGenome, key: str) -> int:
    if key == "flat_embeddings":
        return _flat_input_dim(genome)
    if key == "field_embeddings":
        return _embedding_dim(genome)
    if _is_logit_like(key):
        return 1
    for node in genome.nodes:
        if key not in node.output_keys:
            continue
        if node.skill_id in {"crossnet_mix", "crossnet_v1", "crossnet_v2", "flatten_field_embeddings"}:
            return _flat_input_dim(genome)
        if node.skill_id in {"afm_attention_pooling"}:
            return _embedding_dim(genome)
        if node.skill_id in {"bilinear_interaction"}:
            return _num_fields(genome) * max(1, _num_fields(genome) - 1) // 2 * _embedding_dim(genome)
    return _flat_input_dim(genome)


def _infer_output_dim(card: SkillCard, genome: SkillGenome) -> int:
    if card.skill_id in {"crossnet_mix", "crossnet_v1", "crossnet_v2", "autoint_attention", "senet_feature_gate"}:
        return _flat_input_dim(genome)
    if card.skill_id in {"afm_attention_pooling"}:
        return _embedding_dim(genome)
    if card.skill_id in {"bilinear_interaction"}:
        return _num_fields(genome) * max(1, _num_fields(genome) - 1) // 2 * _embedding_dim(genome)
    return _flat_input_dim(genome) if (_first_output_rank(card) or 2) >= 3 else _embedding_dim(genome)


def _first_input_rank(card: SkillCard) -> int | None:
    return _first_signature_rank(card.manifest.get("input_signature") or card.manifest.get("inputs") or [])


def _first_output_rank(card: SkillCard) -> int | None:
    return _first_signature_rank(card.manifest.get("output_signature") or card.manifest.get("outputs") or [])


def _first_signature_rank(signature: Any) -> int | None:
    if not isinstance(signature, list) or not signature:
        return None
    shape = signature[0].get("shape") if isinstance(signature[0], dict) else None
    if isinstance(shape, str):
        stripped = shape.strip().strip("[]")
        return len([part for part in stripped.split(",") if part.strip()])
    if isinstance(shape, list):
        return len(shape)
    return None


def _unique_node_id(genome: SkillGenome, prefix: str) -> str:
    node_id = prefix
    idx = 1
    existing = genome.node_ids()
    while node_id in existing:
        node_id = f"{prefix}_{idx}"
        idx += 1
    return node_id


def _unique_key(genome: SkillGenome, prefix: str) -> str:
    key = prefix
    idx = 1
    existing = _available_keys(genome)
    while key in existing:
        key = f"{prefix}_{idx}"
        idx += 1
    return key


def _slug(value: str) -> str:
    slug = "".join(char if char.isalnum() or char == "_" else "_" for char in str(value).lower()).strip("_")
    return slug or "skill"


def _actions_key(actions: list[SkillGraphAction]) -> str:
    return "|".join(f"{action.operation}:{action.skill_id}:{action.target_node_id}:{action.target_key}" for action in actions)


def _candidate_diversity_signature(candidate: SkillGraphCandidate) -> str:
    if not candidate.actions:
        return candidate.operation
    first = candidate.actions[0]
    return f"{candidate.operation}:{first.skill_id}:{first.target_node_id}"


def _has_repeated_action(actions: list[SkillGraphAction]) -> bool:
    seen: set[tuple[str, str, str, str]] = set()
    for action in actions:
        key = (action.operation, action.skill_id, action.target_node_id, action.target_key)
        if key in seen:
            return True
        seen.add(key)
    return False


def _synthetic_forward_smoke(model: torch.nn.Module, genome: SkillGenome) -> None:
    batch_size = 2
    batch: dict[str, torch.Tensor] = {}
    required_inputs = set(genome.constraints.required_inputs)
    if "sparse_features" in required_inputs or any(node.skill_id == "field_embedding" for node in genome.nodes):
        batch["sparse_features"] = _synthetic_sparse_features(genome, batch_size=batch_size)
    if "labels" in required_inputs:
        batch["labels"] = torch.tensor([0.0, 1.0], dtype=torch.float32)
    for key in required_inputs:
        if key in batch:
            continue
        batch[key] = torch.randn(batch_size, 4)
    model.eval()
    with torch.no_grad():
        ctx = model(batch)
    for key in genome.constraints.required_outputs:
        if key not in ctx:
            raise AssertionError(f"missing required output after smoke forward: {key}")


def _synthetic_sparse_features(genome: SkillGenome, *, batch_size: int) -> torch.Tensor:
    try:
        vocab_sizes = list(genome.get_node("field_embedding").params.get("vocab_sizes") or [])
    except Exception:
        vocab_sizes = [8]
    if not vocab_sizes:
        vocab_sizes = [8]
    columns = []
    for vocab_size in vocab_sizes:
        high = max(2, int(vocab_size))
        columns.append(torch.randint(1, high, (batch_size, 1), dtype=torch.long))
    return torch.cat(columns, dim=1)
