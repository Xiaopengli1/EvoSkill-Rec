from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from .evolution_memory import EvolutionMemory
from .genome import GenomeConstraints, MutationPlan, SkillEdge, SkillGenome
from .open_ended import OpenEndedProposal
from .skill_library import SkillCard, SkillLibrary


class BaseEvolutionPlanner(ABC):
    @abstractmethod
    def plan(
        self,
        current_genome: SkillGenome,
        skill_library: SkillLibrary,
        diagnosis_report: dict[str, Any],
        evolution_memory: EvolutionMemory | None = None,
        constraints: GenomeConstraints | None = None,
        budget: int = 1,
    ) -> list[MutationPlan | OpenEndedProposal]:
        raise NotImplementedError


class SkillSpacePlanner(BaseEvolutionPlanner):
    pass


class OpenEndedCodingPlanner(BaseEvolutionPlanner):
    pass


class RuleBasedSkillPlanner(SkillSpacePlanner):
    """Deterministic baseline planner for common recommendation failure modes."""

    FAILURE_TO_SKILLS = {
        "high_order_interaction_underfitting": ["crossnet_v2", "crossnet_v1", "fm_interaction", "bilinear_interaction"],
        "recency_insensitivity": ["target_sequence_attention", "sequence_pooling", "gru_sequence_encoder"],
        "target_independent_interest": ["din_target_attention", "target_sequence_attention"],
        "task_conflict": ["mmoe_gate", "ple_gate", "task_specific_towers", "task_tower"],
        "scenario_gap": ["normalize_embedding", "task_tower"],
    }

    def plan(
        self,
        current_genome: SkillGenome,
        skill_library: SkillLibrary,
        diagnosis_report: dict[str, Any],
        evolution_memory: EvolutionMemory | None = None,
        constraints: GenomeConstraints | None = None,
        budget: int = 1,
    ) -> list[MutationPlan]:
        failure_modes = _extract_failure_modes(diagnosis_report)
        target_node_id = diagnosis_report.get("target_node_id") or diagnosis_report.get("affected_node_id")
        plans: list[MutationPlan] = []
        for skill_id, score in _rank_library_skills(current_genome, skill_library, failure_modes, diagnosis_report, budget=max(budget * 4, budget)):
            edges = _suggest_edges(current_genome, skill_library, skill_id, target_node_id)
            if target_node_id and not edges:
                continue
            primary_failure = failure_modes[0] if failure_modes else ""
            mutation_type = "specialize_skill" if primary_failure in {"scenario_gap", "task_conflict"} else "add_skill"
            plans.append(
                MutationPlan(
                    mutation_type=mutation_type,
                    skill_id=skill_id,
                    edges=edges,
                    rationale=f"Address {primary_failure or 'diagnosis'} with library-selected {skill_id}",
                    metadata={
                        "failure_modes": failure_modes,
                        "selection_score": score,
                        "selection_source": "skill_library",
                        "target_node_id": target_node_id,
                    },
                )
            )
            if len(plans) >= budget:
                return plans
        return plans


class ProposalFilePlanner(BaseEvolutionPlanner):
    """Read mutation plans or open-ended proposals generated offline."""

    def __init__(self, proposal_path: str | Path) -> None:
        self.proposal_path = Path(proposal_path)

    def plan(
        self,
        current_genome: SkillGenome,
        skill_library: SkillLibrary,
        diagnosis_report: dict[str, Any],
        evolution_memory: EvolutionMemory | None = None,
        constraints: GenomeConstraints | None = None,
        budget: int = 1,
    ) -> list[MutationPlan | OpenEndedProposal]:
        data = _load_structured(self.proposal_path)
        items = _items_from_file(data)
        planned: list[MutationPlan | OpenEndedProposal] = []
        for item in items:
            if "proposal_type" in item:
                planned.append(OpenEndedProposal.from_dict(item))
            elif "mutation_type" in item:
                planned.append(MutationPlan.from_dict(item))
            else:
                raise ValueError(f"File item is neither MutationPlan nor OpenEndedProposal: {item}")
            if len(planned) >= budget:
                break
        return planned


def _suggest_edges(
    genome: SkillGenome,
    skill_library: SkillLibrary,
    skill_id: str,
    target_node_id: str | None,
) -> list[SkillEdge]:
    if not target_node_id:
        return []
    try:
        target_node = genome.get_node(target_node_id)
        candidate = skill_library.build_node(skill_id)
    except Exception:
        return []
    if not target_node.output_keys or not candidate.input_keys:
        return []
    return [
        SkillEdge(
            src_node_id=target_node.node_id,
            dst_node_id="NEW_NODE",
            src_output_key=target_node.output_keys[0],
            dst_input_key=candidate.input_keys[0],
            tensor_semantics=None,
        )
    ]


def _rank_library_skills(
    genome: SkillGenome,
    skill_library: SkillLibrary,
    failure_modes: list[str],
    diagnosis_report: dict[str, Any],
    *,
    budget: int,
) -> list[tuple[str, int]]:
    required_tasks = {str(task).lower() for task in genome.constraints.task_types}
    preferred_skills = {
        skill_id
        for failure_mode in failure_modes
        for skill_id in RuleBasedSkillPlanner.FAILURE_TO_SKILLS.get(failure_mode, [])
    }
    scored: list[tuple[str, int]] = []
    for skill_id in skill_library.list_skill_ids():
        try:
            card = skill_library.get(skill_id)
        except Exception:
            continue
        if not _card_matches_tasks(card, required_tasks):
            continue
        score = _score_library_card(card, genome, failure_modes, diagnosis_report, preferred=skill_id in preferred_skills)
        if score > 0:
            scored.append((skill_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:budget]


def _card_matches_tasks(card: SkillCard, required_tasks: set[str]) -> bool:
    if not required_tasks:
        return True
    manifest_tasks = {str(task).lower() for task in card.task_types}
    retrieval_tasks = {str(task).lower() for task in (card.manifest.get("retrieval") or {}).get("task_types", [])}
    tasks = manifest_tasks | retrieval_tasks
    return not tasks or bool(tasks & required_tasks)


def _score_library_card(
    card: SkillCard,
    genome: SkillGenome,
    failure_modes: list[str],
    diagnosis_report: dict[str, Any],
    *,
    preferred: bool,
) -> int:
    manifest = card.manifest
    retrieval = manifest.get("retrieval") or {}
    composition = manifest.get("composition") or {}
    text = _lower_text(
        [
            card.skill_id,
            card.category,
            manifest.get("description"),
            retrieval.get("summary"),
            retrieval.get("use_when"),
            retrieval.get("architecture_roles"),
            retrieval.get("objectives"),
            retrieval.get("model_families"),
            manifest.get("inductive_bias"),
            manifest.get("failure_signatures"),
            manifest.get("mutation_roles"),
        ]
    )
    score = 25 if preferred else 1
    for failure_mode in failure_modes:
        tokens = str(failure_mode).lower().replace("_", " ").split()
        score += sum(4 for token in tokens if token and token in text)
    query_tokens = _lower_text([diagnosis_report.get("query"), diagnosis_report.get("rationale")]).replace("_", " ").split()
    score += sum(3 for token in query_tokens if token and token in text)
    produced_keys = genome.produced_keys() | set(genome.constraints.required_inputs)
    requires = set(str(key) for key in card.input_keys)
    if requires and requires <= produced_keys:
        score += 12
    parent_skill_ids = {node.skill_id for node in genome.nodes}
    upstream = set(str(item) for item in composition.get("common_upstream", []) or [])
    downstream = set(str(item) for item in composition.get("common_downstream", []) or [])
    if upstream & parent_skill_ids:
        score += 8
    if downstream & parent_skill_ids:
        score += 4
    return score


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


def _extract_failure_modes(diagnosis_report: dict[str, Any]) -> list[str]:
    for key in ["failure_modes", "target_failure_modes", "failures"]:
        value = diagnosis_report.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
    value = diagnosis_report.get("failure_mode")
    return [str(value)] if value else []


def _load_structured(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        return json.load(f)


def _items_from_file(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "mutation_plans" in data:
            return list(data["mutation_plans"])
        if "open_ended_proposals" in data:
            return list(data["open_ended_proposals"])
        if "plans" in data:
            return list(data["plans"])
        return [data]
    raise TypeError(f"Unsupported proposal file payload: {type(data).__name__}")
