from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .evolution_memory import EvolutionMemory, EvolutionRecord
from .genome import MutationPlan, MutationResult, SkillEdge, SkillGenome
from .mutations import AddSkillMutation
from .skill_library import SkillCard, SkillLibrary
from .verification import ShapeTestRunner, SignatureInferer, SkillCardConsolidator, SkillSignature, SkillUnitTestRunner, StaticCodeChecker, TensorSpec


PROPOSAL_TYPES = {
    "NEW_SKILL_INVENTION",
    "LOCAL_CODE_SURGERY",
    "NEW_BRANCH_DESIGN",
    "NEW_EMBEDDING_DESIGN",
    "NEW_FUSION_DESIGN",
    "NEW_INTERACTION_DESIGN",
    "NEW_OBJECTIVE_DESIGN",
    "NEW_SEQUENCE_DESIGN",
    "NEW_TRAINING_MECHANISM",
    "NEW_ROUTING_OR_GATING_DESIGN",
}

OPEN_ENDED_STRUCTURAL_SCOPES = {"macro"}
DEFAULT_MAX_MACRO_PARAMS = 100000
_DEFAULT_CTR_SMOKE_CASE = {"num_fields": 3, "embedding_dim": 16, "domain_num": 3}
MACRO_ALLOWED_WIRING = {
    "replace_node",
    "insert_between",
    "branch_to_fusion",
    "replace_fusion",
    "generated_fusion",
    "fusion_replacement",
    "macro_fusion",
}


def normalize_open_ended_structural_scope(value: Any, *, allow_empty: bool = True) -> str:
    scope = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "global": "macro",
        "topology": "macro",
        "fusion": "macro",
        "fusion_replacement": "macro",
        "routing": "macro",
        "architecture": "macro",
    }
    scope = aliases.get(scope, scope)
    if not scope and allow_empty:
        return ""
    if scope not in OPEN_ENDED_STRUCTURAL_SCOPES:
        allowed = ", ".join(sorted(OPEN_ENDED_STRUCTURAL_SCOPES))
        raise ValueError(f"Unknown open-ended structural_scope: {value!r}; expected one of: {allowed}")
    return scope


@dataclass
class OpenEndedProposal:
    proposal_id: str
    proposal_type: str
    target_failure_mode: str
    architecture_hypothesis: str
    affected_genome_nodes: list[str]
    code: str
    expected_input_signature: list[dict[str, Any]]
    expected_output_signature: list[dict[str, Any]]
    expected_metric_improvement: dict[str, Any] | str | None = None
    expected_risks: list[str] = field(default_factory=list)
    unit_test_code: str | None = None
    test_spec: dict[str, Any] = field(default_factory=dict)
    ablation_plan: dict[str, Any] | str | None = None
    fallback_plan: dict[str, Any] | str | None = None
    author: str = "human"
    skill_id: str | None = None
    skill_name: str | None = None
    class_name: str | None = None
    task_types: list[str] = field(default_factory=list)
    init_params: dict[str, Any] = field(default_factory=dict)
    structural_scope: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenEndedProposal":
        proposal_type = data["proposal_type"]
        if proposal_type not in PROPOSAL_TYPES:
            raise ValueError(f"Unknown proposal_type: {proposal_type}")
        metadata = dict(data.get("metadata") or {})
        structural_scope = normalize_open_ended_structural_scope(
            data.get("structural_scope") or metadata.get("structural_scope") or metadata.get("scope") or "",
            allow_empty=True,
        )
        if structural_scope:
            metadata["structural_scope"] = structural_scope
        return cls(
            proposal_id=data["proposal_id"],
            proposal_type=proposal_type,
            target_failure_mode=data.get("target_failure_mode", ""),
            architecture_hypothesis=data.get("architecture_hypothesis", ""),
            affected_genome_nodes=list(data.get("affected_genome_nodes") or []),
            code=data.get("code", ""),
            expected_input_signature=list(data.get("expected_input_signature") or []),
            expected_output_signature=list(data.get("expected_output_signature") or []),
            expected_metric_improvement=data.get("expected_metric_improvement"),
            expected_risks=list(data.get("expected_risks") or []),
            unit_test_code=data.get("unit_test_code"),
            test_spec=dict(data.get("test_spec") or {}),
            ablation_plan=data.get("ablation_plan"),
            fallback_plan=data.get("fallback_plan"),
            author=data.get("author", "human"),
            skill_id=data.get("skill_id"),
            skill_name=data.get("skill_name"),
            class_name=data.get("class_name"),
            task_types=list(data.get("task_types") or []),
            init_params=dict(data.get("init_params") or {}),
            structural_scope=structural_scope,
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: str | Path) -> "OpenEndedProposal":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) if path.suffix in {".yaml", ".yml"} else json.load(f)
        return cls.from_dict(data or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeneratedSkillProposal(OpenEndedProposal):
    pass


class CodeSurgeryProposal(OpenEndedProposal):
    pass


class FusionDesignProposal(OpenEndedProposal):
    pass


class ObjectiveDesignProposal(OpenEndedProposal):
    pass


class TrainingMechanismProposal(OpenEndedProposal):
    pass


class RoutingDesignProposal(OpenEndedProposal):
    pass


@dataclass
class OpenEndedIngestionResult:
    success: bool
    proposal_id: str
    message: str
    validation_results: dict[str, Any] = field(default_factory=dict)
    code_path: str | None = None
    skill_card_path: str | None = None
    skill_id: str | None = None
    mutation_result: MutationResult | None = None
    genome: SkillGenome | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "proposal_id": self.proposal_id,
            "message": self.message,
            "validation_results": self.validation_results,
            "code_path": self.code_path,
            "skill_card_path": self.skill_card_path,
            "skill_id": self.skill_id,
            "mutation_result": self.mutation_result.to_dict() if self.mutation_result else None,
            "genome": self.genome.to_dict() if self.genome else None,
        }


class OpenEndedCodingBranch:
    """Validate generated code, consolidate it into a Skill Card, and insert it into a genome."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        generated_root: str | Path | None = None,
        skill_library: SkillLibrary | None = None,
        memory: EvolutionMemory | None = None,
        static_checker: StaticCodeChecker | None = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        self.generated_root = Path(generated_root) if generated_root is not None else self.repo_root / "recskill" / "generated_skills"
        self.skill_library = skill_library or SkillLibrary.from_repo(self.repo_root)
        self.memory = memory or EvolutionMemory(self.repo_root / "recskill" / "evolution_memory.jsonl")
        self.static_checker = static_checker or StaticCodeChecker()
        self.signature_inferer = SignatureInferer()
        self.unit_runner = SkillUnitTestRunner(static_checker=self.static_checker)
        self.shape_runner = ShapeTestRunner()
        self.consolidator = SkillCardConsolidator(repo_root=self.repo_root, generated_root=self.generated_root)

    def ingest(self, proposal: OpenEndedProposal, genome: SkillGenome) -> OpenEndedIngestionResult:
        validation_results: dict[str, Any] = {}
        parent_genome_id = genome.metadata.genome_id
        skill_id = proposal.skill_id or _sanitize_identifier(proposal.proposal_id)
        try:
            static_result = self.static_checker.check(proposal.code)
            validation_results["static_check"] = static_result
            if not static_result["passed"]:
                self._record(proposal, genome, None, "rejected", validation_results, {})
                return OpenEndedIngestionResult(False, proposal.proposal_id, "static check failed", validation_results, skill_id=skill_id)

            self.generated_root.mkdir(parents=True, exist_ok=True)
            code_path = self._write_generated_code(proposal, skill_id)
            signature = self.signature_inferer.infer_from_proposal(proposal)
            if proposal.class_name:
                signature.class_name = proposal.class_name
            if not signature.class_name:
                raise ValueError("Could not infer generated skill class_name")
            effective_params = _effective_generated_skill_params(genome, proposal)
            effective_signature = _effective_generated_skill_signature(genome, proposal, signature, effective_params)
            validation_results["signature"] = signature.to_dict()
            validation_results["effective_signature"] = effective_signature.to_dict()
            validation_results["effective_init_params"] = effective_params
            if (proposal.structural_scope or proposal.metadata.get("structural_scope")) == "macro":
                macro_result = _validate_macro_proposal(proposal, genome)
                validation_results["macro_validation"] = macro_result
                if not macro_result["passed"]:
                    self._record(proposal, genome, None, "rejected", validation_results, {"code_path": str(code_path)})
                    return OpenEndedIngestionResult(False, proposal.proposal_id, "macro validation failed", validation_results, code_path=str(code_path), skill_id=skill_id)

            smoke_signature = _specialize_signature_for_case(effective_signature, _DEFAULT_CTR_SMOKE_CASE)
            smoke_params = _specialize_params_for_case(effective_params, _DEFAULT_CTR_SMOKE_CASE)
            validation_results["smoke_test_init_params"] = smoke_params
            unit_result = self.unit_runner.run(
                code_path,
                smoke_signature.class_name,
                smoke_signature.input_signature,
                smoke_signature.output_signature,
                init_params=smoke_params,
                unit_test_code=proposal.unit_test_code,
            )
            validation_results["unit_tests"] = unit_result
            if not unit_result["passed"]:
                self._record(proposal, genome, None, "failed", validation_results, {"code_path": str(code_path)})
                return OpenEndedIngestionResult(False, proposal.proposal_id, "unit tests failed", validation_results, code_path=str(code_path), skill_id=skill_id)

            shape_result = self.shape_runner.run(
                code_path,
                smoke_signature.class_name,
                smoke_signature.input_signature,
                smoke_signature.output_signature,
                init_params=smoke_params,
            )
            validation_results["shape_tests"] = shape_result
            if not shape_result["passed"]:
                self._record(proposal, genome, None, "failed", validation_results, {"code_path": str(code_path)})
                return OpenEndedIngestionResult(False, proposal.proposal_id, "shape tests failed", validation_results, code_path=str(code_path), skill_id=skill_id)

            portability_result = _run_generated_skill_portability_tests(
                self.shape_runner,
                code_path,
                effective_signature.class_name,
                effective_signature,
                effective_params,
            )
            validation_results["portability_tests"] = portability_result

            card = self.consolidator.consolidate(proposal, code_path, effective_signature, validation_results, init_params=effective_params)
            self.skill_library.add_card(card.path)
            self.skill_library.register_generated_skill(card.skill_id)
            mutation_result = self._insert_generated_skill(genome, proposal, card, effective_signature, init_params=effective_params)
            status = "validated" if mutation_result.success else "failed"
            child_id = mutation_result.genome.metadata.genome_id if mutation_result.genome else None
            artifacts = {"code_path": str(code_path), "skill_card": str(card.path)}
            self._record(proposal, genome, child_id, status, validation_results, artifacts)
            return OpenEndedIngestionResult(
                success=mutation_result.success,
                proposal_id=proposal.proposal_id,
                message="proposal ingested" if mutation_result.success else mutation_result.message,
                validation_results=validation_results,
                code_path=str(code_path),
                skill_card_path=str(card.path),
                skill_id=card.skill_id,
                mutation_result=mutation_result,
                genome=mutation_result.genome,
            )
        except Exception as exc:
            self._record(proposal, genome, None, "failed", validation_results, {})
            return OpenEndedIngestionResult(False, proposal.proposal_id, f"{exc.__class__.__name__}: {exc}", validation_results, skill_id=skill_id)

    def _write_generated_code(self, proposal: OpenEndedProposal, skill_id: str) -> Path:
        base_path = self.generated_root / f"{skill_id}.py"
        path = base_path
        counter = 1
        while path.exists():
            path = self.generated_root / f"{skill_id}_{counter}.py"
            counter += 1
        path.write_text(proposal.code, encoding="utf-8")
        init_path = self.generated_root / "__init__.py"
        if not init_path.exists():
            init_path.write_text("", encoding="utf-8")
        return path

    def _insert_generated_skill(
        self,
        genome: SkillGenome,
        proposal: OpenEndedProposal,
        card: SkillCard,
        signature: SkillSignature,
        init_params: dict[str, Any] | None = None,
    ) -> MutationResult:
        node_id = proposal.metadata.get("node_id") or card.skill_id
        effective_params = dict(init_params if init_params is not None else _effective_generated_skill_params(genome, proposal))
        input_keys = [item.name for item in signature.input_signature]
        output_keys = [item.name for item in signature.output_signature]
        edges = _initial_generated_skill_edges(proposal, node_id)
        if not edges and proposal.affected_genome_nodes:
            parent_node = genome.get_node(proposal.affected_genome_nodes[0])
            if parent_node.output_keys and input_keys:
                src_output_key = proposal.metadata.get("src_output_key") or parent_node.output_keys[0]
                producer_node_id = parent_node.node_id
                direct_producer = _producer_node_for_key(genome, input_keys[0])
                if direct_producer is not None:
                    producer_node_id = direct_producer
                    src_output_key = input_keys[0]
                edges.append(
                    SkillEdge(
                        src_node_id=producer_node_id,
                        dst_node_id="NEW_NODE",
                        src_output_key=src_output_key,
                        dst_input_key=proposal.metadata.get("dst_input_key") or input_keys[0],
                        tensor_semantics=proposal.metadata.get("tensor_semantics"),
                    )
                )
        plan = MutationPlan(
            mutation_type="add_skill",
            skill_id=card.skill_id,
            skill_name=card.skill_name,
            category=card.category,
            params=effective_params,
            input_keys=input_keys,
            output_keys=output_keys,
            task_types=proposal.task_types,
            edges=edges,
            rationale=proposal.architecture_hypothesis,
            metadata={
                "node_id": node_id,
                "source": "generated_skill",
                "node_metadata": {
                    "proposal_id": proposal.proposal_id,
                    "structural_scope": proposal.structural_scope or proposal.metadata.get("structural_scope", ""),
                },
            },
        )
        result = AddSkillMutation(skill_library=self.skill_library).apply(genome, plan)
        if not result.success or result.genome is None:
            return result
        try:
            wire_generated_skill_outputs(result.genome, proposal, node_id, input_keys, output_keys)
            result.genome.assert_valid_graph()
        except Exception as exc:
            return MutationResult(
                False,
                None,
                f"post-insertion wiring failed: {exc.__class__.__name__}: {exc}",
                mutation=result.mutation,
            )
        return result

    def _record(
        self,
        proposal: OpenEndedProposal,
        parent_genome: SkillGenome,
        child_genome_id: str | None,
        status: str,
        validation_results: dict[str, Any],
        artifact_paths: dict[str, str],
    ) -> None:
        self.memory.append(
            EvolutionRecord(
                parent_genome_id=parent_genome.metadata.genome_id,
                child_genome_id=child_genome_id,
                mutation_type=proposal.proposal_type,
                proposal_id=proposal.proposal_id,
                status=status,
                validation_results=validation_results,
                failure_mode=proposal.target_failure_mode,
                rationale=proposal.architecture_hypothesis,
                artifact_paths=artifact_paths,
                task_type=proposal.task_types[0] if proposal.task_types else None,
            )
        )


def _sanitize_identifier(value: str) -> str:
    chars = [char if char.isalnum() or char == "_" else "_" for char in value.lower()]
    return "".join(chars).strip("_") or "generated_skill"


def _initial_generated_skill_edges(proposal: OpenEndedProposal, node_id: str) -> list[SkillEdge]:
    metadata = proposal.metadata or {}
    wiring = str(metadata.get("wiring") or metadata.get("integration") or "").lower().replace("-", "_")
    edges = [SkillEdge.from_dict(item) for item in metadata.get("edges", [])]
    if wiring in {
        "branch_to_fusion",
        "new_branch_to_fusion",
        "add_to_fusion",
        "append_to_fusion",
        "fusion_residual",
        "residual_logit",
        "reroute_logits",
        "replace_logits",
        "calibrate_logits",
        "post_logit",
        "replace_fusion",
        "generated_fusion",
        "fusion_replacement",
        "macro_fusion",
    }:
        new_node_refs = {"NEW_NODE", "$new_node", "${new_node_id}", node_id}
        edges = [edge for edge in edges if edge.dst_node_id in new_node_refs]
    return edges


def wire_generated_skill_outputs(
    genome: SkillGenome,
    proposal: OpenEndedProposal,
    node_id: str,
    input_keys: list[str],
    output_keys: list[str],
) -> None:
    if not output_keys:
        return
    metadata = proposal.metadata or {}
    wiring = str(metadata.get("wiring") or metadata.get("integration") or "").lower().replace("-", "_")
    if not wiring:
        wiring = _infer_wiring(input_keys, output_keys)
    if wiring in {"", "none", "manual"}:
        _add_missing_input_edges(genome, node_id, input_keys)
        return

    _add_missing_input_edges(genome, node_id, input_keys)
    output_key = output_keys[0]
    if wiring == "replace_node":
        _replace_node_with_generated(genome, proposal, node_id, output_keys)
        return
    if wiring == "insert_between":
        _insert_generated_between_nodes(genome, proposal, node_id, input_keys, output_keys)
        return
    if wiring in {"branch_to_fusion", "new_branch_to_fusion"}:
        attach_logit_to_active_terminal(genome, producer_node_id=node_id, logit_key=output_key)
        return
    if wiring in {"add_to_fusion", "append_to_fusion", "fusion_residual", "residual_logit"}:
        fusion_node_id = str(metadata.get("fusion_node_id") or "fusion")
        _append_output_to_fusion(genome, node_id, output_key, fusion_node_id=fusion_node_id)
        return
    if wiring in {"reroute_logits", "replace_logits", "calibrate_logits", "post_logit", "replace_fusion", "generated_fusion", "fusion_replacement", "macro_fusion"}:
        old_key = str(metadata.get("old_logits_key") or metadata.get("replace_logits_key") or _first_logit_key(input_keys) or _current_logits_key(genome) or "logits")
        reroute_prediction_and_loss_logits(genome, old_key=old_key, new_key=output_key, producer_node_id=node_id)
        if wiring in {"replace_fusion", "generated_fusion", "fusion_replacement", "macro_fusion"}:
            _prune_unreachable_from_outputs(genome)
        return
    raise ValueError(f"Unknown generated-skill wiring mode: {wiring}")


_wire_generated_skill_outputs = wire_generated_skill_outputs


def _infer_wiring(input_keys: list[str], output_keys: list[str]) -> str:
    lowered_inputs = {key.lower() for key in input_keys}
    lowered_outputs = {key.lower() for key in output_keys}
    if any("logit" in key for key in lowered_inputs) and any("logit" in key for key in lowered_outputs):
        return "reroute_logits"
    if any("logit" in key for key in lowered_outputs):
        return "add_to_fusion"
    return "none"


def _add_missing_input_edges(genome: SkillGenome, node_id: str, input_keys: list[str]) -> None:
    required_inputs = set(genome.constraints.required_inputs)
    existing = {(edge.dst_node_id, edge.dst_input_key) for edge in genome.edges}
    for input_key in input_keys:
        if (node_id, input_key) in existing or input_key in required_inputs:
            continue
        producer = _producer_node_for_key(genome, input_key)
        if producer is not None:
            genome.edges.append(SkillEdge(producer, node_id, input_key, input_key))


def _append_output_to_fusion(genome: SkillGenome, producer_node_id: str, output_key: str, *, fusion_node_id: str) -> None:
    fusion = genome.get_node(fusion_node_id)
    fusion.params.setdefault("input_keys", list(fusion.input_keys))
    fusion.params["input_keys"] = list(dict.fromkeys(list(fusion.params["input_keys"]) + [output_key]))
    fusion.input_keys = list(dict.fromkeys(list(fusion.input_keys) + [output_key]))
    edge = SkillEdge(producer_node_id, fusion_node_id, output_key, output_key)
    if edge not in genome.edges:
        genome.edges.append(edge)


def reroute_prediction_and_loss_logits(genome: SkillGenome, old_key: str, new_key: str, producer_node_id: str) -> None:
    prediction = genome.get_node("prediction")
    loss = genome.get_node("loss")
    prediction.params["input_key"] = new_key
    prediction.input_keys = [new_key]
    loss.params["logits_key"] = new_key
    non_logit_loss_inputs = [key for key in loss.input_keys if not _is_logit_tensor_key(key)]
    loss.input_keys = list(dict.fromkeys([new_key] + non_logit_loss_inputs))
    genome.edges = [
        edge
        for edge in genome.edges
        if not (
            edge.dst_node_id in {"prediction", "loss"}
            and (
                edge.dst_input_key == old_key
                or edge.src_output_key == old_key
                or _is_logit_tensor_key(edge.dst_input_key)
                or _is_logit_tensor_key(edge.src_output_key)
            )
        )
    ]
    genome.edges.extend(
        [
            SkillEdge(producer_node_id, "prediction", new_key, new_key),
            SkillEdge(producer_node_id, "loss", new_key, new_key),
        ]
    )


_reroute_prediction_and_loss_logits = reroute_prediction_and_loss_logits


def active_logits_terminal(genome: SkillGenome) -> tuple[str | None, str]:
    key = _current_logits_key(genome)
    return _producer_node_for_key(genome, key), key


def attach_logit_to_active_terminal(genome: SkillGenome, *, producer_node_id: str, logit_key: str) -> None:
    terminal_node_id, terminal_key = active_logits_terminal(genome)
    if terminal_node_id is None:
        reroute_prediction_and_loss_logits(genome, old_key=terminal_key, new_key=logit_key, producer_node_id=producer_node_id)
        return
    try:
        terminal = genome.get_node(terminal_node_id)
    except KeyError:
        terminal = None
    if terminal is not None and ("fusion" in terminal.category.lower() or "fusion" in terminal.skill_id.lower() or "fusion" in terminal.node_id.lower()):
        _append_output_to_fusion(genome, producer_node_id, logit_key, fusion_node_id=terminal_node_id)
        return
    reroute_prediction_and_loss_logits(genome, old_key=terminal_key, new_key=logit_key, producer_node_id=producer_node_id)


def _replace_node_with_generated(genome: SkillGenome, proposal: OpenEndedProposal, node_id: str, output_keys: list[str]) -> None:
    target_node_id = str((proposal.metadata or {}).get("target_node_id") or "")
    if not target_node_id:
        raise ValueError("replace_node wiring requires metadata.target_node_id")
    target = genome.get_node(target_node_id)
    target_output_keys = set(target.output_keys)
    if not target_output_keys <= set(output_keys):
        missing = sorted(target_output_keys - set(output_keys))
        raise ValueError(f"replace_node generated output_keys must preserve target outputs: {missing}")
    genome.nodes = [node for node in genome.nodes if node.node_id != target_node_id]
    rewritten_edges: list[SkillEdge] = []
    for edge in genome.edges:
        if edge.src_node_id == target_node_id:
            rewritten_edges.append(SkillEdge(node_id, edge.dst_node_id, edge.src_output_key, edge.dst_input_key, edge.tensor_semantics))
        elif edge.dst_node_id == target_node_id:
            if edge not in rewritten_edges:
                rewritten_edges.append(SkillEdge(edge.src_node_id, node_id, edge.src_output_key, edge.dst_input_key, edge.tensor_semantics))
        else:
            rewritten_edges.append(edge)
    genome.edges = _dedupe_edges(rewritten_edges)
    _sync_consumers_for_output_replacement(genome, target_node_id, node_id, target_output_keys)


def _insert_generated_between_nodes(
    genome: SkillGenome,
    proposal: OpenEndedProposal,
    node_id: str,
    input_keys: list[str],
    output_keys: list[str],
) -> None:
    metadata = proposal.metadata or {}
    upstream_node_id = str(metadata.get("upstream_node_id") or "")
    downstream_node_id = str(metadata.get("downstream_node_id") or "")
    if not upstream_node_id or not downstream_node_id:
        raise ValueError("insert_between wiring requires metadata.upstream_node_id and metadata.downstream_node_id")
    upstream = genome.get_node(upstream_node_id)
    downstream = genome.get_node(downstream_node_id)
    intercepted = [
        edge
        for edge in genome.edges
        if edge.src_node_id == upstream.node_id and edge.dst_node_id == downstream.node_id
    ]
    if not intercepted:
        raise ValueError(f"insert_between found no edge from {upstream_node_id} to {downstream_node_id}")
    input_key = input_keys[0] if input_keys else intercepted[0].dst_input_key
    output_key = output_keys[0] if output_keys else intercepted[0].src_output_key
    rewritten_edges = [edge for edge in genome.edges if edge not in intercepted]
    for edge in intercepted:
        rewritten_edges.append(SkillEdge(upstream.node_id, node_id, edge.src_output_key, input_key, edge.tensor_semantics))
        rewritten_edges.append(SkillEdge(node_id, downstream.node_id, output_key, edge.dst_input_key, edge.tensor_semantics))
    genome.edges = _dedupe_edges(rewritten_edges)


def _sync_consumers_for_output_replacement(
    genome: SkillGenome,
    old_node_id: str,
    new_node_id: str,
    output_keys: set[str],
) -> None:
    for node in genome.nodes:
        if node.node_id == new_node_id:
            continue
        if output_keys & set(node.input_keys):
            node.input_keys = list(dict.fromkeys(node.input_keys))
            if node.node_id == "loss" and "logits_key" in node.params:
                node.params["logits_key"] = node.params.get("logits_key")
            if node.node_id == "prediction" and "input_key" in node.params:
                node.params["input_key"] = node.params.get("input_key")


def _dedupe_edges(edges: list[SkillEdge]) -> list[SkillEdge]:
    deduped: list[SkillEdge] = []
    seen: set[SkillEdge] = set()
    for edge in edges:
        if edge in seen:
            continue
        deduped.append(edge)
        seen.add(edge)
    return deduped


def _prune_unreachable_from_outputs(genome: SkillGenome) -> list[str]:
    required_outputs = set(genome.constraints.required_outputs)
    producer_by_key: dict[str, str] = {}
    for node in genome.nodes:
        for output_key in node.output_keys:
            producer_by_key[output_key] = node.node_id

    needed_nodes: set[str] = set()
    stack = [producer_by_key[key] for key in required_outputs if key in producer_by_key]
    for sink_id in ("prediction", "loss"):
        if any(node.node_id == sink_id for node in genome.nodes):
            stack.append(sink_id)
    while stack:
        node_id = stack.pop()
        if node_id in needed_nodes:
            continue
        needed_nodes.add(node_id)
        for edge in genome.edges:
            if edge.dst_node_id == node_id:
                stack.append(edge.src_node_id)

    if not needed_nodes:
        return []
    removed = [node.node_id for node in genome.nodes if node.node_id not in needed_nodes]
    if not removed:
        return []
    removed_set = set(removed)
    genome.nodes = [node for node in genome.nodes if node.node_id not in removed_set]
    genome.edges = [
        edge
        for edge in genome.edges
        if edge.src_node_id not in removed_set and edge.dst_node_id not in removed_set
    ]
    return removed


def _validate_macro_proposal(proposal: OpenEndedProposal, genome: SkillGenome) -> dict[str, Any]:
    issues: list[str] = []
    metadata = proposal.metadata or {}
    wiring = str(metadata.get("wiring") or metadata.get("integration") or "").lower().replace("-", "_")
    if not wiring:
        issues.append("macro proposal requires metadata.wiring")
    elif wiring not in MACRO_ALLOWED_WIRING:
        issues.append(f"unknown macro wiring: {wiring}")

    judgment = metadata.get("macro_judgment") or {}
    required_judgment = [
        "current_architecture_family",
        "diagnosed_limitation",
        "target_architecture_transformation",
        "why_local_edit_is_insufficient",
        "parent_evidence",
        "proposal_insight",
        "preconditions",
        "wiring_plan",
        "ablation_plan",
    ]
    missing_judgment = [key for key in required_judgment if not judgment.get(key)]
    if missing_judgment:
        issues.append(f"macro_judgment missing required fields: {missing_judgment}")

    node_ids = genome.node_ids()
    if wiring == "replace_node":
        target = metadata.get("target_node_id")
        if not target:
            issues.append("replace_node requires metadata.target_node_id")
        elif target not in node_ids:
            issues.append(f"replace_node target_node_id is absent: {target}")
        elif str(target) in {"prediction", "loss"}:
            issues.append("replace_node cannot target terminal sink nodes prediction/loss")
    if wiring == "insert_between":
        upstream = metadata.get("upstream_node_id")
        downstream = metadata.get("downstream_node_id")
        if not upstream:
            issues.append("insert_between requires metadata.upstream_node_id")
        elif upstream not in node_ids:
            issues.append(f"insert_between upstream_node_id is absent: {upstream}")
        if not downstream:
            issues.append("insert_between requires metadata.downstream_node_id")
        elif downstream not in node_ids:
            issues.append(f"insert_between downstream_node_id is absent: {downstream}")

    available_keys = set(genome.constraints.required_inputs) | genome.produced_keys()
    input_keys = {str(item.get("name")) for item in proposal.expected_input_signature if item.get("name")}
    missing_inputs = sorted(input_keys - available_keys)
    if missing_inputs:
        issues.append(f"macro proposal input keys are not available in parent genome: {missing_inputs}")

    max_params = (proposal.metadata or {}).get("max_params") or (proposal.test_spec or {}).get("max_params")
    if max_params is not None:
        try:
            if int(max_params) > DEFAULT_MAX_MACRO_PARAMS:
                issues.append(f"macro proposal exceeds max parameter budget {DEFAULT_MAX_MACRO_PARAMS}: {max_params}")
        except Exception:
            issues.append(f"macro max_params is not an integer: {max_params}")

    return {"passed": not issues, "issues": issues, "wiring": wiring}


def _producer_node_for_key(genome: SkillGenome, key: str) -> str | None:
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


def _first_logit_key(keys: list[str]) -> str | None:
    for key in keys:
        if "logit" in key.lower():
            return key
    return None


def _is_logit_tensor_key(key: str) -> bool:
    lowered = str(key).lower()
    return lowered == "logits" or "logit" in lowered


def _effective_generated_skill_params(genome: SkillGenome, proposal: OpenEndedProposal) -> dict[str, Any]:
    params = dict(proposal.init_params or {})
    params = _resolve_symbolic_ctr_params(params, _ctr_runtime_case_from_genome(genome))
    wiring = str((proposal.metadata or {}).get("wiring") or (proposal.metadata or {}).get("integration") or "").lower().replace("-", "_")
    if wiring in {"reroute_logits", "replace_logits", "calibrate_logits", "post_logit"}:
        params["input_key"] = _current_logits_key(genome)
    return params


def _ctr_runtime_case_from_genome(genome: SkillGenome) -> dict[str, int]:
    try:
        embedding = genome.get_node("field_embedding")
        vocab_sizes = embedding.params.get("vocab_sizes") or []
        num_fields = len(vocab_sizes) if vocab_sizes else int(embedding.params.get("num_fields", _DEFAULT_CTR_SMOKE_CASE["num_fields"]))
        embedding_dim = int(embedding.params.get("embedding_dim", _DEFAULT_CTR_SMOKE_CASE["embedding_dim"]))
    except Exception:
        num_fields = int(_DEFAULT_CTR_SMOKE_CASE["num_fields"])
        embedding_dim = int(_DEFAULT_CTR_SMOKE_CASE["embedding_dim"])
    domain_num = _domain_num_from_genome(genome)
    domain_representation_dim = _domain_representation_dim_from_genome(genome)
    num_fields = max(1, int(num_fields))
    embedding_dim = max(1, int(embedding_dim))
    flat_input_dim = int((genome.metadata.extras or {}).get("flat_input_dim") or num_fields * embedding_dim)
    return {
        "num_fields": num_fields,
        "embedding_dim": embedding_dim,
        "input_dim": max(1, flat_input_dim),
        "flat_input_dim": max(1, flat_input_dim),
        "domain_num": max(1, int(domain_num)),
        "domain_representation_dim": max(1, int(domain_representation_dim)),
        "representation_dim": max(1, int(domain_representation_dim)),
    }


def _domain_num_from_genome(genome: SkillGenome) -> int:
    extras = genome.metadata.extras or {}
    for key in ("domain_num", "num_domains", "n_task"):
        value = extras.get(key)
        if value is not None:
            try:
                return max(1, int(value))
            except Exception:
                pass
    for node in genome.nodes:
        for key in ("domain_num", "num_domains", "n_task"):
            value = node.params.get(key)
            if value is not None:
                try:
                    return max(1, int(value))
                except Exception:
                    pass
    return int(_DEFAULT_CTR_SMOKE_CASE["domain_num"])


def _domain_representation_dim_from_genome(genome: SkillGenome) -> int:
    extras = genome.metadata.extras or {}
    for key in ("domain_representation_dim", "representation_dim", "expert_output_dim", "bottom_output_dim"):
        value = extras.get(key)
        if value is not None:
            try:
                return max(1, int(value))
            except Exception:
                pass
    try:
        return max(1, int(genome.get_node("domain_towers").params.get("input_dim") or 1))
    except Exception:
        pass
    for node in genome.nodes:
        for key in ("output_dim", "expert_dim", "bottom_output_dim", "input_dim"):
            value = node.params.get(key)
            if value is not None and ("domain" in node.node_id or "mmoe" in node.node_id or "ple" in node.node_id):
                try:
                    return max(1, int(value))
                except Exception:
                    pass
    return 1


def _resolve_symbolic_ctr_params(value: Any, case: dict[str, int]) -> Any:
    if isinstance(value, list):
        return [_resolve_symbolic_ctr_params(item, case) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_symbolic_ctr_params(item, case) for item in value)
    if isinstance(value, dict):
        return {key: _resolve_symbolic_ctr_params(item, case) for key, item in value.items()}
    return _specialize_param_value(value, case)


def _effective_generated_skill_signature(
    genome: SkillGenome,
    proposal: OpenEndedProposal,
    signature: SkillSignature,
    params: dict[str, Any],
) -> SkillSignature:
    wiring = str((proposal.metadata or {}).get("wiring") or (proposal.metadata or {}).get("integration") or "").lower().replace("-", "_")
    if wiring not in {"reroute_logits", "replace_logits", "calibrate_logits", "post_logit"}:
        return signature
    input_key = str(params.get("input_key") or _current_logits_key(genome))
    updated_inputs = []
    replaced = False
    for item in signature.input_signature:
        if not replaced and ("logit" in item.name.lower() or item.name == str((proposal.init_params or {}).get("input_key", ""))):
            updated_inputs.append(type(item)(name=input_key, shape=item.shape, dtype=item.dtype, semantic=item.semantic))
            replaced = True
        else:
            updated_inputs.append(item)
    if not updated_inputs:
        updated_inputs = [type(signature.output_signature[0] if signature.output_signature else _SignatureProxy())(name=input_key, shape=["batch_size", 1], dtype="float32", semantic=None)]
    return SkillSignature(input_signature=updated_inputs, output_signature=signature.output_signature, class_name=signature.class_name)


def _run_generated_skill_portability_tests(
    runner: ShapeTestRunner,
    code_path: str | Path,
    class_name: str,
    signature: SkillSignature,
    init_params: dict[str, Any],
) -> dict[str, Any]:
    if not _signature_has_symbolic_ctr_shapes(signature):
        return {"passed": False, "cases": [], "issues": ["input_signature does not use symbolic CTR dimensions"]}
    cases = [
        {"num_fields": 2, "embedding_dim": 16, "domain_num": 3, "domain_representation_dim": 16},
        {"num_fields": 3, "embedding_dim": 8, "domain_num": 3, "domain_representation_dim": 8},
        {"num_fields": 7, "embedding_dim": 16, "domain_num": 4, "domain_representation_dim": 32},
    ]
    results: list[dict[str, Any]] = []
    issues: list[str] = []
    for case in cases:
        case_signature = _specialize_signature_for_case(signature, case)
        case_params = _specialize_params_for_case(init_params, case)
        result = runner.run(
            code_path,
            class_name,
            case_signature.input_signature,
            case_signature.output_signature,
            init_params=case_params,
        )
        results.append({"case": case, "passed": result.get("passed", False), "result": result})
        if not result.get("passed", False):
            issues.extend(str(issue) for issue in result.get("issues", []))
    return {"passed": not issues, "cases": results, "issues": issues}


def _signature_has_symbolic_ctr_shapes(signature: SkillSignature) -> bool:
    symbols = {
        "num_fields",
        "embedding_dim",
        "input_dim",
        "flat_input_dim",
        "domain_num",
        "num_domains",
        "domain_representation_dim",
        "representation_dim",
    }
    return any(isinstance(dim, str) and dim in symbols for spec in signature.input_signature for dim in spec.shape)


def _specialize_signature_for_case(signature: SkillSignature, case: dict[str, int]) -> SkillSignature:
    return SkillSignature(
        input_signature=[_specialize_tensor_spec(spec, case) for spec in signature.input_signature],
        output_signature=[_specialize_tensor_spec(spec, case) for spec in signature.output_signature],
        class_name=signature.class_name,
    )


def _specialize_tensor_spec(spec: TensorSpec, case: dict[str, int]) -> TensorSpec:
    return TensorSpec(
        name=spec.name,
        shape=[_specialize_shape_dim(dim, case) for dim in spec.shape],
        dtype=spec.dtype,
        semantic=spec.semantic,
    )


def _specialize_shape_dim(dim: Any, case: dict[str, int]) -> Any:
    if dim == "num_fields":
        return int(case["num_fields"])
    if dim == "embedding_dim":
        return int(case["embedding_dim"])
    if dim in {"input_dim", "flat_input_dim"}:
        return int(case.get("flat_input_dim") or case.get("input_dim") or int(case["num_fields"]) * int(case["embedding_dim"]))
    if dim in {"domain_num", "num_domains"}:
        return int(case.get("domain_num", _DEFAULT_CTR_SMOKE_CASE["domain_num"]))
    if dim in {"domain_representation_dim", "representation_dim"}:
        return int(case.get("domain_representation_dim") or case.get("representation_dim") or case.get("embedding_dim", 4))
    return dim


def _specialize_params_for_case(params: dict[str, Any], case: dict[str, int]) -> dict[str, Any]:
    specialized = {}
    for key, value in params.items():
        if key == "num_fields":
            specialized[key] = int(case["num_fields"])
        elif key == "embedding_dim":
            specialized[key] = int(case["embedding_dim"])
        elif key in {"input_dim", "flat_input_dim"}:
            specialized[key] = int(case.get("flat_input_dim") or case.get("input_dim") or int(case["num_fields"]) * int(case["embedding_dim"]))
        elif key in {"domain_num", "num_domains"}:
            specialized[key] = int(case.get("domain_num", _DEFAULT_CTR_SMOKE_CASE["domain_num"]))
        elif key in {"domain_representation_dim", "representation_dim"}:
            specialized[key] = int(case.get("domain_representation_dim") or case.get("representation_dim") or case.get("embedding_dim", 4))
        else:
            specialized[key] = _specialize_param_value(value, case)
    return specialized


def _specialize_param_value(value: Any, case: dict[str, int]) -> Any:
    if isinstance(value, list):
        return [_specialize_param_value(item, case) for item in value]
    if isinstance(value, tuple):
        return tuple(_specialize_param_value(item, case) for item in value)
    if isinstance(value, dict):
        return {key: _specialize_param_value(item, case) for key, item in value.items()}
    if value in {"${num_fields}", "num_fields"}:
        return int(case["num_fields"])
    if value in {"${embedding_dim}", "embedding_dim"}:
        return int(case["embedding_dim"])
    if value in {"${input_dim}", "${flat_input_dim}", "${fusion_dim}", "input_dim", "flat_input_dim"}:
        return int(case.get("flat_input_dim") or case.get("input_dim") or int(case["num_fields"]) * int(case["embedding_dim"]))
    if value in {"${domain_num}", "${num_domains}", "domain_num", "num_domains"}:
        return int(case.get("domain_num", _DEFAULT_CTR_SMOKE_CASE["domain_num"]))
    if value in {
        "${domain_representation_dim}",
        "${representation_dim}",
        "domain_representation_dim",
        "representation_dim",
    }:
        return int(case.get("domain_representation_dim") or case.get("representation_dim") or case.get("embedding_dim", 4))
    return value


@dataclass
class _SignatureProxy:
    name: str = "logits"
    shape: list[str | int] = field(default_factory=lambda: ["batch_size", 1])
    dtype: str = "float32"
    semantic: str | None = None
