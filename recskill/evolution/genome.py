from __future__ import annotations

import copy
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .exceptions import GenomeValidationError


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class SkillNode:
    node_id: str
    skill_id: str
    skill_name: str
    category: str
    params: dict[str, Any] = field(default_factory=dict)
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    source: str = "existing_skill"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillNode":
        return cls(
            node_id=data["node_id"],
            skill_id=data.get("skill_id") or data.get("skill") or data.get("skill_name"),
            skill_name=data.get("skill_name") or data.get("skill_id") or data.get("skill"),
            category=data.get("category", "unknown"),
            params=dict(data.get("params") or {}),
            input_keys=list(data.get("input_keys") or []),
            output_keys=list(data.get("output_keys") or []),
            task_types=list(data.get("task_types") or []),
            source=data.get("source", "existing_skill"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SkillEdge:
    src_node_id: str
    dst_node_id: str
    src_output_key: str
    dst_input_key: str
    tensor_semantics: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillEdge":
        return cls(
            src_node_id=data["src_node_id"],
            dst_node_id=data["dst_node_id"],
            src_output_key=data["src_output_key"],
            dst_input_key=data["dst_input_key"],
            tensor_semantics=data.get("tensor_semantics"),
        )


@dataclass
class GenomeConstraints:
    task_types: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    required_outputs: list[str] = field(default_factory=list)
    shape_specs: dict[str, Any] = field(default_factory=dict)
    max_latency_ms: float | None = None
    max_parameters: int | None = None
    allow_disconnected: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenomeConstraints":
        data = data or {}
        known = {
            "task_types",
            "required_inputs",
            "required_outputs",
            "shape_specs",
            "max_latency_ms",
            "max_parameters",
            "allow_disconnected",
            "extras",
        }
        extras = dict(data.get("extras") or {})
        extras.update({key: value for key, value in data.items() if key not in known})
        return cls(
            task_types=list(data.get("task_types") or []),
            required_inputs=list(data.get("required_inputs") or []),
            required_outputs=list(data.get("required_outputs") or []),
            shape_specs=dict(data.get("shape_specs") or {}),
            max_latency_ms=data.get("max_latency_ms"),
            max_parameters=data.get("max_parameters"),
            allow_disconnected=bool(data.get("allow_disconnected", False)),
            extras=extras,
        )


@dataclass
class GenomeMutation:
    mutation_id: str = field(default_factory=lambda: new_id("mutation"))
    mutation_type: str = ""
    timestamp: str = field(default_factory=utc_now_iso)
    parent_genome_id: str | None = None
    child_genome_id: str | None = None
    plan_id: str | None = None
    description: str = ""
    status: str = "applied"
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenomeMutation":
        return cls(
            mutation_id=data.get("mutation_id") or new_id("mutation"),
            mutation_type=data.get("mutation_type", ""),
            timestamp=data.get("timestamp", utc_now_iso()),
            parent_genome_id=data.get("parent_genome_id"),
            child_genome_id=data.get("child_genome_id"),
            plan_id=data.get("plan_id"),
            description=data.get("description", ""),
            status=data.get("status", "applied"),
            details=dict(data.get("details") or {}),
        )


@dataclass
class GenomeMetadata:
    genome_id: str = field(default_factory=lambda: new_id("genome"))
    parent_genome_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    lineage: list[str] = field(default_factory=list)
    mutation_history: list[GenomeMutation] = field(default_factory=list)
    evaluation_results: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenomeMetadata":
        data = data or {}
        known = {
            "genome_id",
            "parent_genome_ids",
            "created_at",
            "updated_at",
            "lineage",
            "mutation_history",
            "evaluation_results",
            "tags",
            "extras",
        }
        extras = dict(data.get("extras") or {})
        extras.update({key: value for key, value in data.items() if key not in known})
        return cls(
            genome_id=data.get("genome_id") or new_id("genome"),
            parent_genome_ids=list(data.get("parent_genome_ids") or []),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
            lineage=list(data.get("lineage") or []),
            mutation_history=[GenomeMutation.from_dict(item) for item in data.get("mutation_history", [])],
            evaluation_results=dict(data.get("evaluation_results") or {}),
            tags=list(data.get("tags") or []),
            extras=extras,
        )


@dataclass
class MutationPlan:
    mutation_type: str
    plan_id: str = field(default_factory=lambda: new_id("plan"))
    target_node_id: str | None = None
    skill_id: str | None = None
    skill_name: str | None = None
    category: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    nodes: list[SkillNode] = field(default_factory=list)
    edges: list[SkillEdge] = field(default_factory=list)
    parent_genomes: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MutationPlan":
        return cls(
            mutation_type=data["mutation_type"],
            plan_id=data.get("plan_id") or data.get("id") or new_id("plan"),
            target_node_id=data.get("target_node_id"),
            skill_id=data.get("skill_id") or data.get("skill"),
            skill_name=data.get("skill_name") or data.get("skill_id") or data.get("skill"),
            category=data.get("category"),
            params=dict(data.get("params") or {}),
            input_keys=list(data.get("input_keys") or []),
            output_keys=list(data.get("output_keys") or []),
            task_types=list(data.get("task_types") or []),
            nodes=[SkillNode.from_dict(item) for item in data.get("nodes", [])],
            edges=[SkillEdge.from_dict(item) for item in data.get("edges", [])],
            parent_genomes=list(data.get("parent_genomes") or []),
            rationale=data.get("rationale", ""),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MutationResult:
    success: bool
    genome: "SkillGenome | None" = None
    message: str = ""
    validation_results: dict[str, Any] = field(default_factory=dict)
    mutation: GenomeMutation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "genome": self.genome.to_dict() if self.genome else None,
            "validation_results": self.validation_results,
            "mutation": asdict(self.mutation) if self.mutation else None,
        }


@dataclass
class SkillGenome:
    nodes: list[SkillNode] = field(default_factory=list)
    edges: list[SkillEdge] = field(default_factory=list)
    objectives: list[dict[str, Any]] = field(default_factory=list)
    constraints: GenomeConstraints = field(default_factory=GenomeConstraints)
    metadata: GenomeMetadata = field(default_factory=GenomeMetadata)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillGenome":
        return cls(
            nodes=[SkillNode.from_dict(item) for item in data.get("nodes", [])],
            edges=[SkillEdge.from_dict(item) for item in data.get("edges", [])],
            objectives=list(data.get("objectives") or []),
            constraints=GenomeConstraints.from_dict(data.get("constraints")),
            metadata=GenomeMetadata.from_dict(data.get("metadata")),
        )

    @classmethod
    def from_json(cls, text: str) -> "SkillGenome":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> "SkillGenome":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            if path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(f) or {}
            else:
                data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            if path.suffix in {".yaml", ".yml"}:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
            else:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
                f.write("\n")

    def clone(self, preserve_id: bool = False) -> "SkillGenome":
        cloned = SkillGenome.from_dict(copy.deepcopy(self.to_dict()))
        if preserve_id:
            return cloned
        parent_id = self.metadata.genome_id
        cloned.metadata.genome_id = new_id("genome")
        cloned.metadata.parent_genome_ids = list(dict.fromkeys(cloned.metadata.parent_genome_ids + [parent_id]))
        cloned.metadata.lineage = list(dict.fromkeys(cloned.metadata.lineage + [parent_id]))
        cloned.metadata.created_at = utc_now_iso()
        cloned.metadata.updated_at = cloned.metadata.created_at
        return cloned

    def get_node(self, node_id: str) -> SkillNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"Unknown genome node '{node_id}'")

    def node_ids(self) -> set[str]:
        return {node.node_id for node in self.nodes}

    def produced_keys(self) -> set[str]:
        keys = set()
        for node in self.nodes:
            keys.update(node.output_keys)
        return keys

    def consumed_keys(self) -> set[str]:
        keys = set()
        for node in self.nodes:
            keys.update(node.input_keys)
        return keys

    def topological_order(self) -> list[str]:
        node_ids = self.node_ids()
        indegree = {node_id: 0 for node_id in node_ids}
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            if edge.src_node_id in node_ids and edge.dst_node_id in node_ids:
                indegree[edge.dst_node_id] += 1
                outgoing[edge.src_node_id].append(edge.dst_node_id)

        ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for dst in sorted(outgoing[node_id]):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
                    ready.sort()
        if len(order) != len(node_ids):
            raise GenomeValidationError("Genome graph contains a cycle")
        return order

    def validate_graph(self) -> list[str]:
        issues: list[str] = []
        node_ids = [node.node_id for node in self.nodes]
        duplicates = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
        for node_id in duplicates:
            issues.append(f"Duplicate node_id: {node_id}")

        node_by_id = {node.node_id: node for node in self.nodes}
        for node in self.nodes:
            if not node.skill_id:
                issues.append(f"Node {node.node_id} is missing skill_id")
            if not node.skill_name:
                issues.append(f"Node {node.node_id} is missing skill_name")
            if not node.category:
                issues.append(f"Node {node.node_id} is missing category")

        for edge in self.edges:
            src = node_by_id.get(edge.src_node_id)
            dst = node_by_id.get(edge.dst_node_id)
            if src is None:
                issues.append(f"Edge references missing src_node_id: {edge.src_node_id}")
                continue
            if dst is None:
                issues.append(f"Edge references missing dst_node_id: {edge.dst_node_id}")
                continue
            if src.output_keys and edge.src_output_key not in src.output_keys:
                issues.append(f"Edge {edge.src_node_id}->{edge.dst_node_id} uses unknown source output '{edge.src_output_key}'")
            if dst.input_keys and edge.dst_input_key not in dst.input_keys:
                issues.append(f"Edge {edge.src_node_id}->{edge.dst_node_id} uses unknown destination input '{edge.dst_input_key}'")

        try:
            self.topological_order()
        except GenomeValidationError as exc:
            issues.append(str(exc))

        if self.nodes and not self.constraints.allow_disconnected and not self._is_weakly_connected():
            issues.append("Genome graph is disconnected")

        required_outputs = set(self.constraints.required_outputs)
        missing_outputs = sorted(required_outputs - self.produced_keys())
        if missing_outputs:
            issues.append(f"Required outputs are not produced: {missing_outputs}")

        return issues

    def assert_valid_graph(self) -> None:
        issues = self.validate_graph()
        if issues:
            raise GenomeValidationError("; ".join(issues))

    def record_mutation(self, mutation: GenomeMutation) -> None:
        self.metadata.mutation_history.append(mutation)
        if mutation.parent_genome_id:
            self.metadata.parent_genome_ids = list(dict.fromkeys(self.metadata.parent_genome_ids + [mutation.parent_genome_id]))
            self.metadata.lineage = list(dict.fromkeys(self.metadata.lineage + [mutation.parent_genome_id]))
        self.metadata.updated_at = utc_now_iso()

    def _is_weakly_connected(self) -> bool:
        if len(self.nodes) <= 1:
            return True
        node_ids = self.node_ids()
        neighbors = {node_id: set() for node_id in node_ids}
        for edge in self.edges:
            if edge.src_node_id in node_ids and edge.dst_node_id in node_ids:
                neighbors[edge.src_node_id].add(edge.dst_node_id)
                neighbors[edge.dst_node_id].add(edge.src_node_id)
        start = next(iter(node_ids))
        seen = {start}
        stack = [start]
        while stack:
            node_id = stack.pop()
            for neighbor in neighbors[node_id]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return seen == node_ids
