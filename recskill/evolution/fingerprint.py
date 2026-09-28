from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .genome import SkillGenome


IGNORED_NODE_METADATA_KEYS = {
    "template",
    "created_at",
    "updated_at",
    "candidate_id",
    "elapsed_sec",
}


def stable_hash(payload: Any, *, length: int = 16) -> str:
    """Return a deterministic short hash for JSON-serializable payloads."""
    text = json.dumps(_normalize(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def genome_architecture_fingerprint(genome: SkillGenome) -> str:
    """Fingerprint the executable architecture, ignoring lineage and run metadata."""
    return stable_hash(normalize_genome_for_fingerprint(genome), length=20)


def macro_topology_fingerprint(genome: SkillGenome) -> str:
    """Fingerprint macro topology while ignoring concrete skill-id renames."""
    return stable_hash(normalize_genome_for_macro_topology(genome), length=20)


def normalize_genome_for_macro_topology(genome: SkillGenome) -> dict[str, Any]:
    nodes = [
        {
            "node_id": node.node_id,
            "category": node.category,
            "input_keys": list(node.input_keys),
            "output_keys": list(node.output_keys),
            "source": node.source,
        }
        for node in sorted(genome.nodes, key=lambda item: item.node_id)
    ]
    edges = [
        {
            "src_node_id": edge.src_node_id,
            "dst_node_id": edge.dst_node_id,
            "src_output_key": edge.src_output_key,
            "dst_input_key": edge.dst_input_key,
        }
        for edge in sorted(
            genome.edges,
            key=lambda item: (
                item.src_node_id,
                item.dst_node_id,
                item.src_output_key,
                item.dst_input_key,
            ),
        )
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "constraints": {
            "task_types": list(genome.constraints.task_types),
            "required_inputs": list(genome.constraints.required_inputs),
            "required_outputs": list(genome.constraints.required_outputs),
        },
    }


def normalize_genome_for_fingerprint(genome: SkillGenome) -> dict[str, Any]:
    nodes = []
    for node in sorted(genome.nodes, key=lambda item: item.node_id):
        node_metadata = {
            key: value
            for key, value in (node.metadata or {}).items()
            if key not in IGNORED_NODE_METADATA_KEYS
        }
        nodes.append(
            {
                "node_id": node.node_id,
                "skill_id": node.skill_id,
                "category": node.category,
                "params": _normalize(node.params),
                "input_keys": list(node.input_keys),
                "output_keys": list(node.output_keys),
                "task_types": list(node.task_types),
                "source": node.source,
                "metadata": _normalize(node_metadata),
            }
        )
    edges = [
        {
            "src_node_id": edge.src_node_id,
            "dst_node_id": edge.dst_node_id,
            "src_output_key": edge.src_output_key,
            "dst_input_key": edge.dst_input_key,
            "tensor_semantics": edge.tensor_semantics,
        }
        for edge in sorted(
            genome.edges,
            key=lambda item: (
                item.src_node_id,
                item.dst_node_id,
                item.src_output_key,
                item.dst_input_key,
                item.tensor_semantics or "",
            ),
        )
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "objectives": _normalize(genome.objectives),
        "constraints": {
            "task_types": list(genome.constraints.task_types),
            "required_inputs": list(genome.constraints.required_inputs),
            "required_outputs": list(genome.constraints.required_outputs),
            "shape_specs": _normalize(genome.constraints.shape_specs),
            "max_latency_ms": genome.constraints.max_latency_ms,
            "max_parameters": genome.constraints.max_parameters,
            "allow_disconnected": genome.constraints.allow_disconnected,
            "extras": _normalize(genome.constraints.extras),
        },
    }


def config_fingerprint(payload: Any) -> str:
    return stable_hash(payload, length=16)


def diff_payload(baseline: Any, current: Any) -> dict[str, Any]:
    """Return a shallow recursive diff payload for run-level hyperparameter context."""
    base = _normalize(baseline)
    cur = _normalize(current)
    return _diff_normalized(base, cur)


def _diff_normalized(baseline: Any, current: Any) -> Any:
    if isinstance(baseline, dict) and isinstance(current, dict):
        diff = {}
        for key in sorted(set(baseline) | set(current)):
            if key not in baseline:
                diff[key] = {"from": None, "to": current[key]}
            elif key not in current:
                diff[key] = {"from": baseline[key], "to": None}
            else:
                child = _diff_normalized(baseline[key], current[key])
                if child not in ({}, [], None):
                    diff[key] = child
        return diff
    if baseline != current:
        return {"from": baseline, "to": current}
    return {}


def _normalize(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _normalize(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize(item) for item in value)
    return value
