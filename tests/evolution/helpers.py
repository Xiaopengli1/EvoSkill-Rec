from __future__ import annotations

from pathlib import Path

import yaml

from recskill.evolution import GenomeConstraints, SkillEdge, SkillGenome, SkillLibrary, SkillNode


def write_skill_card(root: Path, skill_id: str, category: str, inputs: list[str], outputs: list[str], task_types: list[str] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{skill_id}.skill.yaml"
    manifest = {
        "skill_id": skill_id,
        "name": skill_id,
        "skill_name": skill_id,
        "category": category,
        "task_types": task_types or ["ctr"],
        "input_signature": [{"name": key, "shape": ["batch_size", 4], "dtype": "float32"} for key in inputs],
        "output_signature": [{"name": key, "shape": ["batch_size", 4], "dtype": "float32"} for key in outputs],
        "composition": {
            "requires": inputs,
            "produces": outputs,
            "common_upstream": [],
            "common_downstream": [],
            "example_genome_fragment": {"skill": skill_id, "params": {}},
        },
        "retrieval": {
            "summary": skill_id,
            "use_when": [skill_id],
            "avoid_when": ["never"],
            "task_types": task_types or ["ctr"],
            "architecture_roles": [category],
            "input_modalities": inputs or ["none"],
            "output_semantics": outputs or ["none"],
            "objectives": ["test"],
            "aliases": [skill_id],
        },
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def toy_library(tmp_path: Path) -> SkillLibrary:
    root = tmp_path / "skills"
    write_skill_card(root, "source_skill", "embedding", ["raw"], ["x"])
    write_skill_card(root, "cross_skill", "interaction", ["x"], ["h"])
    write_skill_card(root, "alt_cross_skill", "interaction", ["x"], ["h"])
    write_skill_card(root, "head_skill", "head", ["h"], ["y"])
    write_skill_card(root, "adapter_skill", "adapter", ["h"], ["z"])
    return SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)


def base_genome() -> SkillGenome:
    return SkillGenome(
        nodes=[
            SkillNode(
                node_id="source",
                skill_id="source_skill",
                skill_name="source_skill",
                category="embedding",
                input_keys=["raw"],
                output_keys=["x"],
                task_types=["ctr"],
            )
        ],
        edges=[],
        constraints=GenomeConstraints(task_types=["ctr"], required_inputs=["raw"], required_outputs=["x"]),
    )


def two_node_genome() -> SkillGenome:
    genome = base_genome()
    genome.constraints.required_outputs = ["h"]
    genome.nodes.append(
        SkillNode(
            node_id="cross",
            skill_id="cross_skill",
            skill_name="cross_skill",
            category="interaction",
            input_keys=["x"],
            output_keys=["h"],
            task_types=["ctr"],
        )
    )
    genome.edges.append(SkillEdge("source", "cross", "x", "x"))
    return genome


SAFE_GENERATED_CODE = """
import torch
from torch import nn


class ResidualScale(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, inputs):
        return {"scaled_x": inputs["x"] * self.scale}
"""
