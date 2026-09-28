from pathlib import Path

import yaml

from recskill.evolution import EvolutionMemory, GenomeConstraints, OpenEndedCodingBranch, OpenEndedProposal, SkillEdge, SkillGenome, SkillNode

from .helpers import SAFE_GENERATED_CODE


def test_safe_generated_skill_is_ingested_registered_and_inserted(tmp_path):
    genome = SkillGenome(
        nodes=[
            SkillNode(
                node_id="source",
                skill_id="source",
                skill_name="source",
                category="embedding",
                output_keys=["x"],
                source="generated_skill",
            )
        ],
        constraints=GenomeConstraints(allow_disconnected=False),
    )
    proposal = OpenEndedProposal.from_dict(
        {
            "proposal_id": "safe_scale",
            "proposal_type": "NEW_SKILL_INVENTION",
            "target_failure_mode": "calibration_gap",
            "architecture_hypothesis": "Scale a representation with a trainable residual weight.",
            "affected_genome_nodes": ["source"],
            "code": SAFE_GENERATED_CODE,
            "expected_input_signature": [{"name": "x", "shape": ["batch_size", 4], "dtype": "float32"}],
            "expected_output_signature": [{"name": "scaled_x", "shape": ["batch_size", 4], "dtype": "float32"}],
            "expected_metric_improvement": {"auc": 0.001},
            "expected_risks": ["overfitting"],
            "ablation_plan": "remove scale",
            "fallback_plan": "rollback",
            "author": "human",
            "skill_id": "safe_scale_skill",
            "class_name": "ResidualScale",
            "task_types": ["ctr"],
            "init_params": {"scale": 0.5},
        }
    )
    memory = EvolutionMemory(tmp_path / "memory.jsonl")
    branch = OpenEndedCodingBranch(generated_root=tmp_path / "generated_skills", memory=memory)

    result = branch.ingest(proposal, genome)

    assert result.success, result.to_dict()
    assert result.skill_card_path
    assert result.genome.get_node("safe_scale_skill").source == "generated_skill"
    assert memory.get_generated_skills(task_type="ctr")


def test_rerouted_generated_skill_card_uses_current_logits_key(tmp_path):
    code = """
import torch
from torch import nn


class LogitTanhResidual(nn.Module):
    def __init__(self, input_key="logits", output_key="next_logits", initial_scale=0.1):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, inputs):
        logits = inputs[self.input_key]
        return {self.output_key: logits + self.scale * torch.tanh(logits)}
"""
    genome = SkillGenome(
        nodes=[
            SkillNode(
                node_id="existing_calibrator",
                skill_id="existing_calibrator",
                skill_name="existing_calibrator",
                category="adapter",
                output_keys=["previous_logits"],
                source="generated_skill",
            ),
            SkillNode(
                node_id="prediction",
                skill_id="sigmoid_prediction",
                skill_name="sigmoid_prediction",
                category="head",
                params={"input_key": "previous_logits", "output_key": "prediction"},
                input_keys=["previous_logits"],
                output_keys=["prediction"],
            ),
            SkillNode(
                node_id="loss",
                skill_id="bce_loss",
                skill_name="bce_loss",
                category="loss",
                params={"logits_key": "previous_logits", "labels_key": "labels", "output_key": "loss"},
                input_keys=["previous_logits", "labels"],
                output_keys=["loss"],
            ),
        ],
        edges=[
            SkillEdge("existing_calibrator", "prediction", "previous_logits", "previous_logits"),
            SkillEdge("existing_calibrator", "loss", "previous_logits", "previous_logits"),
        ],
        constraints=GenomeConstraints(task_types=["ctr"], required_inputs=["labels"], required_outputs=["loss"]),
    )
    proposal = OpenEndedProposal.from_dict(
        {
            "proposal_id": "next_logit_residual",
            "proposal_type": "NEW_ROUTING_OR_GATING_DESIGN",
            "target_failure_mode": "calibration_gap",
            "architecture_hypothesis": "Apply a small residual transformation to the current logits.",
            "affected_genome_nodes": ["existing_calibrator"],
            "code": code,
            "expected_input_signature": [{"name": "logits", "shape": ["batch_size", 1], "dtype": "float32"}],
            "expected_output_signature": [{"name": "next_logits", "shape": ["batch_size", 1], "dtype": "float32"}],
            "expected_metric_improvement": {"auc": 0.001},
            "expected_risks": ["overfitting"],
            "author": "human",
            "skill_id": "next_logit_residual",
            "class_name": "LogitTanhResidual",
            "task_types": ["ctr"],
            "init_params": {"input_key": "logits", "output_key": "next_logits", "initial_scale": 0.1},
            "metadata": {"node_id": "next_logit_residual", "wiring": "reroute_logits", "output_key": "next_logits"},
        }
    )
    memory = EvolutionMemory(tmp_path / "memory.jsonl")
    branch = OpenEndedCodingBranch(generated_root=tmp_path / "generated_skills", memory=memory)

    result = branch.ingest(proposal, genome)

    assert result.success, result.to_dict()
    manifest = yaml.safe_load(Path(result.skill_card_path).read_text(encoding="utf-8"))
    assert manifest["input_signature"][0]["name"] == "previous_logits"
    assert manifest["composition"]["requires"] == ["previous_logits"]
    assert manifest["composition"]["example_genome_fragment"]["params"]["input_key"] == "previous_logits"
    node = result.genome.get_node("next_logit_residual")
    assert node.input_keys == ["previous_logits"]
    assert node.params["input_key"] == "previous_logits"
    assert result.genome.get_node("loss").params["logits_key"] == "next_logits"


def test_unsafe_generated_skill_is_rejected_and_recorded(tmp_path):
    proposal = OpenEndedProposal.from_dict(
        {
            "proposal_id": "unsafe",
            "proposal_type": "NEW_SKILL_INVENTION",
            "target_failure_mode": "unknown",
            "architecture_hypothesis": "Unsafe code should not pass.",
            "affected_genome_nodes": [],
            "code": "import requests\nrequests.get('https://example.com')\n",
            "expected_input_signature": [{"name": "x", "shape": ["batch_size", 4], "dtype": "float32"}],
            "expected_output_signature": [{"name": "y", "shape": ["batch_size", 4], "dtype": "float32"}],
            "expected_metric_improvement": None,
            "expected_risks": [],
            "ablation_plan": "",
            "fallback_plan": "",
            "author": "human",
        }
    )
    memory = EvolutionMemory(tmp_path / "memory.jsonl")
    branch = OpenEndedCodingBranch(generated_root=tmp_path / "generated_skills", memory=memory)

    result = branch.ingest(proposal, SkillGenome())

    assert not result.success
    assert result.message == "static check failed"
    assert memory.get_failed_mutations(failure_mode="unknown")
