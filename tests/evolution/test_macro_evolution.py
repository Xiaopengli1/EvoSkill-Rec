"""Tests for macro code-space evolution: validation, wiring, pruning, dedup."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from recskill.evolution import EvolutionMemory, GenomeConstraints, SkillEdge, SkillGenome, SkillNode
from recskill.evolution.code_space import (
    MACRO_TRANSFORMATION_CATALOG,
    available_macro_transformations,
    ingest_open_ended_proposals,
)
from recskill.evolution.fingerprint import macro_topology_fingerprint
from recskill.evolution.open_ended import (
    DEFAULT_MAX_MACRO_PARAMS,
    OpenEndedProposal,
    _prune_unreachable_from_outputs,
    _validate_macro_proposal,
    active_logits_terminal,
    attach_logit_to_active_terminal,
    reroute_prediction_and_loss_logits,
    wire_generated_skill_outputs,
)


def _branched_genome() -> SkillGenome:
    """A small CTR-like genome with a fusion that sums two branch logits."""
    nodes = [
        SkillNode(
            node_id="field_embedding",
            skill_id="field_embedding",
            skill_name="field_embedding",
            category="embedding",
            input_keys=["raw"],
            output_keys=["field_embeddings"],
            task_types=["ctr"],
        ),
        SkillNode(
            node_id="fm",
            skill_id="fm_interaction",
            skill_name="fm_interaction",
            category="interaction",
            input_keys=["field_embeddings"],
            output_keys=["fm_logit"],
            task_types=["ctr"],
        ),
        SkillNode(
            node_id="deep_tower",
            skill_id="mlp_tower",
            skill_name="mlp_tower",
            category="tower",
            input_keys=["field_embeddings"],
            output_keys=["deep_logit"],
            task_types=["ctr"],
        ),
        SkillNode(
            node_id="fusion",
            skill_id="additive_fusion",
            skill_name="additive_fusion",
            category="fusion",
            input_keys=["fm_logit", "deep_logit"],
            output_keys=["logits"],
            params={"input_keys": ["fm_logit", "deep_logit"], "output_key": "logits"},
            task_types=["ctr"],
        ),
        SkillNode(
            node_id="prediction",
            skill_id="sigmoid_head",
            skill_name="sigmoid_head",
            category="head",
            input_keys=["logits"],
            output_keys=["prediction"],
            task_types=["ctr"],
        ),
        SkillNode(
            node_id="loss",
            skill_id="bce_loss",
            skill_name="bce_loss",
            category="loss",
            input_keys=["logits", "label"],
            output_keys=["loss"],
            params={"logits_key": "logits"},
            task_types=["ctr"],
        ),
    ]
    edges = [
        SkillEdge("field_embedding", "fm", "field_embeddings", "field_embeddings"),
        SkillEdge("field_embedding", "deep_tower", "field_embeddings", "field_embeddings"),
        SkillEdge("fm", "fusion", "fm_logit", "fm_logit"),
        SkillEdge("deep_tower", "fusion", "deep_logit", "deep_logit"),
        SkillEdge("fusion", "prediction", "logits", "logits"),
        SkillEdge("fusion", "loss", "logits", "logits"),
    ]
    constraints = GenomeConstraints(
        task_types=["ctr"],
        required_inputs=["raw", "label"],
        required_outputs=["prediction", "loss"],
    )
    return SkillGenome(nodes=nodes, edges=edges, constraints=constraints)


def _macro_proposal(
    *,
    proposal_id: str = "macro_proposal",
    wiring: str = "replace_node",
    target_node_id: str | None = "fm",
    upstream_node_id: str | None = None,
    downstream_node_id: str | None = None,
    omit_judgment: bool = False,
    extra_metadata: dict | None = None,
) -> OpenEndedProposal:
    metadata = {
        "node_id": "new_macro_node",
        "structural_scope": "macro",
        "wiring": wiring,
    }
    if not omit_judgment:
        metadata["macro_judgment"] = {
            "current_architecture_family": "DeepFM-style additive fusion",
            "diagnosed_limitation": "FM ceiling",
            "target_architecture_transformation": "Replace fm with attention",
            "why_local_edit_is_insufficient": "no per-sample re-weighting possible",
            "parent_evidence": ["fm consumes field_embeddings and feeds fm_logit into fusion"],
            "proposal_insight": "attention can allocate interaction capacity across fields instead of fixed FM terms",
            "preconditions": {"met": [], "missing": []},
            "wiring_plan": "replace fm with attention node",
            "ablation_plan": "compare to fm baseline",
        }
    if target_node_id is not None:
        metadata["target_node_id"] = target_node_id
    if upstream_node_id is not None:
        metadata["upstream_node_id"] = upstream_node_id
    if downstream_node_id is not None:
        metadata["downstream_node_id"] = downstream_node_id
    if extra_metadata:
        metadata.update(extra_metadata)
    return OpenEndedProposal(
        proposal_id=proposal_id,
        proposal_type="NEW_INTERACTION_DESIGN",
        target_failure_mode="interaction_ceiling",
        architecture_hypothesis="Replace FM with self-attention.",
        affected_genome_nodes=["fm"],
        code="<unused for unit-level validation>",
        expected_input_signature=[{"name": "field_embeddings", "shape": ["batch_size", 3, 4], "dtype": "float32"}],
        expected_output_signature=[{"name": "fm_logit", "shape": ["batch_size", 1], "dtype": "float32"}],
        skill_id="generated_macro_attention",
        class_name="GeneratedMacroAttention",
        task_types=["ctr"],
        init_params={},
        structural_scope="macro",
        metadata=metadata,
    )


def test_validate_macro_rejects_missing_judgment():
    genome = _branched_genome()
    proposal = _macro_proposal(omit_judgment=True)
    result = _validate_macro_proposal(proposal, genome)
    assert result["passed"] is False
    assert any("macro_judgment" in issue for issue in result["issues"])


def test_validate_macro_rejects_unknown_wiring():
    genome = _branched_genome()
    proposal = _macro_proposal(wiring="teleport_node")
    result = _validate_macro_proposal(proposal, genome)
    assert result["passed"] is False
    assert any("teleport_node" in issue for issue in result["issues"])


def test_validate_macro_rejects_local_logit_wiring():
    genome = _branched_genome()
    proposal = _macro_proposal(wiring="reroute_logits", target_node_id=None)
    result = _validate_macro_proposal(proposal, genome)
    assert result["passed"] is False
    assert any("reroute_logits" in issue for issue in result["issues"])


def test_validate_macro_rejects_replace_node_without_target():
    genome = _branched_genome()
    proposal = _macro_proposal(wiring="replace_node", target_node_id=None)
    result = _validate_macro_proposal(proposal, genome)
    assert result["passed"] is False
    assert any("target_node_id" in issue for issue in result["issues"])


def test_validate_macro_rejects_replace_node_targeting_sink():
    genome = _branched_genome()
    proposal = _macro_proposal(wiring="replace_node", target_node_id="prediction")
    result = _validate_macro_proposal(proposal, genome)
    assert result["passed"] is False
    assert any("sink" in issue for issue in result["issues"])


def test_validate_macro_rejects_insert_between_without_endpoints():
    genome = _branched_genome()
    proposal = _macro_proposal(wiring="insert_between", target_node_id=None)
    result = _validate_macro_proposal(proposal, genome)
    assert result["passed"] is False
    assert any("upstream_node_id" in issue for issue in result["issues"])


def test_validate_macro_rejects_input_keys_not_in_parent_genome():
    genome = _branched_genome()
    proposal = _macro_proposal(wiring="replace_node", target_node_id="fm")
    proposal.expected_input_signature = [{"name": "absent_key", "shape": ["batch_size", 1], "dtype": "float32"}]
    result = _validate_macro_proposal(proposal, genome)
    assert result["passed"] is False
    assert any("absent_key" in issue for issue in result["issues"])


def test_validate_macro_passes_for_valid_replace_node():
    genome = _branched_genome()
    proposal = _macro_proposal(wiring="replace_node", target_node_id="fm")
    result = _validate_macro_proposal(proposal, genome)
    assert result["passed"] is True, result["issues"]


def test_active_logits_terminal_finds_current_logit():
    genome = _branched_genome()
    producer, key = active_logits_terminal(genome)
    assert producer == "fusion"
    assert key == "logits"


def test_attach_logit_to_active_terminal_appends_to_existing_fusion():
    genome = _branched_genome()
    new_node = SkillNode(
        node_id="new_branch_logit",
        skill_id="generated_branch",
        skill_name="generated_branch",
        category="generated_skill",
        source="generated_skill",
        input_keys=["field_embeddings"],
        output_keys=["new_branch_logit"],
        task_types=["ctr"],
    )
    genome.nodes.append(new_node)
    genome.edges.append(SkillEdge("field_embedding", "new_branch_logit", "field_embeddings", "field_embeddings"))

    attach_logit_to_active_terminal(genome, producer_node_id="new_branch_logit", logit_key="new_branch_logit")

    fusion = genome.get_node("fusion")
    assert "new_branch_logit" in fusion.input_keys
    assert "new_branch_logit" in fusion.params.get("input_keys", [])
    assert any(
        edge.src_node_id == "new_branch_logit" and edge.dst_node_id == "fusion"
        for edge in genome.edges
    )


def test_reroute_prediction_and_loss_logits_swaps_terminal():
    genome = _branched_genome()
    new_node = SkillNode(
        node_id="generated_terminal",
        skill_id="generated_terminal",
        skill_name="generated_terminal",
        category="generated_skill",
        source="generated_skill",
        input_keys=["fm_logit", "deep_logit"],
        output_keys=["new_logits"],
        task_types=["ctr"],
    )
    genome.nodes.append(new_node)
    genome.edges.extend([
        SkillEdge("fm", "generated_terminal", "fm_logit", "fm_logit"),
        SkillEdge("deep_tower", "generated_terminal", "deep_logit", "deep_logit"),
    ])

    reroute_prediction_and_loss_logits(genome, old_key="logits", new_key="new_logits", producer_node_id="generated_terminal")

    prediction = genome.get_node("prediction")
    assert prediction.input_keys == ["new_logits"]
    assert genome.get_node("loss").params.get("logits_key") == "new_logits"
    edges_into_prediction = [edge for edge in genome.edges if edge.dst_node_id == "prediction"]
    assert len(edges_into_prediction) == 1
    assert edges_into_prediction[0].src_node_id == "generated_terminal"


def test_prune_unreachable_drops_orphans_after_replace_fusion():
    genome = _branched_genome()
    new_node = SkillNode(
        node_id="generated_terminal",
        skill_id="generated_terminal",
        skill_name="generated_terminal",
        category="generated_skill",
        source="generated_skill",
        input_keys=["fm_logit", "deep_logit"],
        output_keys=["new_logits"],
        task_types=["ctr"],
    )
    genome.nodes.append(new_node)
    genome.edges.extend([
        SkillEdge("fm", "generated_terminal", "fm_logit", "fm_logit"),
        SkillEdge("deep_tower", "generated_terminal", "deep_logit", "deep_logit"),
    ])
    reroute_prediction_and_loss_logits(genome, old_key="logits", new_key="new_logits", producer_node_id="generated_terminal")

    removed = _prune_unreachable_from_outputs(genome)

    assert "fusion" in removed
    assert "fusion" not in genome.node_ids()
    assert all(edge.src_node_id != "fusion" and edge.dst_node_id != "fusion" for edge in genome.edges)


def test_prune_unreachable_keeps_active_branches():
    genome = _branched_genome()
    removed = _prune_unreachable_from_outputs(genome)
    assert removed == []
    assert {"fusion", "fm", "deep_tower", "field_embedding"}.issubset(set(genome.node_ids()))


def test_macro_topology_fingerprint_ignores_skill_id_renames():
    genome_a = _branched_genome()
    genome_b = _branched_genome()
    # Same topology, different skill_id renames must collapse.
    genome_b.get_node("fm").skill_id = "renamed_fm_skill"
    fp_a = macro_topology_fingerprint(genome_a)
    fp_b = macro_topology_fingerprint(genome_b)
    assert fp_a == fp_b


def test_macro_topology_fingerprint_distinguishes_topology_changes():
    genome_a = _branched_genome()
    genome_b = _branched_genome()
    # Drop the fm branch entirely so the topology actually differs.
    genome_b.nodes = [node for node in genome_b.nodes if node.node_id != "fm"]
    genome_b.edges = [
        edge
        for edge in genome_b.edges
        if edge.src_node_id != "fm" and edge.dst_node_id != "fm"
    ]
    fusion = genome_b.get_node("fusion")
    fusion.input_keys = ["deep_logit"]
    fusion.params["input_keys"] = ["deep_logit"]
    assert macro_topology_fingerprint(genome_a) != macro_topology_fingerprint(genome_b)


def test_available_macro_transformations_marks_cue_overlap():
    profile = {
        "diagnostic_cues": [
            "interaction_path_is_limited_to_fm_pairwise_terms",
            "fixed_additive_fusion_limits_sample_adaptive_branch_weighting",
        ],
        "macro_preconditions": {
            "interaction_family_transform": {"met": True},
            "tower_family_transform": {"met": True},
            "embedding_family_transform": {"met": True},
            "routing_family_transform": {"met": True},
            "fusion_family_transform": {"met": True},
        },
    }
    annotated = available_macro_transformations(profile)
    assert annotated, "catalog should not be empty"
    by_kind = {entry["kind"]: entry for entry in annotated}
    assert "interaction_replace" in by_kind
    assert by_kind["interaction_replace"]["available"] is True
    assert "interaction_path_is_limited_to_fm_pairwise_terms" in by_kind["interaction_replace"]["cue_overlap"]
    assert (
        "fixed_additive_fusion_limits_sample_adaptive_branch_weighting"
        in by_kind["routing_introduce"]["cue_overlap"]
    )


def test_available_macro_transformations_marks_missing_preconditions():
    profile = {
        "diagnostic_cues": [],
        "macro_preconditions": {},
    }
    annotated = available_macro_transformations(profile)
    for entry in annotated:
        assert entry["available"] is False
        assert entry["preconditions_status"]["met"] == []


def test_macro_catalog_is_consistent_with_open_ended_wirings():
    from recskill.evolution.open_ended import MACRO_ALLOWED_WIRING

    for entry in MACRO_TRANSFORMATION_CATALOG:
        assert entry["wiring"] in MACRO_ALLOWED_WIRING, entry


# ---------------------------------------------------------------------------
# Wiring dispatch unit tests (no code generation, just genome surgery)
# ---------------------------------------------------------------------------


def _add_generated_node(
    genome: SkillGenome,
    node_id: str,
    *,
    input_keys: list[str],
    output_keys: list[str],
    incoming_edges: list[tuple[str, str, str]] = (),
) -> None:
    genome.nodes.append(
        SkillNode(
            node_id=node_id,
            skill_id=node_id,
            skill_name=node_id,
            category="generated_skill",
            source="generated_skill",
            input_keys=list(input_keys),
            output_keys=list(output_keys),
            task_types=["ctr"],
        )
    )
    for src, src_key, dst_key in incoming_edges:
        genome.edges.append(SkillEdge(src, node_id, src_key, dst_key))


def test_wire_replace_node_swaps_target():
    genome = _branched_genome()
    _add_generated_node(
        genome,
        "new_attn",
        input_keys=["field_embeddings"],
        output_keys=["fm_logit"],
        incoming_edges=[("field_embedding", "field_embeddings", "field_embeddings")],
    )
    proposal = OpenEndedProposal(
        proposal_id="p1",
        proposal_type="NEW_INTERACTION_DESIGN",
        target_failure_mode="x",
        architecture_hypothesis="x",
        affected_genome_nodes=["fm"],
        code="",
        expected_input_signature=[{"name": "field_embeddings", "shape": ["batch_size", 3, 4]}],
        expected_output_signature=[{"name": "fm_logit", "shape": ["batch_size", 1]}],
        skill_id="new_attn",
        class_name="NewAttn",
        task_types=["ctr"],
        init_params={},
        structural_scope="macro",
        metadata={"node_id": "new_attn", "wiring": "replace_node", "target_node_id": "fm", "structural_scope": "macro"},
    )

    wire_generated_skill_outputs(genome, proposal, "new_attn", ["field_embeddings"], ["fm_logit"])

    assert "fm" not in genome.node_ids()
    fusion = genome.get_node("fusion")
    assert "fm_logit" in fusion.input_keys
    assert any(edge.src_node_id == "new_attn" and edge.dst_node_id == "fusion" for edge in genome.edges)


def test_wire_insert_between_intercepts_edge():
    genome = _branched_genome()
    _add_generated_node(
        genome,
        "field_norm",
        input_keys=["field_embeddings"],
        output_keys=["field_embeddings"],
        incoming_edges=[],
    )
    proposal = OpenEndedProposal(
        proposal_id="p2",
        proposal_type="NEW_EMBEDDING_DESIGN",
        target_failure_mode="x",
        architecture_hypothesis="x",
        affected_genome_nodes=["field_embedding", "fm"],
        code="",
        expected_input_signature=[{"name": "field_embeddings", "shape": ["batch_size", 3, 4]}],
        expected_output_signature=[{"name": "field_embeddings", "shape": ["batch_size", 3, 4]}],
        skill_id="field_norm",
        class_name="FieldNorm",
        task_types=["ctr"],
        init_params={},
        structural_scope="macro",
        metadata={
            "node_id": "field_norm",
            "wiring": "insert_between",
            "upstream_node_id": "field_embedding",
            "downstream_node_id": "fm",
            "structural_scope": "macro",
        },
    )

    wire_generated_skill_outputs(genome, proposal, "field_norm", ["field_embeddings"], ["field_embeddings"])

    edges_into_fm = [edge for edge in genome.edges if edge.dst_node_id == "fm"]
    assert all(edge.src_node_id == "field_norm" for edge in edges_into_fm), edges_into_fm
    edges_into_norm = [edge for edge in genome.edges if edge.dst_node_id == "field_norm"]
    assert any(edge.src_node_id == "field_embedding" for edge in edges_into_norm)


def test_wire_branch_to_fusion_routes_to_active_terminal():
    genome = _branched_genome()
    _add_generated_node(
        genome,
        "cross_branch",
        input_keys=["field_embeddings"],
        output_keys=["cross_logit"],
        incoming_edges=[("field_embedding", "field_embeddings", "field_embeddings")],
    )
    proposal = OpenEndedProposal(
        proposal_id="p3",
        proposal_type="NEW_BRANCH_DESIGN",
        target_failure_mode="x",
        architecture_hypothesis="x",
        affected_genome_nodes=["field_embedding"],
        code="",
        expected_input_signature=[{"name": "field_embeddings", "shape": ["batch_size", 3, 4]}],
        expected_output_signature=[{"name": "cross_logit", "shape": ["batch_size", 1]}],
        skill_id="cross_branch",
        class_name="CrossBranch",
        task_types=["ctr"],
        init_params={},
        structural_scope="macro",
        metadata={
            "node_id": "cross_branch",
            "wiring": "branch_to_fusion",
            "upstream_node_id": "field_embedding",
            "structural_scope": "macro",
        },
    )

    wire_generated_skill_outputs(genome, proposal, "cross_branch", ["field_embeddings"], ["cross_logit"])

    fusion = genome.get_node("fusion")
    assert "cross_logit" in fusion.input_keys
    assert any(edge.src_node_id == "cross_branch" and edge.dst_node_id == "fusion" for edge in genome.edges)


def test_branch_to_fusion_ingestion_ignores_premature_fusion_edges(tmp_path):
    code = """
import torch
from torch import nn


class CrossBranch(nn.Module):
    def __init__(self, input_key="field_embeddings", output_key="cross_logit"):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs):
        x = inputs[self.input_key]
        return {self.output_key: x.mean(dim=(1, 2), keepdim=False).unsqueeze(-1) * self.scale}
"""
    proposal = OpenEndedProposal(
        proposal_id="branch_with_declared_fusion_edge",
        proposal_type="NEW_BRANCH_DESIGN",
        target_failure_mode="x",
        architecture_hypothesis="x",
        affected_genome_nodes=["field_embedding"],
        code=code,
        expected_input_signature=[{"name": "field_embeddings", "shape": ["batch_size", 3, 4], "dtype": "float32"}],
        expected_output_signature=[{"name": "cross_logit", "shape": ["batch_size", 1], "dtype": "float32"}],
        skill_id="branch_with_declared_fusion_edge",
        class_name="CrossBranch",
        task_types=["ctr"],
        init_params={"input_key": "field_embeddings", "output_key": "cross_logit"},
        structural_scope="macro",
        metadata={
            "node_id": "cross_branch",
            "wiring": "branch_to_fusion",
            "structural_scope": "macro",
            "edges": [
                {
                    "src_node_id": "field_embedding",
                    "dst_node_id": "NEW_NODE",
                    "src_output_key": "field_embeddings",
                    "dst_input_key": "field_embeddings",
                },
                {
                    "src_node_id": "NEW_NODE",
                    "dst_node_id": "fusion",
                    "src_output_key": "cross_logit",
                    "dst_input_key": "cross_logit",
                },
            ],
            "macro_judgment": {
                "current_architecture_family": "DeepFM-style additive fusion",
                "diagnosed_limitation": "missing cross branch",
                "target_architecture_transformation": "Add a generated branch to fusion",
                "why_local_edit_is_insufficient": "branch topology must change",
                "parent_evidence": ["fusion consumes existing branch logits"],
                "proposal_insight": "new branch can provide extra field interaction evidence",
                "preconditions": {"met": ["interaction_family_transform"], "missing": []},
                "wiring_plan": "append branch logit to fusion",
                "ablation_plan": "remove generated branch",
            },
        },
    )
    memory = EvolutionMemory(tmp_path / "memory.jsonl")
    result = ingest_open_ended_proposals(
        proposals=[proposal],
        parent=_branched_genome(),
        output_dir=tmp_path,
        memory=memory,
        repo_root=tmp_path,
    )

    assert len(result) == 1, [item.diagnostic() for item in result]
    genome = result[0].ingestion.genome
    fusion = genome.get_node("fusion")
    assert "cross_logit" in fusion.input_keys
    assert "cross_logit" in fusion.params["input_keys"]
    assert any(edge.src_node_id == "cross_branch" and edge.dst_node_id == "fusion" for edge in genome.edges)
    genome.assert_valid_graph()
