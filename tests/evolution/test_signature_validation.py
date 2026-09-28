from recskill.evolution import GenomeVerifier, MutationPlan, SkillEdge
from recskill.evolution.mutations import AddSkillMutation

from .helpers import base_genome, toy_library


def test_signature_and_dataflow_validation_accepts_valid_graph(tmp_path):
    library = toy_library(tmp_path)
    genome = AddSkillMutation(library).apply(
        base_genome(),
        MutationPlan(
            mutation_type="add_skill",
            skill_id="cross_skill",
            edges=[SkillEdge("source", "NEW_NODE", "x", "x")],
            metadata={"node_id": "cross"},
        ),
    ).genome
    genome.constraints.required_outputs = ["h"]

    result = GenomeVerifier(library).verify(genome)

    assert result["valid"], result


def test_signature_validation_rejects_missing_inputs(tmp_path):
    library = toy_library(tmp_path)
    genome = base_genome()
    genome.nodes[0].input_keys = ["not_available"]

    result = GenomeVerifier(library).verify(genome)

    assert not result["valid"]
    assert "missing required inputs" in result["issues"][0]
