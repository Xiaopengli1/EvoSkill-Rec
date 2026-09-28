from recskill.evolution import (
    AddSkillMutation,
    HybridizeSkillMutation,
    MutationPlan,
    RemoveSkillMutation,
    ReplaceSkillMutation,
    RewireSkillMutation,
    SkillEdge,
    SkillNode,
    SpecializeSkillMutation,
)

from .helpers import base_genome, toy_library, two_node_genome


def test_add_remove_replace_rewire_hybridize_and_specialize(tmp_path):
    library = toy_library(tmp_path)
    genome = base_genome()

    add_result = AddSkillMutation(library).apply(
        genome,
        MutationPlan(
            mutation_type="add_skill",
            skill_id="cross_skill",
            edges=[SkillEdge("source", "NEW_NODE", "x", "x")],
            metadata={"node_id": "cross"},
        ),
    )
    assert add_result.success, add_result.message
    assert len(add_result.genome.nodes) == 2
    assert len(genome.nodes) == 1

    replace_result = ReplaceSkillMutation(library).apply(
        add_result.genome,
        MutationPlan(mutation_type="replace_skill", target_node_id="cross", skill_id="alt_cross_skill"),
    )
    assert replace_result.success, replace_result.message
    assert replace_result.genome.get_node("cross").skill_id == "alt_cross_skill"

    specialize_result = SpecializeSkillMutation(library).apply(
        replace_result.genome,
        MutationPlan(
            mutation_type="specialize_skill",
            skill_id="adapter_skill",
            edges=[SkillEdge("cross", "NEW_NODE", "h", "h")],
            metadata={"node_id": "scenario_adapter"},
        ),
    )
    assert specialize_result.success, specialize_result.message
    assert specialize_result.genome.get_node("scenario_adapter").category == "adapter"

    rewire_result = RewireSkillMutation(library).apply(
        specialize_result.genome,
        MutationPlan(
            mutation_type="rewire_skill",
            edges=[SkillEdge("source", "scenario_adapter", "x", "h")],
            metadata={"remove_edges": [{"src_node_id": "cross", "dst_node_id": "scenario_adapter", "src_output_key": "h", "dst_input_key": "h"}]},
        ),
    )
    assert rewire_result.success, rewire_result.message
    assert SkillEdge("source", "scenario_adapter", "x", "h") in rewire_result.genome.edges

    remove_genome = two_node_genome()
    remove_genome.constraints.required_outputs = ["x"]
    remove_result = RemoveSkillMutation(library).apply(
        remove_genome,
        MutationPlan(mutation_type="remove_skill", target_node_id="cross"),
    )
    assert remove_result.success, remove_result.message
    assert [node.node_id for node in remove_result.genome.nodes] == ["source"]

    hybrid_result = HybridizeSkillMutation(library).apply(
        genome,
        MutationPlan(
            mutation_type="hybridize_skill",
            nodes=[
                SkillNode(
                    node_id="head",
                    skill_id="head_skill",
                    skill_name="head_skill",
                    category="head",
                    input_keys=["x"],
                    output_keys=["y"],
                    task_types=["ctr"],
                )
            ],
            edges=[SkillEdge("source", "head", "x", "x")],
        ),
    )
    assert hybrid_result.success, hybrid_result.message
    assert hybrid_result.genome.get_node("head").skill_id == "head_skill"


def test_invalid_mutation_is_rejected(tmp_path):
    library = toy_library(tmp_path)
    result = AddSkillMutation(library).apply(
        base_genome(),
        MutationPlan(mutation_type="add_skill", skill_id="cross_skill"),
    )
    assert not result.success
    assert "disconnected" in result.message
