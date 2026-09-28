from recskill.evolution import RuleBasedSkillPlanner

from tests.evolution.helpers import base_genome, toy_library


def test_rule_based_planner_ranks_skill_library_candidates(tmp_path):
    library = toy_library(tmp_path)
    genome = base_genome()

    plans = RuleBasedSkillPlanner().plan(
        genome,
        library,
        {
            "failure_modes": ["cross_skill"],
            "target_node_id": "source",
            "query": "add a cross skill interaction",
        },
        budget=1,
    )

    assert len(plans) == 1
    assert plans[0].skill_id in {"cross_skill", "alt_cross_skill"}
    assert plans[0].metadata["selection_source"] == "skill_library"
    assert plans[0].edges[0].src_node_id == "source"
    assert plans[0].edges[0].dst_node_id == "NEW_NODE"
