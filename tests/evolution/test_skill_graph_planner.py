from recskill.evolution import (
    CTRDatasetBundle,
    CTRGenomeConfig,
    CTRWorkflowConfig,
    CTREvolutionConfig,
    GenomeVerifier,
    SkillGenomeCompiler,
    SkillGraphPlannerConfig,
    SkillGraphSearchPlanner,
    build_default_ctr_genome,
)
from recskill.evolution.ctr_workflow import generate_ctr_candidate_population
from recskill.evolution.skill_library import SkillLibrary


def _deepfm_parent():
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id", "gender"],
        vocab_sizes=[4, 8, 3],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    return build_default_ctr_genome(bundle, genome_cfg), genome_cfg


def test_skill_graph_planner_filters_relevant_cards_and_generates_valid_candidates():
    parent, _ = _deepfm_parent()
    planner = SkillGraphSearchPlanner(
        config=SkillGraphPlannerConfig(
            card_top_k=10,
            per_category_top_k=5,
            beam_width=6,
            max_depth=1,
            candidate_pool_size=12,
            validate_compile=True,
        ),
        seed=7,
    )

    cards = planner.select_relevant_cards(parent, failure_modes=["high_order_interaction_underfitting"])
    candidates = planner.plan(parent, failure_modes=["high_order_interaction_underfitting"], budget=4, round_idx=0)

    assert {card.skill_id for card in cards} & {"crossnet_mix", "crossnet_v2", "afm_attention_pooling", "bilinear_interaction"}
    assert candidates
    assert {candidate.operation for candidate in candidates} <= {"add_branch", "replace_node", "insert_after", "insert_before"}
    for candidate in candidates:
        assert candidate.actions
        assert candidate.architecture_fingerprint
        GenomeVerifier().assert_valid(candidate.genome)
        SkillGenomeCompiler().compile(candidate.genome)


def test_skill_graph_planner_beam_search_can_build_multi_action_candidate():
    parent, _ = _deepfm_parent()
    planner = SkillGraphSearchPlanner(
        config=SkillGraphPlannerConfig(
            card_top_k=8,
            per_category_top_k=5,
            beam_width=8,
            max_depth=2,
            candidate_pool_size=20,
            validate_compile=True,
        ),
        seed=11,
    )

    candidates = planner.plan(parent, failure_modes=["high_order_interaction_underfitting"], budget=8, round_idx=1)

    assert any(candidate.operation == "hybridize" and len(candidate.actions) == 2 for candidate in candidates)
    for candidate in candidates:
        GenomeVerifier().assert_valid(candidate.genome)
        SkillGenomeCompiler().compile(candidate.genome)


def test_skill_graph_planner_strategy_integrates_with_ctr_candidate_generation():
    parent, genome_cfg = _deepfm_parent()
    workflow_cfg = CTRWorkflowConfig(
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            strategy="skill_graph_planner",
            candidate_budget=4,
            code_space_probability=0.0,
            skill_graph_planner=SkillGraphPlannerConfig(
                card_top_k=10,
                beam_width=6,
                max_depth=1,
                candidate_pool_size=12,
                validate_compile=True,
                allow_template_fallback=False,
            ),
        ),
    )

    specs = generate_ctr_candidate_population([parent], workflow_cfg)

    assert specs
    assert len(specs) <= 4
    assert all(spec.evolution_space == "skill_space" for spec in specs)
    assert all(spec.operation in {"add_branch", "replace_node", "insert_after", "insert_before"} for spec in specs)
    assert all(spec.mutation_type.startswith("skill_graph_") for spec in specs)
    for spec in specs:
        GenomeVerifier().assert_valid(spec.genome)
        SkillGenomeCompiler().compile(spec.genome)


def test_skill_graph_planner_can_reuse_promoted_generated_skill_card(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    (root / "field_residual.py").write_text(
        """
import torch
from torch import nn


class FieldResidual(nn.Module):
    def __init__(self, input_key="field_embeddings", output_key="field_residual_logit"):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs):
        x = inputs[self.input_key]
        return {self.output_key: x.mean(dim=(1, 2), keepdim=False).unsqueeze(-1) * self.scale}
""",
        encoding="utf-8",
    )
    (root / "field_residual.skill.yaml").write_text(
        """
skill_id: field_residual
name: field_residual
skill_name: field_residual
category: adapter
implementation_path: field_residual.py
class_name: FieldResidual
task_types: [ctr]
input_signature:
  - name: field_embeddings
    shape: [batch_size, num_fields, embedding_dim]
    dtype: float32
output_signature:
  - name: field_residual_logit
    shape: [batch_size, 1]
    dtype: float32
composition:
  requires: [field_embeddings]
  produces: [field_residual_logit]
  common_upstream: [field_embedding]
  common_downstream: [additive_fusion]
  example_genome_fragment:
    skill: field_residual
    params:
      input_key: field_embeddings
      output_key: field_residual_logit
retrieval:
  aliases: [field_residual, open_ended_evolution]
  task_types: [ctr]
  architecture_roles: [adapter]
  output_semantics: [logit]
promotion_status: promoted
candidate_metrics:
  validation_best_auc: 0.8
created_from_open_ended_evolution: true
""",
        encoding="utf-8",
    )
    parent, _ = _deepfm_parent()
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)
    planner = SkillGraphSearchPlanner(
        skill_library=library,
        config=SkillGraphPlannerConfig(
            card_top_k=4,
            per_category_top_k=4,
            beam_width=4,
            max_depth=1,
            candidate_pool_size=4,
            validate_compile=True,
        ),
        seed=13,
    )

    cards = planner.select_relevant_cards(parent, failure_modes=["high_order_interaction_underfitting"])
    candidates = planner.plan(parent, failure_modes=["high_order_interaction_underfitting"], budget=2, round_idx=2)

    assert "field_residual" in {card.skill_id for card in cards}
    reused_nodes = [
        node
        for candidate in candidates
        for node in candidate.genome.nodes
        if node.skill_id == "field_residual" and node.source == "generated_skill"
    ]
    assert reused_nodes
    assert all(node.metadata.get("skill_graph_planner") for node in reused_nodes)
