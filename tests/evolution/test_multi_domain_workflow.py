import pandas as pd
import torch

from recskill.evolution import (
    AdaptiveCandidateBudgetConfig,
    CodeSpaceConfig,
    CTREvolutionConfig,
    CTRTrainingConfig,
    GenomeVerifier,
    MultiDomainDatasetBundle,
    MultiDomainDatasetConfig,
    MultiDomainGenomeConfig,
    MultiDomainModelEvolutionRunner,
    MultiDomainWorkflowConfig,
    SkillGenomeCompiler,
    SkillLibrary,
    build_multi_domain_genome,
    generate_multi_domain_candidates,
    prepare_multi_domain_data,
)
from recskill.evolution.multi_domain_code_space import (
    build_multi_domain_design_brief,
    build_multi_domain_sketch_prompt,
    build_multi_domain_synthesis_prompt,
    multi_domain_architecture_profile,
    multi_domain_proposal_schema,
)
from recskill.evolution.multi_domain_workflow import MULTI_DOMAIN_TEMPLATES
from recskill.evolution.open_ended import PROPOSAL_TYPES


def _write_multi_domain_csv(path, rows: int = 24) -> None:
    payload = []
    for idx in range(rows):
        domain = idx % 3
        payload.append(
            {
                "user": f"u{idx % 5}",
                "item": f"i{idx % 7}",
                "cate": f"c{idx % 4}",
                "domain_indicator": domain,
                "label": float((idx + domain) % 2),
            }
        )
    pd.DataFrame(payload).to_csv(path, index=False)


def _bundle() -> MultiDomainDatasetBundle:
    return MultiDomainDatasetBundle(
        feature_names=["user", "item", "cate"],
        sparse_feature_names=["user", "item", "cate"],
        dense_feature_names=[],
        vocab_sizes=[5, 7, 4],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
        domain_num=3,
        domain_names=["0", "1", "2"],
    )


def _genome_config(template: str = "star") -> MultiDomainGenomeConfig:
    return MultiDomainGenomeConfig(
        template=template,
        embedding_dim=4,
        shared_hidden_dims=[16, 8],
        tower_hidden_dims=[8],
        expert_dim=8,
        num_experts=3,
        ple_shared_experts=1,
        ple_specific_experts=1,
        star_fcn_dims=[16, 8],
        star_aux_dims=[8],
        sarnet_shared_experts=2,
        sarnet_specific_experts=1,
        sarnet_expert_dim=8,
        adaptdhm_fcn_dims=[16, 8],
        hamur_small_fcn_dims=[16, 8],
        hamur_large_fcn_dims=[32, 16, 8],
        hamur_hyper_dims=[8],
        hamur_k=4,
        hamur_adapter_dim=4,
        m3oe_fcn_dims=[16, 8, 8, 4],
        m3oe_expert_num=2,
        ppnet_fcn_dims=[16, 8],
        domain_embedding_dim=4,
        m2m_num_experts=2,
        m2m_expert_output_size=8,
        m2m_transformer_dims={"num_encoder_layers": 1, "num_decoder_layers": 1, "dim_feedforward": 8},
        adasparse_hidden_dims=[16],
        epnet_fcn_dims=[16, 1],
        dropout=0.0,
    )


def test_prepare_multi_domain_data_excludes_domain_from_sparse_features_by_default(tmp_path):
    data_path = tmp_path / "multi_domain.csv"
    _write_multi_domain_csv(data_path)

    bundle = prepare_multi_domain_data(
        MultiDomainDatasetConfig(path=str(data_path), split_ratio=[0.5, 0.25, 0.25]),
        CTRTrainingConfig(batch_size=4, num_workers=0),
    )
    batch_x, batch_y = next(iter(bundle.train_loader))

    assert bundle.domain_num == 3
    assert bundle.sparse_feature_names == ["user", "item", "cate"]
    assert "domain_indicator" not in bundle.sparse_feature_names
    assert set(batch_x) == {"sparse_features", "domain_indicator"}
    assert batch_x["sparse_features"].shape[1] == 3
    assert batch_x["domain_indicator"].ndim == 1
    assert batch_y.ndim == 1


def test_prepare_movielens_multi_domain_derives_domain_label_and_cate(tmp_path):
    data_path = tmp_path / "ml-1m.csv"
    rows = []
    ages = [1, 18, 25, 35, 45, 50, 56, 25, 18, 35, 1, 56]
    ratings = [5, 3, 4, 2, 5, 1, 4, 3, 5, 2, 4, 1]
    for idx, (age, rating) in enumerate(zip(ages, ratings)):
        rows.append(
            {
                "user_id": idx % 4,
                "movie_id": idx % 5,
                "rating": rating,
                "timestamp": 1000 + idx,
                "title": f"movie-{idx}",
                "genres": "Drama|Action" if idx % 2 == 0 else "Comedy",
                "gender": "F" if idx % 2 == 0 else "M",
                "age": age,
                "occupation": idx % 3,
                "zip": f"000{idx % 4}",
            }
        )
    pd.DataFrame(rows).to_csv(data_path, index=False)

    bundle = prepare_multi_domain_data(
        MultiDomainDatasetConfig(
            path=str(data_path),
            label_col="rating",
            positive_label_threshold=3.0,
            domain_col="domain_indicator",
            domain_source_col="age",
            domain_groups=[[1, 18], [25], [35, 45, 50, 56]],
            genre_col="genres",
            derived_cate_col="cate_id",
            dense_cols=["age"],
            normalize_dense=True,
            exclude_cols=["title", "timestamp"],
            split_ratio=[0.5, 0.25, 0.25],
        ),
        CTRTrainingConfig(batch_size=4, num_workers=0, seed=2022),
    )
    batch_x, batch_y = next(iter(bundle.train_loader))

    assert bundle.domain_num == 3
    assert bundle.metadata["domain_source"] == "derived"
    assert bundle.metadata["positive_label_threshold"] == 3.0
    assert "cate_id" in bundle.sparse_feature_names
    assert "genres" not in bundle.sparse_feature_names
    assert "title" not in bundle.sparse_feature_names
    assert "timestamp" not in bundle.sparse_feature_names
    assert bundle.dense_feature_names == ["age"]
    assert bundle.metadata["normalize_dense"] is True
    assert set(batch_x) == {"sparse_features", "domain_indicator", "dense_values", "dense_features"}
    assert float(batch_x["dense_values"].min()) >= 0.0
    assert float(batch_x["dense_values"].max()) <= 1.0
    assert set(batch_y.tolist()) <= {0.0, 1.0}


def test_multi_domain_templates_compile_and_forward():
    batch = {
        "sparse_features": torch.tensor(
            [
                [0, 0, 0],
                [1, 2, 3],
                [2, 3, 1],
                [4, 6, 2],
            ],
            dtype=torch.long,
        ),
        "domain_indicator": torch.tensor([0, 1, 2, 1], dtype=torch.long),
        "labels": torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }

    for template in MULTI_DOMAIN_TEMPLATES:
        genome = build_multi_domain_genome(_bundle(), _genome_config(template))
        GenomeVerifier().assert_valid(genome)
        model = SkillGenomeCompiler().compile(genome)
        model.eval()
        ctx = model(batch)
        assert ctx["prediction"].shape == (4,)
        assert ctx["loss"].ndim == 0


def test_multi_domain_code_space_profile_exposes_domain_lanes():
    genome = build_multi_domain_genome(_bundle(), _genome_config("mmoe"))

    profile = multi_domain_architecture_profile(genome)

    assert profile["task_family"] == "multi_domain_recommendation"
    assert profile["runtime_dims"]["domain_num"] == 3
    tensor_keys = {point["tensor_key"] for point in profile["injection_points"]}
    assert {"field_embeddings", "flat_embeddings", "domain_representations", "domain_outputs"} <= tensor_keys


def test_multi_domain_code_space_prompt_is_open_ended_but_executable():
    genome = build_multi_domain_genome(_bundle(), _genome_config("mmoe"))

    schema = multi_domain_proposal_schema()
    schema_types = set(schema["proposal_type"].split("|"))
    assert schema_types <= PROPOSAL_TYPES

    brief = build_multi_domain_design_brief(
        parent=genome,
        diagnosis_report={"failure_modes": ["domain_specific_tower_underfitting"]},
        evolution_memory=None,
        budget=2,
        round_idx=0,
    )
    constraints = brief["design_constraints"]
    assert constraints["supported_wiring_modes"] == ["insert_between", "replace_node"]
    assert "protected_nodes" not in constraints
    assert "domain_adapter" in brief["open_search_lanes"]
    assert "domain_selection" in brief["open_search_lanes"]
    assert brief["integration_examples"] == brief["injection_points"]

    sketch_prompt = build_multi_domain_sketch_prompt(design_brief=brief, sketch_budget=2)
    synthesis_prompt = build_multi_domain_synthesis_prompt(
        design_brief=brief,
        sketches=[],
        proposal_budget=2,
    )
    prompt_text = " ".join(
        [
            sketch_prompt["instruction"],
            synthesis_prompt["instruction"],
            constraints["search_space_policy"],
            constraints["boundary_contract_rule"],
        ]
    )
    assert "not the whole search space" in prompt_text
    assert "preserve tensor shapes" not in prompt_text
    assert "never target field_embedding" not in prompt_text


def test_generate_multi_domain_candidates_returns_valid_template_swaps():
    parent = build_multi_domain_genome(_bundle(), _genome_config("star"))
    evolution = CTREvolutionConfig(
        candidate_budget=3,
        templates=[
            {"name": "swap_shared_bottom", "template": "shared_bottom", "operation": "replace"},
            {"name": "swap_mmoe", "template": "mmoe", "operation": "replace"},
            {"name": "swap_ple", "template": "ple", "operation": "replace"},
        ],
    )

    candidates = generate_multi_domain_candidates([parent], _bundle(), _genome_config("star"), evolution, round_idx=0)

    assert [candidate.candidate_id for candidate in candidates] == [
        "round0_swap_mmoe",
        "round0_swap_ple",
        "round0_swap_shared_bottom",
    ]
    for candidate in candidates:
        assert "multi-domain skill selection" in candidate.rationale
        GenomeVerifier().assert_valid(candidate.genome)
        SkillGenomeCompiler().compile(candidate.genome)


def test_generate_multi_domain_candidates_uses_scored_skill_selection_and_family_mix():
    parent = build_multi_domain_genome(_bundle(), _genome_config("star"))
    evolution = CTREvolutionConfig(
        candidate_budget=6,
        templates=[
            {"name": "swap_shared_bottom", "template": "shared_bottom", "operation": "replace"},
            {"name": "swap_mmoe", "template": "mmoe", "operation": "replace"},
            {"name": "swap_ple", "template": "ple", "operation": "replace"},
            {"name": "swap_sarnet", "template": "sarnet", "operation": "replace"},
            {"name": "swap_adaptdhm", "template": "adaptdhm", "operation": "replace"},
        ],
    )

    round0 = generate_multi_domain_candidates([parent], _bundle(), _genome_config("star"), evolution, round_idx=0)
    round1 = generate_multi_domain_candidates([parent], _bundle(), _genome_config("star"), evolution, round_idx=1)

    round0_templates = [candidate.genome.metadata.extras["template"] for candidate in round0]
    round1_ids = [candidate.candidate_id for candidate in round1]

    assert round0[0].candidate_id == "round0_swap_mmoe"
    assert round0[1].candidate_id == "round0_swap_ple"
    assert len(round0_templates) == len(set(round0_templates))
    assert all("multi-domain skill selection" in candidate.rationale for candidate in round0)
    assert round1_ids != [candidate.candidate_id.replace("round0", "round1") for candidate in round0]


def test_generate_multi_domain_candidates_reuses_promoted_generated_skill(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    (root / "md_flat_adapter.py").write_text(
        """
import torch
from torch import nn


class MultiDomainFlatAdapter(nn.Module):
    def __init__(self, input_key="flat_embeddings", output_key="adapted_flat_embeddings", input_dim=1):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.zeros(int(input_dim)))

    def forward(self, inputs):
        x = inputs[self.input_key]
        return {self.output_key: x + self.scale.view(1, -1)}
""",
        encoding="utf-8",
    )
    (root / "md_flat_adapter.skill.yaml").write_text(
        """
skill_id: md_flat_adapter
name: md_flat_adapter
skill_name: md_flat_adapter
category: generated
implementation_path: md_flat_adapter.py
class_name: MultiDomainFlatAdapter
task_types: [multi_domain]
input_signature:
  - name: flat_embeddings
    shape: [batch_size, flat_input_dim]
    dtype: float32
output_signature:
  - name: adapted_flat_embeddings
    shape: [batch_size, flat_input_dim]
    dtype: float32
portability:
  scope: schema_agnostic
  reuse_enabled: true
composition:
  requires: [flat_embeddings]
  produces: [adapted_flat_embeddings]
  common_upstream: [flatten]
  example_genome_fragment:
    skill: md_flat_adapter
    params:
      input_key: flat_embeddings
      output_key: adapted_flat_embeddings
      input_dim: ${flat_input_dim}
retrieval:
  aliases: [md_flat_adapter, open_ended_evolution]
  task_types: [multi_domain]
promotion_status: promoted
candidate_metrics:
  validation_best_auc: 0.9
created_from_open_ended_evolution: true
""",
        encoding="utf-8",
    )
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)
    parent = build_multi_domain_genome(_bundle(), _genome_config("star"))
    evolution = CTREvolutionConfig(
        candidate_budget=2,
        code_space=CodeSpaceConfig(reuse_promoted_generated_skills=True),
    )

    candidates = generate_multi_domain_candidates(
        [parent],
        _bundle(),
        _genome_config("star"),
        evolution,
        round_idx=0,
        skill_library=library,
    )

    generated = [candidate for candidate in candidates if candidate.generated_skill_id == "md_flat_adapter"]
    assert len(generated) == 1
    GenomeVerifier(skill_library=library).assert_valid(generated[0].genome)
    SkillGenomeCompiler(skill_library=library).compile(generated[0].genome)


def test_multi_domain_round_records_skill_and_code_targets(tmp_path, monkeypatch):
    class EmptyProvider:
        def propose(self, **kwargs):
            return []

    monkeypatch.setattr(
        "recskill.evolution.multi_domain_workflow.build_multi_domain_code_space_provider",
        lambda config: EmptyProvider(),
    )
    runner = MultiDomainModelEvolutionRunner(
        MultiDomainWorkflowConfig(
            output_dir=str(tmp_path / "outputs"),
            evolution=CTREvolutionConfig(
                candidate_budget=8,
                code_space_probability=0.25,
                code_space=CodeSpaceConfig(
                    provider="creative_command",
                    planner_command="python tools/code_space_llm_provider.py --mode planner --task multi_domain",
                    synthesizer_command="python tools/code_space_llm_provider.py --mode synthesizer --task multi_domain",
                    allow_fallback_templates=False,
                ),
                adaptive_candidate_budget=AdaptiveCandidateBudgetConfig(enabled=True),
            ),
            write_metadata_files=False,
        )
    )
    runner.bundle = _bundle()
    parent = build_multi_domain_genome(_bundle(), _genome_config("star"))

    candidates = runner._generate_round_candidates([parent], round_idx=0)

    assert len(candidates) == 8
    assert sum(candidate.evolution_space == "code_space" for candidate in candidates) == 0
    assert sum(candidate.evolution_space == "skill_space" for candidate in candidates) == 8
    assert runner.adaptive_budget_state["records"][0]["skill_target"] == 6
    assert runner.adaptive_budget_state["records"][0]["code_target"] == 2


def test_multi_domain_runner_smoke(tmp_path):
    data_path = tmp_path / "multi_domain.csv"
    _write_multi_domain_csv(data_path, rows=36)
    config = MultiDomainWorkflowConfig(
        output_dir=str(tmp_path / "outputs"),
        dataset=MultiDomainDatasetConfig(path=str(data_path), split_ratio=[0.6, 0.2, 0.2]),
        genome=_genome_config("shared_bottom"),
        training=CTRTrainingConfig(
            epoch=1,
            batch_size=8,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            earlystop_patience=2,
        ),
        evolution=CTREvolutionConfig(
            enabled=True,
            rounds=1,
            candidate_budget=1,
            code_space_probability=0.0,
            templates=[{"name": "swap_mmoe", "template": "mmoe", "operation": "replace"}],
        ),
        write_metadata_files=True,
        write_summary_json=True,
        write_evolution_memory=False,
    )

    summary = MultiDomainModelEvolutionRunner(config).run()

    assert summary["domain_names"] == ["0", "1", "2"]
    assert len(summary["results"]) >= 2
    assert (tmp_path / "outputs" / "baseline_genome.json").exists()
    assert (tmp_path / "outputs" / "summary.json").exists()
