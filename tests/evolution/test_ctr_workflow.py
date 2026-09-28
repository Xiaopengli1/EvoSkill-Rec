import json
from pathlib import Path

import pandas as pd

from recskill.evolution import (
    AdaptiveCandidateBudgetConfig,
    CTRDatasetConfig,
    CTRDatasetBundle,
    CTREvolutionConfig,
    CTRGenomeConfig,
    CTRGenomeTrainer,
    CTRModelEvolutionRunner,
    CTRTrainingConfig,
    CTRWorkflowConfig,
    GenomeVerifier,
    SkillGenomeCompiler,
    build_default_ctr_genome,
    load_launcher_config,
    prepare_movielens_ctr_data,
)
from recskill.evolution.ctr_workflow import (
    CandidateResult,
    _adaptive_candidate_targets,
    _best_objective_value,
    _skill_library_for_candidate,
    _rewrite_generated_skill_reference,
    _parse_gpu_ids,
    _allocate_code_budget_by_parent,
    _generate_candidate_from_library_skill,
    _next_parent_population,
    _select_ctr_library_skill_ids,
    _select_survivors,
    _write_tsv,
    generate_ctr_candidate_population,
    generate_ctr_candidates,
)
from recskill.evolution.code_space import CodeSpaceConfig, OpenEndedCandidateBuildResult
from recskill.evolution.fingerprint import genome_architecture_fingerprint
from recskill.evolution.genome import SkillEdge, SkillNode
from recskill.evolution.open_ended import OpenEndedIngestionResult, OpenEndedProposal
from recskill.evolution.skill_library import SkillLibrary
from recskill.evolution.task_launcher import _build_aggregate_results_csv_rows


def test_movielens_ctr_workflow_smoke(tmp_path):
    data_path = tmp_path / "ml.csv"
    rows = []
    ratings = [5, 3, 4, 2, 5, 1, 4, 3, 5, 2, 4, 1]
    for idx, rating in enumerate(ratings):
        rows.append(
            {
                "user_id": idx % 4,
                "movie_id": idx % 6,
                "rating": rating,
                "timestamp": idx,
                "genres": "Drama|Action" if idx % 2 == 0 else "Comedy",
                "gender": "F" if idx % 2 == 0 else "M",
                "age": idx % 5,
                "occupation": idx % 3,
                "zip": f"000{idx % 4}",
            }
        )
    pd.DataFrame(rows).to_csv(data_path, index=False)

    dataset_cfg = CTRDatasetConfig(path=str(data_path), split_ratio=[0.5, 0.25, 0.25])
    training_cfg = CTRTrainingConfig(epoch=1, batch_size=4, device="cpu", max_train_batches=2, max_eval_batches=2)
    bundle = prepare_movielens_ctr_data(dataset_cfg, training_cfg)
    genome = build_default_ctr_genome(bundle, CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0))

    GenomeVerifier().assert_valid(genome)
    metrics = CTRGenomeTrainer(training_cfg).fit_and_evaluate(genome, bundle, tmp_path / "artifact")

    assert metrics["num_examples"] == 3
    assert (tmp_path / "artifact" / "metrics.json").exists()
    assert (tmp_path / "artifact" / "training_log.tsv").exists()
    assert not (tmp_path / "artifact" / "model.pth").exists()


def test_crossnet_mix_template_generates_valid_candidate():
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    workflow_cfg = CTRWorkflowConfig(
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            strategy="templates",
            candidate_budget=1,
            templates=[{"name": "add_crossnet_mix", "num_layers": 2, "low_rank": 4, "num_experts": 2}],
        ),
    )

    candidate_id, candidate, mutation_type, _ = generate_ctr_candidates(parent, workflow_cfg)[0]

    assert candidate_id == "round0_crossnet_mix"
    assert mutation_type == "add_crossnet_mix"
    cross_node = candidate.get_node("crossnet_mix_r0")
    assert cross_node.params["low_rank"] == 4
    assert cross_node.params["num_experts"] == 2
    GenomeVerifier().assert_valid(candidate)


def test_crossnet_mix_without_fm_template_generates_valid_candidate():
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    workflow_cfg = CTRWorkflowConfig(
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            strategy="templates",
            candidate_budget=1,
            templates=[{"name": "add_crossnet_mix_without_fm", "num_layers": 3, "low_rank": 8, "num_experts": 4}],
        ),
    )

    candidate_id, candidate, mutation_type, _ = generate_ctr_candidates(parent, workflow_cfg)[0]

    assert candidate_id == "round0_crossnet_mix_without_fm"
    assert mutation_type == "add_crossnet_mix_without_fm"
    assert "fm" not in candidate.node_ids()
    assert "fm_output" not in candidate.get_node("fusion").input_keys
    GenomeVerifier().assert_valid(candidate)


def test_autoint_attention_template_generates_valid_candidate():
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    workflow_cfg = CTRWorkflowConfig(
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            strategy="templates",
            candidate_budget=1,
            templates=[{"name": "add_autoint_attention", "num_layers": 2, "num_heads": 2}],
        ),
    )

    candidate_id, candidate, mutation_type, _ = generate_ctr_candidates(parent, workflow_cfg)[0]

    assert candidate_id == "round0_autoint_attention"
    assert mutation_type == "add_autoint_attention"
    autoint = candidate.get_node("autoint_r0")
    assert autoint.params["num_heads"] == 2
    assert candidate.get_node("autoint_r0_flatten").params["input_key"] == "autoint_r0_embeddings"
    assert "autoint_r0_logit" in candidate.get_node("fusion").input_keys
    GenomeVerifier().assert_valid(candidate)
    SkillGenomeCompiler().compile(candidate)


def test_default_evolution_generates_skill_candidates_without_code_space_templates():
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id", "gender"],
        vocab_sizes=[4, 8, 3],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    workflow_cfg = CTRWorkflowConfig(
        genome=genome_cfg,
        evolution=CTREvolutionConfig(),
    )

    candidates = generate_ctr_candidate_population([parent], workflow_cfg)

    assert len(candidates) == 10
    assert all(candidate.evolution_space == "skill_space" for candidate in candidates)
    assert sum(candidate.evolution_space == "code_space" for candidate in candidates) == 0
    assert {candidate.operation for candidate in candidates} >= {"add", "replace", "hybridize", "specialize"}
    assert "round0_bilinear_interaction" in {candidate.candidate_id for candidate in candidates}
    for candidate in candidates:
        GenomeVerifier().assert_valid(candidate.genome)
        SkillGenomeCompiler().compile(candidate.genome)


def test_promoted_generated_skill_card_reused_as_skill_space_candidate(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    code_path = root / "field_mean_residual.py"
    code_path.write_text(
        """
import torch
from torch import nn


class ResidualScale(nn.Module):
    def __init__(self, input_key="field_embeddings", output_key="field_mean_residual_logit", scale=1.0):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, inputs):
        x = inputs[self.input_key]
        return {self.output_key: x.mean(dim=(1, 2), keepdim=False).unsqueeze(-1) * self.scale}
""",
        encoding="utf-8",
    )
    card_path = root / "field_mean_residual.skill.yaml"
    card_path.write_text(
        """
skill_id: field_mean_residual
name: field_mean_residual
skill_name: field_mean_residual
category: adapter
description: Reusable generated residual branch.
implementation_path: generated_skills/field_mean_residual.py
class_name: ResidualScale
task_types: [ctr]
input_signature:
  - name: field_embeddings
    shape: [batch_size, num_fields, embedding_dim]
    dtype: float32
output_signature:
  - name: field_mean_residual_logit
    shape: [batch_size, 1]
    dtype: float32
composition:
  requires: [field_embeddings]
  produces: [field_mean_residual_logit]
  common_upstream: [field_embedding]
  common_downstream: [additive_fusion]
  example_genome_fragment:
    skill: field_mean_residual
    params:
      input_key: field_embeddings
      output_key: field_mean_residual_logit
      scale: 0.5
retrieval:
  aliases: [field_mean_residual, open_ended_evolution]
  task_types: [ctr]
  architecture_roles: [adapter]
promotion_status: promoted
candidate_metrics:
  validation_best_auc: 0.8
created_from_open_ended_evolution: true
""",
        encoding="utf-8",
    )
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id", "gender"],
        vocab_sizes=[4, 8, 3],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    config = CTRWorkflowConfig(genome=genome_cfg)
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)

    spec = _generate_candidate_from_library_skill(parent, config, "field_mean_residual", round_idx=3, library=library)

    assert spec.evolution_space == "skill_space"
    assert spec.operation == "reuse_generated"
    assert spec.generated_skill_id == "field_mean_residual"
    node = next(node for node in spec.genome.nodes if node.skill_id == "field_mean_residual")
    assert node.source == "generated_skill"
    assert node.metadata["evolution_space"] == "skill_space"
    assert node.metadata["reused_generated_skill"] is True
    assert node.input_keys == ["field_embeddings"]
    assert node.output_keys == ["field_mean_residual_r3_logit"]
    assert "field_mean_residual_r3_logit" in spec.genome.get_node("fusion").input_keys
    GenomeVerifier(skill_library=library).assert_valid(spec.genome)
    SkillGenomeCompiler(skill_library=library).compile(spec.genome)


def test_promoted_generated_skill_filters_runtime_keys_from_constructor(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    code_path = root / "field_norm_residual.py"
    code_path.write_text(
        """
import torch
from torch import nn


class FieldNormResidual(nn.Module):
    def __init__(self, output_key="field_norm_residual_logit", scale=1.0):
        super().__init__()
        self.output_key = output_key
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, inputs):
        x = inputs["field_embeddings"]
        value = x.norm(dim=-1).mean(dim=1, keepdim=True)
        return {self.output_key: value * self.scale}
""",
        encoding="utf-8",
    )
    card_path = root / "field_norm_residual.skill.yaml"
    card_path.write_text(
        """
skill_id: field_norm_residual
name: field_norm_residual
skill_name: field_norm_residual
category: adapter
description: Reusable generated branch whose constructor does not accept input_key.
implementation_path: generated_skills/field_norm_residual.py
class_name: FieldNormResidual
task_types: [ctr]
input_signature:
  - name: field_embeddings
    shape: [batch_size, num_fields, embedding_dim]
    dtype: float32
output_signature:
  - name: field_norm_residual_logit
    shape: [batch_size, 1]
    dtype: float32
composition:
  requires: [field_embeddings]
  produces: [field_norm_residual_logit]
  common_upstream: [field_embedding]
  common_downstream: [additive_fusion]
  example_genome_fragment:
    skill: field_norm_residual
    params:
      input_key: field_embeddings
      output_key: field_norm_residual_logit
      scale: 0.5
retrieval:
  aliases: [field_norm_residual, open_ended_evolution]
  task_types: [ctr]
  architecture_roles: [adapter]
promotion_status: promoted
candidate_metrics:
  validation_best_auc: 0.8
created_from_open_ended_evolution: true
""",
        encoding="utf-8",
    )
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id", "gender"],
        vocab_sizes=[4, 8, 3],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    config = CTRWorkflowConfig(genome=genome_cfg)
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)

    spec = _generate_candidate_from_library_skill(parent, config, "field_norm_residual", round_idx=3, library=library)

    node = next(node for node in spec.genome.nodes if node.skill_id == "field_norm_residual")
    assert node.params["input_key"] == "field_embeddings"
    assert node.output_keys == ["field_norm_residual_r3_logit"]
    GenomeVerifier(skill_library=library).assert_valid(spec.genome)
    SkillGenomeCompiler(skill_library=library).compile(spec.genome)


def test_promoted_generated_skill_with_fixed_shape_is_not_selected_for_mismatched_schema(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    (root / "fixed_shape.py").write_text(
        """
import torch
from torch import nn


class FixedShapeBranch(nn.Module):
    def __init__(self, flat_key="flat_embeddings", output_key="fixed_shape_logit", num_fields=7, input_dim=112, embedding_dim=16):
        super().__init__()
        self.flat_key = flat_key
        self.output_key = output_key
        self.num_fields = num_fields
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.head = nn.Linear(input_dim, 1)

    def forward(self, inputs):
        return {self.output_key: self.head(inputs[self.flat_key])}
""",
        encoding="utf-8",
    )
    (root / "fixed_shape.skill.yaml").write_text(
        """
skill_id: fixed_shape_branch
name: fixed_shape_branch
skill_name: fixed_shape_branch
category: adapter
implementation_path: generated_skills/fixed_shape.py
class_name: FixedShapeBranch
task_types: [ctr]
input_signature:
  - name: field_embeddings
    shape: [batch_size, 7, 16]
    dtype: float32
  - name: flat_embeddings
    shape: [batch_size, 112]
    dtype: float32
output_signature:
  - name: fixed_shape_logit
    shape: [batch_size, 1]
    dtype: float32
composition:
  requires: [field_embeddings, flat_embeddings]
  produces: [fixed_shape_logit]
  common_upstream: [field_embedding, flatten]
  example_genome_fragment:
    skill: fixed_shape_branch
    params:
      flat_key: flat_embeddings
      output_key: fixed_shape_logit
      num_fields: 7
      embedding_dim: 16
      input_dim: 112
retrieval:
  aliases: [fixed_shape_branch, open_ended_evolution]
  task_types: [ctr]
promotion_status: promoted
candidate_metrics:
  validation_best_auc: 0.9
created_from_open_ended_evolution: true
""",
        encoding="utf-8",
    )
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "item_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=16, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    config = CTRWorkflowConfig(
        genome=genome_cfg,
        evolution=CTREvolutionConfig(code_space=CodeSpaceConfig(reuse_promoted_generated_skills=True)),
    )
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)

    selected = _select_ctr_library_skill_ids(library, [parent], config, 4, code_space=False, round_idx=0)

    assert "fixed_shape_branch" not in selected


def test_reused_promoted_generated_skill_resolves_shape_placeholders(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    (root / "portable_flat.py").write_text(
        """
import torch
from torch import nn


class PortableFlat(nn.Module):
    def __init__(self, flat_key="flat_embeddings", output_key="portable_flat_logit", input_dim=1, num_fields=1, embedding_dim=1):
        super().__init__()
        self.flat_key = flat_key
        self.output_key = output_key
        self.input_dim = int(input_dim)
        self.num_fields = int(num_fields)
        self.embedding_dim = int(embedding_dim)
        self.head = nn.Linear(self.input_dim, 1)

    def forward(self, inputs):
        return {self.output_key: self.head(inputs[self.flat_key])}
""",
        encoding="utf-8",
    )
    (root / "portable_flat.skill.yaml").write_text(
        """
skill_id: portable_flat
name: portable_flat
skill_name: portable_flat
category: adapter
implementation_path: generated_skills/portable_flat.py
class_name: PortableFlat
task_types: [ctr]
input_signature:
  - name: flat_embeddings
    shape: [batch_size, flat_input_dim]
    dtype: float32
output_signature:
  - name: portable_flat_logit
    shape: [batch_size, 1]
    dtype: float32
portability:
  scope: schema_agnostic
  reuse_enabled: true
  constraints:
    min_num_fields: 1
composition:
  requires: [flat_embeddings]
  produces: [portable_flat_logit]
  common_upstream: [flatten]
  example_genome_fragment:
    skill: portable_flat
    params:
      flat_key: flat_embeddings
      output_key: portable_flat_logit
      num_fields: ${num_fields}
      embedding_dim: ${embedding_dim}
      input_dim: ${flat_input_dim}
retrieval:
  aliases: [portable_flat, open_ended_evolution]
  task_types: [ctr]
promotion_status: promoted
candidate_metrics:
  validation_best_auc: 0.9
created_from_open_ended_evolution: true
""",
        encoding="utf-8",
    )
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "item_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=16, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    config = CTRWorkflowConfig(
        genome=genome_cfg,
        evolution=CTREvolutionConfig(code_space=CodeSpaceConfig(reuse_promoted_generated_skills=True)),
    )
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)

    spec = _generate_candidate_from_library_skill(parent, config, "portable_flat", round_idx=2, library=library)

    node = next(node for node in spec.genome.nodes if node.skill_id == "portable_flat")
    assert node.params["num_fields"] == 2
    assert node.params["embedding_dim"] == 16
    assert node.params["input_dim"] == 32
    GenomeVerifier(skill_library=library).assert_valid(spec.genome)
    SkillGenomeCompiler(skill_library=library).compile(spec.genome)


def test_promoted_generated_skill_selection_rotates_in_skill_space_and_not_code_space(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    code_path = root / "logit_adapter.py"
    code_path.write_text(
        """
import torch
from torch import nn


class LogitAdapter(nn.Module):
    def __init__(self, input_key="logits", output_key="adapted_logits"):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        return {self.output_key: inputs[self.input_key] * self.scale}
""",
        encoding="utf-8",
    )
    for name, auc in [
        ("adaptive_logit_affine", 0.801),
        ("tanh_logit_residual", 0.800),
        ("softsign_logit_calibrator", 0.799),
    ]:
        (root / f"{name}.skill.yaml").write_text(
            f"""
skill_id: {name}
name: {name}
skill_name: {name}
category: adapter
implementation_path: {code_path}
class_name: LogitAdapter
task_types: [ctr]
input_signature:
  - name: logits
    shape: [batch_size, 1]
    dtype: float32
output_signature:
  - name: {name}_logits
    shape: [batch_size, 1]
    dtype: float32
composition:
  requires: [logits]
  produces: [{name}_logits]
  common_upstream: [fusion]
  example_genome_fragment:
    skill: {name}
    params:
      input_key: logits
      output_key: {name}_logits
retrieval:
  aliases: [{name}, open_ended_evolution]
  task_types: [ctr]
promotion_status: promoted
candidate_metrics:
  validation_best_auc: {auc}
created_from_open_ended_evolution: true
""",
            encoding="utf-8",
        )
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    config = CTRWorkflowConfig(
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            candidate_budget=3,
            code_space=CodeSpaceConfig(reuse_promoted_generated_skills=True),
        ),
    )
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)

    selected = [
        _select_ctr_library_skill_ids(library, [parent], config, 1, code_space=False, round_idx=round_idx)[0]
        for round_idx in range(4)
    ]

    assert set(selected[:3]) == {"adaptive_logit_affine", "tanh_logit_residual", "softsign_logit_calibrator"}
    assert selected[3] == selected[0]
    assert _select_ctr_library_skill_ids(library, [parent], config, 3, code_space=True, round_idx=0) == []
    selected_with_quota = _select_ctr_library_skill_ids(library, [parent], config, 3, code_space=False, round_idx=0)
    assert any(skill_id not in {"adaptive_logit_affine", "tanh_logit_residual", "softsign_logit_calibrator"} for skill_id in selected_with_quota)

    disabled_config = CTRWorkflowConfig(
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            candidate_budget=3,
            code_space=CodeSpaceConfig(reuse_promoted_generated_skills=False),
        ),
    )
    disabled_selected = _select_ctr_library_skill_ids(library, [parent], disabled_config, 3, code_space=False, round_idx=0)
    assert not (set(disabled_selected) & {"adaptive_logit_affine", "tanh_logit_residual", "softsign_logit_calibrator"})


def test_reused_promoted_logit_skill_retargets_current_logits_and_removes_stale_edges(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    (root / "affine.py").write_text(
        """
import torch
from torch import nn


class Affine(nn.Module):
    def __init__(self, input_key="logits", output_key="affine_logits"):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        return {self.output_key: inputs[self.input_key] * self.scale}
""",
        encoding="utf-8",
    )
    (root / "affine.skill.yaml").write_text(
        """
skill_id: affine
name: affine
skill_name: affine
category: adapter
implementation_path: generated_skills/affine.py
class_name: Affine
task_types: [ctr]
input_signature:
  - name: llm_r00_old_logits
    shape: [batch_size, 1]
    dtype: float32
output_signature:
  - name: llm_r00_affine_logits
    shape: [batch_size, 1]
    dtype: float32
composition:
  requires: [llm_r00_old_logits]
  produces: [llm_r00_affine_logits]
  common_upstream: [fusion]
  example_genome_fragment:
    skill: affine
    params:
      input_key: llm_r00_old_logits
      output_key: llm_r00_affine_logits
retrieval:
  aliases: [affine, open_ended_evolution]
  task_types: [ctr]
promotion_status: promoted
created_from_open_ended_evolution: true
""",
        encoding="utf-8",
    )
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    parent = build_default_ctr_genome(bundle, CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0))
    parent.nodes.append(
        SkillNode(
            node_id="prior_logit_adapter",
            skill_id="affine",
            skill_name="affine",
            category="adapter",
            params={"input_key": "logits", "output_key": "llm_r03_damped_logits"},
            input_keys=["logits"],
            output_keys=["llm_r03_damped_logits"],
            task_types=["ctr"],
            source="generated_skill",
        )
    )
    parent.get_node("prediction").input_keys = ["llm_r03_damped_logits"]
    parent.get_node("prediction").params["input_key"] = "llm_r03_damped_logits"
    parent.get_node("loss").input_keys = ["llm_r03_damped_logits", "labels"]
    parent.get_node("loss").params["logits_key"] = "llm_r03_damped_logits"
    parent.edges = [
        edge
        for edge in parent.edges
        if edge.dst_node_id not in {"prediction", "loss"}
    ]
    parent.edges.extend(
        [
            SkillEdge("fusion", "prior_logit_adapter", "logits", "logits"),
            SkillEdge("prior_logit_adapter", "prediction", "llm_r03_damped_logits", "llm_r03_damped_logits"),
            SkillEdge("prior_logit_adapter", "loss", "llm_r03_damped_logits", "llm_r03_damped_logits"),
        ]
    )
    config = CTRWorkflowConfig(genome=CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0))
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)

    spec = _generate_candidate_from_library_skill(parent, config, "affine", round_idx=4, library=library)

    node = next(node for node in spec.genome.nodes if node.metadata.get("reused_generated_skill"))
    assert node.input_keys == ["llm_r03_damped_logits"]
    assert node.params["input_key"] == "llm_r03_damped_logits"
    assert node.output_keys == ["affine_r4_logit"]
    assert node.params["output_key"] == "affine_r4_logit"
    assert spec.genome.get_node("prediction").input_keys == ["affine_r4_logit"]
    assert spec.genome.get_node("loss").input_keys == ["affine_r4_logit", "labels"]
    assert not any(
        edge.dst_node_id in {"prediction", "loss"} and edge.src_output_key == "llm_r03_damped_logits"
        for edge in spec.genome.edges
    )
    GenomeVerifier(skill_library=library).assert_valid(spec.genome)
    SkillGenomeCompiler(skill_library=library).compile(spec.genome)


def test_reused_generated_logit_skill_without_named_fusion_uses_active_fusion(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    (root / "field_mean.py").write_text(
        """
import torch
from torch import nn


class FieldMean(nn.Module):
    def __init__(self, input_key="field_embeddings", output_key="field_mean_logit"):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        x = inputs[self.input_key]
        return {self.output_key: x.mean(dim=(1, 2), keepdim=False).unsqueeze(-1) * self.scale}
""",
        encoding="utf-8",
    )
    (root / "field_mean.skill.yaml").write_text(
        """
skill_id: field_mean
name: field_mean
skill_name: field_mean
category: adapter
implementation_path: generated_skills/field_mean.py
class_name: FieldMean
task_types: [ctr]
input_signature:
  - name: field_embeddings
    shape: [batch_size, num_fields, embedding_dim]
    dtype: float32
output_signature:
  - name: field_mean_logit
    shape: [batch_size, 1]
    dtype: float32
composition:
  requires: [field_embeddings]
  produces: [field_mean_logit]
  common_upstream: [field_embedding]
  example_genome_fragment:
    skill: field_mean
    params:
      input_key: field_embeddings
      output_key: field_mean_logit
retrieval:
  aliases: [field_mean, open_ended_evolution]
  task_types: [ctr]
promotion_status: promoted
created_from_open_ended_evolution: true
""",
        encoding="utf-8",
    )
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    parent = build_default_ctr_genome(bundle, CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0))
    terminal = parent.get_node("fusion")
    terminal.node_id = "generated_macro_fusion"
    parent.edges = [
        SkillEdge(
            "generated_macro_fusion" if edge.src_node_id == "fusion" else edge.src_node_id,
            "generated_macro_fusion" if edge.dst_node_id == "fusion" else edge.dst_node_id,
            edge.src_output_key,
            edge.dst_input_key,
            edge.tensor_semantics,
        )
        for edge in parent.edges
    ]
    config = CTRWorkflowConfig(genome=CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0))
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)

    spec = _generate_candidate_from_library_skill(parent, config, "field_mean", round_idx=7, library=library)

    node = next(node for node in spec.genome.nodes if node.metadata.get("reused_generated_skill"))
    assert node.output_keys == ["field_mean_r7_logit"]
    active_fusion = spec.genome.get_node("generated_macro_fusion")
    assert "field_mean_r7_logit" in active_fusion.input_keys
    assert "field_mean_r7_logit" in active_fusion.params["input_keys"]
    assert spec.genome.get_node("prediction").input_keys == ["logits"]
    assert spec.genome.get_node("loss").input_keys == ["logits", "labels"]
    assert any(edge.src_node_id == node.node_id and edge.dst_node_id == "generated_macro_fusion" for edge in spec.genome.edges)
    GenomeVerifier(skill_library=library).assert_valid(spec.genome)
    SkillGenomeCompiler(skill_library=library).compile(spec.genome)


def test_skill_library_skips_malformed_skill_cards(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    (root / "good.skill.yaml").write_text(
        """
skill_id: good_skill
name: good_skill
skill_name: good_skill
category: adapter
task_types: [ctr]
input_signature:
  - name: logits
output_signature:
  - name: good_logits
""",
        encoding="utf-8",
    )
    (root / "bad.skill.yaml").write_text("not: [valid\n", encoding="utf-8")
    (root / "missing_id.skill.yaml").write_text("category: adapter\n", encoding="utf-8")

    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)

    assert library.has("good_skill")
    assert len(library.load_errors) == 2
    assert {Path(item["path"]).name for item in library.load_errors} == {"bad.skill.yaml", "missing_id.skill.yaml"}


def test_skill_library_candidate_generation_skips_failed_candidate_builder(monkeypatch):
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    workflow_cfg = CTRWorkflowConfig(
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            strategy="skill_library",
            candidate_budget=2,
            code_space_probability=0.0,
        ),
    )
    original_builder = _generate_candidate_from_library_skill
    failed_skill_ids = []

    def flaky_builder(parent, config, skill_id, round_idx, library=None):
        if skill_id == "crossnet_mix":
            failed_skill_ids.append(skill_id)
            raise KeyError("simulated broken generated candidate")
        return original_builder(parent, config, skill_id, round_idx, library=library)

    monkeypatch.setattr("recskill.evolution.ctr_workflow._generate_candidate_from_library_skill", flaky_builder)

    specs = generate_ctr_candidate_population([parent], workflow_cfg, round_idx=0)

    assert len(specs) == 2
    assert failed_skill_ids == ["crossnet_mix"]
    assert all(spec.candidate_id for spec in specs)


def test_promoted_survivor_genome_carries_skill_card_for_next_round_worker(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    code_path = root / "safe_scale.py"
    code_path.write_text(
        """
import torch
from torch import nn


class SafeScale(nn.Module):
    def __init__(self, input_key="logits", output_key="scaled_logits"):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        return {self.output_key: inputs[self.input_key] * self.scale}
""",
        encoding="utf-8",
    )
    card_path = root / "safe_scale.skill.yaml"
    card_path.write_text(
        f"""
skill_id: safe_scale
name: safe_scale
skill_name: safe_scale
category: adapter
implementation_path: {code_path}
class_name: SafeScale
task_types: [ctr]
input_signature:
  - name: logits
    shape: [batch_size, 1]
    dtype: float32
output_signature:
  - name: scaled_logits
    shape: [batch_size, 1]
    dtype: float32
composition:
  requires: [logits]
  produces: [scaled_logits]
  common_upstream: [fusion]
  example_genome_fragment:
    skill: safe_scale
    params:
      input_key: logits
      output_key: scaled_logits
retrieval:
  aliases: [safe_scale, open_ended_evolution]
  task_types: [ctr]
promotion_status: promoted
created_from_open_ended_evolution: true
""",
        encoding="utf-8",
    )
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome = build_default_ctr_genome(bundle, CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0))
    genome.nodes.append(
        SkillNode(
            node_id="llm_r00_safe_scale",
            skill_id="llm_r00_safe_scale",
            skill_name="llm_r00_safe_scale",
            category="adapter",
            params={"input_key": "logits", "output_key": "scaled_logits"},
            input_keys=["logits"],
            output_keys=["scaled_logits"],
            task_types=["ctr"],
            source="generated_skill",
        )
    )
    genome.edges = [edge for edge in genome.edges if edge.dst_node_id not in {"prediction", "loss"}]
    genome.edges.extend(
        [
            SkillEdge("fusion", "llm_r00_safe_scale", "logits", "logits"),
            SkillEdge("llm_r00_safe_scale", "prediction", "scaled_logits", "scaled_logits"),
            SkillEdge("llm_r00_safe_scale", "loss", "scaled_logits", "scaled_logits"),
        ]
    )
    genome.get_node("prediction").input_keys = ["scaled_logits"]
    genome.get_node("prediction").params["input_key"] = "scaled_logits"
    genome.get_node("loss").input_keys = ["scaled_logits", "labels"]
    genome.get_node("loss").params["logits_key"] = "scaled_logits"

    _rewrite_generated_skill_reference(
        genome,
        "llm_r00_safe_scale",
        "safe_scale",
        persisted_code_path=str(code_path),
        persisted_skill_card_path=str(card_path),
    )
    node = genome.get_node("llm_r00_safe_scale")

    assert node.skill_id == "safe_scale"
    assert node.metadata["origin_skill_id"] == "llm_r00_safe_scale"
    assert node.metadata["persisted_skill_card_path"] == str(card_path)

    worker_library = _skill_library_for_candidate(None, genome=genome)
    GenomeVerifier(skill_library=worker_library).assert_valid(genome)
    SkillGenomeCompiler(skill_library=worker_library).compile(genome)


def test_no_survivor_round_reuses_previous_parent_population():
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    parent = build_default_ctr_genome(bundle, CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0))

    assert _next_parent_population([parent], []) == [parent]


def test_code_space_generation_distributes_budget_across_retained_parents(tmp_path, monkeypatch):
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parents = [
        build_default_ctr_genome(bundle, genome_cfg),
        build_default_ctr_genome(bundle, genome_cfg),
    ]
    parents[0].metadata.genome_id = "parent_a"
    parents[1].metadata.genome_id = "parent_b"
    config = CTRWorkflowConfig(
        output_dir=str(tmp_path),
        write_code_space_diagnostics=False,
        write_evolution_memory=False,
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            candidate_budget=2,
            code_space_probability=1.0,
            code_space=CodeSpaceConfig(
                provider="command",
                command="unused",
                structural_scopes=["macro"],
                reuse_promoted_code_skills=False,
                allow_fallback_templates=False,
            ),
        ),
    )
    runner = CTRModelEvolutionRunner(config)
    calls = []

    class FakeProvider:
        def propose(self, *, parent, diagnosis_report, evolution_memory, budget, round_idx):
            calls.append((parent.metadata.genome_id, diagnosis_report["parent_index"], budget))
            return [
                OpenEndedProposal(
                    proposal_id=f"proposal_{parent.metadata.genome_id}",
                    proposal_type="NEW_FUSION_DESIGN",
                    target_failure_mode="fusion_bottleneck",
                    architecture_hypothesis=f"Macro fusion for {parent.metadata.genome_id}.",
                    affected_genome_nodes=["fusion"],
                    code="",
                    expected_input_signature=[],
                    expected_output_signature=[],
                    skill_id=f"skill_{parent.metadata.genome_id}",
                    class_name="Generated",
                    structural_scope="macro",
                    metadata={"structural_scope": "macro", "wiring": "replace_fusion"},
                )
            ][:budget]

    def fake_build_provider(code_config):
        return FakeProvider()

    def fake_ingest(*, proposals, parent, output_dir, memory, repo_root=None, staging_root=None):
        results = []
        for proposal in proposals:
            child = parent.clone()
            skill_id = proposal.skill_id or proposal.proposal_id
            results.append(
                OpenEndedCandidateBuildResult(
                    proposal=proposal,
                    ingestion=OpenEndedIngestionResult(
                        success=True,
                        proposal_id=proposal.proposal_id,
                        message="ok",
                        skill_id=skill_id,
                        genome=child,
                    ),
                )
            )
        return results

    monkeypatch.setattr("recskill.evolution.ctr_workflow.build_code_space_provider", fake_build_provider)
    monkeypatch.setattr("recskill.evolution.ctr_workflow.inspect_open_ended_proposal_ingestions", fake_ingest)

    specs = runner._generate_open_ended_code_candidates(parents, round_idx=0, target_code=2)

    assert calls == [("parent_a", 0, 1), ("parent_b", 1, 1)]
    assert [spec.parent_genome_id for spec in specs] == ["parent_a", "parent_b"]
    assert [spec.structural_scope for spec in specs] == ["macro", "macro"]
    assert {spec.candidate_id for spec in specs} == {
        "round0_code_skill_parent_a_p0",
        "round0_code_skill_parent_b_p1",
    }


def test_code_space_generation_skips_parent_when_ingestion_crashes(tmp_path, monkeypatch):
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parents = [
        build_default_ctr_genome(bundle, genome_cfg),
        build_default_ctr_genome(bundle, genome_cfg),
    ]
    parents[0].metadata.genome_id = "parent_a"
    parents[1].metadata.genome_id = "parent_b"
    config = CTRWorkflowConfig(
        output_dir=str(tmp_path),
        write_code_space_diagnostics=True,
        write_evolution_memory=False,
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            candidate_budget=2,
            code_space_probability=1.0,
            code_space=CodeSpaceConfig(
                provider="command",
                command="unused",
                structural_scopes=["macro"],
                allow_fallback_templates=False,
            ),
        ),
    )
    runner = CTRModelEvolutionRunner(config)

    class FakeProvider:
        def propose(self, *, parent, diagnosis_report, evolution_memory, budget, round_idx):
            return [
                OpenEndedProposal(
                    proposal_id=f"proposal_{parent.metadata.genome_id}",
                    proposal_type="NEW_FUSION_DESIGN",
                    target_failure_mode="fusion_bottleneck",
                    architecture_hypothesis=f"Macro fusion for {parent.metadata.genome_id}.",
                    affected_genome_nodes=["fusion"],
                    code="",
                    expected_input_signature=[],
                    expected_output_signature=[],
                    skill_id=f"skill_{parent.metadata.genome_id}",
                    class_name="Generated",
                    structural_scope="macro",
                    metadata={"structural_scope": "macro", "wiring": "replace_fusion"},
                )
            ][:budget]

    def fake_build_provider(code_config):
        return FakeProvider()

    def fake_ingest(*, proposals, parent, output_dir, memory, repo_root=None, staging_root=None):
        if parent.metadata.genome_id == "parent_a":
            raise RuntimeError("simulated ingestion failure")
        proposal = proposals[0]
        return [
            OpenEndedCandidateBuildResult(
                proposal=proposal,
                ingestion=OpenEndedIngestionResult(
                    success=True,
                    proposal_id=proposal.proposal_id,
                    message="ok",
                    skill_id=proposal.skill_id,
                    genome=parent.clone(),
                ),
            )
        ]

    monkeypatch.setattr("recskill.evolution.ctr_workflow.build_code_space_provider", fake_build_provider)
    monkeypatch.setattr("recskill.evolution.ctr_workflow.inspect_open_ended_proposal_ingestions", fake_ingest)

    specs = runner._generate_open_ended_code_candidates(parents, round_idx=0, target_code=2)

    assert [spec.parent_genome_id for spec in specs] == ["parent_b"]
    diagnostics = json.loads((tmp_path / "round0_code_space_proposals.json").read_text(encoding="utf-8"))
    assert diagnostics["errors"][0]["stage"] == "ingest_open_ended_proposals"
    assert diagnostics["errors"][0]["parent_genome_id"] == "parent_a"


def test_code_space_generation_records_failed_ingestions_without_silent_zero(tmp_path, monkeypatch):
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    parent.metadata.genome_id = "parent_a"
    config = CTRWorkflowConfig(
        output_dir=str(tmp_path),
        write_code_space_diagnostics=False,
        write_evolution_memory=False,
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            candidate_budget=1,
            code_space_probability=1.0,
            code_space=CodeSpaceConfig(
                provider="command",
                command="unused",
                structural_scopes=["macro"],
                allow_fallback_templates=False,
                max_retries=1,
            ),
        ),
    )
    runner = CTRModelEvolutionRunner(config)

    class FakeProvider:
        def propose(self, *, parent, diagnosis_report, evolution_memory, budget, round_idx):
            return [
                OpenEndedProposal(
                    proposal_id="bad_macro",
                    proposal_type="NEW_FUSION_DESIGN",
                    target_failure_mode="fusion_bottleneck",
                    architecture_hypothesis="Invalid macro fusion.",
                    affected_genome_nodes=["fusion"],
                    code="",
                    expected_input_signature=[],
                    expected_output_signature=[],
                    skill_id="bad_macro_skill",
                    class_name="Generated",
                    structural_scope="macro",
                    metadata={"structural_scope": "macro", "wiring": "replace_fusion"},
                )
            ]

    def fake_build_provider(code_config):
        return FakeProvider()

    def fake_ingest(*, proposals, parent, output_dir, memory, repo_root=None, staging_root=None):
        proposal = proposals[0]
        return [
            OpenEndedCandidateBuildResult(
                proposal=proposal,
                ingestion=OpenEndedIngestionResult(
                    success=False,
                    proposal_id=proposal.proposal_id,
                    message="macro validation failed",
                    skill_id=proposal.skill_id,
                    validation_results={"macro_validation": {"passed": False, "errors": ["missing macro_judgment"]}},
                ),
            )
        ]

    monkeypatch.setattr("recskill.evolution.ctr_workflow.build_code_space_provider", fake_build_provider)
    monkeypatch.setattr("recskill.evolution.ctr_workflow.inspect_open_ended_proposal_ingestions", fake_ingest)

    specs = runner._generate_open_ended_code_candidates([parent], round_idx=0, target_code=1)

    assert specs == []
    diagnostics = json.loads((tmp_path / "round0_code_space_proposals.json").read_text(encoding="utf-8"))
    assert diagnostics["requested"] == 1
    assert diagnostics["generated"] == 0
    assert diagnostics["errors"][0]["proposal_id"] == "bad_macro"
    assert diagnostics["errors"][0]["error"] == "macro validation failed"
    assert diagnostics["errors"][0]["validation_results"]["macro_validation"]["errors"] == ["missing macro_judgment"]
    assert runner.candidate_generation_errors[0]["stage"] == "code_space_parent_generation"


def test_round_generation_supplements_skill_candidates_when_code_space_is_empty(tmp_path, monkeypatch):
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id", "gender"],
        vocab_sizes=[4, 8, 3],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    config = CTRWorkflowConfig(
        output_dir=str(tmp_path),
        write_code_space_diagnostics=False,
        write_evolution_memory=False,
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            strategy="templates",
            candidate_budget=4,
            code_space_probability=0.5,
            code_space=CodeSpaceConfig(
                provider="command",
                command="unused",
                structural_scopes=["macro"],
                allow_fallback_templates=False,
                max_retries=1,
            ),
            templates=[
                {"name": "add_crossnet_mix"},
                {"name": "add_afm_attention"},
                {"name": "add_crossnet_v2"},
                {"name": "replace_fm_with_field_sum"},
                {"name": "remove_fm"},
            ],
        ),
    )
    runner = CTRModelEvolutionRunner(config)

    class EmptyProvider:
        def propose(self, *, parent, diagnosis_report, evolution_memory, budget, round_idx):
            return []

    monkeypatch.setattr("recskill.evolution.ctr_workflow.build_code_space_provider", lambda code_config: EmptyProvider())

    specs = runner._generate_round_candidates([parent], round_idx=0)

    assert len(specs) == 4
    assert sum(spec.evolution_space == "code_space" for spec in specs) == 0
    assert sum(spec.evolution_space == "skill_space" for spec in specs) == 4
    assert any(record["stage"] == "insufficient_open_ended_code_candidates" for record in runner.candidate_generation_errors)
    diagnostics = json.loads((tmp_path / "round0_code_space_error.json").read_text(encoding="utf-8"))
    assert diagnostics["requested"] == 2
    assert diagnostics["generated"] == 0
    assert runner.code_space_generation_records[-1]["stage"] == "skill_space_supplement"


def test_allocate_code_budget_rotates_retained_parents():
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parents = [build_default_ctr_genome(bundle, genome_cfg) for _ in range(3)]
    for idx, parent in enumerate(parents):
        parent.metadata.genome_id = f"parent_{idx}"

    allocations = _allocate_code_budget_by_parent(parents, budget=2, round_idx=1)

    assert [(idx, parent.metadata.genome_id, budget) for idx, parent, budget in allocations] == [
        (1, "parent_1", 1),
        (2, "parent_2", 1),
    ]


def test_architecture_fingerprint_ignores_genome_metadata():
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome = build_default_ctr_genome(bundle, CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0))
    clone = genome.clone()

    assert genome_architecture_fingerprint(genome) == genome_architecture_fingerprint(clone)


def test_architecture_dedup_skips_repeated_remove_fm_candidate(tmp_path):
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome_cfg = CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0)
    parent = build_default_ctr_genome(bundle, genome_cfg)
    workflow_cfg = CTRWorkflowConfig(
        output_dir=str(tmp_path),
        genome=genome_cfg,
        evolution=CTREvolutionConfig(
            strategy="templates",
            candidate_budget=2,
            templates=[{"name": "remove_fm"}, {"name": "remove_fm"}],
        ),
    )
    runner = CTRModelEvolutionRunner(workflow_cfg)

    candidates = generate_ctr_candidate_population([parent], workflow_cfg)
    unique, skipped = runner._deduplicate_candidates(candidates)

    assert len(unique) == 1
    assert len(skipped) == 1
    assert skipped[0].status == "skipped_duplicate"
    assert skipped[0].duplicate_of == unique[0].candidate_id


def test_candidate_result_reports_generation_and_evolution_category():
    result = CandidateResult(
        candidate_id="round2_code_residual_gate_p1",
        genome_id="genome",
        parent_genome_id="parent",
        mutation_type="code_open_ended_residual_gate",
        status="survivor",
        metrics={"validation_best_auc": 0.75},
        artifact_dir="out",
        genome_path="genome.json",
        elapsed_sec=0.0,
        evolution_space="code_space",
        operation="open_ended",
    )

    payload = result.to_dict()

    assert payload["generation"] == "3rd"
    assert payload["evolution_category"] == "code_space:open_ended"
    assert "genome" not in payload


def test_result_report_rows_include_candidate_round_curve_and_fingerprint_summaries(tmp_path):
    config = CTRWorkflowConfig(
        output_dir=str(tmp_path),
        write_metadata_files=False,
        write_summary_json=False,
        write_evolution_memory=False,
        write_code_space_diagnostics=False,
    )
    runner = CTRModelEvolutionRunner(config)
    results = [
        CandidateResult(
            candidate_id="baseline",
            genome_id="genome_base",
            parent_genome_id=None,
            mutation_type="baseline",
            status="baseline",
            metrics={"validation_best_auc": 0.70, "auc": 0.69, "logloss": 0.5},
            artifact_dir="",
            genome_path="",
            elapsed_sec=0.0,
            architecture_fingerprint="arch_base",
            training_config_hash="train_hash",
            genome_config_hash="genome_hash",
            hparam_fingerprint="hparam_hash",
            rationale="baseline",
        ),
        CandidateResult(
            candidate_id="round0_skill_add",
            genome_id="genome_skill",
            parent_genome_id="genome_base",
            mutation_type="add_crossnet",
            status="survivor",
            metrics={"validation_best_auc": 0.75, "auc": 0.74, "logloss": 0.45},
            artifact_dir="",
            genome_path="",
            elapsed_sec=1.0,
            evolution_space="skill_space",
            operation="add",
            rationale="add cross interaction",
            architecture_fingerprint="arch_skill",
            training_config_hash="train_hash",
            genome_config_hash="genome_hash",
            hparam_fingerprint="hparam_hash",
        ),
        CandidateResult(
            candidate_id="round0_code_gate",
            genome_id="genome_code",
            parent_genome_id="genome_base",
            mutation_type="code_open_ended_gate",
            status="skipped_duplicate",
            metrics={},
            artifact_dir="",
            genome_path="",
            elapsed_sec=0.0,
            evolution_space="code_space",
            operation="open_ended",
            rationale="generated gate",
            architecture_fingerprint="arch_skill",
            duplicate_of="round0_skill_add",
            dedup_status="duplicate",
            training_config_hash="train_hash",
            genome_config_hash="genome_hash",
            hparam_fingerprint="hparam_hash",
        ),
    ]

    rows = runner._build_result_report_rows(results)
    row_types = {row["row_type"] for row in rows}

    assert {"candidate", "round_best", "auc_curve", "architecture_fingerprint_summary", "hparam_fingerprint_summary", "status_overview"} <= row_types
    candidate_rows = [row for row in rows if row["row_type"] == "candidate"]
    assert len(candidate_rows) == 2
    assert candidate_rows[0]["round_space_mix"] == "skill=1 code=1"
    assert candidate_rows[0]["test_auc"] == 0.74
    overview = next(row for row in rows if row["row_type"] == "status_overview")
    assert overview["total_skipped_count"] == 1

    _write_tsv(tmp_path / "results.tsv", rows)
    assert "architecture_fingerprint_summary" in (tmp_path / "results.tsv").read_text(encoding="utf-8")


def test_progress_results_write_round_history_during_run(tmp_path):
    runner = CTRModelEvolutionRunner(
        CTRWorkflowConfig(
            output_dir=str(tmp_path),
            write_metadata_files=False,
            write_summary_json=False,
            write_evolution_memory=False,
            write_code_space_diagnostics=False,
        )
    )
    runner.deduplication_records.append(
        {
            "round_idx": 0,
            "generated_candidates": 3,
            "unique_candidates": 2,
            "skipped_duplicates": 1,
            "skipped_candidate_ids": ["round0_code_dup"],
        }
    )
    runner.adaptive_budget_state["records"].append(
        {
            "round_idx": 0,
            "candidate_budget": 3,
            "skill_target": 2,
            "code_target": 1,
            "stagnant_rounds_before_round": 0,
            "stagnant_rounds_after_round": 0,
            "global_best_before_round": 0.70,
            "global_best_after_round": 0.75,
            "improvement": 0.05,
            "improved": True,
        }
    )
    results = [
        CandidateResult(
            candidate_id="baseline",
            genome_id="genome_base",
            parent_genome_id=None,
            mutation_type="baseline",
            status="baseline",
            metrics={"validation_best_auc": 0.70, "auc": 0.69, "logloss": 0.50},
            artifact_dir="",
            genome_path="",
            elapsed_sec=1.0,
            rationale="baseline",
        ),
        CandidateResult(
            candidate_id="round0_skill_add",
            genome_id="genome_skill",
            parent_genome_id="genome_base",
            mutation_type="add_crossnet",
            status="survivor",
            metrics={"validation_best_auc": 0.75, "auc": 0.74, "logloss": 0.45},
            artifact_dir="",
            genome_path="",
            elapsed_sec=2.0,
            evolution_space="skill_space",
            operation="add",
            rationale="add cross interaction",
        ),
        CandidateResult(
            candidate_id="round0_code_gate",
            genome_id="genome_code",
            parent_genome_id="genome_base",
            mutation_type="code_open_ended_gate",
            status="failed",
            metrics={},
            artifact_dir="",
            genome_path="",
            elapsed_sec=0.5,
            evolution_space="code_space",
            operation="open_ended",
            rationale="generated gate",
            error="RuntimeError: failed",
        ),
        CandidateResult(
            candidate_id="round0_code_dup",
            genome_id="genome_dup",
            parent_genome_id="genome_base",
            mutation_type="code_open_ended_dup",
            status="skipped_duplicate",
            metrics={},
            artifact_dir="",
            genome_path="",
            elapsed_sec=0.0,
            evolution_space="code_space",
            operation="open_ended",
            rationale="duplicate gate",
            duplicate_of="round0_skill_add",
            dedup_status="duplicate",
        ),
    ]
    results[1].promotion_result = {"promoted": True, "metadata": {"promotion_status": "promoted"}}

    runner._write_progress_results(results)

    assert (tmp_path / "results.tsv").exists()
    rows = pd.read_csv(tmp_path / "round_history.tsv", sep="\t").fillna("")
    baseline = rows[rows["row_type"] == "baseline"].iloc[0]
    round0 = rows[rows["row_type"] == "round"].iloc[0]
    assert baseline["best_candidate_id_so_far"] == "baseline"
    assert str(round0["round_idx"]) == "0"
    assert round0["generated_candidates"] == 3
    assert round0["unique_candidates"] == 2
    assert round0["skipped_duplicates"] == 1
    assert round0["round_best_candidate_id"] == "round0_skill_add"
    assert round0["best_candidate_id_so_far"] == "round0_skill_add"
    assert round0["survivor_candidate_ids"] == "round0_skill_add"
    assert round0["promoted_candidate_ids"] == "round0_skill_add"
    assert round0["failed_candidate_ids"] == "round0_code_gate"
    assert round0["skipped_duplicate_candidate_ids"] == "round0_code_dup"


def test_round_survivor_artifacts_save_only_survivor_genomes(tmp_path, monkeypatch):
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id"],
        vocab_sizes=[4, 8],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    genome = build_default_ctr_genome(bundle, CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0))
    config = CTRWorkflowConfig(
        output_dir=str(tmp_path),
        write_survivor_artifacts=True,
        survivor_artifacts_dir="kept",
        write_evolution_memory=False,
    )
    runner = CTRModelEvolutionRunner(config)

    def fake_repr(path, genome):
        path.write_text("compiled model\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_write_compiled_model_repr", fake_repr)
    survivor = CandidateResult(
        candidate_id="round0_keep",
        genome_id=genome.metadata.genome_id,
        parent_genome_id=None,
        mutation_type="add_crossnet_mix",
        status="survivor",
        metrics={"validation_best_auc": 0.75, "auc": 0.74, "logloss": 0.45},
        artifact_dir="",
        genome_path="",
        elapsed_sec=0.0,
        architecture_fingerprint=genome_architecture_fingerprint(genome),
        genome=genome,
    )
    discarded = CandidateResult(
        candidate_id="round0_drop",
        genome_id="drop",
        parent_genome_id=None,
        mutation_type="add_crossnet_v2",
        status="discarded",
        metrics={"validation_best_auc": 0.70, "auc": 0.69},
        artifact_dir="",
        genome_path="",
        elapsed_sec=0.0,
        genome=genome,
    )

    runner._write_round_survivor_artifacts(0, [survivor])

    survivor_dir = tmp_path / "kept" / "round00" / "round0_keep"
    assert (survivor_dir / "genome.json").exists()
    assert (survivor_dir / "metrics.json").exists()
    assert (survivor_dir / "structure.md").exists()
    assert (survivor_dir / "model_repr.txt").read_text(encoding="utf-8") == "compiled model\n"
    assert survivor.genome_path == str(survivor_dir / "genome.json")
    assert survivor.artifact_dir == str(survivor_dir)
    assert not (tmp_path / "kept" / "round00" / discarded.candidate_id).exists()


def test_survivor_selection_keeps_average_or_better_models_for_next_round():
    training_cfg = CTRTrainingConfig(objective_metric="auc", min_delta=0.0)
    evolution_cfg = CTREvolutionConfig(candidate_budget=10, survivor_top_k=4)
    results = [
        CandidateResult(
            candidate_id=f"candidate_{idx}",
            genome_id=f"genome_{idx}",
            parent_genome_id="parent",
            mutation_type="test",
            status="candidate",
            metrics={"validation_best_auc": auc, "auc": auc - 0.01},
            artifact_dir="out",
            genome_path="genome.json",
            elapsed_sec=0.0,
        )
        for idx, auc in enumerate([0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88])
    ]

    survivors = _select_survivors(results, training_cfg, evolution_cfg)

    assert [result.candidate_id for result in survivors] == ["candidate_9", "candidate_8", "candidate_7", "candidate_6"]


def test_adaptive_candidate_targets_keep_total_budget_and_increase_code_space_on_stagnation():
    evolution = CTREvolutionConfig(
        candidate_budget=10,
        code_space_probability=0.2,
        code_space=CodeSpaceConfig(provider="creative_command"),
        adaptive_candidate_budget=AdaptiveCandidateBudgetConfig(
            enabled=True,
            stagnation_patience=2,
            base_code_candidates=2,
            code_step=2,
            max_code_candidates=6,
            min_skill_candidates=4,
        ),
    )

    assert _adaptive_candidate_targets(evolution, stagnant_rounds=0) == (8, 2)
    assert _adaptive_candidate_targets(evolution, stagnant_rounds=1) == (8, 2)
    assert _adaptive_candidate_targets(evolution, stagnant_rounds=2) == (6, 4)
    assert _adaptive_candidate_targets(evolution, stagnant_rounds=3) == (4, 6)
    assert _adaptive_candidate_targets(evolution, stagnant_rounds=4) == (4, 6)


def test_best_objective_value_uses_validation_auc_and_logloss_direction():
    auc_results = [
        CandidateResult(
            candidate_id=f"auc_{idx}",
            genome_id=f"genome_{idx}",
            parent_genome_id="parent",
            mutation_type="test",
            status="candidate",
            metrics={"validation_best_auc": value, "auc": value - 0.01},
            artifact_dir="out",
            genome_path="genome.json",
            elapsed_sec=0.0,
        )
        for idx, value in enumerate([0.72, 0.75, 0.74])
    ]
    logloss_results = [
        CandidateResult(
            candidate_id=f"logloss_{idx}",
            genome_id=f"genome_{idx}",
            parent_genome_id="parent",
            mutation_type="test",
            status="candidate",
            metrics={"logloss": value},
            artifact_dir="out",
            genome_path="genome.json",
            elapsed_sec=0.0,
        )
        for idx, value in enumerate([0.61, 0.58, 0.60])
    ]

    assert _best_objective_value(auc_results, CTRTrainingConfig(objective_metric="auc")) == 0.75
    assert _best_objective_value(logloss_results, CTRTrainingConfig(objective_metric="logloss")) == 0.58


def test_gpu_id_parser_skips_disabled_values():
    assert _parse_gpu_ids("0, 1,2") == ["0", "1", "2"]
    assert _parse_gpu_ids("-1") == []
    assert _parse_gpu_ids("NoDevFiles") == []


def test_launcher_config_parses_conda_gpu_tasks(tmp_path):
    path = tmp_path / "launcher.yaml"
    path.write_text(
        """
conda_env: EvoSkillRec
gpu_ids: [0, 1]
max_parallel: 2
write_task_logs: false
write_launcher_summary: false
write_aggregate_outputs: false
write_aggregate_results_csv: true
aggregate_results_csv_path: outputs/evolution/aggregate_results.csv
tasks:
  - name: task0
    config: config0.yaml
    output_dir: out0
    device: cuda:0
""",
        encoding="utf-8",
    )

    config = load_launcher_config(path)

    assert config.conda_env == "EvoSkillRec"
    assert config.gpu_ids == [0, 1]
    assert config.write_task_logs is False
    assert config.write_launcher_summary is False
    assert config.write_aggregate_outputs is False
    assert config.write_aggregate_results_csv is True
    assert config.aggregate_results_csv_path == "outputs/evolution/aggregate_results.csv"
    assert config.tasks[0].name == "task0"


def test_launcher_aggregate_results_csv_rows_summarize_independent_runs(tmp_path):
    run0 = tmp_path / "seed2022"
    run1 = tmp_path / "seed2023"
    run0.mkdir()
    run1.mkdir()
    _write_tsv(
        run0 / "results.tsv",
        [
            {
                "row_type": "baseline",
                "candidate_id": "baseline",
                "candidate_description": "Initial CTR genome",
                "mutation_type": "baseline",
                "evolution_space": "skill_space",
                "validation_best_auc": 0.70,
                "test_auc": 0.69,
                "test_logloss": 0.50,
                "architecture_fingerprint": "arch_base",
                "hparam_fingerprint": "hparam0",
            },
            {
                "row_type": "candidate",
                "round_idx": 0,
                "candidate_id": "round0_skill_add",
                "candidate_description": "Add cross branch",
                "mutation_type": "add_crossnet",
                "evolution_space": "skill_space",
                "operation": "add",
                "status": "survivor",
                "promotion_status": "promoted",
                "validation_best_auc": 0.74,
                "test_auc": 0.73,
                "test_logloss": 0.47,
                "architecture_fingerprint": "arch_add",
                "hparam_fingerprint": "hparam0",
            },
            {
                "row_type": "round_best",
                "round_idx": 0,
                "candidate_id": "round0_skill_add",
                "candidate_description": "Add cross branch",
                "mutation_type": "add_crossnet",
                "evolution_space": "skill_space",
                "operation": "add",
                "validation_best_auc": 0.74,
                "test_auc": 0.73,
                "test_logloss": 0.47,
                "architecture_fingerprint": "arch_add",
                "hparam_fingerprint": "hparam0",
                "best_candidate_id_so_far": "round0_skill_add",
                "best_validation_auc_so_far": 0.74,
                "round_total_candidates": 10,
                "round_skill_candidates": 8,
                "round_code_candidates": 2,
                "round_failed_count": 0,
                "round_skipped_count": 0,
                "round_skipped_duplicate_count": 0,
                "round_promoted_count": 1,
            },
            {
                "row_type": "auc_curve",
                "round_idx": 0,
                "best_candidate_id_so_far": "round0_skill_add",
                "best_validation_auc_so_far": 0.74,
            },
            {
                "row_type": "architecture_fingerprint_summary",
                "round_idx": 0,
                "architecture_fingerprint_unique_count": 10,
                "architecture_fingerprint_counts": '{"arch_add": 1}',
            },
            {
                "row_type": "hparam_fingerprint_summary",
                "round_idx": 0,
                "hparam_fingerprint_unique_count": 1,
                "hparam_fingerprint_counts": '{"hparam0": 10}',
            },
            {
                "row_type": "status_overview",
                "total_failed_count": 0,
                "total_skipped_count": 0,
                "total_skipped_duplicate_count": 0,
                "total_promoted_count": 1,
                "promoted_candidate_ids": "round0_skill_add",
                "status_counts": '{"baseline": 1, "survivor": 1}',
            },
        ],
    )
    _write_tsv(
        run1 / "results.tsv",
        [
            {
                "row_type": "baseline",
                "candidate_id": "baseline",
                "candidate_description": "Initial CTR genome",
                "mutation_type": "baseline",
                "evolution_space": "skill_space",
                "validation_best_auc": 0.72,
                "test_auc": 0.71,
                "test_logloss": 0.49,
                "architecture_fingerprint": "arch_base",
                "hparam_fingerprint": "hparam1",
            },
            {
                "row_type": "candidate",
                "round_idx": 0,
                "candidate_id": "round0_code_gate",
                "candidate_description": "Generated logit gate",
                "mutation_type": "code_open_ended_gate",
                "evolution_space": "code_space",
                "operation": "open_ended",
                "status": "survivor",
                "validation_best_auc": 0.71,
                "test_auc": 0.70,
                "test_logloss": 0.51,
                "architecture_fingerprint": "arch_gate",
                "hparam_fingerprint": "hparam1",
            },
            {
                "row_type": "round_best",
                "round_idx": 0,
                "candidate_id": "round0_code_gate",
                "candidate_description": "Generated logit gate",
                "mutation_type": "code_open_ended_gate",
                "evolution_space": "code_space",
                "operation": "open_ended",
                "validation_best_auc": 0.71,
                "test_auc": 0.70,
                "test_logloss": 0.51,
                "architecture_fingerprint": "arch_gate",
                "hparam_fingerprint": "hparam1",
                "best_candidate_id_so_far": "baseline",
                "best_validation_auc_so_far": 0.72,
                "round_total_candidates": 10,
                "round_skill_candidates": 8,
                "round_code_candidates": 2,
                "round_failed_count": 0,
                "round_skipped_count": 0,
                "round_skipped_duplicate_count": 0,
                "round_promoted_count": 0,
            },
            {
                "row_type": "auc_curve",
                "round_idx": 0,
                "best_candidate_id_so_far": "baseline",
                "best_validation_auc_so_far": 0.72,
            },
            {
                "row_type": "architecture_fingerprint_summary",
                "round_idx": 0,
                "architecture_fingerprint_unique_count": 10,
                "architecture_fingerprint_counts": '{"arch_gate": 1}',
            },
            {
                "row_type": "hparam_fingerprint_summary",
                "round_idx": 0,
                "hparam_fingerprint_unique_count": 1,
                "hparam_fingerprint_counts": '{"hparam1": 10}',
            },
            {
                "row_type": "status_overview",
                "total_failed_count": 0,
                "total_skipped_count": 0,
                "total_skipped_duplicate_count": 0,
                "total_promoted_count": 0,
                "promoted_candidate_ids": "",
                "status_counts": '{"baseline": 1, "survivor": 1}',
            },
        ],
    )
    rows = _build_aggregate_results_csv_rows(
        [
            {
                "task_name": "ctr_movielens_seed2022",
                "output_dir": str(run0),
                "returncode": 0,
                "elapsed_sec": 12.0,
                "cmd": ["conda", "run", "-n", "rechub", "python", "-m", "workflow", "--seed", "2022"],
            },
            {
                "task_name": "ctr_movielens_seed2023",
                "output_dir": str(run1),
                "returncode": 0,
                "elapsed_sec": 13.0,
                "cmd": ["conda", "run", "-n", "rechub", "python", "-m", "workflow", "--seed", "2023"],
            },
        ]
    )

    run_round = next(row for row in rows if row["scope"] == "run" and row["stage"] == "round" and row["seed"] == "2022")
    aggregate_final = next(row for row in rows if row["scope"] == "aggregate" and row["stage"] == "final")
    aggregate_round = next(row for row in rows if row["scope"] == "aggregate" and row["stage"] == "round")

    assert run_round["round_total_candidates"] == 10
    assert abs(run_round["delta_vs_baseline"] - 0.04) < 1e-9
    assert aggregate_final["n_runs"] == 2
    assert aggregate_final["successful_runs"] == 2
    assert aggregate_final["best_seed"] == "2022"
    assert aggregate_final["best_candidate_id_so_far"] == "round0_skill_add"
    assert abs(aggregate_final["mean_best_validation_auc_so_far"] - 0.73) < 1e-9
    assert aggregate_final["improved_vs_baseline_run_count"] == 1
    assert aggregate_round["round_total_candidates"] == 20
    assert aggregate_round["round_skill_candidates"] == 16
    assert aggregate_round["round_code_candidates"] == 4
