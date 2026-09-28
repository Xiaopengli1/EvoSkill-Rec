import json
import textwrap

import pytest

from recskill.evolution.code_space import CodeSpaceConfig, inspect_open_ended_proposal_ingestions
from recskill.evolution.ctr_workflow import (
    AdaptiveCandidateBudgetConfig,
    CandidateResult,
    CandidateSpec,
    CTREvolutionConfig,
    CTRTrainingConfig,
    _skill_library_for_candidate,
)
from recskill.evolution.evolution_memory import EvolutionMemory
from recskill.evolution.mtl_code_space import build_mtl_code_space_provider, mtl_architecture_profile
from recskill.evolution.mtl_workflow import (
    MTLDatasetBundle,
    MTLDatasetConfig,
    MTLGenomeConfig,
    MTLModelEvolutionRunner,
    MTLWorkflowConfig,
    build_mtl_genome,
    generate_mtl_candidates,
    prepare_census_mtl_data,
)
from recskill.evolution.compiler import SkillGenomeCompiler
from recskill.evolution.open_ended import OpenEndedProposal
from recskill.evolution.skill_library import SkillLibrary
from recskill.evolution.verification import GenomeVerifier


def _bundle() -> MTLDatasetBundle:
    return MTLDatasetBundle(
        feature_names=["age", "worker", "education"],
        vocab_sizes=[8, 5, 6],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
        n_task=2,
        task_names=["income_ctr", "marital_cvr"],
        task_types=["classification", "classification"],
    )


def _genome_config(template: str = "shared_bottom") -> MTLGenomeConfig:
    return MTLGenomeConfig(
        template=template,
        embedding_dim=4,
        shared_hidden_dims=[16, 8],
        task_hidden_dims=[8],
        expert_dim=8,
        num_experts=2,
        ple_shared_experts=1,
        ple_specific_experts=1,
        dropout=0.0,
    )


def _mtl_macro_proposal(parent, *, output_key: str = "macro_flat_embeddings") -> OpenEndedProposal:
    downstream = next(
        edge.dst_node_id
        for edge in parent.edges
        if edge.src_node_id == "flatten" and edge.src_output_key == "flat_embeddings"
    )
    macro_code = textwrap.dedent(
        """
        import torch

        class MTLFlatArchitectureGate(torch.nn.Module):
            def __init__(self, input_key="flat_embeddings", output_key="macro_flat_embeddings", init_scale=0.0):
                super().__init__()
                self.input_key = input_key
                self.output_key = output_key
                self.scale = torch.nn.Parameter(torch.tensor(float(init_scale)))

            def forward(self, inputs):
                x = inputs[self.input_key]
                return {self.output_key: x + torch.tanh(self.scale) * x}
        """
    ).strip() + "\n"
    return OpenEndedProposal.from_dict(
        {
            "proposal_id": "mtl_macro_flat_architecture_gate",
            "proposal_type": "NEW_ROUTING_OR_GATING_DESIGN",
            "structural_scope": "macro",
            "target_failure_mode": "small_shared_representation_miscalibration",
            "architecture_hypothesis": "Insert a macro architecture gate on flat_embeddings before the shared transform.",
            "affected_genome_nodes": ["flatten"],
            "code": macro_code,
            "expected_input_signature": [{"name": "flat_embeddings", "shape": ["batch_size", "flat_input_dim"], "dtype": "float32"}],
            "expected_output_signature": [{"name": output_key, "shape": ["batch_size", "flat_input_dim"], "dtype": "float32"}],
            "class_name": "MTLFlatArchitectureGate",
            "skill_id": "mtl_macro_flat_architecture_gate",
            "task_types": ["multitask"],
            "init_params": {
                "input_key": "flat_embeddings",
                "output_key": output_key,
                "init_scale": 0.0,
            },
            "metadata": {
                "node_id": "mtl_macro_flat_architecture_gate",
                "structural_scope": "macro",
                "wiring": "insert_between",
                "upstream_node_id": "flatten",
                "downstream_node_id": downstream,
                "output_key": output_key,
                "macro_judgment": {
                    "current_architecture_family": "shared_transform_input_path",
                    "diagnosed_limitation": "flat shared input has no learned architecture gate before the transform",
                    "target_architecture_transformation": "insert a shape-preserving architecture gate before the shared transform",
                    "why_local_edit_is_insufficient": "a local calibration after the task heads cannot change shared representation routing",
                    "parent_evidence": ["flatten -> shared transform edge consumes flat_embeddings"],
                    "proposal_insight": "a shared-input gate can alter cross-task representation allocation before task splitting",
                    "preconditions": {"met": ["flat_embeddings edge exists"], "missing": []},
                    "wiring_plan": "insert_between flatten and the current shared transform consumer",
                    "ablation_plan": "remove the inserted gate and restore flatten -> shared transform",
                },
            },
        }
    )


def test_census_mtl_data_matches_run_census_sparse_dense_split(tmp_path):
    rows = [
        {
            "age": 0.1,
            "wage per hour": 0.0,
            "class of worker": 2,
            "education": 5,
            "income": 1,
            "marital status": 0,
        },
        {
            "age": 0.9,
            "wage per hour": 0.3,
            "class of worker": 3,
            "education": 6,
            "income": 0,
            "marital status": 1,
        },
    ]
    for split in ["train", "val", "test"]:
        path = tmp_path / f"{split}.csv"
        path.write_text(
            "age,wage per hour,class of worker,education,income,marital status\n"
            + "\n".join(
                ",".join(str(row[col]) for col in ["age", "wage per hour", "class of worker", "education", "income", "marital status"])
                for row in rows
            )
            + "\n",
            encoding="utf-8",
        )

    bundle = prepare_census_mtl_data(
        MTLDatasetConfig(
            train_path=str(tmp_path / "train.csv"),
            val_path=str(tmp_path / "val.csv"),
            test_path=str(tmp_path / "test.csv"),
            dense_cols=["age", "wage per hour"],
            label_cols=["income", "marital status"],
            task_names=["income_ctr", "marital_cvr"],
        ),
        CTRTrainingConfig(batch_size=2, num_workers=0),
    )
    batch_x, batch_y = next(iter(bundle.train_loader))

    assert bundle.sparse_feature_names == ["class of worker", "education"]
    assert bundle.dense_feature_names == ["age", "wage per hour"]
    assert bundle.vocab_sizes == [4, 7]
    assert bundle.metadata["num_dense_binned"] == 0
    assert bundle.metadata["dense_handling"] == "raw_continuous_dense_feature_path"
    assert set(batch_x) == {"sparse_features", "dense_values", "dense_features"}
    assert batch_x["sparse_features"].shape == (2, 2)
    assert batch_x["dense_values"].shape == (2, 2)
    assert batch_y.shape == (2, 2)


def test_mtl_genome_uses_dense_feature_path_for_run_census_dense_columns():
    bundle = MTLDatasetBundle(
        feature_names=["class of worker", "education", "age", "wage per hour"],
        sparse_feature_names=["class of worker", "education"],
        dense_feature_names=["age", "wage per hour"],
        vocab_sizes=[4, 7],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
        n_task=2,
        task_names=["income_ctr", "marital_cvr"],
        task_types=["classification", "classification"],
    )

    genome = build_mtl_genome(bundle, _genome_config("shared_bottom"))

    assert genome.get_node("field_embedding").params["feature_names"] == ["class of worker", "education"]
    assert genome.get_node("dense_feature_path").params["num_dense"] == 2
    assert genome.get_node("shared_bottom").params["input_dim"] == 16
    assert "dense_values" in genome.constraints.required_inputs
    assert any(edge.src_node_id == "dense_feature_path" and edge.dst_node_id == "flatten" for edge in genome.edges)
    GenomeVerifier().assert_valid(genome)
    SkillGenomeCompiler().compile(genome)


def test_mtl_genome_supports_aitm_shared_transform():
    bundle = MTLDatasetBundle(
        feature_names=["class of worker", "education", "age", "wage per hour"],
        sparse_feature_names=["class of worker", "education"],
        dense_feature_names=["age", "wage per hour"],
        vocab_sizes=[4, 7],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
        n_task=2,
        task_names=["income_ctr", "marital_cvr"],
        task_types=["classification", "classification"],
    )

    genome = build_mtl_genome(bundle, _genome_config("aitm"))

    assert genome.get_node("task_specific_towers").params["input_key"] == "flat_embeddings"
    assert genome.get_node("task_specific_towers").params["output_key"] == "task_representations"
    assert genome.get_node("aitm_transfer").params["input_dim"] == 8
    assert genome.get_node("aitm_transfer").params["input_key"] == "task_representations"
    assert genome.get_node("aitm_transfer").params["output_key"] == "task_representations"
    assert genome.get_node("task_tower").params["input_key"] == "task_representations"
    assert genome.metadata.tags[1] == "aitm"
    GenomeVerifier().assert_valid(genome)
    SkillGenomeCompiler().compile(genome)


def test_mtl_code_space_profile_counts_dense_fields_in_runtime_dims():
    bundle = MTLDatasetBundle(
        feature_names=["class of worker", "education", "age", "wage per hour"],
        sparse_feature_names=["class of worker", "education"],
        dense_feature_names=["age", "wage per hour"],
        vocab_sizes=[4, 7],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
        n_task=2,
        task_names=["income_ctr", "marital_cvr"],
        task_types=["classification", "classification"],
    )
    genome = build_mtl_genome(bundle, _genome_config("shared_bottom"))

    profile = mtl_architecture_profile(genome)

    assert profile["runtime_dims"]["num_sparse_fields"] == 2
    assert profile["runtime_dims"]["num_dense_fields"] == 2
    assert profile["runtime_dims"]["num_fields"] == 4
    assert profile["runtime_dims"]["flat_input_dim"] == 16
    assert any("dense_values -> dense_feature_path" in item for item in profile["tensor_flow"])
    field_point = next(point for point in profile["injection_points"] if point["tensor_key"] == "field_embeddings")
    assert field_point["upstream_node_id"] == "dense_feature_path"
    assert field_point["downstream_node_id"] == "flatten"
    assert field_point["tensor_shape"] == ["batch_size", "num_fields", "embedding_dim"]


def test_mtl_skill_space_reuses_promoted_generated_skill_from_global_pool(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    (root / "flat_gate.py").write_text(
        """
import torch
from torch import nn


class FlatGate(nn.Module):
    def __init__(self, input_key="flat_embeddings", output_key="gated_flat_embeddings", input_dim=1):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.ones(int(input_dim)))

    def forward(self, inputs):
        return {self.output_key: inputs[self.input_key] * self.scale}
""",
        encoding="utf-8",
    )
    (root / "flat_gate.skill.yaml").write_text(
        """
skill_id: global_flat_gate
name: global_flat_gate
skill_name: global_flat_gate
category: adapter
implementation_path: generated_skills/flat_gate.py
class_name: FlatGate
task_types: [ctr]
input_signature:
  - name: flat_embeddings
    shape: [batch_size, flat_input_dim]
    dtype: float32
output_signature:
  - name: gated_flat_embeddings
    shape: [batch_size, flat_input_dim]
    dtype: float32
composition:
  requires: [flat_embeddings]
  produces: [gated_flat_embeddings]
  common_upstream: [flatten]
  example_genome_fragment:
    skill: global_flat_gate
    params:
      input_key: flat_embeddings
      output_key: gated_flat_embeddings
      input_dim: ${flat_input_dim}
retrieval:
  aliases: [global_flat_gate, open_ended_evolution]
  task_types: [ctr]
  architecture_roles: [adapter, representation]
  input_modalities: [flat_embeddings]
  output_semantics: [flat_embeddings]
promotion_status: promoted
candidate_metrics:
  validation_best_auc: 0.9
created_from_open_ended_evolution: true
portability:
  scope: schema_agnostic
  reuse_enabled: true
""",
        encoding="utf-8",
    )
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)
    bundle = _bundle()
    parent = build_mtl_genome(bundle, _genome_config("shared_bottom"))
    evolution = CTREvolutionConfig(
        candidate_budget=2,
        templates=[{"name": "swap_mmoe", "template": "mmoe", "operation": "replace"}],
        code_space=CodeSpaceConfig(reuse_promoted_generated_skills=True),
    )

    specs = generate_mtl_candidates([parent], bundle, _genome_config("shared_bottom"), evolution, round_idx=0, skill_library=library)

    assert any(spec.operation in {"replace", "specialize"} for spec in specs)
    reuse = next(spec for spec in specs if spec.operation == "reuse_generated")
    assert reuse.generated_skill_id == "global_flat_gate"
    assert "unified MTL skill pool" in reuse.rationale
    node = next(node for node in reuse.genome.nodes if node.skill_id == "global_flat_gate")
    assert node.source == "generated_skill"
    assert node.task_types == ["multitask"]
    assert node.params["input_dim"] == 12
    assert any(edge.src_node_id == node.node_id and edge.dst_node_id == "shared_bottom" for edge in reuse.genome.edges)
    GenomeVerifier(skill_library=library).assert_valid(reuse.genome)
    SkillGenomeCompiler(skill_library=library).compile(reuse.genome)


def test_mtl_skill_space_scores_predefined_and_generated_in_one_pool(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    (root / "flat_gate.py").write_text(
        """
import torch
from torch import nn


class FlatGate(nn.Module):
    def __init__(self, input_key="flat_embeddings", output_key="gated_flat_embeddings", input_dim=1):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.scale = nn.Parameter(torch.ones(int(input_dim)))

    def forward(self, inputs):
        return {self.output_key: inputs[self.input_key] * self.scale}
""",
        encoding="utf-8",
    )
    for idx in range(3):
        (root / f"flat_gate_{idx}.skill.yaml").write_text(
            f"""
skill_id: generated_mtl_flat_gate_{idx}
name: generated_mtl_flat_gate_{idx}
skill_name: generated_mtl_flat_gate_{idx}
category: multitask
implementation_path: generated_skills/flat_gate.py
class_name: FlatGate
task_types: [multitask]
input_signature:
  - name: flat_embeddings
    shape: [batch_size, flat_input_dim]
    dtype: float32
output_signature:
  - name: generated_mtl_flat_gate_{idx}_embeddings
    shape: [batch_size, flat_input_dim]
    dtype: float32
composition:
  requires: [flat_embeddings]
  produces: [generated_mtl_flat_gate_{idx}_embeddings]
  common_upstream: [flatten]
  example_genome_fragment:
    skill: generated_mtl_flat_gate_{idx}
    params:
      input_key: flat_embeddings
      output_key: generated_mtl_flat_gate_{idx}_embeddings
      input_dim: ${{flat_input_dim}}
retrieval:
  aliases: [generated_mtl_flat_gate_{idx}, open_ended_evolution]
  task_types: [multitask]
  architecture_roles: [task_gate, representation]
  input_modalities: [flat_embeddings]
  output_semantics: [flat_embeddings]
promotion_status: promoted
candidate_metrics:
  validation_best_auc: 0.99
created_from_open_ended_evolution: true
portability:
  scope: schema_agnostic
  reuse_enabled: true
""",
            encoding="utf-8",
        )
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)
    bundle = _bundle()
    parent = build_mtl_genome(bundle, _genome_config("shared_bottom"))
    evolution = CTREvolutionConfig(
        candidate_budget=4,
        templates=[{"name": "swap_mmoe", "template": "mmoe", "operation": "replace"}],
        code_space=CodeSpaceConfig(reuse_promoted_generated_skills=True),
    )

    specs = generate_mtl_candidates([parent], bundle, _genome_config("shared_bottom"), evolution, round_idx=0, skill_library=library)

    assert len(specs) == 4
    assert sum(1 for spec in specs if spec.operation == "reuse_generated") >= 2
    assert any(spec.operation in {"replace", "specialize"} for spec in specs)
    assert all("unified MTL skill pool" in spec.rationale for spec in specs)


def test_mtl_template_candidates_carry_field_embedding_generated_insertions(tmp_path):
    root = tmp_path / "generated_skills"
    root.mkdir(parents=True)
    (root / "field_gate.py").write_text(
        """
import torch
from torch import nn


class FieldGate(nn.Module):
    def __init__(self, input_key="field_embeddings", output_key="field_gated_embeddings", num_fields=1, embedding_dim=1):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.gate = nn.Parameter(torch.zeros(int(num_fields), int(embedding_dim)))

    def forward(self, inputs):
        x = inputs[self.input_key]
        return {self.output_key: x * (1.0 + torch.tanh(self.gate).unsqueeze(0))}
""",
        encoding="utf-8",
    )
    (root / "field_gate.skill.yaml").write_text(
        """
skill_id: global_field_gate
name: global_field_gate
skill_name: global_field_gate
category: adapter
implementation_path: generated_skills/field_gate.py
class_name: FieldGate
task_types: [ctr]
input_signature:
  - name: field_embeddings
    shape: [batch_size, num_fields, embedding_dim]
    dtype: float32
output_signature:
  - name: field_gated_embeddings
    shape: [batch_size, num_fields, embedding_dim]
    dtype: float32
composition:
  requires: [field_embeddings]
  produces: [field_gated_embeddings]
  common_upstream: [field_embedding, dense_feature_path]
  example_genome_fragment:
    skill: global_field_gate
    params:
      input_key: field_embeddings
      output_key: field_gated_embeddings
      num_fields: ${num_fields}
      embedding_dim: ${embedding_dim}
retrieval:
  aliases: [global_field_gate, open_ended_evolution]
  task_types: [ctr]
  architecture_roles: [adapter, embedding_reparameterization]
  input_modalities: [field_embeddings]
  output_semantics: [field_embeddings]
promotion_status: promoted
candidate_metrics:
  validation_best_auc: 0.9
created_from_open_ended_evolution: true
portability:
  scope: schema_agnostic
  reuse_enabled: true
""",
        encoding="utf-8",
    )
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[root], include_generated=False)
    bundle = MTLDatasetBundle(
        feature_names=["class of worker", "education", "age", "wage per hour"],
        sparse_feature_names=["class of worker", "education"],
        dense_feature_names=["age", "wage per hour"],
        vocab_sizes=[4, 7],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
        n_task=2,
        task_names=["income_ctr", "marital_cvr"],
        task_types=["classification", "classification"],
    )
    parent = build_mtl_genome(bundle, _genome_config("shared_bottom"))
    reuse_evolution = CTREvolutionConfig(
        candidate_budget=2,
        templates=[{"name": "swap_mmoe", "template": "mmoe", "operation": "replace"}],
        code_space=CodeSpaceConfig(reuse_promoted_generated_skills=True),
    )

    reused = next(
        spec
        for spec in generate_mtl_candidates([parent], bundle, _genome_config("shared_bottom"), reuse_evolution, round_idx=0, skill_library=library)
        if spec.operation == "reuse_generated"
    )
    field_node = next(node for node in reused.genome.nodes if node.skill_id == "global_field_gate")
    assert field_node.params["num_fields"] == 4
    assert any(edge.src_node_id == "dense_feature_path" and edge.dst_node_id == field_node.node_id for edge in reused.genome.edges)
    assert any(edge.src_node_id == field_node.node_id and edge.dst_node_id == "flatten" for edge in reused.genome.edges)

    template_evolution = CTREvolutionConfig(
        candidate_budget=1,
        templates=[{"name": "swap_mmoe", "template": "mmoe", "operation": "replace"}],
    )
    candidate = generate_mtl_candidates([reused.genome], bundle, _genome_config("shared_bottom"), template_evolution, round_idx=1)[0]

    carried_node = next(node for node in candidate.genome.nodes if node.skill_id == "global_field_gate")
    assert any(edge.src_node_id == "dense_feature_path" and edge.dst_node_id == carried_node.node_id for edge in candidate.genome.edges)
    assert any(edge.src_node_id == carried_node.node_id and edge.dst_node_id == "flatten" for edge in candidate.genome.edges)
    GenomeVerifier(skill_library=library).assert_valid(candidate.genome)
    SkillGenomeCompiler(skill_library=library).compile(candidate.genome)


def test_mtl_code_space_rejects_proposal_file_provider():
    with pytest.raises(ValueError, match="no longer supports proposal_file"):
        build_mtl_code_space_provider(
            CodeSpaceConfig(
                provider="proposal_file",
            )
        )


def test_mtl_code_space_rejects_sketch_file_for_macro_scope(tmp_path):
    sketch_file = tmp_path / "mtl_sketches.json"
    sketch_file.write_text(json.dumps({"sketches": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="sketch_file/offline sketches are not supported"):
        build_mtl_code_space_provider(
            CodeSpaceConfig(
                provider="creative_command",
                sketch_file=str(sketch_file),
                synthesizer_command="python should_not_run.py",
                structural_scopes=["macro"],
            )
        )


def test_mtl_scope_split_rejects_non_macro_scope_provider():
    provider = build_mtl_code_space_provider(
        CodeSpaceConfig(
            provider="scope_split",
            structural_scopes=["macro"],
            scope_providers={"macro": {"provider": "proposal_file", "proposal_file": "deleted.json"}},
            macro_requires_live_provider=False,
            require_active_provider=True,
        )
    )

    with pytest.raises(RuntimeError, match="No active code-space providers"):
        provider.propose(parent=build_mtl_genome(_bundle(), _genome_config("shared_bottom")), diagnosis_report={}, evolution_memory=None, budget=1, round_idx=0)
    assert provider.scope_provider_status["macro"]["reason"] == "proposal_file_provider_removed"


def test_mtl_direct_open_ended_proposal_ingests_on_non_shared_bottom_parent(tmp_path):
    parent = build_mtl_genome(_bundle(), _genome_config("mmoe"))
    proposals = [_mtl_macro_proposal(parent)]
    ingestions = inspect_open_ended_proposal_ingestions(
        proposals=proposals,
        parent=parent,
        output_dir=tmp_path,
        memory=EvolutionMemory(tmp_path / "memory.jsonl", persist=True),
        staging_root=tmp_path / "staging",
    )

    assert [item.success for item in ingestions] == [True]
    for item in ingestions:
        GenomeVerifier(skill_library=_skill_library_for_candidate(item.ingestion.skill_card_path, genome=item.ingestion.genome)).assert_valid(
            item.ingestion.genome
        )


def test_mtl_direct_macro_proposal_ingests_on_current_parent(tmp_path):
    parent = build_mtl_genome(_bundle(), _genome_config("ple"))
    proposals = [_mtl_macro_proposal(parent)]

    assert proposals[0].structural_scope == "macro"
    assert proposals[0].metadata["upstream_node_id"] == "flatten"
    assert proposals[0].metadata["downstream_node_id"] == "ple_gate"

    ingested = inspect_open_ended_proposal_ingestions(
        proposals=proposals,
        parent=parent,
        output_dir=tmp_path,
        memory=EvolutionMemory(tmp_path / "memory.jsonl", persist=True),
        staging_root=tmp_path / "staging",
    )[0]

    assert ingested.success
    library = _skill_library_for_candidate(ingested.ingestion.skill_card_path, genome=ingested.ingestion.genome)
    GenomeVerifier(skill_library=library).assert_valid(ingested.ingestion.genome)
    SkillGenomeCompiler(skill_library=library).compile(ingested.ingestion.genome)


def test_mtl_template_candidates_carry_generated_insertions_from_code_space_parent(tmp_path):
    bundle = _bundle()
    parent = build_mtl_genome(bundle, _genome_config("shared_bottom"))
    proposal = _mtl_macro_proposal(parent)
    ingested = inspect_open_ended_proposal_ingestions(
        proposals=[proposal],
        parent=parent,
        output_dir=tmp_path,
        memory=EvolutionMemory(tmp_path / "memory.jsonl", persist=True),
        staging_root=tmp_path / "staging",
    )[0]
    assert ingested.success

    evolution = CTREvolutionConfig(
        candidate_budget=1,
        templates=[{"name": "swap_mmoe", "template": "mmoe", "operation": "replace"}],
    )
    candidate = generate_mtl_candidates([ingested.ingestion.genome], bundle, _genome_config("shared_bottom"), evolution, round_idx=1)[0]

    assert candidate.genome.get_node("mtl_macro_flat_architecture_gate").skill_id == "mtl_macro_flat_architecture_gate"
    assert any(edge.src_node_id == "mtl_macro_flat_architecture_gate" and edge.dst_node_id == "mmoe_gate" for edge in candidate.genome.edges)
    library = _skill_library_for_candidate(None, genome=candidate.genome)
    GenomeVerifier(skill_library=library).assert_valid(candidate.genome)
    SkillGenomeCompiler(skill_library=library).compile(candidate.genome)


def test_mtl_memory_records_are_marked_multitask(tmp_path):
    runner = MTLModelEvolutionRunner(
        MTLWorkflowConfig(
            output_dir=str(tmp_path),
            memory_path=str(tmp_path / "memory.jsonl"),
            write_evolution_memory=True,
        )
    )
    result = CandidateResult(
        candidate_id="candidate",
        genome_id="genome",
        parent_genome_id=None,
        mutation_type="mutation",
        status="candidate",
        metrics={},
        artifact_dir="",
        genome_path="",
        elapsed_sec=0.0,
    )

    runner._append_memory_record(result, initial_status="candidate")

    record = json.loads((tmp_path / "memory.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["task_type"] == "multitask"


def test_mtl_zero_code_probability_keeps_all_budget_in_skill_space(tmp_path):
    runner = MTLModelEvolutionRunner(
        MTLWorkflowConfig(
            output_dir=str(tmp_path),
            evolution=CTREvolutionConfig(
                candidate_budget=4,
                code_space_probability=0.0,
                code_space=CodeSpaceConfig(
                    provider="creative_command",
                    planner_command="python tools/code_space_llm_provider.py --mode planner --task multitask",
                    synthesizer_command="python tools/code_space_llm_provider.py --mode synthesizer --task multitask",
                    proposals_per_round=2,
                ),
            ),
        )
    )

    assert runner._round_candidate_targets() == (4, 0)
    assert runner.adaptive_budget_state["records"] == []


def test_mtl_adaptive_candidate_targets_match_ctr_runbook_split(tmp_path):
    runner = MTLModelEvolutionRunner(
        MTLWorkflowConfig(
            output_dir=str(tmp_path),
            evolution=CTREvolutionConfig(
                candidate_budget=10,
                code_space_probability=0.2,
                code_space=CodeSpaceConfig(
                    provider="creative_command",
                    planner_command="python tools/code_space_llm_provider.py --mode planner --task multitask",
                    synthesizer_command="python tools/code_space_llm_provider.py --mode synthesizer --task multitask",
                    proposals_per_round=2,
                ),
                adaptive_candidate_budget=AdaptiveCandidateBudgetConfig(
                    enabled=True,
                    stagnation_patience=2,
                    base_code_candidates=2,
                    code_step=2,
                    max_code_candidates=6,
                    min_skill_candidates=4,
                ),
            ),
        )
    )

    assert runner._round_candidate_targets() == (8, 2)
    runner.adaptive_budget_state["stagnant_rounds"] = 2
    assert runner._round_candidate_targets() == (6, 4)
    runner.adaptive_budget_state["stagnant_rounds"] = 3
    assert runner._round_candidate_targets() == (4, 6)


def test_mtl_code_space_diagnosis_includes_recent_history_and_retry_feedback(tmp_path):
    runner = MTLModelEvolutionRunner(
        MTLWorkflowConfig(
            output_dir=str(tmp_path),
            evolution=CTREvolutionConfig(
                candidate_budget=8,
                code_space_probability=0.25,
                code_space=CodeSpaceConfig(
                    provider="creative_command",
                    planner_command="python tools/code_space_llm_provider.py --mode planner --task multitask",
                    synthesizer_command="python tools/code_space_llm_provider.py --mode synthesizer --task multitask",
                    structural_scopes=["macro"],
                    max_retries=2,
                ),
            ),
        )
    )
    runner.bundle = _bundle()
    runner.bundle.metadata.update(
        {
            "num_sparse_features": 32,
            "num_dense_features": 7,
            "num_dense_binned": 0,
            "dense_handling": "raw_continuous_dense_feature_path",
            "flat_input_dim": 312,
        }
    )
    parent = build_mtl_genome(runner.bundle, _genome_config("shared_bottom"))
    runner.completed_results = [
        CandidateResult(
            candidate_id="baseline",
            genome_id="baseline_genome",
            parent_genome_id=None,
            mutation_type="baseline",
            status="baseline",
            metrics={
                "validation_best_auc": 0.969,
                "validation_auc__income_ctr": 0.951,
                "validation_auc__marital_cvr": 0.987,
                "auc": 0.970,
            },
            artifact_dir="",
            genome_path="",
            elapsed_sec=1.0,
        ),
        CandidateResult(
            candidate_id="round0_code_bad_shape",
            genome_id="bad_shape_genome",
            parent_genome_id="baseline_genome",
            mutation_type="code_open_ended_bad_shape",
            status="failed",
            metrics={},
            artifact_dir="",
            genome_path="",
            elapsed_sec=1.0,
            evolution_space="code_space",
            operation="open_ended",
            generated_skill_id="bad_shape",
            proposal_id="bad_shape",
            structural_scope="macro",
            error="RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x312 and 256x64)",
        ),
    ]
    runner.adaptive_budget_state["global_best_score"] = 0.969
    runner.adaptive_budget_state["stagnant_rounds"] = 2
    runner.adaptive_budget_state["active_code_candidates"] = 4
    runner.adaptive_budget_state["records"].append(
        {
            "round_idx": 1,
            "candidate_budget": 8,
            "skill_target": 4,
            "code_target": 4,
            "stagnant_rounds_before_round": 2,
            "global_best_before_round": 0.969,
        }
    )
    runner.candidate_generation_errors.append(
        {
            "round_idx": 0,
            "stage": "mtl_code_genome_invalid",
            "error": "ValueError: generated skill changed the flat_embeddings shape",
        }
    )

    report = runner._build_mtl_code_space_diagnosis(
        parent,
        round_idx=1,
        parent_idx=0,
        parent_budget=4,
        retained_parent_count=1,
        retry_feedback=[
            {
                "stage": "ingest_mtl_open_ended_proposals",
                "proposal_id": "missing_macro",
                "error": "missing macro_judgment fields",
            }
        ],
    )

    feedback = report["evolution_feedback"]
    assert report["previous_code_space_failures"][0]["proposal_id"] == "missing_macro"
    assert "Preserve the exact intercepted tensor shape" in report["repair_instruction"]
    assert feedback["dataset_metadata"]["flat_input_dim"] == 312
    assert feedback["adaptive_candidate_budget"]["current_round_target"]["code_target"] == 4
    assert feedback["recent_rounds"][0]["counts"]["code_space"] == 1
    assert "mat1 and mat2 shapes" in feedback["recent_failed_candidates"][0]["error"]
    assert feedback["recent_generation_errors"][0]["stage"] == "mtl_code_genome_invalid"


def test_mtl_skill_supplement_fills_budget_when_code_space_is_short(tmp_path):
    planner_script = tmp_path / "empty_planner.py"
    synthesizer_script = tmp_path / "unused_synthesizer.py"
    planner_script.write_text("print('')\n", encoding="utf-8")
    synthesizer_script.write_text("raise SystemExit('should not be called')\n", encoding="utf-8")
    runner = MTLModelEvolutionRunner(
        MTLWorkflowConfig(
            output_dir=str(tmp_path),
            write_code_space_diagnostics=False,
            evolution=CTREvolutionConfig(
                candidate_budget=8,
                code_space_probability=0.375,
                templates=[
                    {"name": "swap_mmoe", "template": "mmoe", "operation": "replace"},
                    {"name": "swap_ple", "template": "ple", "operation": "replace"},
                    {"name": "swap_task_specific", "template": "task_specific", "operation": "replace"},
                    {"name": "mmoe_more_experts", "template": "mmoe", "operation": "specialize", "num_experts": 8},
                    {"name": "wider_shared_bottom", "template": "shared_bottom", "operation": "specialize", "shared_hidden_dims": [32, 16]},
                ],
                code_space=CodeSpaceConfig(
                    provider="creative_command",
                    planner_command=f"python {planner_script}",
                    synthesizer_command=f"python {synthesizer_script}",
                    structural_scopes=["macro"],
                    proposals_per_round=3,
                    sketches_per_round=3,
                    max_retries=1,
                ),
            ),
        )
    )
    runner.bundle = _bundle()
    parent = build_mtl_genome(runner.bundle, _genome_config("shared_bottom"))

    specs = runner._generate_round_candidates([parent], round_idx=0)

    assert len(specs) == 8
    assert sum(1 for spec in specs if spec.evolution_space == "code_space") == 0
    assert sum(1 for spec in specs if spec.evolution_space == "skill_space") == 8
    assert runner.code_space_generation_records[-1]["stage"] == "mtl_skill_space_supplement"
    assert runner.code_space_generation_records[-1]["generated"] == 3


def test_mtl_required_live_provider_failure_does_not_fill_with_skill_candidates(tmp_path):
    planner_script = tmp_path / "empty_planner.py"
    synthesizer_script = tmp_path / "unused_synthesizer.py"
    planner_script.write_text("print('')\n", encoding="utf-8")
    synthesizer_script.write_text("raise SystemExit('should not be called')\n", encoding="utf-8")
    runner = MTLModelEvolutionRunner(
        MTLWorkflowConfig(
            output_dir=str(tmp_path),
            write_code_space_diagnostics=False,
            evolution=CTREvolutionConfig(
                candidate_budget=8,
                code_space_probability=0.375,
                code_space=CodeSpaceConfig(
                    provider="creative_command",
                    planner_command=f"python {planner_script}",
                    synthesizer_command=f"python {synthesizer_script}",
                    structural_scopes=["macro"],
                    max_retries=1,
                    require_active_provider=True,
                ),
            ),
        )
    )
    runner.bundle = _bundle()
    parent = build_mtl_genome(runner.bundle, _genome_config("shared_bottom"))

    with pytest.raises(RuntimeError, match="Required active MTL Code-space provider"):
        runner._generate_round_candidates([parent], round_idx=0)

    assert not any(
        record.get("stage") == "mtl_skill_space_supplement"
        for record in runner.code_space_generation_records
    )


def test_mtl_required_live_provider_accepts_partial_valid_candidates(tmp_path, monkeypatch):
    runner = MTLModelEvolutionRunner(
        MTLWorkflowConfig(
            output_dir=str(tmp_path),
            write_code_space_diagnostics=False,
            evolution=CTREvolutionConfig(
                candidate_budget=8,
                code_space_probability=0.375,
                code_space=CodeSpaceConfig(
                    provider="creative_command",
                    structural_scopes=["macro"],
                    require_active_provider=True,
                ),
            ),
        )
    )
    runner.bundle = _bundle()
    parent = build_mtl_genome(runner.bundle, _genome_config("shared_bottom"))
    code_genome = build_mtl_genome(runner.bundle, _genome_config("aitm"))
    code_spec = CandidateSpec(
        candidate_id="round0_code_partial",
        genome=code_genome,
        parent_genome_id=parent.metadata.genome_id,
        mutation_type="code_open_ended_partial",
        rationale="valid partial provider result",
        evolution_space="code_space",
        operation="open_ended",
        proposal_id="partial",
        structural_scope="macro",
    )
    monkeypatch.setattr(
        runner,
        "_generate_open_ended_code_candidates",
        lambda parent_population, round_idx, target_code: [code_spec],
    )

    specs = runner._generate_round_candidates([parent], round_idx=0)

    assert sum(1 for spec in specs if spec.evolution_space == "code_space") == 1
    assert any(spec.candidate_id == "round0_code_partial" for spec in specs)
    assert not any(
        record.get("stage") == "mtl_skill_space_supplement"
        for record in runner.code_space_generation_records
    )
