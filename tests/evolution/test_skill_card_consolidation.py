import yaml

from recskill.evolution import OpenEndedProposal, SkillCardConsolidator, SkillSignature, TensorSpec

from .helpers import SAFE_GENERATED_CODE


def test_skill_card_consolidation_writes_stage1_compatible_yaml(tmp_path):
    code_path = tmp_path / "safe_scale.py"
    code_path.write_text(SAFE_GENERATED_CODE, encoding="utf-8")
    proposal = OpenEndedProposal.from_dict(
        {
            "proposal_id": "card_scale",
            "proposal_type": "NEW_SKILL_INVENTION",
            "target_failure_mode": "calibration_gap",
            "architecture_hypothesis": "Scale dense features.",
            "affected_genome_nodes": ["source"],
            "code": SAFE_GENERATED_CODE,
            "expected_input_signature": [{"name": "x", "shape": ["batch_size", 4], "dtype": "float32"}],
            "expected_output_signature": [{"name": "scaled_x", "shape": ["batch_size", 4], "dtype": "float32"}],
            "expected_metric_improvement": None,
            "expected_risks": ["overfitting"],
            "ablation_plan": "",
            "fallback_plan": "",
            "author": "human",
            "skill_id": "card_scale_skill",
            "class_name": "ResidualScale",
            "task_types": ["ctr"],
        }
    )
    signature = SkillSignature(
        input_signature=[TensorSpec("x", ["batch_size", 4])],
        output_signature=[TensorSpec("scaled_x", ["batch_size", 4])],
        class_name="ResidualScale",
    )

    card = SkillCardConsolidator(repo_root=tmp_path, generated_root=tmp_path / "generated").consolidate(
        proposal,
        code_path,
        signature,
        {"static_check": {"passed": True}},
    )
    manifest = yaml.safe_load(card.path.read_text(encoding="utf-8"))

    assert manifest["skill_id"] == "card_scale_skill"
    assert manifest["implementation_path"] == "safe_scale.py"
    assert manifest["class_name"] == "ResidualScale"
    assert manifest["validation_status"] == "passed"
    assert manifest["composition"]["requires"] == ["x"]
    assert manifest["portability"]["reuse_enabled"] is True


def test_skill_card_consolidation_marks_fixed_ctr_shapes_not_reusable(tmp_path):
    code_path = tmp_path / "safe_scale.py"
    code_path.write_text(SAFE_GENERATED_CODE, encoding="utf-8")
    proposal = OpenEndedProposal.from_dict(
        {
            "proposal_id": "fixed_ctr",
            "proposal_type": "NEW_SKILL_INVENTION",
            "target_failure_mode": "high_order_interaction_underfitting",
            "architecture_hypothesis": "Use fixed field shape.",
            "affected_genome_nodes": ["field_embedding"],
            "code": SAFE_GENERATED_CODE,
            "expected_input_signature": [{"name": "field_embeddings", "shape": ["batch_size", 7, 16], "dtype": "float32"}],
            "expected_output_signature": [{"name": "fixed_logit", "shape": ["batch_size", 1], "dtype": "float32"}],
            "author": "human",
            "skill_id": "fixed_ctr_skill",
            "class_name": "ResidualScale",
            "task_types": ["ctr"],
        }
    )
    signature = SkillSignature(
        input_signature=[TensorSpec("field_embeddings", ["batch_size", 7, 16])],
        output_signature=[TensorSpec("fixed_logit", ["batch_size", 1])],
        class_name="ResidualScale",
    )

    card = SkillCardConsolidator(repo_root=tmp_path, generated_root=tmp_path / "generated").consolidate(
        proposal,
        code_path,
        signature,
        {"static_check": {"passed": True}},
        init_params={"num_fields": 7, "embedding_dim": 16, "input_dim": 112},
    )
    manifest = yaml.safe_load(card.path.read_text(encoding="utf-8"))

    assert manifest["portability"]["scope"] == "dataset_specific"
    assert manifest["portability"]["reuse_enabled"] is False


def test_skill_card_consolidation_writes_portable_ctr_placeholders(tmp_path):
    code_path = tmp_path / "safe_scale.py"
    code_path.write_text(SAFE_GENERATED_CODE, encoding="utf-8")
    proposal = OpenEndedProposal.from_dict(
        {
            "proposal_id": "portable_ctr",
            "proposal_type": "NEW_SKILL_INVENTION",
            "target_failure_mode": "high_order_interaction_underfitting",
            "architecture_hypothesis": "Use symbolic field shape.",
            "affected_genome_nodes": ["field_embedding"],
            "code": SAFE_GENERATED_CODE,
            "expected_input_signature": [{"name": "field_embeddings", "shape": ["batch_size", "num_fields", "embedding_dim"], "dtype": "float32"}],
            "expected_output_signature": [{"name": "portable_logit", "shape": ["batch_size", 1], "dtype": "float32"}],
            "author": "human",
            "skill_id": "portable_ctr_skill",
            "class_name": "ResidualScale",
            "task_types": ["ctr"],
        }
    )
    signature = SkillSignature(
        input_signature=[TensorSpec("field_embeddings", ["batch_size", "num_fields", "embedding_dim"])],
        output_signature=[TensorSpec("portable_logit", ["batch_size", 1])],
        class_name="ResidualScale",
    )

    card = SkillCardConsolidator(repo_root=tmp_path, generated_root=tmp_path / "generated").consolidate(
        proposal,
        code_path,
        signature,
        {"static_check": {"passed": True}, "portability_tests": {"passed": True}},
        init_params={"num_fields": 7, "embedding_dim": 16, "input_dim": 112},
    )
    manifest = yaml.safe_load(card.path.read_text(encoding="utf-8"))

    assert manifest["portability"]["scope"] == "schema_agnostic"
    assert manifest["portability"]["reuse_enabled"] is True
    params = manifest["composition"]["example_genome_fragment"]["params"]
    assert params["num_fields"] == "${num_fields}"
    assert params["embedding_dim"] == "${embedding_dim}"
    assert params["input_dim"] == "${flat_input_dim}"
