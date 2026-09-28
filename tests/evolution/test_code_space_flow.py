import json
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml
import pytest

from recskill.evolution import ArchitectureSketch, CodeSpaceConfig, EvolutionMemory, build_code_space_provider
from recskill.evolution.code_space import ingest_open_ended_proposals, select_architecture_sketches
from recskill.evolution.compiler import SkillGenomeCompiler
from recskill.evolution.ctr_workflow import CTRDatasetBundle, CTRGenomeConfig, build_default_ctr_genome
from recskill.evolution.ctr_workflow import _rewrite_generated_skill_reference
from recskill.evolution.open_ended import OpenEndedProposal
from recskill.evolution.skill_sedimentation import promote_generated_skill, should_promote_generated_skill
from recskill.evolution.skill_library import SkillLibrary
from recskill.evolution.verification import GenomeVerifier

from .helpers import SAFE_GENERATED_CODE, base_genome


def _proposal_payload() -> dict:
    return {
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
        "author": "human",
        "skill_id": "llm_r00_safe_scale_skill",
        "class_name": "ResidualScale",
        "task_types": ["ctr"],
        "init_params": {"scale": 0.5},
    }


def _macro_fusion_payload() -> dict:
    code = textwrap.dedent(
        """
        import torch
        from torch import nn

        class MacroFusion(nn.Module):
            def __init__(self, input_keys=None, output_key="macro_logits"):
                super().__init__()
                self.input_keys = list(input_keys or ["linear_logit", "fm_output", "deep_logit"])
                self.output_key = output_key
                self.scale = nn.Parameter(torch.tensor(0.0))

            def forward(self, inputs):
                logits = torch.cat([inputs[key] for key in self.input_keys], dim=-1)
                baseline = logits.sum(dim=-1, keepdim=True)
                correction = logits.std(dim=-1, keepdim=True, unbiased=False)
                return {self.output_key: baseline + self.scale * correction}
        """
    ).strip() + "\n"
    return {
        "proposal_id": "macro_fusion",
        "proposal_type": "NEW_FUSION_DESIGN",
        "structural_scope": "macro",
        "target_failure_mode": "fusion_bottleneck",
        "architecture_hypothesis": "Replace additive fusion with generated multi-branch fusion.",
        "affected_genome_nodes": ["linear", "fm", "deep_tower"],
        "code": code,
        "expected_input_signature": [
            {"name": "linear_logit", "shape": ["batch_size", 1], "dtype": "float32"},
            {"name": "fm_output", "shape": ["batch_size", 1], "dtype": "float32"},
            {"name": "deep_logit", "shape": ["batch_size", 1], "dtype": "float32"},
        ],
        "expected_output_signature": [{"name": "macro_logits", "shape": ["batch_size", 1], "dtype": "float32"}],
        "skill_id": "llm_r00_macro_fusion",
        "class_name": "MacroFusion",
        "task_types": ["ctr"],
        "init_params": {"input_keys": ["linear_logit", "fm_output", "deep_logit"], "output_key": "macro_logits"},
        "metadata": {
            "node_id": "llm_r00_macro_fusion",
            "structural_scope": "macro",
            "wiring": "replace_fusion",
            "old_logits_key": "logits",
            "output_key": "macro_logits",
            "macro_judgment": {
                "current_architecture_family": "deepfm_additive_linear_fm_mlp",
                "diagnosed_limitation": "fixed additive fusion cannot adapt branch evidence per sample",
                "target_architecture_transformation": "replace additive fusion with generated multi-branch fusion",
                "why_local_edit_is_insufficient": "a local calibration cannot change how branch logits are combined",
                "parent_evidence": ["fusion consumes linear_logit, fm_output, and deep_logit"],
                "proposal_insight": "branch dispersion can indicate when additive evidence needs correction",
                "preconditions": {"met": ["multiple branch logits"], "missing": []},
                "wiring_plan": "replace prediction/loss logits with macro fusion output",
                "ablation_plan": "compare against original additive fusion",
            },
            "edges": [
                {"src_node_id": "linear", "dst_node_id": "NEW_NODE", "src_output_key": "linear_logit", "dst_input_key": "linear_logit"},
                {"src_node_id": "fm", "dst_node_id": "NEW_NODE", "src_output_key": "fm_output", "dst_input_key": "fm_output"},
                {"src_node_id": "deep_tower", "dst_node_id": "NEW_NODE", "src_output_key": "deep_logit", "dst_input_key": "deep_logit"},
            ],
        },
    }


def test_proposal_file_provider_is_removed():
    with pytest.raises(ValueError, match="no longer supports proposal_file"):
        build_code_space_provider(
            CodeSpaceConfig(provider="proposal_file", structural_scopes=["macro"])
        )


def test_code_space_config_defaults_to_macro_only():
    config = CodeSpaceConfig()

    assert config.structural_scopes == ["macro"]
    assert "calibration" not in config.diversity_lanes


def test_default_ctr_config_uses_macro_llm_wrapper():
    config_path = Path(__file__).resolve().parents[2] / "examples" / "evolution" / "ctr_movielens_evolution.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    code_space = config["evolution"]["code_space"]

    assert code_space["provider"] == "creative_command"
    assert code_space["planner_command"] == "python tools/code_space_llm_provider.py --mode planner"
    assert code_space["synthesizer_command"] == "python tools/code_space_llm_provider.py --mode synthesizer"
    assert code_space["structural_scopes"] == ["macro"]
    assert code_space["allow_fallback_templates"] is False
    assert code_space["require_active_provider"] is True


def test_mtl_live_config_uses_multitask_llm_wrapper():
    config_path = Path(__file__).resolve().parents[2] / "examples" / "evolution" / "mtl_census_codespace.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    code_space = config["evolution"]["code_space"]

    assert code_space["provider"] == "creative_command"
    assert code_space["planner_command"] == "python tools/code_space_llm_provider.py --mode planner --task multitask"
    assert code_space["synthesizer_command"] == "python tools/code_space_llm_provider.py --mode synthesizer --task multitask"
    assert code_space["structural_scopes"] == ["macro"]
    assert code_space["allow_fallback_templates"] is False
    assert code_space["require_active_provider"] is True


def test_scope_split_rejects_proposal_file_scope_provider():
    provider = build_code_space_provider(
        CodeSpaceConfig(
            provider="scope_split",
            structural_scopes=["macro"],
            scope_providers={
                "macro": {"provider": "proposal_file", "proposal_file": "deleted.json"},
            },
            require_active_provider=True,
        )
    )

    with pytest.raises(RuntimeError, match="No active code-space providers"):
        provider.propose(parent=base_genome(), diagnosis_report={}, evolution_memory=None, budget=2, round_idx=0)
    assert provider.scope_provider_status["macro"]["reason"] == "proposal_file_provider_removed"


def test_scope_split_accepts_live_macro_command_provider(tmp_path):
    macro_script = tmp_path / "macro_provider.py"
    macro_script.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys

            prompt = json.load(sys.stdin)
            assert prompt["diagnosis_report"]["target_structural_scope"] == "macro"
            proposal = {json.dumps(_macro_fusion_payload())}
            print(json.dumps({{"proposals": [proposal]}}))
            """
        ),
        encoding="utf-8",
    )
    provider = build_code_space_provider(
        CodeSpaceConfig(
            provider="scope_split",
            structural_scopes=["macro"],
            scope_providers={
                "macro": {"provider": "command", "command": _python_command(macro_script)},
            },
            macro_requires_live_provider=True,
        )
    )

    proposals = provider.propose(parent=base_genome(), diagnosis_report={}, evolution_memory=None, budget=2, round_idx=0)

    assert [(proposal.proposal_id, proposal.structural_scope) for proposal in proposals] == [("macro_fusion", "macro")]


def test_scope_split_rejects_macro_sketch_file_when_live_macro_required(tmp_path):
    sketch_path = tmp_path / "macro_sketches.json"
    synthesizer_script = tmp_path / "synthesizer.py"
    sketch_path.write_text(
        json.dumps(
            {
                "sketches": [
                    {
                        "sketch_id": "offline_macro",
                        "innovation_lane": "fusion",
                        "input_keys": ["x"],
                        "output_keys": ["macro_logits"],
                        "wiring": "replace_fusion",
                        "structural_scope": "macro",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    synthesizer_script.write_text("raise SystemExit('should not be called')\n", encoding="utf-8")
    config = CodeSpaceConfig(
        provider="scope_split",
        structural_scopes=["macro"],
        scope_providers={
            "macro": {
                "provider": "creative_command",
                "sketch_file": str(sketch_path),
                "synthesizer_command": _python_command(synthesizer_script),
            },
        },
        macro_requires_live_provider=True,
    )
    provider = build_code_space_provider(config)

    proposals = provider.propose(parent=base_genome(), diagnosis_report={}, evolution_memory=None, budget=2, round_idx=0)

    assert proposals == []
    assert provider.scope_provider_status["macro"]["reason"] == "sketch_file_provider_removed"


def test_live_macro_prompt_includes_architecture_profile_and_judgment_requirements(tmp_path):
    parent = build_default_ctr_genome(
        CTRDatasetBundle(
            feature_names=["user_id", "movie_id", "gender"],
            vocab_sizes=[4, 8, 3],
            train_loader=None,
            val_loader=None,
            test_loader=None,
            metadata={},
        ),
        CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0),
    )
    macro_script = tmp_path / "macro_provider.py"
    macro_script.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys

            prompt = json.load(sys.stdin)
            assert prompt["diagnosis_report"]["target_structural_scope"] == "macro"
            assert prompt["current_architecture_profile"]["inferred_architecture_family"]
            assert "fixed_additive_fusion_limits_sample_adaptive_branch_weighting" in prompt["current_architecture_profile"]["diagnostic_cues"]
            assert "macro_judgment_requirements" in prompt
            proposal = {json.dumps(_macro_fusion_payload())}
            proposal["metadata"]["macro_judgment"] = {{
                "current_architecture_family": prompt["current_architecture_profile"]["inferred_architecture_family"],
                "diagnosed_limitation": "fixed additive fusion",
                "target_architecture_transformation": "sample-adaptive fusion replacement",
                "why_local_edit_is_insufficient": "calibration cannot change branch evidence composition",
                "parent_evidence": ["fusion node has multiple branch logits"],
                "proposal_insight": "adaptive fusion can react to branch disagreement that additive fusion ignores",
                "preconditions": {{"met": ["multiple branch logits"], "missing": []}},
                "wiring_plan": "replace fusion logits",
                "ablation_plan": "fall back to additive fusion",
            }}
            print(json.dumps({{"proposals": [proposal]}}))
            """
        ),
        encoding="utf-8",
    )
    provider = build_code_space_provider(
        CodeSpaceConfig(
            provider="scope_split",
            structural_scopes=["macro"],
            scope_providers={
                "macro": {"provider": "command", "command": _python_command(macro_script)},
            },
            macro_requires_live_provider=True,
        )
    )

    proposals = provider.propose(parent=parent, diagnosis_report={}, evolution_memory=None, budget=1, round_idx=0)

    assert proposals[0].metadata["macro_judgment"]["target_architecture_transformation"] == "sample-adaptive fusion replacement"


def test_architecture_sketch_selection_filters_and_diversifies():
    parent = base_genome()
    sketches = [
        ArchitectureSketch.from_dict(
            {
                "sketch_id": "field_gate",
                "innovation_lane": "gating",
                "target_failure_mode": "weak_field_signal",
                "hypothesis": "Gate a field residual before fusion.",
                "affected_genome_nodes": ["source"],
                "input_keys": ["x"],
                "output_keys": ["field_gate_logit"],
                "wiring": "branch_to_fusion",
                "core_operator": {"type": "field_gate"},
                "complexity_budget": {"max_params": 128},
            }
        ),
        ArchitectureSketch.from_dict(
            {
                "sketch_id": "bad_missing_input",
                "innovation_lane": "interaction",
                "hypothesis": "Consumes a missing tensor.",
                "input_keys": ["missing"],
                "output_keys": ["missing_logit"],
                "wiring": "branch_to_fusion",
            }
        ),
        ArchitectureSketch.from_dict(
            {
                    "sketch_id": "local_logit_calibrator",
                "innovation_lane": "calibration",
                "target_failure_mode": "calibration_gap",
                "hypothesis": "Calibrate logits.",
                "input_keys": ["x"],
                "output_keys": ["calibrated_logit"],
                "wiring": "reroute_logits",
                "core_operator": {"type": "bounded_residual"},
            }
        ),
        ArchitectureSketch.from_dict(
            {
                "sketch_id": "macro_router",
                "innovation_lane": "fusion",
                "target_failure_mode": "fusion_bottleneck",
                "hypothesis": "Replace additive fusion with generated adaptive routing.",
                "input_keys": ["x"],
                "output_keys": ["macro_logit"],
                "wiring": "replace_fusion",
                "core_operator": {"type": "adaptive_fusion_router"},
            }
        ),
    ]

    selected = select_architecture_sketches(
        sketches,
        parent=parent,
        budget=2,
        diversity_lanes=["gating", "fusion", "calibration"],
        structural_scopes=["macro"],
    )

    assert [sketch.sketch_id for sketch in selected] == ["field_gate", "macro_router"]


def test_architecture_sketch_selection_is_macro_only():
    parent = base_genome()
    sketches = [
        ArchitectureSketch.from_dict(
            {
                "sketch_id": "local_calibrator",
                "innovation_lane": "calibration",
                "hypothesis": "Calibrate output.",
                "input_keys": ["x"],
                "output_keys": ["calibrated_logit"],
                "wiring": "reroute_logits",
                "structural_scope": "macro",
            }
        ),
        ArchitectureSketch.from_dict(
            {
                "sketch_id": "macro_router",
                "innovation_lane": "fusion",
                "hypothesis": "Replace fusion with generated routing.",
                "input_keys": ["x"],
                "output_keys": ["macro_logit"],
                "wiring": "replace_fusion",
                "structural_scope": "macro",
            }
        ),
    ]

    selected = select_architecture_sketches(
        sketches,
        parent=parent,
        budget=2,
        structural_scopes=["macro"],
    )

    assert [(sketch.sketch_id, sketch.structural_scope) for sketch in selected] == [
        ("macro_router", "macro"),
    ]


def test_creative_command_provider_plans_sketches_then_synthesizes_proposals(tmp_path):
    sketch_prompt = tmp_path / "sketch_prompt.json"
    synthesis_prompt = tmp_path / "synthesis_prompt.json"
    planner_script = tmp_path / "planner.py"
    synthesizer_script = tmp_path / "synthesizer.py"
    planner_script.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            prompt = json.load(sys.stdin)
            assert "design_brief" in prompt
            print(json.dumps({
                "sketches": [
                    {
                        "sketch_id": "creative_field_gate",
                        "innovation_lane": "gating",
                        "target_failure_mode": "weak_field_signal",
                        "hypothesis": "Create a learnable field gate residual from x.",
                        "affected_genome_nodes": ["source"],
                        "input_keys": ["x"],
                        "output_keys": ["creative_field_gate_logit"],
                        "wiring": "branch_to_fusion",
                        "core_operator": {"type": "learnable_gate"},
                        "complexity_budget": {"max_params": 16},
                    },
                    {
                        "sketch_id": "invalid_missing",
                        "innovation_lane": "interaction",
                        "input_keys": ["missing"],
                        "output_keys": ["invalid_logit"],
                        "wiring": "branch_to_fusion",
                    },
                ]
            }))
            """
        ),
        encoding="utf-8",
    )
    synthesizer_script.write_text(
        textwrap.dedent(
            """
            import json
            import sys
            import textwrap

            prompt = json.load(sys.stdin)
            assert prompt["selected_sketches"][0]["sketch_id"] == "creative_field_gate"
            code = textwrap.dedent('''
                import torch
                from torch import nn

                class CreativeFieldGate(nn.Module):
                    def __init__(self, input_key="x", output_key="creative_field_gate_logit", initial_scale=0.1):
                        super().__init__()
                        self.input_key = input_key
                        self.output_key = output_key
                        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))

                    def forward(self, inputs):
                        x = inputs[self.input_key]
                        return {self.output_key: self.scale * x.mean(dim=-1, keepdim=True)}
            ''').strip() + "\\n"
            proposal = {
                "proposal_id": "creative_field_gate",
                "proposal_type": "NEW_ROUTING_OR_GATING_DESIGN",
                "target_failure_mode": "weak_field_signal",
                "architecture_hypothesis": "Create a learnable field gate residual from x.",
                "affected_genome_nodes": ["source"],
                "code": code,
                "expected_input_signature": [{"name": "x", "shape": ["batch_size", 4], "dtype": "float32"}],
                "expected_output_signature": [
                    {"name": "creative_field_gate_logit", "shape": ["batch_size", 1], "dtype": "float32"}
                ],
                "skill_id": "llm_r00_creative_field_gate",
                "class_name": "CreativeFieldGate",
                "task_types": ["ctr"],
                "init_params": {
                    "input_key": "x",
                    "output_key": "creative_field_gate_logit",
                    "initial_scale": 0.1,
                },
                "metadata": {"source_sketch_id": "creative_field_gate"},
            }
            print(json.dumps({"proposals": [proposal]}))
            """
        ),
        encoding="utf-8",
    )
    provider = build_code_space_provider(
        CodeSpaceConfig(
            provider="creative_command",
            planner_command=_python_command(planner_script),
            synthesizer_command=_python_command(synthesizer_script),
            sketches_per_round=2,
            proposals_per_round=1,
            diversity_lanes=["gating", "interaction"],
            structural_scopes=["macro"],
            sketch_prompt_path=str(sketch_prompt),
            synthesis_prompt_path=str(synthesis_prompt),
        )
    )

    proposals = provider.propose(
        parent=base_genome(),
        diagnosis_report={"failure_modes": ["weak_field_signal"]},
        evolution_memory=None,
        budget=1,
        round_idx=0,
    )

    assert len(proposals) == 1
    assert proposals[0].proposal_id == "creative_field_gate"
    assert proposals[0].metadata["source_sketch_id"] == "creative_field_gate"
    assert proposals[0].metadata["innovation_lane"] == "gating"
    assert proposals[0].metadata["wiring"] == "branch_to_fusion"
    assert sketch_prompt.exists()
    assert synthesis_prompt.exists()


def test_creative_command_provider_rejects_sketch_file_for_macro_scope(tmp_path):
    sketch_file = tmp_path / "sketches.yaml"
    synthesizer_script = tmp_path / "synthesizer.py"
    sketch_file.write_text(
        yaml.safe_dump(
            {
                "sketches": [
                    {
                        "sketch_id": "file_calibrator",
                        "innovation_lane": "calibration",
                        "target_failure_mode": "calibration_gap",
                        "hypothesis": "Add a bounded residual calibration logit.",
                        "affected_genome_nodes": ["source"],
                        "input_keys": ["x"],
                        "output_keys": ["file_calibrator_logit"],
                        "wiring": "branch_to_fusion",
                        "core_operator": {"type": "bounded_residual_calibrator"},
                        "complexity_budget": {"max_params": 8},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    synthesizer_script.write_text("raise SystemExit('should not be called')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sketch_file/offline sketches are not supported"):
        build_code_space_provider(
            CodeSpaceConfig(
                provider="creative_command",
                sketch_file=str(sketch_file),
                synthesizer_command=_python_command(synthesizer_script),
                sketches_per_round=1,
                proposals_per_round=1,
                diversity_lanes=["calibration"],
                structural_scopes=["macro"],
            )
        )


def test_open_ended_candidate_can_stage_and_promote_generated_skill(tmp_path):
    proposals = [OpenEndedProposal.from_dict(_proposal_payload())]
    memory = EvolutionMemory(tmp_path / "memory.jsonl")

    ingested = ingest_open_ended_proposals(proposals=proposals, parent=base_genome(), output_dir=tmp_path / "run", memory=memory)

    assert len(ingested) == 1
    ingestion = ingested[0].ingestion
    assert ingestion.skill_id == "llm_r00_safe_scale_skill"
    assert Path(ingestion.code_path).exists()
    assert Path(ingestion.skill_card_path).exists()
    assert should_promote_generated_skill(candidate_status="survivor", candidate_error=None, policy="survivor")

    promotion = promote_generated_skill(
        skill_id=ingestion.skill_id,
        staged_code_path=ingestion.code_path,
        staged_card_path=ingestion.skill_card_path,
        candidate_id="round0_code_llm_r00_safe_scale_skill",
        candidate_metrics={"validation_best_auc": 0.8},
        candidate_status="survivor",
        genome=ingestion.genome,
        repo_root=tmp_path,
        generated_root=tmp_path / "generated_skills",
    )

    assert promotion.promoted
    assert promotion.skill_id == "safe_scale_skill"
    assert Path(promotion.code_path).exists()
    assert Path(promotion.skill_card_path).exists()
    assert Path(promotion.code_path).name == "safe_scale_skill.py"
    assert Path(promotion.skill_card_path).name == "safe_scale_skill.skill.yaml"
    manifest = yaml.safe_load(Path(promotion.skill_card_path).read_text(encoding="utf-8"))
    assert manifest["skill_id"] == "safe_scale_skill"
    assert manifest["name"] == "safe_scale_skill"
    assert manifest["skill_name"] == "safe_scale_skill"
    assert manifest["promotion_status"] == "promoted"
    assert manifest["candidate_id"] == "round0_code_llm_r00_safe_scale_skill"
    assert manifest["metadata"]["origin_skill_id"] == "llm_r00_safe_scale_skill"
    assert manifest["metadata"]["persisted_skill_id"] == "safe_scale_skill"
    assert manifest["composition"]["example_genome_fragment"]["skill"] == "safe_scale_skill"
    assert "safe_scale_skill" in manifest["retrieval"]["aliases"]
    assert "llm_r00_safe_scale_skill" in manifest["retrieval"]["aliases"]

    _rewrite_generated_skill_reference(ingestion.genome, ingestion.skill_id, promotion.skill_id)
    generated_node = ingestion.genome.get_node("llm_r00_safe_scale_skill")
    assert generated_node.skill_id == "safe_scale_skill"
    assert generated_node.metadata["origin_skill_id"] == "llm_r00_safe_scale_skill"
    library = SkillLibrary(repo_root=tmp_path, skill_roots=[tmp_path / "generated_skills"])
    assert library.has(generated_node.skill_id)
    assert library.get(generated_node.skill_id).implementation_path == "generated_skills/safe_scale_skill.py"


def test_macro_open_ended_candidate_can_replace_prediction_logits(tmp_path):
    bundle = CTRDatasetBundle(
        feature_names=["user_id", "movie_id", "gender"],
        vocab_sizes=[4, 8, 3],
        train_loader=None,
        val_loader=None,
        test_loader=None,
        metadata={},
    )
    parent = build_default_ctr_genome(bundle, CTRGenomeConfig(embedding_dim=4, hidden_dims=[8], dropout=0.0))
    proposals = [OpenEndedProposal.from_dict(_macro_fusion_payload())]

    ingested = ingest_open_ended_proposals(
        proposals=proposals,
        parent=parent,
        output_dir=tmp_path / "run",
        memory=EvolutionMemory(tmp_path / "memory.jsonl"),
    )

    assert len(ingested) == 1
    genome = ingested[0].ingestion.genome
    assert genome.get_node("prediction").input_keys == ["macro_logits"]
    assert genome.get_node("loss").params["logits_key"] == "macro_logits"
    assert genome.get_node("llm_r00_macro_fusion").metadata["structural_scope"] == "macro"
    library = SkillLibrary.from_repo(include_generated=True)
    library.add_card(ingested[0].ingestion.skill_card_path)
    GenomeVerifier(skill_library=library).assert_valid(genome)
    SkillGenomeCompiler(skill_library=library).compile(genome)


def test_llm_provider_wrapper_fails_without_backend():
    script = Path(__file__).resolve().parents[2] / "tools" / "code_space_llm_provider.py"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("EVOSKILLREC_CODE_SPACE_")
    }
    env["EVOSKILLREC_CODE_SPACE_PROVIDER"] = "command"
    proc = subprocess.run(
        [sys.executable, str(script), "--mode", "planner"],
        input=json.dumps({"design_brief": {"available_context_keys": ["x"]}}),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert proc.returncode != 0
    assert "Missing model backend" in proc.stderr


def test_llm_provider_wrapper_validates_macro_outputs_with_command_backend(tmp_path, monkeypatch):
    backend = tmp_path / "backend.py"
    backend.write_text(
        textwrap.dedent(
            """
            import json
            import sys

            payload = json.load(sys.stdin)
            task = payload["task"]
            if task == "planner":
                print(json.dumps({
                    "sketches": [
                        {
                            "sketch_id": "macro_fusion",
                            "innovation_lane": "fusion",
                            "target_failure_mode": "fusion_bottleneck",
                            "hypothesis": "Replace additive fusion.",
                            "affected_genome_nodes": ["linear", "fm", "deep_tower"],
                            "input_keys": ["linear_logit", "fm_output", "deep_logit"],
                            "output_keys": ["macro_logits"],
                            "wiring": "replace_fusion",
                            "structural_scope": "macro",
                        }
                    ]
                }))
            else:
                code = "\\n".join([
                    "import torch",
                    "from torch import nn",
                    "",
                    "class MacroFusion(nn.Module):",
                    "    def __init__(self, input_keys=None, output_key=\\"macro_logits\\"):",
                    "        super().__init__()",
                    "        self.input_keys = list(input_keys or [\\"linear_logit\\", \\"fm_output\\", \\"deep_logit\\"])",
                    "        self.output_key = output_key",
                    "",
                    "    def forward(self, inputs):",
                    "        values = torch.cat([inputs[key] for key in self.input_keys], dim=-1)",
                    "        return {self.output_key: values.sum(dim=-1, keepdim=True)}",
                    "",
                ])
                print(json.dumps({
                    "proposals": [
                        {
                            "proposal_id": "macro_fusion",
                            "proposal_type": "NEW_FUSION_DESIGN",
                            "structural_scope": "macro",
                            "target_failure_mode": "fusion_bottleneck",
                            "architecture_hypothesis": "Replace additive fusion.",
                            "affected_genome_nodes": ["linear", "fm", "deep_tower"],
                            "code": code,
                            "expected_input_signature": [
                                {"name": "linear_logit", "shape": ["batch_size", 1], "dtype": "float32"},
                                {"name": "fm_output", "shape": ["batch_size", 1], "dtype": "float32"},
                                {"name": "deep_logit", "shape": ["batch_size", 1], "dtype": "float32"},
                            ],
                            "expected_output_signature": [
                                {"name": "macro_logits", "shape": ["batch_size", 1], "dtype": "float32"}
                            ],
                            "skill_id": "llm_r00_macro_fusion",
                            "class_name": "MacroFusion",
                            "task_types": ["ctr"],
                            "metadata": {
                                "wiring": "replace_fusion",
                                "old_logits_key": "logits",
                                "output_key": "macro_logits",
                                "macro_judgment": {
                                    "current_architecture_family": "deepfm_additive_linear_fm_mlp",
                                    "diagnosed_limitation": "fixed additive fusion cannot adapt branch evidence",
                                    "target_architecture_transformation": "replace additive fusion",
                                    "why_local_edit_is_insufficient": "local calibration cannot change fusion topology",
                                    "parent_evidence": ["fusion consumes three logits"],
                                    "proposal_insight": "multi-branch evidence should be fused before prediction",
                                    "preconditions": {"met": ["branch logits"], "missing": []},
                                    "wiring_plan": "reroute prediction/loss to macro_logits",
                                    "ablation_plan": "compare against additive fusion",
                                },
                            },
                        }
                    ]
                }))
            """
        ).lstrip(),
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[2] / "tools" / "code_space_llm_provider.py"
    backend_command = _python_command(backend)
    monkeypatch.setenv("EVOSKILLREC_CODE_SPACE_PROVIDER", "command")
    monkeypatch.setenv("EVOSKILLREC_CODE_SPACE_BACKEND", backend_command)
    design_brief = {
        "available_context_keys": ["linear_logit", "fm_output", "deep_logit", "logits"],
        "structural_scopes": ["macro"],
    }

    planner_proc = subprocess.run(
        [sys.executable, str(script), "--mode", "planner"],
        input=json.dumps({"design_brief": design_brief}),
        text=True,
        capture_output=True,
        check=False,
    )
    synthesizer_proc = subprocess.run(
        [sys.executable, str(script), "--mode", "synthesizer"],
        input=json.dumps({"design_brief": design_brief, "selected_sketches": [{"sketch_id": "macro_fusion"}]}),
        text=True,
        capture_output=True,
        check=False,
    )

    assert planner_proc.returncode == 0, planner_proc.stderr
    assert json.loads(planner_proc.stdout)["sketches"][0]["structural_scope"] == "macro"
    assert synthesizer_proc.returncode == 0, synthesizer_proc.stderr
    proposal = json.loads(synthesizer_proc.stdout)["proposals"][0]
    assert proposal["metadata"]["wiring"] == "replace_fusion"
    assert proposal["metadata"]["macro_judgment"]["why_local_edit_is_insufficient"]


def test_llm_provider_wrapper_rejects_hardcoded_ctr_field_shapes(tmp_path, monkeypatch):
    backend = tmp_path / "backend.py"
    backend.write_text(
        textwrap.dedent(
            """
            import json

            code = "\\n".join([
                "import torch",
                "from torch import nn",
                "class FieldBranch(nn.Module):",
                "    def __init__(self, input_key='field_embeddings', output_key='field_logit'):",
                "        super().__init__()",
                "        self.input_key = input_key",
                "        self.output_key = output_key",
                "    def forward(self, inputs):",
                "        return {self.output_key: inputs[self.input_key].mean(dim=(1, 2), keepdim=False).unsqueeze(-1)}",
            ])
            print(json.dumps({
                "proposals": [{
                    "proposal_id": "fixed_field",
                    "proposal_type": "NEW_BRANCH_DESIGN",
                    "structural_scope": "macro",
                    "target_failure_mode": "high_order_interaction_underfitting",
                    "architecture_hypothesis": "fixed shape field branch",
                    "affected_genome_nodes": ["field_embedding"],
                    "code": code,
                    "expected_input_signature": [
                        {"name": "field_embeddings", "shape": ["batch_size", 7, 16], "dtype": "float32"}
                    ],
                    "expected_output_signature": [
                        {"name": "field_logit", "shape": ["batch_size", 1], "dtype": "float32"}
                    ],
                    "skill_id": "fixed_field",
                    "class_name": "FieldBranch",
                    "task_types": ["ctr"],
                    "metadata": {
                        "wiring": "branch_to_fusion",
                        "output_key": "field_logit",
                        "macro_judgment": {
                            "current_architecture_family": "deepfm",
                            "diagnosed_limitation": "needs interaction branch",
                            "target_architecture_transformation": "add branch",
                            "why_local_edit_is_insufficient": "changes topology",
                            "parent_evidence": ["field_embeddings exists"],
                            "proposal_insight": "field branch",
                            "preconditions": {"met": ["field_embeddings"], "missing": []},
                            "wiring_plan": "branch_to_fusion",
                            "ablation_plan": "remove branch"
                        }
                    }
                }]
            }))
            """
        ).lstrip(),
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[2] / "tools" / "code_space_llm_provider.py"
    monkeypatch.setenv("EVOSKILLREC_CODE_SPACE_PROVIDER", "command")
    monkeypatch.setenv("EVOSKILLREC_CODE_SPACE_BACKEND", _python_command(backend))

    proc = subprocess.run(
        [sys.executable, str(script), "--mode", "synthesizer"],
        input=json.dumps({"design_brief": {"available_context_keys": ["field_embeddings"], "structural_scopes": ["macro"]}}),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "hard-codes field_embeddings shape" in proc.stderr


def test_llm_provider_wrapper_defaults_to_codex_provider_without_command_backend():
    script = Path(__file__).resolve().parents[2] / "tools" / "code_space_llm_provider.py"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("EVOSKILLREC_CODE_SPACE_")
    }
    env["EVOSKILLREC_CODE_SPACE_CODEX_COMMAND"] = "definitely_missing_codex_binary"

    proc = subprocess.run(
        [sys.executable, str(script), "--mode", "planner"],
        input=json.dumps({"design_brief": {"available_context_keys": ["x"]}}),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert proc.returncode != 0
    assert "definitely_missing_codex_binary" in proc.stderr or "No such file" in proc.stderr


def test_llm_provider_wrapper_reports_missing_http_provider_keys():
    script = Path(__file__).resolve().parents[2] / "tools" / "code_space_llm_provider.py"
    base_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("EVOSKILLREC_CODE_SPACE_")
        and key not in {"GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY"}
    }

    gemini_proc = subprocess.run(
        [sys.executable, str(script), "--mode", "planner"],
        input=json.dumps({"design_brief": {"available_context_keys": ["x"]}}),
        text=True,
        capture_output=True,
        check=False,
        env={**base_env, "EVOSKILLREC_CODE_SPACE_PROVIDER": "gemini"},
    )
    deepseek_proc = subprocess.run(
        [sys.executable, str(script), "--mode", "planner"],
        input=json.dumps({"design_brief": {"available_context_keys": ["x"]}}),
        text=True,
        capture_output=True,
        check=False,
        env={**base_env, "EVOSKILLREC_CODE_SPACE_PROVIDER": "deepseek"},
    )

    assert gemini_proc.returncode != 0
    assert "Missing Gemini API key" in gemini_proc.stderr
    assert deepseek_proc.returncode != 0
    assert "Missing DeepSeek API key" in deepseek_proc.stderr


def _python_command(script_path: Path) -> str:
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))}"
