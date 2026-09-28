import torch

from recskill.evolution import EvolutionMemory, OpenEndedCodingBranch, OpenEndedProposal, SkillGenome, SkillGenomeCompiler

from .helpers import SAFE_GENERATED_CODE


def test_genome_compiler_executes_generated_skill(tmp_path):
    proposal = OpenEndedProposal.from_dict(
        {
            "proposal_id": "compile_scale",
            "proposal_type": "NEW_SKILL_INVENTION",
            "target_failure_mode": "calibration_gap",
            "architecture_hypothesis": "Scale dense features.",
            "affected_genome_nodes": [],
            "code": SAFE_GENERATED_CODE,
            "expected_input_signature": [{"name": "x", "shape": ["batch_size", 4], "dtype": "float32"}],
            "expected_output_signature": [{"name": "scaled_x", "shape": ["batch_size", 4], "dtype": "float32"}],
            "expected_metric_improvement": None,
            "expected_risks": [],
            "ablation_plan": "",
            "fallback_plan": "",
            "author": "human",
            "skill_id": "compile_scale_skill",
            "class_name": "ResidualScale",
            "task_types": ["ctr"],
            "init_params": {"scale": 2.0},
        }
    )
    branch = OpenEndedCodingBranch(generated_root=tmp_path / "generated_skills", memory=EvolutionMemory(tmp_path / "memory.jsonl"))
    result = branch.ingest(proposal, SkillGenome())
    assert result.success, result.to_dict()

    model = SkillGenomeCompiler(skill_library=branch.skill_library).compile(result.genome)
    x = torch.ones(3, 4)
    output = model({"x": x})

    assert torch.allclose(output["scaled_x"], x * 2.0)
