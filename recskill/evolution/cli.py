from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .evolution_memory import EvolutionMemory, EvolutionRecord
from .genome import SkillGenome
from .mutations import apply_mutation
from .open_ended import OpenEndedCodingBranch, OpenEndedProposal
from .planners import RuleBasedSkillPlanner
from .skill_library import SkillLibrary
from .verification import GenomeVerifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 skill genome evolution utilities.")
    parser.add_argument("--memory", help="Path to evolution JSONL memory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-genome", help="Validate a SkillGenome JSON/YAML file.")
    validate.add_argument("--genome", required=True)

    mutate = subparsers.add_parser("mutate-skill-space", help="Plan and apply safe skill-space mutations.")
    mutate.add_argument("--genome", required=True)
    mutate.add_argument("--diagnosis", required=True)
    mutate.add_argument("--out", required=True)
    mutate.add_argument("--budget", type=int, default=1)

    ingest = subparsers.add_parser("ingest-open-ended-proposal", help="Validate generated code and insert it as a skill.")
    ingest.add_argument("--proposal", required=True)
    ingest.add_argument("--genome", required=True)
    ingest.add_argument("--out", required=True)
    ingest.add_argument("--generated-root")

    lineage = subparsers.add_parser("show-lineage", help="Show recorded lineage for a genome id.")
    lineage.add_argument("--genome-id", required=True)

    args = parser.parse_args(argv)
    memory = EvolutionMemory(args.memory) if args.memory else EvolutionMemory()

    if args.command == "validate-genome":
        genome = SkillGenome.load(args.genome)
        result = GenomeVerifier().verify(genome)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1

    if args.command == "mutate-skill-space":
        return _mutate_skill_space(args, memory)

    if args.command == "ingest-open-ended-proposal":
        return _ingest_open_ended(args, memory)

    if args.command == "show-lineage":
        records = [record.to_dict() for record in memory.get_lineage(args.genome_id)]
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _mutate_skill_space(args: argparse.Namespace, memory: EvolutionMemory) -> int:
    genome = SkillGenome.load(args.genome)
    diagnosis = _load_structured(args.diagnosis)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    skill_library = SkillLibrary.from_repo()
    plans = RuleBasedSkillPlanner().plan(genome, skill_library, diagnosis, memory, genome.constraints, budget=args.budget)
    summaries = []
    for idx, plan in enumerate(plans):
        result = apply_mutation(genome, plan, skill_library=skill_library)
        artifact_paths = {}
        if result.success and result.genome:
            genome_path = out_dir / f"{result.genome.metadata.genome_id}.json"
            result.genome.save(genome_path)
            artifact_paths["genome"] = str(genome_path)
        memory.append(
            EvolutionRecord(
                parent_genome_id=genome.metadata.genome_id,
                child_genome_id=result.genome.metadata.genome_id if result.genome else None,
                mutation_type=plan.mutation_type,
                status="success" if result.success else "failed",
                validation_results=result.validation_results,
                failure_mode=plan.metadata.get("failure_mode"),
                rationale=plan.rationale,
                artifact_paths=artifact_paths,
                task_type=(genome.constraints.task_types[0] if genome.constraints.task_types else None),
            )
        )
        summaries.append({"plan": plan.to_dict(), "result": result.to_dict(), "artifact_paths": artifact_paths, "index": idx})
    summary_path = out_dir / "mutation_results.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"plans": len(plans), "results_path": str(summary_path), "results": summaries}, indent=2, sort_keys=True))
    return 0 if any(item["result"]["success"] for item in summaries) or not summaries else 1


def _ingest_open_ended(args: argparse.Namespace, memory: EvolutionMemory) -> int:
    genome = SkillGenome.load(args.genome)
    proposal = OpenEndedProposal.load(args.proposal)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    branch = OpenEndedCodingBranch(generated_root=args.generated_root, memory=memory)
    result = branch.ingest(proposal, genome)
    result_path = out_dir / f"{proposal.proposal_id}_ingestion_result.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result.genome:
        result.genome.save(out_dir / f"{result.genome.metadata.genome_id}.json")
    print(json.dumps({"result_path": str(result_path), **result.to_dict()}, indent=2, sort_keys=True))
    return 0 if result.success else 1


def _load_structured(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        return json.load(f)


if __name__ == "__main__":
    sys.exit(main())
