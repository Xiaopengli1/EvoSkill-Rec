from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

import recskill.skills as _registered_skills  # noqa: F401

from .skill_library import SkillLibrary


MODEL_DOMAINS = ("ranking", "matching", "multi_task", "generative")


@dataclass(frozen=True)
class ModelSkillAuditIssue:
    severity: str
    model_name: str
    issue: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def audit_model_skill_coverage(repo_root: str | Path | None = None) -> list[ModelSkillAuditIssue]:
    """Check that torch_rechub model coverage points to real reusable skills.

    The audit is intentionally static: it parses exported model names, coverage
    metadata, genome YAML files, and skill manifests without importing optional
    model dependencies such as transformers.
    """

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    coverage = _load_coverage(root)
    coverage_by_name = {model["name"]: model for model in coverage}
    exported = _exported_models(root)
    skill_ids = set(SkillLibrary.from_repo(root).list_skill_ids())
    issues: list[ModelSkillAuditIssue] = []

    for domain, names in exported.items():
        for name in names:
            if name not in coverage_by_name:
                issues.append(ModelSkillAuditIssue("error", name, "missing_model_coverage", f"domain={domain}"))

    for model in coverage:
        name = model["name"]
        status = str(model.get("status", "L0"))
        genome = model.get("genome")
        if status == "L0":
            if name in {item for names in exported.values() for item in names}:
                issues.append(ModelSkillAuditIssue("error", name, "model_not_skillified", "exported torch_rechub model is still L0"))
            continue

        if not genome:
            issues.append(ModelSkillAuditIssue("error", name, "missing_genome", "non-L0 model must declare a genome path"))
            continue
        genome_path = root / genome
        if not genome_path.exists():
            issues.append(ModelSkillAuditIssue("error", name, "missing_genome_file", str(genome_path)))
            continue

        for component in model.get("covered_components") or []:
            if component not in skill_ids:
                issues.append(ModelSkillAuditIssue("error", name, "covered_component_without_skill", str(component)))

        for skill_id in _genome_skill_ids(genome_path):
            if skill_id not in skill_ids:
                issues.append(ModelSkillAuditIssue("error", name, "genome_skill_without_implementation", skill_id))

    return issues


def assert_model_skill_coverage(repo_root: str | Path | None = None) -> None:
    issues = audit_model_skill_coverage(repo_root)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        rendered = "\n".join(f"{issue.model_name}: {issue.issue} ({issue.detail})" for issue in errors)
        raise AssertionError(rendered)


def _load_coverage(root: Path) -> list[dict[str, Any]]:
    with (root / "recskill" / "model_coverage.yaml").open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("models") or [])


def _exported_models(root: Path) -> dict[str, list[str]]:
    exported: dict[str, list[str]] = {}
    for domain in MODEL_DOMAINS:
        init_path = root / "torch_rechub" / "models" / domain / "__init__.py"
        exported[domain] = _read_dunder_all(init_path)
    return exported


def _read_dunder_all(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        return [str(item) for item in value]
    return []


def _genome_skill_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        genome = yaml.safe_load(f) or {}
    return [str(step.get("skill")) for step in genome.get("skills", []) if step.get("skill")]
