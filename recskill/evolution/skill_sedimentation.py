from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .fingerprint import genome_architecture_fingerprint
from .genome import SkillGenome, utc_now_iso


@dataclass
class SkillPromotionResult:
    promoted: bool
    skill_id: str | None = None
    code_path: str | None = None
    skill_card_path: str | None = None
    code_hash: str | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "skill_id": self.skill_id,
            "code_path": self.code_path,
            "skill_card_path": self.skill_card_path,
            "code_hash": self.code_hash,
            "message": self.message,
            "metadata": self.metadata,
        }


def promote_generated_skill(
    *,
    skill_id: str,
    staged_code_path: str | Path,
    staged_card_path: str | Path,
    candidate_id: str,
    candidate_metrics: dict[str, Any],
    candidate_status: str,
    genome: SkillGenome,
    repo_root: str | Path | None = None,
    generated_root: str | Path | None = None,
    promotion_status: str = "promoted",
) -> SkillPromotionResult:
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    generated_root = Path(generated_root) if generated_root is not None else repo_root / "recskill" / "generated_skills"
    generated_root.mkdir(parents=True, exist_ok=True)

    staged_code_path = Path(staged_code_path)
    staged_card_path = Path(staged_card_path)
    if not staged_code_path.exists():
        return SkillPromotionResult(False, skill_id=skill_id, message=f"missing staged code: {staged_code_path}")
    if not staged_card_path.exists():
        return SkillPromotionResult(False, skill_id=skill_id, message=f"missing staged skill card: {staged_card_path}")

    with staged_card_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    source_skill_id = str(manifest.get("skill_id") or manifest.get("name") or skill_id)
    base_persisted_skill_id = _canonical_persisted_skill_id(source_skill_id or skill_id)
    source_skill_ids = _unique_list([skill_id, source_skill_id, manifest.get("name")])
    code_hash = _file_sha256(staged_code_path)
    target_code = _unique_target(generated_root / f"{base_persisted_skill_id}{staged_code_path.suffix or '.py'}", code_hash=code_hash)
    persisted_skill_id = target_code.stem
    if not target_code.exists():
        shutil.copy2(staged_code_path, target_code)

    skill_name = manifest.get("skill_name")
    manifest["skill_id"] = persisted_skill_id
    manifest["name"] = persisted_skill_id
    if not skill_name or str(skill_name) in source_skill_ids or _canonical_persisted_skill_id(str(skill_name)) == persisted_skill_id:
        manifest["skill_name"] = persisted_skill_id
    manifest["implementation_path"] = _relative_or_absolute(target_code, repo_root)
    manifest["promotion_status"] = promotion_status
    manifest["promoted_at"] = utc_now_iso()
    manifest["candidate_id"] = candidate_id
    manifest["candidate_status"] = candidate_status
    manifest["candidate_metrics"] = dict(candidate_metrics)
    manifest["architecture_fingerprint"] = genome_architecture_fingerprint(genome)
    manifest["code_hash"] = code_hash
    manifest["created_from_open_ended_evolution"] = True
    manifest.setdefault("metadata", {})
    manifest["metadata"] = {
        **dict(manifest.get("metadata") or {}),
        "origin_skill_id": source_skill_id,
        "persisted_skill_id": persisted_skill_id,
        "promotion_status": promotion_status,
        "candidate_id": candidate_id,
        "candidate_status": candidate_status,
        "candidate_metrics": dict(candidate_metrics),
        "architecture_fingerprint": genome_architecture_fingerprint(genome),
        "code_hash": code_hash,
    }
    manifest.setdefault("retrieval", {})
    manifest["retrieval"] = {
        **dict(manifest.get("retrieval") or {}),
        "aliases": _unique_list(
            list((manifest.get("retrieval") or {}).get("aliases") or [])
            + [persisted_skill_id]
            + [base_persisted_skill_id]
            + source_skill_ids
            + [promotion_status, "open_ended_evolution"]
        ),
    }
    composition = dict(manifest.get("composition") or {})
    example_fragment = dict(composition.get("example_genome_fragment") or {})
    if example_fragment:
        example_fragment["skill"] = persisted_skill_id
        composition["example_genome_fragment"] = example_fragment
        manifest["composition"] = composition

    target_card = _unique_target(generated_root / f"{persisted_skill_id}.skill.yaml", code_hash=_manifest_hash(manifest), suffix=".skill.yaml")
    with target_card.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    _touch_init(generated_root)
    index_update = _refresh_skill_catalog(repo_root=repo_root, generated_root=generated_root)
    return SkillPromotionResult(
        promoted=True,
        skill_id=persisted_skill_id,
        code_path=str(target_code),
        skill_card_path=str(target_card),
        code_hash=code_hash,
        message="generated skill promoted",
        metadata={
            "candidate_id": candidate_id,
            "candidate_status": candidate_status,
            "origin_skill_id": source_skill_id,
            "persisted_skill_id": persisted_skill_id,
            "base_persisted_skill_id": base_persisted_skill_id,
            "promotion_status": promotion_status,
            "architecture_fingerprint": genome_architecture_fingerprint(genome),
            "skill_catalog_update": index_update,
        },
    )


def should_promote_generated_skill(
    *,
    candidate_status: str,
    candidate_error: str | None,
    policy: str,
) -> bool:
    if candidate_error:
        return False
    normalized = str(policy or "survivor").lower()
    if normalized in {"all_validated", "validated", "trained"}:
        return candidate_status in {"candidate", "survivor", "baseline"}
    if normalized in {"survivor", "survivors"}:
        return candidate_status == "survivor"
    if normalized in {"disabled", "none", "off"}:
        return False
    return candidate_status == "survivor"


def _unique_target(base: Path, *, code_hash: str, suffix: str | None = None) -> Path:
    if base.exists() and _existing_hash(base) == code_hash:
        return base
    stem = base.name[: -len(suffix)] if suffix and base.name.endswith(suffix) else base.stem
    ext = suffix if suffix is not None else "".join(base.suffixes) or base.suffix
    if not ext:
        ext = base.suffix
    candidate = base
    idx = 1
    while candidate.exists():
        candidate = base.with_name(f"{stem}_{idx}{ext}")
        if candidate.exists() and _existing_hash(candidate) == code_hash:
            return candidate
        idx += 1
    return candidate


def _canonical_persisted_skill_id(skill_id: str) -> str:
    canonical = re.sub(r"^llm_r\d+_", "", str(skill_id or ""), flags=re.IGNORECASE).strip("_")
    return canonical or str(skill_id or "generated_skill")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_hash(manifest: dict[str, Any]) -> str:
    text = json.dumps(manifest, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _existing_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return _file_sha256(path)
    except Exception:
        return None


def _touch_init(path: Path) -> None:
    init_path = path / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _unique_list(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _refresh_skill_catalog(*, repo_root: Path, generated_root: Path) -> dict[str, Any]:
    try:
        from tools.build_skill_index import build_skill_catalog, build_skill_index

        index = build_skill_index(repo_root / "recskill" / "skills", repo_root=repo_root, extra_roots=[generated_root])
        (repo_root / "recskill").mkdir(parents=True, exist_ok=True)
        with (repo_root / "recskill" / "skill_index.json").open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
            f.write("\n")
        catalog = build_skill_catalog(index)
        with (repo_root / "recskill" / "skill_catalog.json").open("w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, sort_keys=True)
            f.write("\n")
        return {"updated": True, "num_skills": len(index)}
    except Exception as exc:
        return {"updated": False, "error": f"{exc.__class__.__name__}: {exc}"}
