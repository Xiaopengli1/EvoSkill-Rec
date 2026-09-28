from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def build_skill_index(root: Path, repo_root: Path | None = None, extra_roots: list[Path] | None = None) -> dict:
    """Read all skill manifests and return a deterministic index by skill name."""
    repo_root = repo_root or root.parent.parent
    roots = [root] + list(extra_roots or [])
    manifest_paths = []
    for skill_root in roots:
        if skill_root.exists():
            manifest_paths.extend(skill_root.glob("**/*.skill.yaml"))
    manifest_paths = sorted(manifest_paths)
    index = {}
    for path in manifest_paths:
        with path.open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
        name = manifest.get("name")
        if not name:
            raise ValueError(f"Manifest missing name: {path}")
        if name in index:
            raise ValueError(f"Duplicate skill manifest name: {name}")
        manifest["manifest_path"] = str(path.relative_to(repo_root))
        index[name] = manifest
    return index


def build_skill_catalog(index: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a flattened retrieval catalog from full skill manifests."""
    catalog = {}
    for name, manifest in sorted(index.items()):
        retrieval = manifest.get("retrieval") or {}
        composition = manifest.get("composition") or {}
        compatible_with = manifest.get("compatible_with") or {}
        source = manifest.get("source") or {}

        task_types = _unique_list(_as_list(retrieval.get("task_types")) + _as_list(manifest.get("task_types")))
        architecture_roles = _unique_list(_as_list(retrieval.get("architecture_roles")) + [manifest.get("category")])
        input_modalities = _unique_list(_as_list(retrieval.get("input_modalities")))
        output_semantics = _unique_list(_as_list(retrieval.get("output_semantics")))
        objectives = _unique_list(_as_list(retrieval.get("objectives")))
        aliases = _unique_list(_as_list(retrieval.get("aliases")) + [name, manifest.get("category")])
        model_families = _unique_list(_as_list(retrieval.get("model_families")))
        upstream = _unique_list(_as_list(composition.get("common_upstream")) + _as_list((compatible_with.get("upstream") if compatible_with else None)))
        downstream = _unique_list(_as_list(composition.get("common_downstream")) + _as_list((compatible_with.get("downstream") if compatible_with else None)))
        requires = _unique_list(_as_list(composition.get("requires")))
        produces = _unique_list(_as_list(composition.get("produces")))

        searchable_parts = [
            name,
            manifest.get("category"),
            manifest.get("description"),
            retrieval.get("summary"),
            retrieval.get("use_when"),
            retrieval.get("avoid_when"),
            retrieval.get("query_examples"),
            task_types,
            architecture_roles,
            input_modalities,
            output_semantics,
            objectives,
            aliases,
            model_families,
            upstream,
            downstream,
            requires,
            produces,
            manifest.get("good_for"),
            manifest.get("bad_for"),
            manifest.get("inductive_bias"),
            source.get("class_or_function"),
            source.get("file"),
        ]

        catalog[name] = {
            "name": name,
            "category": manifest.get("category", "unknown"),
            "summary": retrieval.get("summary", manifest.get("description", "")),
            "description": manifest.get("description", ""),
            "task_types": task_types,
            "architecture_roles": architecture_roles,
            "input_modalities": input_modalities,
            "output_semantics": output_semantics,
            "objectives": objectives,
            "aliases": aliases,
            "model_families": model_families,
            "requires": requires,
            "produces": produces,
            "upstream": upstream,
            "downstream": downstream,
            "use_when": _as_list(retrieval.get("use_when")),
            "avoid_when": _as_list(retrieval.get("avoid_when")),
            "query_examples": _as_list(retrieval.get("query_examples")),
            "manifest_path": manifest.get("manifest_path"),
            "source": source,
            "searchable_text": _to_search_text(searchable_parts),
        }
    return catalog


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _unique_list(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _to_search_text(values: list[Any]) -> str:
    tokens = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            tokens.append(_to_search_text(list(value.values())))
        elif isinstance(value, (list, tuple, set)):
            tokens.append(_to_search_text(list(value)))
        else:
            tokens.append(str(value))
    return " ".join(tokens).lower()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    skills_root = repo_root / "recskill" / "skills"
    generated_root = repo_root / "recskill" / "generated_skills"
    output_path = repo_root / "recskill" / "skill_index.json"
    catalog_path = repo_root / "recskill" / "skill_catalog.json"
    index = build_skill_index(skills_root, repo_root=repo_root, extra_roots=[generated_root])
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    catalog = build_skill_catalog(index)
    with catalog_path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {len(index)} skills to {output_path}")
    print(f"Wrote {len(catalog)} catalog entries to {catalog_path}")


if __name__ == "__main__":
    # Example:
    #   python tools/build_skill_index.py
    main()
