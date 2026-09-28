from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from build_skill_index import build_skill_catalog, build_skill_index
except ModuleNotFoundError:
    from tools.build_skill_index import build_skill_catalog, build_skill_index


def load_or_build_catalog(repo_root: Path) -> dict[str, dict[str, Any]]:
    catalog_path = repo_root / "recskill" / "skill_catalog.json"
    if catalog_path.exists():
        with catalog_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    index = build_skill_index(repo_root / "recskill" / "skills", repo_root=repo_root)
    return build_skill_catalog(index)


def search_catalog(
    catalog: dict[str, dict[str, Any]],
    query: str = "",
    task: str | None = None,
    category: str | None = None,
    role: str | None = None,
    modality: str | None = None,
    objective: str | None = None,
    upstream: str | None = None,
    downstream: str | None = None,
    model: str | None = None,
    limit: int = 10,
) -> list[tuple[int, dict[str, Any]]]:
    scored = []
    for entry in catalog.values():
        score = score_skill(
            entry,
            query=query,
            task=task,
            category=category,
            role=role,
            modality=modality,
            objective=objective,
            upstream=upstream,
            downstream=downstream,
            model=model,
        )
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1]["name"]))
    return scored[:limit]


def score_skill(
    entry: dict[str, Any],
    query: str = "",
    task: str | None = None,
    category: str | None = None,
    role: str | None = None,
    modality: str | None = None,
    objective: str | None = None,
    upstream: str | None = None,
    downstream: str | None = None,
    model: str | None = None,
) -> int:
    score = 1
    filters = [
        (task, entry.get("task_types", [])),
        (category, [entry.get("category", "")]),
        (role, entry.get("architecture_roles", [])),
        (modality, entry.get("input_modalities", [])),
        (objective, entry.get("objectives", [])),
        (upstream, entry.get("upstream", [])),
        (downstream, entry.get("downstream", [])),
        (model, entry.get("model_families", [])),
    ]
    for value, candidates in filters:
        if value is None:
            continue
        if not _contains(value, candidates):
            return 0
        score += 10

    query_tokens = [token for token in query.lower().replace("_", " ").split() if token]
    searchable = entry.get("searchable_text", "")
    aliases = " ".join(entry.get("aliases", [])).lower()
    roles = " ".join(entry.get("architecture_roles", [])).lower()
    for token in query_tokens:
        if token in aliases:
            score += 8
        elif token in roles:
            score += 5
        elif token in searchable:
            score += 3
        else:
            score -= 1

    if query_tokens and score <= 1:
        return 0
    return score


def _contains(value: str, candidates: list[str]) -> bool:
    needle = value.lower().replace("_", " ")
    for candidate in candidates:
        haystack = str(candidate).lower().replace("_", " ")
        if needle == haystack or needle in haystack:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Search recskill skills by task, role, purpose, and composition facets.")
    parser.add_argument("query", nargs="?", default="", help="Free-text search query.")
    parser.add_argument("--task", help="Task type, such as ctr, ranking, sequence, matching, or multitask.")
    parser.add_argument("--category", help="Skill category, such as embedding, interaction, sequence, tower, head, loss, utility.")
    parser.add_argument("--role", help="Architecture role, such as explicit_cross, target_attention, or task_gate.")
    parser.add_argument("--modality", help="Input modality, such as sparse_ids, field_embeddings, or sequence_ids.")
    parser.add_argument("--objective", help="Objective or purpose, such as click_prediction or sequential_matching.")
    parser.add_argument("--upstream", help="Require compatibility with an upstream skill.")
    parser.add_argument("--downstream", help="Require compatibility with a downstream skill.")
    parser.add_argument("--model", help="Related model family, such as DCN, DIN, SASRec, or MMoE.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text lines.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    catalog = load_or_build_catalog(repo_root)
    results = search_catalog(
        catalog,
        query=args.query,
        task=args.task,
        category=args.category,
        role=args.role,
        modality=args.modality,
        objective=args.objective,
        upstream=args.upstream,
        downstream=args.downstream,
        model=args.model,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps([{"score": score, **entry} for score, entry in results], indent=2, sort_keys=True))
        return

    for score, entry in results:
        print(f"{score:03d}  {entry['name']}  [{entry['category']}]  {entry['summary']}")


if __name__ == "__main__":
    main()
