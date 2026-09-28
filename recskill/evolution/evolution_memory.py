from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .genome import new_id, utc_now_iso


@dataclass
class EvolutionRecord:
    record_id: str = field(default_factory=lambda: new_id("record"))
    timestamp: str = field(default_factory=utc_now_iso)
    parent_genome_id: str | None = None
    child_genome_id: str | None = None
    mutation_type: str | None = None
    proposal_id: str | None = None
    status: str = "attempted"
    validation_results: dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)
    failure_mode: str | None = None
    rationale: str = ""
    artifact_paths: dict[str, str] = field(default_factory=dict)
    task_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionRecord":
        return cls(
            record_id=data.get("record_id") or new_id("record"),
            timestamp=data.get("timestamp", utc_now_iso()),
            parent_genome_id=data.get("parent_genome_id"),
            child_genome_id=data.get("child_genome_id"),
            mutation_type=data.get("mutation_type"),
            proposal_id=data.get("proposal_id"),
            status=data.get("status", "attempted"),
            validation_results=dict(data.get("validation_results") or {}),
            evaluation_metrics=dict(data.get("evaluation_metrics") or {}),
            failure_mode=data.get("failure_mode"),
            rationale=data.get("rationale", ""),
            artifact_paths=dict(data.get("artifact_paths") or {}),
            task_type=data.get("task_type"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvolutionMemory:
    """Append-only JSONL memory for evolution attempts and outcomes."""

    def __init__(self, path: str | Path | None = None, *, persist: bool = True) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.path = Path(path) if path is not None else repo_root / "recskill" / "evolution_memory.jsonl"
        self.persist = persist
        self._records: list[EvolutionRecord] = []

    def append(self, record: EvolutionRecord) -> EvolutionRecord:
        if not self.persist:
            self._records.append(record)
            return record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, sort_keys=True)
            f.write("\n")
        return record

    def all_records(self) -> list[EvolutionRecord]:
        if not self.persist:
            return self._read_persisted_records() + list(self._records)
        return self._read_persisted_records()

    def _read_persisted_records(self) -> list[EvolutionRecord]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(EvolutionRecord.from_dict(json.loads(line)))
        return records

    def get_successful_mutations(self, failure_mode: str | None = None, task_type: str | None = None) -> list[EvolutionRecord]:
        return [
            record
            for record in self._filter_records(failure_mode=failure_mode, task_type=task_type)
            if record.status in {"success", "improved", "validated"}
        ]

    def get_failed_mutations(self, failure_mode: str | None = None, task_type: str | None = None) -> list[EvolutionRecord]:
        return [
            record
            for record in self._filter_records(failure_mode=failure_mode, task_type=task_type)
            if record.status in {"failed", "rejected", "rolled_back"}
        ]

    def get_generated_skills(self, task_type: str | None = None) -> list[EvolutionRecord]:
        records = self._filter_records(task_type=task_type)
        return [
            record
            for record in records
            if record.artifact_paths.get("skill_card") or record.artifact_paths.get("code_path")
        ]

    def get_lineage(self, genome_id: str) -> list[EvolutionRecord]:
        records = self.all_records()
        by_child = {record.child_genome_id: record for record in records if record.child_genome_id}
        lineage = []
        current = genome_id
        seen = set()
        while current in by_child and current not in seen:
            seen.add(current)
            record = by_child[current]
            lineage.append(record)
            if not record.parent_genome_id:
                break
            current = record.parent_genome_id
        lineage.reverse()
        return lineage

    def _filter_records(self, failure_mode: str | None = None, task_type: str | None = None) -> list[EvolutionRecord]:
        records = self.all_records()
        if failure_mode is not None:
            records = [record for record in records if record.failure_mode == failure_mode]
        if task_type is not None:
            records = [record for record in records if record.task_type == task_type or _record_mentions_task(record, task_type)]
        return records


def _record_mentions_task(record: EvolutionRecord, task_type: str) -> bool:
    blob = json.dumps(
        {
            "validation_results": record.validation_results,
            "evaluation_metrics": record.evaluation_metrics,
            "rationale": record.rationale,
        },
        sort_keys=True,
    )
    return task_type in blob
