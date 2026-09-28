from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LaunchTask:
    name: str
    config: str
    output_dir: str | None = None
    device: str = "cuda:0"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class LauncherConfig:
    conda_env: str
    gpu_ids: list[int] = field(default_factory=lambda: list(range(8)))
    max_parallel: int = 8
    python_module: str = "rec_skill_genome.evolution.ctr_workflow"
    log_dir: str = "outputs/evolution/launcher_logs"
    aggregate_results_path: str | None = None
    aggregate_summary_path: str | None = None
    aggregate_results_csv_path: str | None = None
    snapshot_interval_sec: int = 60
    write_task_logs: bool = True
    write_launcher_summary: bool = True
    write_aggregate_outputs: bool = True
    write_aggregate_results_csv: bool = False
    tasks: list[LaunchTask] = field(default_factory=list)


def load_launcher_config(path: str | Path) -> LauncherConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    tasks = [LaunchTask(**task) for task in raw.get("tasks", [])]
    return LauncherConfig(
        conda_env=raw["conda_env"],
        gpu_ids=list(raw.get("gpu_ids", list(range(8)))),
        max_parallel=int(raw.get("max_parallel", 8)),
        python_module=raw.get("python_module", "rec_skill_genome.evolution.ctr_workflow"),
        log_dir=raw.get("log_dir", "outputs/evolution/launcher_logs"),
        aggregate_results_path=raw.get("aggregate_results_path"),
        aggregate_summary_path=raw.get("aggregate_summary_path"),
        aggregate_results_csv_path=raw.get("aggregate_results_csv_path"),
        snapshot_interval_sec=int(raw.get("snapshot_interval_sec", 60)),
        write_task_logs=_as_bool(raw.get("write_task_logs", True)),
        write_launcher_summary=_as_bool(raw.get("write_launcher_summary", True)),
        write_aggregate_outputs=_as_bool(raw.get("write_aggregate_outputs", True)),
        write_aggregate_results_csv=_as_bool(raw.get("write_aggregate_results_csv", False)),
        tasks=tasks,
    )


def run_launcher(config: LauncherConfig) -> list[dict[str, Any]]:
    if not config.tasks:
        raise ValueError("No tasks configured")
    log_dir = Path(config.log_dir)
    if config.write_task_logs or config.write_launcher_summary or config.write_aggregate_outputs or config.snapshot_interval_sec > 0:
        log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(config.tasks)
    running: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    gpu_pool = list(config.gpu_ids)
    last_snapshot_at = 0.0
    while pending or running:
        while pending and gpu_pool and len(running) < config.max_parallel:
            task = pending.pop(0)
            gpu_id = gpu_pool.pop(0)
            handle = _start_task(config, task, gpu_id, log_dir)
            running.append(handle)
        for handle in list(running):
            returncode = handle["process"].poll()
            if returncode is None:
                continue
            if handle.get("log_file") is not None:
                handle["log_file"].close()
            handle["returncode"] = returncode
            handle["elapsed_sec"] = round(time.time() - handle["started_at"], 4)
            gpu_pool.append(handle["gpu_id"])
            running.remove(handle)
            finished.append({key: value for key, value in handle.items() if key not in {"process", "log_file"}})
        now = time.time()
        if config.write_aggregate_outputs and config.snapshot_interval_sec > 0 and now - last_snapshot_at >= config.snapshot_interval_sec:
            _write_aggregate_outputs(
                config,
                _snapshot_task_records(finished, running, pending),
                log_dir,
                results_filename="aggregate_snapshot.tsv",
                summary_filename="aggregate_snapshot.json",
            )
            last_snapshot_at = now
        if pending or running:
            time.sleep(2)
    if config.write_launcher_summary:
        summary_path = log_dir / "launcher_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(finished, f, indent=2, sort_keys=True)
            f.write("\n")
    if config.write_aggregate_outputs:
        _write_aggregate_outputs(config, finished, log_dir)
    if config.write_aggregate_results_csv:
        _write_aggregate_results_csv(config, finished, log_dir)
    return finished


def _start_task(config: LauncherConfig, task: LaunchTask, gpu_id: int, log_dir: Path) -> dict[str, Any]:
    log_path = log_dir / f"{task.name}.log" if config.write_task_logs else None
    log_file = log_path.open("w", encoding="utf-8") if log_path is not None else None
    cmd = [
        "conda",
        "run",
        "-n",
        config.conda_env,
        "python",
        "-m",
        config.python_module,
        "--config",
        task.config,
        "--device",
        task.device,
    ]
    if task.output_dir:
        cmd.extend(["--output-dir", task.output_dir])
    cmd.extend(task.extra_args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout = log_file if log_file is not None else subprocess.DEVNULL
    process = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT, env=env)
    return {
        "task_name": task.name,
        "gpu_id": gpu_id,
        "cmd": cmd,
        "log_path": str(log_path) if log_path is not None else "",
        "output_dir": task.output_dir or "",
        "started_at": time.time(),
        "process": process,
        "log_file": log_file,
    }


def _write_aggregate_outputs(
    config: LauncherConfig,
    task_records: list[dict[str, Any]],
    log_dir: Path,
    *,
    results_filename: str = "aggregate_results.tsv",
    summary_filename: str = "aggregate_summary.json",
) -> None:
    rows = _collect_task_result_rows(task_records)
    aggregate_results_path = Path(config.aggregate_results_path) if config.aggregate_results_path and results_filename == "aggregate_results.tsv" else log_dir / results_filename
    aggregate_summary_path = Path(config.aggregate_summary_path) if config.aggregate_summary_path and summary_filename == "aggregate_summary.json" else log_dir / summary_filename
    if rows:
        _write_tsv(aggregate_results_path, rows)
    aggregate = {
        "num_tasks": len(task_records),
        "num_successful_tasks": sum(1 for task in task_records if _returncode(task) == 0),
        "num_failed_tasks": sum(1 for task in task_records if _returncode(task) not in {None, 0}),
        "num_running_or_pending_tasks": sum(1 for task in task_records if _returncode(task) is None),
        "tasks": task_records,
        "aggregate_results_path": str(aggregate_results_path),
        "per_generation": _summarize_by_generation(rows),
        "best_by_validation_auc": _best_row(rows, "validation_best_auc"),
        "best_by_test_auc": _best_row(rows, "auc"),
    }
    aggregate_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_summary_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
        f.write("\n")


AGGREGATE_RESULTS_CSV_COLUMNS = [
    "scope",
    "stage",
    "task_name",
    "seed",
    "round_idx",
    "n_runs",
    "successful_runs",
    "failed_runs",
    "best_task_name",
    "best_seed",
    "best_output_dir",
    "round_best_candidate_id",
    "round_best_description",
    "round_best_evolution_space",
    "round_best_operation",
    "round_best_mutation_type",
    "round_best_validation_auc",
    "round_best_test_auc",
    "round_best_test_logloss",
    "round_best_architecture_fingerprint",
    "round_best_hparam_fingerprint",
    "best_candidate_id_so_far",
    "best_description_so_far",
    "best_evolution_space_so_far",
    "best_operation_so_far",
    "best_mutation_type_so_far",
    "best_validation_auc_so_far",
    "best_test_auc_so_far",
    "best_test_logloss_so_far",
    "best_architecture_fingerprint_so_far",
    "best_hparam_fingerprint_so_far",
    "baseline_validation_auc",
    "delta_vs_baseline",
    "delta_vs_previous_best",
    "mean_baseline_validation_auc",
    "std_baseline_validation_auc",
    "min_baseline_validation_auc",
    "max_baseline_validation_auc",
    "mean_round_best_validation_auc",
    "std_round_best_validation_auc",
    "min_round_best_validation_auc",
    "max_round_best_validation_auc",
    "mean_best_validation_auc_so_far",
    "std_best_validation_auc_so_far",
    "min_best_validation_auc_so_far",
    "max_best_validation_auc_so_far",
    "mean_delta_vs_baseline",
    "std_delta_vs_baseline",
    "min_delta_vs_baseline",
    "max_delta_vs_baseline",
    "mean_delta_vs_previous_best",
    "std_delta_vs_previous_best",
    "min_delta_vs_previous_best",
    "max_delta_vs_previous_best",
    "mean_best_test_auc_so_far",
    "std_best_test_auc_so_far",
    "mean_best_test_logloss_so_far",
    "std_best_test_logloss_so_far",
    "improved_vs_baseline_run_count",
    "improved_vs_baseline_run_rate",
    "improved_vs_previous_run_count",
    "improved_vs_previous_run_rate",
    "round_total_candidates",
    "round_skill_candidates",
    "round_code_candidates",
    "round_failed_count",
    "round_skipped_count",
    "round_skipped_duplicate_count",
    "round_promoted_count",
    "total_candidate_count",
    "total_skill_candidate_count",
    "total_code_candidate_count",
    "total_failed_count",
    "total_skipped_count",
    "total_skipped_duplicate_count",
    "total_promoted_count",
    "promoted_candidate_ids",
    "architecture_fingerprint_unique_count",
    "architecture_fingerprint_counts",
    "hparam_fingerprint_unique_count",
    "hparam_fingerprint_counts",
    "status_counts",
    "output_dir",
    "task_returncode",
    "elapsed_sec",
    "error",
]


def _write_aggregate_results_csv(config: LauncherConfig, task_records: list[dict[str, Any]], log_dir: Path) -> None:
    rows = _build_aggregate_results_csv_rows(task_records)
    path = _aggregate_results_csv_path(config, log_dir)
    _write_csv(path, rows, fieldnames=AGGREGATE_RESULTS_CSV_COLUMNS)


def _aggregate_results_csv_path(config: LauncherConfig, log_dir: Path) -> Path:
    if config.aggregate_results_csv_path:
        return Path(config.aggregate_results_csv_path)
    if config.aggregate_results_path and Path(config.aggregate_results_path).suffix.lower() == ".csv":
        return Path(config.aggregate_results_path)
    return log_dir / "aggregate_results.csv"


def _build_aggregate_results_csv_rows(task_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_rows: list[dict[str, Any]] = []
    for task in task_records:
        run_rows.extend(_build_task_evolution_process_rows(task))
    return run_rows + _build_cross_run_process_summary_rows(run_rows)


def _build_task_evolution_process_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = Path(str(task.get("output_dir") or ""))
    result_path = output_dir / "results.tsv"
    base = {
        "scope": "run",
        "task_name": str(task.get("task_name") or ""),
        "seed": _seed_from_task_record(task),
        "output_dir": str(output_dir) if output_dir else "",
        "task_returncode": task.get("returncode", ""),
        "elapsed_sec": task.get("elapsed_sec", ""),
    }
    if not output_dir or not result_path.exists():
        return [
            {
                **base,
                "stage": "missing_results",
                "error": f"missing results.tsv under {output_dir}",
            }
        ]
    try:
        report_rows = _read_tsv(result_path)
    except Exception as exc:
        return [
            {
                **base,
                "stage": "read_error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        ]

    rows_by_type: dict[str, list[dict[str, Any]]] = {}
    for row in report_rows:
        rows_by_type.setdefault(str(row.get("row_type") or ""), []).append(row)

    baseline = _first_row(rows_by_type.get("baseline")) or _find_result_row(report_rows, "baseline")
    candidate_rows = rows_by_type.get("candidate", [])
    candidate_rows_by_id = {str(row.get("candidate_id") or ""): row for row in candidate_rows}
    if baseline:
        candidate_rows_by_id[str(baseline.get("candidate_id") or "baseline")] = baseline
    round_best_by_round = {
        str(row.get("round_idx") or ""): row
        for row in rows_by_type.get("round_best", [])
        if _is_numeric_round_idx(row.get("round_idx"))
    }
    curve_by_round = {
        str(row.get("round_idx") or ""): row
        for row in rows_by_type.get("auc_curve", [])
        if _is_numeric_round_idx(row.get("round_idx"))
    }
    arch_summary_by_round = {
        str(row.get("round_idx") or ""): row
        for row in rows_by_type.get("architecture_fingerprint_summary", [])
        if _is_numeric_round_idx(row.get("round_idx"))
    }
    hparam_summary_by_round = {
        str(row.get("round_idx") or ""): row
        for row in rows_by_type.get("hparam_fingerprint_summary", [])
        if _is_numeric_round_idx(row.get("round_idx"))
    }
    status_overview = _first_row(rows_by_type.get("status_overview")) or {}
    baseline_auc = _first_float(
        _value(baseline, "validation_best_auc"),
        _value(baseline, "best_validation_auc_so_far"),
    )
    process_rows: list[dict[str, Any]] = []
    previous_best = baseline_auc

    baseline_process = {
        **base,
        "stage": "baseline",
        "round_best_candidate_id": "baseline",
        "round_best_description": _value(baseline, "candidate_description"),
        "round_best_evolution_space": _value(baseline, "evolution_space"),
        "round_best_operation": _value(baseline, "operation"),
        "round_best_mutation_type": _value(baseline, "mutation_type"),
        "round_best_validation_auc": baseline_auc,
        "round_best_test_auc": _first_float(_value(baseline, "test_auc"), _value(baseline, "auc")),
        "round_best_test_logloss": _first_float(_value(baseline, "test_logloss"), _value(baseline, "logloss")),
        "round_best_architecture_fingerprint": _value(baseline, "architecture_fingerprint"),
        "round_best_hparam_fingerprint": _value(baseline, "hparam_fingerprint"),
        "best_candidate_id_so_far": "baseline",
        "best_description_so_far": _value(baseline, "candidate_description"),
        "best_evolution_space_so_far": _value(baseline, "evolution_space"),
        "best_operation_so_far": _value(baseline, "operation"),
        "best_mutation_type_so_far": _value(baseline, "mutation_type"),
        "best_validation_auc_so_far": baseline_auc,
        "best_test_auc_so_far": _first_float(_value(baseline, "test_auc"), _value(baseline, "auc")),
        "best_test_logloss_so_far": _first_float(_value(baseline, "test_logloss"), _value(baseline, "logloss")),
        "best_architecture_fingerprint_so_far": _value(baseline, "architecture_fingerprint"),
        "best_hparam_fingerprint_so_far": _value(baseline, "hparam_fingerprint"),
        "baseline_validation_auc": baseline_auc,
        "delta_vs_baseline": 0.0 if baseline_auc is not None else "",
        "delta_vs_previous_best": 0.0 if baseline_auc is not None else "",
    }
    process_rows.append(baseline_process)

    round_ids = _round_ids_from_report(candidate_rows, round_best_by_round, curve_by_round)
    for round_idx in round_ids:
        round_candidates = [row for row in candidate_rows if str(row.get("round_idx") or "") == round_idx]
        round_best = round_best_by_round.get(round_idx) or _best_report_row_by_auc(round_candidates)
        curve = curve_by_round.get(round_idx) or {}
        best_candidate_id = _first_text(
            _value(curve, "best_candidate_id_so_far"),
            _value(round_best, "best_candidate_id_so_far"),
            _value(round_best, "candidate_id"),
        )
        best_row = _find_result_row(report_rows, best_candidate_id) or round_best or baseline
        round_best_auc = _first_float(
            _value(round_best, "validation_best_auc"),
            _value(round_best, "round_best_validation_auc"),
            _value(curve, "round_best_validation_auc"),
        )
        best_auc = _first_float(_value(curve, "best_validation_auc_so_far"), _value(round_best, "best_validation_auc_so_far"), round_best_auc)
        promoted_candidate_ids = [
            str(row.get("candidate_id") or "")
            for row in round_candidates
            if str(row.get("promotion_status") or "").lower() == "promoted"
        ]
        arch_summary = arch_summary_by_round.get(round_idx) or {}
        hparam_summary = hparam_summary_by_round.get(round_idx) or {}
        row = {
            **base,
            "stage": "round",
            "round_idx": round_idx,
            "round_best_candidate_id": _value(round_best, "candidate_id"),
            "round_best_description": _value(round_best, "candidate_description"),
            "round_best_evolution_space": _value(round_best, "evolution_space"),
            "round_best_operation": _value(round_best, "operation"),
            "round_best_mutation_type": _value(round_best, "mutation_type"),
            "round_best_validation_auc": round_best_auc,
            "round_best_test_auc": _first_float(_value(round_best, "test_auc"), _value(round_best, "auc")),
            "round_best_test_logloss": _first_float(_value(round_best, "test_logloss"), _value(round_best, "logloss")),
            "round_best_architecture_fingerprint": _value(round_best, "architecture_fingerprint"),
            "round_best_hparam_fingerprint": _value(round_best, "hparam_fingerprint"),
            "best_candidate_id_so_far": _value(best_row, "candidate_id") or best_candidate_id,
            "best_description_so_far": _value(best_row, "candidate_description"),
            "best_evolution_space_so_far": _value(best_row, "evolution_space"),
            "best_operation_so_far": _value(best_row, "operation"),
            "best_mutation_type_so_far": _value(best_row, "mutation_type"),
            "best_validation_auc_so_far": best_auc,
            "best_test_auc_so_far": _first_float(_value(best_row, "test_auc"), _value(best_row, "auc")),
            "best_test_logloss_so_far": _first_float(_value(best_row, "test_logloss"), _value(best_row, "logloss")),
            "best_architecture_fingerprint_so_far": _value(best_row, "architecture_fingerprint"),
            "best_hparam_fingerprint_so_far": _value(best_row, "hparam_fingerprint"),
            "baseline_validation_auc": baseline_auc,
            "delta_vs_baseline": _subtract(best_auc, baseline_auc),
            "delta_vs_previous_best": _subtract(best_auc, previous_best),
            "round_total_candidates": _first_int(_value(round_best, "round_total_candidates"), len(round_candidates)),
            "round_skill_candidates": _first_int(
                _value(round_best, "round_skill_candidates"),
                sum(1 for row in round_candidates if str(row.get("evolution_space") or "").lower() == "skill_space"),
            ),
            "round_code_candidates": _first_int(
                _value(round_best, "round_code_candidates"),
                sum(1 for row in round_candidates if str(row.get("evolution_space") or "").lower() == "code_space"),
            ),
            "round_failed_count": _first_int(_value(round_best, "round_failed_count"), sum(1 for row in round_candidates if row.get("status") == "failed")),
            "round_skipped_count": _first_int(_value(round_best, "round_skipped_count"), sum(1 for row in round_candidates if _is_skipped(row))),
            "round_skipped_duplicate_count": _first_int(
                _value(round_best, "round_skipped_duplicate_count"),
                sum(1 for row in round_candidates if _is_duplicate_skip(row)),
            ),
            "round_promoted_count": _first_int(_value(round_best, "round_promoted_count"), len(promoted_candidate_ids)),
            "promoted_candidate_ids": ",".join(promoted_candidate_ids),
            "architecture_fingerprint_unique_count": _first_int(_value(arch_summary, "architecture_fingerprint_unique_count")),
            "architecture_fingerprint_counts": _value(arch_summary, "architecture_fingerprint_counts"),
            "hparam_fingerprint_unique_count": _first_int(_value(hparam_summary, "hparam_fingerprint_unique_count")),
            "hparam_fingerprint_counts": _value(hparam_summary, "hparam_fingerprint_counts"),
            "status_counts": _json_counts(_status_counts(round_candidates)),
        }
        process_rows.append(row)
        if best_auc is not None:
            previous_best = best_auc

    final_best_id = _value(process_rows[-1], "best_candidate_id_so_far") if process_rows else "baseline"
    final_best_row = _find_result_row(report_rows, final_best_id) or baseline
    final_best_auc = _first_float(_value(process_rows[-1], "best_validation_auc_so_far"), _value(final_best_row, "validation_best_auc")) if process_rows else baseline_auc
    final_promoted_ids = _promoted_ids_from_report(candidate_rows)
    process_rows.append(
        {
            **base,
            "stage": "final",
            "best_candidate_id_so_far": _value(final_best_row, "candidate_id") or final_best_id,
            "best_description_so_far": _value(final_best_row, "candidate_description"),
            "best_evolution_space_so_far": _value(final_best_row, "evolution_space"),
            "best_operation_so_far": _value(final_best_row, "operation"),
            "best_mutation_type_so_far": _value(final_best_row, "mutation_type"),
            "best_validation_auc_so_far": final_best_auc,
            "best_test_auc_so_far": _first_float(_value(final_best_row, "test_auc"), _value(final_best_row, "auc")),
            "best_test_logloss_so_far": _first_float(_value(final_best_row, "test_logloss"), _value(final_best_row, "logloss")),
            "best_architecture_fingerprint_so_far": _value(final_best_row, "architecture_fingerprint"),
            "best_hparam_fingerprint_so_far": _value(final_best_row, "hparam_fingerprint"),
            "baseline_validation_auc": baseline_auc,
            "delta_vs_baseline": _subtract(final_best_auc, baseline_auc),
            "total_candidate_count": len(candidate_rows),
            "total_skill_candidate_count": sum(1 for row in candidate_rows if str(row.get("evolution_space") or "").lower() == "skill_space"),
            "total_code_candidate_count": sum(1 for row in candidate_rows if str(row.get("evolution_space") or "").lower() == "code_space"),
            "total_failed_count": _first_int(_value(status_overview, "total_failed_count"), sum(1 for row in candidate_rows if row.get("status") == "failed")),
            "total_skipped_count": _first_int(_value(status_overview, "total_skipped_count"), sum(1 for row in candidate_rows if _is_skipped(row))),
            "total_skipped_duplicate_count": _first_int(
                _value(status_overview, "total_skipped_duplicate_count"),
                sum(1 for row in candidate_rows if _is_duplicate_skip(row)),
            ),
            "total_promoted_count": _first_int(_value(status_overview, "total_promoted_count"), len(final_promoted_ids)),
            "promoted_candidate_ids": _value(status_overview, "promoted_candidate_ids") or ",".join(final_promoted_ids),
            "status_counts": _value(status_overview, "status_counts") or _json_counts(_status_counts(candidate_rows)),
        }
    )
    return process_rows


def _build_cross_run_process_summary_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    baseline_rows = [row for row in run_rows if row.get("scope") == "run" and row.get("stage") == "baseline"]
    if baseline_rows:
        aggregate_rows.append(_aggregate_process_group("baseline", "", baseline_rows))
    round_ids = sorted(
        {str(row.get("round_idx") or "") for row in run_rows if row.get("scope") == "run" and row.get("stage") == "round" and str(row.get("round_idx") or "") != ""},
        key=_round_sort_key,
    )
    for round_idx in round_ids:
        grouped = [row for row in run_rows if row.get("scope") == "run" and row.get("stage") == "round" and str(row.get("round_idx") or "") == round_idx]
        aggregate_rows.append(_aggregate_process_group("round", round_idx, grouped))
    final_rows = [row for row in run_rows if row.get("scope") == "run" and row.get("stage") == "final"]
    if final_rows:
        aggregate_rows.append(_aggregate_process_group("final", "", final_rows))
    return aggregate_rows


def _aggregate_process_group(stage: str, round_idx: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_row = _best_process_row(rows, "best_validation_auc_so_far") or {}
    aggregate = {
        "scope": "aggregate",
        "stage": stage,
        "task_name": "all",
        "seed": "all",
        "round_idx": round_idx,
        "n_runs": len(rows),
        "successful_runs": sum(1 for row in rows if _text(row.get("task_returncode")) == "0"),
        "failed_runs": sum(1 for row in rows if _text(row.get("task_returncode")) not in {"", "0"}),
        "best_task_name": _value(best_row, "task_name"),
        "best_seed": _value(best_row, "seed"),
        "best_output_dir": _value(best_row, "output_dir"),
        "round_best_candidate_id": _value(best_row, "round_best_candidate_id"),
        "round_best_description": _value(best_row, "round_best_description"),
        "round_best_evolution_space": _value(best_row, "round_best_evolution_space"),
        "round_best_operation": _value(best_row, "round_best_operation"),
        "round_best_mutation_type": _value(best_row, "round_best_mutation_type"),
        "round_best_validation_auc": _value(best_row, "round_best_validation_auc"),
        "round_best_test_auc": _value(best_row, "round_best_test_auc"),
        "round_best_test_logloss": _value(best_row, "round_best_test_logloss"),
        "round_best_architecture_fingerprint": _value(best_row, "round_best_architecture_fingerprint"),
        "round_best_hparam_fingerprint": _value(best_row, "round_best_hparam_fingerprint"),
        "best_candidate_id_so_far": _value(best_row, "best_candidate_id_so_far"),
        "best_description_so_far": _value(best_row, "best_description_so_far"),
        "best_evolution_space_so_far": _value(best_row, "best_evolution_space_so_far"),
        "best_operation_so_far": _value(best_row, "best_operation_so_far"),
        "best_mutation_type_so_far": _value(best_row, "best_mutation_type_so_far"),
        "best_validation_auc_so_far": _value(best_row, "best_validation_auc_so_far"),
        "best_test_auc_so_far": _value(best_row, "best_test_auc_so_far"),
        "best_test_logloss_so_far": _value(best_row, "best_test_logloss_so_far"),
        "best_architecture_fingerprint_so_far": _value(best_row, "best_architecture_fingerprint_so_far"),
        "best_hparam_fingerprint_so_far": _value(best_row, "best_hparam_fingerprint_so_far"),
        "baseline_validation_auc": _value(best_row, "baseline_validation_auc"),
        "delta_vs_baseline": _value(best_row, "delta_vs_baseline"),
        "delta_vs_previous_best": _value(best_row, "delta_vs_previous_best"),
        "round_total_candidates": _sum_int(rows, "round_total_candidates"),
        "round_skill_candidates": _sum_int(rows, "round_skill_candidates"),
        "round_code_candidates": _sum_int(rows, "round_code_candidates"),
        "round_failed_count": _sum_int(rows, "round_failed_count"),
        "round_skipped_count": _sum_int(rows, "round_skipped_count"),
        "round_skipped_duplicate_count": _sum_int(rows, "round_skipped_duplicate_count"),
        "round_promoted_count": _sum_int(rows, "round_promoted_count"),
        "total_candidate_count": _sum_int(rows, "total_candidate_count"),
        "total_skill_candidate_count": _sum_int(rows, "total_skill_candidate_count"),
        "total_code_candidate_count": _sum_int(rows, "total_code_candidate_count"),
        "total_failed_count": _sum_int(rows, "total_failed_count"),
        "total_skipped_count": _sum_int(rows, "total_skipped_count"),
        "total_skipped_duplicate_count": _sum_int(rows, "total_skipped_duplicate_count"),
        "total_promoted_count": _sum_int(rows, "total_promoted_count"),
        "promoted_candidate_ids": _join_promoted_ids(rows),
        "status_counts": _json_counts(_merge_count_columns(rows, "status_counts")),
        "output_dir": "multiple",
    }
    _add_metric_stats(aggregate, rows, "baseline_validation_auc", "baseline_validation_auc")
    _add_metric_stats(aggregate, rows, "round_best_validation_auc", "round_best_validation_auc")
    _add_metric_stats(aggregate, rows, "best_validation_auc_so_far", "best_validation_auc_so_far")
    _add_metric_stats(aggregate, rows, "delta_vs_baseline", "delta_vs_baseline")
    _add_metric_stats(aggregate, rows, "delta_vs_previous_best", "delta_vs_previous_best")
    _add_metric_stats(aggregate, rows, "best_test_auc_so_far", "best_test_auc_so_far", include_min_max=False)
    _add_metric_stats(aggregate, rows, "best_test_logloss_so_far", "best_test_logloss_so_far", include_min_max=False)
    baseline_deltas = [_as_float(row.get("delta_vs_baseline")) for row in rows]
    baseline_deltas = [value for value in baseline_deltas if value is not None]
    previous_deltas = [_as_float(row.get("delta_vs_previous_best")) for row in rows]
    previous_deltas = [value for value in previous_deltas if value is not None]
    aggregate["improved_vs_baseline_run_count"] = sum(1 for value in baseline_deltas if value > 0)
    aggregate["improved_vs_baseline_run_rate"] = _rate(aggregate["improved_vs_baseline_run_count"], len(baseline_deltas))
    aggregate["improved_vs_previous_run_count"] = sum(1 for value in previous_deltas if value > 0)
    aggregate["improved_vs_previous_run_rate"] = _rate(aggregate["improved_vs_previous_run_count"], len(previous_deltas))
    arch_counts = _merge_count_columns(rows, "architecture_fingerprint_counts")
    hparam_counts = _merge_count_columns(rows, "hparam_fingerprint_counts")
    if arch_counts:
        aggregate["architecture_fingerprint_unique_count"] = len(arch_counts)
        aggregate["architecture_fingerprint_counts"] = _json_counts(arch_counts)
    if hparam_counts:
        aggregate["hparam_fingerprint_unique_count"] = len(hparam_counts)
        aggregate["hparam_fingerprint_counts"] = _json_counts(hparam_counts)
    return aggregate


def _seed_from_task_record(task: dict[str, Any]) -> str:
    extra_seed = _seed_from_cmd(task.get("cmd") or [])
    if extra_seed:
        return extra_seed
    name = str(task.get("task_name") or "")
    marker = "seed"
    if marker in name:
        tail = name.split(marker, 1)[1]
        digits = []
        for char in tail:
            if char.isdigit():
                digits.append(char)
            elif digits:
                break
        if digits:
            return "".join(digits)
    return ""


def _seed_from_cmd(cmd: Any) -> str:
    if not isinstance(cmd, list):
        return ""
    for idx, item in enumerate(cmd):
        if str(item) == "--seed" and idx + 1 < len(cmd):
            return str(cmd[idx + 1])
    return ""


def _first_row(rows: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    return rows[0] if rows else None


def _find_result_row(rows: list[dict[str, Any]], candidate_id: str | None) -> dict[str, Any] | None:
    if not candidate_id:
        return None
    for row in rows:
        if str(row.get("candidate_id") or "") == str(candidate_id):
            return row
    return None


def _value(row: dict[str, Any] | None, key: str) -> Any:
    if not row:
        return ""
    value = row.get(key, "")
    return "" if value is None else value


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_int(*values: Any) -> int | str:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return ""


def _first_text(*values: Any) -> str:
    for value in values:
        if value not in {None, ""}:
            return str(value)
    return ""


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _round_ids_from_report(
    candidate_rows: list[dict[str, Any]],
    round_best_by_round: dict[str, dict[str, Any]],
    curve_by_round: dict[str, dict[str, Any]],
) -> list[str]:
    round_ids = set(round_best_by_round) | set(curve_by_round)
    for row in candidate_rows:
        round_idx = str(row.get("round_idx") or "")
        if _is_numeric_round_idx(round_idx):
            round_ids.add(round_idx)
    return sorted((round_idx for round_idx in round_ids if _is_numeric_round_idx(round_idx)), key=_round_sort_key)


def _is_numeric_round_idx(value: Any) -> bool:
    return str(value or "").isdigit()


def _round_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value or "")
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _best_report_row_by_auc(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [(_as_float(row.get("validation_best_auc")), row) for row in rows]
    scored = [(value, row) for value, row in scored if value is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _subtract(left: Any, right: Any) -> float | str:
    left_value = _as_float(left)
    right_value = _as_float(right)
    if left_value is None or right_value is None:
        return ""
    return left_value - right_value


def _is_skipped(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    return status.startswith("skipped")


def _is_duplicate_skip(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    dedup_status = str(row.get("dedup_status") or "").lower()
    return status == "skipped_duplicate" or dedup_status == "duplicate"


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _json_counts(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def _promoted_ids_from_report(candidate_rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(row.get("candidate_id") or "")
        for row in candidate_rows
        if str(row.get("promotion_status") or "").lower() == "promoted"
    ]


def _best_process_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    scored = [(_as_float(row.get(metric)), row) for row in rows]
    scored = [(value, row) for value, row in scored if value is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _sum_int(rows: list[dict[str, Any]], key: str) -> int | str:
    values = []
    for row in rows:
        value = _first_int(row.get(key))
        if value != "":
            values.append(int(value))
    return sum(values) if values else ""


def _join_promoted_ids(rows: list[dict[str, Any]]) -> str:
    promoted: list[str] = []
    for row in rows:
        for candidate_id in str(row.get("promoted_candidate_ids") or "").split(","):
            candidate_id = candidate_id.strip()
            if candidate_id and candidate_id not in promoted:
                promoted.append(candidate_id)
    return ",".join(promoted)


def _merge_count_columns(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if not value:
            continue
        if isinstance(value, dict):
            counts = value
        else:
            try:
                counts = json.loads(str(value))
            except json.JSONDecodeError:
                continue
        if not isinstance(counts, dict):
            continue
        for count_key, count_value in counts.items():
            try:
                count_int = int(count_value)
            except (TypeError, ValueError):
                continue
            merged[str(count_key)] = merged.get(str(count_key), 0) + count_int
    return merged


def _add_metric_stats(
    aggregate: dict[str, Any],
    rows: list[dict[str, Any]],
    source_key: str,
    output_stem: str,
    *,
    include_min_max: bool = True,
) -> None:
    values = [_as_float(row.get(source_key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return
    aggregate[f"mean_{output_stem}"] = statistics.fmean(values)
    aggregate[f"std_{output_stem}"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    if include_min_max:
        aggregate[f"min_{output_stem}"] = min(values)
        aggregate[f"max_{output_stem}"] = max(values)


def _rate(numerator: int | str, denominator: int) -> float | str:
    if denominator <= 0:
        return ""
    return int(numerator) / denominator


def _collect_task_result_rows(finished: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in finished:
        output_dir = Path(str(task.get("output_dir") or ""))
        task_name = str(task.get("task_name") or "")
        if output_dir and (output_dir / "results.tsv").exists():
            for row in _read_tsv(output_dir / "results.tsv"):
                rows.append({"task_name": task_name, "task_returncode": task.get("returncode", ""), **row})
            continue
        summary_path = output_dir / "summary.json"
        if output_dir and summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            for result in summary.get("results", []):
                rows.append(_row_from_summary_result(task_name, task, result))
            continue
        if output_dir:
            partial_rows = _collect_partial_candidate_rows(task_name, task, output_dir)
            if partial_rows:
                rows.extend(partial_rows)
                continue
        rows.append(
            {
                "task_name": task_name,
                "task_returncode": task.get("returncode", ""),
                "candidate_id": "",
                "generation": "unknown",
                "evolution_category": "unknown",
                "status": "missing_results",
                "validation_best_auc": "",
                "auc": "",
                "logloss": "",
                "artifact_dir": str(output_dir),
                "error": f"missing results.tsv/summary.json under {output_dir}",
            }
        )
    return rows


def _snapshot_task_records(
    finished: list[dict[str, Any]],
    running: list[dict[str, Any]],
    pending: list[LaunchTask],
) -> list[dict[str, Any]]:
    records = [{key: value for key, value in task.items() if key not in {"process", "log_file"}} for task in finished]
    for handle in running:
        records.append({key: value for key, value in handle.items() if key not in {"process", "log_file"}})
    for task in pending:
        records.append(
            {
                "task_name": task.name,
                "gpu_id": "",
                "cmd": [],
                "log_path": "",
                "output_dir": task.output_dir or "",
                "started_at": None,
                "returncode": None,
            }
        )
    return records


def _collect_partial_candidate_rows(task_name: str, task: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(output_dir.glob("*/metrics.json")):
        candidate_dir = metrics_path.parent
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics_payload = json.load(f)
        except Exception as exc:
            rows.append(
                {
                    "task_name": task_name,
                    "task_returncode": task.get("returncode", ""),
                    "candidate_id": candidate_dir.name,
                    "generation": _generation_from_candidate_id(candidate_dir.name),
                    "evolution_category": "unknown",
                    "status": "partial_metrics_error",
                    "validation_best_auc": "",
                    "auc": "",
                    "logloss": "",
                    "artifact_dir": str(candidate_dir),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue
        test_metrics = metrics_payload.get("test") or {}
        rows.append(
            {
                "task_name": task_name,
                "task_returncode": task.get("returncode", ""),
                "candidate_id": candidate_dir.name,
                "generation": _generation_from_candidate_id(candidate_dir.name),
                "evolution_category": _category_from_candidate_id(candidate_dir.name),
                "status": "partial",
                "validation_best_auc": metrics_payload.get("validation_best_auc", ""),
                "auc": test_metrics.get("auc", ""),
                "logloss": test_metrics.get("logloss", ""),
                "loss": test_metrics.get("loss", ""),
                "artifact_dir": str(candidate_dir),
                "error": "",
            }
        )
    return rows


def _row_from_summary_result(task_name: str, task: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or {}
    return {
        "task_name": task_name,
        "task_returncode": task.get("returncode", ""),
        "candidate_id": result.get("candidate_id", ""),
        "generation": result.get("generation", ""),
        "evolution_category": result.get("evolution_category", ""),
        "genome_id": result.get("genome_id", ""),
        "parent_genome_id": result.get("parent_genome_id", ""),
        "mutation_type": result.get("mutation_type", ""),
        "evolution_space": result.get("evolution_space", ""),
        "operation": result.get("operation", ""),
        "structure_category": result.get("structure_category", ""),
        "hparam_category": result.get("hparam_category", ""),
        "architecture_fingerprint": result.get("architecture_fingerprint", ""),
        "duplicate_of": result.get("duplicate_of", ""),
        "dedup_status": result.get("dedup_status", ""),
        "training_config_hash": result.get("training_config_hash", ""),
        "genome_config_hash": result.get("genome_config_hash", ""),
        "hparam_fingerprint": result.get("hparam_fingerprint", ""),
        "hparam_changes": json.dumps(result.get("hparam_changes") or {}, sort_keys=True),
        "generated_skill_id": result.get("generated_skill_id", ""),
        "proposal_id": result.get("proposal_id", ""),
        "promotion_status": ((result.get("promotion_result") or {}).get("metadata") or {}).get("promotion_status", ""),
        "status": result.get("status", ""),
        "validation_best_auc": metrics.get("validation_best_auc", ""),
        "auc": metrics.get("auc", ""),
        "logloss": metrics.get("logloss", ""),
        "loss": metrics.get("loss", ""),
        "elapsed_sec": result.get("elapsed_sec", ""),
        "artifact_dir": result.get("artifact_dir", ""),
        "error": result.get("error", ""),
    }


def _summarize_by_generation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("generation") or "unknown"), []).append(row)
    summaries = []
    for generation, generation_rows in sorted(grouped.items()):
        values = [_as_float(row.get("validation_best_auc")) for row in generation_rows]
        values = [value for value in values if value is not None]
        summaries.append(
            {
                "generation": generation,
                "num_rows": len(generation_rows),
                "num_survivors": sum(1 for row in generation_rows if row.get("status") == "survivor"),
                "num_code_space": sum(1 for row in generation_rows if str(row.get("evolution_space") or "").lower() == "code_space"),
                "num_skill_space": sum(1 for row in generation_rows if str(row.get("evolution_space") or "").lower() == "skill_space"),
                "validation_best_auc_mean": statistics.fmean(values) if values else None,
                "validation_best_auc_max": max(values) if values else None,
                "best_row": _best_row(generation_rows, "validation_best_auc"),
            }
        )
    return summaries


def _best_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    scored = [(_as_float(row.get(metric)), row) for row in rows]
    scored = [(value, row) for value, row in scored if value is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expanded_fieldnames = list(fieldnames)
    for row in rows:
        for key in row:
            if key not in expanded_fieldnames:
                expanded_fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=expanded_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"0", "false", "no", "off", ""}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return bool(value)


def _returncode(task: dict[str, Any]) -> int | None:
    value = task.get("returncode")
    if value is None or value == "":
        return None
    return int(value)


def _generation_from_candidate_id(candidate_id: str) -> str:
    if candidate_id == "baseline":
        return "baseline"
    if candidate_id.startswith("round"):
        digits = []
        for char in candidate_id[len("round") :]:
            if not char.isdigit():
                break
            digits.append(char)
        if digits:
            return str(int("".join(digits)) + 1)
    return "unknown"


def _category_from_candidate_id(candidate_id: str) -> str:
    if candidate_id == "baseline":
        return "baseline"
    if "_code_" in candidate_id:
        return "code_space:open_ended"
    return "skill_space:unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch multiple CTR evolution tasks with one visible GPU per task.")
    parser.add_argument("--config", required=True, help="YAML launcher config.")
    args = parser.parse_args(argv)
    finished = run_launcher(load_launcher_config(args.config))
    failed = [task for task in finished if task["returncode"] != 0]
    print(json.dumps(finished, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
