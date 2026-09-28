#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "examples" / "evolution" / "multi_domain_movielens_evolution.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "baselines" / "multi_domain_movielens_seed2022"
DEFAULT_MODELS = [
    "sarnet",
    "m3oe",
    "adasparse",
    "mmoe",
    "ppnet",
    "hamur_small",
    "hamur_large",
    "shared_bottom",
    "epnet",
    "ple",
    "star",
    "adaptdhm",
    "m2m",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MovieLens multi-domain baseline templates.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Base multi-domain MovieLens workflow config.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for all baseline outputs.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Template names to run.")
    parser.add_argument("--gpu-ids", default=os.environ.get("CTR_EVOLUTION_GPU_IDS", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--device", default="cuda:0", help="Device visible inside each worker.")
    parser.add_argument("--max-parallel", type=int, default=0, help="Maximum concurrent workers; defaults to GPU count.")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        if not args.model:
            raise SystemExit("--worker requires --model")
        return _run_worker(args)
    return _run_launcher(args)


def _run_launcher(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [item.strip() for item in str(args.gpu_ids).split(",") if item.strip()]
    if not gpu_ids and str(args.device).startswith("cuda"):
        raise SystemExit("No GPU ids configured. Pass --gpu-ids or use --device cpu.")
    if not gpu_ids:
        gpu_ids = ["cpu"]
    max_parallel = args.max_parallel or len(gpu_ids)
    max_parallel = max(1, min(max_parallel, len(gpu_ids), len(args.models)))
    pending = list(dict.fromkeys(_normalize_model_name(model) for model in args.models))
    running: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    started_at = time.time()

    print(f"output_root={output_root}", flush=True)
    print(f"models={','.join(pending)}", flush=True)
    print(f"gpu_ids={','.join(gpu_ids)} max_parallel={max_parallel}", flush=True)

    while pending or running:
        while pending and len(running) < max_parallel:
            model = pending.pop(0)
            gpu_id = gpu_ids[len(running) % len(gpu_ids)]
            log_path = log_dir / f"{model}.log"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--config",
                str(Path(args.config).resolve()),
                "--output-root",
                str(output_root),
                "--model",
                model,
                "--gpu-id",
                gpu_id,
                "--device",
                args.device,
                "--seed",
                str(args.seed),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
            if gpu_id != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            running.append(
                {
                    "model": model,
                    "gpu_id": gpu_id,
                    "proc": proc,
                    "log_path": log_path,
                    "started": time.time(),
                }
            )
            print(f"started model={model} gpu={gpu_id} pid={proc.pid} log={log_path}", flush=True)

        time.sleep(5)
        still_running: list[dict[str, Any]] = []
        for item in running:
            proc: subprocess.Popen[Any] = item["proc"]
            rc = proc.poll()
            if rc is None:
                still_running.append(item)
                continue
            elapsed = round(time.time() - item["started"], 3)
            record = {
                "model": item["model"],
                "gpu_id": item["gpu_id"],
                "returncode": rc,
                "elapsed_sec": elapsed,
                "log_path": str(item["log_path"]),
            }
            completed.append(record)
            print(
                f"finished model={item['model']} rc={rc} elapsed_sec={elapsed} log={item['log_path']}",
                flush=True,
            )
        running = still_running

    aggregate = _write_aggregate(output_root, completed, elapsed_sec=round(time.time() - started_at, 3))
    print(f"wrote aggregate={aggregate}", flush=True)
    failed = [item for item in completed if item["returncode"] != 0]
    return 1 if failed else 0


def _run_worker(args: argparse.Namespace) -> int:
    if str(args.gpu_id or "") and args.gpu_id != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    sys.path.insert(0, str(REPO_ROOT))

    from recskill.evolution import CTREvolutionConfig
    from recskill.evolution.multi_domain_workflow import MultiDomainModelEvolutionRunner, load_multi_domain_workflow_config

    model = _normalize_model_name(args.model)
    output_dir = Path(args.output_root).resolve() / model
    config = load_multi_domain_workflow_config(args.config)
    config.experiment_name = f"multi_domain_movielens_baseline_{model}"
    config.output_dir = str(output_dir)
    config.genome.template = model
    config.training.device = args.device
    config.training.seed = int(args.seed)
    config.evolution = CTREvolutionConfig(enabled=False, rounds=0, candidate_budget=0)
    config.memory_path = str(output_dir / "evolution_memory.jsonl")
    config.write_metadata_files = True
    config.write_summary_json = True
    config.write_evolution_memory = True
    config.write_code_space_diagnostics = False

    print(
        f"worker model={model} gpu_id={args.gpu_id} visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"device={config.training.device} output_dir={output_dir}",
        flush=True,
    )
    summary = MultiDomainModelEvolutionRunner(config).run()
    _write_worker_result(output_dir, model, summary)
    return 0


def _write_worker_result(output_dir: Path, model: str, summary: dict[str, Any]) -> None:
    rows = []
    for result in summary.get("results") or []:
        if result.get("candidate_id") == "baseline":
            rows.append(_row_from_result(model, result))
    if not rows:
        rows.append({"model": model, "status": "missing_baseline"})
    path = output_dir / "baseline_result.tsv"
    _write_tsv(path, rows)


def _write_aggregate(output_root: Path, completed: list[dict[str, Any]], *, elapsed_sec: float) -> Path:
    rows: list[dict[str, Any]] = []
    by_model = {item["model"]: item for item in completed}
    for model in DEFAULT_MODELS:
        if model not in by_model and not (output_root / model).exists():
            continue
        row = {
            "model": model,
            "returncode": by_model.get(model, {}).get("returncode", ""),
            "worker_elapsed_sec": by_model.get(model, {}).get("elapsed_sec", ""),
            "log_path": by_model.get(model, {}).get("log_path", ""),
        }
        summary_path = output_root / model / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            baseline = next(
                (item for item in summary.get("results") or [] if item.get("candidate_id") == "baseline"),
                None,
            )
            if baseline:
                row.update(_row_from_result(model, baseline))
                row["run_elapsed_sec"] = summary.get("elapsed_sec", "")
        rows.append(row)
    for model, item in sorted(by_model.items()):
        if model in {row.get("model") for row in rows}:
            continue
        rows.append(
            {
                "model": model,
                "returncode": item.get("returncode", ""),
                "worker_elapsed_sec": item.get("elapsed_sec", ""),
                "log_path": item.get("log_path", ""),
            }
        )
    rows.sort(key=lambda row: DEFAULT_MODELS.index(row["model"]) if row.get("model") in DEFAULT_MODELS else 999)
    path = output_root / "baseline_results.tsv"
    _write_tsv(path, rows)
    metadata = {
        "elapsed_sec": elapsed_sec,
        "completed": completed,
        "aggregate_path": str(path),
    }
    (output_root / "baseline_run_summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _row_from_result(model: str, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or {}
    return {
        "model": model,
        "candidate_id": result.get("candidate_id", ""),
        "status": result.get("status", ""),
        "validation_auc": metrics.get("validation_auc", ""),
        "validation_best_auc": metrics.get("validation_best_auc", ""),
        "validation_logloss": metrics.get("validation_logloss", ""),
        "validation_loss": metrics.get("validation_loss", ""),
        "test_auc": metrics.get("auc", ""),
        "test_logloss": metrics.get("logloss", ""),
        "test_loss": metrics.get("loss", ""),
        "validation_auc__0": metrics.get("validation_auc__0", ""),
        "validation_logloss__0": metrics.get("validation_logloss__0", ""),
        "validation_num_examples__0": metrics.get("validation_num_examples__0", ""),
        "validation_auc__1": metrics.get("validation_auc__1", ""),
        "validation_logloss__1": metrics.get("validation_logloss__1", ""),
        "validation_num_examples__1": metrics.get("validation_num_examples__1", ""),
        "validation_auc__2": metrics.get("validation_auc__2", ""),
        "validation_logloss__2": metrics.get("validation_logloss__2", ""),
        "validation_num_examples__2": metrics.get("validation_num_examples__2", ""),
        "test_auc__0": metrics.get("auc__0", ""),
        "test_logloss__0": metrics.get("logloss__0", ""),
        "test_num_examples__0": metrics.get("num_examples__0", ""),
        "test_auc__1": metrics.get("auc__1", ""),
        "test_logloss__1": metrics.get("logloss__1", ""),
        "test_num_examples__1": metrics.get("num_examples__1", ""),
        "test_auc__2": metrics.get("auc__2", ""),
        "test_logloss__2": metrics.get("logloss__2", ""),
        "test_num_examples__2": metrics.get("num_examples__2", ""),
        "elapsed_sec": result.get("elapsed_sec", ""),
        "architecture_fingerprint": result.get("architecture_fingerprint", ""),
        "error": result.get("error", ""),
    }


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_model_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "sharedbottom": "shared_bottom",
        "sharebottom": "shared_bottom",
        "hamur": "hamur_small",
    }
    return aliases.get(normalized, normalized)


if __name__ == "__main__":
    raise SystemExit(main())
