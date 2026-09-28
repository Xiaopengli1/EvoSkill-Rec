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


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "examples/evolution/run_movielens_ctr_baselines.py"
DEFAULT_MODELS = [
    "widedeep",
    "deepfm",
    "dcn",
    "dcn_v2",
    "edcn",
    "afm",
    "autoint",
    "fibinet",
    "deepffm",
    "fat_deepffm",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/baselines/movielens_ctr_parallel"))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--poll-sec", type=float, default=5.0)
    return parser.parse_known_args()


def launch_model(model: str, gpu: str, output_dir: Path, extra_args: list[str]) -> tuple[subprocess.Popen, Any]:
    model_dir = output_dir / model
    log_dir = output_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"{model}.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--models",
        model,
        "--output-dir",
        str(model_dir),
        "--device",
        "cuda:0",
        *extra_args,
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"launched\tmodel={model}\tgpu={gpu}\tpid={proc.pid}\tlog={log_dir / (model + '.log')}", flush=True)
    return proc, log_file


def load_model_row(model_dir: Path, model: str) -> dict[str, Any]:
    summary_path = model_dir / "summary.json"
    if not summary_path.exists():
        return {"model": model, "source": "torch_rechub", "status": "failed", "error": "missing summary.json"}
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    for row in summary.get("results", []):
        if row.get("model") == model:
            return row
    return {"model": model, "source": "torch_rechub", "status": "failed", "error": "missing model row"}


def load_reference_rows(output_dir: Path) -> list[dict[str, Any]]:
    for summary_path in sorted(output_dir.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        rows = [row for row in summary.get("results", []) if row.get("source") == "evoskillrec_reference"]
        if rows:
            return rows
    return []


def write_aggregate(output_dir: Path, models: list[str], statuses: dict[str, int]) -> list[dict[str, Any]]:
    rows = load_reference_rows(output_dir)
    rows.extend(load_model_row(output_dir / model, model) for model in models)
    for row in rows:
        model = row.get("model")
        if model in statuses and statuses[model] != 0 and row.get("status") == "ok":
            row["status"] = "failed"
            row["error"] = f"process exited {statuses[model]}"

    fieldnames = ["model", "source", "status", "validation_best_auc", "test_auc", "test_logloss", "test_num_examples", "elapsed_sec", "error", "artifact_dir"]
    with (output_dir / "results.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"results": rows, "statuses": statuses}, f, indent=2, sort_keys=True)
        f.write("\n")
    return rows


def main() -> int:
    args, extra_args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required")

    pending = list(models)
    running: dict[str, tuple[subprocess.Popen, Any, str]] = {}
    statuses: dict[str, int] = {}
    free_gpus = list(gpus)
    started = time.time()

    while pending or running:
        while pending and free_gpus:
            model = pending.pop(0)
            gpu = free_gpus.pop(0)
            proc, log_file = launch_model(model, gpu, output_dir, extra_args)
            running[model] = (proc, log_file, gpu)

        time.sleep(args.poll_sec)
        for model, (proc, log_file, gpu) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log_file.close()
            statuses[model] = int(code)
            free_gpus.append(gpu)
            del running[model]
            print(f"finished\tmodel={model}\tgpu={gpu}\texit={code}", flush=True)
            write_aggregate(output_dir, models, statuses)

    rows = write_aggregate(output_dir, models, statuses)
    print(f"aggregate\telapsed_sec={time.time() - started:.2f}\trows={len(rows)}\toutput={output_dir}", flush=True)
    return 0 if all(code == 0 for code in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
