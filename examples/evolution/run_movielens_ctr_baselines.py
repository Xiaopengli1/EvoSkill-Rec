from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torch_rechub.basic.features import SparseFeature
from torch_rechub.models.ranking import AFM, DCN, EDCN, AutoInt, DCNv2, DeepFFM, DeepFM, FatDeepFFM, FiBiNet, WideDeep
from torch_rechub.utils.data import TorchDataset


MODEL_NAMES = [
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_movielens_inputs(args: argparse.Namespace) -> tuple[list[SparseFeature], list[SparseFeature], list[SparseFeature], DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    data = pd.read_csv(args.dataset_path)
    if args.limit_rows:
        data = data.head(args.limit_rows).copy()
    if args.derived_cate_col not in data.columns and args.genre_col in data.columns:
        data[args.derived_cate_col] = data[args.genre_col].fillna("unknown").map(lambda value: str(value).split("|")[0])

    if args.label_col and args.label_col in data.columns:
        labels = data[args.label_col].astype(np.float32).to_numpy()
    else:
        labels = (data[args.rating_col].astype(float) >= args.positive_rating_threshold).astype(np.float32).to_numpy()
    if args.timestamp_col in data.columns:
        data = data.assign(__label=labels).sort_values(args.timestamp_col)
        labels = data.pop("__label").to_numpy(dtype=np.float32)
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(data))
        data = data.iloc[order].reset_index(drop=True)
        labels = labels[order]

    base_feature_names = [name for name in args.categorical_cols.split(",") if name in data.columns]
    train_idx, val_idx, test_idx = split_indices(len(labels), args.split_ratio)
    feature_names = list(base_feature_names)
    if args.add_count_bucket_features:
        feature_names.extend(add_train_count_bucket_features(data, base_feature_names, train_idx))

    x: dict[str, np.ndarray] = {}
    vocab_sizes: dict[str, int] = {}
    for name in feature_names:
        values = data[name].fillna("__missing__").astype(str)
        if args.category_encoding == "train_oov":
            ids, vocab_size = encode_train_oov(values, train_idx, args.min_category_frequency)
        else:
            encoder = LabelEncoder()
            ids = encoder.fit_transform(values) + 1
            vocab_size = int(ids.max()) + 1
        x[name] = ids.astype(np.int64)
        vocab_sizes[name] = vocab_size

    train_loader = build_loader(x, labels, train_idx, args.batch_size, shuffle=True, seed=args.seed, num_workers=args.num_workers)
    val_loader = build_loader(x, labels, val_idx, args.batch_size, shuffle=False, seed=args.seed, num_workers=args.num_workers)
    test_loader = build_loader(x, labels, test_idx, args.batch_size, shuffle=False, seed=args.seed, num_workers=args.num_workers)

    sparse_features = [SparseFeature(name, vocab_size=vocab_sizes[name], embed_dim=args.embedding_dim) for name in feature_names]
    ffm_linear_features = [SparseFeature(name, vocab_size=vocab_sizes[name], embed_dim=1) for name in feature_names]
    num_fields = len(feature_names)
    ffm_cross_features = [SparseFeature(name, vocab_size=vocab_sizes[name] * num_fields, embed_dim=args.ffm_embedding_dim) for name in feature_names]
    metadata = {
        "dataset_path": args.dataset_path,
        "num_rows": int(len(labels)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "positive_rate": float(labels.mean()),
        "train_positive_rate": float(labels[train_idx].mean()) if len(train_idx) else None,
        "category_encoding": args.category_encoding,
        "min_category_frequency": args.min_category_frequency,
        "add_count_bucket_features": args.add_count_bucket_features,
        "feature_names": feature_names,
        "base_feature_names": base_feature_names,
        "vocab_sizes": vocab_sizes,
        "split_ratio": args.split_ratio,
    }
    return sparse_features, ffm_linear_features, ffm_cross_features, train_loader, val_loader, test_loader, metadata


def split_indices(num_rows: int, split_ratio: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(num_rows * split_ratio[0])
    val_end = train_end + int(num_rows * split_ratio[1])
    indices = np.arange(num_rows)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def add_train_count_bucket_features(data: pd.DataFrame, feature_names: list[str], train_idx: np.ndarray) -> list[str]:
    added: list[str] = []
    train_data = data.iloc[train_idx]
    for name in feature_names:
        count_col = f"{name}_train_count_bin"
        train_counts = train_data[name].fillna("__missing__").astype(str).value_counts()
        counts = data[name].fillna("__missing__").astype(str).map(train_counts).fillna(0).astype(np.int64)
        data[count_col] = bucketize_counts(counts)
        added.append(count_col)
    return added


def bucketize_counts(counts: pd.Series) -> np.ndarray:
    values = counts.to_numpy(dtype=np.int64)
    # Log buckets keep the feature compact while separating cold, rare, and popular IDs.
    return np.floor(np.log2(values + 1)).astype(np.int64)


def encode_train_oov(values: pd.Series, train_idx: np.ndarray, min_frequency: int) -> tuple[np.ndarray, int]:
    min_frequency = max(1, int(min_frequency))
    train_values = values.iloc[train_idx]
    counts = train_values.value_counts()
    known_values = sorted(counts[counts >= min_frequency].index.tolist())
    mapping = {value: idx + 1 for idx, value in enumerate(known_values)}
    ids = values.map(mapping).fillna(0).to_numpy(dtype=np.int64)
    return ids, len(mapping) + 1


def build_loader(x: dict[str, np.ndarray], y: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    subset_x = {key: values[indices] for key, values in x.items()}
    subset_y = y[indices]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(TorchDataset(subset_x, subset_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator)


def build_model(model_name: str, sparse_features: list[SparseFeature], ffm_linear_features: list[SparseFeature], ffm_cross_features: list[SparseFeature], args: argparse.Namespace) -> torch.nn.Module:
    mlp_params = {"dims": args.hidden_dims, "dropout": args.dropout, "activation": args.activation}
    if model_name == "widedeep":
        return WideDeep(wide_features=sparse_features, deep_features=sparse_features, mlp_params=dict(mlp_params))
    if model_name == "deepfm":
        return DeepFM(deep_features=sparse_features, fm_features=sparse_features, mlp_params=dict(mlp_params))
    if model_name == "dcn":
        return DCN(features=sparse_features, n_cross_layers=args.cross_layers, mlp_params=dict(mlp_params))
    if model_name == "dcn_v2":
        return DCNv2(features=sparse_features, n_cross_layers=args.cross_layers, mlp_params=dict(mlp_params), low_rank=args.low_rank, num_experts=args.num_experts)
    if model_name == "edcn":
        return EDCN(features=sparse_features, n_cross_layers=args.cross_layers, mlp_params=dict(mlp_params))
    if model_name == "afm":
        return AFM(fm_features=sparse_features, embed_dim=args.embedding_dim, t=args.afm_attention_dim)
    if model_name == "autoint":
        return AutoInt(sparse_features=sparse_features, dense_features=[], num_layers=args.autoint_layers, num_heads=args.autoint_heads, dropout=args.dropout, mlp_params=dict(mlp_params))
    if model_name == "fibinet":
        return FiBiNet(features=sparse_features, reduction_ratio=args.reduction_ratio, mlp_params=dict(mlp_params))
    if model_name == "deepffm":
        return DeepFFM(linear_features=ffm_linear_features, cross_features=ffm_cross_features, embed_dim=args.ffm_embedding_dim, mlp_params=dict(mlp_params))
    if model_name == "fat_deepffm":
        return FatDeepFFM(linear_features=ffm_linear_features, cross_features=ffm_cross_features, embed_dim=args.ffm_embedding_dim, reduction_ratio=args.reduction_ratio, mlp_params=dict(mlp_params))
    raise ValueError(f"Unsupported model: {model_name}")


def train_and_evaluate(model_name: str, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCELoss(reduction="none" if args.loss_balance == "inverse_frequency" else "mean")
    best_auc = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    training_log: list[dict[str, Any]] = []

    for epoch in range(args.epoch):
        model.train()
        losses = []
        for x_dict, y in train_loader:
            x_dict = {key: value.to(device) for key, value in x_dict.items()}
            y = y.to(device).float()
            optimizer.zero_grad()
            pred = model(x_dict).view_as(y)
            loss = criterion(pred, y)
            if args.loss_balance == "inverse_frequency":
                loss = apply_inverse_frequency_weights(loss, y, args.train_positive_rate)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, **{f"val_{key}": value for key, value in val_metrics.items()}}
        training_log.append(row)
        print(f"{model_name}\tepoch={epoch}\ttrain_loss={row['train_loss']:.6f}\tval_auc={val_metrics['auc']:.6f}\tval_logloss={val_metrics['logloss']:.6f}", flush=True)
        if not math.isnan(val_metrics["auc"]) and val_metrics["auc"] > best_auc + args.min_delta:
            best_auc = float(val_metrics["auc"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.earlystop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    test_metrics = evaluate(model, test_loader, device)
    return {
        "model": model_name,
        "source": "torch_rechub",
        "status": "ok",
        "validation_best_auc": best_auc,
        "test_auc": test_metrics["auc"],
        "test_logloss": test_metrics["logloss"],
        "test_num_examples": test_metrics["num_examples"],
        "elapsed_sec": round(time.time() - started, 4),
        "training_log": training_log,
    }


def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    labels: list[float] = []
    predictions: list[float] = []
    with torch.no_grad():
        for x_dict, y in data_loader:
            x_dict = {key: value.to(device) for key, value in x_dict.items()}
            pred = model(x_dict).detach().cpu().view(-1).numpy()
            labels.extend(y.detach().cpu().view(-1).numpy().astype(float).tolist())
            predictions.extend(pred.astype(float).tolist())
    target = np.asarray(labels, dtype=np.float64)
    pred = np.clip(np.asarray(predictions, dtype=np.float64), 1e-7, 1 - 1e-7)
    return {
        "auc": safe_auc(target, pred),
        "logloss": float(log_loss(target, pred, labels=[0, 1])),
        "num_examples": int(len(target)),
    }


def apply_inverse_frequency_weights(loss: torch.Tensor, labels: torch.Tensor, positive_rate: float | None) -> torch.Tensor:
    if positive_rate is None or positive_rate <= 0.0 or positive_rate >= 1.0:
        return loss.mean()
    pos_weight = 0.5 / positive_rate
    neg_weight = 0.5 / (1.0 - positive_rate)
    weights = torch.where(labels > 0.5, torch.as_tensor(pos_weight, device=labels.device), torch.as_tensor(neg_weight, device=labels.device))
    return (loss * weights).mean()


def safe_auc(target: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(target)) < 2:
        return float("nan")
    return float(roc_auc_score(target, pred))


def load_reference_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    rows = []
    aliases = {
        "baseline": "evoskillrec_genome_baseline",
        "round0_crossnet_mix": "evoskillrec_crossnet_mix",
    }
    for item in summary.get("results", []):
        candidate_id = item.get("candidate_id")
        if candidate_id not in aliases:
            continue
        metrics = item.get("metrics") or {}
        rows.append(
            {
                "model": aliases[candidate_id],
                "source": "evoskillrec_reference",
                "status": item.get("status", "reference"),
                "validation_best_auc": metrics.get("validation_best_auc"),
                "test_auc": metrics.get("auc"),
                "test_logloss": metrics.get("logloss"),
                "test_num_examples": metrics.get("num_examples"),
                "elapsed_sec": item.get("elapsed_sec"),
                "training_log": [],
                "artifact_dir": item.get("artifact_dir"),
            }
        )
    return rows


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_rows = [{key: value for key, value in row.items() if key != "training_log"} for row in rows]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": serializable_rows}, f, indent=2, sort_keys=True)
        f.write("\n")
    fieldnames = ["model", "source", "status", "validation_best_auc", "test_auc", "test_logloss", "test_num_examples", "elapsed_sec", "error", "artifact_dir"]
    with (output_dir / "results.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(serializable_rows)
    with (output_dir / "training_logs.json").open("w", encoding="utf-8") as f:
        json.dump({row["model"]: row.get("training_log", []) for row in rows if row.get("training_log")}, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=str(ROOT / "examples/matching/data/ml-1m/ml-1m.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/baselines/movielens_ctr"))
    parser.add_argument("--reference-summary", default=None, help="Optional evolution summary JSON for comparison.")
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--categorical-cols", default="user_id,movie_id,gender,age,occupation,zip,cate_id")
    parser.add_argument("--genre-col", default="genres")
    parser.add_argument("--derived-cate-col", default="cate_id")
    parser.add_argument("--label-col")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--positive-rating-threshold", type=float, default=4.0)
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--earlystop-patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--ffm-embedding-dim", type=int, default=10)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--cross-layers", type=int, default=3)
    parser.add_argument("--low-rank", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--afm-attention-dim", type=int, default=64)
    parser.add_argument("--autoint-layers", type=int, default=3)
    parser.add_argument("--autoint-heads", type=int, default=2)
    parser.add_argument("--reduction-ratio", type=int, default=3)
    parser.add_argument("--loss-balance", choices=["none", "inverse_frequency"], default="none")
    parser.add_argument("--category-encoding", choices=["global", "train_oov"], default="global")
    parser.add_argument("--min-category-frequency", type=int, default=1)
    parser.add_argument("--add-count-bucket-features", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    sparse_features, ffm_linear_features, ffm_cross_features, train_loader, val_loader, test_loader, metadata = prepare_movielens_inputs(args)
    args.train_positive_rate = metadata.get("train_positive_rate")
    rows = load_reference_rows(Path(args.reference_summary)) if args.reference_summary else []
    write_outputs(output_dir, rows, metadata)

    for model_name in [name.strip() for name in args.models.split(",") if name.strip()]:
        print(f"start\t{model_name}", flush=True)
        try:
            set_seed(args.seed)
            model = build_model(model_name, sparse_features, ffm_linear_features, ffm_cross_features, args)
            row = train_and_evaluate(model_name, model, train_loader, val_loader, test_loader, args)
        except Exception as exc:
            row = {
                "model": model_name,
                "source": "torch_rechub",
                "status": "failed",
                "validation_best_auc": None,
                "test_auc": None,
                "test_logloss": None,
                "test_num_examples": None,
                "elapsed_sec": None,
                "error": f"{exc.__class__.__name__}: {exc}",
                "training_log": [],
            }
            print(f"failed\t{model_name}\t{row['error']}", flush=True)
        rows.append(row)
        write_outputs(output_dir, rows, metadata)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
