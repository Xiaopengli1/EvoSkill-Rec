from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from torch_rechub.utils.data import TorchDataset

from recskill.core.registry import SKILL_REGISTRY

from .compiler import SkillGenomeCompiler
from .code_space import (
    CodeSpaceConfig,
    build_code_space_provider,
    code_space_provider_diagnostics,
    inspect_open_ended_proposal_ingestions,
)
from .evolution_memory import EvolutionMemory, EvolutionRecord
from .fingerprint import config_fingerprint, diff_payload, genome_architecture_fingerprint
from .genome import GenomeConstraints, GenomeMutation, SkillEdge, SkillGenome, SkillNode, new_id, utc_now_iso
from .skill_sedimentation import promote_generated_skill, should_promote_generated_skill
from .skill_graph_planner import SkillGraphPlannerConfig, SkillGraphSearchPlanner
from .skill_library import SkillCard, SkillLibrary
from .verification import GenomeVerifier


@dataclass
class CTRDatasetConfig:
    name: str = "movielens_ctr"
    type: str = "movielens_ctr"
    path: str = "examples/matching/data/ml-1m/ml-1m.csv"
    categorical_cols: list[str] = field(default_factory=lambda: ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"])
    genre_col: str = "genres"
    derived_cate_col: str = "cate_id"
    label_col: str | None = None
    rating_col: str = "rating"
    positive_rating_threshold: float = 4.0
    timestamp_col: str | None = "timestamp"
    temporal_split: bool = True
    split_ratio: list[float] = field(default_factory=lambda: [0.7, 0.1, 0.2])
    limit_rows: int | None = None


@dataclass
class CTRTrainingConfig:
    epoch: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 2048
    weight_decay: float = 1e-5
    device: str = "cpu"
    seed: int = 2022
    num_workers: int = 0
    earlystop_patience: int = 3
    objective_metric: str = "auc"
    min_delta: float = 0.0
    loss_key: str = "loss"
    logits_key: str = "logits"
    prediction_key: str = "prediction"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None


@dataclass
class AdaptiveCandidateBudgetConfig:
    enabled: bool = False
    stagnation_patience: int = 2
    min_improvement: float = 0.001
    base_code_candidates: int | None = None
    code_step: int = 2
    max_code_candidates: int | None = None
    min_skill_candidates: int = 4
    reset_on_improvement: bool = True
    cooldown_code_step: int = 1

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "AdaptiveCandidateBudgetConfig":
        raw = raw or {}
        return cls(
            enabled=_as_bool(raw.get("enabled", False)),
            stagnation_patience=max(0, int(raw.get("stagnation_patience", 2))),
            min_improvement=float(raw.get("min_improvement", 0.001)),
            base_code_candidates=int(raw["base_code_candidates"]) if raw.get("base_code_candidates") is not None else None,
            code_step=max(0, int(raw.get("code_step", 2))),
            max_code_candidates=int(raw["max_code_candidates"]) if raw.get("max_code_candidates") is not None else None,
            min_skill_candidates=max(0, int(raw.get("min_skill_candidates", 4))),
            reset_on_improvement=_as_bool(raw.get("reset_on_improvement", True)),
            cooldown_code_step=max(0, int(raw.get("cooldown_code_step", 1))),
        )


@dataclass
class CTREvolutionConfig:
    enabled: bool = True
    rounds: int = 3
    candidate_budget: int = 10
    strategy: str = "skill_library"
    code_space_probability: float = 0.2
    failure_modes: list[str] = field(default_factory=lambda: ["high_order_interaction_underfitting"])
    templates: list[dict[str, Any]] = field(default_factory=lambda: _default_evolution_templates())
    keep_top_k: int = 4
    survivor_top_k: int = 4
    deduplicate_architectures: bool = True
    deduplicate_against_memory: bool = True
    code_space: CodeSpaceConfig = field(default_factory=CodeSpaceConfig)
    skill_graph_planner: SkillGraphPlannerConfig = field(default_factory=SkillGraphPlannerConfig)
    adaptive_candidate_budget: AdaptiveCandidateBudgetConfig = field(default_factory=AdaptiveCandidateBudgetConfig)

    def __post_init__(self) -> None:
        if isinstance(self.code_space, dict):
            _, target_code = _candidate_targets(self)
            self.code_space = CodeSpaceConfig.from_raw(self.code_space, fallback_budget=target_code)
        if isinstance(self.skill_graph_planner, dict):
            self.skill_graph_planner = SkillGraphPlannerConfig.from_raw(self.skill_graph_planner)
        if isinstance(self.adaptive_candidate_budget, dict):
            self.adaptive_candidate_budget = AdaptiveCandidateBudgetConfig.from_raw(self.adaptive_candidate_budget)


@dataclass
class CTRGenomeConfig:
    genome_path: str | None = None
    template: str = "deepfm"
    embedding_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    activation: str = "relu"


@dataclass
class CTRWorkflowConfig:
    experiment_name: str = "ctr_evolution_movielens"
    output_dir: str = "outputs/evolution/ctr_movielens"
    dataset: CTRDatasetConfig = field(default_factory=CTRDatasetConfig)
    genome: CTRGenomeConfig = field(default_factory=CTRGenomeConfig)
    training: CTRTrainingConfig = field(default_factory=CTRTrainingConfig)
    evolution: CTREvolutionConfig = field(default_factory=CTREvolutionConfig)
    memory_path: str | None = None
    write_metadata_files: bool = True
    write_summary_json: bool = True
    write_evolution_memory: bool = True
    write_code_space_diagnostics: bool = True
    write_survivor_artifacts: bool = False
    survivor_artifacts_dir: str = "survivors"


@dataclass
class CTRDatasetBundle:
    feature_names: list[str]
    vocab_sizes: list[int]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    metadata: dict[str, Any]


@dataclass
class CandidateResult:
    candidate_id: str
    genome_id: str
    parent_genome_id: str | None
    mutation_type: str
    status: str
    metrics: dict[str, Any]
    artifact_dir: str
    genome_path: str
    elapsed_sec: float
    evolution_space: str = "skill_space"
    operation: str = ""
    rationale: str = ""
    architecture_fingerprint: str = ""
    duplicate_of: str | None = None
    dedup_status: str = "unique"
    structure_category: str = ""
    hparam_category: str = "fixed_training_config"
    training_config_hash: str = ""
    genome_config_hash: str = ""
    hparam_fingerprint: str = ""
    hparam_changes: dict[str, Any] = field(default_factory=dict)
    generated_skill_id: str | None = None
    staged_code_path: str | None = None
    staged_skill_card_path: str | None = None
    proposal_id: str | None = None
    structural_scope: str = ""
    promotion_result: dict[str, Any] | None = None
    error: str | None = None
    genome: SkillGenome | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        payload = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "genome"}
        payload["generation"] = _candidate_generation_label(self)
        payload["evolution_category"] = _candidate_evolution_category(self)
        payload["structure_category"] = self.structure_category or payload["evolution_category"]
        return payload


@dataclass
class CandidateSpec:
    candidate_id: str
    genome: SkillGenome
    parent_genome_id: str | None
    mutation_type: str
    rationale: str
    evolution_space: str = "skill_space"
    operation: str = ""
    architecture_fingerprint: str = ""
    duplicate_of: str | None = None
    dedup_status: str = "unique"
    generated_skill_id: str | None = None
    staged_code_path: str | None = None
    staged_skill_card_path: str | None = None
    proposal_id: str | None = None
    structural_scope: str = ""


def load_workflow_config(path: str | Path) -> CTRWorkflowConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return CTRWorkflowConfig(
        experiment_name=raw.get("experiment_name", "ctr_evolution_movielens"),
        output_dir=raw.get("output_dir", "outputs/evolution/ctr_movielens"),
        dataset=CTRDatasetConfig(**(raw.get("dataset") or {})),
        genome=CTRGenomeConfig(**(raw.get("genome") or {})),
        training=CTRTrainingConfig(**(raw.get("training") or {})),
        evolution=CTREvolutionConfig(**(raw.get("evolution") or {})),
        memory_path=raw.get("memory_path"),
        write_metadata_files=_as_bool(raw.get("write_metadata_files", True)),
        write_summary_json=_as_bool(raw.get("write_summary_json", True)),
        write_evolution_memory=_as_bool(raw.get("write_evolution_memory", True)),
        write_code_space_diagnostics=_as_bool(raw.get("write_code_space_diagnostics", True)),
        write_survivor_artifacts=_as_bool(raw.get("write_survivor_artifacts", False)),
        survivor_artifacts_dir=str(raw.get("survivor_artifacts_dir", "survivors")),
    )


def prepare_movielens_ctr_data(config: CTRDatasetConfig, training: CTRTrainingConfig) -> CTRDatasetBundle:
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path)
    if config.limit_rows is not None:
        data = data.head(config.limit_rows).copy()
    if config.derived_cate_col not in data.columns and config.genre_col in data.columns:
        data[config.derived_cate_col] = data[config.genre_col].fillna("unknown").map(lambda value: str(value).split("|")[0])

    categorical_cols = [col for col in config.categorical_cols if col in data.columns]
    if not categorical_cols:
        raise ValueError("No categorical columns are available for sparse_features")

    labels = _build_ctr_labels(data, config)
    if config.timestamp_col and config.timestamp_col in data.columns and config.temporal_split:
        data = data.assign(__label=labels).sort_values(config.timestamp_col)
        labels = data.pop("__label").to_numpy(dtype=np.float32)
    else:
        data = data.copy()
        labels = labels.astype(np.float32)
        rng = np.random.default_rng(training.seed)
        order = rng.permutation(len(data))
        data = data.iloc[order].reset_index(drop=True)
        labels = labels[order]

    encoded = []
    vocab_sizes = []
    encoders: dict[str, dict[str, int]] = {}
    for col in categorical_cols:
        lbe = LabelEncoder()
        values = data[col].fillna("__missing__").astype(str)
        ids = lbe.fit_transform(values) + 1
        encoded.append(ids.astype(np.int64))
        vocab_sizes.append(int(ids.max()) + 1)
        encoders[col] = {str(raw): int(idx + 1) for idx, raw in enumerate(lbe.classes_)}
    sparse_features = np.stack(encoded, axis=1)
    x = {"sparse_features": sparse_features}

    train_idx, val_idx, test_idx = _split_indices(len(labels), config.split_ratio)
    train_loader = _build_loader(x, labels, train_idx, training.batch_size, shuffle=True, seed=training.seed, num_workers=training.num_workers)
    val_loader = _build_loader(x, labels, val_idx, training.batch_size, shuffle=False, seed=training.seed, num_workers=training.num_workers)
    test_loader = _build_loader(x, labels, test_idx, training.batch_size, shuffle=False, seed=training.seed, num_workers=training.num_workers)
    metadata = {
        "dataset_path": str(path),
        "num_rows": int(len(labels)),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "positive_rate": float(labels.mean()) if len(labels) else None,
        "feature_names": categorical_cols,
        "vocab_sizes": vocab_sizes,
        "encoders": encoders,
    }
    return CTRDatasetBundle(categorical_cols, vocab_sizes, train_loader, val_loader, test_loader, metadata)


def build_default_ctr_genome(bundle: CTRDatasetBundle, config: CTRGenomeConfig) -> SkillGenome:
    if config.template != "deepfm":
        raise ValueError(f"Unsupported built-in genome template: {config.template}")
    num_fields = len(bundle.vocab_sizes)
    flat_input_dim = num_fields * config.embedding_dim
    hidden_dims = list(config.hidden_dims)
    genome = SkillGenome(
        nodes=[
            SkillNode(
                node_id="field_embedding",
                skill_id="field_embedding",
                skill_name="field_embedding",
                category="embedding",
                params={
                    "vocab_sizes": bundle.vocab_sizes,
                    "embedding_dim": config.embedding_dim,
                    "feature_names": bundle.feature_names,
                },
                input_keys=["sparse_features"],
                output_keys=["field_embeddings"],
                task_types=["ctr"],
            ),
            SkillNode(
                node_id="flatten",
                skill_id="flatten_field_embeddings",
                skill_name="flatten_field_embeddings",
                category="utility",
                input_keys=["field_embeddings"],
                output_keys=["flat_embeddings"],
                task_types=["ctr"],
            ),
            SkillNode(
                node_id="linear",
                skill_id="linear_logit",
                skill_name="linear_logit",
                category="head",
                params={"input_key": "flat_embeddings", "output_key": "linear_logit", "input_dim": flat_input_dim},
                input_keys=["flat_embeddings"],
                output_keys=["linear_logit"],
                task_types=["ctr"],
            ),
            SkillNode(
                node_id="fm",
                skill_id="fm_interaction",
                skill_name="fm_interaction",
                category="interaction",
                input_keys=["field_embeddings"],
                output_keys=["fm_output"],
                task_types=["ctr"],
            ),
            SkillNode(
                node_id="deep_tower",
                skill_id="mlp_tower",
                skill_name="mlp_tower",
                category="tower",
                params={
                    "input_key": "flat_embeddings",
                    "output_key": "deep_logit",
                    "input_dim": flat_input_dim,
                    "hidden_dims": hidden_dims,
                    "dropout": config.dropout,
                    "activation": config.activation,
                    "output_layer": True,
                },
                input_keys=["flat_embeddings"],
                output_keys=["deep_logit"],
                task_types=["ctr"],
            ),
            SkillNode(
                node_id="fusion",
                skill_id="additive_fusion",
                skill_name="additive_fusion",
                category="fusion",
                params={"input_keys": ["linear_logit", "fm_output", "deep_logit"], "output_key": "logits"},
                input_keys=["linear_logit", "fm_output", "deep_logit"],
                output_keys=["logits"],
                task_types=["ctr"],
            ),
            SkillNode(
                node_id="prediction",
                skill_id="sigmoid_prediction",
                skill_name="sigmoid_prediction",
                category="head",
                params={"input_key": "logits", "output_key": "prediction"},
                input_keys=["logits"],
                output_keys=["prediction"],
                task_types=["ctr"],
            ),
            SkillNode(
                node_id="loss",
                skill_id="bce_loss",
                skill_name="bce_loss",
                category="objective",
                params={"logits_key": "logits", "labels_key": "labels", "output_key": "loss", "from_logits": True},
                input_keys=["logits", "labels"],
                output_keys=["loss"],
                task_types=["ctr"],
            ),
        ],
        edges=[
            SkillEdge("field_embedding", "flatten", "field_embeddings", "field_embeddings"),
            SkillEdge("field_embedding", "fm", "field_embeddings", "field_embeddings"),
            SkillEdge("flatten", "linear", "flat_embeddings", "flat_embeddings"),
            SkillEdge("flatten", "deep_tower", "flat_embeddings", "flat_embeddings"),
            SkillEdge("linear", "fusion", "linear_logit", "linear_logit"),
            SkillEdge("fm", "fusion", "fm_output", "fm_output"),
            SkillEdge("deep_tower", "fusion", "deep_logit", "deep_logit"),
            SkillEdge("fusion", "prediction", "logits", "logits"),
            SkillEdge("fusion", "loss", "logits", "logits"),
        ],
        objectives=[{"skill_id": "bce_loss", "output_key": "loss"}],
        constraints=GenomeConstraints(task_types=["ctr"], required_inputs=["sparse_features", "labels"], required_outputs=["logits", "loss"]),
    )
    genome.metadata.tags = ["ctr", config.template]
    return genome


class CTRGenomeTrainer:
    def __init__(self, training: CTRTrainingConfig) -> None:
        self.training = training
        self.device = torch.device(training.device)

    def fit_and_evaluate(
        self,
        genome: SkillGenome,
        bundle: CTRDatasetBundle,
        artifact_dir: str | Path,
        skill_library: SkillLibrary | None = None,
        write_artifacts: bool = True,
    ) -> dict[str, Any]:
        artifact_dir = Path(artifact_dir)
        if write_artifacts:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        _set_seed(self.training.seed)
        model = SkillGenomeCompiler(skill_library=skill_library).compile(genome)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.training.learning_rate, weight_decay=self.training.weight_decay)
        best_auc = -math.inf
        best_state = None
        best_val_metrics: dict[str, Any] | None = None
        patience = 0
        log_rows: list[dict[str, Any]] = []
        for epoch in range(self.training.epoch):
            train_loss = self._train_one_epoch(model, optimizer, bundle.train_loader)
            val_metrics = self.evaluate(model, bundle.val_loader)
            row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{key}": value for key, value in val_metrics.items()}}
            log_rows.append(row)
            auc = val_metrics.get("auc")
            if auc is not None and not math.isnan(float(auc)) and float(auc) > best_auc + self.training.min_delta:
                best_auc = float(auc)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_val_metrics = dict(val_metrics)
                patience = 0
            else:
                patience += 1
                if patience >= self.training.earlystop_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        test_metrics = self.evaluate(model, bundle.test_loader)
        if write_artifacts:
            _write_tsv(artifact_dir / "training_log.tsv", log_rows)
            with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "validation_best_auc": best_auc,
                        "validation": best_val_metrics or {},
                        "test": test_metrics,
                        "training_log": log_rows,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
                f.write("\n")
        validation_metrics = {
            f"validation_{key}": value
            for key, value in (best_val_metrics or {}).items()
        }
        return {"validation_best_auc": best_auc, **validation_metrics, **test_metrics}

    def _train_one_epoch(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, data_loader: DataLoader) -> float:
        model.train()
        losses = []
        for batch_idx, (x_dict, y) in enumerate(data_loader):
            if self.training.max_train_batches is not None and batch_idx >= self.training.max_train_batches:
                break
            batch = self._batch_to_device(x_dict, y)
            optimizer.zero_grad()
            ctx = model(batch)
            loss = ctx.get(self.training.loss_key)
            if loss is None:
                logits = ctx[self.training.logits_key]
                labels = batch["labels"].float().reshape_as(logits)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader) -> dict[str, Any]:
        model.eval()
        labels: list[float] = []
        predictions: list[float] = []
        losses: list[float] = []
        with torch.no_grad():
            for batch_idx, (x_dict, y) in enumerate(data_loader):
                if self.training.max_eval_batches is not None and batch_idx >= self.training.max_eval_batches:
                    break
                batch = self._batch_to_device(x_dict, y)
                ctx = model(batch)
                pred = ctx.get(self.training.prediction_key)
                if pred is None:
                    pred = torch.sigmoid(ctx[self.training.logits_key]).squeeze(-1)
                loss = ctx.get(self.training.loss_key)
                if loss is not None:
                    losses.append(float(loss.detach().cpu()))
                labels.extend(batch["labels"].detach().cpu().view(-1).tolist())
                predictions.extend(pred.detach().cpu().view(-1).tolist())
        clipped = np.clip(np.asarray(predictions, dtype=np.float64), 1e-7, 1 - 1e-7)
        target = np.asarray(labels, dtype=np.float64)
        metrics = {
            "auc": _safe_auc(target, clipped),
            "logloss": _safe_logloss(target, clipped),
            "loss": float(np.mean(losses)) if losses else None,
            "num_examples": int(len(target)),
        }
        return metrics

    def _batch_to_device(self, x_dict: dict[str, torch.Tensor], y: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = {key: value.to(self.device) for key, value in x_dict.items()}
        batch["labels"] = y.float().to(self.device)
        return batch


class CTRModelEvolutionRunner:
    """End-to-end CTR training, evaluation, and Stage 2 evolution runner."""

    def __init__(self, config: CTRWorkflowConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory = EvolutionMemory(config.memory_path or self.output_dir / "evolution_memory.jsonl", persist=config.write_evolution_memory)
        self.trainer = CTRGenomeTrainer(config.training)
        self.bundle: CTRDatasetBundle | None = None
        self.candidate_parallelism: list[dict[str, Any]] = []
        self.candidate_generation_errors: list[dict[str, Any]] = []
        self.deduplication_records: list[dict[str, Any]] = []
        self.code_space_generation_records: list[dict[str, Any]] = []
        self._temporary_staging: tempfile.TemporaryDirectory[str] | None = None
        self._code_space_staging_root = config.evolution.code_space.staging_root
        if self._code_space_staging_root is None and not config.write_metadata_files:
            self._temporary_staging = tempfile.TemporaryDirectory(prefix="evoskillrec_code_space_")
            self._code_space_staging_root = self._temporary_staging.name
        self.training_config_hash = config_fingerprint(asdict(config.training))
        self.genome_config_hash = config_fingerprint(asdict(config.genome))
        self.hparam_fingerprint = config_fingerprint({"training": asdict(config.training), "genome": asdict(config.genome)})
        self.hparam_changes = {
            "training": diff_payload(asdict(CTRTrainingConfig()), asdict(config.training)),
            "genome": diff_payload(asdict(CTRGenomeConfig()), asdict(config.genome)),
        }
        self.adaptive_budget_state = {
            "global_best_score": None,
            "stagnant_rounds": 0,
            "active_code_candidates": None,
            "records": [],
        }
        self.architecture_index: dict[str, str] = {}
        if config.evolution.deduplicate_against_memory:
            for record in self.memory.all_records():
                fingerprint = (record.validation_results or {}).get("architecture_fingerprint")
                if fingerprint:
                    self.architecture_index.setdefault(str(fingerprint), record.child_genome_id or record.record_id)

    def run(self) -> dict[str, Any]:
        try:
            return self._run()
        finally:
            self._cleanup_temporary_staging()

    def _run(self) -> dict[str, Any]:
        started = time.time()
        self.bundle = prepare_movielens_ctr_data(self.config.dataset, self.config.training)
        if self.config.write_metadata_files:
            self._write_json("dataset_metadata.json", self.bundle.metadata)
        baseline = self._load_or_build_genome(self.bundle)
        GenomeVerifier().assert_valid(baseline)
        if self.config.write_metadata_files:
            baseline_path = self.output_dir / "baseline_genome.json"
            baseline.save(baseline_path)
        baseline_fingerprint = _safe_architecture_fingerprint(baseline, fallback="baseline")
        self.architecture_index[baseline_fingerprint] = "baseline"
        results = [self._train_candidate("baseline", baseline, None, "baseline", "baseline", "Initial CTR genome")]
        self._initialize_adaptive_candidate_budget(results[0])
        self._write_progress_results(results)

        parent_population = [baseline]
        for round_idx in range(self.config.evolution.rounds if self.config.evolution.enabled else 0):
            candidates = self._generate_round_candidates(parent_population, round_idx=round_idx)
            unique_candidates, skipped_results = self._deduplicate_candidates(candidates)
            self.deduplication_records.append(
                {
                    "round_idx": round_idx,
                    "generated_candidates": len(candidates),
                    "unique_candidates": len(unique_candidates),
                    "skipped_duplicates": len(skipped_results),
                    "skipped_candidate_ids": [result.candidate_id for result in skipped_results],
                }
            )
            round_results = self._train_round_candidates(unique_candidates, round_idx=round_idx)
            result_by_candidate_id = {result.candidate_id: result for result in [*skipped_results, *round_results]}
            results.extend([result_by_candidate_id[spec.candidate_id] for spec in candidates if spec.candidate_id in result_by_candidate_id])
            survivors = _select_survivors(round_results, self.config.training, self.config.evolution)
            survivor_ids = {result.candidate_id for result in survivors}
            for result in round_results:
                if result.error is None:
                    result.status = "survivor" if result.candidate_id in survivor_ids else "discarded"
                    self._maybe_promote_generated_skill(result)
            self._write_round_survivor_artifacts(round_idx, survivors)
            self._update_adaptive_candidate_budget(round_idx, round_results)
            parent_population = _next_parent_population(parent_population, survivors)
            self._write_progress_results(results)

        self._write_progress_results(results)
        summary = {
            "experiment_name": self.config.experiment_name,
            "output_dir": str(self.output_dir),
            "elapsed_sec": round(time.time() - started, 4),
            "candidate_parallelism": self.candidate_parallelism,
            "candidate_generation_errors": self.candidate_generation_errors,
            "code_space_generation": self.code_space_generation_records,
            "deduplication": self.deduplication_records,
            "training_config_hash": self.training_config_hash,
            "genome_config_hash": self.genome_config_hash,
            "hparam_fingerprint": self.hparam_fingerprint,
            "hparam_changes": self.hparam_changes,
            "adaptive_candidate_budget": list(self.adaptive_budget_state["records"]),
            "results": [result.to_dict() for result in results],
        }
        if self.config.write_summary_json:
            self._write_json("summary.json", summary)
        return summary

    def _generate_round_candidates(self, parent_population: list[SkillGenome], round_idx: int) -> list[CandidateSpec]:
        code_config = self.config.evolution.code_space
        provider_name = str(code_config.provider).lower().replace("-", "_")
        if provider_name in {"fallback", "disabled", "none"}:
            try:
                return generate_ctr_candidate_population(parent_population, self.config, round_idx=round_idx)
            except Exception as exc:
                self._record_candidate_generation_error(round_idx, "skill_candidate_generation", exc)
                return []

        target_skill, target_code = self._candidate_targets_for_round(round_idx)
        skill_evolution = replace(self.config.evolution, candidate_budget=target_skill, code_space_probability=0.0)
        skill_config = replace(self.config, evolution=skill_evolution)
        try:
            specs = generate_ctr_candidate_population(parent_population, skill_config, round_idx=round_idx)[:target_skill]
        except Exception as exc:
            self._record_candidate_generation_error(round_idx, "skill_candidate_generation", exc, target_skill=target_skill)
            specs = []
        try:
            code_specs = self._generate_open_ended_code_candidates(
                parent_population,
                round_idx=round_idx,
                target_code=target_code,
            )
        except Exception as exc:
            self._record_candidate_generation_error(round_idx, "code_candidate_generation", exc, target_code=target_code)
            code_specs = []
        specs.extend(code_specs)
        if len(code_specs) < target_code and not code_config.allow_fallback_templates:
            missing = target_code - len(code_specs)
            self._record_insufficient_code_space_candidates(round_idx, target_code, len(code_specs))
            supplement = self._generate_skill_space_supplement(
                parent_population,
                round_idx=round_idx,
                count=missing,
                reason="code_space_insufficient",
                skip_candidate_ids={spec.candidate_id for spec in specs},
            )
            specs.extend(supplement)
        if len(code_specs) < target_code and code_config.allow_fallback_templates:
            fallback_evolution = replace(self.config.evolution, candidate_budget=target_code - len(code_specs), code_space_probability=1.0, strategy="templates")
            fallback_config = replace(self.config, evolution=fallback_evolution)
            try:
                fallback = [
                    spec
                    for spec in generate_ctr_candidate_population(parent_population, fallback_config, round_idx=round_idx)
                    if spec.evolution_space == "code_space"
                ]
            except Exception as exc:
                self._record_candidate_generation_error(round_idx, "code_fallback_template_generation", exc, target_code=target_code - len(code_specs))
                fallback = []
            specs.extend(fallback[: target_code - len(code_specs)])
        seen_ids: set[str] = set()
        for spec in specs:
            spec.candidate_id = _dedupe_candidate_id(spec.candidate_id, seen_ids)
        return specs[: self.config.evolution.candidate_budget]

    def _record_insufficient_code_space_candidates(self, round_idx: int, requested: int, generated: int) -> None:
        code_config = self.config.evolution.code_space
        self.candidate_generation_errors.append(
            {
                "round_idx": round_idx,
                "stage": "insufficient_open_ended_code_candidates",
                "error": f"requested {requested} code-space candidates but generated {generated}",
                "target_code": requested,
                "generated": generated,
                "provider": code_config.provider,
                "allow_fallback_templates": code_config.allow_fallback_templates,
                "require_active_provider": code_config.require_active_provider,
            }
        )
        provider_diagnostics = _safe_code_space_provider_diagnostics(code_config)
        self._write_json(
            f"round{round_idx}_code_space_error.json",
            {
                "error": "insufficient_open_ended_code_candidates",
                "provider": code_config.provider,
                "structural_scopes": code_config.structural_scopes,
                "scope_providers": provider_diagnostics.get("scope_providers", {}),
                "macro_requires_live_provider": code_config.macro_requires_live_provider,
                "requested": requested,
                "generated": generated,
                "allow_fallback_templates": code_config.allow_fallback_templates,
                "require_active_provider": code_config.require_active_provider,
            },
        )

    def _generate_skill_space_supplement(
        self,
        parent_population: list[SkillGenome],
        *,
        round_idx: int,
        count: int,
        reason: str,
        skip_candidate_ids: set[str] | None = None,
    ) -> list[CandidateSpec]:
        if count <= 0:
            return []
        skip_candidate_ids = set(skip_candidate_ids or set())
        supplement_evolution = replace(
            self.config.evolution,
            candidate_budget=max(self.config.evolution.candidate_budget, count + len(skip_candidate_ids)),
            code_space_probability=0.0,
        )
        supplement_config = replace(self.config, evolution=supplement_evolution)
        try:
            pool = generate_ctr_candidate_population(parent_population, supplement_config, round_idx=round_idx)
        except Exception as exc:
            self._record_candidate_generation_error(round_idx, "skill_space_supplement_generation", exc, target_skill=count, reason=reason)
            return []
        supplement = [spec for spec in pool if spec.candidate_id not in skip_candidate_ids][:count]
        for idx, spec in enumerate(supplement):
            spec.candidate_id = f"{spec.candidate_id}_supplement{idx}"
            spec.rationale = f"{spec.rationale} (skill-space supplement: {reason})"
        self.code_space_generation_records.append(
            {
                "round_idx": round_idx,
                "stage": "skill_space_supplement",
                "reason": reason,
                "requested": count,
                "generated": len(supplement),
            }
        )
        return supplement

    def _record_candidate_generation_error(self, round_idx: int, stage: str, exc: Exception, **metadata: Any) -> None:
        self.candidate_generation_errors.append(
            {
                "round_idx": round_idx,
                "stage": stage,
                "error": f"{exc.__class__.__name__}: {exc}",
                **metadata,
            }
        )

    def _initialize_adaptive_candidate_budget(self, baseline: CandidateResult) -> None:
        value = _objective_value(baseline.metrics, self.config.training)
        if value is not None and not _is_nan(float(value)):
            self.adaptive_budget_state["global_best_score"] = float(value)

    def _candidate_targets_for_round(self, round_idx: int) -> tuple[int, int]:
        targets = _adaptive_candidate_targets(
            self.config.evolution,
            stagnant_rounds=int(self.adaptive_budget_state.get("stagnant_rounds") or 0),
            active_code_candidates=self.adaptive_budget_state.get("active_code_candidates"),
        )
        self.adaptive_budget_state["active_code_candidates"] = targets[1]
        self.adaptive_budget_state["records"].append(
            {
                "round_idx": round_idx,
                "candidate_budget": targets[0] + targets[1],
                "skill_target": targets[0],
                "code_target": targets[1],
                "stagnant_rounds_before_round": int(self.adaptive_budget_state.get("stagnant_rounds") or 0),
                "global_best_before_round": self.adaptive_budget_state.get("global_best_score"),
            }
        )
        return targets

    def _update_adaptive_candidate_budget(self, round_idx: int, round_results: list[CandidateResult]) -> None:
        config = self.config.evolution.adaptive_candidate_budget
        if not config.enabled:
            return
        round_best = _best_objective_value(round_results, self.config.training)
        record = next(
            (item for item in reversed(self.adaptive_budget_state["records"]) if item.get("round_idx") == round_idx),
            None,
        )
        if record is not None:
            record["round_best_score"] = round_best
        if round_best is None:
            self.adaptive_budget_state["stagnant_rounds"] = int(self.adaptive_budget_state.get("stagnant_rounds") or 0) + 1
            if record is not None:
                record["improved"] = False
                record["stagnation_reason"] = "no_successful_round_result"
                record["stagnant_rounds_after_round"] = int(self.adaptive_budget_state.get("stagnant_rounds") or 0)
            return
        global_best = self.adaptive_budget_state.get("global_best_score")
        improvement = _objective_improvement(round_best, global_best, self.config.training)
        improved = improvement is None or improvement >= float(config.min_improvement)
        if improved:
            self.adaptive_budget_state["global_best_score"] = float(round_best)
            if config.reset_on_improvement:
                self.adaptive_budget_state["stagnant_rounds"] = 0
                self.adaptive_budget_state["active_code_candidates"] = None
            else:
                self.adaptive_budget_state["stagnant_rounds"] = max(
                    0,
                    int(self.adaptive_budget_state.get("stagnant_rounds") or 0) - int(config.cooldown_code_step),
                )
        else:
            self.adaptive_budget_state["stagnant_rounds"] = int(self.adaptive_budget_state.get("stagnant_rounds") or 0) + 1
        if record is not None:
            record["improvement"] = improvement
            record["improved"] = bool(improved)
            record["stagnant_rounds_after_round"] = int(self.adaptive_budget_state.get("stagnant_rounds") or 0)
            record["global_best_after_round"] = self.adaptive_budget_state.get("global_best_score")

    def _generate_open_ended_code_candidates(self, parents: list[SkillGenome], round_idx: int, target_code: int) -> list[CandidateSpec]:
        if target_code <= 0:
            return []
        try:
            provider = build_code_space_provider(self.config.evolution.code_space)
        except Exception as exc:
            self._record_candidate_generation_error(round_idx, "build_code_space_provider", exc, target_code=target_code)
            return []
        specs: list[CandidateSpec] = []
        proposal_reports: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        parent_allocations = _allocate_code_budget_by_parent(parents, budget=target_code, round_idx=round_idx)
        for parent_idx, parent, parent_budget in parent_allocations:
            diagnosis_report = {
                "failure_modes": list(self.config.evolution.failure_modes),
                "target_node_id": "fusion",
                "round_idx": round_idx,
                "parent_index": parent_idx,
                "parent_rank": parent_idx,
                "parent_genome_id": parent.metadata.genome_id,
                "retained_parent_count": len(parents),
                "structural_scopes": list(self.config.evolution.code_space.structural_scopes),
            }
            parent_specs: list[CandidateSpec] = []
            retry_feedback: list[dict[str, Any]] = []
            attempts = max(1, int(self.config.evolution.code_space.max_retries))
            for attempt_idx in range(attempts):
                attempt_report = dict(diagnosis_report)
                if retry_feedback:
                    attempt_report["previous_code_space_failures"] = retry_feedback[-8:]
                    attempt_report["repair_instruction"] = (
                        "Regenerate different valid macro proposals that avoid the previous ingestion failures. "
                        "Fix missing macro_judgment fields, unavailable input keys, unsafe code, import errors, "
                        "shape mismatches, and invalid wiring before returning JSON."
                    )
                try:
                    proposals = provider.propose(
                        parent=parent,
                        diagnosis_report=attempt_report,
                        evolution_memory=self.memory,
                        budget=parent_budget,
                        round_idx=round_idx,
                    )
                except Exception as exc:
                    error = {
                        "parent_index": parent_idx,
                        "parent_genome_id": parent.metadata.genome_id,
                        "attempt": attempt_idx + 1,
                        "stage": "provider_propose",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                    errors.append(error)
                    retry_feedback.append(error)
                    continue
                proposal_reports.append(
                    {
                        "parent_index": parent_idx,
                        "parent_genome_id": parent.metadata.genome_id,
                        "attempt": attempt_idx + 1,
                        "requested": parent_budget,
                        "received": len(proposals),
                        "proposal_ids": [proposal.proposal_id for proposal in proposals],
                        "proposal_structural_scopes": {
                            proposal.proposal_id: proposal.structural_scope or proposal.metadata.get("structural_scope", "")
                            for proposal in proposals
                        },
                    }
                )
                if not proposals:
                    error = {
                        "parent_index": parent_idx,
                        "parent_genome_id": parent.metadata.genome_id,
                        "attempt": attempt_idx + 1,
                        "stage": "provider_propose",
                        "error": "provider returned no proposals",
                    }
                    errors.append(error)
                    retry_feedback.append(error)
                    continue
                try:
                    ingested = inspect_open_ended_proposal_ingestions(
                        proposals=proposals,
                        parent=parent,
                        output_dir=self.output_dir,
                        memory=self.memory,
                        staging_root=self._code_space_staging_root,
                    )
                except Exception as exc:
                    error = {
                        "parent_index": parent_idx,
                        "parent_genome_id": parent.metadata.genome_id,
                        "attempt": attempt_idx + 1,
                        "stage": "ingest_open_ended_proposals",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                    errors.append(error)
                    retry_feedback.append(error)
                    continue
                failed_ingestions = [item for item in ingested if not item.success]
                for item in failed_ingestions:
                    diagnostic = item.diagnostic()
                    error = {
                        "parent_index": parent_idx,
                        "parent_genome_id": parent.metadata.genome_id,
                        "attempt": attempt_idx + 1,
                        "stage": "ingest_open_ended_proposals",
                        "proposal_id": diagnostic.get("proposal_id", ""),
                        "skill_id": diagnostic.get("skill_id", ""),
                        "error": diagnostic.get("message", "ingestion failed"),
                        "validation_results": diagnostic.get("validation_results", {}),
                    }
                    errors.append(error)
                    retry_feedback.append(error)
                successful_ingestions = [item for item in ingested if item.success]
                for item in successful_ingestions[:parent_budget]:
                    if len(parent_specs) + len(specs) >= target_code:
                        break
                    spec = self._candidate_spec_from_open_ended_result(
                        item,
                        parent=parent,
                        parent_idx=parent_idx,
                        round_idx=round_idx,
                        retained_parent_count=len(parents),
                        errors=errors,
                    )
                    if spec is not None:
                        parent_specs.append(spec)
                if len(parent_specs) >= parent_budget:
                    break
            specs.extend(parent_specs[:parent_budget])
            if not parent_specs:
                self.candidate_generation_errors.append(
                    {
                        "round_idx": round_idx,
                        "stage": "code_space_parent_generation",
                        "error": "no valid code-space candidates for retained parent",
                        "parent_index": parent_idx,
                        "parent_genome_id": parent.metadata.genome_id,
                        "target_code": parent_budget,
                    }
                )
            if len(specs) >= target_code:
                break
        diagnostic_payload = {
            "provider": self.config.evolution.code_space.provider,
            "provider_diagnostics": _safe_code_space_provider_diagnostics(self.config.evolution.code_space),
            "requested": target_code,
            "received": sum(report["received"] for report in proposal_reports),
            "generated": len(specs),
            "retained_parent_count": len(parents),
            "parent_budgets": [
                {"parent_index": parent_idx, "parent_genome_id": parent.metadata.genome_id, "budget": budget}
                for parent_idx, parent, budget in parent_allocations
            ],
            "parent_proposals": proposal_reports,
            "proposal_ids": [
                proposal_id
                for report in proposal_reports
                for proposal_id in report["proposal_ids"]
            ],
            "proposal_structural_scopes": {
                proposal_id: scope
                for report in proposal_reports
                for proposal_id, scope in report["proposal_structural_scopes"].items()
            },
            "errors": errors,
        }
        self.code_space_generation_records.append(
            {
                "round_idx": round_idx,
                "requested": target_code,
                "generated": len(specs),
                "received": diagnostic_payload["received"],
                "errors": len(errors),
            }
        )
        if self.config.write_code_space_diagnostics or len(specs) < target_code or errors:
            self._write_json(f"round{round_idx}_code_space_proposals.json", diagnostic_payload)
        return specs

    def _candidate_spec_from_open_ended_result(
        self,
        item: Any,
        *,
        parent: SkillGenome,
        parent_idx: int,
        round_idx: int,
        retained_parent_count: int,
        errors: list[dict[str, Any]],
    ) -> CandidateSpec | None:
        try:
            ingestion = item.ingestion
            genome = ingestion.genome
            if genome is None:
                return None
            skill_id = ingestion.skill_id or item.proposal.skill_id or item.proposal.proposal_id
            code_hash = _path_sha256(ingestion.code_path) if ingestion.code_path else None
            if code_hash:
                _annotate_generated_skill_hash(genome, skill_id, code_hash)
            _annotate_generated_skill_artifacts(
                genome,
                skill_id,
                staged_code_path=ingestion.code_path,
                staged_skill_card_path=ingestion.skill_card_path,
            )
            candidate_id = f"round{round_idx}_code_{skill_id}"
            if retained_parent_count > 1:
                candidate_id = f"{candidate_id}_p{parent_idx}"
            return CandidateSpec(
                candidate_id=candidate_id,
                genome=genome,
                parent_genome_id=parent.metadata.genome_id,
                mutation_type=f"code_open_ended_{skill_id}",
                rationale=item.proposal.architecture_hypothesis,
                evolution_space="code_space",
                operation="open_ended",
                generated_skill_id=skill_id,
                staged_code_path=ingestion.code_path,
                staged_skill_card_path=ingestion.skill_card_path,
                proposal_id=item.proposal.proposal_id,
                structural_scope=item.proposal.structural_scope or item.proposal.metadata.get("structural_scope", ""),
            )
        except Exception as exc:
            proposal = getattr(item, "proposal", None)
            errors.append(
                {
                    "parent_index": parent_idx,
                    "parent_genome_id": parent.metadata.genome_id,
                    "proposal_id": getattr(proposal, "proposal_id", ""),
                    "stage": "build_code_candidate_spec",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            return None

    def _deduplicate_candidates(self, candidates: list[CandidateSpec]) -> tuple[list[CandidateSpec], list[CandidateResult]]:
        if not self.config.evolution.deduplicate_architectures:
            for spec in candidates:
                spec.architecture_fingerprint = _safe_architecture_fingerprint(spec.genome, fallback=spec.candidate_id)
            return candidates, []
        unique = []
        skipped = []
        for spec in candidates:
            fingerprint = _safe_architecture_fingerprint(spec.genome, fallback=spec.candidate_id)
            spec.architecture_fingerprint = fingerprint
            duplicate_of = self.architecture_index.get(fingerprint)
            if duplicate_of:
                spec.duplicate_of = duplicate_of
                spec.dedup_status = "duplicate"
                skipped.append(self._skipped_duplicate_result(spec, duplicate_of))
                continue
            self.architecture_index[fingerprint] = spec.candidate_id
            unique.append(spec)
        return unique, skipped

    def _skipped_duplicate_result(self, spec: CandidateSpec, duplicate_of: str) -> CandidateResult:
        result = CandidateResult(
            candidate_id=spec.candidate_id,
            genome_id=spec.genome.metadata.genome_id,
            parent_genome_id=spec.parent_genome_id,
            mutation_type=spec.mutation_type,
            status="skipped_duplicate",
            metrics={},
            artifact_dir="",
            genome_path="",
            elapsed_sec=0.0,
            evolution_space=spec.evolution_space,
            operation=spec.operation,
            rationale=spec.rationale,
            architecture_fingerprint=spec.architecture_fingerprint,
            duplicate_of=duplicate_of,
            dedup_status="duplicate",
            structure_category=_structure_category(spec.evolution_space, spec.operation, spec.mutation_type),
            hparam_category="fixed_training_config",
            training_config_hash=self.training_config_hash,
            genome_config_hash=self.genome_config_hash,
            hparam_fingerprint=self.hparam_fingerprint,
            hparam_changes=self.hparam_changes,
            generated_skill_id=spec.generated_skill_id,
            staged_code_path=spec.staged_code_path,
            staged_skill_card_path=spec.staged_skill_card_path,
            proposal_id=spec.proposal_id,
            structural_scope=spec.structural_scope,
            genome=spec.genome,
        )
        self._append_memory_record(result, initial_status="skipped_duplicate")
        return result

    def _load_or_build_genome(self, bundle: CTRDatasetBundle) -> SkillGenome:
        if self.config.genome.genome_path:
            return SkillGenome.load(self.config.genome.genome_path)
        return build_default_ctr_genome(bundle, self.config.genome)

    def _train_candidate(
        self,
        candidate_id: str,
        genome: SkillGenome,
        parent_genome_id: str | None,
        mutation_type: str,
        initial_status: str,
        rationale: str,
        evolution_space: str = "skill_space",
        operation: str = "",
        architecture_fingerprint: str | None = None,
        generated_skill_id: str | None = None,
        staged_code_path: str | None = None,
        staged_skill_card_path: str | None = None,
        proposal_id: str | None = None,
        structural_scope: str = "",
    ) -> CandidateResult:
        if self.bundle is None:
            self.bundle = prepare_movielens_ctr_data(self.config.dataset, self.config.training)
        result = _train_candidate_artifact(
            candidate_id=candidate_id,
            genome=genome,
            parent_genome_id=parent_genome_id,
            mutation_type=mutation_type,
            initial_status=initial_status,
            rationale=rationale,
            evolution_space=evolution_space,
            operation=operation,
            trainer=self.trainer,
            bundle=self.bundle,
            output_dir=self.output_dir,
            training_config_hash=self.training_config_hash,
            genome_config_hash=self.genome_config_hash,
            hparam_fingerprint=self.hparam_fingerprint,
            hparam_changes=self.hparam_changes,
            architecture_fingerprint=architecture_fingerprint,
            generated_skill_id=generated_skill_id,
            staged_code_path=staged_code_path,
            staged_skill_card_path=staged_skill_card_path,
            proposal_id=proposal_id,
            structural_scope=structural_scope,
        )
        self._append_memory_record(result, initial_status=initial_status)
        return result

    def _train_round_candidates(self, candidates: list[CandidateSpec], round_idx: int) -> list[CandidateResult]:
        if not candidates:
            return []
        gpu_ids = _available_candidate_gpu_ids() if _uses_cuda(self.config.training.device) else []
        if len(candidates) == 1 or len(gpu_ids) < 2:
            self.candidate_parallelism.append(
                {
                    "round_idx": round_idx,
                    "mode": "serial",
                    "num_candidates": len(candidates),
                    "gpu_ids": gpu_ids[:1],
                    "reason": "fewer than 2 available GPUs" if _uses_cuda(self.config.training.device) else "non-CUDA training device",
                }
            )
            return [
                self._train_candidate(
                    spec.candidate_id,
                    spec.genome,
                    spec.parent_genome_id,
                    spec.mutation_type,
                    "candidate",
                    spec.rationale,
                    evolution_space=spec.evolution_space,
                    operation=spec.operation,
                    architecture_fingerprint=spec.architecture_fingerprint,
                    generated_skill_id=spec.generated_skill_id,
                    staged_code_path=spec.staged_code_path,
                    staged_skill_card_path=spec.staged_skill_card_path,
                    proposal_id=spec.proposal_id,
                    structural_scope=spec.structural_scope,
                )
                for spec in candidates
            ]

        processes = min(len(candidates), len(gpu_ids))
        parallel_record = {
            "round_idx": round_idx,
            "mode": "multiprocess",
            "num_candidates": len(candidates),
            "processes": processes,
            "gpu_ids": gpu_ids[:processes],
            "waves": [],
        }
        self.candidate_parallelism.append(parallel_record)
        results: list[CandidateResult] = []
        for start_idx in range(0, len(candidates), processes):
            wave = candidates[start_idx : start_idx + processes]
            wave_gpu_ids = gpu_ids[: len(wave)]
            wave_record = {
                "wave_idx": start_idx // processes,
                "num_candidates": len(wave),
                "gpu_ids": wave_gpu_ids,
            }
            parallel_record["waves"].append(wave_record)
            payloads = [
                {
                    "config": self.config,
                    "output_dir": str(self.output_dir),
                    "spec": spec,
                    "gpu_id": wave_gpu_ids[idx],
                    "training_config_hash": self.training_config_hash,
                    "genome_config_hash": self.genome_config_hash,
                    "hparam_fingerprint": self.hparam_fingerprint,
                    "hparam_changes": self.hparam_changes,
                }
                for idx, spec in enumerate(wave)
            ]
            try:
                with mp.get_context("spawn").Pool(processes=len(wave)) as pool:
                    wave_results = pool.map(_train_candidate_worker, payloads)
            except Exception as exc:
                wave_record["fallback_error"] = f"{exc.__class__.__name__}: {exc}"
                wave_results = [
                    self._train_candidate(
                        spec.candidate_id,
                        spec.genome,
                        spec.parent_genome_id,
                        spec.mutation_type,
                        "candidate",
                        spec.rationale,
                        evolution_space=spec.evolution_space,
                        operation=spec.operation,
                        architecture_fingerprint=spec.architecture_fingerprint,
                        generated_skill_id=spec.generated_skill_id,
                        staged_code_path=spec.staged_code_path,
                        staged_skill_card_path=spec.staged_skill_card_path,
                        proposal_id=spec.proposal_id,
                        structural_scope=spec.structural_scope,
                    )
                    for spec in wave
                ]
            else:
                for result in wave_results:
                    self._append_memory_record(result, initial_status="candidate")
            results.extend(wave_results)
        return results

    def _append_memory_record(self, result: CandidateResult, initial_status: str) -> None:
        self.memory.append(
            _memory_record_from_result(
                result,
                self.config.evolution.failure_modes,
                initial_status,
                task_type=self._memory_task_type(),
            )
        )

    def _memory_task_type(self) -> str:
        return "ctr"

    def _maybe_promote_generated_skill(self, result: CandidateResult) -> None:
        if not result.generated_skill_id or not result.staged_code_path or not result.staged_skill_card_path:
            return
        if not should_promote_generated_skill(
            candidate_status=result.status,
            candidate_error=result.error,
            policy=self.config.evolution.code_space.promotion_policy,
        ):
            return
        if result.genome is None:
            return
        try:
            promotion = promote_generated_skill(
                skill_id=result.generated_skill_id,
                staged_code_path=result.staged_code_path,
                staged_card_path=result.staged_skill_card_path,
                candidate_id=result.candidate_id,
                candidate_metrics=result.metrics,
                candidate_status=result.status,
                genome=result.genome,
                promotion_status="promoted" if result.status == "survivor" else "validated",
            )
        except Exception as exc:
            result.promotion_result = {
                "promoted": False,
                "skill_id": result.generated_skill_id,
                "message": f"{exc.__class__.__name__}: {exc}",
                "metadata": {"candidate_id": result.candidate_id, "promotion_status": "promotion_failed"},
            }
            self.memory.append(
                EvolutionRecord(
                    parent_genome_id=result.parent_genome_id,
                    child_genome_id=result.genome_id,
                    mutation_type=result.mutation_type,
                    proposal_id=result.proposal_id,
                    status="promotion_failed",
                    validation_results={
                        "promotion_result": result.promotion_result,
                        "architecture_fingerprint": result.architecture_fingerprint,
                        "generated_skill_id": result.generated_skill_id,
                    },
                    evaluation_metrics=result.metrics,
                    failure_mode=",".join(self.config.evolution.failure_modes),
                    rationale=result.rationale,
                    artifact_paths=_artifact_paths_for_result(result),
                    task_type=self._memory_task_type(),
                )
            )
            return
        result.promotion_result = promotion.to_dict()
        if promotion.promoted and promotion.skill_id and result.genome is not None:
            _rewrite_generated_skill_reference(
                result.genome,
                result.generated_skill_id,
                promotion.skill_id,
                persisted_code_path=promotion.code_path,
                persisted_skill_card_path=promotion.skill_card_path,
            )
            _register_promoted_generated_skill(promotion.skill_id, promotion.skill_card_path)
            result.generated_skill_id = promotion.skill_id
        artifact_paths = {
            "staged_code": result.staged_code_path or "",
            "staged_skill_card": result.staged_skill_card_path or "",
            "promoted_code": promotion.code_path or "",
            "promoted_skill_card": promotion.skill_card_path or "",
        }
        if result.artifact_dir:
            artifact_paths["artifact_dir"] = result.artifact_dir
        self.memory.append(
            EvolutionRecord(
                parent_genome_id=result.parent_genome_id,
                child_genome_id=result.genome_id,
                mutation_type=result.mutation_type,
                proposal_id=result.proposal_id,
                status="promoted" if promotion.promoted else "promotion_failed",
                validation_results={
                    "promotion_result": promotion.to_dict(),
                    "architecture_fingerprint": result.architecture_fingerprint,
                    "generated_skill_id": result.generated_skill_id,
                },
                evaluation_metrics=result.metrics,
                failure_mode=",".join(self.config.evolution.failure_modes),
                rationale=result.rationale,
                artifact_paths=artifact_paths,
                task_type=self._memory_task_type(),
            )
        )

    def _write_results(self, results: list[CandidateResult]) -> None:
        self._write_progress_results(results)

    def _write_progress_results(self, results: list[CandidateResult]) -> None:
        rows = self._build_result_report_rows(results)
        _write_tsv(self.output_dir / "results.tsv", rows)
        _write_tsv(self.output_dir / "round_history.tsv", self._build_round_history_rows(results))

    def _write_round_survivor_artifacts(self, round_idx: int, survivors: list[CandidateResult]) -> None:
        if not self.config.write_survivor_artifacts:
            return
        for rank, result in enumerate(survivors, start=1):
            if result.genome is None:
                continue
            artifact_dir = self.output_dir / self.config.survivor_artifacts_dir / f"round{round_idx:02d}" / result.candidate_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            genome_path = artifact_dir / "genome.json"
            result.genome.save(genome_path)
            result.genome_path = str(genome_path)
            result.artifact_dir = str(artifact_dir)
            metrics_payload = {
                "rank": rank,
                "candidate_id": result.candidate_id,
                "genome_id": result.genome_id,
                "parent_genome_id": result.parent_genome_id,
                "mutation_type": result.mutation_type,
                "status": result.status,
                "evolution_space": result.evolution_space,
                "operation": result.operation,
                "rationale": result.rationale,
                "architecture_fingerprint": result.architecture_fingerprint,
                "metrics": result.metrics,
                "generated_skill_id": result.generated_skill_id,
                "proposal_id": result.proposal_id,
                "structural_scope": result.structural_scope,
            }
            with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=2, sort_keys=True)
                f.write("\n")
            _write_genome_structure_markdown(artifact_dir / "structure.md", result.genome, result=result, rank=rank)
            self._write_compiled_model_repr(artifact_dir / "model_repr.txt", result.genome)

    def _write_compiled_model_repr(self, path: Path, genome: SkillGenome) -> None:
        try:
            model = SkillGenomeCompiler(skill_library=_skill_library_for_candidate(None, genome=genome)).compile(genome)
            text = repr(model)
        except Exception as exc:
            text = f"Failed to compile model representation: {exc.__class__.__name__}: {exc}"
        path.write_text(text + "\n", encoding="utf-8")

    def _build_result_report_rows(self, results: list[CandidateResult]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        round_groups = _round_result_groups(results)
        round_counts = {round_idx: _round_space_counts(group) for round_idx, group in round_groups.items()}
        round_best = {round_idx: _best_result_by_validation_auc(group) for round_idx, group in round_groups.items()}
        candidate_slots = _candidate_slots(round_groups)
        auc_curve = _validation_auc_curve(results, round_best)

        for result in results:
            round_idx = _round_idx_from_candidate_id(result.candidate_id)
            counts = round_counts.get(round_idx, {}) if round_idx is not None else {}
            best = round_best.get(round_idx) if round_idx is not None else None
            curve = auc_curve.get(round_idx if round_idx is not None else "baseline", {})
            row = {
                "row_type": "baseline" if result.candidate_id == "baseline" else "candidate",
                "round_idx": "" if round_idx is None else round_idx,
                "candidate_slot": candidate_slots.get(result.candidate_id, ""),
                "candidate_id": result.candidate_id,
                "candidate_description": result.rationale,
                "generation": _candidate_generation_label(result),
                "evolution_category": _candidate_evolution_category(result),
                "genome_id": result.genome_id,
                "parent_genome_id": result.parent_genome_id or "",
                "mutation_type": result.mutation_type,
                "evolution_space": result.evolution_space,
                "operation": result.operation,
                "structure_category": result.structure_category or _candidate_evolution_category(result),
                "hparam_category": result.hparam_category,
                "architecture_fingerprint": result.architecture_fingerprint,
                "duplicate_of": result.duplicate_of or "",
                "dedup_status": result.dedup_status,
                "training_config_hash": result.training_config_hash,
                "genome_config_hash": result.genome_config_hash,
                "hparam_fingerprint": result.hparam_fingerprint,
                "hparam_changes": json.dumps(result.hparam_changes, sort_keys=True),
                "generated_skill_id": result.generated_skill_id or "",
                "proposal_id": result.proposal_id or "",
                "structural_scope": result.structural_scope,
                "promotion_status": (result.promotion_result or {}).get("metadata", {}).get("promotion_status", "") if result.promotion_result else "",
                "status": result.status,
                "validation_best_auc": result.metrics.get("validation_best_auc"),
                "round_best_candidate_id": best.candidate_id if best else "",
                "round_best_validation_auc": _validation_auc(best) if best else "",
                "is_round_best": bool(best and result.candidate_id == best.candidate_id),
                "best_candidate_id_so_far": curve.get("best_candidate_id_so_far", ""),
                "best_validation_auc_so_far": curve.get("best_validation_auc_so_far", ""),
                "round_total_candidates": counts.get("total", ""),
                "round_skill_candidates": counts.get("skill_space", ""),
                "round_code_candidates": counts.get("code_space", ""),
                "round_space_mix": _round_space_mix_label(counts),
                "round_failed_count": counts.get("failed", ""),
                "round_skipped_count": counts.get("skipped_duplicate", ""),
                "round_skipped_duplicate_count": counts.get("skipped_duplicate", ""),
                "round_promoted_count": counts.get("promoted", ""),
                "auc": result.metrics.get("auc"),
                "test_auc": result.metrics.get("auc"),
                "logloss": result.metrics.get("logloss"),
                "test_logloss": result.metrics.get("logloss"),
                "loss": result.metrics.get("loss"),
                "elapsed_sec": result.elapsed_sec,
                "artifact_dir": result.artifact_dir,
                "error": result.error or "",
            }
            row.update(_metric_report_columns(result.metrics))
            rows.append(row)
        rows.extend(_round_best_report_rows(round_groups, round_best, round_counts, auc_curve))
        rows.extend(_auc_curve_report_rows(auc_curve, round_best))
        rows.extend(_fingerprint_summary_rows(round_groups))
        rows.append(_status_overview_row(results))
        return rows

    def _build_round_history_rows(self, results: list[CandidateResult]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        round_groups = _round_result_groups(results)
        round_counts = {round_idx: _round_space_counts(group) for round_idx, group in round_groups.items()}
        round_best = {round_idx: _best_result_by_validation_auc(group) for round_idx, group in round_groups.items()}
        auc_curve = _validation_auc_curve(results, round_best)
        dedupe_by_round = {int(record["round_idx"]): record for record in self.deduplication_records if "round_idx" in record}
        adaptive_by_round = {
            int(record["round_idx"]): record for record in self.adaptive_budget_state["records"] if "round_idx" in record
        }

        baseline = next((result for result in results if result.candidate_id == "baseline"), None)
        if baseline is not None:
            baseline_auc = _validation_auc(baseline)
            baseline_row = {
                "row_type": "baseline",
                "round_idx": "baseline",
                "generation": "baseline",
                "status": baseline.status,
                "candidate_id": baseline.candidate_id,
                "candidate_description": baseline.rationale,
                "validation_best_auc": baseline_auc if baseline_auc is not None else "",
                "test_auc": baseline.metrics.get("auc", ""),
                "test_logloss": baseline.metrics.get("logloss", ""),
                "loss": baseline.metrics.get("loss", ""),
                "elapsed_sec": baseline.elapsed_sec,
                "best_candidate_id_so_far": baseline.candidate_id if baseline_auc is not None else "",
                "best_validation_auc_so_far": baseline_auc if baseline_auc is not None else "",
                "error": baseline.error or "",
            }
            baseline_row.update(_metric_report_columns(baseline.metrics))
            rows.append(baseline_row)

        for round_idx in sorted(round_groups):
            group = round_groups[round_idx]
            counts = round_counts.get(round_idx, {})
            best = round_best.get(round_idx)
            curve = auc_curve.get(round_idx, {})
            dedupe = dedupe_by_round.get(round_idx, {})
            adaptive = adaptive_by_round.get(round_idx, {})
            survivors = [result.candidate_id for result in group if result.status == "survivor"]
            promoted = [result.candidate_id for result in group if (result.promotion_result or {}).get("promoted")]
            failed = [result.candidate_id for result in group if result.status == "failed" or result.error]
            skipped = [result.candidate_id for result in group if result.status == "skipped_duplicate"]
            row = {
                "row_type": "round",
                "round_idx": round_idx,
                "generation": _ordinal(round_idx + 1),
                "status": "completed",
                "generated_candidates": dedupe.get("generated_candidates", len(group)),
                "unique_candidates": dedupe.get("unique_candidates", counts.get("total", 0) - counts.get("skipped_duplicate", 0)),
                "skipped_duplicates": dedupe.get("skipped_duplicates", counts.get("skipped_duplicate", 0)),
                "round_total_candidates": counts.get("total", ""),
                "round_skill_candidates": counts.get("skill_space", ""),
                "round_code_candidates": counts.get("code_space", ""),
                "round_space_mix": _round_space_mix_label(counts),
                "round_failed_count": counts.get("failed", ""),
                "round_skipped_count": counts.get("skipped_duplicate", ""),
                "round_skipped_duplicate_count": counts.get("skipped_duplicate", ""),
                "round_promoted_count": counts.get("promoted", ""),
                "candidate_budget": adaptive.get("candidate_budget", ""),
                "skill_target": adaptive.get("skill_target", ""),
                "code_target": adaptive.get("code_target", ""),
                "stagnant_rounds_before_round": adaptive.get("stagnant_rounds_before_round", ""),
                "stagnant_rounds_after_round": adaptive.get("stagnant_rounds_after_round", ""),
                "global_best_before_round": adaptive.get("global_best_before_round", ""),
                "global_best_after_round": adaptive.get("global_best_after_round", ""),
                "improvement": adaptive.get("improvement", ""),
                "improved": adaptive.get("improved", ""),
                "round_best_candidate_id": best.candidate_id if best else "",
                "round_best_validation_auc": _validation_auc(best) if best else "",
                "round_best_test_auc": best.metrics.get("auc", "") if best else "",
                "round_best_test_logloss": best.metrics.get("logloss", "") if best else "",
                "best_candidate_id_so_far": curve.get("best_candidate_id_so_far", ""),
                "best_validation_auc_so_far": curve.get("best_validation_auc_so_far", ""),
                "survivor_candidate_ids": ",".join(survivors),
                "promoted_candidate_ids": ",".join(promoted),
                "failed_candidate_ids": ",".join(failed),
                "skipped_duplicate_candidate_ids": ",".join(skipped),
                "status_counts": json.dumps(_fingerprint_counts(result.status for result in group), sort_keys=True),
                "elapsed_sec": round(sum(float(result.elapsed_sec or 0.0) for result in group), 4),
            }
            if best is not None:
                row.update({f"round_best_{key}": value for key, value in _metric_report_columns(best.metrics).items()})
            rows.append(row)
        return rows

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        with (self.output_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    def _cleanup_temporary_staging(self) -> None:
        if self._temporary_staging is None:
            return
        self._temporary_staging.cleanup()
        self._temporary_staging = None


def _train_candidate_worker(payload: dict[str, Any]) -> CandidateResult:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    gpu_id = str(payload["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config: CTRWorkflowConfig = payload["config"]
    if _uses_cuda(config.training.device):
        config.training.device = "cuda:0"
    bundle = prepare_movielens_ctr_data(config.dataset, config.training)
    trainer = CTRGenomeTrainer(config.training)
    spec: CandidateSpec = payload["spec"]
    return _train_candidate_artifact(
        candidate_id=spec.candidate_id,
        genome=spec.genome,
        parent_genome_id=spec.parent_genome_id,
        mutation_type=spec.mutation_type,
        initial_status="candidate",
        rationale=spec.rationale,
        evolution_space=spec.evolution_space,
        operation=spec.operation,
        trainer=trainer,
        bundle=bundle,
        output_dir=Path(payload["output_dir"]),
        training_config_hash=payload.get("training_config_hash", ""),
        genome_config_hash=payload.get("genome_config_hash", ""),
        hparam_fingerprint=payload.get("hparam_fingerprint", ""),
        hparam_changes=dict(payload.get("hparam_changes") or {}),
        architecture_fingerprint=spec.architecture_fingerprint,
        generated_skill_id=spec.generated_skill_id,
        staged_code_path=spec.staged_code_path,
        staged_skill_card_path=spec.staged_skill_card_path,
        proposal_id=spec.proposal_id,
        structural_scope=spec.structural_scope,
    )


def _train_candidate_artifact(
    *,
    candidate_id: str,
    genome: SkillGenome,
    parent_genome_id: str | None,
    mutation_type: str,
    initial_status: str,
    rationale: str,
    evolution_space: str,
    operation: str,
    trainer: CTRGenomeTrainer,
    bundle: CTRDatasetBundle,
    output_dir: str | Path,
    training_config_hash: str = "",
    genome_config_hash: str = "",
    hparam_fingerprint: str = "",
    hparam_changes: dict[str, Any] | None = None,
    architecture_fingerprint: str | None = None,
    generated_skill_id: str | None = None,
    staged_code_path: str | None = None,
    staged_skill_card_path: str | None = None,
    proposal_id: str | None = None,
    structural_scope: str = "",
) -> CandidateResult:
    started = time.time()
    artifact_dir = Path(output_dir) / candidate_id
    try:
        skill_library = _skill_library_for_candidate(staged_skill_card_path, genome=genome)
        GenomeVerifier(skill_library=skill_library).assert_valid(genome)
        metrics = trainer.fit_and_evaluate(genome, bundle, artifact_dir, skill_library=skill_library, write_artifacts=False)
        status = initial_status
        error = None
    except Exception as exc:
        metrics = {}
        status = "failed"
        error = f"{exc.__class__.__name__}: {exc}"
    return CandidateResult(
        candidate_id=candidate_id,
        genome_id=genome.metadata.genome_id,
        parent_genome_id=parent_genome_id,
        mutation_type=mutation_type,
        status=status,
        metrics=metrics,
        artifact_dir="",
        genome_path="",
        elapsed_sec=round(time.time() - started, 4),
        evolution_space=evolution_space,
        operation=operation,
        rationale=rationale,
        architecture_fingerprint=architecture_fingerprint or _safe_architecture_fingerprint(genome, fallback=candidate_id),
        structure_category=_structure_category(evolution_space, operation, mutation_type),
        hparam_category="fixed_training_config",
        training_config_hash=training_config_hash,
        genome_config_hash=genome_config_hash,
        hparam_fingerprint=hparam_fingerprint,
        hparam_changes=dict(hparam_changes or {}),
        generated_skill_id=generated_skill_id,
        staged_code_path=staged_code_path,
        staged_skill_card_path=staged_skill_card_path,
        proposal_id=proposal_id,
        structural_scope=structural_scope,
        error=error,
        genome=genome,
    )


def _memory_record_from_result(
    result: CandidateResult,
    failure_modes: list[str],
    initial_status: str,
    *,
    task_type: str = "ctr",
) -> EvolutionRecord:
    return EvolutionRecord(
        parent_genome_id=result.parent_genome_id,
        child_genome_id=result.genome_id,
        mutation_type=result.mutation_type,
        status="failed" if result.error else ("success" if initial_status == "baseline" else "validated"),
        validation_results=(
            {
                "error": result.error,
                "evolution_space": result.evolution_space,
                "operation": result.operation,
                "structure_category": result.structure_category,
                "architecture_fingerprint": result.architecture_fingerprint,
                "duplicate_of": result.duplicate_of,
                "dedup_status": result.dedup_status,
                "training_config_hash": result.training_config_hash,
                "genome_config_hash": result.genome_config_hash,
                "hparam_fingerprint": result.hparam_fingerprint,
                "hparam_changes": result.hparam_changes,
                "generated_skill_id": result.generated_skill_id,
                "proposal_id": result.proposal_id,
                "structural_scope": result.structural_scope,
                "promotion_result": result.promotion_result,
            }
            if result.error
            else {
                "trained": initial_status != "skipped_duplicate",
                "evolution_space": result.evolution_space,
                "operation": result.operation,
                "structure_category": result.structure_category,
                "architecture_fingerprint": result.architecture_fingerprint,
                "duplicate_of": result.duplicate_of,
                "dedup_status": result.dedup_status,
                "training_config_hash": result.training_config_hash,
                "genome_config_hash": result.genome_config_hash,
                "hparam_fingerprint": result.hparam_fingerprint,
                "hparam_changes": result.hparam_changes,
                "generated_skill_id": result.generated_skill_id,
                "proposal_id": result.proposal_id,
                "structural_scope": result.structural_scope,
                "promotion_result": result.promotion_result,
            }
        ),
        evaluation_metrics=result.metrics,
        failure_mode=",".join(failure_modes),
        rationale=result.rationale,
        artifact_paths=_artifact_paths_for_result(result),
        task_type=task_type,
    )


def _artifact_paths_for_result(result: CandidateResult) -> dict[str, str]:
    paths: dict[str, str] = {}
    if result.artifact_dir:
        paths["artifact_dir"] = result.artifact_dir
    if result.genome_path:
        paths["genome"] = result.genome_path
    if result.staged_code_path:
        paths["staged_code"] = result.staged_code_path
    if result.staged_skill_card_path:
        paths["staged_skill_card"] = result.staged_skill_card_path
    return paths


def _write_genome_structure_markdown(path: Path, genome: SkillGenome, *, result: CandidateResult, rank: int) -> None:
    lines = [
        f"# {result.candidate_id}",
        "",
        f"- survivor_rank: `{rank}`",
        f"- genome_id: `{result.genome_id}`",
        f"- parent_genome_id: `{result.parent_genome_id or ''}`",
        f"- mutation_type: `{result.mutation_type}`",
        f"- structural_scope: `{result.structural_scope}`",
        f"- status: `{result.status}`",
        f"- validation_best_auc: `{result.metrics.get('validation_best_auc', '')}`",
        f"- test_auc: `{result.metrics.get('auc', '')}`",
        f"- test_logloss: `{result.metrics.get('logloss', '')}`",
        f"- architecture_fingerprint: `{result.architecture_fingerprint}`",
        "",
    ]
    try:
        fusion = genome.get_node("fusion")
        lines.extend(
            [
                "## Fusion",
                "",
                f"- inputs: `{fusion.params.get('input_keys', fusion.input_keys)}`",
                f"- output: `{fusion.params.get('output_key', fusion.output_keys[0] if fusion.output_keys else '')}`",
                "",
            ]
        )
    except Exception:
        pass
    try:
        prediction = genome.get_node("prediction")
        loss = genome.get_node("loss")
        lines.extend(
            [
                "## Prediction And Loss",
                "",
                f"- prediction input: `{prediction.params.get('input_key', prediction.input_keys[0] if prediction.input_keys else '')}`",
                f"- loss logits_key: `{loss.params.get('logits_key', '')}`",
                "",
            ]
        )
    except Exception:
        pass
    lines.extend(["## Nodes", "", "| node_id | skill_id | inputs | outputs | params |", "|---|---|---|---|---|"])
    for node in genome.nodes:
        params = json.dumps(node.params, sort_keys=True, default=str).replace("|", "\\|")
        lines.append(f"| `{node.node_id}` | `{node.skill_id}` | `{node.input_keys}` | `{node.output_keys}` | `{params}` |")
    lines.extend(["", "## Edges", "", "| src | dst | src_output | dst_input |", "|---|---|---|---|"])
    for edge in genome.edges:
        lines.append(f"| `{edge.src_node_id}` | `{edge.dst_node_id}` | `{edge.src_output_key}` | `{edge.dst_input_key}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _round_result_groups(results: list[CandidateResult]) -> dict[int, list[CandidateResult]]:
    groups: dict[int, list[CandidateResult]] = {}
    for result in results:
        round_idx = _round_idx_from_candidate_id(result.candidate_id)
        if round_idx is None:
            continue
        groups.setdefault(round_idx, []).append(result)
    return dict(sorted(groups.items()))


def _candidate_slots(round_groups: dict[int, list[CandidateResult]]) -> dict[str, int]:
    slots: dict[str, int] = {}
    for group in round_groups.values():
        for idx, result in enumerate(group, start=1):
            slots[result.candidate_id] = idx
    return slots


def _round_space_counts(results: list[CandidateResult]) -> dict[str, int]:
    return {
        "total": len(results),
        "skill_space": sum(1 for result in results if result.evolution_space == "skill_space"),
        "code_space": sum(1 for result in results if result.evolution_space == "code_space"),
        "failed": sum(1 for result in results if result.status == "failed" or result.error),
        "skipped_duplicate": sum(1 for result in results if result.status == "skipped_duplicate"),
        "promoted": sum(1 for result in results if (result.promotion_result or {}).get("promoted")),
    }


def _metric_report_columns(metrics: dict[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    prefixes = (
        "validation_auc__",
        "validation_logloss__",
        "auc__",
        "logloss__",
    )
    for key, value in (metrics or {}).items():
        if key in {"validation_auc", "validation_mean_auc", "validation_loss", "mean_auc"}:
            columns[key] = value
            continue
        if any(str(key).startswith(prefix) for prefix in prefixes):
            columns[key] = value
    return columns


def _round_space_mix_label(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return f"skill={counts.get('skill_space', 0)} code={counts.get('code_space', 0)}"


def _best_result_by_validation_auc(results: list[CandidateResult]) -> CandidateResult | None:
    scored = [(result, _validation_auc(result)) for result in results if result.error is None and result.status != "skipped_duplicate"]
    scored = [(result, value) for result, value in scored if value is not None and not _is_nan(value)]
    if not scored:
        return None
    return max(scored, key=lambda item: item[1])[0]


def _validation_auc(result: CandidateResult | None) -> float | None:
    if result is None:
        return None
    value = result.metrics.get("validation_best_auc")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validation_auc_curve(
    results: list[CandidateResult],
    round_best: dict[int, CandidateResult | None],
) -> dict[int | str, dict[str, Any]]:
    curve: dict[int | str, dict[str, Any]] = {}
    baseline = next((result for result in results if result.candidate_id == "baseline"), None)
    best_result = baseline if _validation_auc(baseline) is not None else None
    best_auc = _validation_auc(best_result)
    if baseline is not None:
        curve["baseline"] = {
            "best_candidate_id_so_far": best_result.candidate_id if best_result else "",
            "best_validation_auc_so_far": best_auc if best_auc is not None else "",
        }
    for round_idx in sorted(round_best):
        candidate = round_best[round_idx]
        candidate_auc = _validation_auc(candidate)
        if candidate is not None and candidate_auc is not None and (best_auc is None or candidate_auc > best_auc):
            best_result = candidate
            best_auc = candidate_auc
        curve[round_idx] = {
            "best_candidate_id_so_far": best_result.candidate_id if best_result else "",
            "best_validation_auc_so_far": best_auc if best_auc is not None else "",
        }
    return curve


def _round_best_report_rows(
    round_groups: dict[int, list[CandidateResult]],
    round_best: dict[int, CandidateResult | None],
    round_counts: dict[int, dict[str, int]],
    auc_curve: dict[int | str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_idx in sorted(round_groups):
        best = round_best.get(round_idx)
        counts = round_counts.get(round_idx, {})
        curve = auc_curve.get(round_idx, {})
        rows.append(
            {
                "row_type": "round_best",
                "round_idx": round_idx,
                "candidate_id": best.candidate_id if best else "",
                "candidate_description": best.rationale if best else "",
                "generation": _ordinal(round_idx + 1),
                "evolution_category": _candidate_evolution_category(best) if best else "",
                "mutation_type": best.mutation_type if best else "",
                "evolution_space": best.evolution_space if best else "",
                "operation": best.operation if best else "",
                "structure_category": best.structure_category if best else "",
                "architecture_fingerprint": best.architecture_fingerprint if best else "",
                "training_config_hash": best.training_config_hash if best else "",
                "genome_config_hash": best.genome_config_hash if best else "",
                "hparam_fingerprint": best.hparam_fingerprint if best else "",
                "hparam_changes": json.dumps(best.hparam_changes if best else {}, sort_keys=True),
                "generated_skill_id": best.generated_skill_id if best else "",
                "proposal_id": best.proposal_id if best else "",
                "structural_scope": best.structural_scope if best else "",
                "promotion_status": ((best.promotion_result or {}).get("metadata") or {}).get("promotion_status", "") if best else "",
                "status": best.status if best else "no_valid_candidate",
                "validation_best_auc": _validation_auc(best) if best else "",
                "round_best_candidate_id": best.candidate_id if best else "",
                "round_best_validation_auc": _validation_auc(best) if best else "",
                "best_candidate_id_so_far": curve.get("best_candidate_id_so_far", ""),
                "best_validation_auc_so_far": curve.get("best_validation_auc_so_far", ""),
                "round_total_candidates": counts.get("total", ""),
                "round_skill_candidates": counts.get("skill_space", ""),
                "round_code_candidates": counts.get("code_space", ""),
                "round_space_mix": _round_space_mix_label(counts),
                "round_failed_count": counts.get("failed", ""),
                "round_skipped_count": counts.get("skipped_duplicate", ""),
                "round_skipped_duplicate_count": counts.get("skipped_duplicate", ""),
                "round_promoted_count": counts.get("promoted", ""),
                "auc": best.metrics.get("auc") if best else "",
                "test_auc": best.metrics.get("auc") if best else "",
                "logloss": best.metrics.get("logloss") if best else "",
                "test_logloss": best.metrics.get("logloss") if best else "",
                "loss": best.metrics.get("loss") if best else "",
                "elapsed_sec": best.elapsed_sec if best else "",
                "error": best.error if best and best.error else "",
            }
        )
    return rows


def _auc_curve_report_rows(
    auc_curve: dict[int | str, dict[str, Any]],
    round_best: dict[int, CandidateResult | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "baseline" in auc_curve:
        rows.append(
            {
                "row_type": "auc_curve",
                "round_idx": "baseline",
                "generation": "baseline",
                "best_candidate_id_so_far": auc_curve["baseline"].get("best_candidate_id_so_far", ""),
                "best_validation_auc_so_far": auc_curve["baseline"].get("best_validation_auc_so_far", ""),
            }
        )
    for round_idx in sorted(round_best):
        best = round_best.get(round_idx)
        rows.append(
            {
                "row_type": "auc_curve",
                "round_idx": round_idx,
                "generation": _ordinal(round_idx + 1),
                "candidate_id": best.candidate_id if best else "",
                "round_best_candidate_id": best.candidate_id if best else "",
                "round_best_validation_auc": _validation_auc(best) if best else "",
                "best_candidate_id_so_far": auc_curve.get(round_idx, {}).get("best_candidate_id_so_far", ""),
                "best_validation_auc_so_far": auc_curve.get(round_idx, {}).get("best_validation_auc_so_far", ""),
            }
        )
    return rows


def _fingerprint_summary_rows(round_groups: dict[int, list[CandidateResult]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_idx, group in round_groups.items():
        architecture_counts = _fingerprint_counts(result.architecture_fingerprint for result in group)
        hparam_counts = _fingerprint_counts(result.hparam_fingerprint for result in group)
        rows.append(
            {
                "row_type": "architecture_fingerprint_summary",
                "round_idx": round_idx,
                "generation": _ordinal(round_idx + 1),
                "architecture_fingerprint_unique_count": len(architecture_counts),
                "architecture_fingerprint_counts": json.dumps(architecture_counts, sort_keys=True),
            }
        )
        rows.append(
            {
                "row_type": "hparam_fingerprint_summary",
                "round_idx": round_idx,
                "generation": _ordinal(round_idx + 1),
                "hparam_fingerprint_unique_count": len(hparam_counts),
                "hparam_fingerprint_counts": json.dumps(hparam_counts, sort_keys=True),
                "training_config_hash": group[0].training_config_hash if group else "",
                "genome_config_hash": group[0].genome_config_hash if group else "",
                "hparam_fingerprint": group[0].hparam_fingerprint if group else "",
                "hparam_changes": json.dumps(group[0].hparam_changes if group else {}, sort_keys=True),
            }
        )
    return rows


def _fingerprint_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _status_overview_row(results: list[CandidateResult]) -> dict[str, Any]:
    promoted = [result for result in results if (result.promotion_result or {}).get("promoted")]
    failed = [result for result in results if result.status == "failed" or result.error]
    skipped = [result for result in results if result.status == "skipped_duplicate"]
    return {
        "row_type": "status_overview",
        "round_idx": "all",
        "candidate_id": "overview",
        "round_total_candidates": sum(1 for result in results if result.candidate_id != "baseline"),
        "total_failed_count": len(failed),
        "total_skipped_count": len(skipped),
        "total_skipped_duplicate_count": len(skipped),
        "total_promoted_count": len(promoted),
        "failed_candidate_ids": ",".join(result.candidate_id for result in failed),
        "skipped_candidate_ids": ",".join(result.candidate_id for result in skipped),
        "promoted_candidate_ids": ",".join(result.candidate_id for result in promoted),
        "status_counts": json.dumps(_fingerprint_counts(result.status for result in results), sort_keys=True),
    }


def _skill_library_for_candidate(staged_skill_card_path: str | None, genome: SkillGenome | None = None) -> SkillLibrary:
    library = SkillLibrary.from_repo(include_generated=True)
    for card_path in _candidate_skill_card_paths(staged_skill_card_path, genome):
        library.add_card(card_path)
    return library


def _candidate_skill_card_paths(staged_skill_card_path: str | None, genome: SkillGenome | None) -> list[str]:
    paths: list[str] = []
    if staged_skill_card_path:
        paths.append(staged_skill_card_path)
    if genome is not None:
        for node in genome.nodes:
            for key in ("persisted_skill_card_path", "staged_skill_card_path", "runtime_skill_card_path"):
                value = node.metadata.get(key)
                if value:
                    paths.append(str(value))
    return list(dict.fromkeys(paths))


def _path_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _annotate_generated_skill_hash(genome: SkillGenome, skill_id: str, code_hash: str) -> None:
    for node in genome.nodes:
        if node.skill_id == skill_id:
            node.metadata = {**node.metadata, "generated_code_hash": code_hash}


def _annotate_generated_skill_artifacts(
    genome: SkillGenome,
    skill_id: str,
    *,
    staged_code_path: str | None = None,
    staged_skill_card_path: str | None = None,
    persisted_code_path: str | None = None,
    persisted_skill_card_path: str | None = None,
) -> None:
    metadata = {
        key: value
        for key, value in {
            "staged_code_path": staged_code_path,
            "staged_skill_card_path": staged_skill_card_path,
            "persisted_code_path": persisted_code_path,
            "persisted_skill_card_path": persisted_skill_card_path,
            "runtime_code_path": persisted_code_path or staged_code_path,
            "runtime_skill_card_path": persisted_skill_card_path or staged_skill_card_path,
        }.items()
        if value
    }
    if not metadata:
        return
    for node in genome.nodes:
        if node.skill_id == skill_id:
            node.metadata = {**node.metadata, **metadata}


def _rewrite_generated_skill_reference(
    genome: SkillGenome,
    old_skill_id: str,
    new_skill_id: str,
    *,
    persisted_code_path: str | None = None,
    persisted_skill_card_path: str | None = None,
) -> None:
    if old_skill_id == new_skill_id:
        _annotate_generated_skill_artifacts(
            genome,
            old_skill_id,
            persisted_code_path=persisted_code_path,
            persisted_skill_card_path=persisted_skill_card_path,
        )
        return
    for node in genome.nodes:
        if node.skill_id != old_skill_id:
            continue
        node.metadata = {
            **node.metadata,
            "origin_skill_id": old_skill_id,
            "persisted_skill_id": new_skill_id,
            **{
                key: value
                for key, value in {
                    "persisted_code_path": persisted_code_path,
                    "persisted_skill_card_path": persisted_skill_card_path,
                    "runtime_code_path": persisted_code_path,
                    "runtime_skill_card_path": persisted_skill_card_path,
                }.items()
                if value
            },
        }
        node.skill_id = new_skill_id
        if node.skill_name == old_skill_id:
            node.skill_name = new_skill_id


def _register_promoted_generated_skill(skill_id: str | None, skill_card_path: str | None) -> None:
    if not skill_id or not skill_card_path:
        return
    try:
        library = SkillLibrary.from_repo(include_generated=True)
        library.add_card(skill_card_path)
        library.register_generated_skill(skill_id)
    except Exception:
        return


DEFAULT_SKILL_SPACE_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "add_crossnet_mix",
        "num_layers": 3,
        "low_rank": 8,
        "num_experts": 4,
        "evolution_space": "skill_space",
        "operation": "add",
    },
    {"name": "add_afm_attention", "attention_dim": 64, "evolution_space": "skill_space", "operation": "add"},
    {"name": "add_crossnet_v2", "num_layers": 3, "evolution_space": "skill_space", "operation": "add"},
    {"name": "replace_fm_with_field_sum", "evolution_space": "skill_space", "operation": "replace"},
    {"name": "remove_fm", "evolution_space": "skill_space", "operation": "replace"},
    {
        "name": "hybridize_crossnet_afm",
        "num_layers": 2,
        "attention_dim": 32,
        "evolution_space": "skill_space",
        "operation": "hybridize",
    },
    {
        "name": "add_autoint_attention",
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "evolution_space": "skill_space",
        "operation": "add",
    },
    {"name": "specialize_senet_fm", "reduction_ratio": 3, "evolution_space": "skill_space", "operation": "specialize"},
]

DEFAULT_CODE_SPACE_TEMPLATES: list[dict[str, Any]] = [
    {"name": "code_logit_calibrator", "evolution_space": "code_space", "operation": "open_ended"},
    {"name": "code_residual_gate", "hidden_dim": 32, "evolution_space": "code_space", "operation": "open_ended"},
]


LIBRARY_STRATEGIES = {"skill_library", "library", "dynamic", "hybrid"}
SKILL_GRAPH_PLANNER_STRATEGIES = {"skill_graph_planner", "skill_graph", "graph_planner", "card_graph"}

CTR_LIBRARY_SKILL_ORDER = [
    "crossnet_mix",
    "afm_attention_pooling",
    "crossnet_v2",
    "field_sum",
    "bilinear_interaction",
    "hybridize_crossnet_afm",
    "autoint_attention",
    "senet_feature_gate",
    "crossnet_v1",
    "fm_interaction",
]

CTR_LIBRARY_CODE_SKILL_ORDER = [
    "generated_logit_temperature_calibrator",
    "generated_residual_logit_gate",
]

LEGACY_TEMPLATE_TO_LIBRARY_SKILL = {
    "add_crossnet_mix": "crossnet_mix",
    "add_crossnet_v1": "crossnet_v1",
    "add_crossnet_v2": "crossnet_v2",
    "add_afm_attention": "afm_attention_pooling",
    "replace_fm_with_field_sum": "field_sum",
    "add_fm_interaction": "fm_interaction",
    "add_bilinear_interaction": "bilinear_interaction",
    "hybridize_crossnet_afm": "hybridize_crossnet_afm",
    "add_autoint_attention": "autoint_attention",
    "specialize_senet_fm": "senet_feature_gate",
    "code_logit_calibrator": "generated_logit_temperature_calibrator",
    "code_residual_gate": "generated_residual_logit_gate",
}

LIBRARY_SKILL_TO_TEMPLATE = {
    value: key for key, value in LEGACY_TEMPLATE_TO_LIBRARY_SKILL.items()
}


def _default_evolution_templates() -> list[dict[str, Any]]:
    return [dict(template) for template in DEFAULT_SKILL_SPACE_TEMPLATES]


def generate_ctr_candidate_population(parents: list[SkillGenome], config: CTRWorkflowConfig, round_idx: int = 0) -> list[CandidateSpec]:
    if not parents:
        return []
    if _uses_skill_graph_planner_strategy(config.evolution):
        specs = _generate_ctr_candidate_population_from_skill_graph_planner(parents, config, round_idx=round_idx)
        if len(specs) >= config.evolution.candidate_budget or not config.evolution.skill_graph_planner.allow_template_fallback:
            return specs[: config.evolution.candidate_budget]
        seen_ids = {spec.candidate_id for spec in specs}
        fallback_evolution = replace(config.evolution, strategy="skill_library")
        fallback_config = replace(config, evolution=fallback_evolution)
        for spec in generate_ctr_candidate_population(parents, fallback_config, round_idx=round_idx):
            if spec.evolution_space != "skill_space" or spec.candidate_id in seen_ids:
                continue
            spec.candidate_id = _dedupe_candidate_id(spec.candidate_id, seen_ids)
            specs.append(spec)
            if len(specs) >= config.evolution.candidate_budget:
                break
        return specs[: config.evolution.candidate_budget]
    if _uses_skill_library_strategy(config.evolution):
        specs = _generate_ctr_candidate_population_from_skill_library(parents, config, round_idx=round_idx)
        if len(specs) >= config.evolution.candidate_budget:
            return specs[: config.evolution.candidate_budget]
        seen_ids = {spec.candidate_id for spec in specs}
        for spec in _generate_ctr_candidate_population_from_templates(parents, config, round_idx=round_idx):
            if spec.candidate_id in seen_ids:
                continue
            specs.append(spec)
            seen_ids.add(spec.candidate_id)
            if len(specs) >= config.evolution.candidate_budget:
                break
        return specs[: config.evolution.candidate_budget]

    return _generate_ctr_candidate_population_from_templates(parents, config, round_idx=round_idx)


def _generate_ctr_candidate_population_from_templates(parents: list[SkillGenome], config: CTRWorkflowConfig, round_idx: int = 0) -> list[CandidateSpec]:
    templates = _effective_evolution_templates(config.evolution)
    specs: list[CandidateSpec] = []
    seen_ids: set[str] = set()
    for idx, template in enumerate(templates):
        parent = parents[idx % len(parents)]
        try:
            spec = _generate_candidate_from_template(parent, config, template, round_idx=round_idx)
        except Exception:
            continue
        if len(parents) > 1:
            spec.candidate_id = f"{spec.candidate_id}_p{idx % len(parents)}"
        spec.candidate_id = _dedupe_candidate_id(spec.candidate_id, seen_ids)
        specs.append(spec)
    return specs[: config.evolution.candidate_budget]


def generate_ctr_candidates(configured_parent: SkillGenome, config: CTRWorkflowConfig, round_idx: int = 0) -> list[tuple[str, SkillGenome, str, str]]:
    specs = generate_ctr_candidate_population([configured_parent], config, round_idx=round_idx)
    return [(spec.candidate_id, spec.genome, spec.mutation_type, spec.rationale) for spec in specs[: config.evolution.candidate_budget]]


def _uses_skill_library_strategy(evolution: CTREvolutionConfig) -> bool:
    return str(evolution.strategy or "").lower().replace("-", "_") in LIBRARY_STRATEGIES


def _uses_skill_graph_planner_strategy(evolution: CTREvolutionConfig) -> bool:
    return str(evolution.strategy or "").lower().replace("-", "_") in SKILL_GRAPH_PLANNER_STRATEGIES


def _candidate_targets(evolution: CTREvolutionConfig) -> tuple[int, int]:
    budget = max(0, int(evolution.candidate_budget))
    if not _code_space_quota_enabled(evolution):
        return budget, 0
    target_code = min(budget, max(0, int(round(budget * float(evolution.code_space_probability)))))
    return budget - target_code, target_code


def _adaptive_candidate_targets(
    evolution: CTREvolutionConfig,
    *,
    stagnant_rounds: int,
    active_code_candidates: Any = None,
) -> tuple[int, int]:
    base_skill, base_code = _candidate_targets(evolution)
    budget = base_skill + base_code
    config = evolution.adaptive_candidate_budget
    if budget <= 0 or not config.enabled or base_code <= 0:
        return base_skill, base_code
    base_code = int(config.base_code_candidates) if config.base_code_candidates is not None else base_code
    base_code = max(0, min(budget, base_code))
    max_code = int(config.max_code_candidates) if config.max_code_candidates is not None else budget
    max_code = max(base_code, min(budget, max_code))
    min_skill = max(0, min(budget, int(config.min_skill_candidates)))
    max_code = min(max_code, budget - min_skill)
    if max_code < base_code:
        base_code = max_code
    if stagnant_rounds < int(config.stagnation_patience):
        code_target = base_code
    else:
        escalation = stagnant_rounds - int(config.stagnation_patience) + 1
        code_target = base_code + escalation * int(config.code_step)
        if active_code_candidates is not None:
            code_target = max(code_target, int(active_code_candidates))
    code_target = max(0, min(max_code, code_target))
    skill_target = budget - code_target
    if skill_target < min_skill:
        skill_target = min_skill
        code_target = budget - skill_target
    return skill_target, code_target


def _code_space_quota_enabled(evolution: CTREvolutionConfig) -> bool:
    code_space = evolution.code_space
    if isinstance(code_space, dict):
        provider = str(code_space.get("provider", "fallback")).lower().replace("-", "_")
    else:
        provider = str(code_space.provider).lower().replace("-", "_")
    if provider not in {"fallback", "disabled", "none"}:
        return True
    return any(_template_evolution_space(template) == "code_space" for template in (evolution.templates or []))


def _reuse_promoted_generated_skills_enabled(config: CTRWorkflowConfig) -> bool:
    code_space = config.evolution.code_space
    if isinstance(code_space, dict):
        return bool(
            code_space.get(
                "reuse_promoted_generated_skills",
                code_space.get("reuse_promoted_code_skills", False),
            )
        )
    return bool(
        getattr(
            code_space,
            "reuse_promoted_generated_skills",
            getattr(code_space, "reuse_promoted_code_skills", False),
        )
    )


def _allocate_code_budget_by_parent(
    parents: list[SkillGenome],
    *,
    budget: int,
    round_idx: int = 0,
) -> list[tuple[int, SkillGenome, int]]:
    if not parents or budget <= 0:
        return []
    count = min(len(parents), budget)
    start = round_idx % len(parents)
    ordered = [(start + offset) % len(parents) for offset in range(count)]
    allocations = {idx: budget // count for idx in ordered}
    for idx in ordered[: budget % count]:
        allocations[idx] += 1
    return [(idx, parents[idx], allocations[idx]) for idx in ordered if allocations[idx] > 0]


def _next_parent_population(current: list[SkillGenome], survivors: list[CandidateResult]) -> list[SkillGenome]:
    next_population = [result.genome for result in survivors if result.genome is not None]
    return next_population or current


def _generate_promoted_code_library_candidates(
    parents: list[SkillGenome],
    config: CTRWorkflowConfig,
    *,
    round_idx: int,
    target_code: int,
) -> list[CandidateSpec]:
    if target_code <= 0:
        return []
    try:
        library = SkillLibrary.from_repo(include_generated=True)
    except Exception:
        return []
    reusable_target = min(target_code, 1)
    skill_ids = _select_ctr_library_skill_ids(library, parents, config, reusable_target, code_space=True, round_idx=round_idx)
    specs: list[CandidateSpec] = []
    seen_ids: set[str] = set()
    for idx, skill_id in enumerate(skill_ids):
        card = library.get(skill_id)
        if not _is_reusable_generated_skill_card(card):
            continue
        parent = parents[idx % len(parents)]
        try:
            spec = _generate_candidate_from_library_skill(parent, config, skill_id, round_idx=round_idx, library=library)
        except Exception:
            continue
        if len(parents) > 1:
            spec.candidate_id = f"{spec.candidate_id}_p{idx % len(parents)}"
        spec.candidate_id = _dedupe_candidate_id(spec.candidate_id, seen_ids)
        specs.append(spec)
        if len(specs) >= reusable_target:
            break
    return specs


def _generate_ctr_candidate_population_from_skill_library(parents: list[SkillGenome], config: CTRWorkflowConfig, round_idx: int = 0) -> list[CandidateSpec]:
    target_skill, target_code = _candidate_targets(config.evolution)
    try:
        library = SkillLibrary.from_repo(include_generated=True)
    except Exception:
        return []
    selected_skill_ids = _select_ctr_library_skill_ids(library, parents, config, target_skill, code_space=False, round_idx=round_idx)
    selected_code_ids = _select_ctr_library_skill_ids(library, parents, config, target_code, code_space=True, round_idx=round_idx)

    specs: list[CandidateSpec] = []
    seen_ids: set[str] = set()
    for idx, skill_id in enumerate(selected_skill_ids + selected_code_ids):
        parent = parents[idx % len(parents)]
        try:
            spec = _generate_candidate_from_library_skill(parent, config, skill_id, round_idx=round_idx, library=library)
        except Exception:
            continue
        if len(parents) > 1:
            spec.candidate_id = f"{spec.candidate_id}_p{idx % len(parents)}"
        spec.candidate_id = _dedupe_candidate_id(spec.candidate_id, seen_ids)
        specs.append(spec)
    return specs


def _generate_ctr_candidate_population_from_skill_graph_planner(
    parents: list[SkillGenome],
    config: CTRWorkflowConfig,
    round_idx: int = 0,
) -> list[CandidateSpec]:
    target_skill, _ = _candidate_targets(config.evolution)
    target_skill = min(target_skill, config.evolution.candidate_budget)
    if target_skill <= 0:
        return []
    try:
        library = SkillLibrary.from_repo(include_generated=True)
    except Exception:
        return []
    specs: list[CandidateSpec] = []
    seen_ids: set[str] = set()
    for parent_idx, parent in enumerate(parents):
        remaining = target_skill - len(specs)
        if remaining <= 0:
            break
        planner = SkillGraphSearchPlanner(
            skill_library=library,
            config=config.evolution.skill_graph_planner,
            seed=int(config.training.seed) + round_idx * 1009 + parent_idx,
        )
        try:
            planned = planner.plan(
                parent,
                failure_modes=config.evolution.failure_modes,
                budget=remaining,
                round_idx=round_idx,
            )
        except Exception:
            continue
        for candidate in planned:
            try:
                first_action = candidate.actions[0] if candidate.actions else None
                skill_label = _slug_for_candidate_id(first_action.skill_id if first_action else candidate.operation)
                candidate_id = f"round{round_idx}_skill_graph_{candidate.operation}_{skill_label}"
                if len(parents) > 1:
                    candidate_id = f"{candidate_id}_p{parent_idx}"
                candidate_id = _dedupe_candidate_id(candidate_id, seen_ids)
                specs.append(
                    CandidateSpec(
                        candidate_id=candidate_id,
                        genome=candidate.genome,
                        parent_genome_id=parent.metadata.genome_id,
                        mutation_type=candidate.mutation_type,
                        rationale=f"{candidate.rationale} (skill graph score={candidate.score:.4f})",
                        evolution_space="skill_space",
                        operation=candidate.operation,
                        architecture_fingerprint=candidate.architecture_fingerprint,
                    )
                )
            except Exception:
                continue
            if len(specs) >= target_skill:
                break
    return specs[:target_skill]


def _select_ctr_library_skill_ids(
    library: SkillLibrary,
    parents: list[SkillGenome],
    config: CTRWorkflowConfig,
    target: int,
    *,
    code_space: bool,
    round_idx: int = 0,
) -> list[str]:
    if target <= 0:
        return []
    ordered_ids = CTR_LIBRARY_CODE_SKILL_ORDER if code_space else CTR_LIBRARY_SKILL_ORDER
    scored: list[tuple[int, str]] = []
    for skill_id in ordered_ids:
        if skill_id == "hybridize_crossnet_afm":
            if code_space or not (library.has("crossnet_v2") and library.has("afm_attention_pooling")):
                continue
            scored.append((80, skill_id))
            continue
        if not library.has(skill_id):
            continue
        try:
            card = library.get(skill_id)
            if not _is_ctr_library_candidate_compatible(card, parents, config=config, code_space=code_space):
                continue
            scored.append((_score_ctr_library_candidate(card, parents, config, code_space=code_space), skill_id))
        except Exception:
            continue

    seen = {skill_id for _, skill_id in scored}
    reuse_generated = _reuse_promoted_generated_skills_enabled(config)
    for skill_id in library.list_skill_ids():
        if skill_id in seen:
            continue
        try:
            card = library.get(skill_id)
            if _is_reusable_generated_skill_card(card) and not reuse_generated:
                continue
            if code_space != _is_ctr_code_space_library_skill(card):
                continue
            if not _has_ctr_library_builder_for_card(card):
                continue
            if _is_ctr_library_candidate_compatible(card, parents, config=config, code_space=code_space):
                scored.append((_score_ctr_library_candidate(card, parents, config, code_space=code_space), skill_id))
        except Exception:
            continue

    if code_space:
        scored.sort(key=lambda item: (-item[0], ordered_ids.index(item[1]) if item[1] in ordered_ids else len(ordered_ids), item[1]))
    else:
        scored.sort(key=lambda item: (-item[0], ordered_ids.index(item[1]) if item[1] in ordered_ids else len(ordered_ids), item[1]))
        scored = _rotate_reusable_generated_skills(scored, library, round_idx=round_idx)
        scored = _ensure_non_generated_skill_quota(scored, library, target=target)
    return [skill_id for _, skill_id in scored[:target]]


def _rotate_reusable_generated_skills(scored: list[tuple[int, str]], library: SkillLibrary, *, round_idx: int) -> list[tuple[int, str]]:
    reusable: list[tuple[int, str]] = []
    other: list[tuple[int, str]] = []
    for item in scored:
        _, skill_id = item
        try:
            card = library.get(skill_id)
        except Exception:
            other.append(item)
            continue
        if _is_reusable_generated_skill_card(card):
            reusable.append(item)
        else:
            other.append(item)
    if len(reusable) <= 1:
        return scored
    reusable.sort(key=lambda item: (-item[0], item[1]))
    shift = round_idx % len(reusable)
    rotated = reusable[shift:] + reusable[:shift]
    other.sort(key=lambda item: (-item[0], item[1]))
    return rotated + other


def _ensure_non_generated_skill_quota(scored: list[tuple[int, str]], library: SkillLibrary, *, target: int) -> list[tuple[int, str]]:
    if target <= 1 or len(scored) <= target:
        return scored
    selected = scored[:target]
    if any(not _is_reusable_generated_skill_id(library, skill_id) for _, skill_id in selected):
        return scored
    replacement_idx = None
    for idx, (_, skill_id) in enumerate(scored[target:], start=target):
        if not _is_reusable_generated_skill_id(library, skill_id):
            replacement_idx = idx
            break
    if replacement_idx is None:
        return scored
    replacement = scored[replacement_idx]
    return selected[:-1] + [replacement] + scored[target:replacement_idx] + selected[-1:] + scored[replacement_idx + 1 :]


def _is_reusable_generated_skill_id(library: SkillLibrary, skill_id: str) -> bool:
    try:
        return _is_reusable_generated_skill_card(library.get(skill_id))
    except Exception:
        return False


def _has_ctr_library_builder(skill_id: str) -> bool:
    return skill_id in {
        "crossnet_mix",
        "crossnet_v1",
        "crossnet_v2",
        "afm_attention_pooling",
        "autoint_attention",
        "field_sum",
        "fm_interaction",
        "bilinear_interaction",
        "senet_feature_gate",
        "generated_logit_temperature_calibrator",
        "generated_residual_logit_gate",
    }


def _has_ctr_library_builder_for_card(card: SkillCard) -> bool:
    return _has_ctr_library_builder(card.skill_id) or _is_reusable_generated_skill_card(card)


def _is_ctr_library_candidate_compatible(
    card: SkillCard,
    parents: list[SkillGenome],
    *,
    config: CTRWorkflowConfig,
    code_space: bool,
) -> bool:
    if code_space:
        if not _is_ctr_code_space_library_skill(card) or not _skill_card_has_loadable_implementation(card):
            return False
        if card.skill_id in CTR_LIBRARY_CODE_SKILL_ORDER:
            return True
        requirements = set(_resolved_generated_input_keys(card, parents[0]))
        if not requirements:
            return True
        return any(requirements <= _available_ctr_tensor_keys(parent) for parent in parents)
    if _is_reusable_generated_skill_card(card):
        if not _reuse_promoted_generated_skills_enabled(config) or not _skill_card_has_loadable_implementation(card):
            return False
        if not _generated_skill_portability_allows_reuse(card):
            return False
        requirements = set(_resolved_generated_input_keys(card, parents[0]))
        if not requirements:
            return any(_generated_skill_shape_compatible(card, parent, config) for parent in parents)
        return any(requirements <= _available_ctr_tensor_keys(parent) and _generated_skill_shape_compatible(card, parent, config) for parent in parents)
    if not _has_ctr_library_builder(card.skill_id):
        return False
    task_types = {task.lower() for task in card.task_types}
    retrieval_tasks = {str(task).lower() for task in (card.manifest.get("retrieval") or {}).get("task_types", [])}
    if task_types or retrieval_tasks:
        if not ((task_types | retrieval_tasks) & {"ctr", "ranking"}):
            return False
    requirements = set(card.input_keys)
    if not requirements:
        return True
    for parent in parents:
        available = _available_ctr_tensor_keys(parent)
        if requirements <= available:
            return True
        if card.skill_id in {"field_sum", "bilinear_interaction", "senet_feature_gate"} and "field_embeddings" in available:
            return True
        if card.skill_id in {"crossnet_mix", "crossnet_v1", "crossnet_v2"} and "flat_embeddings" in available:
            return True
    return False


def _skill_card_has_loadable_implementation(card: SkillCard) -> bool:
    if card.skill_id in SKILL_REGISTRY:
        return True
    implementation_path = card.implementation_path
    if not implementation_path:
        return False
    path = Path(implementation_path)
    if not path.is_absolute():
        if card.path is not None:
            for base in [card.path.parent, card.path.parent.parent]:
                candidate = base / path
                if candidate.exists():
                    return True
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / path
    return path.exists()


def _generated_skill_portability_allows_reuse(card: SkillCard) -> bool:
    portability = card.manifest.get("portability") or {}
    if not isinstance(portability, dict):
        return True
    if portability.get("reuse_enabled") is False:
        return False
    scope = str(portability.get("scope") or "").lower()
    return scope not in {"dataset_specific", "local_only"}


def _generated_skill_shape_compatible(card: SkillCard, parent: SkillGenome, config: CTRWorkflowConfig) -> bool:
    current = _generated_skill_runtime_shape(parent, config)
    if current["num_fields"] < _as_int(_portability_constraint(card, "min_num_fields"), default=1):
        return False
    if current["num_fields"] > _as_int(_portability_constraint(card, "max_num_fields"), default=current["num_fields"]):
        return False
    params = _generated_fragment_params(card)
    for name, current_value in [
        ("num_fields", current["num_fields"]),
        ("embedding_dim", current["embedding_dim"]),
        ("input_dim", current["flat_input_dim"]),
    ]:
        if not _static_shape_value_compatible(params.get(name), current_value):
            return False
    for item in _signature_items_for_shape_check(card.manifest.get("input_signature") or []):
        shape = item.get("shape") or []
        name = str(item.get("name") or "")
        if not _shape_spec_compatible(shape, name=name, current=current):
            return False
    return True


def _generated_skill_runtime_shape(parent: SkillGenome, config: CTRWorkflowConfig) -> dict[str, int]:
    embedding_dim = _embedding_dim(parent, config)
    try:
        num_fields = _num_fields(parent)
    except Exception:
        num_fields = max(1, _flat_input_dim(parent, config) // max(1, embedding_dim))
    return {
        "num_fields": int(num_fields),
        "embedding_dim": int(embedding_dim),
        "flat_input_dim": int(num_fields) * int(embedding_dim),
    }


def _portability_constraint(card: SkillCard, key: str) -> Any:
    portability = card.manifest.get("portability") or {}
    if not isinstance(portability, dict):
        return None
    constraints = portability.get("constraints") or {}
    if not isinstance(constraints, dict):
        return None
    return constraints.get(key)


def _static_shape_value_compatible(value: Any, current_value: int) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        value = value.strip()
        if not value or value.startswith("${"):
            return True
        if value in {"num_fields", "embedding_dim", "input_dim", "flat_input_dim"}:
            return True
        if not value.isdigit():
            return True
    try:
        return int(value) == int(current_value)
    except Exception:
        return True


def _signature_items_for_shape_check(signature: Any) -> list[dict[str, Any]]:
    if not isinstance(signature, list):
        return []
    return [item for item in signature if isinstance(item, dict)]


def _shape_spec_compatible(shape: Any, *, name: str, current: dict[str, int]) -> bool:
    if not isinstance(shape, list):
        return True
    if len(shape) == 3 and "field" in name.lower():
        return (
            _shape_dim_compatible(shape[1], current["num_fields"])
            and _shape_dim_compatible(shape[2], current["embedding_dim"])
        )
    if len(shape) == 2 and ("flat" in name.lower() or "input" in name.lower()):
        return _shape_dim_compatible(shape[1], current["flat_input_dim"])
    return True


def _shape_dim_compatible(dim: Any, current_value: int) -> bool:
    if isinstance(dim, str):
        dim = dim.strip()
        if dim in {"num_fields", "embedding_dim", "input_dim", "flat_input_dim"}:
            return True
        if dim.startswith("${"):
            return True
        if not dim.isdigit():
            return True
    try:
        return int(dim) == int(current_value)
    except Exception:
        return True


def _is_ctr_code_space_library_skill(card: SkillCard) -> bool:
    return card.skill_id in CTR_LIBRARY_CODE_SKILL_ORDER


def _is_reusable_generated_skill_card(card: SkillCard) -> bool:
    manifest = card.manifest
    if card.skill_id in CTR_LIBRARY_CODE_SKILL_ORDER:
        return False
    aliases = {str(item).lower() for item in (manifest.get("retrieval") or {}).get("aliases", [])}
    return bool(
        manifest.get("created_from_open_ended_evolution")
        or str(manifest.get("promotion_status", "")).lower() in {"promoted", "validated"}
        or "open_ended_evolution" in aliases
    )


def _score_ctr_library_candidate(card: SkillCard, parents: list[SkillGenome], config: CTRWorkflowConfig, *, code_space: bool) -> int:
    score = 10
    if code_space:
        score += 100
        return score
    if _is_reusable_generated_skill_card(card):
        score += 26
        metrics = card.manifest.get("candidate_metrics") or (card.manifest.get("metadata") or {}).get("candidate_metrics") or {}
        try:
            score += min(20, max(0, int(round(float(metrics.get("validation_best_auc", 0.0)) * 10))))
        except Exception:
            pass
    manifest = card.manifest
    retrieval = manifest.get("retrieval") or {}
    composition = manifest.get("composition") or {}
    text = " ".join(
        str(item)
        for item in [
            card.skill_id,
            card.category,
            retrieval.get("architecture_roles", []),
            retrieval.get("objectives", []),
            retrieval.get("model_families", []),
            manifest.get("inductive_bias", []),
            manifest.get("failure_signatures", []),
        ]
    ).lower()
    for failure_mode in config.evolution.failure_modes:
        for token in str(failure_mode).lower().replace("_", " ").split():
            if token in text:
                score += 2
    parent_skill_ids = {node.skill_id for parent in parents for node in parent.nodes}
    upstream = set(str(item) for item in composition.get("common_upstream", []) or [])
    downstream = set(str(item) for item in composition.get("common_downstream", []) or [])
    if upstream & parent_skill_ids:
        score += 8
    if downstream & {"additive_fusion", "binary_ctr_head", "linear_logit", "sigmoid_prediction"}:
        score += 4
    if card.skill_id in {"crossnet_mix", "crossnet_v2", "bilinear_interaction", "afm_attention_pooling", "autoint_attention"}:
        score += 12
    if card.skill_id in {"field_sum", "senet_feature_gate"}:
        score += 8
    return score


def _available_ctr_tensor_keys(genome: SkillGenome) -> set[str]:
    return set(genome.constraints.required_inputs) | genome.produced_keys() | {"sparse_features", "labels"}


def _generate_candidate_from_library_skill(parent: SkillGenome, config: CTRWorkflowConfig, skill_id: str, round_idx: int, library: SkillLibrary | None = None) -> CandidateSpec:
    library = library or SkillLibrary.from_repo(include_generated=True)
    card = library.get(skill_id) if library.has(skill_id) else None
    template = _template_for_library_skill(config.evolution, skill_id, card=card)
    if skill_id in {"crossnet_mix", "crossnet_v1", "crossnet_v2"}:
        candidate_id, genome, mutation_type, rationale = _add_crossnet_candidate(parent, config, template, skill_id=skill_id, round_idx=round_idx)
    elif skill_id == "afm_attention_pooling":
        candidate_id, genome, mutation_type, rationale = _add_afm_attention_candidate(parent, config, template, round_idx=round_idx)
    elif skill_id == "autoint_attention":
        candidate_id, genome, mutation_type, rationale = _add_autoint_attention_candidate(parent, config, template, round_idx=round_idx)
    elif skill_id == "field_sum":
        candidate_id, genome, mutation_type, rationale = _replace_fm_with_field_sum_candidate(parent, template, round_idx=round_idx)
    elif skill_id == "fm_interaction":
        candidate_id, genome, mutation_type, rationale = _add_fm_interaction_candidate(parent, template, round_idx=round_idx)
    elif skill_id == "bilinear_interaction":
        candidate_id, genome, mutation_type, rationale = _add_bilinear_interaction_candidate(parent, config, template, round_idx=round_idx)
    elif skill_id == "hybridize_crossnet_afm":
        candidate_id, genome, mutation_type, rationale = _hybridize_crossnet_afm_candidate(parent, config, template, round_idx=round_idx)
    elif skill_id == "senet_feature_gate":
        candidate_id, genome, mutation_type, rationale = _specialize_senet_fm_candidate(parent, config, template, round_idx=round_idx)
    elif skill_id == "generated_logit_temperature_calibrator":
        candidate_id, genome, mutation_type, rationale = _code_logit_calibrator_candidate(parent, template, round_idx=round_idx)
    elif skill_id == "generated_residual_logit_gate":
        candidate_id, genome, mutation_type, rationale = _code_residual_gate_candidate(parent, config, template, round_idx=round_idx)
    elif card is not None and _is_reusable_generated_skill_card(card):
        candidate_id, genome, mutation_type, rationale = _reuse_generated_skill_candidate(parent, config, card, template, round_idx=round_idx)
    else:
        raise ValueError(f"No CTR skill-library builder for skill: {skill_id}")

    return CandidateSpec(
        candidate_id=candidate_id,
        genome=genome,
        parent_genome_id=parent.metadata.genome_id,
        mutation_type=mutation_type,
        rationale=rationale,
        evolution_space=_template_evolution_space(template),
        operation=_template_operation(template),
        generated_skill_id=skill_id if card is not None and _is_reusable_generated_skill_card(card) else None,
        proposal_id=card.manifest.get("source_proposal_id") if card is not None else None,
    )


def _template_for_library_skill(evolution: CTREvolutionConfig, skill_id: str, card: SkillCard | None = None) -> dict[str, Any]:
    template_name = LIBRARY_SKILL_TO_TEMPLATE.get(skill_id, skill_id)
    is_code_space = _is_ctr_code_space_library_skill(card) if card is not None else skill_id in CTR_LIBRARY_CODE_SKILL_ORDER
    operation = "reuse_generated" if card is not None and _is_reusable_generated_skill_card(card) else _library_skill_operation(skill_id)
    template = {
        "name": template_name,
        "skill_id": skill_id,
        "selected_from": "skill_library",
        "evolution_space": "code_space" if is_code_space else "skill_space",
        "operation": "open_ended" if is_code_space else operation,
    }
    for configured in evolution.templates or []:
        configured_skill = configured.get("skill_id") or configured.get("skill") or LEGACY_TEMPLATE_TO_LIBRARY_SKILL.get(str(configured.get("name", "")))
        if configured_skill == skill_id or configured.get("name") == template_name:
            template.update(dict(configured))
            template["skill_id"] = skill_id
            template["selected_from"] = "skill_library"
    return template


def _library_skill_operation(skill_id: str) -> str:
    try:
        card = SkillLibrary.from_repo(include_generated=True).get(skill_id)
        if _is_reusable_generated_skill_card(card):
            return "reuse_generated"
        if _is_ctr_code_space_library_skill(card):
            return "open_ended"
    except Exception:
        pass
    if skill_id in CTR_LIBRARY_CODE_SKILL_ORDER:
        return "open_ended"
    if skill_id in {"field_sum"}:
        return "replace"
    if skill_id in {"hybridize_crossnet_afm"}:
        return "hybridize"
    if skill_id in {"senet_feature_gate"}:
        return "specialize"
    return "add"


def _effective_evolution_templates(evolution: CTREvolutionConfig) -> list[dict[str, Any]]:
    budget = max(0, int(evolution.candidate_budget))
    if budget == 0:
        return []
    target_code = min(budget, max(0, int(round(budget * float(evolution.code_space_probability)))))
    target_skill = budget - target_code
    provided = [dict(template) for template in (evolution.templates or [])]
    skill_templates = [template for template in provided if _template_evolution_space(template) != "code_space"]
    code_templates = [template for template in provided if _template_evolution_space(template) == "code_space"]
    skill_templates = _fill_templates(skill_templates, DEFAULT_SKILL_SPACE_TEMPLATES, target_skill)
    code_templates = code_templates[:target_code]
    return skill_templates[:target_skill] + code_templates[:target_code]


def _fill_templates(templates: list[dict[str, Any]], defaults: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    filled = [dict(template) for template in templates[:target]]
    seen = {template.get("name") for template in filled}
    for template in defaults:
        if len(filled) >= target:
            break
        if template.get("name") in seen:
            continue
        filled.append(dict(template))
        seen.add(template.get("name"))
    default_idx = 0
    while len(filled) < target and defaults:
        filled.append(dict(defaults[default_idx % len(defaults)]))
        default_idx += 1
    return filled


def _generate_candidate_from_template(parent: SkillGenome, config: CTRWorkflowConfig, template: dict[str, Any], round_idx: int) -> CandidateSpec:
    name = template.get("name")
    if name == "add_crossnet_v2":
        candidate_id, genome, mutation_type, rationale = _add_crossnet_candidate(parent, config, template, skill_id="crossnet_v2", round_idx=round_idx)
    elif name == "add_crossnet_v1":
        candidate_id, genome, mutation_type, rationale = _add_crossnet_candidate(parent, config, template, skill_id="crossnet_v1", round_idx=round_idx)
    elif name == "add_crossnet_mix":
        candidate_id, genome, mutation_type, rationale = _add_crossnet_candidate(parent, config, template, skill_id="crossnet_mix", round_idx=round_idx)
    elif name == "add_crossnet_mix_without_fm":
        candidate_id, genome, mutation_type, rationale = _add_crossnet_without_fm_candidate(parent, config, template, skill_id="crossnet_mix", round_idx=round_idx)
    elif name == "add_afm_attention":
        candidate_id, genome, mutation_type, rationale = _add_afm_attention_candidate(parent, config, template, round_idx=round_idx)
    elif name == "add_autoint_attention":
        candidate_id, genome, mutation_type, rationale = _add_autoint_attention_candidate(parent, config, template, round_idx=round_idx)
    elif name == "replace_fm_with_field_sum":
        candidate_id, genome, mutation_type, rationale = _replace_fm_with_field_sum_candidate(parent, template, round_idx=round_idx)
    elif name == "hybridize_crossnet_afm":
        candidate_id, genome, mutation_type, rationale = _hybridize_crossnet_afm_candidate(parent, config, template, round_idx=round_idx)
    elif name == "specialize_deep_tower":
        candidate_id, genome, mutation_type, rationale = _specialize_deep_tower_candidate(parent, template, round_idx=round_idx)
    elif name == "specialize_senet_fm":
        candidate_id, genome, mutation_type, rationale = _specialize_senet_fm_candidate(parent, config, template, round_idx=round_idx)
    elif name == "remove_fm":
        candidate_id, genome, mutation_type, rationale = _remove_fm_candidate(parent, template, round_idx=round_idx)
    elif name == "increase_deep_tower":
        candidate_id, genome, mutation_type, rationale = _increase_deep_tower_candidate(parent, template, round_idx=round_idx)
    elif name == "code_logit_calibrator":
        candidate_id, genome, mutation_type, rationale = _code_logit_calibrator_candidate(parent, template, round_idx=round_idx)
    elif name == "code_residual_gate":
        candidate_id, genome, mutation_type, rationale = _code_residual_gate_candidate(parent, config, template, round_idx=round_idx)
    else:
        raise ValueError(f"Unknown CTR evolution template: {name}")
    return CandidateSpec(
        candidate_id=candidate_id,
        genome=genome,
        parent_genome_id=parent.metadata.genome_id,
        mutation_type=mutation_type,
        rationale=rationale,
        evolution_space=_template_evolution_space(template),
        operation=_template_operation(template),
    )


def _dedupe_candidate_id(candidate_id: str, seen_ids: set[str]) -> str:
    base_id = candidate_id
    suffix = 1
    while candidate_id in seen_ids:
        candidate_id = f"{base_id}_{suffix}"
        suffix += 1
    seen_ids.add(candidate_id)
    return candidate_id


def _add_crossnet_candidate(parent: SkillGenome, config: CTRWorkflowConfig, template: dict[str, Any], skill_id: str, round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    cross_logit = _append_crossnet_branch(genome, parent, config, template, skill_id=skill_id, round_idx=round_idx)
    _record_template_mutation(genome, parent, f"add_{skill_id}", template)
    rationale = f"Add {skill_id} branch for high-order CTR feature interactions"
    return f"round{round_idx}_{skill_id}", genome, f"add_{skill_id}", rationale


def _append_crossnet_branch(
    genome: SkillGenome,
    parent: SkillGenome,
    config: CTRWorkflowConfig,
    template: dict[str, Any],
    skill_id: str,
    round_idx: int,
    prefix: str | None = None,
) -> str:
    flat_input_dim = _flat_input_dim(parent, config)
    suffix = prefix or f"{skill_id}_r{round_idx}"
    cross_node_id = _unique_node_id(genome, suffix)
    cross_logit_id = _unique_node_id(genome, f"{suffix}_head")
    cross_output = f"{suffix}_output"
    cross_logit = f"{suffix}_logit"
    genome.nodes.extend(
        [
            SkillNode(
                node_id=cross_node_id,
                skill_id=skill_id,
                skill_name=skill_id,
                category="interaction",
                params={
                    "input_dim": flat_input_dim,
                    "num_layers": int(template.get("num_layers", 2)),
                    "input_key": "flat_embeddings",
                    "output_key": cross_output,
                    **_optional_int_params(template, ["low_rank", "num_experts"]),
                },
                input_keys=["flat_embeddings"],
                output_keys=[cross_output],
                task_types=["ctr"],
                metadata={"template": template},
            ),
            SkillNode(
                node_id=cross_logit_id,
                skill_id="binary_ctr_head",
                skill_name="binary_ctr_head",
                category="head",
                params={"input_key": cross_output, "input_dim": flat_input_dim, "output_key": cross_logit},
                input_keys=[cross_output],
                output_keys=[cross_logit],
                task_types=["ctr"],
                metadata={"template": template},
            ),
        ]
    )
    genome.edges.extend(
        [
            SkillEdge("flatten", cross_node_id, "flat_embeddings", "flat_embeddings"),
            SkillEdge(cross_node_id, cross_logit_id, cross_output, cross_output),
        ]
    )
    _append_logit_to_fusion(genome, cross_logit_id, cross_logit)
    return cross_logit


def _add_afm_attention_candidate(parent: SkillGenome, config: CTRWorkflowConfig, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    _append_afm_attention_branch(genome, config, template, round_idx=round_idx)
    _record_template_mutation(genome, parent, "add_afm_attention", template)
    return f"round{round_idx}_afm_attention", genome, "add_afm_attention", "Add AFM attention branch for weighted pairwise field interactions"


def _append_afm_attention_branch(genome: SkillGenome, config: CTRWorkflowConfig, template: dict[str, Any], round_idx: int, prefix: str | None = None) -> str:
    num_fields = _num_fields(genome)
    embedding_dim = _embedding_dim(genome, config)
    suffix = prefix or f"afm_r{round_idx}"
    afm_node_id = _unique_node_id(genome, suffix)
    afm_logit_id = _unique_node_id(genome, f"{suffix}_head")
    afm_output = f"{suffix}_output"
    afm_logit = f"{suffix}_logit"
    genome.nodes.extend(
        [
            SkillNode(
                node_id=afm_node_id,
                skill_id="afm_attention_pooling",
                skill_name="afm_attention_pooling",
                category="interaction",
                params={
                    "embedding_dim": embedding_dim,
                    "num_fields": num_fields,
                    "attention_dim": int(template.get("attention_dim", 64)),
                    "input_key": "field_embeddings",
                    "output_key": afm_output,
                },
                input_keys=["field_embeddings"],
                output_keys=[afm_output],
                task_types=["ctr"],
                metadata={"template": template},
            ),
            SkillNode(
                node_id=afm_logit_id,
                skill_id="binary_ctr_head",
                skill_name="binary_ctr_head",
                category="head",
                params={"input_key": afm_output, "input_dim": embedding_dim, "output_key": afm_logit},
                input_keys=[afm_output],
                output_keys=[afm_logit],
                task_types=["ctr"],
                metadata={"template": template},
            ),
        ]
    )
    genome.edges.extend(
        [
            SkillEdge("field_embedding", afm_node_id, "field_embeddings", "field_embeddings"),
            SkillEdge(afm_node_id, afm_logit_id, afm_output, afm_output),
        ]
    )
    _append_logit_to_fusion(genome, afm_logit_id, afm_logit)
    return afm_logit


def _add_autoint_attention_candidate(parent: SkillGenome, config: CTRWorkflowConfig, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    flat_input_dim = _flat_input_dim(parent, config)
    embedding_dim = _embedding_dim(genome, config)
    suffix = f"autoint_r{round_idx}"
    autoint_node_id = _unique_node_id(genome, suffix)
    flatten_node_id = _unique_node_id(genome, f"{suffix}_flatten")
    head_node_id = _unique_node_id(genome, f"{suffix}_head")
    attention_output = f"{suffix}_embeddings"
    flat_output = f"{suffix}_flat"
    logit_key = f"{suffix}_logit"
    genome.nodes.extend(
        [
            SkillNode(
                node_id=autoint_node_id,
                skill_id="autoint_attention",
                skill_name="autoint_attention",
                category="interaction",
                params={
                    "embedding_dim": embedding_dim,
                    "num_layers": int(template.get("num_layers", 2)),
                    "num_heads": int(template.get("num_heads", 4)),
                    "dropout": float(template.get("dropout", 0.0)),
                    "residual": bool(template.get("residual", True)),
                    "input_key": "field_embeddings",
                    "output_key": attention_output,
                },
                input_keys=["field_embeddings"],
                output_keys=[attention_output],
                task_types=["ctr"],
                metadata={"template": template},
            ),
            SkillNode(
                node_id=flatten_node_id,
                skill_id="flatten_field_embeddings",
                skill_name="flatten_field_embeddings",
                category="utility",
                params={"input_key": attention_output, "output_key": flat_output},
                input_keys=[attention_output],
                output_keys=[flat_output],
                task_types=["ctr"],
                metadata={"template": template},
            ),
            SkillNode(
                node_id=head_node_id,
                skill_id="binary_ctr_head",
                skill_name="binary_ctr_head",
                category="head",
                params={"input_key": flat_output, "input_dim": flat_input_dim, "output_key": logit_key},
                input_keys=[flat_output],
                output_keys=[logit_key],
                task_types=["ctr"],
                metadata={"template": template},
            ),
        ]
    )
    genome.edges.extend(
        [
            SkillEdge("field_embedding", autoint_node_id, "field_embeddings", "field_embeddings"),
            SkillEdge(autoint_node_id, flatten_node_id, attention_output, attention_output),
            SkillEdge(flatten_node_id, head_node_id, flat_output, flat_output),
        ]
    )
    _append_logit_to_fusion(genome, head_node_id, logit_key)
    _record_template_mutation(genome, parent, "add_autoint_attention", template)
    return f"round{round_idx}_autoint_attention", genome, "add_autoint_attention", "Add AutoInt self-attention branch for field interactions"


def _replace_fm_with_field_sum_candidate(parent: SkillGenome, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    if "fm" in genome.node_ids():
        fm = genome.get_node("fm")
        fm.skill_id = "field_sum"
        fm.skill_name = "field_sum"
        fm.category = "utility"
        fm.params = {"input_key": fm.input_keys[0] if fm.input_keys else "field_embeddings", "output_key": "fm_output"}
        fm.input_keys = [fm.params["input_key"]]
        fm.output_keys = ["fm_output"]
        fm.metadata = {**fm.metadata, "template": template, "replaced_skill_id": "fm_interaction"}
    else:
        field_sum_id = _unique_node_id(genome, f"field_sum_r{round_idx}")
        field_sum_logit = f"field_sum_r{round_idx}_logit"
        genome.nodes.append(
            SkillNode(
                node_id=field_sum_id,
                skill_id="field_sum",
                skill_name="field_sum",
                category="utility",
                params={"input_key": "field_embeddings", "output_key": field_sum_logit},
                input_keys=["field_embeddings"],
                output_keys=[field_sum_logit],
                task_types=["ctr"],
                metadata={"template": template, "replace_fallback": True},
            )
        )
        genome.edges.append(SkillEdge("field_embedding", field_sum_id, "field_embeddings", "field_embeddings"))
        _append_logit_to_fusion(genome, field_sum_id, field_sum_logit)
    _record_template_mutation(genome, parent, "replace_fm_with_field_sum", template)
    rationale = "Replace FM interaction with a field-sum logit to test simpler second-order aggregation"
    return f"round{round_idx}_replace_fm_field_sum", genome, "replace_fm_with_field_sum", rationale


def _hybridize_crossnet_afm_candidate(parent: SkillGenome, config: CTRWorkflowConfig, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    cross_skill = str(template.get("cross_skill", "crossnet_v2"))
    _append_crossnet_branch(genome, parent, config, template, skill_id=cross_skill, round_idx=round_idx, prefix=f"hybrid_{cross_skill}_r{round_idx}")
    _append_afm_attention_branch(genome, config, template, round_idx=round_idx, prefix=f"hybrid_afm_r{round_idx}")
    _record_template_mutation(genome, parent, "hybridize_crossnet_afm", template)
    rationale = "Hybridize explicit CrossNet and AFM attention branches in one child genome"
    return f"round{round_idx}_hybrid_crossnet_afm", genome, "hybridize_crossnet_afm", rationale


def _specialize_deep_tower_candidate(parent: SkillGenome, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    tower = genome.get_node(template.get("target_node_id", "deep_tower"))
    hidden_dims = list(template.get("hidden_dims") or tower.params.get("hidden_dims") or [256, 128, 64])
    tower.params["hidden_dims"] = hidden_dims
    if "dropout" in template:
        tower.params["dropout"] = float(template["dropout"])
    if "activation" in template:
        tower.params["activation"] = str(template["activation"])
    tower.metadata["template"] = template
    _record_template_mutation(genome, parent, "specialize_deep_tower", template)
    rationale = f"Specialize deep tower hidden_dims={hidden_dims}, dropout={tower.params.get('dropout')}"
    return f"round{round_idx}_specialize_deep_tower", genome, "specialize_deep_tower", rationale


def _specialize_senet_fm_candidate(parent: SkillGenome, config: CTRWorkflowConfig, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    num_fields = _num_fields(genome)
    senet_node_id = _unique_node_id(genome, f"senet_fm_r{round_idx}")
    gated_key = f"senet_fm_r{round_idx}_embeddings"
    genome.nodes.append(
        SkillNode(
            node_id=senet_node_id,
            skill_id="senet_feature_gate",
            skill_name="senet_feature_gate",
            category="interaction",
            params={
                "num_fields": num_fields,
                "reduction_ratio": int(template.get("reduction_ratio", 3)),
                "input_key": "field_embeddings",
                "output_key": gated_key,
            },
            input_keys=["field_embeddings"],
            output_keys=[gated_key],
            task_types=["ctr"],
            metadata={"template": template},
        )
    )
    genome.edges.append(SkillEdge("field_embedding", senet_node_id, "field_embeddings", "field_embeddings"))
    if "fm" in genome.node_ids():
        fm = genome.get_node("fm")
        old_fm_inputs = set(fm.input_keys)
        genome.edges = [edge for edge in genome.edges if not (edge.dst_node_id == "fm" and edge.dst_input_key in old_fm_inputs)]
        fm.params["input_key"] = gated_key
        fm.input_keys = [gated_key]
        genome.edges.append(SkillEdge(senet_node_id, "fm", gated_key, gated_key))
    else:
        field_sum_id = _unique_node_id(genome, f"senet_field_sum_r{round_idx}")
        field_sum_logit = f"senet_field_sum_r{round_idx}_logit"
        genome.nodes.append(
            SkillNode(
                node_id=field_sum_id,
                skill_id="field_sum",
                skill_name="field_sum",
                category="utility",
                params={"input_key": gated_key, "output_key": field_sum_logit},
                input_keys=[gated_key],
                output_keys=[field_sum_logit],
                task_types=["ctr"],
                metadata={"template": template},
            )
        )
        genome.edges.append(SkillEdge(senet_node_id, field_sum_id, gated_key, gated_key))
        _append_logit_to_fusion(genome, field_sum_id, field_sum_logit)
    _record_template_mutation(genome, parent, "specialize_senet_fm", template)
    return f"round{round_idx}_specialize_senet_fm", genome, "specialize_senet_fm", "Specialize FM path with SENET feature gating"


def _code_logit_calibrator_candidate(parent: SkillGenome, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    input_key = _current_logits_key(genome)
    output_key = f"generated_calibrated_r{round_idx}_logits"
    node_id = _unique_node_id(genome, f"generated_calibrator_r{round_idx}")
    producer = _producer_node_for_key(genome, input_key)
    genome.nodes.append(
        SkillNode(
            node_id=node_id,
            skill_id="generated_logit_temperature_calibrator",
            skill_name="generated_logit_temperature_calibrator",
            category="generated",
            params={
                "input_key": input_key,
                "output_key": output_key,
                "initial_temperature": float(template.get("initial_temperature", 1.0)),
                "initial_bias": float(template.get("initial_bias", 0.0)),
            },
            input_keys=[input_key],
            output_keys=[output_key],
            task_types=["ctr"],
            source="generated_skill",
            metadata={"template": template, "evolution_space": "code_space"},
        )
    )
    if producer is not None:
        genome.edges.append(SkillEdge(producer, node_id, input_key, input_key))
    _reroute_prediction_and_loss_logits(genome, old_key=input_key, new_key=output_key, producer_node_id=node_id)
    _record_template_mutation(genome, parent, "code_open_ended_logit_calibrator", template)
    rationale = "Code-space generated logit temperature and bias calibrator"
    return f"round{round_idx}_code_logit_calibrator", genome, "code_open_ended_logit_calibrator", rationale


def _code_residual_gate_candidate(parent: SkillGenome, config: CTRWorkflowConfig, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    flat_input_dim = _flat_input_dim(parent, config)
    node_id = _unique_node_id(genome, f"generated_gate_r{round_idx}")
    output_key = f"generated_gate_r{round_idx}_logit"
    genome.nodes.append(
        SkillNode(
            node_id=node_id,
            skill_id="generated_residual_logit_gate",
            skill_name="generated_residual_logit_gate",
            category="generated",
            params={
                "input_key": "flat_embeddings",
                "output_key": output_key,
                "input_dim": flat_input_dim,
                "hidden_dim": int(template.get("hidden_dim", 32)),
                "initial_scale": float(template.get("initial_scale", 0.1)),
            },
            input_keys=["flat_embeddings"],
            output_keys=[output_key],
            task_types=["ctr"],
            source="generated_skill",
            metadata={"template": template, "evolution_space": "code_space"},
        )
    )
    genome.edges.append(SkillEdge("flatten", node_id, "flat_embeddings", "flat_embeddings"))
    _append_logit_to_fusion(genome, node_id, output_key)
    _record_template_mutation(genome, parent, "code_open_ended_residual_gate", template)
    rationale = "Code-space generated residual gate branch over flat embeddings"
    return f"round{round_idx}_code_residual_gate", genome, "code_open_ended_residual_gate", rationale


def _reuse_generated_skill_candidate(parent: SkillGenome, config: CTRWorkflowConfig, card: SkillCard, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    base_id = _slug(card.skill_id)
    node_id = _unique_node_id(genome, f"{base_id}_r{round_idx}")
    input_keys = _resolved_generated_input_keys(card, genome)
    output_keys = _resolved_generated_output_keys(card, round_idx)
    if len(output_keys) == 1 and _is_logit_tensor_key(output_keys[0]):
        output_keys = [f"{base_id}_r{round_idx}_logit"]
    params = _generated_skill_params(card, input_keys=input_keys, output_keys=output_keys, round_idx=round_idx, parent=parent, config=config)
    node = SkillNode(
        node_id=node_id,
        skill_id=card.skill_id,
        skill_name=card.skill_name,
        category=card.category or "generated",
        params=params,
        input_keys=input_keys,
        output_keys=output_keys,
        task_types=card.task_types or ["ctr"],
        source="generated_skill",
        metadata={
            "template": template,
            "evolution_space": "skill_space",
            "source_proposal_id": card.manifest.get("source_proposal_id"),
            "reused_generated_skill": True,
        },
    )
    genome.nodes.append(node)
    _add_missing_input_edges(genome, node_id, input_keys)
    _wire_reused_generated_skill_output(genome, node_id=node_id, input_keys=input_keys, output_keys=output_keys)
    _record_template_mutation(genome, parent, f"skill_reuse_generated_{card.skill_id}", template)
    rationale = f"Reuse promoted generated skill {card.skill_id} from the accumulated skill library"
    return f"round{round_idx}_skill_reuse_{base_id}", genome, f"skill_reuse_generated_{card.skill_id}", rationale


def _add_crossnet_without_fm_candidate(
    parent: SkillGenome,
    config: CTRWorkflowConfig,
    template: dict[str, Any],
    skill_id: str,
    round_idx: int,
) -> tuple[str, SkillGenome, str, str]:
    candidate_id, genome, mutation_type, _ = _add_crossnet_candidate(parent, config, template, skill_id=skill_id, round_idx=round_idx)
    _remove_fm_branch(genome)
    mutation_type = f"{mutation_type}_without_fm"
    _record_template_mutation(genome, parent, mutation_type, template)
    rationale = f"Add {skill_id} branch and remove FM to avoid redundant second-order interactions"
    return f"{candidate_id}_without_fm", genome, mutation_type, rationale


def _remove_fm_candidate(parent: SkillGenome, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    _remove_fm_branch(genome)
    _record_template_mutation(genome, parent, "remove_fm", template)
    return f"round{round_idx}_remove_fm", genome, "remove_fm", "Remove FM branch to test whether second-order interactions hurt validation"


def _add_fm_interaction_candidate(parent: SkillGenome, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    node_id = _unique_node_id(genome, f"fm_extra_r{round_idx}")
    output_key = f"fm_extra_r{round_idx}_logit"
    genome.nodes.append(
        SkillNode(
            node_id=node_id,
            skill_id="fm_interaction",
            skill_name="fm_interaction",
            category="interaction",
            params={
                "input_key": "field_embeddings",
                "output_key": output_key,
                "reduce_sum": bool(template.get("reduce_sum", True)),
            },
            input_keys=["field_embeddings"],
            output_keys=[output_key],
            task_types=["ctr"],
            metadata={"template": template},
        )
    )
    genome.edges.append(SkillEdge("field_embedding", node_id, "field_embeddings", "field_embeddings"))
    _append_logit_to_fusion(genome, node_id, output_key)
    _record_template_mutation(genome, parent, "add_fm_interaction", template)
    return f"round{round_idx}_fm_interaction", genome, "add_fm_interaction", "Add an extra FM interaction branch from the skill library"


def _add_bilinear_interaction_candidate(parent: SkillGenome, config: CTRWorkflowConfig, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    num_fields = _num_fields(genome)
    embedding_dim = _embedding_dim(genome, config)
    num_pairs = num_fields * (num_fields - 1) // 2
    suffix = f"bilinear_r{round_idx}"
    bilinear_node_id = _unique_node_id(genome, suffix)
    flatten_node_id = _unique_node_id(genome, f"{suffix}_flatten")
    head_node_id = _unique_node_id(genome, f"{suffix}_head")
    bilinear_output = f"{suffix}_interactions"
    flat_output = f"{suffix}_flat"
    logit_key = f"{suffix}_logit"
    genome.nodes.extend(
        [
            SkillNode(
                node_id=bilinear_node_id,
                skill_id="bilinear_interaction",
                skill_name="bilinear_interaction",
                category="interaction",
                params={
                    "embedding_dim": embedding_dim,
                    "num_fields": num_fields,
                    "bilinear_type": str(template.get("bilinear_type", "field_interaction")),
                    "input_key": "field_embeddings",
                    "output_key": bilinear_output,
                },
                input_keys=["field_embeddings"],
                output_keys=[bilinear_output],
                task_types=["ctr"],
                metadata={"template": template},
            ),
            SkillNode(
                node_id=flatten_node_id,
                skill_id="flatten_field_embeddings",
                skill_name="flatten_field_embeddings",
                category="utility",
                params={"input_key": bilinear_output, "output_key": flat_output},
                input_keys=[bilinear_output],
                output_keys=[flat_output],
                task_types=["ctr"],
                metadata={"template": template},
            ),
            SkillNode(
                node_id=head_node_id,
                skill_id="binary_ctr_head",
                skill_name="binary_ctr_head",
                category="head",
                params={"input_key": flat_output, "input_dim": num_pairs * embedding_dim, "output_key": logit_key},
                input_keys=[flat_output],
                output_keys=[logit_key],
                task_types=["ctr"],
                metadata={"template": template},
            ),
        ]
    )
    genome.edges.extend(
        [
            SkillEdge("field_embedding", bilinear_node_id, "field_embeddings", "field_embeddings"),
            SkillEdge(bilinear_node_id, flatten_node_id, bilinear_output, bilinear_output),
            SkillEdge(flatten_node_id, head_node_id, flat_output, flat_output),
        ]
    )
    _append_logit_to_fusion(genome, head_node_id, logit_key)
    _record_template_mutation(genome, parent, "add_bilinear_interaction", template)
    return f"round{round_idx}_bilinear_interaction", genome, "add_bilinear_interaction", "Add FiBiNet-style bilinear interaction branch from the skill library"


def _remove_fm_branch(genome: SkillGenome) -> None:
    genome.nodes = [node for node in genome.nodes if node.node_id != "fm"]
    genome.edges = [edge for edge in genome.edges if edge.src_node_id != "fm" and edge.dst_node_id != "fm"]
    fusion = _active_fusion_node(genome)
    if fusion is None:
        return
    fusion.input_keys = [key for key in fusion.input_keys if key != "fm_output"]
    if "input_keys" in fusion.params:
        fusion.params["input_keys"] = [key for key in fusion.params["input_keys"] if key != "fm_output"]


def _increase_deep_tower_candidate(parent: SkillGenome, template: dict[str, Any], round_idx: int) -> tuple[str, SkillGenome, str, str]:
    genome = parent.clone()
    tower = genome.get_node(template.get("target_node_id", "deep_tower"))
    hidden_dims = list(template.get("hidden_dims") or tower.params.get("hidden_dims") or [128, 64])
    tower.params["hidden_dims"] = hidden_dims
    tower.metadata["template"] = template
    _record_template_mutation(genome, parent, "increase_deep_tower", template)
    return f"round{round_idx}_deep_tower", genome, "increase_deep_tower", f"Set deep tower hidden_dims={hidden_dims}"


def _optional_int_params(template: dict[str, Any], keys: list[str]) -> dict[str, int]:
    return {key: int(template[key]) for key in keys if key in template}


def _template_evolution_space(template: dict[str, Any]) -> str:
    value = str(template.get("evolution_space") or template.get("space") or "").lower()
    if value in {"code", "code_space", "code-space", "open_ended", "open-ended"}:
        return "code_space"
    return "skill_space"


def _template_operation(template: dict[str, Any]) -> str:
    if template.get("operation"):
        return str(template["operation"])
    name = str(template.get("name", ""))
    if name.startswith("add_"):
        return "add"
    if name.startswith("replace_"):
        return "replace"
    if name.startswith("hybridize_"):
        return "hybridize"
    if name.startswith("specialize_") or name.startswith("increase_"):
        return "specialize"
    if name.startswith("code_"):
        return "open_ended"
    return "template"


def _candidate_generation_label(result: CandidateResult) -> str:
    if result.candidate_id == "baseline" or result.mutation_type == "baseline":
        return "baseline"
    round_idx = _round_idx_from_candidate_id(result.candidate_id)
    if round_idx is None:
        return "unknown"
    return _ordinal(round_idx + 1)


def _round_idx_from_candidate_id(candidate_id: str) -> int | None:
    if not candidate_id.startswith("round"):
        return None
    digits = []
    for char in candidate_id[len("round") :]:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return None
    return int("".join(digits))


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _candidate_evolution_category(result: CandidateResult) -> str:
    if result.candidate_id == "baseline" or result.mutation_type == "baseline":
        return "baseline"
    space = result.evolution_space or "unknown"
    operation = result.operation or _operation_from_mutation_type(result.mutation_type)
    if space == "code_space":
        return f"code_space:{operation or 'open_ended'}"
    if space == "skill_space":
        return f"skill_space:{operation or 'unknown'}"
    return "unknown"


def _structure_category(evolution_space: str, operation: str, mutation_type: str) -> str:
    if mutation_type == "baseline":
        return "baseline"
    operation = operation or _operation_from_mutation_type(mutation_type)
    if evolution_space == "code_space":
        return f"code_space:{operation or 'open_ended'}"
    if evolution_space == "skill_space":
        return f"skill_space:{operation or 'unknown'}"
    return "unknown"


def _operation_from_mutation_type(mutation_type: str) -> str:
    if mutation_type.startswith("add_"):
        return "add"
    if mutation_type.startswith(("replace_", "remove_")):
        return "replace"
    if mutation_type.startswith("hybridize_"):
        return "hybridize"
    if mutation_type.startswith(("specialize_", "increase_")):
        return "specialize"
    if mutation_type.startswith("code_"):
        return "open_ended"
    return ""


def _available_candidate_gpu_ids() -> list[str]:
    override = os.environ.get("CTR_EVOLUTION_GPU_IDS")
    if override:
        return _parse_gpu_ids(override)
    physical_ids = _nvidia_smi_gpu_ids()
    if physical_ids:
        return physical_ids
    visible = _parse_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if visible:
        return visible
    if torch.cuda.is_available():
        return [str(idx) for idx in range(torch.cuda.device_count())]
    return []


def _nvidia_smi_gpu_ids() -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return []
    return _parse_gpu_ids(proc.stdout.replace("\n", ","))


def _parse_gpu_ids(raw: str) -> list[str]:
    blocked_values = {"", "-1", "none", "nodevfiles", "no_dev_files"}
    ids = []
    for item in raw.split(","):
        value = item.strip()
        if value.lower() in blocked_values:
            continue
        ids.append(value)
    return ids


def _uses_cuda(device: str) -> bool:
    return str(device).lower().startswith("cuda")


def _append_logit_to_fusion(genome: SkillGenome, producer_node_id: str, logit_key: str, *, fusion_node_id: str = "fusion") -> None:
    if fusion_node_id not in genome.node_ids():
        active_fusion = _active_fusion_node(genome)
        if active_fusion is None:
            _reroute_prediction_and_loss_logits(
                genome,
                old_key=_current_logits_key(genome),
                new_key=logit_key,
                producer_node_id=producer_node_id,
            )
            return
        fusion_node_id = active_fusion.node_id
    genome.edges.append(SkillEdge(producer_node_id, fusion_node_id, logit_key, logit_key))
    fusion = genome.get_node(fusion_node_id)
    fusion.params.setdefault("input_keys", list(fusion.input_keys))
    fusion.params["input_keys"] = list(dict.fromkeys(list(fusion.params["input_keys"]) + [logit_key]))
    fusion.input_keys = list(dict.fromkeys(list(fusion.input_keys) + [logit_key]))


def _add_missing_input_edges(genome: SkillGenome, node_id: str, input_keys: list[str]) -> None:
    required_inputs = set(genome.constraints.required_inputs)
    existing = {(edge.dst_node_id, edge.dst_input_key) for edge in genome.edges}
    for input_key in input_keys:
        if (node_id, input_key) in existing or input_key in required_inputs:
            continue
        producer = _producer_node_for_key(genome, input_key)
        if producer is not None:
            genome.edges.append(SkillEdge(producer, node_id, input_key, input_key))


def _wire_reused_generated_skill_output(genome: SkillGenome, *, node_id: str, input_keys: list[str], output_keys: list[str]) -> None:
    if not output_keys:
        return
    output_key = output_keys[0]
    if _is_logit_tensor_key(output_key) and any(_is_logit_tensor_key(key) for key in input_keys):
        old_key = next((key for key in input_keys if _is_logit_tensor_key(key)), _current_logits_key(genome))
        _reroute_prediction_and_loss_logits(genome, old_key=old_key, new_key=output_key, producer_node_id=node_id)
        return
    if _is_logit_tensor_key(output_key):
        _attach_logit_to_active_terminal(genome, producer_node_id=node_id, logit_key=output_key)


def _attach_logit_to_active_terminal(genome: SkillGenome, *, producer_node_id: str, logit_key: str) -> None:
    terminal_key = _current_logits_key(genome)
    terminal_node_id = _producer_node_for_key(genome, terminal_key)
    if terminal_node_id is None:
        _reroute_prediction_and_loss_logits(genome, old_key=terminal_key, new_key=logit_key, producer_node_id=producer_node_id)
        return
    try:
        terminal = genome.get_node(terminal_node_id)
    except KeyError:
        terminal = None
    if terminal is not None and (
        "fusion" in terminal.category.lower()
        or "fusion" in terminal.skill_id.lower()
        or "fusion" in terminal.node_id.lower()
    ):
        _append_logit_to_fusion(genome, producer_node_id, logit_key, fusion_node_id=terminal_node_id)
        return
    _reroute_prediction_and_loss_logits(genome, old_key=terminal_key, new_key=logit_key, producer_node_id=producer_node_id)


def _active_fusion_node(genome: SkillGenome) -> SkillNode | None:
    if "fusion" in genome.node_ids():
        try:
            return genome.get_node("fusion")
        except KeyError:
            pass
    terminal_node_id = _producer_node_for_key(genome, _current_logits_key(genome))
    if terminal_node_id:
        try:
            terminal = genome.get_node(terminal_node_id)
        except KeyError:
            terminal = None
        if terminal is not None and _is_fusion_like_node(terminal):
            return terminal
    for node in genome.nodes:
        if _is_fusion_like_node(node):
            return node
    return None


def _is_fusion_like_node(node: SkillNode) -> bool:
    return (
        "fusion" in node.category.lower()
        or "fusion" in node.skill_id.lower()
        or "fusion" in node.node_id.lower()
    )


def _reroute_prediction_and_loss_logits(genome: SkillGenome, old_key: str, new_key: str, producer_node_id: str) -> None:
    prediction = genome.get_node("prediction")
    loss = genome.get_node("loss")
    prediction.params["input_key"] = new_key
    prediction.input_keys = [new_key]
    loss.params["logits_key"] = new_key
    non_logit_loss_inputs = [key for key in loss.input_keys if not _is_logit_tensor_key(key)]
    loss.input_keys = list(dict.fromkeys([new_key] + non_logit_loss_inputs))
    genome.edges = [
        edge
        for edge in genome.edges
        if not (
            edge.dst_node_id in {"prediction", "loss"}
            and (
                edge.dst_input_key == old_key
                or edge.src_output_key == old_key
                or _is_logit_tensor_key(edge.dst_input_key)
                or _is_logit_tensor_key(edge.src_output_key)
            )
        )
    ]
    genome.edges.extend(
        [
            SkillEdge(producer_node_id, "prediction", new_key, new_key),
            SkillEdge(producer_node_id, "loss", new_key, new_key),
        ]
    )


def _producer_node_for_key(genome: SkillGenome, key: str) -> str | None:
    for node in genome.nodes:
        if key in node.output_keys:
            return node.node_id
    return None


def _current_logits_key(genome: SkillGenome) -> str:
    try:
        loss = genome.get_node("loss")
        if loss.params.get("logits_key"):
            return str(loss.params["logits_key"])
    except Exception:
        pass
    try:
        prediction = genome.get_node("prediction")
        if prediction.params.get("input_key"):
            return str(prediction.params["input_key"])
    except Exception:
        pass
    return "logits"


def _resolved_generated_input_keys(card: SkillCard, genome: SkillGenome) -> list[str]:
    keys = [key for key in card.input_keys if key and not _is_param_placeholder_key(key)]
    if len(keys) == 1 and _is_logit_tensor_key(keys[0]):
        return [_current_logits_key(genome)]
    if keys:
        return keys
    params = _generated_fragment_params(card)
    if params.get("input_key"):
        if _is_logit_tensor_key(str(params["input_key"])):
            return [_current_logits_key(genome)]
        return [str(params["input_key"])]
    common_upstream = [str(item) for item in (card.manifest.get("composition") or {}).get("common_upstream", []) or []]
    if any(item in {"fusion", "additive_fusion"} for item in common_upstream):
        return [_current_logits_key(genome)]
    if any(item in {"flatten", "flatten_field_embeddings"} for item in common_upstream):
        return ["flat_embeddings"]
    if any(item in {"field_embedding"} for item in common_upstream):
        return ["field_embeddings"]
    return [card.input_keys[0]] if card.input_keys else []


def _resolved_generated_output_keys(card: SkillCard, round_idx: int) -> list[str]:
    keys = [key for key in card.output_keys if key and not _is_param_placeholder_key(key)]
    if keys:
        return keys
    params = _generated_fragment_params(card)
    if params.get("output_key"):
        return [str(params["output_key"])]
    base = _slug(card.skill_id)
    return [f"{base}_r{round_idx}_logit"]


def _generated_skill_params(
    card: SkillCard,
    *,
    input_keys: list[str],
    output_keys: list[str],
    round_idx: int,
    parent: SkillGenome | None = None,
    config: CTRWorkflowConfig | None = None,
) -> dict[str, Any]:
    params = _generated_fragment_params(card)
    if len(input_keys) == 1:
        params["input_key"] = input_keys[0]
    elif "input_key" not in params and input_keys:
        params["input_key"] = input_keys[0]
    if len(output_keys) == 1:
        params["output_key"] = output_keys[0]
    elif "output_key" not in params and output_keys:
        params["output_key"] = output_keys[0]
    if output_keys and str(params.get("output_key")) in {"output_key", ""}:
        params["output_key"] = output_keys[0]
    if input_keys and str(params.get("input_key")) in {"input_key", ""}:
        params["input_key"] = input_keys[0]
    if parent is not None:
        params = {key: _resolve_generated_param_value(value, parent, config) for key, value in params.items()}
    return params


def _resolve_generated_param_value(value: Any, parent: SkillGenome, config: CTRWorkflowConfig | None = None) -> Any:
    if isinstance(value, str):
        normalized = value.strip()
        runtime = _generated_skill_runtime_shape(parent, config or CTRWorkflowConfig())
        if normalized in {"${num_fields}", "${field_count}"}:
            return runtime["num_fields"]
        if normalized == "${embedding_dim}":
            return runtime["embedding_dim"]
        if normalized in {"${input_dim}", "${flat_input_dim}", "${fusion_dim}"}:
            return runtime["flat_input_dim"]
    return value


def _generated_fragment_params(card: SkillCard) -> dict[str, Any]:
    fragment = (card.manifest.get("composition") or {}).get("example_genome_fragment") or {}
    return dict(fragment.get("params") or {})


def _is_param_placeholder_key(key: str) -> bool:
    return key in {"input_key", "output_key", "logits_key", "labels_key"}


def _is_logit_tensor_key(key: str) -> bool:
    return "logit" in key.lower() or key == "logits"


def _slug(value: str) -> str:
    slug = "".join(char if char.isalnum() or char == "_" else "_" for char in value.lower()).strip("_")
    return slug or "generated_skill"


def _slug_for_candidate_id(value: str) -> str:
    slug = "".join(char if char.isalnum() or char == "_" else "_" for char in str(value).lower()).strip("_")
    return slug or "skill"


def _num_fields(genome: SkillGenome) -> int:
    embedding = genome.get_node("field_embedding")
    vocab_sizes = embedding.params.get("vocab_sizes") or []
    if not vocab_sizes:
        raise ValueError("field_embedding must expose vocab_sizes to infer num_fields")
    return len(vocab_sizes)


def _embedding_dim(genome: SkillGenome, config: CTRWorkflowConfig) -> int:
    try:
        return int(genome.get_node("field_embedding").params.get("embedding_dim", config.genome.embedding_dim))
    except Exception:
        return int(config.genome.embedding_dim)


def _record_template_mutation(genome: SkillGenome, parent: SkillGenome, mutation_type: str, template: dict[str, Any]) -> None:
    genome.record_mutation(
        GenomeMutation(
            mutation_type=mutation_type,
            parent_genome_id=parent.metadata.genome_id,
            child_genome_id=genome.metadata.genome_id,
            description=f"CTR template mutation: {mutation_type}",
            details={"template": template},
        )
    )


def _select_survivors(candidates: list[CandidateResult], training: CTRTrainingConfig, evolution: CTREvolutionConfig) -> list[CandidateResult]:
    successful = [candidate for candidate in candidates if candidate.error is None and _objective_value(candidate.metrics, training) is not None]
    scored = [(candidate, float(_objective_value(candidate.metrics, training))) for candidate in successful]
    scored = [(candidate, value) for candidate, value in scored if not _is_nan(value)]
    if not scored:
        return []
    reverse = training.objective_metric != "logloss"
    average = float(np.mean([value for _, value in scored]))
    if training.objective_metric == "logloss":
        survivors = [(candidate, value) for candidate, value in scored if value <= average + training.min_delta]
    else:
        survivors = [(candidate, value) for candidate, value in scored if value >= average - training.min_delta]
    survivors.sort(key=lambda item: item[1], reverse=reverse)
    top_k = int(getattr(evolution, "survivor_top_k", 0) or 0)
    if top_k > 0:
        survivors = survivors[:top_k]
    return [candidate for candidate, _ in survivors]


def _objective_value(metrics: dict[str, Any], training: CTRTrainingConfig) -> float | None:
    metric = training.objective_metric
    if metric == "auc" and metrics.get("validation_best_auc") is not None:
        return metrics.get("validation_best_auc")
    return metrics.get(metric)


def _best_objective_value(candidates: list[CandidateResult], training: CTRTrainingConfig) -> float | None:
    values = []
    for candidate in candidates:
        if candidate.error is not None:
            continue
        value = _objective_value(candidate.metrics, training)
        if value is None:
            continue
        value = float(value)
        if not _is_nan(value):
            values.append(value)
    if not values:
        return None
    return min(values) if training.objective_metric == "logloss" else max(values)


def _objective_improvement(round_best: float, global_best: Any, training: CTRTrainingConfig) -> float | None:
    if global_best is None:
        return None
    global_best = float(global_best)
    round_best = float(round_best)
    if training.objective_metric == "logloss":
        return global_best - round_best
    return round_best - global_best


def _flat_input_dim(parent: SkillGenome, config: CTRWorkflowConfig) -> int:
    try:
        embedding = parent.get_node("field_embedding")
        vocab_sizes = embedding.params.get("vocab_sizes") or []
        embedding_dim = int(embedding.params.get("embedding_dim", config.genome.embedding_dim))
        if vocab_sizes:
            return len(vocab_sizes) * embedding_dim
    except Exception:
        pass
    return int(config.genome.embedding_dim)


def _unique_node_id(genome: SkillGenome, prefix: str) -> str:
    node_id = prefix
    idx = 1
    existing = genome.node_ids()
    while node_id in existing:
        node_id = f"{prefix}_{idx}"
        idx += 1
    return node_id


def _select_improved(candidates: list[CandidateResult], baseline: CandidateResult, training: CTRTrainingConfig) -> CandidateResult | None:
    improved = [candidate for candidate in candidates if candidate.error is None and _is_improved(candidate.metrics, baseline.metrics, training)]
    if not improved:
        return None
    reverse = training.objective_metric != "logloss"
    improved.sort(key=lambda item: _objective_value(item.metrics, training) or (-math.inf if reverse else math.inf), reverse=reverse)
    return improved[0]


def _is_improved(candidate: dict[str, Any], baseline: dict[str, Any], training: CTRTrainingConfig) -> bool:
    candidate_value = _objective_value(candidate, training)
    baseline_value = _objective_value(baseline, training)
    if candidate_value is None or baseline_value is None or _is_nan(candidate_value) or _is_nan(baseline_value):
        return False
    if training.objective_metric == "logloss":
        return float(candidate_value) < float(baseline_value) - training.min_delta
    return float(candidate_value) > float(baseline_value) + training.min_delta


def _build_ctr_labels(data: pd.DataFrame, config: CTRDatasetConfig) -> np.ndarray:
    if config.label_col and config.label_col in data.columns:
        return data[config.label_col].astype(np.float32).to_numpy()
    if config.rating_col not in data.columns:
        raise ValueError(f"Neither label_col nor rating_col is available. Missing rating_col={config.rating_col}")
    return (data[config.rating_col].astype(float) >= config.positive_rating_threshold).astype(np.float32).to_numpy()


def _split_indices(n_rows: int, split_ratio: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_rows < 3:
        raise ValueError("Need at least 3 rows to create train/val/test splits")
    total = sum(split_ratio)
    ratios = [item / total for item in split_ratio]
    train_end = max(1, int(n_rows * ratios[0]))
    val_end = max(train_end + 1, train_end + int(n_rows * ratios[1]))
    val_end = min(val_end, n_rows - 1)
    indices = np.arange(n_rows)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _build_loader(
    x: dict[str, np.ndarray],
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    subset_x = {key: value[indices] for key, value in x.items()}
    subset_y = y[indices]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(TorchDataset(subset_x, subset_y), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator)


def _safe_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    try:
        if len(set(labels.tolist())) < 2:
            return float("nan")
        return float(roc_auc_score(labels, predictions))
    except Exception:
        return float("nan")


def _safe_logloss(labels: np.ndarray, predictions: np.ndarray) -> float:
    try:
        labels_arg = [0, 1] if len(set(labels.tolist())) < 2 else None
        return float(log_loss(labels, predictions, labels=labels_arg))
    except Exception:
        return float("nan")


def _safe_architecture_fingerprint(genome: SkillGenome, *, fallback: str) -> str:
    try:
        return genome_architecture_fingerprint(genome)
    except Exception as exc:
        return f"fingerprint_error_{_slug_for_candidate_id(fallback)}_{hashlib.sha256(f'{exc.__class__.__name__}:{exc}'.encode('utf-8')).hexdigest()[:12]}"


def _safe_code_space_provider_diagnostics(code_config: CodeSpaceConfig) -> dict[str, Any]:
    try:
        return code_space_provider_diagnostics(code_config)
    except Exception as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def _write_tsv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _as_int(value: Any, *, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run automated CTR training, evaluation, and Stage 2 evolution.")
    parser.add_argument("--config", required=True, help="YAML workflow config.")
    parser.add_argument("--device", help="Override training.device, for launchers that set CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--output-dir", help="Override output_dir.")
    parser.add_argument("--seed", type=int, help="Override training.seed for independent random runs.")
    args = parser.parse_args(argv)
    config = load_workflow_config(args.config)
    if args.device:
        config.training.device = args.device
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.training.seed = args.seed
    summary = CTRModelEvolutionRunner(config).run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
