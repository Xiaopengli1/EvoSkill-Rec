"""Multi-task (MTL) recommendation-model evolution workflow.

This module mirrors :mod:`recskill.evolution.ctr_workflow` but targets multi-task
learning instead of single-task CTR. The reference dataset is census-income, where
two binary heads are trained jointly:

* ``income``         -> task 0, modelled as the CTR-style task
* ``marital status`` -> task 1, modelled as the CVR-style task

The architecture being evolved is the *shared transform* that turns the flattened
field embeddings into per-task representations. The fixed scaffold around it is::

    sparse_features -> field_embedding
    dense_values    -> dense_feature_path -> field_embeddings -> flatten
                                           -> <shared transform> -> task_representations [B, n_task, R]
                                           -> task_tower -> task_outputs [B, n_task]
    task_outputs + task_labels -> multitask_loss -> loss

The shared transform is one of ``shared_bottom`` (+ ``task_replication``),
``aitm`` (+ ``aitm_transfer``), ``mmoe``, ``ple``, or ``task_specific``.
Evolution swaps it and tunes its size.

The runner reuses as much of the CTR workflow infrastructure as possible
(evolution memory, deduplication, survivor selection, reporting, fingerprinting).
Only the MTL-specific pieces are new: data preparation, baseline genome
construction, per-task evaluation, and the candidate generators.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import json
import math
import copy
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from torch_rechub.utils.data import TorchDataset

from .compiler import SkillGenomeCompiler
from .code_space import inspect_open_ended_proposal_ingestions
from .ctr_workflow import (
    CandidateResult,
    CandidateSpec,
    CTREvolutionConfig,
    CTRGenomeTrainer,
    CTRModelEvolutionRunner,
    CTRTrainingConfig,
    _allocate_code_budget_by_parent,
    _adaptive_candidate_targets,
    _annotate_generated_skill_artifacts,
    _annotate_generated_skill_hash,
    _as_bool,
    _available_candidate_gpu_ids,
    _build_loader,
    _dedupe_candidate_id,
    _best_result_by_validation_auc,
    _next_parent_population,
    _path_sha256,
    _round_result_groups,
    _round_space_counts,
    _safe_architecture_fingerprint,
    _safe_code_space_provider_diagnostics,
    _safe_auc,
    _safe_logloss,
    _select_survivors,
    _set_seed,
    _generated_skill_portability_allows_reuse,
    _is_reusable_generated_skill_card,
    _skill_card_has_loadable_implementation,
    _train_candidate_artifact,
    _uses_cuda,
)
from .mtl_code_space import build_mtl_code_space_provider
from .fingerprint import config_fingerprint, diff_payload
from .genome import GenomeConstraints, GenomeMutation, SkillEdge, SkillGenome, SkillNode
from .skill_library import SkillCard, SkillLibrary
from .verification import GenomeVerifier


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MTLDatasetConfig:
    name: str = "census_income_mtl"
    type: str = "census_mtl"
    # Pre-split files (the workflow concatenates them and reuses the split order).
    train_path: str = "examples/ranking/data/census-income/census_income_train.csv"
    val_path: str = "examples/ranking/data/census-income/census_income_val.csv"
    test_path: str = "examples/ranking/data/census-income/census_income_test.csv"
    # Single-file fallback: if `path` is set, it is split with `split_ratio`.
    path: str | None = None
    split_ratio: list[float] = field(default_factory=lambda: [0.7, 0.1, 0.2])
    # Task definition. label_cols[i] -> task i. income=CTR task, marital=CVR task.
    label_cols: list[str] = field(default_factory=lambda: ["income", "marital status"])
    task_names: list[str] = field(default_factory=lambda: ["income_ctr", "marital_cvr"])
    task_types: list[str] = field(default_factory=lambda: ["classification", "classification"])
    # Feature handling follows the Census preprocessing contract: categorical columns
    # stay as sparse ids and dense columns stay as continuous numeric values.
    dense_cols: list[str] = field(
        default_factory=lambda: [
            "age",
            "wage per hour",
            "capital gains",
            "capital losses",
            "divdends from stocks",
            "num persons worked for employer",
            "weeks worked in year",
        ]
    )
    # Kept for backward YAML compatibility. MTL no longer bins dense columns.
    dense_bins: int = 32
    limit_rows: int | None = None


@dataclass
class MTLGenomeConfig:
    genome_path: str | None = None
    # Shared-transform template: shared_bottom | aitm | mmoe | ple | task_specific
    template: str = "shared_bottom"
    embedding_dim: int = 8
    shared_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    task_hidden_dims: list[int] = field(default_factory=lambda: [32])
    expert_dim: int = 32
    num_experts: int = 4
    ple_shared_experts: int = 1
    ple_specific_experts: int = 1
    dropout: float = 0.1
    activation: str = "relu"
    task_weights: list[float] | None = None


@dataclass
class MTLWorkflowConfig:
    experiment_name: str = "mtl_evolution_census"
    output_dir: str = "outputs/evolution/mtl_census"
    dataset: MTLDatasetConfig = field(default_factory=MTLDatasetConfig)
    genome: MTLGenomeConfig = field(default_factory=MTLGenomeConfig)
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
class MTLDatasetBundle:
    feature_names: list[str]
    vocab_sizes: list[int]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    metadata: dict[str, Any]
    n_task: int
    task_names: list[str]
    task_types: list[str]
    sparse_feature_names: list[str] = field(default_factory=list)
    dense_feature_names: list[str] = field(default_factory=list)


def load_mtl_workflow_config(path: str | Path) -> MTLWorkflowConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return MTLWorkflowConfig(
        experiment_name=raw.get("experiment_name", "mtl_evolution_census"),
        output_dir=raw.get("output_dir", "outputs/evolution/mtl_census"),
        dataset=MTLDatasetConfig(**(raw.get("dataset") or {})),
        genome=MTLGenomeConfig(**(raw.get("genome") or {})),
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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _encode_sparse_column(values: pd.Series) -> tuple[np.ndarray, int]:
    """Return sparse ids using the census preprocessing contract when possible."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        ids = numeric.fillna(0).round().astype(np.int64).to_numpy()
        if ids.size and ids.min() < 0:
            ids = ids - int(ids.min())
        vocab_size = int(ids.max()) + 1 if ids.size else 1
        return ids, max(1, vocab_size)

    filled = values.fillna("__missing__").astype(str)
    codes, _ = pd.factorize(filled, sort=True)
    ids = codes.astype(np.int64)
    vocab_size = int(ids.max()) + 1 if len(ids) else 1
    return ids, max(1, vocab_size)


def _dense_values(values: pd.Series) -> np.ndarray:
    """Keep dense census features as continuous values, matching the Census preprocessing contract."""
    return pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def _binary_label(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    uniques = np.unique(numeric[~np.isnan(numeric)])
    if uniques.size and not np.array_equal(np.sort(uniques), np.array([0.0, 1.0])):
        # Treat any positive value as 1, others as 0 (defensive for non-0/1 encodings).
        if not np.all(np.isin(numeric, [0.0, 1.0])):
            numeric = (numeric > 0).astype(np.float32)
    return numeric


def prepare_census_mtl_data(config: MTLDatasetConfig, training: CTRTrainingConfig) -> MTLDatasetBundle:
    """Load census-income (or any tabular MTL CSV) into a multi-task bundle."""
    if config.path:
        single = Path(config.path)
        if not single.exists():
            raise FileNotFoundError(f"Dataset not found: {single}")
        data = pd.read_csv(single)
        train_idx_end = val_idx_end = None  # signal ratio split below
    else:
        paths = [Path(config.train_path), Path(config.val_path), Path(config.test_path)]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Dataset not found: {p}")
        frames = [pd.read_csv(p) for p in paths]
        train_idx_end = len(frames[0])
        val_idx_end = train_idx_end + len(frames[1])
        data = pd.concat(frames, axis=0, ignore_index=True)

    if config.limit_rows is not None:
        data = data.head(config.limit_rows).copy()
        train_idx_end = val_idx_end = None

    missing_labels = [col for col in config.label_cols if col not in data.columns]
    if missing_labels:
        raise ValueError(f"Label columns not found in dataset: {missing_labels}")
    n_task = len(config.label_cols)
    if n_task < 2:
        raise ValueError("MTL workflow requires at least 2 label columns")
    if len(config.task_types) != n_task:
        raise ValueError("task_types length must match label_cols length")
    if len(config.task_names) != n_task:
        raise ValueError("task_names length must match label_cols length")

    labels = np.stack([_binary_label(data[col]) for col in config.label_cols], axis=1).astype(np.float32)

    data = data.fillna(0)
    feature_cols = [col for col in data.columns if col not in config.label_cols]
    dense_set = {col for col in config.dense_cols if col in feature_cols}
    dense_feature_names = [col for col in feature_cols if col in dense_set]
    sparse_feature_names = [col for col in feature_cols if col not in dense_set]

    encoded: list[np.ndarray] = []
    vocab_sizes: list[int] = []
    for col in sparse_feature_names:
        ids, vocab = _encode_sparse_column(data[col])
        encoded.append(ids)
        vocab_sizes.append(vocab)
    if not encoded:
        raise ValueError("No feature columns available after removing labels")
    sparse_features = np.stack(encoded, axis=1)
    x = {"sparse_features": sparse_features}
    if dense_feature_names:
        dense_matrix = np.stack([_dense_values(data[col]) for col in dense_feature_names], axis=1)
        x["dense_values"] = dense_matrix
        x["dense_features"] = dense_matrix

    n_rows = len(labels)
    if train_idx_end is not None and val_idx_end is not None:
        train_idx = np.arange(0, train_idx_end)
        val_idx = np.arange(train_idx_end, val_idx_end)
        test_idx = np.arange(val_idx_end, n_rows)
    else:
        rng = np.random.default_rng(training.seed)
        order = rng.permutation(n_rows)
        total = sum(config.split_ratio) or 1.0
        ratios = [r / total for r in config.split_ratio]
        train_end = max(1, int(n_rows * ratios[0]))
        val_end = min(max(train_end + 1, train_end + int(n_rows * ratios[1])), n_rows - 1)
        train_idx, val_idx, test_idx = order[:train_end], order[train_end:val_end], order[val_end:]

    train_loader = _build_loader(x, labels, train_idx, training.batch_size, shuffle=True, seed=training.seed, num_workers=training.num_workers)
    val_loader = _build_loader(x, labels, val_idx, training.batch_size, shuffle=False, seed=training.seed, num_workers=training.num_workers)
    test_loader = _build_loader(x, labels, test_idx, training.batch_size, shuffle=False, seed=training.seed, num_workers=training.num_workers)

    metadata = {
        "dataset_paths": [config.path] if config.path else [config.train_path, config.val_path, config.test_path],
        "num_rows": int(n_rows),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_idx)),
        "n_task": n_task,
        "task_names": list(config.task_names),
        "task_types": list(config.task_types),
        "label_cols": list(config.label_cols),
        "positive_rates": {
            name: float(labels[:, idx].mean()) if n_rows else None
            for idx, name in enumerate(config.task_names)
        },
        "num_features": len(feature_cols),
        "num_sparse_features": len(sparse_feature_names),
        "num_dense_features": len(dense_feature_names),
        "num_dense_binned": 0,
        "dense_handling": "raw_continuous_dense_feature_path",
        "feature_names": feature_cols,
        "sparse_feature_names": sparse_feature_names,
        "dense_feature_names": dense_feature_names,
        "vocab_sizes": vocab_sizes,
        "dense_bins": None,
    }
    return MTLDatasetBundle(
        feature_names=feature_cols,
        vocab_sizes=vocab_sizes,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        metadata=metadata,
        n_task=n_task,
        task_names=list(config.task_names),
        task_types=list(config.task_types),
        sparse_feature_names=sparse_feature_names,
        dense_feature_names=dense_feature_names,
    )


# ---------------------------------------------------------------------------
# Genome construction
# ---------------------------------------------------------------------------


MTL_TASK_TAG = "multitask"
MTL_TEMPLATES = ("shared_bottom", "aitm", "mmoe", "ple", "task_specific")


def _mtl_sparse_feature_names(bundle: MTLDatasetBundle) -> list[str]:
    return list(bundle.sparse_feature_names or bundle.feature_names[: len(bundle.vocab_sizes)])


def _mtl_num_dense_features(bundle: MTLDatasetBundle) -> int:
    return len(bundle.dense_feature_names or [])


def _mtl_num_total_fields(bundle: MTLDatasetBundle) -> int:
    return len(bundle.vocab_sizes) + _mtl_num_dense_features(bundle)


def _embedding_and_flatten_nodes(bundle: MTLDatasetBundle, config: MTLGenomeConfig) -> list[SkillNode]:
    nodes = [
        SkillNode(
            node_id="field_embedding",
            skill_id="field_embedding",
            skill_name="field_embedding",
            category="embedding",
            params={
                "vocab_sizes": bundle.vocab_sizes,
                "embedding_dim": config.embedding_dim,
                "feature_names": _mtl_sparse_feature_names(bundle),
            },
            input_keys=["sparse_features"],
            output_keys=["field_embeddings"],
            task_types=[MTL_TASK_TAG],
        ),
    ]
    if _mtl_num_dense_features(bundle) > 0:
        nodes.append(
            SkillNode(
                node_id="dense_feature_path",
                skill_id="dense_feature_path",
                skill_name="dense_feature_path",
                category="utility",
                params={
                    "num_dense": _mtl_num_dense_features(bundle),
                    "embedding_dim": config.embedding_dim,
                    "input_key": "dense_values",
                    "output_key": "dense_embeddings",
                    "field_embeddings_key": "field_embeddings",
                    "merged_output_key": "field_embeddings",
                },
                input_keys=["dense_values", "field_embeddings"],
                output_keys=["dense_embeddings", "field_embeddings"],
                task_types=[MTL_TASK_TAG],
            )
        )
    nodes.append(
        SkillNode(
            node_id="flatten",
            skill_id="flatten_field_embeddings",
            skill_name="flatten_field_embeddings",
            category="utility",
            params={"input_key": "field_embeddings", "output_key": "flat_embeddings"},
            input_keys=["field_embeddings"],
            output_keys=["flat_embeddings"],
            task_types=[MTL_TASK_TAG],
        ),
    )
    return nodes


def _shared_transform_nodes(
    template: str,
    bundle: MTLDatasetBundle,
    config: MTLGenomeConfig,
    flat_input_dim: int,
) -> tuple[list[SkillNode], list[SkillEdge], int]:
    """Build the swappable shared-transform subgraph.

    Returns (nodes, edges, representation_dim). Every variant consumes
    ``flat_embeddings`` and produces ``task_representations`` [B, n_task, R].
    """
    n_task = bundle.n_task
    nodes: list[SkillNode] = []
    edges: list[SkillEdge] = []

    if template == "shared_bottom":
        rep_dim = config.shared_hidden_dims[-1]
        nodes.append(
            SkillNode(
                node_id="shared_bottom",
                skill_id="shared_bottom_tower",
                skill_name="shared_bottom_tower",
                category="multitask",
                params={
                    "input_dim": flat_input_dim,
                    "hidden_dims": list(config.shared_hidden_dims),
                    "input_key": "flat_embeddings",
                    "output_key": "shared_representation",
                    "activation": config.activation,
                    "dropout": config.dropout,
                },
                input_keys=["flat_embeddings"],
                output_keys=["shared_representation"],
                task_types=[MTL_TASK_TAG],
            )
        )
        nodes.append(
            SkillNode(
                node_id="task_replication",
                skill_id="task_replication",
                skill_name="task_replication",
                category="multitask",
                params={
                    "n_task": n_task,
                    "input_key": "shared_representation",
                    "output_key": "task_representations",
                },
                input_keys=["shared_representation"],
                output_keys=["task_representations"],
                task_types=[MTL_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "shared_bottom", "flat_embeddings", "flat_embeddings"))
        edges.append(SkillEdge("shared_bottom", "task_replication", "shared_representation", "shared_representation"))
        return nodes, edges, rep_dim

    if template == "task_specific":
        rep_dim = config.shared_hidden_dims[-1]
        nodes.append(
            SkillNode(
                node_id="task_specific_towers",
                skill_id="task_specific_towers",
                skill_name="task_specific_towers",
                category="multitask",
                params={
                    "input_dim": flat_input_dim,
                    "n_task": n_task,
                    "hidden_dims": list(config.shared_hidden_dims),
                    "input_key": "flat_embeddings",
                    "output_key": "task_representations",
                    "activation": config.activation,
                    "dropout": config.dropout,
                },
                input_keys=["flat_embeddings"],
                output_keys=["task_representations"],
                task_types=[MTL_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "task_specific_towers", "flat_embeddings", "flat_embeddings"))
        return nodes, edges, rep_dim

    if template == "aitm":
        rep_dim = config.shared_hidden_dims[-1]
        nodes.append(
            SkillNode(
                node_id="task_specific_towers",
                skill_id="task_specific_towers",
                skill_name="task_specific_towers",
                category="multitask",
                params={
                    "input_dim": flat_input_dim,
                    "n_task": n_task,
                    "hidden_dims": list(config.shared_hidden_dims),
                    "input_key": "flat_embeddings",
                    "output_key": "task_representations",
                    "activation": config.activation,
                    "dropout": config.dropout,
                },
                input_keys=["flat_embeddings"],
                output_keys=["task_representations"],
                task_types=[MTL_TASK_TAG],
            )
        )
        nodes.append(
            SkillNode(
                node_id="aitm_transfer",
                skill_id="aitm_transfer",
                skill_name="aitm_transfer",
                category="multitask",
                params={
                    "input_dim": rep_dim,
                    "n_task": n_task,
                    "input_key": "task_representations",
                    "output_key": "task_representations",
                },
                input_keys=["task_representations"],
                output_keys=["task_representations"],
                task_types=[MTL_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "task_specific_towers", "flat_embeddings", "flat_embeddings"))
        edges.append(SkillEdge("task_specific_towers", "aitm_transfer", "task_representations", "task_representations"))
        return nodes, edges, rep_dim

    if template == "mmoe":
        rep_dim = config.expert_dim
        nodes.append(
            SkillNode(
                node_id="mmoe_gate",
                skill_id="mmoe_gate",
                skill_name="mmoe_gate",
                category="multitask",
                params={
                    "input_dim": flat_input_dim,
                    "n_expert": config.num_experts,
                    "n_task": n_task,
                    "expert_hidden_dims": [*config.shared_hidden_dims[:-1], config.expert_dim] if config.shared_hidden_dims else [config.expert_dim],
                    "input_key": "flat_embeddings",
                    "output_key": "task_representations",
                    "activation": config.activation,
                    "dropout": config.dropout,
                },
                input_keys=["flat_embeddings"],
                output_keys=["task_representations"],
                task_types=[MTL_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "mmoe_gate", "flat_embeddings", "flat_embeddings"))
        return nodes, edges, rep_dim

    if template == "ple":
        rep_dim = config.expert_dim
        nodes.append(
            SkillNode(
                node_id="ple_gate",
                skill_id="ple_gate",
                skill_name="ple_gate",
                category="multitask",
                params={
                    "input_dim": flat_input_dim,
                    "n_task": n_task,
                    "n_expert_shared": config.ple_shared_experts,
                    "n_expert_specific": config.ple_specific_experts,
                    "expert_hidden_dims": [*config.shared_hidden_dims[:-1], config.expert_dim] if config.shared_hidden_dims else [config.expert_dim],
                    "input_key": "flat_embeddings",
                    "output_key": "task_representations",
                    "activation": config.activation,
                    "dropout": config.dropout,
                },
                input_keys=["flat_embeddings"],
                output_keys=["task_representations"],
                task_types=[MTL_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "ple_gate", "flat_embeddings", "flat_embeddings"))
        return nodes, edges, rep_dim

    raise ValueError(f"Unsupported MTL genome template: {template}")


def build_mtl_genome(bundle: MTLDatasetBundle, config: MTLGenomeConfig, template: str | None = None) -> SkillGenome:
    """Construct a complete MTL genome for the requested shared-transform template."""
    template = template or config.template
    if template not in MTL_TEMPLATES:
        raise ValueError(f"Unsupported MTL genome template: {template} (expected one of {MTL_TEMPLATES})")
    n_task = bundle.n_task
    flat_input_dim = _mtl_num_total_fields(bundle) * config.embedding_dim
    task_weights = list(config.task_weights) if config.task_weights else [1.0] * n_task
    if len(task_weights) != n_task:
        raise ValueError("task_weights length must match label_cols length")

    nodes = _embedding_and_flatten_nodes(bundle, config)
    transform_nodes, transform_edges, rep_dim = _shared_transform_nodes(template, bundle, config, flat_input_dim)
    nodes.extend(transform_nodes)

    nodes.append(
        SkillNode(
            node_id="task_tower",
            skill_id="task_tower",
            skill_name="task_tower",
            category="multitask",
            params={
                "input_dim": rep_dim,
                "task_types": list(bundle.task_types),
                "tower_hidden_dims": list(config.task_hidden_dims),
                "input_key": "task_representations",
                "output_key": "task_outputs",
                "activation": config.activation,
                "dropout": config.dropout,
            },
            input_keys=["task_representations"],
            output_keys=["task_outputs"],
            task_types=[MTL_TASK_TAG],
        )
    )
    nodes.append(
        SkillNode(
            node_id="loss",
            skill_id="multitask_loss",
            skill_name="multitask_loss",
            category="loss",
            params={
                "task_types": list(bundle.task_types),
                "weights": task_weights,
                "predictions_key": "task_outputs",
                "labels_key": "task_labels",
                "output_key": "loss",
            },
            input_keys=["task_outputs", "task_labels"],
            output_keys=["loss"],
            task_types=[MTL_TASK_TAG],
        )
    )

    edges = list(transform_edges)
    if _mtl_num_dense_features(bundle) > 0:
        edges.append(SkillEdge("field_embedding", "dense_feature_path", "field_embeddings", "field_embeddings"))
        edges.append(SkillEdge("dense_feature_path", "flatten", "field_embeddings", "field_embeddings"))
    else:
        edges.append(SkillEdge("field_embedding", "flatten", "field_embeddings", "field_embeddings"))
    # task_representations is produced by the shared transform's terminal node.
    transform_output_node = transform_nodes[-1].node_id
    edges.append(SkillEdge(transform_output_node, "task_tower", "task_representations", "task_representations"))
    edges.append(SkillEdge("task_tower", "loss", "task_outputs", "task_outputs"))

    genome = SkillGenome(
        nodes=nodes,
        edges=edges,
        objectives=[{"skill_id": "multitask_loss", "output_key": "loss"}],
        constraints=GenomeConstraints(
            task_types=[MTL_TASK_TAG],
            required_inputs=["sparse_features", *(["dense_values"] if _mtl_num_dense_features(bundle) > 0 else []), "task_labels"],
            required_outputs=["task_outputs", "loss"],
        ),
    )
    genome.metadata.tags = ["multitask", template, *bundle.task_names]
    genome.metadata.extras = {
        **genome.metadata.extras,
        "sparse_feature_names": _mtl_sparse_feature_names(bundle),
        "dense_feature_names": list(bundle.dense_feature_names or []),
        "num_sparse_fields": len(bundle.vocab_sizes),
        "num_dense_fields": _mtl_num_dense_features(bundle),
        "num_fields": _mtl_num_total_fields(bundle),
        "embedding_dim": config.embedding_dim,
        "flat_input_dim": flat_input_dim,
    }
    genome.record_mutation(
        GenomeMutation(
            mutation_type="mtl_baseline",
            child_genome_id=genome.metadata.genome_id,
            description=f"MTL baseline genome with {template} shared transform",
            details={"template": template, "representation_dim": rep_dim},
        )
    )
    return genome


# ---------------------------------------------------------------------------
# Trainer (per-task AUC)
# ---------------------------------------------------------------------------


class MTLGenomeTrainer(CTRGenomeTrainer):
    """Train MTL genomes and report per-task plus mean AUC.

    Reuses the parent fit/early-stop loop (which keys on the ``auc`` metric).
    The overridden :meth:`evaluate` returns ``auc`` = mean of per-task AUCs so the
    parent's best-checkpoint and objective logic works unchanged, and also exposes
    each task's AUC under ``auc__<task_name>`` for inspection.
    """

    def __init__(self, training: CTRTrainingConfig, task_names: list[str]) -> None:
        super().__init__(training)
        self.task_names = list(task_names)

    def _batch_to_device(self, x_dict: dict[str, torch.Tensor], y: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = {key: value.to(self.device) for key, value in x_dict.items()}
        labels = y.float()
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        batch["task_labels"] = labels.to(self.device)
        # Keep a generic `labels` alias so any single-task skill still resolves.
        batch["labels"] = batch["task_labels"]
        return batch

    def _train_one_epoch(self, model, optimizer, data_loader) -> float:
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
                raise RuntimeError("MTL genome did not produce a 'loss' output")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    def evaluate(self, model, data_loader) -> dict[str, Any]:
        model.eval()
        n_task = len(self.task_names)
        per_task_labels: list[list[float]] = [[] for _ in range(n_task)]
        per_task_preds: list[list[float]] = [[] for _ in range(n_task)]
        losses: list[float] = []
        with torch.no_grad():
            for batch_idx, (x_dict, y) in enumerate(data_loader):
                if self.training.max_eval_batches is not None and batch_idx >= self.training.max_eval_batches:
                    break
                batch = self._batch_to_device(x_dict, y)
                ctx = model(batch)
                outputs = ctx.get("task_outputs")
                if outputs is None:
                    raise RuntimeError("MTL genome did not produce 'task_outputs'")
                outputs = outputs.detach().cpu()
                labels = batch["task_labels"].detach().cpu()
                for task_idx in range(n_task):
                    per_task_preds[task_idx].extend(outputs[:, task_idx].reshape(-1).tolist())
                    per_task_labels[task_idx].extend(labels[:, task_idx].reshape(-1).tolist())
                loss = ctx.get(self.training.loss_key)
                if loss is not None:
                    losses.append(float(loss.detach().cpu()))

        metrics: dict[str, Any] = {}
        task_aucs: list[float] = []
        for task_idx, name in enumerate(self.task_names):
            target = np.asarray(per_task_labels[task_idx], dtype=np.float64)
            pred = np.clip(np.asarray(per_task_preds[task_idx], dtype=np.float64), 1e-7, 1 - 1e-7)
            auc = _safe_auc(target, pred)
            logloss = _safe_logloss(target, pred)
            metrics[f"auc__{name}"] = auc
            metrics[f"logloss__{name}"] = logloss
            if not math.isnan(auc):
                task_aucs.append(auc)
        mean_auc = float(np.mean(task_aucs)) if task_aucs else float("nan")
        metrics["auc"] = mean_auc
        metrics["mean_auc"] = mean_auc
        metrics["loss"] = float(np.mean(losses)) if losses else None
        metrics["num_examples"] = int(len(per_task_labels[0])) if per_task_labels else 0
        return metrics


# ---------------------------------------------------------------------------
# Candidate generation (template-based architecture search)
# ---------------------------------------------------------------------------


def _default_mtl_templates() -> list[dict[str, Any]]:
    return [
        {"name": "swap_aitm", "template": "aitm", "operation": "replace"},
        {"name": "swap_ple", "template": "ple", "operation": "replace"},
        {"name": "swap_mmoe", "template": "mmoe", "operation": "replace"},
        {"name": "swap_task_specific", "template": "task_specific", "operation": "replace"},
        {"name": "swap_shared_bottom", "template": "shared_bottom", "operation": "replace"},
        {"name": "mmoe_more_experts", "template": "mmoe", "operation": "specialize", "num_experts": 8},
        {"name": "ple_more_experts", "template": "ple", "operation": "specialize", "ple_specific_experts": 2, "ple_shared_experts": 2},
        {"name": "wider_shared_bottom", "template": "shared_bottom", "operation": "specialize", "shared_hidden_dims": [256, 128]},
        {"name": "deeper_task_tower", "template": "shared_bottom", "operation": "specialize", "task_hidden_dims": [64, 32]},
    ]


MTL_TEMPLATE_SKILL_IDS: dict[str, tuple[str, ...]] = {
    "shared_bottom": ("shared_bottom_tower", "task_replication"),
    "mmoe": ("mmoe_gate",),
    "ple": ("ple_gate",),
    "aitm": ("aitm_transfer", "task_specific_towers"),
    "task_specific": ("task_specific_towers",),
}


@dataclass(frozen=True)
class _MTLSkillChoice:
    source: str
    score: int
    name: str
    order: int
    template: dict[str, Any] | None = None
    skill_id: str | None = None
    card: SkillCard | None = None


def _genome_template(genome: SkillGenome) -> str:
    node_ids = genome.node_ids()
    if "aitm_transfer" in node_ids:
        return "aitm"
    if "mmoe_gate" in node_ids:
        return "mmoe"
    if "ple_gate" in node_ids:
        return "ple"
    if "task_specific_towers" in node_ids:
        return "task_specific"
    return "shared_bottom"


def _genome_config_from_parent(genome: SkillGenome, base: MTLGenomeConfig) -> MTLGenomeConfig:
    """Reconstruct an MTLGenomeConfig that reproduces ``genome``'s architecture."""
    cfg = replace(base)
    node_ids = genome.node_ids()
    try:
        embedding = genome.get_node("field_embedding")
        cfg.embedding_dim = int(embedding.params.get("embedding_dim", base.embedding_dim))
    except KeyError:
        pass
    if "shared_bottom" in node_ids:
        cfg.shared_hidden_dims = list(genome.get_node("shared_bottom").params.get("hidden_dims", base.shared_hidden_dims))
    elif "task_specific_towers" in node_ids:
        cfg.shared_hidden_dims = list(genome.get_node("task_specific_towers").params.get("hidden_dims", base.shared_hidden_dims))
    elif "mmoe_gate" in node_ids:
        node = genome.get_node("mmoe_gate")
        cfg.num_experts = int(node.params.get("n_expert", base.num_experts))
        dims = list(node.params.get("expert_hidden_dims", []))
        if dims:
            cfg.expert_dim = int(dims[-1])
            cfg.shared_hidden_dims = dims
    elif "ple_gate" in node_ids:
        node = genome.get_node("ple_gate")
        cfg.ple_shared_experts = int(node.params.get("n_expert_shared", base.ple_shared_experts))
        cfg.ple_specific_experts = int(node.params.get("n_expert_specific", base.ple_specific_experts))
        dims = list(node.params.get("expert_hidden_dims", []))
        if dims:
            cfg.expert_dim = int(dims[-1])
            cfg.shared_hidden_dims = dims
    if "task_tower" in node_ids:
        cfg.task_hidden_dims = list(genome.get_node("task_tower").params.get("tower_hidden_dims", base.task_hidden_dims))
    return cfg


def _candidate_from_template(
    parent: SkillGenome,
    bundle: MTLDatasetBundle,
    base_genome_config: MTLGenomeConfig,
    template: dict[str, Any],
    round_idx: int,
) -> CandidateSpec:
    target_template = str(template.get("template") or _genome_template(parent))
    cfg = _genome_config_from_parent(parent, base_genome_config)
    cfg.template = target_template
    # Apply per-template specialization overrides.
    for key in ("num_experts", "ple_shared_experts", "ple_specific_experts", "shared_hidden_dims", "task_hidden_dims", "expert_dim", "embedding_dim"):
        if key in template:
            setattr(cfg, key, template[key])

    child = build_mtl_genome(bundle, cfg, template=target_template)
    carried = _carry_forward_mtl_generated_insertions(parent, child)
    child.metadata.parent_genome_ids = list(dict.fromkeys([*child.metadata.parent_genome_ids, parent.metadata.genome_id]))
    child.metadata.lineage = list(dict.fromkeys([*child.metadata.lineage, parent.metadata.genome_id]))

    name = str(template.get("name", target_template))
    operation = str(template.get("operation", "replace"))
    candidate_id = f"round{round_idx}_{name}"
    rationale = f"MTL {operation}: shared transform -> {target_template} ({name})"
    if carried:
        rationale = f"{rationale}; carried {carried} generated MTL insertion(s) from parent"
    return CandidateSpec(
        candidate_id=candidate_id,
        genome=child,
        parent_genome_id=parent.metadata.genome_id,
        mutation_type=name,
        rationale=rationale,
        evolution_space="skill_space",
        operation=operation,
        architecture_fingerprint=_safe_architecture_fingerprint(child, fallback=candidate_id),
    )


def _mtl_reuse_promoted_generated_skills_enabled(evolution: CTREvolutionConfig) -> bool:
    code_space = evolution.code_space
    if isinstance(code_space, dict):
        return bool(code_space.get("reuse_promoted_generated_skills", code_space.get("reuse_promoted_code_skills", False)))
    return bool(getattr(code_space, "reuse_promoted_generated_skills", getattr(code_space, "reuse_promoted_code_skills", False)))


def _is_mtl_reusable_generated_skill_card(card: SkillCard, parents: list[SkillGenome]) -> bool:
    if not _is_reusable_generated_skill_card(card):
        return False
    if not _skill_card_has_loadable_implementation(card):
        return False
    if not _generated_skill_portability_allows_reuse(card):
        return False
    return any(_mtl_reuse_lane_for_card(card, parent) is not None for parent in parents)


def _score_mtl_reusable_generated_card(card: SkillCard, parents: list[SkillGenome]) -> int:
    score = 20
    tasks = {str(task).lower() for task in card.task_types}
    retrieval_tasks = {str(task).lower() for task in (card.manifest.get("retrieval") or {}).get("task_types", [])}
    if "multitask" in tasks or "multitask" in retrieval_tasks:
        score += 20
    text = _mtl_card_text(card)
    for token in ["task", "shared", "gate", "routing", "representation", "flat", "field"]:
        if token in text:
            score += 2
    if any(_mtl_reuse_lane_for_card(card, parent) == "task_representations" for parent in parents):
        score += 6
    metrics = card.manifest.get("candidate_metrics") or (card.manifest.get("metadata") or {}).get("candidate_metrics") or {}
    try:
        score += min(10, max(0, int(round(float(metrics.get("validation_best_auc", metrics.get("auc", 0.0))) * 10))))
    except Exception:
        pass
    return score


def _build_unified_mtl_skill_choices(
    evolution: CTREvolutionConfig,
    parents: list[SkillGenome],
    *,
    target: int,
    round_idx: int,
    skill_library: SkillLibrary | None = None,
) -> tuple[list[_MTLSkillChoice], SkillLibrary | None]:
    """Return one scored MTL skill-space pool covering predefined and generated skills."""
    library = _mtl_choice_skill_library(evolution, skill_library)
    choices: list[_MTLSkillChoice] = []
    for idx, template in enumerate(_mtl_predefined_template_pool(evolution, target=target)):
        template = dict(template)
        template_name = str(template.get("name") or template.get("template") or f"template_{idx}")
        primary_skill_id = _mtl_primary_skill_id_for_template(template)
        choices.append(
            _MTLSkillChoice(
                source="predefined",
                score=_score_mtl_predefined_template(template, parents, library),
                name=template_name,
                order=idx,
                template=template,
                skill_id=primary_skill_id,
                card=_mtl_template_primary_card(template, library),
            )
        )
    if library is not None and _mtl_reuse_promoted_generated_skills_enabled(evolution):
        choices.extend(_mtl_reusable_generated_skill_choices(library, parents, round_idx=round_idx))
    return choices, library


def _mtl_choice_skill_library(evolution: CTREvolutionConfig, skill_library: SkillLibrary | None) -> SkillLibrary | None:
    if skill_library is not None:
        return skill_library
    try:
        return SkillLibrary.from_repo(include_generated=_mtl_reuse_promoted_generated_skills_enabled(evolution))
    except Exception:
        return None


def _mtl_predefined_template_pool(evolution: CTREvolutionConfig, *, target: int) -> list[dict[str, Any]]:
    desired = max(target, len(evolution.templates or []) + len(_default_mtl_templates()))
    templates: list[dict[str, Any]] = []
    seen_default_names: set[str] = set()
    for template in evolution.templates or []:
        item = dict(template)
        item["_mtl_configured_template"] = True
        templates.append(item)
        name = str(item.get("name") or "")
        if name:
            seen_default_names.add(name)
    defaults = _default_mtl_templates()
    for template in defaults:
        name = str(template.get("name") or "")
        if name in seen_default_names:
            continue
        item = dict(template)
        item["_mtl_configured_template"] = False
        templates.append(item)
        seen_default_names.add(name)
    idx = 0
    while len(templates) < desired and defaults:
        item = dict(defaults[idx % len(defaults)])
        item["_mtl_configured_template"] = False
        templates.append(item)
        idx += 1
    return templates


def _mtl_primary_skill_id_for_template(template: dict[str, Any]) -> str:
    target_template = str(template.get("template") or "")
    skill_ids = MTL_TEMPLATE_SKILL_IDS.get(target_template) or ()
    return skill_ids[0] if skill_ids else target_template


def _mtl_template_primary_card(template: dict[str, Any], library: SkillLibrary | None) -> SkillCard | None:
    if library is None:
        return None
    primary_skill_id = _mtl_primary_skill_id_for_template(template)
    try:
        return library.get(primary_skill_id) if library.has(primary_skill_id) else None
    except Exception:
        return None


def _mtl_template_cards(template: dict[str, Any], library: SkillLibrary | None) -> list[SkillCard]:
    if library is None:
        return []
    cards: list[SkillCard] = []
    for skill_id in MTL_TEMPLATE_SKILL_IDS.get(str(template.get("template") or ""), ()):
        try:
            if library.has(skill_id):
                cards.append(library.get(skill_id))
        except Exception:
            continue
    return cards


def _score_mtl_predefined_template(template: dict[str, Any], parents: list[SkillGenome], library: SkillLibrary | None) -> int:
    score = 20
    target_template = str(template.get("template") or "")
    operation = str(template.get("operation") or "")
    if template.get("_mtl_configured_template"):
        score += 8
    if operation == "replace":
        score += 4
    elif operation == "specialize":
        score += 5
    if target_template in {"aitm", "ple"}:
        score += 7
    elif target_template == "mmoe":
        score += 6
    elif target_template == "task_specific":
        score += 5
    elif target_template == "shared_bottom":
        score += 3
    parent_templates = {_genome_template(parent) for parent in parents}
    if target_template and target_template not in parent_templates:
        score += 4
    elif operation != "specialize":
        score -= 2
    cards = _mtl_template_cards(template, library)
    if cards:
        if any("multitask" in {str(task).lower() for task in card.task_types} for card in cards):
            score += 8
        text = " ".join(_mtl_card_text(card) for card in cards)
        token_score = 0
        for token in ["task", "shared", "gate", "expert", "representation", "routing", "specific", "bottom"]:
            if token in text:
                token_score += 2
        score += min(10, token_score)
        parent_skill_ids = {node.skill_id for parent in parents for node in parent.nodes}
        for card in cards:
            composition = card.manifest.get("composition") or {}
            upstream = {str(item) for item in composition.get("common_upstream", []) or []}
            downstream = {str(item) for item in composition.get("common_downstream", []) or []}
            if upstream & parent_skill_ids:
                score += 2
            if downstream & {"task_tower", "task_replication"}:
                score += 3
    return score


def _mtl_reusable_generated_skill_choices(
    library: SkillLibrary,
    parents: list[SkillGenome],
    *,
    round_idx: int,
) -> list[_MTLSkillChoice]:
    scored: list[tuple[int, str, SkillCard]] = []
    for skill_id in library.list_skill_ids():
        try:
            card = library.get(skill_id)
        except Exception:
            continue
        if not _is_mtl_reusable_generated_skill_card(card, parents):
            continue
        scored.append((_score_mtl_reusable_generated_card(card, parents), skill_id, card))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1]))
    shift = round_idx % len(scored)
    rotated = scored[shift:] + scored[:shift]
    return [
        _MTLSkillChoice(
            source="generated",
            score=score,
            name=skill_id,
            order=idx,
            skill_id=skill_id,
            card=card,
        )
        for idx, (score, skill_id, card) in enumerate(rotated)
    ]


def _rank_mtl_skill_choices(choices: list[_MTLSkillChoice], *, target: int) -> list[_MTLSkillChoice]:
    source_priority = {"predefined": 0, "generated": 1}
    ranked = sorted(
        choices,
        key=lambda choice: (
            -choice.score,
            source_priority.get(choice.source, 9),
            choice.order,
            choice.name,
        ),
    )
    return _ensure_mtl_skill_source_mix(ranked, target=target)


def _ensure_mtl_skill_source_mix(choices: list[_MTLSkillChoice], *, target: int) -> list[_MTLSkillChoice]:
    if target <= 1 or len(choices) <= target:
        return choices
    available_sources = {choice.source for choice in choices}
    required_sources = [source for source in ("predefined", "generated") if source in available_sources]
    if len(required_sources) <= 1:
        return choices
    selected = list(choices[:target])
    tail = list(choices[target:])
    for source in required_sources:
        if any(choice.source == source for choice in selected):
            continue
        replacement_idx = next((idx for idx, choice in enumerate(tail) if choice.source == source), None)
        if replacement_idx is None:
            continue
        replace_pos = _mtl_replaceable_selected_choice_index(selected)
        replacement = tail.pop(replacement_idx)
        displaced = selected[replace_pos]
        selected[replace_pos] = replacement
        tail.insert(0, displaced)
    return selected + tail


def _mtl_replaceable_selected_choice_index(selected: list[_MTLSkillChoice]) -> int:
    counts: dict[str, int] = {}
    for choice in selected:
        counts[choice.source] = counts.get(choice.source, 0) + 1
    for idx in range(len(selected) - 1, -1, -1):
        if counts.get(selected[idx].source, 0) > 1:
            return idx
    return len(selected) - 1


def _mtl_parent_for_skill_choice(
    choice: _MTLSkillChoice,
    parents: list[SkillGenome],
    *,
    round_idx: int,
    choice_idx: int,
) -> tuple[SkillGenome, int]:
    if not parents:
        raise ValueError("No MTL parents available for candidate generation")
    if choice.source == "generated" and choice.card is not None:
        for offset in range(len(parents)):
            parent_idx = (round_idx + choice_idx + offset) % len(parents)
            if _mtl_reuse_lane_for_card(choice.card, parents[parent_idx]) is not None:
                return parents[parent_idx], parent_idx
    parent_idx = choice_idx % len(parents)
    return parents[parent_idx], parent_idx


def _candidate_from_mtl_skill_choice(
    choice: _MTLSkillChoice,
    parent: SkillGenome,
    bundle: MTLDatasetBundle,
    base_genome_config: MTLGenomeConfig,
    *,
    round_idx: int,
    library: SkillLibrary | None,
) -> CandidateSpec:
    if choice.source == "generated":
        if choice.card is None or library is None:
            raise ValueError("Generated MTL skill choice requires a skill card and library")
        spec = _candidate_from_mtl_promoted_generated_skill(parent, choice.card, round_idx=round_idx, library=library)
    else:
        if choice.template is None:
            raise ValueError("Predefined MTL skill choice requires a template")
        spec = _candidate_from_template(parent, bundle, base_genome_config, choice.template, round_idx)
    spec.rationale = f"{spec.rationale} (unified MTL skill pool: source={choice.source}, score={choice.score})"
    return spec


def _candidate_from_mtl_promoted_generated_skill(
    parent: SkillGenome,
    card: SkillCard,
    *,
    round_idx: int,
    library: SkillLibrary,
) -> CandidateSpec:
    lane = _mtl_reuse_lane_for_card(card, parent)
    if lane is None:
        raise ValueError(f"Promoted generated skill is not compatible with this MTL parent: {card.skill_id}")
    input_key = lane
    output_key = _resolved_mtl_generated_output_key(card, parent, round_idx=round_idx, input_key=input_key)
    params = _mtl_generated_skill_params(card, parent, input_key=input_key, output_key=output_key)
    base_id = _mtl_slug(card.skill_id)
    node = SkillNode(
        node_id=_unique_mtl_node_id(parent, f"{base_id}_r{round_idx}"),
        skill_id=card.skill_id,
        skill_name=card.skill_name,
        category=card.category or "generated",
        params=params,
        input_keys=[input_key],
        output_keys=[output_key],
        task_types=[MTL_TASK_TAG],
        source="generated_skill",
        metadata={
            "evolution_space": "skill_space",
            "reused_generated_skill": True,
            "source_proposal_id": card.manifest.get("source_proposal_id"),
            "runtime_skill_card_path": str(card.path) if card.path else "",
        },
    )
    child = parent.clone()
    if not _insert_mtl_generated_node_on_lane(child, node, input_key=input_key, output_key=output_key):
        raise ValueError(f"Could not insert promoted generated skill on MTL lane: {input_key}")
    GenomeVerifier(skill_library=library).assert_valid(child)
    child.record_mutation(
        GenomeMutation(
            mutation_type=f"mtl_skill_reuse_generated_{card.skill_id}",
            parent_genome_id=parent.metadata.genome_id,
            child_genome_id=child.metadata.genome_id,
            description=f"Reuse promoted generated skill {card.skill_id} from the global generated-skill library",
            details={"skill_id": card.skill_id, "lane": input_key, "output_key": output_key},
        )
    )
    candidate_id = f"round{round_idx}_skill_reuse_{base_id}"
    return CandidateSpec(
        candidate_id=candidate_id,
        genome=child,
        parent_genome_id=parent.metadata.genome_id,
        mutation_type=f"mtl_skill_reuse_generated_{card.skill_id}",
        rationale=f"Reuse promoted generated skill {card.skill_id} on MTL {input_key} lane",
        evolution_space="skill_space",
        operation="reuse_generated",
        architecture_fingerprint=_safe_architecture_fingerprint(child, fallback=candidate_id),
        generated_skill_id=card.skill_id,
        proposal_id=card.manifest.get("source_proposal_id"),
    )


def _mtl_reuse_lane_for_card(card: SkillCard, parent: SkillGenome) -> str | None:
    input_keys = _resolved_mtl_generated_input_keys(card, parent)
    if len(input_keys) != 1:
        return None
    input_key = input_keys[0]
    if input_key not in {"field_embeddings", "flat_embeddings", "task_representations"}:
        return None
    if any(_is_mtl_logit_key(key) for key in [input_key, *_resolved_mtl_generated_output_keys(card)]):
        return None
    if _current_mtl_lane_edge(parent, input_key) is None:
        return None
    if not _mtl_generated_shape_compatible(card, parent, lane=input_key):
        return None
    return input_key


def _resolved_mtl_generated_input_keys(card: SkillCard, parent: SkillGenome) -> list[str]:
    available = _available_mtl_tensor_keys(parent)
    keys = [key for key in card.input_keys if key and not _is_mtl_param_placeholder(key)]
    if keys:
        return keys if set(keys) <= available else []
    params = _mtl_generated_fragment_params(card)
    for param_name in ("input_key", "flat_key", "field_key", "context_key"):
        value = str(params.get(param_name) or "")
        if value in available:
            return [value]
    text = _mtl_card_text(card)
    for key in ("task_representations", "flat_embeddings", "field_embeddings"):
        if key in available and key in text:
            return [key]
    return []


def _resolved_mtl_generated_output_keys(card: SkillCard) -> list[str]:
    keys = [key for key in card.output_keys if key and not _is_mtl_param_placeholder(key)]
    if keys:
        return keys
    params = _mtl_generated_fragment_params(card)
    for param_name in ("output_key", "flat_output_key", "field_output_key"):
        value = str(params.get(param_name) or "")
        if value and not _is_mtl_param_placeholder(value):
            return [value]
    return []


def _resolved_mtl_generated_output_key(card: SkillCard, parent: SkillGenome, *, round_idx: int, input_key: str) -> str:
    output_keys = _resolved_mtl_generated_output_keys(card)
    output_key = output_keys[0] if len(output_keys) == 1 else ""
    if not output_key or output_key == input_key or output_key in parent.produced_keys():
        output_key = _unique_mtl_key(parent, f"{_mtl_slug(card.skill_id)}_r{round_idx}_{input_key}")
    return output_key


def _mtl_generated_skill_params(card: SkillCard, parent: SkillGenome, *, input_key: str, output_key: str) -> dict[str, Any]:
    params = _mtl_generated_fragment_params(card)
    params["input_key"] = input_key
    params["output_key"] = output_key
    if input_key == "flat_embeddings" and "flat_key" in params:
        params["flat_key"] = input_key
    if input_key == "field_embeddings" and "field_key" in params:
        params["field_key"] = input_key
    if input_key == "task_representations" and "task_key" in params:
        params["task_key"] = input_key
    for key, value in list(params.items()):
        params[key] = _resolve_mtl_generated_param_value(value, parent)
    return params


def _mtl_generated_fragment_params(card: SkillCard) -> dict[str, Any]:
    fragment = (card.manifest.get("composition") or {}).get("example_genome_fragment") or {}
    if isinstance(fragment, str):
        try:
            fragment = yaml.safe_load(fragment) or {}
        except Exception:
            fragment = {}
    return dict(fragment.get("params") or {}) if isinstance(fragment, dict) else {}


def _resolve_mtl_generated_param_value(value: Any, parent: SkillGenome) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    runtime = _mtl_runtime_shape(parent)
    if normalized in {"${num_fields}", "${field_count}"}:
        return runtime["num_fields"]
    if normalized == "${embedding_dim}":
        return runtime["embedding_dim"]
    if normalized in {"${input_dim}", "${flat_input_dim}", "${fusion_dim}"}:
        return runtime["flat_input_dim"]
    if normalized in {"${n_task}", "${num_tasks}"}:
        return runtime["n_task"]
    if normalized in {"${task_representation_dim}", "${representation_dim}"}:
        return runtime["task_representation_dim"]
    return value


def _mtl_generated_shape_compatible(card: SkillCard, parent: SkillGenome, *, lane: str) -> bool:
    runtime = _mtl_runtime_shape(parent)
    params = _mtl_generated_fragment_params(card)
    for name, current_value in [
        ("num_fields", runtime["num_fields"]),
        ("embedding_dim", runtime["embedding_dim"]),
        ("input_dim", runtime["flat_input_dim"]),
    ]:
        if not _mtl_static_shape_value_compatible(params.get(name), current_value):
            return False
    input_shape = _mtl_signature_shape(card.manifest.get("input_signature") or card.manifest.get("inputs") or [], lane)
    output_shape = _mtl_first_signature_shape(card.manifest.get("output_signature") or card.manifest.get("outputs") or [])
    if input_shape and not _mtl_shape_spec_compatible(input_shape, name=lane, current=runtime):
        return False
    if output_shape:
        if not _mtl_shape_spec_compatible(output_shape, name=lane, current=runtime):
            return False
        if lane == "field_embeddings" and len(output_shape) != 3:
            return False
        if lane == "flat_embeddings" and len(output_shape) != 2:
            return False
        if lane == "task_representations" and len(output_shape) != 3:
            return False
        if len(output_shape) == 2 and _mtl_shape_dim_is_static_one(output_shape[-1]):
            return False
        return True
    text = " ".join([_mtl_card_text(card), " ".join(_resolved_mtl_generated_output_keys(card))])
    if lane == "field_embeddings":
        return "field" in text and "embedding" in text
    if lane == "flat_embeddings":
        return "flat" in text or "representation" in text
    return "task" in text and "representation" in text


def _mtl_runtime_shape(parent: SkillGenome) -> dict[str, int]:
    extras = parent.metadata.extras or {}
    embedding_dim = int(extras.get("embedding_dim") or _mtl_embedding_dim(parent))
    num_fields = int(extras.get("num_fields") or (_mtl_sparse_field_count(parent) + _mtl_dense_field_count(parent)))
    task_representation_dim = 1
    try:
        task_tower = parent.get_node("task_tower")
        task_representation_dim = int(task_tower.params.get("input_dim") or task_representation_dim)
    except Exception:
        pass
    return {
        "num_fields": max(1, num_fields),
        "embedding_dim": max(1, embedding_dim),
        "flat_input_dim": max(1, num_fields) * max(1, embedding_dim),
        "n_task": _mtl_task_count(parent),
        "task_representation_dim": max(1, task_representation_dim),
    }


def _mtl_sparse_field_count(parent: SkillGenome) -> int:
    try:
        return len(parent.get_node("field_embedding").params.get("vocab_sizes") or [])
    except Exception:
        return 0


def _mtl_dense_field_count(parent: SkillGenome) -> int:
    try:
        return int(parent.get_node("dense_feature_path").params.get("num_dense") or 0)
    except Exception:
        return 0


def _mtl_embedding_dim(parent: SkillGenome) -> int:
    try:
        return int(parent.get_node("field_embedding").params.get("embedding_dim") or 1)
    except Exception:
        return 1


def _mtl_task_count(parent: SkillGenome) -> int:
    try:
        return len(parent.get_node("task_tower").params.get("task_types") or [])
    except Exception:
        return 1


def _mtl_signature_shape(signature: Any, name: str) -> list[Any]:
    if not isinstance(signature, list):
        return []
    for item in signature:
        if isinstance(item, dict) and str(item.get("name") or "") == name:
            shape = item.get("shape") or []
            return list(shape) if isinstance(shape, list) else _shape_list_from_string(shape)
    return []


def _mtl_first_signature_shape(signature: Any) -> list[Any]:
    if not isinstance(signature, list) or not signature or not isinstance(signature[0], dict):
        return []
    shape = signature[0].get("shape") or []
    return list(shape) if isinstance(shape, list) else _shape_list_from_string(shape)


def _shape_list_from_string(shape: Any) -> list[str]:
    if not isinstance(shape, str):
        return []
    return [part.strip() for part in shape.strip().strip("[]").split(",") if part.strip()]


def _mtl_shape_spec_compatible(shape: list[Any], *, name: str, current: dict[str, int]) -> bool:
    if not shape:
        return True
    if len(shape) == 3 and name == "field_embeddings":
        return _mtl_shape_dim_compatible(shape[1], current["num_fields"]) and _mtl_shape_dim_compatible(shape[2], current["embedding_dim"])
    if len(shape) == 2 and name == "flat_embeddings":
        return _mtl_shape_dim_compatible(shape[1], current["flat_input_dim"])
    if len(shape) == 3 and name == "task_representations":
        return _mtl_shape_dim_compatible(shape[1], current["n_task"]) and _mtl_shape_dim_compatible(shape[2], current["task_representation_dim"])
    return True


def _mtl_shape_dim_compatible(dim: Any, current_value: int) -> bool:
    if isinstance(dim, str):
        dim = dim.strip()
        if dim in {"num_fields", "embedding_dim", "input_dim", "flat_input_dim", "n_task", "num_tasks", "task_representation_dim"}:
            return True
        if dim.startswith("${") or not dim.isdigit():
            return True
    try:
        return int(dim) == int(current_value)
    except Exception:
        return True


def _mtl_static_shape_value_compatible(value: Any, current_value: int) -> bool:
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


def _mtl_shape_dim_is_static_one(dim: Any) -> bool:
    try:
        return int(dim) == 1
    except Exception:
        return False


def _available_mtl_tensor_keys(parent: SkillGenome) -> set[str]:
    return set(parent.constraints.required_inputs) | parent.produced_keys() | {"sparse_features", "dense_values", "task_labels"}


def _is_mtl_param_placeholder(key: str) -> bool:
    return key in {"input_key", "input_keys", "output_key", "output_keys", "logits_key", "labels_key"}


def _is_mtl_logit_key(key: str) -> bool:
    lowered = str(key).lower()
    return lowered == "logits" or "logit" in lowered


def _mtl_card_text(card: SkillCard) -> str:
    manifest = card.manifest
    retrieval = manifest.get("retrieval") or {}
    values = [
        card.skill_id,
        card.category,
        manifest.get("description"),
        manifest.get("function_description"),
        manifest.get("inductive_bias"),
        manifest.get("failure_signatures"),
        manifest.get("failure_modes_addressed"),
        retrieval.get("summary"),
        retrieval.get("use_when"),
        retrieval.get("architecture_roles"),
        retrieval.get("input_modalities"),
        retrieval.get("output_semantics"),
        retrieval.get("objectives"),
        retrieval.get("model_families"),
        retrieval.get("aliases"),
    ]
    return _lower_mtl_text(values)


def _lower_mtl_text(values: list[Any]) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            parts.append(_lower_mtl_text(list(value.values())))
        elif isinstance(value, (list, tuple, set)):
            parts.append(_lower_mtl_text(list(value)))
        else:
            parts.append(str(value))
    return " ".join(parts).lower()


def _mtl_slug(value: str) -> str:
    slug = "".join(char if char.isalnum() or char == "_" else "_" for char in str(value).lower()).strip("_")
    return slug or "generated_skill"


def _carry_forward_mtl_generated_insertions(parent: SkillGenome, child: SkillGenome) -> int:
    """Carry shape-preserving generated insert_between nodes into rebuilt MTL templates.

    Template mutations rebuild the fixed MTL scaffold, so generated insertions from
    a code-space survivor would otherwise be dropped. We only carry generated
    single-input/single-output nodes that intercept the stable MTL tensor lanes:
    ``field_embeddings``, ``flat_embeddings``, and ``task_representations``.
    """
    carried = 0
    for node in parent.nodes:
        if not _is_carryable_mtl_generated_node(node):
            continue
        input_key = node.input_keys[0] if len(node.input_keys) == 1 else ""
        output_key = node.output_keys[0] if len(node.output_keys) == 1 else ""
        if input_key not in {"field_embeddings", "flat_embeddings", "task_representations"} or not output_key:
            continue
        snapshot = SkillGenome.from_dict(copy.deepcopy(child.to_dict()))
        try:
            if _insert_mtl_generated_node_on_lane(child, node, input_key=input_key, output_key=output_key):
                child.assert_valid_graph()
                carried += 1
            else:
                child.nodes = snapshot.nodes
                child.edges = snapshot.edges
        except Exception:
            child.nodes = snapshot.nodes
            child.edges = snapshot.edges
    if carried:
        child.record_mutation(
            GenomeMutation(
                mutation_type="carry_forward_generated_mtl_insertions",
                parent_genome_id=parent.metadata.genome_id,
                child_genome_id=child.metadata.genome_id,
                description="Carry forward compatible generated MTL insert_between nodes into rebuilt template genome",
                details={"carried_count": carried},
            )
        )
    return carried


def _is_carryable_mtl_generated_node(node: SkillNode) -> bool:
    if node.source == "generated_skill":
        return True
    metadata = node.metadata or {}
    return any(
        metadata.get(key)
        for key in (
            "runtime_skill_card_path",
            "staged_skill_card_path",
            "persisted_skill_card_path",
            "generated_code_hash",
        )
    )


def _insert_mtl_generated_node_on_lane(
    genome: SkillGenome,
    node: SkillNode,
    *,
    input_key: str,
    output_key: str,
) -> bool:
    if output_key in genome.produced_keys():
        return False
    edge = _current_mtl_lane_edge(genome, input_key)
    if edge is None:
        return False
    cloned = SkillNode.from_dict(copy.deepcopy(asdict(node)))
    cloned.node_id = _unique_mtl_node_id(genome, cloned.node_id)
    genome.nodes.append(cloned)
    genome.edges = [item for item in genome.edges if item != edge]
    genome.edges.append(SkillEdge(edge.src_node_id, cloned.node_id, edge.src_output_key, input_key, edge.tensor_semantics))
    genome.edges.append(SkillEdge(cloned.node_id, edge.dst_node_id, output_key, edge.dst_input_key, edge.tensor_semantics))
    return True


def _current_mtl_lane_edge(genome: SkillGenome, tensor_key: str) -> SkillEdge | None:
    if tensor_key == "field_embeddings":
        return next(
            (
                edge
                for edge in genome.edges
                if edge.dst_node_id == "flatten" and edge.dst_input_key == "field_embeddings"
            ),
            None,
        )
    if tensor_key == "flat_embeddings":
        return next(
            (
                edge
                for edge in genome.edges
                if edge.src_node_id == "flatten" and edge.src_output_key == "flat_embeddings"
            ),
            None,
        )
    if tensor_key == "task_representations":
        return next(
            (
                edge
                for edge in genome.edges
                if edge.dst_node_id == "task_tower" and edge.dst_input_key == "task_representations"
            ),
            None,
        )
    return None


def _unique_mtl_node_id(genome: SkillGenome, prefix: str) -> str:
    node_id = prefix
    idx = 1
    existing = genome.node_ids()
    while node_id in existing:
        node_id = f"{prefix}_{idx}"
        idx += 1
    return node_id


def _unique_mtl_key(genome: SkillGenome, prefix: str) -> str:
    key = prefix
    idx = 1
    existing = set(genome.constraints.required_inputs) | genome.produced_keys()
    while key in existing:
        key = f"{prefix}_{idx}"
        idx += 1
    return key


def _mtl_prompt_result_summary(result: CandidateResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    return {
        "candidate_id": result.candidate_id,
        "status": result.status,
        "evolution_space": result.evolution_space,
        "operation": result.operation,
        "mutation_type": result.mutation_type,
        "generated_skill_id": result.generated_skill_id or "",
        "proposal_id": result.proposal_id or "",
        "structural_scope": result.structural_scope,
        "parent_genome_id": result.parent_genome_id or "",
        "validation_best_auc": result.metrics.get("validation_best_auc"),
        "test_mean_auc": result.metrics.get("auc"),
        "metrics": _mtl_prompt_metric_subset(result.metrics),
        "error": _mtl_prompt_short_text(result.error or "", limit=600),
        "rationale": _mtl_prompt_short_text(result.rationale or "", limit=400),
    }


def _mtl_prompt_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    exact_keys = {
        "validation_best_auc",
        "validation_auc",
        "validation_mean_auc",
        "validation_loss",
        "auc",
        "mean_auc",
        "logloss",
        "loss",
    }
    prefixes = (
        "validation_auc__",
        "validation_logloss__",
        "auc__",
        "logloss__",
    )
    for key, value in (metrics or {}).items():
        if key in exact_keys or any(str(key).startswith(prefix) for prefix in prefixes):
            selected[key] = value
    return selected


def _mtl_prompt_recent_rounds(results: list[CandidateResult], *, limit: int = 5) -> list[dict[str, Any]]:
    groups = _round_result_groups(results)
    rows: list[dict[str, Any]] = []
    for round_idx in sorted(groups)[-limit:]:
        group = groups[round_idx]
        best = _best_result_by_validation_auc(group)
        rows.append(
            {
                "round_idx": round_idx,
                "counts": _round_space_counts(group),
                "best_candidate": _mtl_prompt_result_summary(best) if best else {},
                "failed_candidates": [
                    _mtl_prompt_result_summary(result)
                    for result in group
                    if result.error or result.status == "failed"
                ][:6],
                "code_space_candidates": [
                    _mtl_prompt_result_summary(result)
                    for result in group
                    if result.evolution_space == "code_space"
                ][:6],
            }
        )
    return rows


def _mtl_prompt_recent_failures(results: list[CandidateResult], *, limit: int = 12) -> list[dict[str, Any]]:
    failures = [
        result
        for result in results
        if result.error or result.status == "failed"
    ]
    return [_mtl_prompt_result_summary(result) for result in failures[-limit:]]


def _mtl_prompt_short_text(text: str, *, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def generate_mtl_candidates(
    parents: list[SkillGenome],
    bundle: MTLDatasetBundle,
    base_genome_config: MTLGenomeConfig,
    evolution: CTREvolutionConfig,
    round_idx: int,
    skill_library: SkillLibrary | None = None,
) -> list[CandidateSpec]:
    target = max(0, int(evolution.candidate_budget))
    if target <= 0 or not parents:
        return []
    specs: list[CandidateSpec] = []
    seen: set[str] = set()
    choices, library = _build_unified_mtl_skill_choices(
        evolution,
        parents,
        target=target,
        round_idx=round_idx,
        skill_library=skill_library,
    )
    for idx, choice in enumerate(_rank_mtl_skill_choices(choices, target=target)):
        parent, parent_idx = _mtl_parent_for_skill_choice(choice, parents, round_idx=round_idx, choice_idx=idx)
        try:
            spec = _candidate_from_mtl_skill_choice(
                choice,
                parent,
                bundle,
                base_genome_config,
                round_idx=round_idx,
                library=library,
            )
        except Exception:
            continue
        if len(parents) > 1:
            spec.candidate_id = f"{spec.candidate_id}_p{parent_idx}"
        spec.candidate_id = _dedupe_candidate_id(spec.candidate_id, seen)
        specs.append(spec)
        if len(specs) >= target:
            break
    return specs[:target]


def _effective_mtl_templates(evolution: CTREvolutionConfig, *, target: int) -> list[dict[str, Any]]:
    if target <= 0:
        return []
    provided = [dict(template) for template in (evolution.templates or [])]
    if not provided:
        provided = []
    filled = provided[:target]
    seen_names = {str(template.get("name") or "") for template in filled}
    defaults = _default_mtl_templates()
    for template in defaults:
        if len(filled) >= target:
            break
        name = str(template.get("name") or "")
        if name in seen_names:
            continue
        filled.append(dict(template))
        seen_names.add(name)
    idx = 0
    while len(filled) < target and defaults:
        filled.append(dict(defaults[idx % len(defaults)]))
        idx += 1
    return filled[:target]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class MTLModelEvolutionRunner(CTRModelEvolutionRunner):
    """MTL evolution runner reusing CTR infrastructure with MTL-specific data,
    genome, trainer, and candidate generation."""

    def __init__(self, config: MTLWorkflowConfig) -> None:
        # Bypass CTRModelEvolutionRunner.__init__ (which builds CTR-specific state)
        # and reproduce the parts we need against the MTL config.
        self.config = config
        if not getattr(self.config.evolution.code_space, "reuse_promoted_generated_skills", False):
            self.config.evolution.code_space.reuse_promoted_generated_skills = True
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        from .evolution_memory import EvolutionMemory

        self.memory = EvolutionMemory(
            config.memory_path or self.output_dir / "evolution_memory.jsonl",
            persist=config.write_evolution_memory,
        )
        self.bundle: MTLDatasetBundle | None = None
        self.trainer: MTLGenomeTrainer | None = None
        self.candidate_parallelism: list[dict[str, Any]] = []
        self.candidate_generation_errors: list[dict[str, Any]] = []
        self.deduplication_records: list[dict[str, Any]] = []
        self.code_space_generation_records: list[dict[str, Any]] = []
        self.completed_results: list[CandidateResult] = []
        self._temporary_staging = None
        self._code_space_staging_root = config.evolution.code_space.staging_root
        self.training_config_hash = config_fingerprint(asdict(config.training))
        self.genome_config_hash = config_fingerprint(asdict(config.genome))
        self.hparam_fingerprint = config_fingerprint({"training": asdict(config.training), "genome": asdict(config.genome)})
        self.hparam_changes = {
            "training": diff_payload(asdict(CTRTrainingConfig()), asdict(config.training)),
            "genome": diff_payload(asdict(MTLGenomeConfig()), asdict(config.genome)),
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

    # -- data + genome -----------------------------------------------------

    def _ensure_bundle_and_trainer(self) -> None:
        if self.bundle is None:
            self.bundle = prepare_census_mtl_data(self.config.dataset, self.config.training)
        if self.trainer is None:
            self.trainer = MTLGenomeTrainer(self.config.training, self.bundle.task_names)

    def _load_or_build_genome(self, bundle: MTLDatasetBundle) -> SkillGenome:
        if self.config.genome.genome_path:
            return SkillGenome.load(self.config.genome.genome_path)
        return build_mtl_genome(bundle, self.config.genome)

    def _validate_training_runtime(self) -> None:
        if not _uses_cuda(self.config.training.device) or torch.cuda.is_available():
            return
        gpu_ids = os.environ.get("CTR_EVOLUTION_GPU_IDS", "") or "<auto-detect>"
        raise RuntimeError(
            "CUDA training was requested, but PyTorch cannot access CUDA in this process. "
            "Run the MTL command with `conda run -n rechub` in a GPU-visible shell and "
            f"verify torch.cuda.is_available() before starting the evolution. "
            f"CTR_EVOLUTION_GPU_IDS={gpu_ids}"
        )

    # -- main loop ---------------------------------------------------------

    def _run(self) -> dict[str, Any]:
        started = time.time()
        self._validate_training_runtime()
        self._ensure_bundle_and_trainer()
        assert self.bundle is not None and self.trainer is not None
        if self.config.write_metadata_files:
            self._write_json("dataset_metadata.json", self.bundle.metadata)

        baseline = self._load_or_build_genome(self.bundle)
        GenomeVerifier().assert_valid(baseline)
        if self.config.write_metadata_files:
            baseline.save(self.output_dir / "baseline_genome.json")
        baseline_fingerprint = _safe_architecture_fingerprint(baseline, fallback="baseline")
        self.architecture_index[baseline_fingerprint] = "baseline"

        results = [self._train_candidate("baseline", baseline, None, "baseline", "baseline", "Initial MTL genome")]
        self.completed_results = list(results)
        self._initialize_adaptive_candidate_budget(results[0])
        self._write_progress_results(results)

        parent_population = [baseline]
        rounds = self.config.evolution.rounds if self.config.evolution.enabled else 0
        for round_idx in range(rounds):
            candidates = self._generate_round_candidates(parent_population, round_idx=round_idx)
            unique_candidates, skipped_results = self._deduplicate_candidates(candidates)
            self.deduplication_records.append(
                {
                    "round_idx": round_idx,
                    "generated_candidates": len(candidates),
                    "unique_candidates": len(unique_candidates),
                    "skipped_duplicates": len(skipped_results),
                    "skipped_candidate_ids": [r.candidate_id for r in skipped_results],
                }
            )
            round_results = self._train_round_candidates(unique_candidates, round_idx=round_idx)
            result_by_id = {r.candidate_id: r for r in [*skipped_results, *round_results]}
            results.extend([result_by_id[spec.candidate_id] for spec in candidates if spec.candidate_id in result_by_id])
            survivors = _select_survivors(round_results, self.config.training, self.config.evolution)
            survivor_ids = {r.candidate_id for r in survivors}
            for result in round_results:
                if result.error is None:
                    result.status = "survivor" if result.candidate_id in survivor_ids else "discarded"
                    self._maybe_promote_generated_skill(result)
            self._write_round_survivor_artifacts(round_idx, survivors)
            self._update_adaptive_candidate_budget(round_idx, round_results)
            parent_population = _next_parent_population(parent_population, survivors)
            self.completed_results = list(results)
            self._write_progress_results(results)

        self._write_progress_results(results)
        summary = {
            "experiment_name": self.config.experiment_name,
            "output_dir": str(self.output_dir),
            "task_names": self.bundle.task_names,
            "task_label_cols": self.config.dataset.label_cols,
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

    # -- candidate generation ---------------------------------------------

    def _code_space_enabled(self) -> bool:
        provider = str(self.config.evolution.code_space.provider).lower().replace("-", "_")
        return provider not in {"fallback", "disabled", "none"}

    def _round_candidate_targets(self) -> tuple[int, int]:
        """Return (skill_target, code_target) splitting the candidate budget."""
        if self._code_space_enabled():
            return _adaptive_candidate_targets(
                self.config.evolution,
                stagnant_rounds=int(self.adaptive_budget_state.get("stagnant_rounds") or 0),
                active_code_candidates=self.adaptive_budget_state.get("active_code_candidates"),
            )
        return int(self.config.evolution.candidate_budget), 0

    def _generate_round_candidates(self, parent_population: list[SkillGenome], round_idx: int) -> list[CandidateSpec]:
        if not self._code_space_enabled():
            skill_target, code_target = int(self.config.evolution.candidate_budget), 0
        else:
            skill_target, code_target = self._candidate_targets_for_round(round_idx)
        skill_evolution = replace(self.config.evolution, candidate_budget=skill_target)
        specs = generate_mtl_candidates(
            parent_population,
            self.bundle,
            self.config.genome,
            skill_evolution,
            round_idx=round_idx,
        ) if skill_target > 0 else []
        if code_target > 0:
            try:
                code_specs = self._generate_open_ended_code_candidates(
                    parent_population, round_idx=round_idx, target_code=code_target
                )
            except Exception as exc:
                self._record_candidate_generation_error(round_idx, "mtl_code_candidate_generation", exc, target_code=code_target)
                code_specs = []
            specs.extend(code_specs)
            if len(code_specs) < code_target:
                self._record_insufficient_code_space_candidates(round_idx, code_target, len(code_specs))
                if self.config.evolution.code_space.require_active_provider and not code_specs:
                    raise RuntimeError(
                        "Required active MTL Code-space provider did not produce any valid candidates "
                        f"in round {round_idx}: generated 0 of {code_target}. "
                        "Inspect the round*_mtl_code_space_proposals.json diagnostic before retrying."
                    )
                if not self.config.evolution.code_space.require_active_provider:
                    specs.extend(
                        self._generate_skill_space_supplement(
                            parent_population,
                            round_idx=round_idx,
                            count=code_target - len(code_specs),
                            reason="mtl_code_space_insufficient",
                            skip_candidate_ids={spec.candidate_id for spec in specs},
                        )
                    )
        seen_ids: set[str] = set()
        for spec in specs:
            spec.candidate_id = _dedupe_candidate_id(spec.candidate_id, seen_ids)
        return specs[: self.config.evolution.candidate_budget]

    def _memory_task_type(self) -> str:
        return MTL_TASK_TAG

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
        try:
            pool = generate_mtl_candidates(
                parent_population,
                self.bundle,
                self.config.genome,
                supplement_evolution,
                round_idx=round_idx,
            )
        except Exception as exc:
            self._record_candidate_generation_error(
                round_idx,
                "mtl_skill_space_supplement_generation",
                exc,
                target_skill=count,
                reason=reason,
            )
            return []
        supplement = [spec for spec in pool if spec.candidate_id not in skip_candidate_ids][:count]
        for idx, spec in enumerate(supplement):
            spec.candidate_id = f"{spec.candidate_id}_supplement{idx}"
            spec.rationale = f"{spec.rationale} (skill-space supplement: {reason})"
        self.code_space_generation_records.append(
            {
                "round_idx": round_idx,
                "stage": "mtl_skill_space_supplement",
                "reason": reason,
                "requested": count,
                "generated": len(supplement),
            }
        )
        return supplement

    def _generate_open_ended_code_candidates(
        self, parents: list[SkillGenome], round_idx: int, target_code: int
    ) -> list[CandidateSpec]:
        if target_code <= 0 or not parents:
            return []
        try:
            provider = build_mtl_code_space_provider(self.config.evolution.code_space)
        except Exception as exc:
            self._record_candidate_generation_error(round_idx, "build_mtl_code_space_provider", exc, target_code=target_code)
            return []
        staging_root = self._code_space_staging_root or str(self.output_dir / "generated_skill_staging")
        specs: list[CandidateSpec] = []
        errors: list[dict[str, Any]] = []
        proposal_reports: list[dict[str, Any]] = []
        allocations = _allocate_code_budget_by_parent(parents, budget=target_code, round_idx=round_idx)
        retained_parent_count = len(parents)
        for parent_idx, parent, parent_budget in allocations:
            parent_specs: list[CandidateSpec] = []
            retry_feedback: list[dict[str, Any]] = []
            attempts = max(1, int(self.config.evolution.code_space.max_retries))
            for attempt_idx in range(attempts):
                diagnosis_report = self._build_mtl_code_space_diagnosis(
                    parent,
                    round_idx=round_idx,
                    parent_idx=parent_idx,
                    parent_budget=parent_budget,
                    retained_parent_count=retained_parent_count,
                    retry_feedback=retry_feedback,
                )
                try:
                    proposals = provider.propose(
                        parent=parent,
                        diagnosis_report=diagnosis_report,
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
                    ingestions = inspect_open_ended_proposal_ingestions(
                        proposals=proposals,
                        parent=parent,
                        output_dir=self.output_dir,
                        memory=self.memory,
                        staging_root=staging_root,
                    )
                except Exception as exc:
                    error = {
                        "parent_index": parent_idx,
                        "parent_genome_id": parent.metadata.genome_id,
                        "attempt": attempt_idx + 1,
                        "stage": "ingest_mtl_open_ended_proposals",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                    errors.append(error)
                    retry_feedback.append(error)
                    continue
                for item in ingestions:
                    diagnostic = item.diagnostic()
                    if not diagnostic.get("success"):
                        error = {
                            "parent_index": parent_idx,
                            "parent_genome_id": parent.metadata.genome_id,
                            "attempt": attempt_idx + 1,
                            "stage": "ingest_mtl_open_ended_proposals",
                            "proposal_id": diagnostic.get("proposal_id", ""),
                            "skill_id": diagnostic.get("skill_id", ""),
                            "error": diagnostic.get("message", "ingestion failed"),
                            "validation_results": diagnostic.get("validation_results", {}),
                        }
                        errors.append(error)
                        retry_feedback.append(error)
                for item in [item for item in ingestions if item.success]:
                    if len(parent_specs) >= parent_budget or len(parent_specs) + len(specs) >= target_code:
                        break
                    spec = self._candidate_spec_from_mtl_result(
                        item,
                        parent=parent,
                        parent_idx=parent_idx,
                        round_idx=round_idx,
                        retained_parent_count=retained_parent_count,
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
                        "stage": "mtl_code_space_parent_generation",
                        "error": "no valid MTL code-space candidates for retained parent",
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
                for parent_idx, parent, budget in allocations
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
            "evolution_feedback": self._mtl_code_space_feedback_snapshot(round_idx),
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
            self._write_json(f"round{round_idx}_mtl_code_space_proposals.json", diagnostic_payload)
        return specs[:target_code]

    def _build_mtl_code_space_diagnosis(
        self,
        parent: SkillGenome,
        *,
        round_idx: int,
        parent_idx: int,
        parent_budget: int,
        retained_parent_count: int,
        retry_feedback: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        shared_transform_node = next(
            (n.node_id for n in parent.nodes if n.category == "multitask" and "task_representations" in n.output_keys),
            None,
        )
        report: dict[str, Any] = {
            "failure_modes": list(self.config.evolution.failure_modes),
            "task_family": "multi_task_recommendation",
            "round_idx": round_idx,
            "parent_index": parent_idx,
            "parent_rank": parent_idx,
            "parent_genome_id": parent.metadata.genome_id,
            "retained_parent_count": retained_parent_count,
            "parent_budget": parent_budget,
            "shared_transform_node": shared_transform_node,
            "structural_scopes": list(self.config.evolution.code_space.structural_scopes),
            "objective": {
                "selection_metric": "validation mean AUC",
                "training_objective_metric": self.config.training.objective_metric,
                "min_delta": self.config.training.min_delta,
                "task_names": list(self.bundle.task_names if self.bundle is not None else self.config.dataset.task_names),
            },
            "evolution_feedback": self._mtl_code_space_feedback_snapshot(round_idx),
            "real_time_feedback_contract": (
                "Use the current parent genome, recent round outcomes, recent failed generated skills, "
                "provider/ingestion failures, and dataset/runtime dimensions when planning this round. "
                "Do not replay fixed ideas; propose repairs or new architectures that directly address the "
                "observed MTL failures."
            ),
        }
        if retry_feedback:
            report["previous_code_space_failures"] = retry_feedback[-8:]
            report["repair_instruction"] = (
                "Regenerate different valid MTL macro proposals that avoid the previous provider or ingestion "
                "failures. Fix missing macro_judgment fields, unavailable input keys, unsafe code, invalid "
                "insert_between/replace_node wiring, hard-coded dimensions, and shape mismatches. Preserve the "
                "exact intercepted tensor shape advertised by current_architecture_profile.injection_points."
            )
        return report

    def _mtl_code_space_feedback_snapshot(self, round_idx: int) -> dict[str, Any]:
        results = list(self.completed_results or [])
        baseline = next((result for result in results if result.candidate_id == "baseline"), None)
        adaptive_record = next(
            (item for item in reversed(self.adaptive_budget_state["records"]) if item.get("round_idx") == round_idx),
            None,
        )
        return {
            "round_idx": round_idx,
            "dataset_metadata": dict(self.bundle.metadata) if self.bundle is not None else {},
            "task_names": list(self.bundle.task_names if self.bundle is not None else self.config.dataset.task_names),
            "baseline": _mtl_prompt_result_summary(baseline) if baseline is not None else {},
            "recent_rounds": _mtl_prompt_recent_rounds(results, limit=5),
            "recent_failed_candidates": _mtl_prompt_recent_failures(results, limit=12),
            "recent_code_space_generation": list(self.code_space_generation_records[-8:]),
            "recent_generation_errors": list(self.candidate_generation_errors[-12:]),
            "adaptive_candidate_budget": {
                "stagnant_rounds": int(self.adaptive_budget_state.get("stagnant_rounds") or 0),
                "global_best_score": self.adaptive_budget_state.get("global_best_score"),
                "active_code_candidates": self.adaptive_budget_state.get("active_code_candidates"),
                "current_round_target": dict(adaptive_record or {}),
                "recent_records": list(self.adaptive_budget_state["records"][-6:]),
            },
        }

    def _candidate_spec_from_mtl_result(
        self,
        item: Any,
        *,
        parent: SkillGenome,
        parent_idx: int,
        round_idx: int,
        retained_parent_count: int,
        errors: list[dict[str, Any]] | None = None,
    ) -> CandidateSpec | None:
        ingestion = item.ingestion
        genome = getattr(ingestion, "genome", None)
        if genome is None or not getattr(ingestion, "success", False):
            return None
        skill_id = ingestion.skill_id or item.proposal.skill_id or item.proposal.proposal_id
        try:
            GenomeVerifier().assert_valid(genome)
        except Exception as exc:
            if errors is not None:
                errors.append(
                    {
                        "parent_index": parent_idx,
                        "parent_genome_id": parent.metadata.genome_id,
                        "proposal_id": getattr(item.proposal, "proposal_id", ""),
                        "stage": "build_mtl_code_candidate_spec",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
            self._record_candidate_generation_error(
                round_idx, "mtl_code_genome_invalid", exc, skill_id=skill_id
            )
            return None
        if ingestion.code_path:
            _annotate_generated_skill_hash(genome, skill_id, _path_sha256(ingestion.code_path))
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
            architecture_fingerprint=_safe_architecture_fingerprint(genome, fallback=candidate_id),
            generated_skill_id=skill_id,
            staged_code_path=ingestion.code_path,
            staged_skill_card_path=ingestion.skill_card_path,
            proposal_id=item.proposal.proposal_id,
            structural_scope=item.proposal.structural_scope or item.proposal.metadata.get("structural_scope", ""),
        )

    # -- training ----------------------------------------------------------

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
        self._ensure_bundle_and_trainer()
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
                    wave_results = pool.map(_train_mtl_candidate_worker, payloads)
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


def _train_mtl_candidate_worker(payload: dict[str, Any]) -> CandidateResult:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    gpu_id = str(payload["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config: MTLWorkflowConfig = payload["config"]
    if _uses_cuda(config.training.device):
        config.training.device = "cuda:0"
    bundle = prepare_census_mtl_data(config.dataset, config.training)
    trainer = MTLGenomeTrainer(config.training, bundle.task_names)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run multi-task (MTL) recommendation-model evolution.")
    parser.add_argument("--config", required=True, help="YAML workflow config.")
    parser.add_argument("--device", help="Override training.device.")
    parser.add_argument("--output-dir", help="Override output_dir.")
    parser.add_argument("--seed", type=int, help="Override training.seed.")
    args = parser.parse_args(argv)
    config = load_mtl_workflow_config(args.config)
    if args.device:
        config.training.device = args.device
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.training.seed = args.seed
    summary = MTLModelEvolutionRunner(config).run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
