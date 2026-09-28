"""Multi-domain recommendation-model evolution workflow.

This module adds the multi-domain counterpart of the CTR/MTL evolution runners.
The fixed scaffold is intentionally small for the first supported slice:

    sparse_features -> field_embedding -> flatten -> <domain transform>
    domain_indicator -> domain_indicator_adapter -> domain_id
    <domain transform> -> domain_outputs/prediction -> bce_loss

The supported built-in templates are ``star``, ``shared_bottom``, ``mmoe``, and
``ple``. They reuse the existing scenario and multitask skills, but the genome
task contract is ``multi_domain`` and the trainer reports both global and
per-domain metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .code_space import inspect_open_ended_proposal_ingestions
from .ctr_workflow import (
    CandidateResult,
    CandidateSpec,
    CTREvolutionConfig,
    CTRGenomeTrainer,
    CTRModelEvolutionRunner,
    CTRTrainingConfig,
    _adaptive_candidate_targets,
    _allocate_code_budget_by_parent,
    _annotate_generated_skill_artifacts,
    _annotate_generated_skill_hash,
    _as_bool,
    _available_candidate_gpu_ids,
    _build_loader,
    _dedupe_candidate_id,
    _generated_skill_portability_allows_reuse,
    _is_reusable_generated_skill_card,
    _next_parent_population,
    _path_sha256,
    _safe_architecture_fingerprint,
    _safe_code_space_provider_diagnostics,
    _safe_auc,
    _safe_logloss,
    _select_survivors,
    _skill_card_has_loadable_implementation,
    _train_candidate_artifact,
    _uses_cuda,
)
from .evolution_memory import EvolutionMemory
from .fingerprint import config_fingerprint, diff_payload
from .genome import GenomeConstraints, GenomeMutation, SkillEdge, SkillGenome, SkillNode
from .multi_domain_code_space import build_multi_domain_code_space_provider
from .skill_library import SkillCard, SkillLibrary
from .verification import GenomeVerifier


MULTI_DOMAIN_TASK_TAG = "multi_domain"
MULTI_DOMAIN_TEMPLATES = (
    "star",
    "shared_bottom",
    "mmoe",
    "ple",
    "sarnet",
    "adaptdhm",
    "hamur_small",
    "hamur_large",
    "m3oe",
    "ppnet",
    "m2m",
    "adasparse",
    "epnet",
)

MULTI_DOMAIN_TEMPLATE_SKILL_IDS: dict[str, tuple[str, ...]] = {
    "star": ("star_domain_fcn", "domain_select"),
    "shared_bottom": ("shared_bottom_tower", "task_replication", "task_tower", "domain_select"),
    "mmoe": ("mmoe_gate", "task_tower", "domain_select"),
    "ple": ("ple_gate", "task_tower", "domain_select"),
    "sarnet": ("sarnet_expert_mixer",),
    "adaptdhm": ("adaptdhm_cluster_tower",),
    "hamur_small": ("hamur_domain_adapter_tower",),
    "hamur_large": ("hamur_domain_adapter_tower",),
    "m3oe": ("m3oe_expert_fusion",),
    "ppnet": ("ppnet_domain_towers", "domain_select"),
    "m2m": ("m2m_meta_tower",),
    "adasparse": ("adasparse_pruned_tower",),
    "epnet": ("gate_nu_feature_gate", "mlp_tower", "sigmoid_prediction"),
}


@dataclass(frozen=True)
class _MultiDomainSkillChoice:
    source: str
    score: int
    name: str
    order: int
    template: dict[str, Any] | None = None
    skill_id: str | None = None
    card: SkillCard | None = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MultiDomainDatasetConfig:
    name: str = "multi_domain_dataset"
    type: str = "multi_domain_csv"
    path: str = ""
    label_col: str = "label"
    domain_col: str = "domain_indicator"
    positive_label_threshold: float | None = None
    domain_source_col: str | None = None
    domain_groups: list[list[Any]] | None = None
    domain_mapping: dict[str, int] | None = None
    domain_default: int | None = None
    genre_col: str | None = None
    derived_cate_col: str | None = None
    categorical_cols: list[str] = field(default_factory=list)
    dense_cols: list[str] = field(default_factory=list)
    normalize_dense: bool = False
    exclude_cols: list[str] = field(default_factory=list)
    include_domain_as_feature: bool = False
    split_ratio: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    limit_rows: int | None = None


@dataclass
class MultiDomainGenomeConfig:
    genome_path: str | None = None
    template: str = "star"
    embedding_dim: int = 16
    shared_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    tower_hidden_dims: list[int] = field(default_factory=lambda: [32])
    expert_dim: int = 32
    num_experts: int = 3
    ple_shared_experts: int = 1
    ple_specific_experts: int = 2
    star_fcn_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    star_aux_dims: list[int] = field(default_factory=lambda: [32])
    sarnet_shared_experts: int = 8
    sarnet_specific_experts: int = 2
    sarnet_expert_dim: int = 16
    adaptdhm_fcn_dims: list[int] = field(default_factory=lambda: [64, 64])
    adaptdhm_cluster_num: int | None = None
    adaptdhm_beta: float = 0.9
    hamur_small_fcn_dims: list[int] = field(default_factory=lambda: [256, 128])
    hamur_large_fcn_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    hamur_hyper_dims: list[int] = field(default_factory=lambda: [64])
    hamur_k: int = 35
    hamur_adapter_dim: int = 32
    m3oe_fcn_dims: list[int] = field(default_factory=lambda: [128, 64, 64, 32])
    m3oe_expert_num: int = 4
    m3oe_exp_d: float = 1.0
    m3oe_bal_d: float = 1.0
    ppnet_fcn_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    domain_embedding_dim: int = 16
    m2m_num_experts: int = 4
    m2m_expert_output_size: int = 16
    m2m_transformer_dims: dict[str, int] = field(
        default_factory=lambda: {"num_encoder_layers": 2, "num_decoder_layers": 2, "dim_feedforward": 16}
    )
    adasparse_hidden_dims: list[int] = field(default_factory=lambda: [32, 32])
    adasparse_form: str = "Fusion"
    adasparse_epsilon: float = 1e-2
    adasparse_beta: float = 2.0
    adasparse_alpha: float = 1.0
    adasparse_delta_alpha: float = 1e-4
    epnet_fcn_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    epnet_gamma: float = 2.0
    dropout: float = 0.0
    activation: str = "relu"


@dataclass
class MultiDomainWorkflowConfig:
    experiment_name: str = "multi_domain_evolution"
    output_dir: str = "outputs/evolution/multi_domain"
    dataset: MultiDomainDatasetConfig = field(default_factory=MultiDomainDatasetConfig)
    genome: MultiDomainGenomeConfig = field(default_factory=MultiDomainGenomeConfig)
    training: CTRTrainingConfig = field(default_factory=CTRTrainingConfig)
    evolution: CTREvolutionConfig = field(default_factory=lambda: CTREvolutionConfig(templates=_default_multi_domain_templates()))
    memory_path: str | None = None
    write_metadata_files: bool = True
    write_summary_json: bool = True
    write_evolution_memory: bool = True
    write_code_space_diagnostics: bool = True
    write_survivor_artifacts: bool = False
    survivor_artifacts_dir: str = "survivors"


@dataclass
class MultiDomainDatasetBundle:
    feature_names: list[str]
    vocab_sizes: list[int]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    metadata: dict[str, Any]
    domain_num: int
    domain_names: list[str]
    sparse_feature_names: list[str] = field(default_factory=list)
    dense_feature_names: list[str] = field(default_factory=list)


def load_multi_domain_workflow_config(path: str | Path) -> MultiDomainWorkflowConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    evolution_raw = dict(raw.get("evolution") or {})
    if "templates" not in evolution_raw:
        evolution_raw["templates"] = _default_multi_domain_templates()
    return MultiDomainWorkflowConfig(
        experiment_name=raw.get("experiment_name", "multi_domain_evolution"),
        output_dir=raw.get("output_dir", "outputs/evolution/multi_domain"),
        dataset=MultiDomainDatasetConfig(**(raw.get("dataset") or {})),
        genome=MultiDomainGenomeConfig(**(raw.get("genome") or {})),
        training=CTRTrainingConfig(**(raw.get("training") or {})),
        evolution=CTREvolutionConfig(**evolution_raw),
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


def _factorize_column(values: pd.Series) -> tuple[np.ndarray, int, dict[str, int]]:
    filled = values.fillna("__missing__").astype(str)
    codes, uniques = pd.factorize(filled, sort=True)
    ids = codes.astype(np.int64)
    vocab_size = int(ids.max()) + 1 if len(ids) else 1
    mapping = {str(raw): int(idx) for idx, raw in enumerate(uniques)}
    return ids, max(1, vocab_size), mapping


def _binary_label(values: pd.Series, positive_threshold: float | None = None) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if positive_threshold is not None:
        return (numeric > float(positive_threshold)).astype(np.float32)
    if not np.all(np.isin(numeric, [0.0, 1.0])):
        numeric = (numeric > 0).astype(np.float32)
    return numeric


def _dense_values(values: pd.Series, *, normalize: bool = False) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if normalize and len(arr):
        min_value = float(np.min(arr))
        max_value = float(np.max(arr))
        span = max_value - min_value
        if span > 0:
            arr = ((arr - min_value) / span).astype(np.float32)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def _derive_domain_indicator(data: pd.DataFrame, config: MultiDomainDatasetConfig) -> pd.Series:
    source_col = config.domain_source_col
    if not source_col:
        raise ValueError(
            f"Domain column {config.domain_col!r} not found and no domain_source_col was configured"
        )
    if source_col not in data.columns:
        raise ValueError(f"domain_source_col not found in dataset: {source_col}")
    source = data[source_col]
    if config.domain_groups:
        group_lookup: dict[str, int] = {}
        for idx, group in enumerate(config.domain_groups):
            for value in group:
                group_lookup[_domain_lookup_key(value)] = idx
        mapped = source.map(lambda value: group_lookup.get(_domain_lookup_key(value), config.domain_default))
    elif config.domain_mapping:
        mapping = {_domain_lookup_key(key): int(value) for key, value in config.domain_mapping.items()}
        mapped = source.map(lambda value: mapping.get(_domain_lookup_key(value), config.domain_default))
    else:
        raise ValueError(
            f"Domain column {config.domain_col!r} not found; configure domain_groups or domain_mapping"
        )
    if mapped.isna().any():
        missing = sorted({_domain_lookup_key(value) for value in source[mapped.isna()].head(10).tolist()})
        raise ValueError(
            f"Could not derive {config.domain_col!r} from {source_col!r}; unmapped sample values: {missing}"
        )
    return mapped.astype(np.int64)


def _domain_lookup_key(value: Any) -> str:
    if pd.isna(value):
        return "__missing__"
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return str(value)


def prepare_multi_domain_data(config: MultiDomainDatasetConfig, training: CTRTrainingConfig) -> MultiDomainDatasetBundle:
    if not str(config.path or "").strip():
        raise ValueError("dataset.path is required for multi-domain evolution")
    path = Path(config.path)
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path)
    if config.limit_rows is not None and config.limit_rows < len(data):
        data = data.sample(n=int(config.limit_rows), random_state=int(training.seed)).reset_index(drop=True)
    if config.derived_cate_col and config.genre_col and config.derived_cate_col not in data.columns and config.genre_col in data.columns:
        data[config.derived_cate_col] = data[config.genre_col].fillna("unknown").map(lambda value: str(value).split("|")[0])
    if config.label_col not in data.columns:
        raise ValueError(f"Label column not found in dataset: {config.label_col}")
    domain_source = "dataset_column"
    if config.domain_col not in data.columns:
        data[config.domain_col] = _derive_domain_indicator(data, config)
        domain_source = "derived"
    if len(data) < 3:
        raise ValueError("Need at least 3 rows to create train/val/test splits")

    labels = _binary_label(data[config.label_col], positive_threshold=config.positive_label_threshold)
    domain_ids, domain_num, domain_mapping = _factorize_column(data[config.domain_col])
    if domain_num < 2:
        raise ValueError(
            f"Multi-domain evolution requires at least 2 domains in {config.domain_col}; "
            f"found {domain_num}. Increase limit_rows or verify the dataset split."
        )
    domain_names = [name for name, _ in sorted(domain_mapping.items(), key=lambda item: item[1])]

    excluded = {config.label_col, *config.exclude_cols}
    if config.derived_cate_col and config.genre_col and config.derived_cate_col != config.genre_col:
        excluded.add(config.genre_col)
    if not config.include_domain_as_feature:
        excluded.add(config.domain_col)
    all_feature_cols = [col for col in data.columns if col not in excluded]
    dense_set = {col for col in config.dense_cols if col in all_feature_cols}
    if config.categorical_cols:
        sparse_feature_names = [col for col in config.categorical_cols if col in all_feature_cols and col not in dense_set]
    else:
        sparse_feature_names = [col for col in all_feature_cols if col not in dense_set]
    dense_feature_names = [col for col in all_feature_cols if col in dense_set]
    if not sparse_feature_names:
        raise ValueError("No sparse feature columns are available for multi-domain evolution")

    sparse_arrays: list[np.ndarray] = []
    vocab_sizes: list[int] = []
    encoders: dict[str, dict[str, int]] = {}
    for col in sparse_feature_names:
        ids, vocab, mapping = _factorize_column(data[col])
        sparse_arrays.append(ids)
        vocab_sizes.append(vocab)
        encoders[col] = mapping
    x: dict[str, np.ndarray] = {
        "sparse_features": np.stack(sparse_arrays, axis=1),
        "domain_indicator": domain_ids.astype(np.int64),
    }
    if dense_feature_names:
        dense_matrix = np.stack(
            [_dense_values(data[col], normalize=config.normalize_dense) for col in dense_feature_names],
            axis=1,
        )
        x["dense_values"] = dense_matrix
        x["dense_features"] = dense_matrix

    rng = np.random.default_rng(training.seed)
    order = rng.permutation(len(labels))
    for key, value in list(x.items()):
        x[key] = value[order]
    labels = labels[order]
    domain_ids = domain_ids[order]

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
        "domain_col": config.domain_col,
        "domain_source": domain_source,
        "domain_source_col": config.domain_source_col,
        "positive_label_threshold": config.positive_label_threshold,
        "domain_num": int(domain_num),
        "domain_names": domain_names,
        "domain_mapping": domain_mapping,
        "domain_counts": {domain_names[idx]: int((domain_ids == idx).sum()) for idx in range(domain_num)},
        "feature_names": [*sparse_feature_names, *dense_feature_names],
        "sparse_feature_names": sparse_feature_names,
        "dense_feature_names": dense_feature_names,
        "normalize_dense": bool(config.normalize_dense),
        "vocab_sizes": vocab_sizes,
        "encoders": encoders,
        "include_domain_as_feature": bool(config.include_domain_as_feature),
    }
    return MultiDomainDatasetBundle(
        feature_names=[*sparse_feature_names, *dense_feature_names],
        vocab_sizes=vocab_sizes,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        metadata=metadata,
        domain_num=domain_num,
        domain_names=domain_names,
        sparse_feature_names=sparse_feature_names,
        dense_feature_names=dense_feature_names,
    )


def _split_indices(n_rows: int, split_ratio: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_rows < 3:
        raise ValueError("Need at least 3 rows to create train/val/test splits")
    total = sum(split_ratio) or 1.0
    ratios = [item / total for item in split_ratio]
    train_end = max(1, int(n_rows * ratios[0]))
    val_end = max(train_end + 1, train_end + int(n_rows * ratios[1]))
    val_end = min(val_end, n_rows - 1)
    indices = np.arange(n_rows)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


# ---------------------------------------------------------------------------
# Genome construction
# ---------------------------------------------------------------------------


def _num_dense_features(bundle: MultiDomainDatasetBundle) -> int:
    return len(bundle.dense_feature_names or [])


def _num_total_fields(bundle: MultiDomainDatasetBundle) -> int:
    return len(bundle.vocab_sizes) + _num_dense_features(bundle)


def _domain_task_types(domain_num: int) -> list[str]:
    return ["classification"] * int(domain_num)


def _input_nodes(
    bundle: MultiDomainDatasetBundle,
    config: MultiDomainGenomeConfig,
    *,
    include_domain_adapter: bool = True,
) -> list[SkillNode]:
    nodes = []
    if include_domain_adapter:
        nodes.append(
            SkillNode(
                node_id="domain_adapter",
                skill_id="domain_indicator_adapter",
                skill_name="domain_indicator_adapter",
                category="scenario",
                params={
                    "input_key": "domain_indicator",
                    "output_key": "domain_id",
                    "num_domains": bundle.domain_num,
                },
                input_keys=["domain_indicator"],
                output_keys=["domain_id"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
    nodes.append(
        SkillNode(
            node_id="field_embedding",
            skill_id="field_embedding",
            skill_name="field_embedding",
            category="embedding",
            params={
                "vocab_sizes": bundle.vocab_sizes,
                "embedding_dim": config.embedding_dim,
                "feature_names": list(bundle.sparse_feature_names or bundle.feature_names[: len(bundle.vocab_sizes)]),
            },
            input_keys=["sparse_features"],
            output_keys=["field_embeddings"],
            task_types=[MULTI_DOMAIN_TASK_TAG],
        ),
    )
    if _num_dense_features(bundle) > 0:
        nodes.append(
            SkillNode(
                node_id="dense_feature_path",
                skill_id="dense_feature_path",
                skill_name="dense_feature_path",
                category="utility",
                params={
                    "num_dense": _num_dense_features(bundle),
                    "embedding_dim": config.embedding_dim,
                    "input_key": "dense_values",
                    "output_key": "dense_embeddings",
                    "field_embeddings_key": "field_embeddings",
                    "merged_output_key": "field_embeddings",
                },
                input_keys=["dense_values", "field_embeddings"],
                output_keys=["dense_embeddings", "field_embeddings"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
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
            task_types=[MULTI_DOMAIN_TASK_TAG],
        )
    )
    return nodes


def _template_uses_domain_adapter(template: str) -> bool:
    return template in {
        "star",
        "shared_bottom",
        "mmoe",
        "ple",
        "sarnet",
        "hamur_small",
        "hamur_large",
        "m3oe",
        "ppnet",
    }


def _domain_transform_nodes(
    template: str,
    bundle: MultiDomainDatasetBundle,
    config: MultiDomainGenomeConfig,
    flat_input_dim: int,
) -> tuple[list[SkillNode], list[SkillEdge], str, int | None, str]:
    nodes: list[SkillNode] = []
    edges: list[SkillEdge] = []

    if template == "star":
        nodes.append(
            SkillNode(
                node_id="star_domain_fcn",
                skill_id="star_domain_fcn",
                skill_name="star_domain_fcn",
                category="scenario",
                params={
                    "input_dim": flat_input_dim,
                    "domain_num": bundle.domain_num,
                    "fcn_dims": list(config.star_fcn_dims),
                    "aux_dims": list(config.star_aux_dims),
                    "input_key": "flat_embeddings",
                    "output_key": "domain_outputs",
                },
                input_keys=["flat_embeddings"],
                output_keys=["domain_outputs"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "star_domain_fcn", "flat_embeddings", "flat_embeddings"))
        return nodes, edges, "star_domain_fcn", None, "domain_outputs"

    if template == "shared_bottom":
        rep_dim = int(config.shared_hidden_dims[-1])
        nodes.extend(
            [
                SkillNode(
                    node_id="shared_bottom",
                    skill_id="shared_bottom_tower",
                    skill_name="shared_bottom_tower",
                    category="scenario",
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
                    task_types=[MULTI_DOMAIN_TASK_TAG],
                ),
                SkillNode(
                    node_id="domain_replication",
                    skill_id="task_replication",
                    skill_name="task_replication",
                    category="scenario",
                    params={
                        "n_task": bundle.domain_num,
                        "input_key": "shared_representation",
                        "output_key": "domain_representations",
                    },
                    input_keys=["shared_representation"],
                    output_keys=["domain_representations"],
                    task_types=[MULTI_DOMAIN_TASK_TAG],
                ),
            ]
        )
        edges.append(SkillEdge("flatten", "shared_bottom", "flat_embeddings", "flat_embeddings"))
        edges.append(SkillEdge("shared_bottom", "domain_replication", "shared_representation", "shared_representation"))
        return nodes, edges, "domain_replication", rep_dim, "domain_representations"

    if template == "mmoe":
        rep_dim = int(config.expert_dim)
        expert_hidden_dims = [*config.shared_hidden_dims[:-1], config.expert_dim] if config.shared_hidden_dims else [config.expert_dim]
        nodes.append(
            SkillNode(
                node_id="domain_mmoe_gate",
                skill_id="mmoe_gate",
                skill_name="mmoe_gate",
                category="scenario",
                params={
                    "input_dim": flat_input_dim,
                    "n_expert": config.num_experts,
                    "n_task": bundle.domain_num,
                    "expert_hidden_dims": expert_hidden_dims,
                    "input_key": "flat_embeddings",
                    "output_key": "domain_representations",
                    "activation": config.activation,
                    "dropout": config.dropout,
                },
                input_keys=["flat_embeddings"],
                output_keys=["domain_representations"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "domain_mmoe_gate", "flat_embeddings", "flat_embeddings"))
        return nodes, edges, "domain_mmoe_gate", rep_dim, "domain_representations"

    if template == "ple":
        rep_dim = int(config.expert_dim)
        expert_hidden_dims = [*config.shared_hidden_dims[:-1], config.expert_dim] if config.shared_hidden_dims else [config.expert_dim]
        nodes.append(
            SkillNode(
                node_id="domain_ple_gate",
                skill_id="ple_gate",
                skill_name="ple_gate",
                category="scenario",
                params={
                    "input_dim": flat_input_dim,
                    "n_task": bundle.domain_num,
                    "n_expert_shared": config.ple_shared_experts,
                    "n_expert_specific": config.ple_specific_experts,
                    "expert_hidden_dims": expert_hidden_dims,
                    "input_key": "flat_embeddings",
                    "output_key": "domain_representations",
                    "activation": config.activation,
                    "dropout": config.dropout,
                },
                input_keys=["flat_embeddings"],
                output_keys=["domain_representations"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "domain_ple_gate", "flat_embeddings", "flat_embeddings"))
        return nodes, edges, "domain_ple_gate", rep_dim, "domain_representations"

    if template == "sarnet":
        nodes.append(
            SkillNode(
                node_id="sarnet_expert_mixer",
                skill_id="sarnet_expert_mixer",
                skill_name="sarnet_expert_mixer",
                category="scenario",
                params={
                    "input_dim": flat_input_dim,
                    "domain_num": bundle.domain_num,
                    "domain_shared_expert_num": config.sarnet_shared_experts,
                    "domain_specific_expert_num": config.sarnet_specific_experts,
                    "expert_dim": config.sarnet_expert_dim,
                    "input_key": "flat_embeddings",
                    "domain_key": "domain_id",
                    "output_key": "prediction",
                },
                input_keys=["flat_embeddings", "domain_id"],
                output_keys=["prediction"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "sarnet_expert_mixer", "flat_embeddings", "flat_embeddings"))
        edges.append(SkillEdge("domain_adapter", "sarnet_expert_mixer", "domain_id", "domain_id"))
        return nodes, edges, "sarnet_expert_mixer", None, "prediction"

    if template == "adaptdhm":
        cluster_num = int(config.adaptdhm_cluster_num or bundle.domain_num)
        nodes.append(
            SkillNode(
                node_id="adaptdhm_cluster_tower",
                skill_id="adaptdhm_cluster_tower",
                skill_name="adaptdhm_cluster_tower",
                category="scenario",
                params={
                    "input_dim": flat_input_dim,
                    "fcn_dims": list(config.adaptdhm_fcn_dims),
                    "cluster_num": cluster_num,
                    "beta": config.adaptdhm_beta,
                    "input_key": "flat_embeddings",
                    "output_key": "prediction",
                    "cluster_key": "cluster_id",
                },
                input_keys=["flat_embeddings"],
                output_keys=["prediction", "cluster_id"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "adaptdhm_cluster_tower", "flat_embeddings", "flat_embeddings"))
        return nodes, edges, "adaptdhm_cluster_tower", None, "prediction"

    if template in {"hamur_small", "hamur_large"}:
        fcn_dims = config.hamur_large_fcn_dims if template == "hamur_large" else config.hamur_small_fcn_dims
        nodes.append(
            SkillNode(
                node_id="hamur_domain_adapter_tower",
                skill_id="hamur_domain_adapter_tower",
                skill_name="hamur_domain_adapter_tower",
                category="scenario",
                params={
                    "input_dim": flat_input_dim,
                    "domain_num": bundle.domain_num,
                    "fcn_dims": list(fcn_dims),
                    "hyper_dims": list(config.hamur_hyper_dims),
                    "k": config.hamur_k,
                    "adapter_dim": config.hamur_adapter_dim,
                    "input_key": "flat_embeddings",
                    "domain_key": "domain_id",
                    "output_key": "prediction",
                },
                input_keys=["flat_embeddings", "domain_id"],
                output_keys=["prediction"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "hamur_domain_adapter_tower", "flat_embeddings", "flat_embeddings"))
        edges.append(SkillEdge("domain_adapter", "hamur_domain_adapter_tower", "domain_id", "domain_id"))
        return nodes, edges, "hamur_domain_adapter_tower", None, "prediction"

    if template == "m3oe":
        nodes.append(
            SkillNode(
                node_id="m3oe_expert_fusion",
                skill_id="m3oe_expert_fusion",
                skill_name="m3oe_expert_fusion",
                category="scenario",
                params={
                    "input_dim": flat_input_dim,
                    "domain_num": bundle.domain_num,
                    "fcn_dims": list(config.m3oe_fcn_dims),
                    "expert_num": config.m3oe_expert_num,
                    "exp_d": config.m3oe_exp_d,
                    "bal_d": config.m3oe_bal_d,
                    "input_key": "flat_embeddings",
                    "domain_key": "domain_id",
                    "output_key": "prediction",
                },
                input_keys=["flat_embeddings", "domain_id"],
                output_keys=["prediction"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "m3oe_expert_fusion", "flat_embeddings", "flat_embeddings"))
        edges.append(SkillEdge("domain_adapter", "m3oe_expert_fusion", "domain_id", "domain_id"))
        return nodes, edges, "m3oe_expert_fusion", None, "prediction"

    if template == "ppnet":
        nodes.append(
            SkillNode(
                node_id="ppnet_domain_towers",
                skill_id="ppnet_domain_towers",
                skill_name="ppnet_domain_towers",
                category="scenario",
                params={
                    "input_dim": flat_input_dim,
                    "domain_num": bundle.domain_num,
                    "fcn_dims": list(config.ppnet_fcn_dims),
                    "input_key": "flat_embeddings",
                    "output_key": "domain_outputs",
                },
                input_keys=["flat_embeddings"],
                output_keys=["domain_outputs"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "ppnet_domain_towers", "flat_embeddings", "flat_embeddings"))
        return nodes, edges, "ppnet_domain_towers", None, "domain_outputs"

    if template in {"m2m", "adasparse", "epnet"}:
        domain_nodes, domain_edges = _domain_context_nodes(bundle, config)
        nodes.extend(domain_nodes)
        edges.extend(domain_edges)

    if template == "m2m":
        nodes.append(
            SkillNode(
                node_id="m2m_meta_tower",
                skill_id="m2m_meta_tower",
                skill_name="m2m_meta_tower",
                category="scenario",
                params={
                    "input_dim": flat_input_dim,
                    "domain_dim": config.domain_embedding_dim,
                    "num_experts": config.m2m_num_experts,
                    "expert_output_size": config.m2m_expert_output_size,
                    "transformer_dims": dict(config.m2m_transformer_dims),
                    "input_key": "flat_embeddings",
                    "domain_key": "domain_context",
                    "output_key": "prediction",
                },
                input_keys=["flat_embeddings", "domain_context"],
                output_keys=["prediction"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten", "m2m_meta_tower", "flat_embeddings", "flat_embeddings"))
        edges.append(SkillEdge("flatten_domain", "m2m_meta_tower", "domain_context", "domain_context"))
        return nodes, edges, "m2m_meta_tower", None, "prediction"

    if template == "adasparse":
        nodes.append(
            SkillNode(
                node_id="adasparse_pruned_tower",
                skill_id="adasparse_pruned_tower",
                skill_name="adasparse_pruned_tower",
                category="scenario",
                params={
                    "scenario_dim": config.domain_embedding_dim,
                    "agnostic_dim": flat_input_dim,
                    "hidden_dims": list(config.adasparse_hidden_dims),
                    "scenario_key": "domain_context",
                    "agnostic_key": "flat_embeddings",
                    "logits_key": "logits",
                    "output_key": "prediction",
                    "form": config.adasparse_form,
                    "epsilon": config.adasparse_epsilon,
                    "beta": config.adasparse_beta,
                    "alpha": config.adasparse_alpha,
                    "delta_alpha": config.adasparse_delta_alpha,
                    "activation": config.activation,
                    "dropout": config.dropout,
                },
                input_keys=["domain_context", "flat_embeddings"],
                output_keys=["logits", "prediction"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
        edges.append(SkillEdge("flatten_domain", "adasparse_pruned_tower", "domain_context", "domain_context"))
        edges.append(SkillEdge("flatten", "adasparse_pruned_tower", "flat_embeddings", "flat_embeddings"))
        return nodes, edges, "adasparse_pruned_tower", None, "prediction"

    if template == "epnet":
        nodes.extend(
            [
                SkillNode(
                    node_id="epnet_gate",
                    skill_id="gate_nu_feature_gate",
                    skill_name="gate_nu_feature_gate",
                    category="scenario",
                    params={
                        "scenario_dim": config.domain_embedding_dim,
                        "feature_dim": flat_input_dim,
                        "scenario_key": "domain_context",
                        "feature_key": "flat_embeddings",
                        "output_key": "gated_agnostic",
                        "gate_output_key": "scenario_gate",
                        "gamma": config.epnet_gamma,
                    },
                    input_keys=["domain_context", "flat_embeddings"],
                    output_keys=["gated_agnostic", "scenario_gate"],
                    task_types=[MULTI_DOMAIN_TASK_TAG],
                ),
                SkillNode(
                    node_id="epnet_prediction_tower",
                    skill_id="mlp_tower",
                    skill_name="mlp_tower",
                    category="tower",
                    params={
                        "input_key": "gated_agnostic",
                        "output_key": "logits",
                        "input_dim": flat_input_dim,
                        "hidden_dims": list(config.epnet_fcn_dims),
                        "dropout": config.dropout,
                        "activation": config.activation,
                        "output_layer": True,
                    },
                    input_keys=["gated_agnostic"],
                    output_keys=["logits"],
                    task_types=[MULTI_DOMAIN_TASK_TAG],
                ),
                SkillNode(
                    node_id="epnet_prediction",
                    skill_id="sigmoid_prediction",
                    skill_name="sigmoid_prediction",
                    category="head",
                    params={"input_key": "logits", "output_key": "prediction"},
                    input_keys=["logits"],
                    output_keys=["prediction"],
                    task_types=[MULTI_DOMAIN_TASK_TAG],
                ),
            ]
        )
        edges.append(SkillEdge("flatten_domain", "epnet_gate", "domain_context", "domain_context"))
        edges.append(SkillEdge("flatten", "epnet_gate", "flat_embeddings", "flat_embeddings"))
        edges.append(SkillEdge("epnet_gate", "epnet_prediction_tower", "gated_agnostic", "gated_agnostic"))
        edges.append(SkillEdge("epnet_prediction_tower", "epnet_prediction", "logits", "logits"))
        return nodes, edges, "epnet_prediction", None, "prediction"

    raise ValueError(f"Unsupported multi-domain genome template: {template}")


def _domain_context_nodes(
    bundle: MultiDomainDatasetBundle,
    config: MultiDomainGenomeConfig,
) -> tuple[list[SkillNode], list[SkillEdge]]:
    nodes = [
        SkillNode(
            node_id="domain_embedding",
            skill_id="field_embedding",
            skill_name="field_embedding",
            category="embedding",
            params={
                "input_key": "domain_indicator",
                "output_key": "domain_embeddings",
                "vocab_sizes": [bundle.domain_num],
                "embedding_dim": config.domain_embedding_dim,
                "feature_names": ["domain_indicator"],
            },
            input_keys=["domain_indicator"],
            output_keys=["domain_embeddings"],
            task_types=[MULTI_DOMAIN_TASK_TAG],
        ),
        SkillNode(
            node_id="flatten_domain",
            skill_id="flatten_field_embeddings",
            skill_name="flatten_field_embeddings",
            category="utility",
            params={"input_key": "domain_embeddings", "output_key": "domain_context"},
            input_keys=["domain_embeddings"],
            output_keys=["domain_context"],
            task_types=[MULTI_DOMAIN_TASK_TAG],
        ),
    ]
    edges = [SkillEdge("domain_embedding", "flatten_domain", "domain_embeddings", "domain_embeddings")]
    return nodes, edges


def build_multi_domain_genome(
    bundle: MultiDomainDatasetBundle,
    config: MultiDomainGenomeConfig,
    template: str | None = None,
) -> SkillGenome:
    template = template or config.template
    if template not in MULTI_DOMAIN_TEMPLATES:
        raise ValueError(f"Unsupported multi-domain genome template: {template} (expected one of {MULTI_DOMAIN_TEMPLATES})")
    flat_input_dim = _num_total_fields(bundle) * config.embedding_dim

    nodes = _input_nodes(bundle, config, include_domain_adapter=_template_uses_domain_adapter(template))
    transform_nodes, transform_edges, transform_output_node, rep_dim, transform_output_key = _domain_transform_nodes(
        template, bundle, config, flat_input_dim
    )
    nodes.extend(transform_nodes)
    if rep_dim is not None:
        nodes.append(
            SkillNode(
                node_id="domain_towers",
                skill_id="task_tower",
                skill_name="task_tower",
                category="scenario",
                params={
                    "input_dim": rep_dim,
                    "task_types": _domain_task_types(bundle.domain_num),
                    "tower_hidden_dims": list(config.tower_hidden_dims),
                    "input_key": "domain_representations",
                    "output_key": "domain_outputs",
                    "activation": config.activation,
                    "dropout": config.dropout,
                },
                input_keys=["domain_representations"],
                output_keys=["domain_outputs"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
    uses_domain_select = rep_dim is not None or transform_output_key == "domain_outputs"
    if uses_domain_select:
        nodes.append(
            SkillNode(
                node_id="domain_select",
                skill_id="domain_select",
                skill_name="domain_select",
                category="scenario",
                params={
                    "input_key": "domain_outputs",
                    "domain_key": "domain_id",
                    "output_key": "prediction",
                },
                input_keys=["domain_outputs", "domain_id"],
                output_keys=["prediction"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            )
        )
    nodes.extend(
        [
            SkillNode(
                node_id="loss",
                skill_id="bce_loss",
                skill_name="bce_loss",
                category="loss",
                params={
                    "logits_key": "prediction",
                    "labels_key": "labels",
                    "output_key": "loss",
                    "from_logits": False,
                },
                input_keys=["prediction", "labels"],
                output_keys=["loss"],
                task_types=[MULTI_DOMAIN_TASK_TAG],
            ),
        ]
    )

    edges = list(transform_edges)
    if _num_dense_features(bundle) > 0:
        edges.append(SkillEdge("field_embedding", "dense_feature_path", "field_embeddings", "field_embeddings"))
        edges.append(SkillEdge("dense_feature_path", "flatten", "field_embeddings", "field_embeddings"))
    else:
        edges.append(SkillEdge("field_embedding", "flatten", "field_embeddings", "field_embeddings"))
    if rep_dim is not None:
        edges.append(SkillEdge("domain_adapter", "domain_select", "domain_id", "domain_id"))
        edges.append(SkillEdge(transform_output_node, "domain_towers", "domain_representations", "domain_representations"))
        edges.append(SkillEdge("domain_towers", "domain_select", "domain_outputs", "domain_outputs"))
        edges.append(SkillEdge("domain_select", "loss", "prediction", "prediction"))
    elif transform_output_key == "domain_outputs":
        edges.append(SkillEdge("domain_adapter", "domain_select", "domain_id", "domain_id"))
        edges.append(SkillEdge(transform_output_node, "domain_select", "domain_outputs", "domain_outputs"))
        edges.append(SkillEdge("domain_select", "loss", "prediction", "prediction"))
    elif transform_output_key == "prediction":
        edges.append(SkillEdge(transform_output_node, "loss", "prediction", "prediction"))
    else:
        raise ValueError(f"Unsupported multi-domain transform output key: {transform_output_key}")

    required_inputs = ["sparse_features", "domain_indicator", "labels"]
    if _num_dense_features(bundle) > 0:
        required_inputs.append("dense_values")
    genome = SkillGenome(
        nodes=nodes,
        edges=edges,
        objectives=[{"skill_id": "bce_loss", "output_key": "loss"}],
        constraints=GenomeConstraints(
            task_types=[MULTI_DOMAIN_TASK_TAG],
            required_inputs=required_inputs,
            required_outputs=["prediction", "loss"],
        ),
    )
    genome.metadata.tags = [MULTI_DOMAIN_TASK_TAG, template, *bundle.domain_names]
    genome.metadata.extras = {
        **genome.metadata.extras,
        "domain_num": bundle.domain_num,
        "domain_names": list(bundle.domain_names),
        "sparse_feature_names": list(bundle.sparse_feature_names or []),
        "dense_feature_names": list(bundle.dense_feature_names or []),
        "num_sparse_fields": len(bundle.vocab_sizes),
        "num_dense_fields": _num_dense_features(bundle),
        "num_fields": _num_total_fields(bundle),
        "embedding_dim": config.embedding_dim,
        "flat_input_dim": flat_input_dim,
        "template": template,
    }
    genome.record_mutation(
        GenomeMutation(
            mutation_type="multi_domain_baseline",
            child_genome_id=genome.metadata.genome_id,
            description=f"Multi-domain baseline genome with {template} transform",
            details={"template": template, "domain_num": bundle.domain_num},
        )
    )
    return genome


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class MultiDomainGenomeTrainer(CTRGenomeTrainer):
    """Train multi-domain genomes and report global plus per-domain metrics."""

    def __init__(self, training: CTRTrainingConfig, domain_num: int, domain_names: list[str] | None = None) -> None:
        super().__init__(training)
        self.domain_num = int(domain_num)
        self.domain_names = list(domain_names or [f"domain_{idx}" for idx in range(self.domain_num)])

    def _batch_to_device(self, x_dict: dict[str, torch.Tensor], y: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = {key: value.to(self.device) for key, value in x_dict.items()}
        batch["labels"] = y.float().to(self.device)
        return batch

    def _train_one_epoch(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, data_loader: DataLoader) -> float:
        model.train()
        losses: list[float] = []
        for batch_idx, (x_dict, y) in enumerate(data_loader):
            if self.training.max_train_batches is not None and batch_idx >= self.training.max_train_batches:
                break
            batch = self._batch_to_device(x_dict, y)
            optimizer.zero_grad()
            ctx = model(batch)
            loss = ctx.get(self.training.loss_key)
            if loss is None:
                raise RuntimeError("Multi-domain genome did not produce a 'loss' output")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader) -> dict[str, Any]:
        model.eval()
        labels: list[float] = []
        predictions: list[float] = []
        domains: list[int] = []
        losses: list[float] = []
        with torch.no_grad():
            for batch_idx, (x_dict, y) in enumerate(data_loader):
                if self.training.max_eval_batches is not None and batch_idx >= self.training.max_eval_batches:
                    break
                batch = self._batch_to_device(x_dict, y)
                ctx = model(batch)
                pred = ctx.get(self.training.prediction_key)
                if pred is None:
                    raise RuntimeError("Multi-domain genome did not produce 'prediction'")
                loss = ctx.get(self.training.loss_key)
                if loss is not None:
                    losses.append(float(loss.detach().cpu()))
                labels.extend(batch["labels"].detach().cpu().view(-1).tolist())
                predictions.extend(pred.detach().cpu().view(-1).tolist())
                domains.extend(batch["domain_indicator"].detach().cpu().view(-1).long().tolist())

        target = np.asarray(labels, dtype=np.float64)
        pred = np.clip(np.asarray(predictions, dtype=np.float64), 1e-7, 1 - 1e-7)
        domain_arr = np.asarray(domains, dtype=np.int64)
        metrics: dict[str, Any] = {
            "auc": _safe_auc(target, pred),
            "logloss": _safe_logloss(target, pred),
            "loss": float(np.mean(losses)) if losses else None,
            "num_examples": int(len(target)),
        }
        domain_aucs: list[float] = []
        for domain_idx in range(self.domain_num):
            name = self.domain_names[domain_idx] if domain_idx < len(self.domain_names) else f"domain_{domain_idx}"
            key = _metric_safe_domain_name(name, domain_idx)
            mask = domain_arr == domain_idx
            d_target = target[mask]
            d_pred = pred[mask]
            d_auc = _safe_auc(d_target, d_pred) if len(d_target) else float("nan")
            d_logloss = _safe_logloss(d_target, d_pred) if len(d_target) else float("nan")
            metrics[f"auc__{key}"] = d_auc
            metrics[f"logloss__{key}"] = d_logloss
            metrics[f"num_examples__{key}"] = int(len(d_target))
            if not math.isnan(d_auc):
                domain_aucs.append(d_auc)
        metrics["mean_domain_auc"] = float(np.mean(domain_aucs)) if domain_aucs else float("nan")
        return metrics


def _metric_safe_domain_name(name: str, idx: int) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in str(name).lower()).strip("_")
    return slug or f"domain_{idx}"


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _default_multi_domain_templates() -> list[dict[str, Any]]:
    return [
        {"name": "swap_star", "template": "star", "operation": "replace"},
        {"name": "swap_shared_bottom", "template": "shared_bottom", "operation": "replace"},
        {"name": "swap_mmoe", "template": "mmoe", "operation": "replace"},
        {"name": "swap_ple", "template": "ple", "operation": "replace"},
        {"name": "swap_sarnet", "template": "sarnet", "operation": "replace"},
        {"name": "swap_adaptdhm", "template": "adaptdhm", "operation": "replace"},
        {"name": "swap_hamur_small", "template": "hamur_small", "operation": "replace"},
        {"name": "swap_hamur_large", "template": "hamur_large", "operation": "replace"},
        {"name": "swap_m3oe", "template": "m3oe", "operation": "replace"},
        {"name": "swap_ppnet", "template": "ppnet", "operation": "replace"},
        {"name": "swap_m2m", "template": "m2m", "operation": "replace"},
        {"name": "swap_adasparse", "template": "adasparse", "operation": "replace"},
        {"name": "swap_epnet", "template": "epnet", "operation": "replace"},
        {"name": "star_wider", "template": "star", "operation": "specialize", "star_fcn_dims": [256, 128, 64]},
        {"name": "mmoe_more_experts", "template": "mmoe", "operation": "specialize", "num_experts": 6},
        {"name": "ple_more_experts", "template": "ple", "operation": "specialize", "ple_specific_experts": 3, "ple_shared_experts": 2},
        {"name": "deeper_domain_tower", "template": "shared_bottom", "operation": "specialize", "tower_hidden_dims": [64, 32]},
    ]


def _genome_template(genome: SkillGenome) -> str:
    node_ids = genome.node_ids()
    if "star_domain_fcn" in node_ids:
        return "star"
    if "domain_mmoe_gate" in node_ids:
        return "mmoe"
    if "domain_ple_gate" in node_ids:
        return "ple"
    if "sarnet_expert_mixer" in node_ids:
        return "sarnet"
    if "adaptdhm_cluster_tower" in node_ids:
        return "adaptdhm"
    if "hamur_domain_adapter_tower" in node_ids:
        dims = list(genome.get_node("hamur_domain_adapter_tower").params.get("fcn_dims", []))
        return "hamur_large" if len(dims) >= 3 else "hamur_small"
    if "m3oe_expert_fusion" in node_ids:
        return "m3oe"
    if "ppnet_domain_towers" in node_ids:
        return "ppnet"
    if "m2m_meta_tower" in node_ids:
        return "m2m"
    if "adasparse_pruned_tower" in node_ids:
        return "adasparse"
    if "epnet_gate" in node_ids:
        return "epnet"
    return "shared_bottom"


def _config_from_parent(genome: SkillGenome, base: MultiDomainGenomeConfig) -> MultiDomainGenomeConfig:
    cfg = replace(base)
    cfg.template = _genome_template(genome)
    try:
        cfg.embedding_dim = int(genome.get_node("field_embedding").params.get("embedding_dim", cfg.embedding_dim))
    except KeyError:
        pass
    node_ids = genome.node_ids()
    if "star_domain_fcn" in node_ids:
        node = genome.get_node("star_domain_fcn")
        cfg.star_fcn_dims = list(node.params.get("fcn_dims", cfg.star_fcn_dims))
        cfg.star_aux_dims = list(node.params.get("aux_dims", cfg.star_aux_dims))
    if "shared_bottom" in node_ids:
        cfg.shared_hidden_dims = list(genome.get_node("shared_bottom").params.get("hidden_dims", cfg.shared_hidden_dims))
    if "domain_mmoe_gate" in node_ids:
        node = genome.get_node("domain_mmoe_gate")
        cfg.num_experts = int(node.params.get("n_expert", cfg.num_experts))
        dims = list(node.params.get("expert_hidden_dims", []))
        if dims:
            cfg.expert_dim = int(dims[-1])
            cfg.shared_hidden_dims = dims
    if "domain_ple_gate" in node_ids:
        node = genome.get_node("domain_ple_gate")
        cfg.ple_shared_experts = int(node.params.get("n_expert_shared", cfg.ple_shared_experts))
        cfg.ple_specific_experts = int(node.params.get("n_expert_specific", cfg.ple_specific_experts))
        dims = list(node.params.get("expert_hidden_dims", []))
        if dims:
            cfg.expert_dim = int(dims[-1])
            cfg.shared_hidden_dims = dims
    if "sarnet_expert_mixer" in node_ids:
        node = genome.get_node("sarnet_expert_mixer")
        cfg.sarnet_shared_experts = int(node.params.get("domain_shared_expert_num", cfg.sarnet_shared_experts))
        cfg.sarnet_specific_experts = int(node.params.get("domain_specific_expert_num", cfg.sarnet_specific_experts))
        cfg.sarnet_expert_dim = int(node.params.get("expert_dim", cfg.sarnet_expert_dim))
    if "adaptdhm_cluster_tower" in node_ids:
        node = genome.get_node("adaptdhm_cluster_tower")
        cfg.adaptdhm_fcn_dims = list(node.params.get("fcn_dims", cfg.adaptdhm_fcn_dims))
        cfg.adaptdhm_cluster_num = int(node.params.get("cluster_num", cfg.adaptdhm_cluster_num or 0)) or cfg.adaptdhm_cluster_num
        cfg.adaptdhm_beta = float(node.params.get("beta", cfg.adaptdhm_beta))
    if "hamur_domain_adapter_tower" in node_ids:
        node = genome.get_node("hamur_domain_adapter_tower")
        dims = list(node.params.get("fcn_dims", []))
        if _genome_template(genome) == "hamur_large":
            cfg.hamur_large_fcn_dims = dims or cfg.hamur_large_fcn_dims
        else:
            cfg.hamur_small_fcn_dims = dims or cfg.hamur_small_fcn_dims
        cfg.hamur_hyper_dims = list(node.params.get("hyper_dims", cfg.hamur_hyper_dims))
        cfg.hamur_k = int(node.params.get("k", cfg.hamur_k))
        cfg.hamur_adapter_dim = int(node.params.get("adapter_dim", cfg.hamur_adapter_dim))
    if "m3oe_expert_fusion" in node_ids:
        node = genome.get_node("m3oe_expert_fusion")
        cfg.m3oe_fcn_dims = list(node.params.get("fcn_dims", cfg.m3oe_fcn_dims))
        cfg.m3oe_expert_num = int(node.params.get("expert_num", cfg.m3oe_expert_num))
        cfg.m3oe_exp_d = float(node.params.get("exp_d", cfg.m3oe_exp_d))
        cfg.m3oe_bal_d = float(node.params.get("bal_d", cfg.m3oe_bal_d))
    if "ppnet_domain_towers" in node_ids:
        cfg.ppnet_fcn_dims = list(genome.get_node("ppnet_domain_towers").params.get("fcn_dims", cfg.ppnet_fcn_dims))
    if "domain_embedding" in node_ids:
        cfg.domain_embedding_dim = int(genome.get_node("domain_embedding").params.get("embedding_dim", cfg.domain_embedding_dim))
    if "m2m_meta_tower" in node_ids:
        node = genome.get_node("m2m_meta_tower")
        cfg.m2m_num_experts = int(node.params.get("num_experts", cfg.m2m_num_experts))
        cfg.m2m_expert_output_size = int(node.params.get("expert_output_size", cfg.m2m_expert_output_size))
        cfg.m2m_transformer_dims = dict(node.params.get("transformer_dims", cfg.m2m_transformer_dims))
    if "adasparse_pruned_tower" in node_ids:
        node = genome.get_node("adasparse_pruned_tower")
        cfg.adasparse_hidden_dims = list(node.params.get("hidden_dims", cfg.adasparse_hidden_dims))
        cfg.adasparse_form = str(node.params.get("form", cfg.adasparse_form))
        cfg.adasparse_epsilon = float(node.params.get("epsilon", cfg.adasparse_epsilon))
        cfg.adasparse_beta = float(node.params.get("beta", cfg.adasparse_beta))
        cfg.adasparse_alpha = float(node.params.get("alpha", cfg.adasparse_alpha))
        cfg.adasparse_delta_alpha = float(node.params.get("delta_alpha", cfg.adasparse_delta_alpha))
    if "epnet_gate" in node_ids:
        cfg.epnet_gamma = float(genome.get_node("epnet_gate").params.get("gamma", cfg.epnet_gamma))
    if "epnet_prediction_tower" in node_ids:
        cfg.epnet_fcn_dims = list(genome.get_node("epnet_prediction_tower").params.get("hidden_dims", cfg.epnet_fcn_dims))
    if "domain_towers" in node_ids:
        cfg.tower_hidden_dims = list(genome.get_node("domain_towers").params.get("tower_hidden_dims", cfg.tower_hidden_dims))
    return cfg


def _candidate_from_template(
    parent: SkillGenome,
    bundle: MultiDomainDatasetBundle,
    base_genome_config: MultiDomainGenomeConfig,
    template: dict[str, Any],
    round_idx: int,
) -> CandidateSpec:
    cfg = _config_from_parent(parent, base_genome_config)
    target_template = str(template.get("template") or cfg.template)
    cfg.template = target_template
    for key in (
        "embedding_dim",
        "shared_hidden_dims",
        "tower_hidden_dims",
        "expert_dim",
        "num_experts",
        "ple_shared_experts",
        "ple_specific_experts",
        "star_fcn_dims",
        "star_aux_dims",
        "sarnet_shared_experts",
        "sarnet_specific_experts",
        "sarnet_expert_dim",
        "adaptdhm_fcn_dims",
        "adaptdhm_cluster_num",
        "adaptdhm_beta",
        "hamur_small_fcn_dims",
        "hamur_large_fcn_dims",
        "hamur_hyper_dims",
        "hamur_k",
        "hamur_adapter_dim",
        "m3oe_fcn_dims",
        "m3oe_expert_num",
        "m3oe_exp_d",
        "m3oe_bal_d",
        "ppnet_fcn_dims",
        "domain_embedding_dim",
        "m2m_num_experts",
        "m2m_expert_output_size",
        "m2m_transformer_dims",
        "adasparse_hidden_dims",
        "adasparse_form",
        "adasparse_epsilon",
        "adasparse_beta",
        "adasparse_alpha",
        "adasparse_delta_alpha",
        "epnet_fcn_dims",
        "epnet_gamma",
        "dropout",
    ):
        if key in template:
            setattr(cfg, key, template[key])
    child = build_multi_domain_genome(bundle, cfg, template=target_template)
    child.metadata.parent_genome_ids = list(dict.fromkeys([*child.metadata.parent_genome_ids, parent.metadata.genome_id]))
    child.metadata.lineage = list(dict.fromkeys([*child.metadata.lineage, parent.metadata.genome_id]))
    name = str(template.get("name") or target_template)
    operation = str(template.get("operation", "replace"))
    candidate_id = f"round{round_idx}_{name}"
    return CandidateSpec(
        candidate_id=candidate_id,
        genome=child,
        parent_genome_id=parent.metadata.genome_id,
        mutation_type=name,
        rationale=f"Multi-domain {operation}: transform -> {target_template} ({name})",
        evolution_space="skill_space",
        operation=operation,
        architecture_fingerprint=_safe_architecture_fingerprint(child, fallback=candidate_id),
    )


def _multi_domain_reuse_promoted_generated_skills_enabled(evolution: CTREvolutionConfig) -> bool:
    code_space = evolution.code_space
    if isinstance(code_space, dict):
        return bool(code_space.get("reuse_promoted_generated_skills", code_space.get("reuse_promoted_code_skills", False)))
    return bool(getattr(code_space, "reuse_promoted_generated_skills", getattr(code_space, "reuse_promoted_code_skills", False)))


def _multi_domain_choice_skill_library(evolution: CTREvolutionConfig, skill_library: SkillLibrary | None) -> SkillLibrary | None:
    if skill_library is not None:
        return skill_library
    try:
        return SkillLibrary.from_repo(include_generated=_multi_domain_reuse_promoted_generated_skills_enabled(evolution))
    except Exception:
        return None


def _multi_domain_template_pool(evolution: CTREvolutionConfig, *, target: int) -> list[dict[str, Any]]:
    desired = max(target, len(evolution.templates or []) + len(_default_multi_domain_templates()))
    templates: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for template in evolution.templates or []:
        item = dict(template)
        item["_multi_domain_configured_template"] = True
        templates.append(item)
        name = str(item.get("name") or "")
        if name:
            seen_names.add(name)
    defaults = _default_multi_domain_templates()
    for template in defaults:
        name = str(template.get("name") or "")
        if name in seen_names:
            continue
        item = dict(template)
        item["_multi_domain_configured_template"] = False
        templates.append(item)
        seen_names.add(name)
    idx = 0
    while len(templates) < desired and defaults:
        item = dict(defaults[idx % len(defaults)])
        item["_multi_domain_configured_template"] = False
        templates.append(item)
        idx += 1
    return templates


def _multi_domain_primary_skill_id_for_template(template: dict[str, Any]) -> str:
    target_template = str(template.get("template") or "")
    skill_ids = MULTI_DOMAIN_TEMPLATE_SKILL_IDS.get(target_template) or ()
    return skill_ids[0] if skill_ids else target_template


def _multi_domain_template_cards(template: dict[str, Any], library: SkillLibrary | None) -> list[SkillCard]:
    if library is None:
        return []
    cards: list[SkillCard] = []
    for skill_id in MULTI_DOMAIN_TEMPLATE_SKILL_IDS.get(str(template.get("template") or ""), ()):
        try:
            if library.has(skill_id):
                cards.append(library.get(skill_id))
        except Exception:
            continue
    return cards


def _multi_domain_template_primary_card(template: dict[str, Any], library: SkillLibrary | None) -> SkillCard | None:
    cards = _multi_domain_template_cards(template, library)
    return cards[0] if cards else None


def _score_multi_domain_predefined_template(
    template: dict[str, Any],
    parents: list[SkillGenome],
    evolution: CTREvolutionConfig,
    library: SkillLibrary | None,
) -> int:
    score = 20
    target_template = str(template.get("template") or "")
    operation = str(template.get("operation") or "")
    if template.get("_multi_domain_configured_template"):
        score += 8
    if operation == "replace":
        score += 4
    elif operation == "specialize":
        score += 5
    if target_template in {"mmoe", "ple", "m3oe"}:
        score += 7
    elif target_template in {"sarnet", "hamur_small", "hamur_large", "m2m"}:
        score += 6
    elif target_template in {"star", "ppnet"}:
        score += 5
    elif target_template in {"adaptdhm", "adasparse", "epnet"}:
        score += 4
    parent_templates = {_genome_template(parent) for parent in parents}
    if target_template and target_template not in parent_templates:
        score += 4
    elif operation != "specialize":
        score -= 3

    cards = _multi_domain_template_cards(template, library)
    if not cards:
        return score
    if any(MULTI_DOMAIN_TASK_TAG in {str(task).lower() for task in card.task_types} for card in cards):
        score += 10
    text = " ".join(_multi_domain_card_text(card) for card in cards)
    for token in ["domain", "scenario", "routing", "gate", "expert", "tower", "adapter", "selection", "multi-domain"]:
        if token in text:
            score += 2
    failure_text = " ".join(str(item).lower() for item in evolution.failure_modes or [])
    if failure_text:
        for token in ["underfitting", "domain", "scenario", "transfer", "tower", "routing", "expert"]:
            if token in failure_text and token in text:
                score += 2
    parent_skill_ids = {node.skill_id for parent in parents for node in parent.nodes}
    for card in cards:
        composition = card.manifest.get("composition") or {}
        upstream = {str(item) for item in composition.get("common_upstream", []) or []}
        downstream = {str(item) for item in composition.get("common_downstream", []) or []}
        if upstream & parent_skill_ids:
            score += 2
        if downstream & parent_skill_ids:
            score += 2
        retrieval_tasks = {str(task).lower() for task in (card.manifest.get("retrieval") or {}).get("task_types", [])}
        if MULTI_DOMAIN_TASK_TAG in retrieval_tasks:
            score += 3
    return score


def _build_unified_multi_domain_skill_choices(
    evolution: CTREvolutionConfig,
    parents: list[SkillGenome],
    *,
    target: int,
    round_idx: int,
    skill_library: SkillLibrary | None = None,
) -> tuple[list[_MultiDomainSkillChoice], SkillLibrary | None]:
    library = _multi_domain_choice_skill_library(evolution, skill_library)
    choices: list[_MultiDomainSkillChoice] = []
    for idx, template in enumerate(_multi_domain_template_pool(evolution, target=target)):
        template = dict(template)
        template_name = str(template.get("name") or template.get("template") or f"template_{idx}")
        choices.append(
            _MultiDomainSkillChoice(
                source="predefined",
                score=_score_multi_domain_predefined_template(template, parents, evolution, library),
                name=template_name,
                order=idx,
                template=template,
                skill_id=_multi_domain_primary_skill_id_for_template(template),
                card=_multi_domain_template_primary_card(template, library),
            )
        )
    if library is not None and _multi_domain_reuse_promoted_generated_skills_enabled(evolution):
        choices.extend(_multi_domain_reusable_generated_skill_choices(library, parents, round_idx=round_idx))
    return choices, library


def _rank_multi_domain_skill_choices(
    choices: list[_MultiDomainSkillChoice],
    *,
    target: int,
    round_idx: int,
) -> list[_MultiDomainSkillChoice]:
    if target <= 0:
        return []
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
    ranked = _ensure_multi_domain_skill_source_mix(ranked, target=target)
    ranked = _ensure_multi_domain_template_family_mix(ranked, target=target)
    return _rotate_multi_domain_tail_for_exploration(ranked, target=target, round_idx=round_idx)


def _ensure_multi_domain_skill_source_mix(choices: list[_MultiDomainSkillChoice], *, target: int) -> list[_MultiDomainSkillChoice]:
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
        replace_pos = _multi_domain_replaceable_selected_choice_index(selected)
        replacement = tail.pop(replacement_idx)
        displaced = selected[replace_pos]
        selected[replace_pos] = replacement
        tail.insert(0, displaced)
    return selected + tail


def _ensure_multi_domain_template_family_mix(
    choices: list[_MultiDomainSkillChoice],
    *,
    target: int,
) -> list[_MultiDomainSkillChoice]:
    if target <= 1 or len(choices) <= target:
        return choices
    selected = list(choices[:target])
    tail = list(choices[target:])
    used = {_multi_domain_choice_family(choice) for choice in selected if _multi_domain_choice_family(choice)}
    counts: dict[str, int] = {}
    for choice in selected:
        family = _multi_domain_choice_family(choice)
        if family:
            counts[family] = counts.get(family, 0) + 1
    for selected_idx in range(len(selected) - 1, -1, -1):
        family = _multi_domain_choice_family(selected[selected_idx])
        if not family or counts.get(family, 0) <= 1:
            continue
        replacement_idx = next(
            (
                idx
                for idx, choice in enumerate(tail)
                if _multi_domain_choice_family(choice) and _multi_domain_choice_family(choice) not in used
            ),
            None,
        )
        if replacement_idx is None:
            break
        replacement = tail.pop(replacement_idx)
        displaced = selected[selected_idx]
        selected[selected_idx] = replacement
        tail.insert(0, displaced)
        counts[family] -= 1
        replacement_family = _multi_domain_choice_family(replacement)
        if replacement_family:
            used.add(replacement_family)
            counts[replacement_family] = counts.get(replacement_family, 0) + 1
    return selected + tail


def _multi_domain_choice_family(choice: _MultiDomainSkillChoice) -> str:
    if choice.template is not None:
        return str(choice.template.get("template") or choice.name)
    return str(choice.skill_id or choice.name)


def _multi_domain_replaceable_selected_choice_index(selected: list[_MultiDomainSkillChoice]) -> int:
    counts: dict[str, int] = {}
    for choice in selected:
        counts[choice.source] = counts.get(choice.source, 0) + 1
    for idx in range(len(selected) - 1, -1, -1):
        if counts.get(selected[idx].source, 0) > 1:
            return idx
    return len(selected) - 1


def _rotate_multi_domain_tail_for_exploration(
    choices: list[_MultiDomainSkillChoice],
    *,
    target: int,
    round_idx: int,
) -> list[_MultiDomainSkillChoice]:
    if target <= 0 or len(choices) <= target:
        return choices
    if round_idx <= 0:
        return choices
    selected = list(choices[:target])
    tail = list(choices[target:])
    if not tail:
        return choices
    replace_pos = max(0, target - 1)
    tail_idx = (round_idx - 1) % len(tail)
    selected[replace_pos], tail[tail_idx] = tail[tail_idx], selected[replace_pos]
    return selected + tail


def _multi_domain_parent_for_skill_choice(
    choice: _MultiDomainSkillChoice,
    parents: list[SkillGenome],
    *,
    round_idx: int,
    choice_idx: int,
) -> tuple[SkillGenome, int]:
    if not parents:
        raise ValueError("No multi-domain parents available for candidate generation")
    if choice.source == "generated" and choice.card is not None:
        for offset in range(len(parents)):
            parent_idx = (round_idx + choice_idx + offset) % len(parents)
            if _multi_domain_reuse_lane_for_card(choice.card, parents[parent_idx]) is not None:
                return parents[parent_idx], parent_idx
    parent_idx = choice_idx % len(parents)
    return parents[parent_idx], parent_idx


def _candidate_from_multi_domain_skill_choice(
    choice: _MultiDomainSkillChoice,
    parent: SkillGenome,
    bundle: MultiDomainDatasetBundle,
    base_genome_config: MultiDomainGenomeConfig,
    *,
    round_idx: int,
    library: SkillLibrary | None,
) -> CandidateSpec:
    if choice.source == "generated":
        if choice.card is None or library is None:
            raise ValueError("Generated multi-domain skill choice requires a skill card and library")
        spec = _candidate_from_multi_domain_promoted_generated_skill(parent, choice.card, round_idx=round_idx, library=library)
    else:
        if choice.template is None:
            raise ValueError("Predefined multi-domain skill choice requires a template")
        spec = _candidate_from_template(parent, bundle, base_genome_config, choice.template, round_idx)
    spec.rationale = f"{spec.rationale} (multi-domain skill selection: source={choice.source}, score={choice.score})"
    return spec


def _multi_domain_reusable_generated_skill_choices(
    library: SkillLibrary,
    parents: list[SkillGenome],
    *,
    round_idx: int,
) -> list[_MultiDomainSkillChoice]:
    scored: list[tuple[int, str, SkillCard]] = []
    for skill_id in library.list_skill_ids():
        try:
            card = library.get(skill_id)
        except Exception:
            continue
        if not _is_multi_domain_reusable_generated_skill_card(card, parents):
            continue
        scored.append((_score_multi_domain_reusable_generated_card(card, parents), skill_id, card))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1]))
    shift = round_idx % len(scored)
    rotated = scored[shift:] + scored[:shift]
    return [
        _MultiDomainSkillChoice(
            source="generated",
            score=score,
            name=skill_id,
            order=idx,
            skill_id=skill_id,
            card=card,
        )
        for idx, (score, skill_id, card) in enumerate(rotated)
    ]


def _is_multi_domain_reusable_generated_skill_card(card: SkillCard, parents: list[SkillGenome]) -> bool:
    if not _is_reusable_generated_skill_card(card):
        return False
    if not _skill_card_has_loadable_implementation(card):
        return False
    if not _generated_skill_portability_allows_reuse(card):
        return False
    tasks = {str(task).lower() for task in card.task_types}
    retrieval_tasks = {str(task).lower() for task in (card.manifest.get("retrieval") or {}).get("task_types", [])}
    if tasks or retrieval_tasks:
        if MULTI_DOMAIN_TASK_TAG not in (tasks | retrieval_tasks):
            return False
    return any(_multi_domain_reuse_lane_for_card(card, parent) is not None for parent in parents)


def _score_multi_domain_reusable_generated_card(card: SkillCard, parents: list[SkillGenome]) -> int:
    score = 35
    tasks = {str(task).lower() for task in card.task_types}
    retrieval_tasks = {str(task).lower() for task in (card.manifest.get("retrieval") or {}).get("task_types", [])}
    if MULTI_DOMAIN_TASK_TAG in tasks or MULTI_DOMAIN_TASK_TAG in retrieval_tasks:
        score += 20
    text = _multi_domain_card_text(card)
    for token in ["domain", "scenario", "routing", "gate", "expert", "representation", "flat", "field"]:
        if token in text:
            score += 2
    if any(_multi_domain_reuse_lane_for_card(card, parent) in {"domain_representations", "domain_outputs"} for parent in parents):
        score += 6
    metrics = card.manifest.get("candidate_metrics") or (card.manifest.get("metadata") or {}).get("candidate_metrics") or {}
    try:
        score += min(10, max(0, int(round(float(metrics.get("validation_best_auc", metrics.get("auc", 0.0))) * 10))))
    except Exception:
        pass
    return score


def _candidate_from_multi_domain_promoted_generated_skill(
    parent: SkillGenome,
    card: SkillCard,
    *,
    round_idx: int,
    library: SkillLibrary,
) -> CandidateSpec:
    lane = _multi_domain_reuse_lane_for_card(card, parent)
    if lane is None:
        raise ValueError(f"Promoted generated skill is not compatible with this multi-domain parent: {card.skill_id}")
    input_key = lane
    output_key = _resolved_multi_domain_generated_output_key(card, parent, round_idx=round_idx, input_key=input_key)
    params = _multi_domain_generated_skill_params(card, parent, input_key=input_key, output_key=output_key)
    base_id = _multi_domain_slug(card.skill_id)
    node = SkillNode(
        node_id=_unique_multi_domain_node_id(parent, f"{base_id}_r{round_idx}"),
        skill_id=card.skill_id,
        skill_name=card.skill_name,
        category=card.category or "generated",
        params=params,
        input_keys=[input_key],
        output_keys=[output_key],
        task_types=[MULTI_DOMAIN_TASK_TAG],
        source="generated_skill",
        metadata={
            "evolution_space": "skill_space",
            "reused_generated_skill": True,
            "source_proposal_id": card.manifest.get("source_proposal_id"),
            "runtime_skill_card_path": str(card.path) if card.path else "",
        },
    )
    child = parent.clone()
    if not _insert_multi_domain_generated_node_on_lane(child, node, input_key=input_key, output_key=output_key):
        raise ValueError(f"Could not insert promoted generated skill on multi-domain lane: {input_key}")
    GenomeVerifier(skill_library=library).assert_valid(child)
    child.record_mutation(
        GenomeMutation(
            mutation_type=f"multi_domain_skill_reuse_generated_{card.skill_id}",
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
        mutation_type=f"multi_domain_skill_reuse_generated_{card.skill_id}",
        rationale=f"Reuse promoted generated skill {card.skill_id} on multi-domain {input_key} lane",
        evolution_space="skill_space",
        operation="reuse_generated",
        architecture_fingerprint=_safe_architecture_fingerprint(child, fallback=candidate_id),
        generated_skill_id=card.skill_id,
        proposal_id=card.manifest.get("source_proposal_id"),
    )


def _multi_domain_reuse_lane_for_card(card: SkillCard, parent: SkillGenome) -> str | None:
    input_keys = _resolved_multi_domain_generated_input_keys(card, parent)
    if len(input_keys) != 1:
        return None
    input_key = input_keys[0]
    if input_key not in {"field_embeddings", "flat_embeddings", "domain_representations", "domain_outputs"}:
        return None
    if _is_multi_domain_logit_key(input_key) or any(_is_multi_domain_logit_key(key) for key in _resolved_multi_domain_generated_output_keys(card)):
        return None
    if _current_multi_domain_lane_edge(parent, input_key) is None:
        return None
    if not _multi_domain_generated_shape_compatible(card, parent, lane=input_key):
        return None
    return input_key


def _resolved_multi_domain_generated_input_keys(card: SkillCard, parent: SkillGenome) -> list[str]:
    available = _available_multi_domain_tensor_keys(parent)
    keys = [key for key in card.input_keys if key and not _is_multi_domain_param_placeholder(key)]
    if keys:
        return keys if set(keys) <= available else []
    params = _multi_domain_generated_fragment_params(card)
    for param_name in ("input_key", "flat_key", "field_key", "context_key", "domain_key"):
        value = str(params.get(param_name) or "")
        if value in available:
            return [value]
    text = _multi_domain_card_text(card)
    for key in ("domain_outputs", "domain_representations", "flat_embeddings", "field_embeddings"):
        if key in available and key in text:
            return [key]
    return []


def _resolved_multi_domain_generated_output_keys(card: SkillCard) -> list[str]:
    keys = [key for key in card.output_keys if key and not _is_multi_domain_param_placeholder(key)]
    if keys:
        return keys
    params = _multi_domain_generated_fragment_params(card)
    for param_name in ("output_key", "flat_output_key", "field_output_key", "domain_output_key"):
        value = str(params.get(param_name) or "")
        if value and not _is_multi_domain_param_placeholder(value):
            return [value]
    return []


def _resolved_multi_domain_generated_output_key(card: SkillCard, parent: SkillGenome, *, round_idx: int, input_key: str) -> str:
    output_keys = _resolved_multi_domain_generated_output_keys(card)
    output_key = output_keys[0] if len(output_keys) == 1 else ""
    if not output_key or output_key == input_key or output_key in parent.produced_keys():
        output_key = _unique_multi_domain_key(parent, f"{_multi_domain_slug(card.skill_id)}_r{round_idx}_{input_key}")
    return output_key


def _multi_domain_generated_skill_params(card: SkillCard, parent: SkillGenome, *, input_key: str, output_key: str) -> dict[str, Any]:
    params = _multi_domain_generated_fragment_params(card)
    params["input_key"] = input_key
    params["output_key"] = output_key
    if input_key == "flat_embeddings" and "flat_key" in params:
        params["flat_key"] = input_key
    if input_key == "field_embeddings" and "field_key" in params:
        params["field_key"] = input_key
    if input_key in {"domain_representations", "domain_outputs"} and "domain_key" in params:
        params["domain_key"] = input_key
    for key, value in list(params.items()):
        params[key] = _resolve_multi_domain_generated_param_value(value, parent)
    return params


def _multi_domain_generated_fragment_params(card: SkillCard) -> dict[str, Any]:
    fragment = (card.manifest.get("composition") or {}).get("example_genome_fragment") or {}
    if isinstance(fragment, str):
        try:
            fragment = yaml.safe_load(fragment) or {}
        except Exception:
            fragment = {}
    return dict(fragment.get("params") or {}) if isinstance(fragment, dict) else {}


def _resolve_multi_domain_generated_param_value(value: Any, parent: SkillGenome) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    runtime = _multi_domain_runtime_shape(parent)
    if normalized in {"${num_fields}", "${field_count}"}:
        return runtime["num_fields"]
    if normalized == "${embedding_dim}":
        return runtime["embedding_dim"]
    if normalized in {"${input_dim}", "${flat_input_dim}", "${fusion_dim}"}:
        return runtime["flat_input_dim"]
    if normalized in {"${domain_num}", "${num_domains}", "${n_task}"}:
        return runtime["domain_num"]
    if normalized in {"${domain_representation_dim}", "${representation_dim}"}:
        return runtime["domain_representation_dim"]
    return value


def _multi_domain_generated_shape_compatible(card: SkillCard, parent: SkillGenome, *, lane: str) -> bool:
    runtime = _multi_domain_runtime_shape(parent)
    params = _multi_domain_generated_fragment_params(card)
    for name, current_value in [
        ("num_fields", runtime["num_fields"]),
        ("embedding_dim", runtime["embedding_dim"]),
        ("input_dim", runtime["flat_input_dim"]),
        ("domain_num", runtime["domain_num"]),
    ]:
        if not _multi_domain_static_shape_value_compatible(params.get(name), current_value):
            return False
    input_shape = _multi_domain_signature_shape(card.manifest.get("input_signature") or card.manifest.get("inputs") or [], lane)
    output_shape = _multi_domain_first_signature_shape(card.manifest.get("output_signature") or card.manifest.get("outputs") or [])
    if input_shape and not _multi_domain_shape_spec_compatible(input_shape, name=lane, current=runtime):
        return False
    if output_shape:
        if not _multi_domain_shape_spec_compatible(output_shape, name=lane, current=runtime):
            return False
        if lane == "field_embeddings" and len(output_shape) != 3:
            return False
        if lane == "flat_embeddings" and len(output_shape) != 2:
            return False
        if lane == "domain_representations" and len(output_shape) != 3:
            return False
        if lane == "domain_outputs" and len(output_shape) != 2:
            return False
        if len(output_shape) == 2 and _multi_domain_shape_dim_is_static_one(output_shape[-1]):
            return False
        return True
    text = " ".join([_multi_domain_card_text(card), " ".join(_resolved_multi_domain_generated_output_keys(card))])
    if lane == "field_embeddings":
        return "field" in text and "embedding" in text
    if lane == "flat_embeddings":
        return "flat" in text or "representation" in text
    if lane == "domain_representations":
        return "domain" in text and "representation" in text
    return "domain" in text and ("output" in text or "score" in text)


def _multi_domain_runtime_shape(parent: SkillGenome) -> dict[str, int]:
    extras = parent.metadata.extras or {}
    embedding_dim = int(extras.get("embedding_dim") or _multi_domain_embedding_dim(parent))
    num_fields = int(extras.get("num_fields") or (_multi_domain_sparse_field_count(parent) + _multi_domain_dense_field_count(parent)))
    domain_num = int(extras.get("domain_num") or _multi_domain_domain_count(parent))
    domain_representation_dim = int(extras.get("domain_representation_dim") or 1)
    try:
        domain_tower = parent.get_node("domain_towers")
        domain_representation_dim = int(domain_tower.params.get("input_dim") or domain_representation_dim)
    except Exception:
        pass
    return {
        "num_fields": max(1, num_fields),
        "embedding_dim": max(1, embedding_dim),
        "flat_input_dim": max(1, num_fields) * max(1, embedding_dim),
        "domain_num": max(1, domain_num),
        "domain_representation_dim": max(1, domain_representation_dim),
    }


def _multi_domain_sparse_field_count(parent: SkillGenome) -> int:
    try:
        return len(parent.get_node("field_embedding").params.get("vocab_sizes") or [])
    except Exception:
        return 0


def _multi_domain_dense_field_count(parent: SkillGenome) -> int:
    try:
        return int(parent.get_node("dense_feature_path").params.get("num_dense") or 0)
    except Exception:
        return 0


def _multi_domain_embedding_dim(parent: SkillGenome) -> int:
    try:
        return int(parent.get_node("field_embedding").params.get("embedding_dim") or 1)
    except Exception:
        return 1


def _multi_domain_domain_count(parent: SkillGenome) -> int:
    for node_id in ("domain_adapter", "domain_towers", "domain_select"):
        try:
            node = parent.get_node(node_id)
        except Exception:
            continue
        for key in ("num_domains", "domain_num", "n_task"):
            if node.params.get(key) is not None:
                try:
                    return int(node.params[key])
                except Exception:
                    continue
    return 1


def _multi_domain_signature_shape(signature: Any, name: str) -> list[Any]:
    if not isinstance(signature, list):
        return []
    for item in signature:
        if isinstance(item, dict) and str(item.get("name") or "") == name:
            shape = item.get("shape") or []
            return list(shape) if isinstance(shape, list) else _multi_domain_shape_list_from_string(shape)
    return []


def _multi_domain_first_signature_shape(signature: Any) -> list[Any]:
    if not isinstance(signature, list) or not signature or not isinstance(signature[0], dict):
        return []
    shape = signature[0].get("shape") or []
    return list(shape) if isinstance(shape, list) else _multi_domain_shape_list_from_string(shape)


def _multi_domain_shape_list_from_string(shape: Any) -> list[str]:
    if not isinstance(shape, str):
        return []
    return [part.strip() for part in shape.strip().strip("[]").split(",") if part.strip()]


def _multi_domain_shape_spec_compatible(shape: list[Any], *, name: str, current: dict[str, int]) -> bool:
    if not shape:
        return True
    if len(shape) == 3 and name == "field_embeddings":
        return _multi_domain_shape_dim_compatible(shape[1], current["num_fields"]) and _multi_domain_shape_dim_compatible(shape[2], current["embedding_dim"])
    if len(shape) == 2 and name == "flat_embeddings":
        return _multi_domain_shape_dim_compatible(shape[1], current["flat_input_dim"])
    if len(shape) == 3 and name == "domain_representations":
        return _multi_domain_shape_dim_compatible(shape[1], current["domain_num"]) and _multi_domain_shape_dim_compatible(shape[2], current["domain_representation_dim"])
    if len(shape) == 2 and name == "domain_outputs":
        return _multi_domain_shape_dim_compatible(shape[1], current["domain_num"])
    return True


def _multi_domain_shape_dim_compatible(dim: Any, current_value: int) -> bool:
    if isinstance(dim, str):
        dim = dim.strip()
        if dim in {
            "num_fields",
            "embedding_dim",
            "input_dim",
            "flat_input_dim",
            "domain_num",
            "num_domains",
            "n_task",
            "domain_representation_dim",
        }:
            return True
        if dim.startswith("${") or not dim.isdigit():
            return True
    try:
        return int(dim) == int(current_value)
    except Exception:
        return True


def _multi_domain_static_shape_value_compatible(value: Any, current_value: int) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        value = value.strip()
        if not value or value.startswith("${"):
            return True
        if value in {"num_fields", "embedding_dim", "input_dim", "flat_input_dim", "domain_num", "num_domains"}:
            return True
        if not value.isdigit():
            return True
    try:
        return int(value) == int(current_value)
    except Exception:
        return True


def _multi_domain_shape_dim_is_static_one(dim: Any) -> bool:
    try:
        return int(dim) == 1
    except Exception:
        return False


def _available_multi_domain_tensor_keys(parent: SkillGenome) -> set[str]:
    return set(parent.constraints.required_inputs) | parent.produced_keys() | {"sparse_features", "dense_values", "domain_indicator", "labels"}


def _is_multi_domain_param_placeholder(key: str) -> bool:
    return key in {"input_key", "input_keys", "output_key", "output_keys", "logits_key", "labels_key", "domain_key"}


def _is_multi_domain_logit_key(key: str) -> bool:
    lowered = str(key).lower()
    return lowered == "logits" or "logit" in lowered


def _multi_domain_card_text(card: SkillCard) -> str:
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
    return _lower_multi_domain_text(values)


def _lower_multi_domain_text(values: list[Any]) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            parts.append(_lower_multi_domain_text(list(value.values())))
        elif isinstance(value, (list, tuple, set)):
            parts.append(_lower_multi_domain_text(list(value)))
        else:
            parts.append(str(value))
    return " ".join(parts).lower()


def _multi_domain_slug(value: str) -> str:
    slug = "".join(char if char.isalnum() or char == "_" else "_" for char in str(value).lower()).strip("_")
    return slug or "generated_skill"


def _insert_multi_domain_generated_node_on_lane(
    genome: SkillGenome,
    node: SkillNode,
    *,
    input_key: str,
    output_key: str,
) -> bool:
    if output_key in genome.produced_keys():
        return False
    edge = _current_multi_domain_lane_edge(genome, input_key)
    if edge is None:
        return False
    cloned = SkillNode.from_dict(copy.deepcopy(asdict(node)))
    cloned.node_id = _unique_multi_domain_node_id(genome, cloned.node_id)
    genome.nodes.append(cloned)
    genome.edges = [item for item in genome.edges if item != edge]
    genome.edges.append(SkillEdge(edge.src_node_id, cloned.node_id, edge.src_output_key, input_key, edge.tensor_semantics))
    genome.edges.append(SkillEdge(cloned.node_id, edge.dst_node_id, output_key, edge.dst_input_key, edge.tensor_semantics))
    return True


def _current_multi_domain_lane_edge(genome: SkillGenome, tensor_key: str) -> SkillEdge | None:
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
    if tensor_key == "domain_representations":
        return next(
            (
                edge
                for edge in genome.edges
                if edge.dst_node_id == "domain_towers" and edge.dst_input_key == "domain_representations"
            ),
            None,
        )
    if tensor_key == "domain_outputs":
        return next(
            (
                edge
                for edge in genome.edges
                if edge.dst_node_id == "domain_select" and edge.dst_input_key == "domain_outputs"
            ),
            None,
        )
    return None


def _unique_multi_domain_node_id(genome: SkillGenome, prefix: str) -> str:
    node_id = prefix
    idx = 1
    existing = genome.node_ids()
    while node_id in existing:
        node_id = f"{prefix}_{idx}"
        idx += 1
    return node_id


def _unique_multi_domain_key(parent: SkillGenome, prefix: str) -> str:
    key = prefix
    idx = 1
    existing = parent.produced_keys() | set(parent.constraints.required_inputs)
    while key in existing:
        key = f"{prefix}_{idx}"
        idx += 1
    return key


def generate_multi_domain_candidates(
    parents: list[SkillGenome],
    bundle: MultiDomainDatasetBundle,
    base_genome_config: MultiDomainGenomeConfig,
    evolution: CTREvolutionConfig,
    round_idx: int,
    skill_library: SkillLibrary | None = None,
) -> list[CandidateSpec]:
    target = max(0, int(evolution.candidate_budget))
    if target <= 0 or not parents:
        return []
    specs: list[CandidateSpec] = []
    seen: set[str] = set()
    choices, library = _build_unified_multi_domain_skill_choices(
        evolution,
        parents,
        target=target,
        round_idx=round_idx,
        skill_library=skill_library,
    )
    for idx, choice in enumerate(_rank_multi_domain_skill_choices(choices, target=target, round_idx=round_idx)):
        parent, parent_idx = _multi_domain_parent_for_skill_choice(choice, parents, round_idx=round_idx, choice_idx=idx)
        try:
            spec = _candidate_from_multi_domain_skill_choice(
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class MultiDomainModelEvolutionRunner(CTRModelEvolutionRunner):
    """End-to-end multi-domain training and evolution runner."""

    def __init__(self, config: MultiDomainWorkflowConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory = EvolutionMemory(
            config.memory_path or self.output_dir / "evolution_memory.jsonl",
            persist=config.write_evolution_memory,
        )
        self.bundle: MultiDomainDatasetBundle | None = None
        self.trainer: MultiDomainGenomeTrainer | None = None
        self.candidate_parallelism: list[dict[str, Any]] = []
        self.candidate_generation_errors: list[dict[str, Any]] = []
        self.deduplication_records: list[dict[str, Any]] = []
        self.code_space_generation_records: list[dict[str, Any]] = []
        self.completed_results: list[CandidateResult] = []
        self._temporary_staging: tempfile.TemporaryDirectory[str] | None = None
        self._code_space_staging_root = config.evolution.code_space.staging_root
        if self._code_space_staging_root is None and not config.write_metadata_files:
            self._temporary_staging = tempfile.TemporaryDirectory(prefix="evoskillrec_multi_domain_code_space_")
            self._code_space_staging_root = self._temporary_staging.name
        self.training_config_hash = config_fingerprint(asdict(config.training))
        self.genome_config_hash = config_fingerprint(asdict(config.genome))
        self.hparam_fingerprint = config_fingerprint({"training": asdict(config.training), "genome": asdict(config.genome)})
        self.hparam_changes = {
            "training": diff_payload(asdict(CTRTrainingConfig()), asdict(config.training)),
            "genome": diff_payload(asdict(MultiDomainGenomeConfig()), asdict(config.genome)),
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

    def _ensure_bundle_and_trainer(self) -> None:
        if self.bundle is None:
            self.bundle = prepare_multi_domain_data(self.config.dataset, self.config.training)
        if self.trainer is None:
            self.trainer = MultiDomainGenomeTrainer(self.config.training, self.bundle.domain_num, self.bundle.domain_names)

    def _load_or_build_genome(self, bundle: MultiDomainDatasetBundle) -> SkillGenome:
        if self.config.genome.genome_path:
            return SkillGenome.load(self.config.genome.genome_path)
        return build_multi_domain_genome(bundle, self.config.genome)

    def _run(self) -> dict[str, Any]:
        started = time.time()
        self._ensure_bundle_and_trainer()
        assert self.bundle is not None
        if self.config.write_metadata_files:
            self._write_json("dataset_metadata.json", self.bundle.metadata)
        baseline = self._load_or_build_genome(self.bundle)
        GenomeVerifier().assert_valid(baseline)
        if self.config.write_metadata_files:
            baseline.save(self.output_dir / "baseline_genome.json")
        baseline_fingerprint = _safe_architecture_fingerprint(baseline, fallback="baseline")
        self.architecture_index[baseline_fingerprint] = "baseline"

        results = [self._train_candidate("baseline", baseline, None, "baseline", "baseline", "Initial multi-domain genome")]
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
                    "skipped_candidate_ids": [result.candidate_id for result in skipped_results],
                }
            )
            round_results = self._train_round_candidates(unique_candidates, round_idx=round_idx)
            result_by_id = {r.candidate_id: r for r in [*skipped_results, *round_results]}
            results.extend([result_by_id[spec.candidate_id] for spec in candidates if spec.candidate_id in result_by_id])
            survivors = _select_survivors(round_results, self.config.training, self.config.evolution)
            survivor_ids = {result.candidate_id for result in survivors}
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
            "domain_names": list(self.bundle.domain_names),
            "domain_col": self.config.dataset.domain_col,
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

    def _code_space_enabled(self) -> bool:
        provider = str(self.config.evolution.code_space.provider).lower().replace("-", "_")
        return provider not in {"fallback", "disabled", "none"}

    def _generate_round_candidates(self, parent_population: list[SkillGenome], round_idx: int) -> list[CandidateSpec]:
        if not self._code_space_enabled():
            skill_target, code_target = int(self.config.evolution.candidate_budget), 0
        else:
            skill_target, code_target = self._candidate_targets_for_round(round_idx)
        skill_evolution = replace(self.config.evolution, candidate_budget=skill_target)
        specs = (
            generate_multi_domain_candidates(parent_population, self.bundle, self.config.genome, skill_evolution, round_idx=round_idx)
            if skill_target > 0 and self.bundle is not None
            else []
        )
        if code_target > 0:
            try:
                code_specs = self._generate_open_ended_code_candidates(parent_population, round_idx=round_idx, target_code=code_target)
            except Exception as exc:
                self._record_candidate_generation_error(round_idx, "multi_domain_code_candidate_generation", exc, target_code=code_target)
                code_specs = []
            specs.extend(code_specs)
            if len(code_specs) < code_target:
                missing = code_target - len(code_specs)
                self._record_insufficient_code_space_candidates(round_idx, code_target, len(code_specs))
                specs.extend(
                    self._generate_skill_space_supplement(
                        parent_population,
                        round_idx=round_idx,
                        count=missing,
                        reason="multi_domain_code_space_insufficient",
                        skip_candidate_ids={spec.candidate_id for spec in specs},
                    )
                )
        seen_ids: set[str] = set()
        for spec in specs:
            spec.candidate_id = _dedupe_candidate_id(spec.candidate_id, seen_ids)
        return specs[: self.config.evolution.candidate_budget]

    def _memory_task_type(self) -> str:
        return MULTI_DOMAIN_TASK_TAG

    def _generate_skill_space_supplement(
        self,
        parent_population: list[SkillGenome],
        *,
        round_idx: int,
        count: int,
        reason: str,
        skip_candidate_ids: set[str] | None = None,
    ) -> list[CandidateSpec]:
        if count <= 0 or self.bundle is None:
            return []
        skip_candidate_ids = set(skip_candidate_ids or set())
        supplement_evolution = replace(
            self.config.evolution,
            candidate_budget=max(self.config.evolution.candidate_budget, count + len(skip_candidate_ids)),
            code_space_probability=0.0,
        )
        try:
            pool = generate_multi_domain_candidates(
                parent_population,
                self.bundle,
                self.config.genome,
                supplement_evolution,
                round_idx=round_idx,
            )
        except Exception as exc:
            self._record_candidate_generation_error(
                round_idx,
                "multi_domain_skill_space_supplement_generation",
                exc,
                target_skill=count,
                reason=reason,
            )
            return []
        supplement = [spec for spec in pool if spec.candidate_id not in skip_candidate_ids][:count]
        for idx, spec in enumerate(supplement):
            spec.candidate_id = f"{spec.candidate_id}_supplement{idx}"
            spec.rationale = f"{spec.rationale} (skill-space supplement: {reason})"
        return supplement

    def _generate_open_ended_code_candidates(
        self, parents: list[SkillGenome], round_idx: int, target_code: int
    ) -> list[CandidateSpec]:
        if target_code <= 0 or not parents:
            return []
        provider = build_multi_domain_code_space_provider(self.config.evolution.code_space)
        staging_root = self._code_space_staging_root or str(self.output_dir / "generated_skill_staging")
        specs: list[CandidateSpec] = []
        errors: list[dict[str, Any]] = []
        proposal_reports: list[dict[str, Any]] = []
        allocations = _allocate_code_budget_by_parent(parents, budget=target_code, round_idx=round_idx)
        for parent_idx, parent, parent_budget in allocations:
            diagnosis_report = {
                "failure_modes": list(self.config.evolution.failure_modes),
                "task_family": "multi_domain_recommendation",
                "round_idx": round_idx,
                "parent_index": parent_idx,
                "parent_genome_id": parent.metadata.genome_id,
                "parent_budget": parent_budget,
                "objective": {
                    "selection_metric": "validation AUC",
                    "training_objective_metric": self.config.training.objective_metric,
                    "domain_names": list(self.bundle.domain_names if self.bundle is not None else []),
                },
            }
            try:
                proposals = provider.propose(
                    parent=parent,
                    diagnosis_report=diagnosis_report,
                    evolution_memory=self.memory,
                    budget=parent_budget,
                    round_idx=round_idx,
                )
                ingestions = inspect_open_ended_proposal_ingestions(
                    proposals=proposals,
                    parent=parent,
                    output_dir=self.output_dir,
                    memory=self.memory,
                    staging_root=staging_root,
                )
            except Exception as exc:
                errors.append(
                    {
                        "parent_index": parent_idx,
                        "parent_genome_id": parent.metadata.genome_id,
                        "stage": "multi_domain_code_space",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
                continue
            proposal_reports.append(
                {
                    "parent_index": parent_idx,
                    "parent_genome_id": parent.metadata.genome_id,
                    "requested": parent_budget,
                    "received": len(proposals),
                    "proposal_ids": [proposal.proposal_id for proposal in proposals],
                }
            )
            for item in ingestions:
                diagnostic = item.diagnostic()
                if not diagnostic.get("success"):
                    errors.append(
                        {
                            "parent_index": parent_idx,
                            "parent_genome_id": parent.metadata.genome_id,
                            "proposal_id": diagnostic.get("proposal_id", ""),
                            "skill_id": diagnostic.get("skill_id", ""),
                            "stage": "ingest_multi_domain_open_ended_proposals",
                            "error": diagnostic.get("message", "ingestion failed"),
                            "validation_results": diagnostic.get("validation_results", {}),
                        }
                    )
                    continue
                spec = self._candidate_spec_from_code_result(item, parent=parent, parent_idx=parent_idx, round_idx=round_idx, retained_parent_count=len(parents), errors=errors)
                if spec is not None:
                    specs.append(spec)
                if len(specs) >= target_code:
                    break
            if len(specs) >= target_code:
                break
        diagnostic_payload = {
            "provider": self.config.evolution.code_space.provider,
            "provider_diagnostics": _safe_code_space_provider_diagnostics(self.config.evolution.code_space),
            "requested": target_code,
            "generated": len(specs),
            "parent_proposals": proposal_reports,
            "errors": errors,
        }
        self.code_space_generation_records.append(
            {"round_idx": round_idx, "requested": target_code, "generated": len(specs), "errors": len(errors)}
        )
        if self.config.write_code_space_diagnostics or len(specs) < target_code or errors:
            self._write_json(f"round{round_idx}_multi_domain_code_space_proposals.json", diagnostic_payload)
        return specs[:target_code]

    def _candidate_spec_from_code_result(
        self,
        item: Any,
        *,
        parent: SkillGenome,
        parent_idx: int,
        round_idx: int,
        retained_parent_count: int,
        errors: list[dict[str, Any]],
    ) -> CandidateSpec | None:
        ingestion = item.ingestion
        genome = getattr(ingestion, "genome", None)
        if genome is None or not getattr(ingestion, "success", False):
            return None
        skill_id = ingestion.skill_id or item.proposal.skill_id or item.proposal.proposal_id
        try:
            GenomeVerifier().assert_valid(genome)
        except Exception as exc:
            errors.append(
                {
                    "parent_index": parent_idx,
                    "parent_genome_id": parent.metadata.genome_id,
                    "proposal_id": getattr(item.proposal, "proposal_id", ""),
                    "stage": "build_multi_domain_code_candidate_spec",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
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
            wave_record = {"wave_idx": start_idx // processes, "num_candidates": len(wave), "gpu_ids": wave_gpu_ids}
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
                    wave_results = pool.map(_train_multi_domain_candidate_worker, payloads)
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


def _train_multi_domain_candidate_worker(payload: dict[str, Any]) -> CandidateResult:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    gpu_id = str(payload["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config: MultiDomainWorkflowConfig = payload["config"]
    if _uses_cuda(config.training.device):
        config.training.device = "cuda:0"
    bundle = prepare_multi_domain_data(config.dataset, config.training)
    trainer = MultiDomainGenomeTrainer(config.training, bundle.domain_num, bundle.domain_names)
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
    parser = argparse.ArgumentParser(description="Run multi-domain recommendation-model evolution.")
    parser.add_argument("--config", required=True, help="YAML workflow config.")
    parser.add_argument("--device", help="Override training.device.")
    parser.add_argument("--output-dir", help="Override output_dir.")
    parser.add_argument("--seed", type=int, help="Override training.seed.")
    args = parser.parse_args(argv)
    config = load_multi_domain_workflow_config(args.config)
    if args.device:
        config.training.device = args.device
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.training.seed = args.seed
    summary = MultiDomainModelEvolutionRunner(config).run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
