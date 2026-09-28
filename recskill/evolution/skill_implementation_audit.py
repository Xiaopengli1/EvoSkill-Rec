from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path

import recskill.skills as _registered_skills  # noqa: F401

from recskill.core.registry import SKILL_REGISTRY

from .skill_library import SkillLibrary


MODEL_DOMAINS = ("ranking", "matching", "multi_task", "generative")


REQUIRED_MODEL_SKILLS: dict[str, list[str]] = {
    "DeepFM": ["named_feature_dict_adapter", "field_embedding", "flatten_field_embeddings", "linear_logit", "fm_interaction", "mlp_tower", "additive_fusion", "sigmoid_prediction", "bce_loss"],
    "WideDeep": ["named_feature_dict_adapter", "dense_feature_path", "field_embedding", "flatten_field_embeddings", "linear_logit", "mlp_tower", "additive_fusion", "sigmoid_prediction", "bce_loss"],
    "DCN": ["named_feature_dict_adapter", "dense_feature_path", "field_embedding", "flatten_field_embeddings", "crossnet_v1", "mlp_tower", "concat_fusion", "linear_logit", "sigmoid_prediction", "bce_loss"],
    "DCNv2": ["field_embedding", "flatten_field_embeddings", "crossnet_v2", "crossnet_mix", "mlp_tower", "concat_fusion", "linear_logit", "sigmoid_prediction"],
    "EDCN": ["field_embedding", "flatten_field_embeddings", "crossnet_v1", "edcn_bridge_stack", "mlp_tower", "linear_logit", "sigmoid_prediction"],
    "AFM": ["field_embedding", "linear_logit", "fm_interaction", "afm_attention_pooling", "sigmoid_prediction"],
    "AutoInt": ["field_embedding", "dense_feature_path", "flatten_field_embeddings", "autoint_attention", "linear_logit", "mlp_tower", "sigmoid_prediction"],
    "FiBiNet": ["field_embedding", "senet_feature_gate", "bilinear_interaction", "flatten_field_embeddings", "mlp_tower", "sigmoid_prediction"],
    "DeepFFM": ["field_embedding", "field_aware_embedding_adapter", "field_sum", "ffm_interaction", "flatten_field_embeddings", "mlp_tower", "additive_fusion", "sigmoid_prediction"],
    "FatDeepFFM": ["field_embedding", "field_aware_embedding_adapter", "field_sum", "ffm_interaction", "cen_feature_gate", "mlp_tower", "additive_fusion", "sigmoid_prediction"],
    "DIN": ["field_embedding", "sequence_field_embedding", "sequence_mask", "din_target_attention", "flatten_field_embeddings", "concat_fusion", "mlp_tower", "sigmoid_prediction"],
    "DIEN": ["field_embedding", "sequence_field_embedding", "sequence_mask", "gru_sequence_encoder", "full_augru_evolution", "dien_auxiliary_interest_loss", "flatten_field_embeddings", "concat_fusion", "mlp_tower", "sigmoid_prediction"],
    "BST": ["field_embedding", "sequence_field_embedding", "target_append_sequence", "exact_leakyrelu_transformer_encoder", "sequence_pooling", "flatten_field_embeddings", "concat_fusion", "mlp_tower", "sigmoid_prediction"],
    "DSSM": ["named_feature_dict_adapter", "field_embedding", "flatten_field_embeddings", "mlp_tower", "normalize_embedding", "dot_product_logits", "mode_user_item_adapter", "sigmoid_prediction"],
    "DSSM_SENET": ["named_feature_dict_adapter", "field_embedding", "senet_feature_gate", "flatten_field_embeddings", "mlp_tower", "normalize_embedding", "dot_product_logits", "mode_user_item_adapter", "sigmoid_prediction"],
    "FaceBookDSSM": ["named_feature_dict_adapter", "field_embedding", "flatten_field_embeddings", "mlp_tower", "shared_mlp_tower", "normalize_embedding", "dot_product_logits", "mode_user_item_adapter", "hinge_loss"],
    "YoutubeDNN": ["named_feature_dict_adapter", "field_embedding", "flatten_field_embeddings", "sequence_pooling", "mlp_tower", "normalize_embedding", "candidate_dot_logits", "mode_user_item_adapter", "sampled_softmax_loss"],
    "YoutubeSBC": ["named_feature_dict_adapter", "field_embedding", "flatten_field_embeddings", "mlp_tower", "inbatch_sampled_logits", "in_batch_negative_head", "in_batch_nce_loss", "mode_user_item_adapter"],
    "GRU4Rec": ["named_feature_dict_adapter", "field_embedding", "sequence_field_embedding", "gru_sequence_encoder", "mlp_tower", "normalize_embedding", "candidate_dot_logits", "mode_user_item_adapter", "sampled_softmax_loss"],
    "NARM": ["shared_sequence_embedding", "narm_session_encoder", "candidate_dot_logits", "all_item_scoring_adapter", "mode_user_item_adapter"],
    "SASRec": ["shared_sequence_embedding", "sasrec_sequence_encoder", "sasrec_pairwise_logits", "in_batch_negative_head", "mode_user_item_adapter"],
    "MIND": ["field_embedding", "sequence_field_embedding", "multi_interest_extractor", "multi_interest_fusion", "normalize_embedding", "candidate_dot_logits", "mode_user_item_adapter", "sampled_softmax_loss"],
    "ComirecSA": ["field_embedding", "sequence_field_embedding", "multi_interest_extractor", "multi_interest_fusion", "normalize_embedding", "candidate_dot_logits", "mode_user_item_adapter", "sampled_softmax_loss"],
    "ComirecDR": ["field_embedding", "sequence_field_embedding", "multi_interest_extractor", "multi_interest_fusion", "normalize_embedding", "candidate_dot_logits", "mode_user_item_adapter", "sampled_softmax_loss"],
    "SINE": ["shared_sequence_embedding", "sine_interest_encoder", "candidate_dot_logits", "covariance_regularizer", "mode_user_item_adapter", "sampled_softmax_loss"],
    "STAMP": ["shared_sequence_embedding", "stamp_session_encoder", "candidate_dot_logits", "all_item_scoring_adapter", "mode_user_item_adapter"],
    "SharedBottom": ["named_feature_dict_adapter", "field_embedding", "flatten_field_embeddings", "shared_bottom_tower", "task_replication", "task_tower", "multitask_loss"],
    "MMOE": ["named_feature_dict_adapter", "field_embedding", "flatten_field_embeddings", "mmoe_gate", "task_tower", "multitask_loss"],
    "PLE": ["named_feature_dict_adapter", "field_embedding", "flatten_field_embeddings", "multi_level_cgc", "task_tower", "multitask_loss"],
    "ESMM": ["named_feature_dict_adapter", "field_embedding", "flatten_field_embeddings", "mlp_tower", "esmm_chain_head", "multitask_loss"],
    "AITM": ["named_feature_dict_adapter", "field_embedding", "flatten_field_embeddings", "task_specific_towers", "exact_attention_transfer", "task_tower", "multitask_loss"],
    "HSTUModel": ["shared_sequence_embedding", "time_bucket_embedding", "hstu_sequence_encoder", "tied_embedding_lm_head", "sequence_cross_entropy_loss"],
    "HLLMModel": ["precomputed_item_embedding_adapter", "time_bucket_embedding", "relative_position_bias", "hllm_transformer_block", "normalized_item_dot_head", "sequence_cross_entropy_loss"],
    "RQVAEModel": ["mlp_tower", "exact_residual_quantizer_stack", "kmeans_sinkhorn_initialization", "semantic_id_export_adapter"],
    "TIGERModel": ["exact_t5_encoder_decoder", "ranking_loss_temperature", "trie_constrained_decoding"],
}


@dataclass(frozen=True)
class SkillImplementationAuditIssue:
    severity: str
    model_name: str
    issue: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def audit_torch_rechub_skill_implementations(repo_root: str | Path | None = None) -> list[SkillImplementationAuditIssue]:
    """Audit model ability coverage using only recskill/skills implementations."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    library = SkillLibrary.from_repo(root, include_generated=False)
    exported = {name for names in _exported_models(root).values() for name in names}
    issues: list[SkillImplementationAuditIssue] = []

    for model_name in sorted(exported):
        if model_name not in REQUIRED_MODEL_SKILLS:
            issues.append(SkillImplementationAuditIssue("error", model_name, "missing_required_skill_map", "exported model has no skill implementation map"))

    for model_name in sorted(set(REQUIRED_MODEL_SKILLS) - exported):
        if model_name != "DSSM_SENET":
            issues.append(SkillImplementationAuditIssue("warning", model_name, "skill_map_for_non_exported_model", "model is mapped but not exported by torch_rechub.models.__all__"))

    for model_name, skill_ids in sorted(REQUIRED_MODEL_SKILLS.items()):
        for skill_id in skill_ids:
            if not library.has(skill_id):
                issues.append(SkillImplementationAuditIssue("error", model_name, "missing_skill_manifest", skill_id))
            if skill_id not in SKILL_REGISTRY:
                issues.append(SkillImplementationAuditIssue("error", model_name, "missing_runtime_skill", skill_id))

    return issues


def assert_torch_rechub_skill_implementations(repo_root: str | Path | None = None) -> None:
    errors = [issue for issue in audit_torch_rechub_skill_implementations(repo_root) if issue.severity == "error"]
    if errors:
        rendered = "\n".join(f"{issue.model_name}: {issue.issue} ({issue.detail})" for issue in errors)
        raise AssertionError(rendered)


def _exported_models(root: Path) -> dict[str, list[str]]:
    return {domain: _read_dunder_all(root / "torch_rechub" / "models" / domain / "__init__.py") for domain in MODEL_DOMAINS}


def _read_dunder_all(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            return [str(item) for item in ast.literal_eval(node.value)]
    return []
