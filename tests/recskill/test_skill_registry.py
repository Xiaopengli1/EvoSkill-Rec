import recskill


def test_generated_skills_are_registered():
    expected = {
        "bce_loss",
        "binary_ctr_head",
        "candidate_dot_logits",
        "cen_feature_gate",
        "concat_fusion",
        "crossnet_mix",
        "crossnet_v1",
        "dien_interest_evolution",
        "din_target_attention",
        "edcn_bridge_stack",
        "field_embedding",
        "field_sum",
        "flatten_field_embeddings",
        "fm_interaction",
        "inbatch_sampled_logits",
        "mlp_tower",
        "mmoe_gate",
        "multi_interest_fusion",
        "narm_session_encoder",
        "normalize_embedding",
        "sasrec_pairwise_logits",
        "sasrec_sequence_encoder",
        "sequence_field_embedding",
        "shared_sequence_embedding",
        "shared_mlp_tower",
        "sigmoid_prediction",
        "sine_interest_encoder",
        "stamp_session_encoder",
        "target_append_sequence",
        "task_replication",
        "task_specific_towers",
        "task_tower",
    }

    assert expected.issubset(set(recskill.list_skills()))


def test_build_skill_from_registry():
    skill = recskill.build_skill("fm_interaction")

    assert skill.__class__.__name__ == "FMInteractionSkill"
