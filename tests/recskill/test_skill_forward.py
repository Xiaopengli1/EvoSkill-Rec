import torch

from recskill import SkillContext, build_skill


def test_required_skill_forward_shapes():
    batch_size = 4
    vocab_sizes = [11, 13, 17]
    embedding_dim = 5
    sparse_features = torch.stack(
        [
            torch.randint(0, vocab_sizes[0], (batch_size,)),
            torch.randint(0, vocab_sizes[1], (batch_size,)),
            torch.randint(0, vocab_sizes[2], (batch_size,)),
        ],
        dim=1,
    )
    ctx = SkillContext({"sparse_features": sparse_features, "labels": torch.randint(0, 2, (batch_size,)).float()})

    ctx = build_skill("field_embedding", vocab_sizes=vocab_sizes, embedding_dim=embedding_dim)(ctx)
    assert ctx["field_embeddings"].shape == (batch_size, len(vocab_sizes), embedding_dim)

    ctx = build_skill("flatten_field_embeddings")(ctx)
    assert ctx["flat_embeddings"].shape == (batch_size, len(vocab_sizes) * embedding_dim)

    ctx = build_skill("fm_interaction")(ctx)
    assert ctx["fm_output"].shape == (batch_size, 1)

    ctx = build_skill("mlp_tower", input_key="flat_embeddings", output_key="deep_output", input_dim=15, hidden_dims=[8], dropout=0.0)(ctx)
    assert ctx["deep_output"].shape == (batch_size, 8)

    ctx = build_skill("concat_fusion", input_keys=["fm_output", "deep_output"], output_key="fused")(ctx)
    assert ctx["fused"].shape == (batch_size, 9)

    ctx = build_skill("binary_ctr_head", input_key="fused", input_dim=9)(ctx)
    assert ctx["logits"].shape == (batch_size, 1)

    ctx = build_skill("bce_loss")(ctx)
    assert ctx["loss"].dim() == 0
    assert torch.isfinite(ctx["loss"])


def test_crossnet_v1_forward_shape():
    ctx = SkillContext({"flat_embeddings": torch.randn(4, 12)})

    ctx = build_skill("crossnet_v1", input_dim=12, num_layers=2)(ctx)

    assert ctx["cross_output"].shape == (4, 12)


def test_exact_torch_rechub_module_skills_forward_shapes():
    ctx = SkillContext({"sequence_ids": torch.tensor([[1, 2, 0], [3, 0, 0]])})
    ctx = build_skill("sequence_mask")(ctx)
    assert ctx["sequence_mask"].shape == (2, 3)

    ctx = SkillContext(
        {
            "interest_sequence": torch.randn(2, 1, 3, 4),
            "target_embeddings": torch.randn(2, 1, 4),
            "sequence_mask": torch.tensor([[[1, 1, 0]], [[1, 0, 0]]], dtype=torch.bool),
        }
    )
    ctx = build_skill("full_augru_evolution", embedding_dim=4, num_fields=1)(ctx)
    assert ctx["interest_evolving"].shape == (2, 1, 4)

    ctx = SkillContext({"shared_input": torch.randn(2, 3)})
    ctx = build_skill(
        "multi_level_cgc",
        input_dim=3,
        n_task=2,
        n_level=2,
        n_expert_specific=1,
        n_expert_shared=1,
        expert_params={"dims": [4], "activation": "relu", "dropout": 0.0},
    )(ctx)
    assert ctx["task_representations"].shape == (2, 2, 4)

    ctx = SkillContext({"task_representations": torch.randn(2, 2, 4)})
    ctx = build_skill("exact_attention_transfer", input_dim=4, n_task=2)(ctx)
    assert ctx["task_representations"].shape == (2, 2, 4)

    ctx = SkillContext({"latent": torch.randn(2, 3)})
    ctx = build_skill("exact_residual_quantizer_stack", n_e_list=[4, 4], e_dim=3, sk_epsilons=[0.0, 0.0], use_sk=False)(ctx)
    assert ctx["quantized"].shape == (2, 3)
    assert ctx["code_indices"].shape == (2, 2)

    ctx = SkillContext({"seq_tokens": torch.tensor([[1, 2, 0]])})
    ctx = build_skill("precomputed_item_embedding_adapter", item_embeddings=torch.randn(5, 4))(ctx)
    assert ctx["sequence_embeddings"].shape == (1, 3, 4)

    ctx = SkillContext({"sequence_output": torch.randn(1, 3, 4), "item_embeddings": torch.randn(5, 4)})
    ctx = build_skill("normalized_item_dot_head")(ctx)
    assert ctx["logits"].shape == (1, 3, 5)

    ctx = SkillContext({"sequence_output": torch.randn(1, 3, 4)})
    ctx = build_skill("tied_embedding_lm_head", vocab_size=5, d_model=4, embedding_weight_key=None)(ctx)
    assert ctx["logits"].shape == (1, 3, 5)

    ctx = SkillContext({"concept_embeddings": torch.randn(6, 4)})
    ctx = build_skill("covariance_regularizer")(ctx)
    assert ctx["covariance_regularizer"].dim() == 0
