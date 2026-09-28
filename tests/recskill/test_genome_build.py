from pathlib import Path

import torch

from recskill import build_model_from_genome, validate_model_forward


def test_build_deepfm_genome_and_validate_forward():
    genome_path = Path(__file__).resolve().parents[2] / "recskill" / "genomes" / "deepfm_skillified.yaml"
    runtime_params = {
        "vocab_sizes": [10, 12, 14],
        "embedding_dim": 4,
        "flat_input_dim": 12,
        "hidden_dims": [8],
        "fusion_dim": 9,
    }
    model = build_model_from_genome(genome_path, runtime_params=runtime_params)
    dummy_batch = {
        "sparse_features": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]], dtype=torch.long),
        "labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }

    ok, message = validate_model_forward(model, dummy_batch, required_outputs=["logits", "loss"])

    assert ok, message
