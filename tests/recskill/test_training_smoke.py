from pathlib import Path

import torch

from recskill import build_model_from_genome


def test_deepfm_skillified_training_smoke():
    torch.manual_seed(7)
    genome_path = Path(__file__).resolve().parents[2] / "recskill" / "genomes" / "deepfm_skillified.yaml"
    runtime_params = {
        "vocab_sizes": [20, 30, 40],
        "embedding_dim": 4,
        "flat_input_dim": 12,
        "hidden_dims": [8],
        "fusion_dim": 9,
    }
    model = build_model_from_genome(genome_path, runtime_params=runtime_params)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    saw_gradient = False
    for _ in range(5):
        batch = {
            "sparse_features": torch.stack(
                [
                    torch.randint(0, 20, (16,)),
                    torch.randint(0, 30, (16,)),
                    torch.randint(0, 40, (16,)),
                ],
                dim=1,
            ),
            "labels": torch.randint(0, 2, (16,)).float(),
        }
        optimizer.zero_grad()
        ctx = model(batch)
        loss = ctx["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        saw_gradient = any(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters() if param.requires_grad)
        optimizer.step()

    assert saw_gradient
