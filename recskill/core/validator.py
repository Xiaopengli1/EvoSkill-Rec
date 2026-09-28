from __future__ import annotations

import torch


def validate_model_forward(model, dummy_batch, required_outputs=None) -> tuple[bool, str]:
    """Run a guarded forward pass and check required context outputs."""
    required_outputs = required_outputs or []
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)
        missing = [key for key in required_outputs if key not in output]
        if missing:
            return False, f"Missing required outputs: {missing}"
        return True, "forward ok"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    finally:
        if was_training:
            model.train()
