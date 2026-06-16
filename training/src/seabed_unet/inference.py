"""Checkpoint loading + tile-level prediction shared by evaluate.py and predict.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .unet import UNet


def load_checkpoint(path: Path, device: torch.device) -> tuple[UNet, dict]:
    """Rebuild the model from the config stored inside the checkpoint."""
    ckpt = torch.load(path, map_location=device)
    c = ckpt["config"]
    model = UNet(
        in_channels=len(c["bands"]),
        num_classes=len(c["classes"]),
        base_filters=c["model"]["base_filters"],
        depth=c["model"]["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_probs(
    model: UNet, inputs: np.ndarray, device: torch.device
) -> np.ndarray:
    """(C_in, H, W) normalized inputs -> (num_classes, H, W) softmax probabilities."""
    x = torch.from_numpy(inputs[np.newaxis]).to(device)
    return F.softmax(model(x), dim=1)[0].cpu().numpy()
