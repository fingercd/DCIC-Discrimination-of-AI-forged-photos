# Plan B: Segmentation model for forgery region (binary mask). Supports UNet and DeepLabV3Plus.
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

CONFIG = {
    "architecture": "deeplabv3plus",
    "encoder_name": "resnet50",
    "encoder_weights": "imagenet",
    "in_channels": 3,
}


def build_unet(
    encoder_name: str | None = None,
    encoder_weights: str | None = None,
    in_channels: int = 3,
    architecture: str | None = None,
) -> nn.Module:
    """Build segmentation model. architecture: 'unet' | 'deeplabv3plus'. Output single-channel logits."""
    if smp is None:
        raise ImportError("segmentation_models_pytorch is required: pip install segmentation-models-pytorch")
    arch = (architecture or CONFIG["architecture"]).lower()
    enc = encoder_name or CONFIG["encoder_name"]
    w = encoder_weights or CONFIG["encoder_weights"]
    common = {"encoder_name": enc, "encoder_weights": w, "in_channels": in_channels, "classes": 1}
    if arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(**common)
    else:
        model = smp.Unet(**common)
    return model


def load_unet_checkpoint(model: nn.Module, path: str | None) -> nn.Module:
    if path:
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    return model
