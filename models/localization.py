"""Localization modules
"""

import torch
import torch.nn as nn

from .layers import CustomDropout
from .vgg11 import VGG11


def _state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _load_flexible(module: nn.Module, state_dict: dict):
    try:
        module.load_state_dict(state_dict, strict=False)
        return
    except RuntimeError:
        pass

    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module.") :]
        if new_k.startswith("vgg."):
            new_k = new_k[len("vgg.") :]
        cleaned[new_k] = v
    module.load_state_dict(cleaned, strict=False)

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(
        self,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
        pretrained_vgg_path: str = None,
        freeze_backbone: str = "none",
    ):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.vgg = VGG11(
            in_channels=in_channels,
            num_classes=1000,
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
        )
        self.backbone = self.vgg.features

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(256, 4),
        )

        if pretrained_vgg_path is not None:
            ckpt = torch.load(pretrained_vgg_path, map_location="cpu")
            state_dict = _state_dict_from_checkpoint(ckpt)
            _load_flexible(self.vgg, state_dict)

        freeze_mode = freeze_backbone
        if isinstance(freeze_mode, bool):
            freeze_mode = "all" if freeze_mode else "none"
        if freeze_mode not in {"none", "all", "partial"}:
            raise ValueError("freeze_backbone must be one of {'none', 'all', 'partial'}")

        if freeze_mode == "all":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif freeze_mode == "partial":
            pool_count = 0
            for layer in self.backbone:
                for p in layer.parameters():
                    p.requires_grad = False
                if isinstance(layer, nn.MaxPool2d):
                    pool_count += 1
                    if pool_count >= 3:
                        break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        feats = self.backbone(x)
        return self.regressor(feats)
