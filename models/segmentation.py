"""Segmentation model
"""

import torch
import torch.nn as nn

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


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


class DiceLoss(nn.Module):
    """Multiclass Dice loss over softmax probabilities."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_oh = _one_hot(target, num_classes)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_oh, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(target_oh, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice for stable pixelwise learning and overlap optimization."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        target_oh = _one_hot(target, num_classes)
        return self.bce_weight * self.bce(logits, target_oh) + self.dice_weight * self.dice(logits, target)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
        pretrained_vgg_path: str = None,
        freeze_backbone: str = "none",
    ):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        self.num_classes = num_classes
        self.encoder = VGG11(
            in_channels=in_channels,
            num_classes=1000,
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
        )

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec0 = DoubleConv(64, 64)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        # BCE handles per-pixel class supervision while Dice improves mask overlap quality.
        self.loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

        if pretrained_vgg_path is not None:
            ckpt = torch.load(pretrained_vgg_path, map_location="cpu")
            state_dict = _state_dict_from_checkpoint(ckpt)
            _load_flexible(self.encoder, state_dict)

        freeze_mode = freeze_backbone
        if isinstance(freeze_mode, bool):
            freeze_mode = "all" if freeze_mode else "none"
        if freeze_mode not in {"none", "all", "partial"}:
            raise ValueError("freeze_backbone must be one of {'none', 'all', 'partial'}")

        if freeze_mode == "all":
            for p in self.encoder.features.parameters():
                p.requires_grad = False
        elif freeze_mode == "partial":
            pool_count = 0
            for layer in self.encoder.features:
                for p in layer.parameters():
                    p.requires_grad = False
                if isinstance(layer, nn.MaxPool2d):
                    pool_count += 1
                    if pool_count >= 3:
                        break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        p1, p2, p3, p4, p5 = self.encoder.forward_features(x)

        d4 = self.up4(p5)
        d4 = self.dec4(torch.cat([d4, p4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, p3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, p2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, p1], dim=1))

        d0 = self.up0(d1)
        d0 = self.dec0(d0)

        return self.head(d0)
