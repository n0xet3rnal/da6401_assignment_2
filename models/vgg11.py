"""VGG11 backbone and classifier."""

import torch
import torch.nn as nn

from .layers import CustomDropout


class VGG11(nn.Module):
    """VGG11 (config A) with optional BatchNorm and CustomDropout."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

        layers = []
        curr_in = in_channels
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(curr_in, v, kernel_size=3, padding=1))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                curr_in = v

        self.features = nn.Sequential(*layers)

        # Dropout is placed before the two largest FC layers to regularize dense capacity.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward_features(self, x: torch.Tensor):
        """Return feature maps after each maxpool stage (5 maps)."""
        pool_features = []
        out = x
        for layer in self.features:
            out = layer(out)
            if isinstance(layer, nn.MaxPool2d):
                pool_features.append(out)
        return pool_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG11Encoder(VGG11):
    """Backward-compatible alias for existing skeleton naming."""

    def __init__(self, in_channels: int = 3, use_batchnorm: bool = True, dropout_p: float = 0.5):
        super().__init__(in_channels=in_channels, num_classes=1000, use_batchnorm=use_batchnorm, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        pool_features = self.forward_features(x)
        bottleneck = pool_features[-1]
        if return_features:
            feature_dict = {
                "pool1": pool_features[0],
                "pool2": pool_features[1],
                "pool3": pool_features[2],
                "pool4": pool_features[3],
                "pool5": pool_features[4],
            }
            return bottleneck, feature_dict
        return bottleneck
