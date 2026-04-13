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
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),      # 0
            nn.BatchNorm2d(64),                  # 1
            nn.ReLU(inplace=True),               # 2
            nn.MaxPool2d(2, 2),                  # 3
            nn.Conv2d(64, 128, 3, padding=1),    # 4
            nn.BatchNorm2d(128),                 # 5
            nn.ReLU(inplace=True),               # 6
            nn.MaxPool2d(2, 2),                  # 7
            nn.Conv2d(128, 256, 3, padding=1),   # 8
            nn.BatchNorm2d(256),                 # 9
            nn.ReLU(inplace=True),               # 10
            nn.Conv2d(256, 256, 3, padding=1),   # 11
            nn.BatchNorm2d(256),                 # 12
            nn.ReLU(inplace=True),               # 13
            nn.MaxPool2d(2, 2),                  # 14
            nn.Conv2d(256, 512, 3, padding=1),   # 15
            nn.BatchNorm2d(512),                 # 16
            nn.ReLU(inplace=True),               # 17
            nn.Conv2d(512, 512, 3, padding=1),   # 18
            nn.BatchNorm2d(512),                 # 19
            nn.ReLU(inplace=True),               # 20
            nn.MaxPool2d(2, 2),                  # 21
            nn.Conv2d(512, 512, 3, padding=1),   # 22
            nn.BatchNorm2d(512),                 # 23
            nn.ReLU(inplace=True),               # 24
            nn.Conv2d(512, 512, 3, padding=1),   # 25
            nn.BatchNorm2d(512),                 # 26
            nn.ReLU(inplace=True),               # 27
            nn.MaxPool2d(2, 2),                  # 28
        )

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
