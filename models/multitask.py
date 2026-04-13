"""Unified multi-task model
"""

import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .layers import CustomDropout
from .localization import VGG11Localizer
from .segmentation import DoubleConv
from .segmentation import VGG11UNet
from .vgg11 import VGG11

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "checkpoints/classifier.pth", localizer_path: str = "checkpoints/localizer.pth", unet_path: str = "checkpoints/unet.pth"):
        #per assignment instructions :
        import gdown
        gdown.download(id="1Kpe28u6Lw1KHhBT_pe720ZA5iqa-jyMo", output=classifier_path, quiet=False)
        gdown.download(id="12YSVAel-GjHI9O4ou42hO3pLwt0R6Yy3", output=localizer_path, quiet=False)
        gdown.download(id="1AjB-cwlbPl5z03UhpsmQ67wOzlMPQDTo", output=unet_path, quiet=False)
    
        
        
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.

        """
        '''''
        import gdown
        classifier_path = "checkpoints/classifier.pth"
        localizer_path = "checkpoints/localizer.pth"
        unet_path = "checkpoints/unet.pth"
        gdown.download(id="CLASSIFIER_DRIVE_ID", output=classifier_path, quiet=False)
        gdown.download(id="LOCALIZER_DRIVE_ID", output=localizer_path, quiet=False)
        gdown.download(id="UNET_DRIVE_ID", output=unet_path, quiet=False)
        '''
        
        super().__init__()
        self.encoder = VGG11(in_channels=in_channels, num_classes=1000)

        self.classifier_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_breeds),
        )

        self.regressor_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(256, 4),
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
        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

        self._load_pretrained_heads(classifier_path, localizer_path, unet_path, num_breeds, seg_classes, in_channels)

    def _safe_load(self, path: str):
        try:
            ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                return ckpt["state_dict"]
            return ckpt
        except Exception as e:
            print(f"[MultiTaskPerceptionModel] Warning: could not load '{path}': {e}")
            return None

    def _load_pretrained_heads(self, classifier_path: str, localizer_path: str, unet_path: str, num_breeds: int, seg_classes: int, in_channels: int):
        cls_ckpt = self._safe_load(classifier_path)
        if cls_ckpt is not None:
            cls_model = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
            cls_model.load_state_dict(cls_ckpt, strict=False)
            self.encoder.features.load_state_dict(cls_model.vgg.features.state_dict(), strict=False)
            self.classifier_head.load_state_dict(cls_model.vgg.classifier.state_dict(), strict=False)

        loc_ckpt = self._safe_load(localizer_path)
        if loc_ckpt is not None:
            loc_model = VGG11Localizer(in_channels=in_channels)
            loc_model.load_state_dict(loc_ckpt, strict=False)
            self.encoder.features.load_state_dict(loc_model.backbone.state_dict(), strict=False)
            self.regressor_head[0].load_state_dict(loc_model.regressor[2].state_dict())
            self.regressor_head[3].load_state_dict(loc_model.regressor[5].state_dict())
            self.regressor_head[6].load_state_dict(loc_model.regressor[8].state_dict())

        seg_ckpt = self._safe_load(unet_path)
        if seg_ckpt is not None:
            seg_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)
            seg_model.load_state_dict(seg_ckpt, strict=False)
            self.encoder.features.load_state_dict(seg_model.encoder.features.state_dict(), strict=False)
            self.up4.load_state_dict(seg_model.up4.state_dict())
            self.dec4.load_state_dict(seg_model.dec4.state_dict())
            self.up3.load_state_dict(seg_model.up3.state_dict())
            self.dec3.load_state_dict(seg_model.dec3.state_dict())
            self.up2.load_state_dict(seg_model.up2.state_dict())
            self.dec2.load_state_dict(seg_model.dec2.state_dict())
            self.up1.load_state_dict(seg_model.up1.state_dict())
            self.dec1.load_state_dict(seg_model.dec1.state_dict())
            self.up0.load_state_dict(seg_model.up0.state_dict())
            self.dec0.load_state_dict(seg_model.dec0.state_dict())
            self.seg_head.load_state_dict(seg_model.head.state_dict())

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        p1, p2, p3, p4, p5 = self.encoder.forward_features(x)

        pooled = torch.nn.functional.adaptive_avg_pool2d(p5, (7, 7))
        flat = torch.flatten(pooled, 1)
        class_logits = self.classifier_head(flat)
        bbox = self.regressor_head(flat)

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
        seg_mask = self.seg_head(d0)

        return {
            'classification': class_logits,
            'localization': bbox,
            'segmentation': seg_mask
        }
