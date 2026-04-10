"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        if pred_boxes.ndim != 2 or target_boxes.ndim != 2 or pred_boxes.shape[-1] != 4 or target_boxes.shape[-1] != 4:
            raise ValueError("pred_boxes and target_boxes must be shaped [B, 4]")
        if pred_boxes.shape[0] != target_boxes.shape[0]:
            raise ValueError("pred_boxes and target_boxes must have the same batch size")

        pred_wh = torch.clamp(pred_boxes[:, 2:], min=0.0)
        targ_wh = torch.clamp(target_boxes[:, 2:], min=0.0)

        pred_xy1 = pred_boxes[:, :2] - pred_wh / 2.0
        pred_xy2 = pred_boxes[:, :2] + pred_wh / 2.0
        targ_xy1 = target_boxes[:, :2] - targ_wh / 2.0
        targ_xy2 = target_boxes[:, :2] + targ_wh / 2.0

        inter_xy1 = torch.maximum(pred_xy1, targ_xy1)
        inter_xy2 = torch.minimum(pred_xy2, targ_xy2)
        inter_wh = torch.clamp(inter_xy2 - inter_xy1, min=0.0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        pred_area = pred_wh[:, 0] * pred_wh[:, 1]
        targ_area = targ_wh[:, 0] * targ_wh[:, 1]
        union_area = pred_area + targ_area - inter_area

        iou = inter_area / (union_area + self.eps)
        iou = torch.clamp(iou, min=0.0, max=1.0)
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss