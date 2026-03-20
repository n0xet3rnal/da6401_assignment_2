"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement dropout.
        raise NotImplementedError("Implement CustomDropout.forward")
