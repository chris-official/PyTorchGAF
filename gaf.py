import torch
from torch import nn
from typing import Literal


class GAFTransform(nn.Module):
    """Transforms a batch of times eries data to a Gramian Angular Field (GAF)"""
    def __init__(self, method: Literal["summation", "difference"] = "summation", eps: float = 1e-6):
        super().__init__()
        assert method in ["summation", "difference"], "Method must be either 'summation' or 'difference'"
        self.method = method
        self.eps = eps

    def forward(self, x):
        """
        Expects x to be of shape (batch_size, channels, seq_len)
        Returns image tensor of shape (batch_size, channels, seq_len, seq_len)
        """
        x_cos = self.min_max_scale(x)  # Min-Max scaling to range [-1, 1]

        # Calculate GAF
        x_sin = (1 - x_cos ** 2) ** 0.5
        if self.method == "summation":
            gaf = torch.einsum("bci,bcj->bcij", x_cos, x_cos) - torch.einsum("bci,bcj->bcij", x_sin, x_sin)
        else:
            gaf = torch.einsum("bci,bcj->bcij", x_sin, x_cos) - torch.einsum("bci,bcj->bcij", x_cos, x_sin)

        gaf = gaf / 2 + 0.5  # Scale images to range [0, 1]
        return gaf

    def min_max_scale(self, x):
        """Min-Max scaling each sequence to range [-1, 1]"""
        mins = x.min(dim=-1, keepdim=True).values
        maxs = x.max(dim=-1, keepdim=True).values
        return 2 * (x - mins) / (maxs - mins + self.eps) - 1
      
