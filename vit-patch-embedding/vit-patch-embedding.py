import numpy as np
import torch
from torch import nn

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> torch.Tensor:
    """
    Convert image to patch embeddings using Conv2d projection.

    Args:
        image: (B, H, W, C) numpy array
        patch_size: P
        embed_dim: D

    Returns:
        embeddings: (B, N, D)
    """
    B, H, W, C = image.shape

    # Convert to torch tensor and change layout to (B, C, H, W)
    x = torch.from_numpy(image).permute(0, 3, 1, 2).float()

    # Conv2d projection (equivalent to linear projection of patches)
    conv = nn.Conv2d(
        in_channels=C,
        out_channels=embed_dim,
        kernel_size=patch_size,
        stride=patch_size
    )

    # Apply convolution
    x = conv(x)  # (B, D, H/P, W/P)

    # Flatten patches
    x = x.flatten(2)       # (B, D, N)
    x = x.transpose(1, 2)  # (B, N, D)

    return x
