"""Visualization utilities for image display and debugging.

This module provides utilities for denormalizing and displaying images
that have been normalized for model training.
"""

import numpy as np
import torch


def denormalize(
    tensor: torch.Tensor,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """Denormalize a tensor image with given mean and std.

    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W]
        mean: Mean used for normalization (ImageNet default)
        std: Std used for normalization (ImageNet default)

    Returns:
        Denormalized tensor in range [0, 1]
    """
    # Clone to avoid modifying original
    tensor = tensor.clone()

    # Convert mean and std to tensors
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)

    # Handle batch dimension
    if tensor.ndim == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    # Denormalize: x_denorm = x_norm * std + mean
    tensor = tensor * std + mean

    # Clamp to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)

    return tensor


def tensor_to_image(
    tensor: torch.Tensor,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """Convert a normalized tensor to a displayable numpy image.

    Args:
        tensor: Image tensor [C, H, W]
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Numpy array [H, W, C] in range [0, 1]
    """
    # Denormalize
    tensor = denormalize(tensor, mean=mean, std=std)

    # Move to CPU and convert to numpy
    image = tensor.cpu().numpy()

    # Transpose from [C, H, W] to [H, W, C]
    if image.ndim == 3:
        image = image.transpose(1, 2, 0)

    return image


def batch_to_grid(
    batch: torch.Tensor,
    nrow: int = 4,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """Convert a batch of images to a grid for display.

    Args:
        batch: Batch tensor [B, C, H, W]
        nrow: Number of images per row
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Numpy array representing the grid [H, W, C]

    Example:
        >>> import matplotlib.pyplot as plt
        >>> batch = next(iter(dataloader))[0][:16]
        >>> grid = batch_to_grid(batch, nrow=4)
        >>> plt.imshow(grid)
        >>> plt.show()
    """
    from torchvision.utils import make_grid

    # Denormalize
    batch = denormalize(batch, mean=mean, std=std)

    # Create grid
    grid = make_grid(batch, nrow=nrow, padding=2, normalize=False)

    # Convert to numpy [H, W, C]
    grid = grid.cpu().numpy().transpose(1, 2, 0)

    return grid
