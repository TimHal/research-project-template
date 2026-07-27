"""Shared image transform pipeline used by the image DataModules.

Centralizing the default pipeline keeps every DataModule consistent and avoids
copy-pasted transform code. Callables are module-level functions (not lambdas)
so the resulting transforms are picklable and safe with ``num_workers > 0``.
"""

from typing import Callable, Optional

import torchvision.transforms as transforms


def to_rgb(image):
    """Convert a PIL image to RGB. A named function so it stays picklable."""
    return image.convert("RGB")


def build_image_transform(
    img_size: tuple[int, int],
    img_mode: str = "RGB",
    augmentation: str = "none",
    norm_mean: Optional[list[float]] = None,
    norm_std: Optional[list[float]] = None,
    train: bool = True,
    extra_transforms: Optional[list[Callable]] = None,
) -> transforms.Compose:
    """Build the default image transform pipeline.

    Order: resize → channel conversion → extra_transforms → augmentation
    (train only) → ToTensor → normalize (when stats given).

    Args:
        img_size: Target ``(height, width)``.
        img_mode: ``"RGB"`` or ``"L"`` (grayscale).
        augmentation: ``"none"`` or ``"basic"`` (horizontal flip + small rotation).
            Applied only when ``train`` is True.
        norm_mean: Per-channel normalization means (with ``norm_std``).
        norm_std: Per-channel normalization stds (with ``norm_mean``).
        train: Whether this is the training pipeline (enables augmentation).
        extra_transforms: Dataset-specific transforms inserted before
            augmentation (e.g. the EMNIST orientation fix).

    Returns:
        A composed transform producing a normalized tensor.
    """
    pipeline: list[Callable] = [transforms.Resize(img_size)]

    if img_mode == "L":
        pipeline.append(transforms.Grayscale(num_output_channels=1))
    else:
        pipeline.append(transforms.Lambda(to_rgb))

    if extra_transforms:
        pipeline.extend(extra_transforms)

    if train and augmentation == "basic":
        pipeline.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ]
        )

    pipeline.append(transforms.ToTensor())

    if norm_mean is not None and norm_std is not None:
        pipeline.append(transforms.Normalize(mean=norm_mean, std=norm_std))

    return transforms.Compose(pipeline)
