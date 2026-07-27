"""Tests for the shared image transform builder.

Locks in the two guarantees of ``build_image_transform``: correct output
tensors, and picklability (so transforms are safe with ``num_workers > 0``).
"""

import pickle

import torch
from PIL import Image

from data.transforms import build_image_transform


def test_rgb_output_shape_and_range():
    transform = build_image_transform((8, 8), img_mode="RGB")
    out = transform(Image.new("RGB", (16, 16), color=(120, 30, 200)))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 8, 8)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_grayscale_output_has_one_channel():
    transform = build_image_transform((8, 8), img_mode="L")
    out = transform(Image.new("RGB", (16, 16)))
    assert out.shape == (1, 8, 8)


def test_normalization_applied():
    transform = build_image_transform(
        (8, 8), norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]
    )
    out = transform(Image.new("RGB", (16, 16), color=(0, 0, 0)))
    # A black pixel normalized by (x-0.5)/0.5 becomes -1.
    assert torch.allclose(out, torch.full_like(out, -1.0))


def test_augmentation_only_when_training():
    train = build_image_transform((8, 8), augmentation="basic", train=True)
    val = build_image_transform((8, 8), augmentation="basic", train=False)
    train_types = [type(t).__name__ for t in train.transforms]
    val_types = [type(t).__name__ for t in val.transforms]
    assert "RandomHorizontalFlip" in train_types
    assert "RandomHorizontalFlip" not in val_types


def test_transform_is_picklable():
    # Named callables (not lambdas) keep the pipeline picklable — required for
    # DataLoader workers under the 'spawn' start method.
    transform = build_image_transform((8, 8), img_mode="RGB", augmentation="basic")
    pickle.loads(pickle.dumps(transform))
