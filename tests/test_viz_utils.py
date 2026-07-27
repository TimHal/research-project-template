"""Tests for the notebook visualization helpers."""

import torch

from util.viz_utils import batch_to_grid, denormalize


def test_denormalize_shape_and_range():
    x = torch.randn(3, 8, 8)
    out = denormalize(x)
    assert out.shape == x.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_batch_to_grid():
    batch = torch.randn(4, 3, 8, 8)
    grid = batch_to_grid(batch, nrow=2)
    assert grid.ndim == 3
    assert grid.shape[-1] == 3  # [H, W, C]
