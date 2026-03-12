"""TIMM Model Wrapper for PyTorch Image Models.

This module provides a wrapper around timm models to make them compatible
with the research framework, supporting both classification and feature extraction.

Available models: https://huggingface.co/timm

Common models:
- ResNet: resnet18, resnet50, resnet101
- EfficientNet: efficientnet_b0, efficientnet_b4
- ViT: vit_base_patch16_224, vit_small_patch16_224
- ConvNeXt: convnext_tiny, convnext_small
- Swin: swin_tiny_patch4_window7_224
"""

from typing import Optional

import timm
import torch
import torch.nn as nn


class TIMMModel(nn.Module):
    """Wrapper for TIMM (PyTorch Image Models) with feature extraction support.

    Args:
        model_name: Name of the timm model (e.g., 'resnet50', 'vit_base_patch16_224')
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes (0 for feature extraction only)
        in_chans: Number of input channels
        features_only: Whether to return features only (for reconstruction tasks)
        drop_rate: Dropout rate
        **kwargs: Additional arguments passed to timm.create_model

    Example:
        >>> model = TIMMModel("resnet18", pretrained=True, num_classes=10)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)  # [1, 10]
        >>> features = model.get_features(x)  # [1, 512]
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = False,
        num_classes: int = 1000,
        in_chans: int = 3,
        features_only: bool = False,
        drop_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.model_name = model_name
        self.features_only = features_only

        # Create the TIMM model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            features_only=features_only,
            drop_rate=drop_rate,
            **kwargs,
        )

        # Get model configuration
        self.num_features = (
            self.model.num_features if hasattr(self.model, "num_features") else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor [N, C, H, W]

        Returns:
            Output tensor (features or logits depending on configuration)
        """
        return self.model(x)

    def get_features(self, x: torch.Tensor, layer: Optional[str] = None) -> torch.Tensor:
        """Extract features from the model.

        Args:
            x: Input tensor [N, C, H, W]
            layer: Layer name to extract from (use model.named_modules() to
                list available names). If None, returns features from the
                penultimate layer (before the classifier head).

        Returns:
            Feature tensor. Shape depends on architecture and layer.

        Raises:
            ValueError: If the specified layer is not found.
        """
        if layer is not None:
            features = {}

            def hook_fn(module, input, output):
                features["out"] = output

            handle = None
            for name, module in self.model.named_modules():
                if name == layer:
                    handle = module.register_forward_hook(hook_fn)
                    break

            if handle is None:
                raise ValueError(
                    f"Layer '{layer}' not found. "
                    f"Available: {[n for n, _ in self.model.named_modules() if n]}"
                )

            self.model(x)
            handle.remove()
            return features["out"]
        else:
            # Return final features before classifier
            if self.features_only:
                return self.model(x)[-1]
            elif hasattr(self.model, "forward_features"):
                return self.model.forward_features(x)
            else:
                return self.model(x)
