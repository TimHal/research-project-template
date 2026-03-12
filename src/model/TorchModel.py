"""Generic PyTorch Model Wrapper.

A thin factory wrapper that instantiates any nn.Module by its Python import
path, making it usable from YAML configs without writing a dedicated wrapper.

This is the escape hatch for models not covered by TIMMModel,
TorchvisionModel, or HuggingFaceModel.

Example config:
    model:
      class_path: model.TorchModel.TorchModel
      init_args:
        class_path: "torchvision.models.resnet18"
        init_args:
          num_classes: 10
"""

import importlib
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class TorchModel(nn.Module):
    """Generic wrapper that instantiates any nn.Module by import path.

    Args:
        class_path: Dotted Python path to the model class or factory function
            (e.g., 'torchvision.models.resnet18', 'my_package.MyModel')
        init_args: Dictionary of constructor/factory arguments
        features_layer: Optional layer name for get_features() hook-based
            extraction. Use model.named_modules() to find layer names.

    Example:
        >>> model = TorchModel(
        ...     class_path="torchvision.models.resnet18",
        ...     init_args={"num_classes": 10},
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)  # [1, 10]
    """

    def __init__(
        self,
        class_path: str,
        init_args: Optional[Dict[str, Any]] = None,
        features_layer: Optional[str] = None,
    ):
        super().__init__()
        self.features_layer = features_layer

        init_args = init_args or {}
        module_name, cls_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls_or_fn = getattr(module, cls_name)
        self.model = cls_or_fn(**init_args)

        if not isinstance(self.model, nn.Module):
            raise TypeError(
                f"Expected nn.Module, got {type(self.model)}. "
                f"Make sure '{class_path}' returns an nn.Module."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — delegates to the wrapped model."""
        return self.model(x)

    def get_features(self, x: torch.Tensor, layer: Optional[str] = None) -> torch.Tensor:
        """Extract features from a named layer using a forward hook.

        Args:
            x: Input tensor
            layer: Layer name (from model.named_modules()). Falls back to
                features_layer set in constructor. If neither is set,
                raises ValueError.

        Returns:
            Feature tensor captured from the specified layer
        """
        target_layer = layer or self.features_layer
        if target_layer is None:
            raise ValueError(
                "No layer specified for feature extraction. Pass 'layer' argument "
                "or set 'features_layer' in constructor. "
                f"Available layers: {[n for n, _ in self.model.named_modules() if n]}"
            )

        features = {}

        def hook_fn(module, input, output):
            features["out"] = output

        handle = None
        for name, module in self.model.named_modules():
            if name == target_layer:
                handle = module.register_forward_hook(hook_fn)
                break

        if handle is None:
            raise ValueError(
                f"Layer '{target_layer}' not found. "
                f"Available: {[n for n, _ in self.model.named_modules() if n]}"
            )

        self.model(x)
        handle.remove()

        return features["out"]
