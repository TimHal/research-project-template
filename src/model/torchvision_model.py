"""Torchvision Model Wrapper.

Wraps torchvision.models for use with the research framework, supporting
both classification and feature extraction.

Available models: https://pytorch.org/vision/stable/models.html

Common models:
- ResNet: resnet18, resnet50, resnet101
- EfficientNet: efficientnet_b0, efficientnet_b4, efficientnet_v2_s
- ViT: vit_b_16, vit_b_32, vit_l_16
- ConvNeXt: convnext_tiny, convnext_small
- Swin: swin_t, swin_s, swin_b
- MobileNet: mobilenet_v3_small, mobilenet_v3_large
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

# Classifier head attribute names used by different torchvision architectures
_CLASSIFIER_ATTR_NAMES = ("fc", "classifier", "head", "heads")


def _get_classifier_attr(model: nn.Module) -> tuple[str, nn.Module]:
    """Find the classifier head attribute on a torchvision model."""
    for name in _CLASSIFIER_ATTR_NAMES:
        if hasattr(model, name):
            return name, getattr(model, name)
    raise ValueError(
        f"Cannot find classifier head. Looked for: {_CLASSIFIER_ATTR_NAMES}. "
        f"Available attributes: {[n for n, _ in model.named_children()]}"
    )


def _get_classifier_in_features(classifier: nn.Module) -> int:
    """Extract in_features from a classifier head (handles Linear and Sequential)."""
    if isinstance(classifier, nn.Linear):
        return classifier.in_features
    if isinstance(classifier, nn.Sequential):
        for layer in reversed(list(classifier.children())):
            if isinstance(layer, nn.Linear):
                return layer.in_features
    raise ValueError(f"Cannot determine in_features from classifier: {type(classifier)}")


class TorchvisionModel(nn.Module):
    """Wrapper for torchvision models with feature extraction support.

    Args:
        model_name: Name of the torchvision model (e.g., 'resnet50', 'vit_b_16')
        weights: Pretrained weights. Use a string like "IMAGENET1K_V1",
            "IMAGENET1K_V2", "DEFAULT", True (= "DEFAULT"), or None/False.
        num_classes: Number of output classes. If different from the model's
            default, the classifier head is replaced.
        **kwargs: Additional arguments passed to the model constructor.

    Example:
        >>> model = TorchvisionModel("resnet18", weights="IMAGENET1K_V1", num_classes=10)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)  # [1, 10]
        >>> features = model.get_features(x)  # [1, 512]
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        weights: Optional[str | bool] = None,
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name

        # Resolve weights argument
        if weights is True:
            weights = "DEFAULT"
        elif weights is False:
            weights = None

        # Create the model
        model_fn = getattr(tv_models, model_name, None)
        if model_fn is None:
            raise ValueError(
                f"Unknown torchvision model: {model_name}. "
                f"See https://pytorch.org/vision/stable/models.html"
            )
        self.model = model_fn(weights=weights, **kwargs)

        # Replace classifier head if num_classes differs
        self._classifier_attr, classifier = _get_classifier_attr(self.model)
        default_out = None
        if isinstance(classifier, nn.Linear):
            default_out = classifier.out_features
        elif isinstance(classifier, nn.Sequential):
            for layer in reversed(list(classifier.children())):
                if isinstance(layer, nn.Linear):
                    default_out = layer.out_features
                    break

        if default_out is not None and num_classes != default_out:
            in_features = _get_classifier_in_features(classifier)
            setattr(self.model, self._classifier_attr, nn.Linear(in_features, num_classes))

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def get_features(self, x: torch.Tensor, layer: Optional[str] = None) -> torch.Tensor:
        """Extract features from the model.

        Args:
            x: Input tensor [N, C, H, W]
            layer: Layer name to extract from (use model.named_modules() to
                list available names). If None, returns features from the
                penultimate layer (the input to the classifier head).

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

        # Default: extract features before the classifier head
        features = {}

        def capture_input(module, input, output):
            features["penultimate"] = input[0] if isinstance(input, tuple) else input

        classifier = getattr(self.model, self._classifier_attr)
        # For Sequential classifiers, hook into the first Linear layer
        target = classifier
        if isinstance(classifier, nn.Sequential):
            for child in classifier.children():
                if isinstance(child, nn.Linear):
                    target = child
                    break

        handle = target.register_forward_hook(capture_input)
        self.model(x)
        handle.remove()

        return features["penultimate"]
