"""HuggingFace Transformers Model Wrapper.

Wraps HuggingFace transformers AutoModel classes for use with the research
framework, supporting image classification, sequence classification, and
feature extraction.

Requires: pip install transformers

Available models: https://huggingface.co/models

Common vision models:
- ViT: google/vit-base-patch16-224, google/vit-large-patch16-224
- DeiT: facebook/deit-base-patch16-224
- Swin: microsoft/swin-base-patch4-window7-224
- BEiT: microsoft/beit-base-patch16-224

Common text models:
- BERT: bert-base-uncased, bert-large-uncased
- RoBERTa: roberta-base, roberta-large
- DistilBERT: distilbert-base-uncased
"""

from typing import Any, Optional

import torch
import torch.nn as nn


def _import_transformers():
    """Lazy import of transformers with clear error message."""
    try:
        import transformers
        return transformers
    except ImportError:
        raise ImportError(
            "HuggingFaceModel requires the 'transformers' package. "
            "Install it with: pip install transformers"
        ) from None


# Maps task strings to AutoModel class names
_TASK_TO_AUTO_CLASS = {
    "image-classification": "AutoModelForImageClassification",
    "sequence-classification": "AutoModelForSequenceClassification",
    "feature-extraction": "AutoModel",
}


class HuggingFaceModel(nn.Module):
    """Wrapper for HuggingFace Transformers models.

    Args:
        model_name: HuggingFace model identifier
            (e.g., 'google/vit-base-patch16-224', 'bert-base-uncased')
        task: Model task type. One of:
            - "image-classification": AutoModelForImageClassification
            - "sequence-classification": AutoModelForSequenceClassification
            - "feature-extraction": AutoModel (no classification head)
        num_labels: Number of output classes (for classification tasks)
        pretrained: Whether to load pretrained weights (default True)
        **kwargs: Additional arguments passed to from_pretrained or from_config

    Example:
        >>> model = HuggingFaceModel(
        ...     "google/vit-base-patch16-224",
        ...     task="image-classification",
        ...     num_labels=10,
        ... )
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)  # [1, 10]
        >>> features = model.get_features(x)  # [1, 197, 768]
    """

    def __init__(
        self,
        model_name: str,
        task: str = "image-classification",
        num_labels: int = 1000,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        transformers = _import_transformers()

        self.model_name = model_name
        self.task = task
        self.num_labels = num_labels

        if task not in _TASK_TO_AUTO_CLASS:
            raise ValueError(
                f"Unknown task: {task}. Choose from: {list(_TASK_TO_AUTO_CLASS.keys())}"
            )

        auto_cls_name = _TASK_TO_AUTO_CLASS[task]
        auto_cls = getattr(transformers, auto_cls_name)

        if pretrained:
            model_kwargs: dict[str, Any] = dict(kwargs)
            if task != "feature-extraction":
                model_kwargs["num_labels"] = num_labels
                model_kwargs["ignore_mismatched_sizes"] = True
            self.model = auto_cls.from_pretrained(model_name, **model_kwargs)
        else:
            config_cls = transformers.AutoConfig
            config = config_cls.from_pretrained(model_name, **kwargs)
            if task != "feature-extraction":
                config.num_labels = num_labels
            self.model = auto_cls.from_config(config)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor. For vision models: [N, C, H, W].
                For text models: token IDs [N, seq_len] (pass attention_mask via kwargs).

        Returns:
            Logits tensor [N, num_labels] for classification, or
            last hidden state [N, seq_len, hidden_dim] for feature extraction.
        """
        output = self.model(x, **kwargs)

        if self.task == "feature-extraction":
            return output.last_hidden_state

        return output.logits

    def get_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract hidden-state features (before the classification head).

        Uses output_hidden_states=True to get the last hidden state from the
        same model, without loading a separate feature extractor.

        Args:
            x: Input tensor (same format as forward)

        Returns:
            Last hidden state tensor [N, seq_len, hidden_dim]
        """
        output = self.model(x, output_hidden_states=True, **kwargs)
        return output.hidden_states[-1]
