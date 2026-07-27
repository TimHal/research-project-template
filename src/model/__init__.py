"""Model architectures (nn.Module wrappers) for research experiments.

Every wrapper is re-exported here so configs can reference it by the short path
``model.<ClassName>`` (e.g. ``model.TIMMModel``). HuggingFaceModel imports
``transformers`` lazily, so importing this package never requires it.
"""

from model.huggingface_model import HuggingFaceModel
from model.timm_model import TIMMModel
from model.torch_model import TorchModel
from model.torchvision_model import TorchvisionModel

__all__ = ["HuggingFaceModel", "TIMMModel", "TorchModel", "TorchvisionModel"]
