"""
Grad-CAM Wrapper
================
A thin wrapper around ``pytorch_grad_cam`` that provides a consistent
interface with FEM and CS-FEM, making it easy to compare methods
side-by-side.

Dependency:
    pip install grad-cam

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

import numpy as np
import torch
from pytorch_grad_cam import GradCAM as _GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAM:
    """
    Grad-CAM wrapper with a consistent interface matching FEM and CS-FEM.

    Args:
        model (nn.Module): A CNN model in eval mode.
        target_layer (nn.Module): The convolutional layer to explain
            (e.g. ``model.layer4[-1]`` for ResNet-50).

    Example::

        model = torchvision.models.resnet50(pretrained=True).eval()
        gradcam = GradCAM(model, target_layer=model.layer4[-1])
        heatmap = gradcam(input_tensor)               # top predicted class
        heatmap = gradcam(input_tensor, class_idx=243) # specific class
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._cam = _GradCAM(model=model, target_layers=[target_layer])

    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM saliency heatmap.

        Args:
            input_tensor: Preprocessed image tensor of shape ``[1, 3, H, W]``.
            class_idx: Class to explain. If ``None``, the top predicted class
                is used automatically.

        Returns:
            Saliency map as a numpy array of shape ``(H, W)`` in ``[0, 1]``.
        """
        targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
        heatmap = self._cam(input_tensor, targets=targets)
        return heatmap[0]   # [H, W]

    def release(self):
        """No-op — included for API consistency with FEM and CS-FEM."""
        pass

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        return self.explain(input_tensor, class_idx=class_idx)