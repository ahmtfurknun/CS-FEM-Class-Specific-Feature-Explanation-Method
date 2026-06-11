"""
Feature Explanation Method (FEM)
=================================
Class-agnostic saliency method based on statistical thresholding of CNN
feature map activations.

Reference:
    Fuad, K. A. A., Martin, P. E., Giot, R., Bourqui, R., Benois-Pineau, J., & Zemmari, A. (2020, November). 
    Features understanding in 3D CNNS for actions recognition in video. 
    In 2020 Tenth International Conference on Image Processing Theory, Tools and Applications (IPTA) (pp. 1-6). IEEE.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation & gradient extraction
# ---------------------------------------------------------------------------

class ActivationsAndGradients:
    """
    Registers forward hooks on target layers to capture activations and
    gradients during a forward (and optional backward) pass.

    Args:
        model (nn.Module): The neural network model.
        target_layers (list[nn.Module]): Layers to hook into.
    """

    def __init__(self, model: torch.nn.Module, target_layers: list):
        self.model = model
        self.activations: list[torch.Tensor] = []
        self.gradients: list[torch.Tensor] = []
        self._handles = []

        for layer in target_layers:
            self._handles.append(layer.register_forward_hook(self._save_activation))
            self._handles.append(layer.register_forward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations.append(output.cpu().detach())

    def _save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store)

    def release(self):
        """Remove all hooks. Call this after you are done with the explainer."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.activations = []
        self.gradients = []
        return self.model(x)


# ---------------------------------------------------------------------------
# FEM
# ---------------------------------------------------------------------------

class FEM:
    """
    Feature Explanation Method (FEM).

    Produces a class-agnostic saliency map by:
        1. Computing per-channel binary masks via statistical thresholding
           (threshold = mean + K * std over spatial dimensions).
        2. Weighting masked channels by their mean activation magnitude.
        3. Summing and normalising the result.

    Args:
        model (nn.Module): A CNN model in eval mode.
        target_layer (nn.Module): The convolutional layer to explain
            (e.g. ``model.layer4[-1]`` for ResNet-50).
        k (float): Threshold multiplier for the statistical gate.
            Higher values produce sparser, more confident masks. Default: 2.
    
    Example::

        model = torchvision.models.resnet50(pretrained=True).eval()
        fem = FEM(model, target_layer=model.layer4[-1])
        heatmap = fem(input_tensor)          # numpy array, H x W in [0, 1]
        fem.release()
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, k: float = 2.0):
        self.model = model
        self.k = k
        self._ang = ActivationsAndGradients(model, [target_layer])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_binary_maps(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Threshold each channel independently using mean + K * std.

        Args:
            activations: Shape ``[1, C, H, W]``.

        Returns:
            Binary tensor of the same shape.
        """
        mean = activations.mean(dim=(2, 3), keepdim=True)
        std  = activations.std(dim=(2, 3),  keepdim=True)
        return (activations >= mean + self.k * std).float()

    def _aggregate(
        self,
        weighted_maps: torch.Tensor,
        output_size: tuple[int, int],
        positive_only: bool = True,
    ) -> np.ndarray:
        """
        Aggregate channel-wise weighted binary maps into a single heatmap.

        Args:
            weighted_maps: Shape ``[C, H, W]`` — already channel-weighted binary maps.
            output_size: ``(height, width)`` of the original input image.
            positive_only: If ``True`` (recommended), keep only positive contributions.

        Returns:
            Normalised heatmap as a numpy array of shape ``(H, W)`` in ``[0, 1]``.
        """
        if positive_only:
            weighted_maps = F.relu(weighted_maps)

        heatmap = weighted_maps.sum(dim=0)   # [H, W]
        heatmap = F.relu(heatmap)

        # Normalise to [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = cv2.resize(heatmap_np, (output_size[1], output_size[0]))

        # Stretch to full [0, 1] range after resizing
        lo, hi = heatmap_np.min(), heatmap_np.max()
        if hi > lo:
            heatmap_np = (heatmap_np - lo) / (hi - lo)

        return heatmap_np

    # ------------------------------------------------------------------
    # Core weighting strategy (overridden in CS-FEM)
    # ------------------------------------------------------------------

    def _compute_channel_weights(
        self,
        activations: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-channel weights.

        FEM uses the spatial mean of each activation channel, making the
        method class-agnostic.

        Args:
            activations: Shape ``[1, C, H, W]``.
            output: Model logits (unused in base FEM).

        Returns:
            Weight tensor of shape ``[C, 1, 1]``.
        """
        return activations[0].mean(dim=(1, 2)).reshape(-1, 1, 1)  # [C, 1, 1]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain(
        self,
        input_tensor: torch.Tensor,
        positive_only: bool = True,
    ) -> np.ndarray:
        """
        Generate a saliency heatmap for ``input_tensor``.

        Args:
            input_tensor: Preprocessed image tensor of shape ``[1, 3, H, W]``.
            positive_only: Keep only positively contributing channels.
                Set to ``False`` to visualise suppressive features as well.

        Returns:
            Saliency map as a numpy array of shape ``(H, W)`` in ``[0, 1]``.
        """
        _, _, H, W = input_tensor.shape

        output = self._ang(input_tensor)
        activations = self._ang.activations[0]          # [1, C, h, w]

        binary_maps     = self._compute_binary_maps(activations)          # [1, C, h, w]
        channel_weights = self._compute_channel_weights(activations, output)  # [C, 1, 1]
        weighted_maps   = binary_maps[0] * channel_weights                # [C, h, w]

        return self._aggregate(weighted_maps, output_size=(H, W), positive_only=positive_only)

    def release(self):
        """Release all forward hooks. Call when done."""
        self._ang.release()

    def __call__(self, input_tensor: torch.Tensor, positive_only: bool = True) -> np.ndarray:
        return self.explain(input_tensor, positive_only=positive_only)