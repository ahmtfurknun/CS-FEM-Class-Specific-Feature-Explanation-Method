"""
Class-Specific Feature Explanation Method (CS-FEM)
====================================================
A class-discriminative saliency method that extends FEM by injecting
class-specific weights into the statistical thresholding framework,
acting as a "semantic noise gate".

CS-FEM supports two equivalent weighting modes:

* ``"weights"``   — uses the classification head weights for the predicted
                    (or specified) class. Fast and exact; requires the model
                    to expose a linear classifier layer.

* ``"gradients"`` — uses the global average pooled gradients of the class
                    score w.r.t. the target layer activations (Grad-CAM
                    style). Works on any architecture. Theoretically
                    equivalent to ``"weights"`` for models with a GAP +
                    linear classifier head (see paper, Sec. 3.2).
"""

import numpy as np
import torch
import torch.nn.functional as F

from .fem import FEM


class CSFEM(FEM):
    """
    Class-Specific Feature Explanation Method (CS-FEM).

    Inherits the statistical thresholding pipeline from :class:`FEM` and
    replaces the class-agnostic channel weights with class-specific weights,
    making the method discriminative across categories.

    Args:
        model (nn.Module): A CNN model in eval mode.
        target_layer (nn.Module): The convolutional layer to explain
            (e.g. ``model.layer4[-1]`` for ResNet-50).
        classifier_layer (nn.Linear, optional): The final linear
            classification layer (e.g. ``model.fc``). Required when
            ``mode="weights"``. Ignored when ``mode="gradients"``.
        mode (str): Weighting strategy — ``"weights"`` or ``"gradients"``.
            Default: ``"weights"``.
        k (float): Statistical threshold multiplier (inherited from FEM).
            Default: 2.

    Raises:
        ValueError: If ``mode="weights"`` but ``classifier_layer`` is not provided.
        ValueError: If ``mode`` is not ``"weights"`` or ``"gradients"``.

    Example — weights mode (ResNet-50)::

        model = torchvision.models.resnet50(pretrained=True).eval()
        csfem = CSFEM(
            model,
            target_layer=model.layer4[-1],
            classifier_layer=model.fc,
            mode="weights",
        )
        heatmap = csfem(input_tensor)
        csfem.release()

    Example — gradients mode (any CNN)::

        csfem = CSFEM(model, target_layer=model.features[-1], mode="gradients")
        heatmap = csfem(input_tensor)
        csfem.release()

    Example — explain a specific class::

        heatmap = csfem(input_tensor, class_idx=243)
    """

    _VALID_MODES = {"weights", "gradients"}

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        classifier_layer: torch.nn.Module | None = None,
        mode: str = "weights",
        k: float = 2.0,
    ):
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}, got '{mode}'.")
        if mode == "weights" and classifier_layer is None:
            raise ValueError(
                "classifier_layer must be provided when mode='weights'. "
                "Pass the model's final linear layer (e.g. model.fc) or "
                "switch to mode='gradients'."
            )

        super().__init__(model, target_layer, k=k)
        self.classifier_layer = classifier_layer
        self.mode = mode

    # ------------------------------------------------------------------
    # Class-specific weighting strategies
    # ------------------------------------------------------------------

    def _weights_from_classifier(
        self,
        output: torch.Tensor,
        class_idx: int | None,
    ) -> torch.Tensor:
        """
        Extract per-channel weights from the classification head.

        For a model with a GAP → linear head this is theoretically
        equivalent to the GAP of Grad-CAM gradients.

        Args:
            output: Model logits of shape ``[1, num_classes]``.
            class_idx: Target class. If ``None``, uses the top predicted class.

        Returns:
            Weight tensor of shape ``[C, 1, 1]``.
        """
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        weights = self.classifier_layer.weight[class_idx].detach().cpu()
        return weights.reshape(-1, 1, 1)   # [C, 1, 1]

    def _weights_from_gradients(
        self,
        output: torch.Tensor,
        activations: torch.Tensor,
        class_idx: int | None,
    ) -> torch.Tensor:
        """
        Compute per-channel weights as the global average pooled gradients
        of the target class score w.r.t. the activation maps (Grad-CAM).

        Args:
            output: Model logits of shape ``[1, num_classes]``.
            activations: Activation tensor of shape ``[1, C, H, W]``.
            class_idx: Target class. If ``None``, uses the top predicted class.

        Returns:
            Weight tensor of shape ``[C, 1, 1]``.
        """
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        # Gradients are stored by ActivationsAndGradients in reverse order
        gradients = self._ang.gradients[0]  # [1, C, H, W]
        weights = gradients.mean(dim=(2, 3))[0]  # GAP → [C]
        return weights.reshape(-1, 1, 1)   # [C, 1, 1]

    # ------------------------------------------------------------------
    # Override FEM hook
    # ------------------------------------------------------------------

    def _compute_channel_weights(
        self,
        activations: torch.Tensor,
        output: torch.Tensor,
        class_idx: int | None = None,
    ) -> torch.Tensor:
        """
        Dispatch to the selected weighting mode.

        Args:
            activations: Shape ``[1, C, H, W]``.
            output: Model logits.
            class_idx: Target class index (``None`` → top predicted class).

        Returns:
            Weight tensor of shape ``[C, 1, 1]``.
        """
        if self.mode == "weights":
            class_weights = self._weights_from_classifier(output, class_idx)
        else:
            class_weights = self._weights_from_gradients(output, activations, class_idx)
        
        activation_means = super()._compute_channel_weights(activations, output)
        return class_weights * activation_means

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        positive_only: bool = True,
    ) -> np.ndarray:
        """
        Generate a class-specific saliency heatmap.

        Args:
            input_tensor: Preprocessed image tensor of shape ``[1, 3, H, W]``.
            class_idx: Class to explain. If ``None``, the top predicted class
                is used automatically.
            positive_only: Keep only positively contributing channels.

        Returns:
            Saliency map as a numpy array of shape ``(H, W)`` in ``[0, 1]``.
        """
        _, _, H, W = input_tensor.shape

        output = self._ang(input_tensor)
        activations = self._ang.activations[0]              # [1, C, h, w]

        binary_maps     = self._compute_binary_maps(activations)                           # [1, C, h, w]
        channel_weights = self._compute_channel_weights(activations, output, class_idx)    # [C, 1, 1]
        weighted_maps   = binary_maps[0] * channel_weights                                 # [C, h, w]

        return self._aggregate(weighted_maps, output_size=(H, W), positive_only=positive_only)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        positive_only: bool = True,
    ) -> np.ndarray:
        return self.explain(input_tensor, class_idx=class_idx, positive_only=positive_only)