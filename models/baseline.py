"""
Baseline models for GEDI AGB prediction using foundation model embeddings + lat/lon.

These baselines test whether the CNP's context aggregation mechanism adds value
beyond simple point-to-point mapping with embeddings + coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.neural_process import EmbeddingEncoder


class SimpleMLPBaseline(nn.Module):
    """
    Simple MLP baseline: embedding + lat/lon -> AGBD prediction.

    Uses the same EmbeddingEncoder as the CNP for fair comparison,
    but skips the context/target mechanism and just learns a direct mapping.
    """

    def __init__(
        self,
        patch_size: int = 3,
        embedding_channels: int = 128,
        embedding_feature_dim: int = 128,
        hidden_dim: int = 512,
        output_uncertainty: bool = True,
        num_hidden_layers: int = 3
    ):
        """
        Initialize simple MLP baseline.

        Args:
            patch_size: Size of embedding patches
            embedding_channels: Number of channels in embeddings
            embedding_feature_dim: Dimension of encoded embedding features
            hidden_dim: Hidden layer dimension
            output_uncertainty: Whether to predict uncertainty
            num_hidden_layers: Number of hidden layers in the MLP
        """
        super().__init__()

        self.output_uncertainty = output_uncertainty

        # Reuse the same embedding encoder as CNP for fair comparison
        self.embedding_encoder = EmbeddingEncoder(
            patch_size=patch_size,
            in_channels=embedding_channels,
            hidden_dim=hidden_dim,
            output_dim=embedding_feature_dim
        )

        # Simple MLP: [embedding_features + coords] -> AGBD
        input_dim = embedding_feature_dim + 2  # embedding features + lat/lon

        layers = []

        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ])

        # Hidden layers with residual connections
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])

        self.mlp = nn.ModuleList(layers)

        # Output heads
        self.mean_head = nn.Linear(hidden_dim, 1)

        if output_uncertainty:
            self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        coords: torch.Tensor,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            coords: (batch, 2) - normalized lat/lon coordinates
            embeddings: (batch, patch_size, patch_size, channels) - embedding patches

        Returns:
            (predicted_agbd, log_variance) for each point
            Shape: (batch, 1) for each
        """
        # Encode embeddings to features
        embedding_features = self.embedding_encoder(embeddings)

        # Concatenate with coordinates
        x = torch.cat([embedding_features, coords], dim=-1)

        # Pass through MLP with residual connections
        for i in range(0, len(self.mlp), 3):  # Each block is 3 layers (Linear, LayerNorm, ReLU)
            linear = self.mlp[i]
            norm = self.mlp[i + 1]
            activation = self.mlp[i + 2]

            if i == 0:
                # First layer - no residual
                x = activation(norm(linear(x)))
            else:
                # Hidden layers - with residual
                identity = x
                x = activation(norm(linear(x)) + identity)

        # Output heads
        mean = self.mean_head(x)

        if self.output_uncertainty:
            log_var = self.log_var_head(x)
            # Clamp log_var to prevent numerical instability (same as CNP)
            log_var = torch.clamp(log_var, min=-7, max=5)
            return mean, log_var
        else:
            return mean, None

    def predict(
        self,
        coords: torch.Tensor,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty.

        Returns:
            (mean, std) predictions
        """
        pred_mean, pred_log_var = self.forward(coords, embeddings)

        if pred_log_var is not None:
            pred_std = torch.exp(0.5 * pred_log_var)
        else:
            pred_std = torch.zeros_like(pred_mean)

        return pred_mean, pred_std


class FlatMLPBaseline(nn.Module):
    """
    Alternative MLP baseline that flattens the embedding patch instead of using CNN.

    This tests whether the CNN encoding of spatial structure is valuable.
    """

    def __init__(
        self,
        patch_size: int = 3,
        embedding_channels: int = 128,
        hidden_dim: int = 512,
        output_uncertainty: bool = True,
        num_hidden_layers: int = 3
    ):
        """
        Initialize flat MLP baseline.

        Args:
            patch_size: Size of embedding patches
            embedding_channels: Number of channels in embeddings
            hidden_dim: Hidden layer dimension
            output_uncertainty: Whether to predict uncertainty
            num_hidden_layers: Number of hidden layers in the MLP
        """
        super().__init__()

        self.output_uncertainty = output_uncertainty
        self.patch_size = patch_size
        self.embedding_channels = embedding_channels

        # Input: flattened embedding + coords
        input_dim = patch_size * patch_size * embedding_channels + 2  # 3*3*128 + 2 = 1154

        layers = []

        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ])

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])

        self.mlp = nn.ModuleList(layers)

        # Output heads
        self.mean_head = nn.Linear(hidden_dim, 1)

        if output_uncertainty:
            self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        coords: torch.Tensor,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            coords: (batch, 2)
            embeddings: (batch, patch_size, patch_size, channels)

        Returns:
            (predicted_agbd, log_variance)
        """
        batch_size = coords.shape[0]

        # Flatten embeddings
        flat_embeddings = embeddings.reshape(batch_size, -1)

        # Concatenate with coordinates
        x = torch.cat([flat_embeddings, coords], dim=-1)

        # Pass through MLP with residual connections
        for i in range(0, len(self.mlp), 3):
            linear = self.mlp[i]
            norm = self.mlp[i + 1]
            activation = self.mlp[i + 2]

            if i == 0:
                x = activation(norm(linear(x)))
            else:
                identity = x
                x = activation(norm(linear(x)) + identity)

        # Output heads
        mean = self.mean_head(x)

        if self.output_uncertainty:
            log_var = self.log_var_head(x)
            log_var = torch.clamp(log_var, min=-7, max=5)
            return mean, log_var
        else:
            return mean, None

    def predict(
        self,
        coords: torch.Tensor,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty."""
        pred_mean, pred_log_var = self.forward(coords, embeddings)

        if pred_log_var is not None:
            pred_std = torch.exp(0.5 * pred_log_var)
        else:
            pred_std = torch.zeros_like(pred_mean)

        return pred_mean, pred_std


def baseline_loss(
    pred_mean: torch.Tensor,
    pred_log_var: Optional[torch.Tensor],
    target: torch.Tensor
) -> torch.Tensor:
    """
    Loss function for baseline models (same as neural_process_loss).

    Args:
        pred_mean: Predicted means (batch, 1)
        pred_log_var: Predicted log variances (batch, 1) or None
        target: Target values (batch, 1)

    Returns:
        Scalar loss
    """
    if pred_log_var is not None:
        # Gaussian negative log-likelihood
        loss = 0.5 * (
            pred_log_var +
            torch.exp(-pred_log_var) * (target - pred_mean) ** 2
        )
    else:
        # MSE loss
        loss = (target - pred_mean) ** 2

    return loss.mean()
