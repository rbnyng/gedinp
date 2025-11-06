"""
Convolutional Conditional Neural Process for GEDI AGB prediction.

Uses UNet to process sparse GEDI observations + dense embeddings,
then decodes to dense AGBD predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .unet import UNet, UNetSmall


class ConvCNPDecoder(nn.Module):
    """
    Decoder MLP for ConvCNP.

    Takes feature vector at a pixel and outputs (mean, log_variance).
    """

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        output_uncertainty: bool = True
    ):
        super().__init__()
        self.output_uncertainty = output_uncertainty

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, 1)

        if output_uncertainty:
            self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode features to predictions.

        Args:
            features: Feature tensor (B, feature_dim, H, W) or (B, feature_dim)

        Returns:
            (mean, log_variance) predictions
            Shape: (B, 1, H, W) or (B, 1) depending on input
        """
        # Handle both spatial and flat inputs
        spatial_input = features.dim() == 4

        if spatial_input:
            B, C, H, W = features.shape
            # Flatten spatial dimensions: (B, C, H, W) -> (B*H*W, C)
            features = features.permute(0, 2, 3, 1).reshape(-1, C)

        # MLP
        x = self.mlp(features)
        mean = self.mean_head(x)

        if self.output_uncertainty:
            log_var = self.log_var_head(x)
        else:
            log_var = None

        # Reshape back to spatial if needed
        if spatial_input:
            mean = mean.reshape(B, H, W, 1).permute(0, 3, 1, 2)  # (B, 1, H, W)
            if log_var is not None:
                log_var = log_var.reshape(B, H, W, 1).permute(0, 3, 1, 2)

        return mean, log_var


class GEDIConvCNP(nn.Module):
    """
    Convolutional Conditional Neural Process for GEDI AGB interpolation.

    Architecture:
    1. Input: Sparse AGBD + mask + dense embeddings (B, 130, H, W)
    2. UNet encoder: Process to dense feature map (B, feature_dim, H, W)
    3. Decoder: Predict (mean, variance) at each pixel

    Key advantages over baseline CNP:
    - Preserves spatial structure (no mean pooling bottleneck)
    - Efficient dense predictions (process tile once)
    - Strong inductive bias for gridded data
    """

    def __init__(
        self,
        embedding_channels: int = 128,
        feature_dim: int = 128,
        base_channels: int = 64,
        unet_depth: int = 3,
        decoder_hidden_dim: int = 128,
        output_uncertainty: bool = True,
        use_small_unet: bool = False
    ):
        """
        Initialize ConvCNP.

        Args:
            embedding_channels: Number of embedding channels (128 for GeoTessera)
            feature_dim: Dimension of UNet output features
            base_channels: Base channels for UNet
            unet_depth: Number of UNet levels
            decoder_hidden_dim: Hidden dimension for decoder MLP
            output_uncertainty: Whether to predict uncertainty
            use_small_unet: Use smaller UNet architecture
        """
        super().__init__()

        self.embedding_channels = embedding_channels
        self.feature_dim = feature_dim
        self.output_uncertainty = output_uncertainty

        # Input: 1 (AGBD) + 1 (mask) + 128 (embeddings) = 130 channels
        in_channels = 2 + embedding_channels

        # UNet encoder
        if use_small_unet:
            self.encoder = UNetSmall(
                in_channels=in_channels,
                feature_dim=feature_dim,
                base_channels=base_channels
            )
        else:
            self.encoder = UNet(
                in_channels=in_channels,
                feature_dim=feature_dim,
                base_channels=base_channels,
                depth=unet_depth
            )

        # Decoder
        self.decoder = ConvCNPDecoder(
            feature_dim=feature_dim,
            hidden_dim=decoder_hidden_dim,
            output_uncertainty=output_uncertainty
        )

    def forward(
        self,
        tile_embedding: torch.Tensor,
        context_agbd: torch.Tensor,
        context_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            tile_embedding: Dense embeddings (B, C, H, W)
            context_agbd: Sparse AGBD values (B, 1, H, W)
            context_mask: Binary mask (B, 1, H, W)

        Returns:
            (pred_mean, pred_log_var) for all pixels
            Shape: (B, 1, H, W) each
        """
        # Concatenate inputs: (B, 130, H, W)
        x = torch.cat([context_agbd, context_mask, tile_embedding], dim=1)

        # Encode to feature map
        features = self.encoder(x)  # (B, feature_dim, H, W)

        # Decode to predictions
        pred_mean, pred_log_var = self.decoder(features)

        return pred_mean, pred_log_var

    def predict(
        self,
        tile_embedding: torch.Tensor,
        context_agbd: torch.Tensor,
        context_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty.

        Returns:
            (mean, std) predictions for all pixels
        """
        pred_mean, pred_log_var = self.forward(
            tile_embedding,
            context_agbd,
            context_mask
        )

        if pred_log_var is not None:
            pred_std = torch.exp(0.5 * pred_log_var)
        else:
            pred_std = torch.zeros_like(pred_mean)

        return pred_mean, pred_std


def convcnp_loss(
    pred_mean: torch.Tensor,
    pred_log_var: Optional[torch.Tensor],
    target_agbd: torch.Tensor,
    target_mask: torch.Tensor
) -> torch.Tensor:
    """
    Negative log-likelihood loss for ConvCNP.

    Only computes loss at target locations (where mask=1).

    Args:
        pred_mean: Predicted means (B, 1, H, W)
        pred_log_var: Predicted log variances (B, 1, H, W) or None
        target_agbd: Target AGBD values (B, 1, H, W)
        target_mask: Binary mask indicating target locations (B, 1, H, W)

    Returns:
        Scalar loss
    """
    # Only compute loss where we have targets
    if pred_log_var is not None:
        # Gaussian negative log-likelihood
        loss = 0.5 * (
            pred_log_var +
            torch.exp(-pred_log_var) * (target_agbd - pred_mean) ** 2
        )
    else:
        # MSE loss
        loss = (target_agbd - pred_mean) ** 2

    # Apply mask and average
    masked_loss = loss * target_mask
    n_targets = target_mask.sum()

    if n_targets > 0:
        return masked_loss.sum() / n_targets
    else:
        return torch.tensor(0.0, device=loss.device)


def compute_metrics(
    pred_mean: torch.Tensor,
    pred_std: Optional[torch.Tensor],
    target_agbd: torch.Tensor,
    target_mask: torch.Tensor
) -> dict:
    """
    Compute evaluation metrics at target locations.

    Args:
        pred_mean: Predicted means (B, 1, H, W)
        pred_std: Predicted standard deviations (B, 1, H, W) or None
        target_agbd: Target AGBD values (B, 1, H, W)
        target_mask: Binary mask (B, 1, H, W)

    Returns:
        Dictionary of metrics
    """
    # Extract values at target locations
    mask_bool = target_mask > 0.5

    pred_values = pred_mean[mask_bool].detach().cpu().numpy().flatten()
    target_values = target_agbd[mask_bool].detach().cpu().numpy().flatten()

    if len(pred_values) == 0:
        return {}

    # Compute metrics
    mse = ((pred_values - target_values) ** 2).mean()
    rmse = mse ** 0.5
    mae = abs(pred_values - target_values).mean()

    # R^2 score
    ss_res = ((target_values - pred_values) ** 2).sum()
    ss_tot = ((target_values - target_values.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse,
        'n_samples': len(pred_values)
    }

    if pred_std is not None:
        std_values = pred_std[mask_bool].detach().cpu().numpy().flatten()
        metrics['mean_uncertainty'] = std_values.mean()

    return metrics


if __name__ == "__main__":
    # Test ConvCNP
    model = GEDIConvCNP(
        embedding_channels=128,
        feature_dim=128,
        base_channels=32,
        unet_depth=3
    )

    # Test input
    B, H, W = 2, 256, 256
    tile_embedding = torch.randn(B, 128, H, W)
    context_agbd = torch.randn(B, 1, H, W)
    context_mask = torch.randint(0, 2, (B, 1, H, W)).float()

    # Forward
    pred_mean, pred_log_var = model(tile_embedding, context_agbd, context_mask)

    print(f"Input shapes:")
    print(f"  tile_embedding: {tile_embedding.shape}")
    print(f"  context_agbd: {context_agbd.shape}")
    print(f"  context_mask: {context_mask.shape}")
    print(f"\nOutput shapes:")
    print(f"  pred_mean: {pred_mean.shape}")
    print(f"  pred_log_var: {pred_log_var.shape}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
