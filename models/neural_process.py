import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EmbeddingEncoder(nn.Module):
    """Encode GeoTessera embedding patches into feature vectors."""

    def __init__(
        self,
        patch_size: int = 3,
        in_channels: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        """
        Initialize embedding encoder.

        Args:
            patch_size: Size of embedding patch (e.g., 3x3)
            in_channels: Number of embedding channels (128 for GeoTessera)
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super().__init__()

        # CNN to process spatial structure of embedding patch
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embedding patches.

        Args:
            embeddings: (batch, patch_size, patch_size, channels)

        Returns:
            Feature vectors (batch, output_dim)
        """
        # Reshape to (batch, channels, height, width)
        x = embeddings.permute(0, 3, 1, 2)

        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Global pooling
        x = self.pool(x).squeeze(-1).squeeze(-1)

        # Projection
        x = self.fc(x)

        return x


class ContextEncoder(nn.Module):
    """Encode context points (coord + embedding + agbd) into representations."""

    def __init__(
        self,
        coord_dim: int = 2,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        """
        Initialize context encoder.

        Args:
            coord_dim: Coordinate dimension (2 for lon/lat)
            embedding_dim: Embedding feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output representation dimension
        """
        super().__init__()

        input_dim = coord_dim + embedding_dim + 1  # coords + embedding + agbd

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        coords: torch.Tensor,
        embedding_features: torch.Tensor,
        agbd: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode context points.

        Args:
            coords: (batch, coord_dim)
            embedding_features: (batch, embedding_dim)
            agbd: (batch, 1)

        Returns:
            Representations (batch, output_dim)
        """
        x = torch.cat([coords, embedding_features, agbd], dim=-1)
        return self.mlp(x)


class Decoder(nn.Module):
    """Decode query point + context representation to AGBD prediction."""

    def __init__(
        self,
        coord_dim: int = 2,
        embedding_dim: int = 128,
        context_dim: int = 128,
        hidden_dim: int = 256,
        output_uncertainty: bool = True
    ):
        """
        Initialize decoder.

        Args:
            coord_dim: Coordinate dimension
            embedding_dim: Embedding feature dimension
            context_dim: Context representation dimension
            hidden_dim: Hidden layer dimension
            output_uncertainty: Whether to output uncertainty estimate
        """
        super().__init__()

        self.output_uncertainty = output_uncertainty
        input_dim = coord_dim + embedding_dim + context_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output heads
        self.mean_head = nn.Linear(hidden_dim, 1)

        if output_uncertainty:
            self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        query_coords: torch.Tensor,
        query_embedding_features: torch.Tensor,
        context_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode query points.

        Args:
            query_coords: (batch, coord_dim)
            query_embedding_features: (batch, embedding_dim)
            context_repr: (batch, context_dim)

        Returns:
            (mean, log_variance) predictions, where log_variance is None if not output_uncertainty
        """
        x = torch.cat([query_coords, query_embedding_features, context_repr], dim=-1)
        x = self.mlp(x)

        mean = self.mean_head(x)

        if self.output_uncertainty:
            log_var = self.log_var_head(x)
            # Clamp log_var to prevent numerical instability
            # Range: variance between ~0.001 and ~150
            log_var = torch.clamp(log_var, min=-7, max=5)
            return mean, log_var
        else:
            return mean, None


class GEDINeuralProcess(nn.Module):
    """
    Neural Process for GEDI AGB interpolation with foundation model embeddings.

    Architecture:
    1. Encode embedding patches to feature vectors
    2. Encode context points (coord + embedding feature + agbd)
    3. Aggregate context representations (mean pooling)
    4. Decode query points to AGBD predictions with uncertainty
    """

    def __init__(
        self,
        patch_size: int = 3,
        embedding_channels: int = 128,
        embedding_feature_dim: int = 128,
        context_repr_dim: int = 128,
        hidden_dim: int = 256,
        output_uncertainty: bool = True
    ):
        """
        Initialize Neural Process.

        Args:
            patch_size: Size of embedding patches
            embedding_channels: Number of channels in embeddings
            embedding_feature_dim: Dimension of encoded embedding features
            context_repr_dim: Dimension of context representations
            hidden_dim: Hidden layer dimension
            output_uncertainty: Whether to predict uncertainty
        """
        super().__init__()

        self.output_uncertainty = output_uncertainty

        # Embedding encoder (shared for context and query)
        self.embedding_encoder = EmbeddingEncoder(
            patch_size=patch_size,
            in_channels=embedding_channels,
            hidden_dim=hidden_dim,
            output_dim=embedding_feature_dim
        )

        # Context encoder
        self.context_encoder = ContextEncoder(
            coord_dim=2,
            embedding_dim=embedding_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=context_repr_dim
        )

        # Decoder
        self.decoder = Decoder(
            coord_dim=2,
            embedding_dim=embedding_feature_dim,
            context_dim=context_repr_dim,
            hidden_dim=hidden_dim,
            output_uncertainty=output_uncertainty
        )

    def forward(
        self,
        context_coords: torch.Tensor,
        context_embeddings: torch.Tensor,
        context_agbd: torch.Tensor,
        query_coords: torch.Tensor,
        query_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            context_coords: (n_context, 2)
            context_embeddings: (n_context, patch_size, patch_size, channels)
            context_agbd: (n_context, 1)
            query_coords: (n_query, 2)
            query_embeddings: (n_query, patch_size, patch_size, channels)

        Returns:
            (predicted_agbd, log_variance) for query points
            Shape: (n_query, 1) for each
        """
        # Encode embeddings
        context_emb_features = self.embedding_encoder(context_embeddings)
        query_emb_features = self.embedding_encoder(query_embeddings)

        # Encode context points
        context_repr = self.context_encoder(
            context_coords,
            context_emb_features,
            context_agbd
        )

        # Aggregate context (mean pooling)
        aggregated_context = context_repr.mean(dim=0, keepdim=True)

        # Expand to match query batch size
        aggregated_context = aggregated_context.expand(query_coords.shape[0], -1)

        # Decode query points
        pred_mean, pred_log_var = self.decoder(
            query_coords,
            query_emb_features,
            aggregated_context
        )

        return pred_mean, pred_log_var

    def predict(
        self,
        context_coords: torch.Tensor,
        context_embeddings: torch.Tensor,
        context_agbd: torch.Tensor,
        query_coords: torch.Tensor,
        query_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty.

        Returns:
            (mean, std) predictions
        """
        pred_mean, pred_log_var = self.forward(
            context_coords,
            context_embeddings,
            context_agbd,
            query_coords,
            query_embeddings
        )

        if pred_log_var is not None:
            pred_std = torch.exp(0.5 * pred_log_var)
        else:
            pred_std = torch.zeros_like(pred_mean)

        return pred_mean, pred_std


def neural_process_loss(
    pred_mean: torch.Tensor,
    pred_log_var: Optional[torch.Tensor],
    target: torch.Tensor
) -> torch.Tensor:
    """
    Negative log-likelihood loss for Neural Process.

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


def compute_metrics(
    pred_mean: torch.Tensor,
    pred_std: Optional[torch.Tensor],
    target: torch.Tensor
) -> dict:
    """
    Compute evaluation metrics.

    Args:
        pred_mean: Predicted means
        pred_std: Predicted standard deviations (or None)
        target: Target values

    Returns:
        Dictionary of metrics
    """
    pred_mean = pred_mean.detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()

    mse = ((pred_mean - target) ** 2).mean()
    rmse = mse ** 0.5
    mae = abs(pred_mean - target).mean()

    # R^2 score
    ss_res = ((target - pred_mean) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse
    }

    if pred_std is not None:
        pred_std = pred_std.detach().cpu().numpy().flatten()
        metrics['mean_uncertainty'] = pred_std.mean()

    return metrics
