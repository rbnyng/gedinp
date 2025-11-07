import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class FourierCoordinateEncoding(nn.Module):
    """
    Fourier feature encoding for coordinates.

    Transforms 2D coordinates into high-dimensional features using sinusoidal functions
    at multiple frequency scales. This helps the model capture multi-scale spatial patterns.
    """

    def __init__(self, num_frequencies: int = 10, include_original: bool = True):
        """
        Initialize Fourier encoding.

        Args:
            num_frequencies: Number of frequency scales to use
            include_original: Whether to include original coordinates in output
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_original = include_original

        # Create frequency scales: 2^0, 2^1, ..., 2^(num_frequencies-1)
        # These will be used to create sinusoidal features at different scales
        frequencies = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('frequencies', frequencies)

        # Output dimension: original coords (2) + sin/cos for each frequency (2 * 2 * num_frequencies)
        self.output_dim = (2 if include_original else 0) + 4 * num_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates with Fourier features.

        Args:
            coords: (batch, 2) normalized coordinates in [0, 1]

        Returns:
            Fourier features (batch, output_dim)
        """
        # coords: (batch, 2)
        # frequencies: (num_frequencies,)

        # Compute freq * coords for all frequencies
        # Shape: (batch, 2, num_frequencies)
        scaled_coords = coords.unsqueeze(-1) * self.frequencies.unsqueeze(0).unsqueeze(0)

        # Apply sin and cos
        # Shape: (batch, 2, num_frequencies) each
        sin_features = torch.sin(2 * math.pi * scaled_coords)
        cos_features = torch.cos(2 * math.pi * scaled_coords)

        # Flatten frequency dimension: (batch, 2 * num_frequencies) each
        sin_features = sin_features.reshape(coords.shape[0], -1)
        cos_features = cos_features.reshape(coords.shape[0], -1)

        if self.include_original:
            # Concatenate: original + sin + cos
            return torch.cat([coords, sin_features, cos_features], dim=-1)
        else:
            return torch.cat([sin_features, cos_features], dim=-1)


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

        # Deeper CNN to process spatial structure of embedding patch
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        # Residual connection for first block
        self.residual_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

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

        # First block with residual
        identity = self.residual_proj(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)) + identity)

        # Third conv layer
        x = F.relu(self.bn3(self.conv3(x)) + x)

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

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

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

        # First layer
        x = F.relu(self.ln1(self.fc1(x)))

        # Second layer with residual
        identity = x
        x = F.relu(self.ln2(self.fc2(x)) + identity)

        # Third layer with residual
        x = F.relu(self.ln3(self.fc3(x)) + x)

        # Output projection
        x = self.fc_out(x)

        return x


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

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

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

        # First layer
        x = F.relu(self.ln1(self.fc1(x)))

        # Second layer with residual
        identity = x
        x = F.relu(self.ln2(self.fc2(x)) + identity)

        # Third layer with residual
        x = F.relu(self.ln3(self.fc3(x)) + x)

        mean = self.mean_head(x)

        if self.output_uncertainty:
            log_var = self.log_var_head(x)
            # Clamp log_var to prevent numerical instability
            # Range: variance between ~0.001 and ~150
            log_var = torch.clamp(log_var, min=-7, max=5)
            return mean, log_var
        else:
            return mean, None


class AttentionAggregator(nn.Module):
    """Attention-based aggregation of context representations with distance bias."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_distance_bias: bool = False,
        distance_bias_scale: float = 1.0
    ):
        """
        Initialize attention aggregator.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_distance_bias: Whether to add distance-based bias to attention
            distance_bias_scale: Initial scale for distance bias (learnable)
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.use_distance_bias = use_distance_bias

        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable parameters for distance bias
        if use_distance_bias:
            # Scale parameter for distance (similar to temperature in attention)
            # Initialized to provide reasonable bias, then learned during training
            self.log_distance_scale = nn.Parameter(
                torch.log(torch.tensor(distance_bias_scale))
            )
            # Optional: per-head scaling for distance bias
            self.distance_bias_per_head = nn.Parameter(
                torch.ones(num_heads)
            )

    def forward(
        self,
        query_repr: torch.Tensor,
        context_repr: torch.Tensor,
        query_coords: Optional[torch.Tensor] = None,
        context_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate context using attention with optional distance bias.

        Args:
            query_repr: Query representations (n_query, dim)
            context_repr: Context representations (n_context, dim)
            query_coords: Query coordinates (n_query, 2) - required if use_distance_bias
            context_coords: Context coordinates (n_context, 2) - required if use_distance_bias

        Returns:
            Aggregated context for each query (n_query, dim)
        """
        # Expand dimensions for batch processing
        query = query_repr.unsqueeze(0)  # (1, n_query, dim)
        context = context_repr.unsqueeze(0)  # (1, n_context, dim)

        # Compute distance-based attention mask if enabled
        attn_mask = None
        if self.use_distance_bias:
            if query_coords is None or context_coords is None:
                raise ValueError("Coordinates required when use_distance_bias=True")

            # Compute pairwise distances: (n_query, n_context)
            # Using Euclidean distance in normalized coordinate space
            distances = self._compute_pairwise_distances(query_coords, context_coords)

            # Convert distances to attention bias
            # Negative bias for larger distances (reduces attention weight)
            # Scale is learned: larger scale = distance matters more
            distance_scale = torch.exp(self.log_distance_scale)
            distance_bias = -distance_scale * distances  # (n_query, n_context)

            # Optionally scale per head
            # Reshape for multi-head: (num_heads, n_query, n_context)
            distance_bias = distance_bias.unsqueeze(0).expand(self.num_heads, -1, -1)
            distance_bias = distance_bias * self.distance_bias_per_head.view(-1, 1, 1)

            # Reshape for batch dimension: (1 * num_heads, n_query, n_context)
            attn_mask = distance_bias.reshape(self.num_heads, query_coords.shape[0], context_coords.shape[0])

        # Apply cross-attention with distance bias
        attended, _ = self.attention(
            query, context, context,
            attn_mask=attn_mask,
            need_weights=False
        )

        # Apply dropout and residual connection
        attended = self.dropout(attended)
        output = self.norm(attended.squeeze(0) + query_repr)

        return output

    def _compute_pairwise_distances(
        self,
        query_coords: torch.Tensor,
        context_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances between query and context points.

        Args:
            query_coords: (n_query, 2)
            context_coords: (n_context, 2)

        Returns:
            Distances (n_query, n_context)
        """
        # Efficient distance computation using broadcasting
        # query: (n_query, 1, 2), context: (1, n_context, 2)
        diff = query_coords.unsqueeze(1) - context_coords.unsqueeze(0)  # (n_query, n_context, 2)
        distances = torch.norm(diff, dim=-1)  # (n_query, n_context)
        return distances


class GEDINeuralProcess(nn.Module):
    """
    Neural Process for GEDI AGB interpolation with foundation model embeddings.

    Architecture:
    1. Encode embedding patches to feature vectors
    2. Encode context points (coord + embedding feature + agbd)
    3. Aggregate context representations with attention mechanism
    4. Decode query points to AGBD predictions with uncertainty
    """

    def __init__(
        self,
        patch_size: int = 3,
        embedding_channels: int = 128,
        embedding_feature_dim: int = 128,
        context_repr_dim: int = 128,
        hidden_dim: int = 256,
        output_uncertainty: bool = True,
        use_attention: bool = True,
        num_attention_heads: int = 4,
        use_fourier_encoding: bool = False,
        fourier_frequencies: int = 10,
        use_distance_bias: bool = False,
        distance_bias_scale: float = 1.0
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
            use_attention: Use attention for context aggregation (vs mean pooling)
            num_attention_heads: Number of attention heads
            use_fourier_encoding: Use Fourier features for coordinates
            fourier_frequencies: Number of frequency scales for Fourier encoding
            use_distance_bias: Add distance bias to attention mechanism
            distance_bias_scale: Initial scale for distance bias (learnable)
        """
        super().__init__()

        self.output_uncertainty = output_uncertainty
        self.use_attention = use_attention
        self.use_fourier_encoding = use_fourier_encoding
        self.use_distance_bias = use_distance_bias

        # Coordinate encoding
        if use_fourier_encoding:
            self.coord_encoder = FourierCoordinateEncoding(
                num_frequencies=fourier_frequencies,
                include_original=True
            )
            coord_dim = self.coord_encoder.output_dim
        else:
            self.coord_encoder = None
            coord_dim = 2

        # Embedding encoder (shared for context and query)
        self.embedding_encoder = EmbeddingEncoder(
            patch_size=patch_size,
            in_channels=embedding_channels,
            hidden_dim=hidden_dim,
            output_dim=embedding_feature_dim
        )

        # Context encoder
        self.context_encoder = ContextEncoder(
            coord_dim=coord_dim,
            embedding_dim=embedding_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=context_repr_dim
        )

        # Attention aggregator (optional)
        if use_attention:
            self.attention_aggregator = AttentionAggregator(
                dim=context_repr_dim,
                num_heads=num_attention_heads,
                use_distance_bias=use_distance_bias,
                distance_bias_scale=distance_bias_scale
            )
            # Query projection for attention (coord + embedding -> context_repr_dim)
            self.query_proj = nn.Linear(coord_dim + embedding_feature_dim, context_repr_dim)

        # Decoder
        self.decoder = Decoder(
            coord_dim=coord_dim,
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
            context_coords: (n_context, 2) - normalized coordinates
            context_embeddings: (n_context, patch_size, patch_size, channels)
            context_agbd: (n_context, 1)
            query_coords: (n_query, 2) - normalized coordinates
            query_embeddings: (n_query, patch_size, patch_size, channels)

        Returns:
            (predicted_agbd, log_variance) for query points
            Shape: (n_query, 1) for each
        """
        # Store original coordinates for distance computation (if needed)
        context_coords_original = context_coords
        query_coords_original = query_coords

        # Apply Fourier encoding to coordinates if enabled
        if self.use_fourier_encoding:
            context_coords_encoded = self.coord_encoder(context_coords)
            query_coords_encoded = self.coord_encoder(query_coords)
        else:
            context_coords_encoded = context_coords
            query_coords_encoded = query_coords

        # Encode embeddings
        context_emb_features = self.embedding_encoder(context_embeddings)
        query_emb_features = self.embedding_encoder(query_embeddings)

        # Encode context points (with Fourier-encoded or original coords)
        context_repr = self.context_encoder(
            context_coords_encoded,
            context_emb_features,
            context_agbd
        )

        # Aggregate context
        if self.use_attention:
            # Create query representations for attention
            # Use query coords + embeddings as query features
            query_repr = torch.cat([query_coords_encoded, query_emb_features], dim=-1)

            # Project to context dimension
            query_repr_projected = self.query_proj(query_repr)

            # Use attention to aggregate context
            # Pass original (non-Fourier) coordinates for distance computation
            aggregated_context = self.attention_aggregator(
                query_repr_projected,
                context_repr,
                query_coords=query_coords_original if self.use_distance_bias else None,
                context_coords=context_coords_original if self.use_distance_bias else None
            )
        else:
            # Mean pooling (original method)
            aggregated_context = context_repr.mean(dim=0, keepdim=True)
            # Expand to match query batch size
            aggregated_context = aggregated_context.expand(query_coords.shape[0], -1)

        # Decode query points (with Fourier-encoded or original coords)
        pred_mean, pred_log_var = self.decoder(
            query_coords_encoded,
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
