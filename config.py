"""
Configuration system for GEDI Neural Process experiments.

Provides type-safe, documented configuration with JSON serialization.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Neural Process model architecture configuration."""

    # Embedding parameters
    patch_size: int = 3
    """Spatial patch size for GeoTessera embeddings (e.g., 3 = 3x3 = ~30m patch)"""

    embedding_channels: int = 128
    """GeoTessera embedding dimension (fixed by pretrained model)"""

    embedding_feature_dim: int = 128
    """Dimension after embedding encoder (flattened patch -> features)"""

    # Context representation
    context_repr_dim: int = 128
    """Dimension of aggregated context representation"""

    use_attention: bool = True
    """Use attention-based context aggregation (vs mean pooling)"""

    num_attention_heads: int = 4
    """Number of attention heads for context aggregation"""

    # Decoder
    hidden_dim: int = 512
    """Hidden layer dimension in decoder MLP"""

    output_uncertainty: bool = True
    """Output predictive uncertainty (mean + variance)"""

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    # GEDI target variable
    target_variable: str = 'agbd'
    """GEDI target variable to predict. Options:
    - 'agbd': Aboveground biomass density (Mg/ha) from L4A
    - 'rh98': 98th percentile relative height (m) from L2A
    - 'cover': Total canopy cover (%) from L2B
    - 'pai': Plant Area Index (m²/m²) from L2B
    - 'fhd': Foliage height diversity from L2B
    """

    target_scale: float = 200.0
    """Normalization scale for target variable.
    Default 200.0 for AGBD (~95th percentile).
    Adjust for other variables (e.g., 50.0 for rh98, 100.0 for cover)
    """

    log_transform_target: bool = True
    """Apply log(1+x) transform to target variable (recommended for AGBD)"""

    # Spatial preprocessing
    min_shots_per_tile: int = 10
    """Minimum GEDI shots required per tile"""

    max_shots_per_tile: Optional[int] = None
    """Maximum shots per tile (subsample if exceeded). None = no limit"""

    tile_size: float = 0.1
    """Tile size in degrees (~11km at equator)"""

    # Context/target split
    context_ratio_range: Tuple[float, float] = (0.3, 0.7)
    """Range of context/total ratios for Neural Process training"""

    # Coordinate normalization
    normalize_coords: bool = True
    """Normalize coordinates using global bounds"""

    # Embeddings
    embedding_year: int = 2024
    """Year of GeoTessera embeddings"""

    def to_dict(self):
        """Convert to dictionary."""
        d = asdict(self)
        # Convert tuple to list for JSON serialization
        if 'context_ratio_range' in d:
            d['context_ratio_range'] = list(d['context_ratio_range'])
        return d

    @classmethod
    def from_dict(cls, d: dict):
        """Create from dictionary."""
        # Convert list back to tuple
        if 'context_ratio_range' in d and isinstance(d['context_ratio_range'], list):
            d['context_ratio_range'] = tuple(d['context_ratio_range'])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""

    augment_coords: bool = True
    """Add random noise to coordinates during training"""

    coord_noise_std: float = 0.01
    """Standard deviation for coordinate noise (~1km at global scale)"""

    augment_embeddings: bool = False
    """Add noise to embeddings (not typically recommended)"""

    embedding_noise_std: float = 0.0
    """Standard deviation for embedding noise"""

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    # Optimization
    learning_rate: float = 5e-4
    """Initial learning rate (Adam optimizer)"""

    batch_size: int = 16
    """Number of tiles per batch"""

    epochs: int = 100
    """Maximum number of training epochs"""

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    """Use ReduceLROnPlateau scheduler"""

    lr_scheduler_patience: int = 5
    """Epochs without improvement before reducing LR"""

    lr_scheduler_factor: float = 0.5
    """Multiplicative factor for LR reduction"""

    # Early stopping
    early_stopping: bool = True
    """Enable early stopping"""

    early_stopping_patience: int = 15
    """Epochs without improvement before stopping"""

    # Gradient clipping
    gradient_clip_norm: float = 1.0
    """Max norm for gradient clipping (prevents exploding gradients)"""

    # Checkpointing
    save_every: int = 10
    """Save checkpoint every N epochs"""

    # Data splits
    val_ratio: float = 0.15
    """Validation set ratio"""

    test_ratio: float = 0.15
    """Test set ratio"""

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration combining all sub-configs."""

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment metadata
    experiment_name: str = "gedi_cnp"
    """Descriptive name for this experiment"""

    seed: int = 42
    """Random seed for reproducibility"""

    # Paths
    output_dir: str = "./outputs"
    """Base directory for outputs"""

    cache_dir: str = "./cache"
    """Directory for caching tiles and embeddings"""

    # Region (set at runtime)
    region_bbox: Optional[Tuple[float, float, float, float]] = None
    """Region bounding box (lon_min, lat_min, lon_max, lat_max)"""

    start_time: str = "2019-01-01"
    """Start date for GEDI data (YYYY-MM-DD)"""

    end_time: str = "2023-12-31"
    """End date for GEDI data (YYYY-MM-DD)"""

    # Runtime
    device: str = "cuda"
    """Device to use (cuda/cpu)"""

    num_workers: int = 4
    """Number of data loading workers"""

    # Global coordinate bounds (computed during training)
    global_bounds: Optional[Tuple[float, float, float, float]] = None
    """Global coordinate bounds (lon_min, lat_min, lon_max, lat_max)"""

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        d = {
            'model': self.model.to_dict(),
            'data': self.data.to_dict(),
            'augmentation': self.augmentation.to_dict(),
            'training': self.training.to_dict(),
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'output_dir': self.output_dir,
            'cache_dir': self.cache_dir,
            'region_bbox': list(self.region_bbox) if self.region_bbox else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'device': self.device,
            'num_workers': self.num_workers,
            'global_bounds': list(self.global_bounds) if self.global_bounds else None
        }
        return d

    @classmethod
    def from_dict(cls, d: dict):
        """Create from dictionary."""
        # Handle nested configs
        model = ModelConfig.from_dict(d.get('model', {}))
        data = DataConfig.from_dict(d.get('data', {}))
        augmentation = AugmentationConfig.from_dict(d.get('augmentation', {}))
        training = TrainingConfig.from_dict(d.get('training', {}))

        # Convert lists back to tuples
        region_bbox = tuple(d['region_bbox']) if d.get('region_bbox') else None
        global_bounds = tuple(d['global_bounds']) if d.get('global_bounds') else None

        return cls(
            model=model,
            data=data,
            augmentation=augmentation,
            training=training,
            experiment_name=d.get('experiment_name', 'gedi_cnp'),
            seed=d.get('seed', 42),
            output_dir=d.get('output_dir', './outputs'),
            cache_dir=d.get('cache_dir', './cache'),
            region_bbox=region_bbox,
            start_time=d.get('start_time', '2019-01-01'),
            end_time=d.get('end_time', '2023-12-31'),
            device=d.get('device', 'cuda'),
            num_workers=d.get('num_workers', 4),
            global_bounds=global_bounds
        )

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)

    def get_output_dir(self) -> Path:
        """Get output directory path."""
        return Path(self.output_dir) / self.experiment_name


# Preset configurations for different targets
def get_agbd_config(**kwargs) -> ExperimentConfig:
    """Get configuration for AGBD prediction (default)."""
    config = ExperimentConfig(**kwargs)
    config.data.target_variable = 'agbd'
    config.data.target_scale = 200.0
    config.data.log_transform_target = True
    return config


def get_height_config(**kwargs) -> ExperimentConfig:
    """Get configuration for canopy height (RH98) prediction."""
    config = ExperimentConfig(**kwargs)
    config.data.target_variable = 'rh98'
    config.data.target_scale = 50.0  # ~95th percentile height
    config.data.log_transform_target = True
    return config


def get_cover_config(**kwargs) -> ExperimentConfig:
    """Get configuration for canopy cover prediction."""
    config = ExperimentConfig(**kwargs)
    config.data.target_variable = 'cover'
    config.data.target_scale = 100.0  # Cover is 0-100%
    config.data.log_transform_target = False  # Cover is already bounded
    return config


def get_pai_config(**kwargs) -> ExperimentConfig:
    """Get configuration for Plant Area Index prediction."""
    config = ExperimentConfig(**kwargs)
    config.data.target_variable = 'pai'
    config.data.target_scale = 10.0  # PAI typically 0-10
    config.data.log_transform_target = False
    return config


if __name__ == '__main__':
    # Example: Create and save a config
    config = ExperimentConfig(
        experiment_name='test_experiment',
        region_bbox=(-180, -90, 180, 90)
    )

    print("Example Configuration:")
    print(json.dumps(config.to_dict(), indent=2))

    # Test save/load
    config.save('test_config.json')
    loaded = ExperimentConfig.load('test_config.json')
    print("\nConfiguration loaded successfully!")

    # Clean up
    Path('test_config.json').unlink()
