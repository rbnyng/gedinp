"""General utility functions."""

import numpy as np
try:
    import torch
except ImportError:
    torch = None


def convert_to_serializable(obj):
    """
    Convert numpy types to native Python types for JSON serialization.

    Recursively handles dictionaries, lists, and tuples.

    Args:
        obj: Object to convert (can be numpy types, dict, list, tuple, etc.)

    Returns:
        Serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Sets seed for:
    - NumPy random number generator
    - PyTorch CPU random number generator
    - PyTorch CUDA random number generator (if available)

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
