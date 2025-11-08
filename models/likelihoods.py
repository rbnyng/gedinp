"""
Likelihood functions for Neural Process models.

This module implements different likelihood distributions for modeling
Above-Ground Biomass Density (AGBD) data.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def gaussian_log_nll(
    pred_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood on log-transformed data.

    This is the current default approach: apply log(1+x) transform to AGBD,
    then model with Gaussian likelihood. This implicitly assumes a log-normal
    distribution for the original AGBD values.

    NLL = 0.5 * (log(var) + (target - mean)^2 / var)
        = 0.5 * (log_var + exp(-log_var) * (target - mean)^2)

    Args:
        pred_mean: Predicted means (batch, 1) in log-transformed space
        pred_log_var: Predicted log variances (batch, 1)
        target: Target values (batch, 1) in log-transformed space

    Returns:
        Negative log-likelihood (scalar)
    """
    nll = 0.5 * (
        pred_log_var +
        torch.exp(-pred_log_var) * (target - pred_mean) ** 2
    )
    return nll.mean()


def lognormal_nll(
    pred_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Explicit log-normal negative log-likelihood.

    This is mathematically equivalent to gaussian_log_nll but makes the
    log-normal assumption explicit. The model predicts parameters of the
    log-normal distribution directly.

    For X ~ LogNormal(μ, σ²):
        p(x) = 1/(x*σ*sqrt(2π)) * exp(-(log(x) - μ)² / (2σ²))

    NLL = log(x) + 0.5*log(2π) + 0.5*log(σ²) + (log(x) - μ)² / (2σ²)

    Since we work with log-transformed data, this simplifies to the same
    formula as gaussian_log_nll. This function is provided for clarity
    and documentation purposes.

    Args:
        pred_mean: Predicted log-means (batch, 1) - μ parameter
        pred_log_var: Predicted log variances (batch, 1) - log(σ²) parameter
        target: Target values (batch, 1) in log-transformed space

    Returns:
        Negative log-likelihood (scalar)
    """
    # In log-transformed space, log-normal reduces to Gaussian
    # This is the same as gaussian_log_nll but named explicitly
    nll = 0.5 * (
        pred_log_var +
        torch.exp(-pred_log_var) * (target - pred_mean) ** 2
    )
    return nll.mean()


def gamma_nll(
    pred_mean: torch.Tensor,
    pred_log_concentration: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Gamma negative log-likelihood for positive continuous data.

    The Gamma distribution is another common choice for modeling positive
    continuous variables like biomass. It's parameterized by shape (α) and
    rate (β), or equivalently concentration and rate.

    We use the mean and concentration parameterization:
        mean = α/β
        concentration = α
        rate = α/mean

    For X ~ Gamma(α, β):
        p(x) = (β^α / Γ(α)) * x^(α-1) * exp(-β*x)

    NLL = -log(p(x)) = α*log(mean/α) + log(Γ(α)) - (α-1)*log(x) + α*x/mean

    Args:
        pred_mean: Predicted means (batch, 1) - must be positive
        pred_log_concentration: Predicted log concentration parameter (batch, 1)
            Concentration α controls the shape. Higher α = less variance.
        target: Target values (batch, 1) - must be positive
        eps: Small constant for numerical stability

    Returns:
        Negative log-likelihood (scalar)

    Note:
        The model should predict mean in the ORIGINAL (not log-transformed) space
        when using this likelihood. This requires different data preprocessing.
    """
    # Ensure positive values
    pred_mean = pred_mean.clamp(min=eps)
    target = target.clamp(min=eps)

    # Get concentration parameter (α)
    concentration = torch.exp(pred_log_concentration).clamp(min=eps, max=1e3)

    # Rate parameter: β = α / mean
    rate = concentration / pred_mean

    # Gamma NLL using PyTorch's lgamma (log of gamma function)
    # NLL = -log(p(x))
    #     = -[α*log(β) - log(Γ(α)) + (α-1)*log(x) - β*x]
    #     = log(Γ(α)) - α*log(β) - (α-1)*log(x) + β*x
    nll = (
        torch.lgamma(concentration)
        - concentration * torch.log(rate + eps)
        - (concentration - 1) * torch.log(target + eps)
        + rate * target
    )

    return nll.mean()


def get_likelihood_function(likelihood_type: str):
    """
    Get the appropriate likelihood function based on type.

    Args:
        likelihood_type: One of ['gaussian-log', 'lognormal', 'gamma']

    Returns:
        Likelihood function that takes (pred_mean, pred_params, target)

    Raises:
        ValueError: If likelihood_type is not recognized
    """
    likelihood_functions = {
        'gaussian-log': gaussian_log_nll,
        'lognormal': lognormal_nll,
        'gamma': gamma_nll,
    }

    if likelihood_type not in likelihood_functions:
        raise ValueError(
            f"Unknown likelihood type: {likelihood_type}. "
            f"Must be one of {list(likelihood_functions.keys())}"
        )

    return likelihood_functions[likelihood_type]


def get_likelihood_param_name(likelihood_type: str) -> str:
    """
    Get the name of the second parameter for the likelihood.

    Args:
        likelihood_type: One of ['gaussian-log', 'lognormal', 'gamma']

    Returns:
        Parameter name (e.g., 'log_var', 'log_concentration')
    """
    param_names = {
        'gaussian-log': 'log_var',
        'lognormal': 'log_var',
        'gamma': 'log_concentration',
    }

    if likelihood_type not in param_names:
        raise ValueError(
            f"Unknown likelihood type: {likelihood_type}. "
            f"Must be one of {list(param_names.keys())}"
        )

    return param_names[likelihood_type]


def requires_original_scale(likelihood_type: str) -> bool:
    """
    Check if likelihood requires data in original (non-log-transformed) scale.

    Args:
        likelihood_type: One of ['gaussian-log', 'lognormal', 'gamma']

    Returns:
        True if likelihood expects original scale data, False if log-transformed
    """
    original_scale_likelihoods = {'gamma'}
    return likelihood_type in original_scale_likelihoods
