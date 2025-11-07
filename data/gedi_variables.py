"""
GEDI Variable Configurations

Provides recommended settings for different GEDI variables across L2A, L2B, and L4A products.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class GEDIVariableConfig:
    """Configuration for a GEDI variable."""
    name: str
    """Variable name in gediDB"""

    product: str
    """GEDI product (L2A, L2B, L4A)"""

    description: str
    """Human-readable description"""

    unit: str
    """Units of measurement"""

    recommended_scale: float
    """Recommended normalization scale (~95th percentile)"""

    log_transform: bool
    """Whether to apply log(1+x) transform"""

    typical_range: tuple
    """Typical (min, max) range for the variable"""


# Available GEDI variables with recommended settings
GEDI_VARIABLES: Dict[str, GEDIVariableConfig] = {
    'agbd': GEDIVariableConfig(
        name='agbd',
        product='L4A',
        description='Aboveground Biomass Density',
        unit='Mg/ha',
        recommended_scale=200.0,
        log_transform=True,
        typical_range=(0, 500)
    ),
    'rh98': GEDIVariableConfig(
        name='rh98',
        product='L2A',
        description='98th percentile relative height',
        unit='m',
        recommended_scale=50.0,
        log_transform=True,
        typical_range=(0, 70)
    ),
    'rh100': GEDIVariableConfig(
        name='rh100',
        product='L2A',
        description='Maximum canopy height',
        unit='m',
        recommended_scale=50.0,
        log_transform=True,
        typical_range=(0, 80)
    ),
    'cover': GEDIVariableConfig(
        name='cover',
        product='L2B',
        description='Total canopy cover',
        unit='%',
        recommended_scale=100.0,
        log_transform=False,  # Already bounded 0-100%
        typical_range=(0, 100)
    ),
    'pai': GEDIVariableConfig(
        name='pai',
        product='L2B',
        description='Plant Area Index',
        unit='m²/m²',
        recommended_scale=10.0,
        log_transform=False,
        typical_range=(0, 12)
    ),
    'fhd_normal': GEDIVariableConfig(
        name='fhd_normal',
        product='L2B',
        description='Foliage Height Diversity (normalized)',
        unit='dimensionless',
        recommended_scale=1.0,
        log_transform=False,
        typical_range=(0, 1)
    ),
}


def get_variable_config(variable_name: str) -> GEDIVariableConfig:
    """
    Get configuration for a GEDI variable.

    Args:
        variable_name: Name of the GEDI variable

    Returns:
        GEDIVariableConfig with recommended settings

    Raises:
        KeyError: If variable is not recognized
    """
    if variable_name not in GEDI_VARIABLES:
        available = ', '.join(GEDI_VARIABLES.keys())
        raise KeyError(
            f"Unknown GEDI variable '{variable_name}'. "
            f"Available variables: {available}"
        )
    return GEDI_VARIABLES[variable_name]


def list_available_variables() -> List[str]:
    """Get list of available GEDI variables."""
    return list(GEDI_VARIABLES.keys())


def print_variable_info(variable_name: str = None):
    """
    Print information about GEDI variables.

    Args:
        variable_name: Specific variable to show, or None for all
    """
    if variable_name:
        var_config = get_variable_config(variable_name)
        print(f"\nGEDI Variable: {var_config.name}")
        print(f"Product: {var_config.product}")
        print(f"Description: {var_config.description}")
        print(f"Unit: {var_config.unit}")
        print(f"Typical Range: {var_config.typical_range[0]}-{var_config.typical_range[1]} {var_config.unit}")
        print(f"Recommended Scale: {var_config.recommended_scale}")
        print(f"Log Transform: {var_config.log_transform}")
    else:
        print("\nAvailable GEDI Variables:")
        print("-" * 80)
        for name, config in GEDI_VARIABLES.items():
            print(f"{name:15} | {config.product:5} | {config.description:40} | {config.unit}")


if __name__ == '__main__':
    # Print all variables
    print_variable_info()

    print("\n" + "=" * 80)
    print("\nExample: AGBD configuration")
    print_variable_info('agbd')
