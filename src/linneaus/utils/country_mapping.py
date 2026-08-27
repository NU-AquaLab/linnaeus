"""
Country-RIR registry utilities.

This module provides functions to load and work with the country to
Regional Internet Registry (RIR) mapping data from the official registry.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..models.schemas import CountryInfo, CountryRIRRegistry, RIRInfo


def load_country_rir_registry() -> CountryRIRRegistry:
    """Load the country-RIR registry data from the package resources.

    Returns:
        CountryRIRRegistry: The loaded and validated registry data

    Raises:
        FileNotFoundError: If the registry file is not found
        ValueError: If the registry data is invalid
    """
    # Get the registry file path relative to this module
    registry_file = (
        Path(__file__).parent.parent / "resources" / "country_rir_registry.json"
    )

    if not registry_file.exists():
        raise FileNotFoundError(f"Country-RIR registry file not found: {registry_file}")

    with open(registry_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return CountryRIRRegistry(**data)


def get_rir_for_country(country_code: str) -> Optional[str]:
    """Get the RIR for a given country code.

    Args:
        country_code: ISO 3166-1 alpha-2 country code

    Returns:
        RIR code if found, None otherwise
    """
    registry = load_country_rir_registry()
    return registry.get_rir_for_country(country_code)


def get_countries_by_rir(rir: str) -> List[str]:
    """Get list of country codes for a given RIR.

    Args:
        rir: RIR code (e.g., 'ARIN', 'RIPE NCC', 'APNIC', etc.)

    Returns:
        List of ISO country codes
    """
    registry = load_country_rir_registry()
    return registry.get_countries_by_rir(rir)


def get_countries_by_continent(continent: str) -> List[str]:
    """Get list of country codes for a given continent.

    Args:
        continent: Continent name

    Returns:
        List of ISO country codes
    """
    registry = load_country_rir_registry()
    return registry.get_countries_by_continent(continent)


def get_country_info(country_code: str) -> Optional[CountryInfo]:
    """Get complete information for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code

    Returns:
        CountryInfo object if found, None otherwise
    """
    registry = load_country_rir_registry()
    return registry.get_country_info(country_code)


def get_rir_info(rir: str) -> Optional[RIRInfo]:
    """Get complete information for a Regional Internet Registry.

    Args:
        rir: RIR code (e.g., 'ARIN', 'RIPE NCC', 'APNIC', etc.)

    Returns:
        RIRInfo object if found, None otherwise
    """
    registry = load_country_rir_registry()
    return registry.get_rir_info(rir)


def get_all_rirs() -> List[str]:
    """Get list of all available RIR codes.

    Returns:
        List of RIR codes
    """
    registry = load_country_rir_registry()
    return list(registry.registries.keys())


def get_registry_metadata() -> Dict[str, str]:
    """Get registry metadata information.

    Returns:
        Dictionary with registry metadata
    """
    registry = load_country_rir_registry()
    return registry.metadata
