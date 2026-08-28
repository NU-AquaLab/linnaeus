"""Configuration management for Linnaeus."""

from pathlib import Path
from typing import Optional

from .settings import Config, LLMConfig, OpenAIConfig, get_config
from .two_stage_config import ConfigLoader, LinnaeusConfig


def load_config(config_path: Optional[Path] = None) -> LinnaeusConfig:
    """Load full Linnaeus configuration from YAML + env vars."""
    loader = ConfigLoader(config_path=config_path)
    return loader.load_config()


__all__ = [
    "Config",
    "LLMConfig",
    "LinnaeusConfig",
    "OpenAIConfig",
    "get_config",
    "load_config",
]
