"""Configuration management for Linneaus."""

from pathlib import Path
from typing import Optional

from .settings import Config, LLMConfig, OpenAIConfig, get_config
from .two_stage_config import ConfigLoader, LinneausConfig


def load_config(config_path: Optional[Path] = None) -> LinneausConfig:
    """Load full Linneaus configuration from YAML + env vars."""
    loader = ConfigLoader(config_path=config_path)
    return loader.load_config()


__all__ = [
    "Config",
    "LLMConfig",
    "LinneausConfig",
    "OpenAIConfig",
    "get_config",
    "load_config",
]
