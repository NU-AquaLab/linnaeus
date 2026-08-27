"""
Configuration management for Linneaus.

This module provides centralized configuration management using YAML files
and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration.

    Works with any OpenAI-compatible API.  Set ``base_url`` to point at
    alternative providers::

        https://api.anthropic.com/v1/   # Claude (Anthropic)
        http://localhost:11434/v1        # Ollama (local)
        https://api.together.xyz/v1      # Together AI
        https://api.fireworks.ai/inference/v1  # Fireworks AI
        https://api.groq.com/openai/v1   # Groq
        http://your-server:8000/v1       # vLLM (self-hosted)
    """

    api_key: Optional[str] = Field(None, description="API key for the LLM provider")
    base_url: Optional[str] = Field(
        None,
        description="Base URL for OpenAI-compatible API (None = default OpenAI)",
    )
    org_id: Optional[str] = Field(None, description="Organization ID (OpenAI-specific)")
    default_model: str = Field("gpt-4o-mini", description="Default model to use")
    temperature: float = Field(0.0001, description="Temperature for model inference")
    batch_size: int = Field(10, description="Batch size for processing")
    max_concurrent_fine_tunes: int = Field(
        3, description="Max concurrent fine-tuning jobs"
    )
    daily_fine_tune_limit: int = Field(10, description="Daily fine-tuning limit")


# Backward-compatible alias
OpenAIConfig = LLMConfig


class DataConfig(BaseModel):
    """Data management configuration."""

    data_dir: Path = Field(Path("./data"), description="Base data directory")
    raw_data_dir: Path = Field(
        Path("./data/raw_data"), description="Raw data directory"
    )
    processed_data_dir: Path = Field(
        Path("./data/processed_data"), description="Processed data directory"
    )
    labeled_data_dir: Path = Field(
        Path("./data/labeled_data"), description="Labeled data directory"
    )
    cache_expiry_hours: int = Field(24, description="Cache expiry time in hours")

    def model_post_init(self, __context):
        """Ensure all directories are absolute paths."""
        self.data_dir = self.data_dir.resolve()
        self.raw_data_dir = self.raw_data_dir.resolve()
        self.processed_data_dir = self.processed_data_dir.resolve()
        self.labeled_data_dir = self.labeled_data_dir.resolve()


class APIConfig(BaseModel):
    """External API configuration."""

    ipinfo_token: Optional[str] = Field(None, description="IPinfo API token")
    asrank_url: str = Field(
        "https://api.asrank.caida.org/v2/graphql", description="ASRank GraphQL API URL"
    )
    peeringdb_base_url: str = Field(
        "https://publicdata.caida.org/datasets/peeringdb",
        description="PeeringDB data URL",
    )
    apnic_aspop_url: str = Field(
        "https://stats.labs.apnic.net/cgi-bin/aspop", description="APNIC ASPOP data URL"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field("INFO", description="Log level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format"
    )
    file_path: Optional[Path] = Field(None, description="Log file path")


class Config(BaseModel):
    """Main application configuration."""

    environment: str = Field("development", description="Application environment")
    openai: OpenAIConfig
    data: DataConfig = Field(default_factory=DataConfig)
    apis: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from YAML file and environment variables.

        Parameters
        ----------
        config_path : Path, optional
            Path to configuration YAML file. If None, looks for config.yaml
            in the current directory or project root.

        Returns
        -------
        Config
            Configuration instance.
        """
        # Load environment variables
        load_dotenv()

        # Find config file
        if config_path is None:
            config_path = Path("config.yaml")
            if not config_path.exists():
                config_path = Path("config") / "config.yaml"

        # Load YAML config if it exists
        config_data = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

        # Override with environment variables
        config_data = cls._apply_env_overrides(config_data)

        return cls(**config_data)

    @staticmethod
    def _apply_env_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Accept 'llm_provider' as YAML alias for 'openai'
        if "llm_provider" in config_data and "openai" not in config_data:
            config_data["openai"] = config_data.pop("llm_provider")
        elif "llm_provider" in config_data:
            config_data.pop("llm_provider")

        # LLM / OpenAI configuration
        openai_config = config_data.setdefault("openai", {})
        # LLM_API_KEY takes priority over OPENAI_API_KEY
        if api_key := os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
            openai_config["api_key"] = api_key
        if base_url := os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"):
            openai_config["base_url"] = base_url
        if org_id := os.getenv("OPENAI_ORG_ID"):
            openai_config["org_id"] = org_id
        if model := os.getenv("DEFAULT_MODEL"):
            openai_config["default_model"] = model
        if temp := os.getenv("TEMPERATURE"):
            openai_config["temperature"] = float(temp)
        if batch_size := os.getenv("BATCH_SIZE"):
            openai_config["batch_size"] = int(batch_size)
        if max_ft := os.getenv("MAX_CONCURRENT_FINE_TUNES"):
            openai_config["max_concurrent_fine_tunes"] = int(max_ft)
        if daily_ft := os.getenv("DAILY_FINE_TUNE_LIMIT"):
            openai_config["daily_fine_tune_limit"] = int(daily_ft)

        # API configuration
        apis_config = config_data.setdefault("apis", {})
        if ipinfo_token := os.getenv("IPINFO_TOKEN"):
            apis_config["ipinfo_token"] = ipinfo_token

        # Data configuration
        data_config = config_data.setdefault("data", {})
        if data_dir := os.getenv("DEFAULT_DATA_DIR"):
            data_config["data_dir"] = Path(data_dir)
        if cache_expiry := os.getenv("CACHE_EXPIRY_HOURS"):
            data_config["cache_expiry_hours"] = int(cache_expiry)

        # Logging configuration
        logging_config = config_data.setdefault("logging", {})
        if log_level := os.getenv("LOG_LEVEL"):
            logging_config["level"] = log_level

        # Environment
        if env := os.getenv("ENVIRONMENT"):
            config_data["environment"] = env

        return config_data


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get the global configuration instance.

    Parameters
    ----------
    config_path : Path, optional
        Path to configuration file. Only used on first call.

    Returns
    -------
    Config
        Global configuration instance.
    """
    global _config
    if _config is None:
        _config = Config.from_file(config_path)
    return _config
