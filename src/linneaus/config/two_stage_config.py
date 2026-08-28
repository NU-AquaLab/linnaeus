"""
Configuration management for two-stage hierarchical pipeline.

This module handles loading and validation of configuration from both
config.yaml and environment variables, with proper Pydantic validation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SVMConfig(BaseModel):
    """SVM configuration for Stage 1."""

    kernel: str = Field(default="rbf", description="SVM kernel")
    C: float = Field(default=1.0, ge=0.0, description="SVM regularization parameter")
    probability: bool = Field(default=True, description="Enable probability estimates")
    feature_selection: bool = Field(
        default=True, description="Enable feature selection"
    )
    n_features: int = Field(
        default=50, ge=1, description="Number of features to select"
    )


class MetaLearnerConfig(BaseModel):
    """Meta-learner configuration for Stage 1."""

    max_iter: int = Field(default=1000, ge=1, description="Maximum iterations")
    random_state: int = Field(default=42, description="Random state")
    cv_folds: int = Field(default=3, ge=2, description="Cross-validation folds")


class Stage1Config(BaseModel):
    """Configuration for Stage 1 assembled classifier."""

    svm_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="SVM weight in ensemble"
    )
    llm_weight: float = Field(
        default=0.6, ge=0.0, le=1.0, description="LLM weight in ensemble"
    )
    meta_learner: str = Field(
        default="LogisticRegression", description="Meta-learner type"
    )
    parallel_batch_size: int = Field(
        default=10, ge=1, description="Batch size for parallel processing"
    )
    confidence_weight: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Weight in overall confidence"
    )
    svm: SVMConfig = Field(default_factory=SVMConfig, description="SVM configuration")
    meta_learner_params: MetaLearnerConfig = Field(
        default_factory=MetaLearnerConfig, description="Meta-learner params"
    )

    @field_validator("svm_weight", "llm_weight")
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Validate that weights sum to reasonable value."""
        # This is a simplified validation - full validation happens at model level
        if v < 0.0 or v > 1.0:
            raise ValueError("Weights must be between 0.0 and 1.0")
        return v


class Stage2Config(BaseModel):
    """Configuration for Stage 2 sublevel classifier."""

    confidence_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight in overall confidence"
    )
    batch_size: int = Field(default=10, ge=1, description="Batch size for processing")
    temperature: float = Field(
        default=0.0001, ge=0.0, le=2.0, description="LLM temperature"
    )
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    category_models: Dict[str, str] = Field(
        default_factory=dict, description="Category to model ID mapping"
    )

    @field_validator("category_models")
    @classmethod
    def validate_category_models(cls, v):
        """Validate that category models are provided."""
        if not v:
            logger.warning("No category models configured for Stage 2")
        return v


class GeneralConfig(BaseModel):
    """General pipeline configuration."""

    max_workers: int = Field(default=4, ge=1, description="Maximum parallel workers")
    enable_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )
    graceful_degradation: bool = Field(
        default=True, description="Enable graceful degradation"
    )
    cache_predictions: bool = Field(
        default=False, description="Enable prediction caching"
    )


class PromptsConfig(BaseModel):
    """Configuration for prompts."""

    stage1_system: str = Field(..., description="Stage 1 system prompt")
    stage2_system_templates: Dict[str, str] = Field(
        default_factory=dict, description="Stage 2 category templates"
    )


class TwoStagePipelineConfig(BaseModel):
    """Complete configuration for two-stage pipeline."""

    stage1: Stage1Config = Field(
        default_factory=Stage1Config, description="Stage 1 configuration"
    )
    stage2: Stage2Config = Field(
        default_factory=Stage2Config, description="Stage 2 configuration"
    )
    general: GeneralConfig = Field(
        default_factory=GeneralConfig, description="General configuration"
    )
    prompts: PromptsConfig = Field(..., description="Prompts configuration")

    @field_validator("stage1", "stage2")
    @classmethod
    def validate_stage_weights(cls, v, info):
        """Validate stage configuration."""
        if info.field_name == "stage1":
            total_weight = v.svm_weight + v.llm_weight
            if not (0.8 <= total_weight <= 1.2):
                logger.warning(
                    f"Stage 1 weights sum to {total_weight}, should be close to 1.0"
                )
        return v


class LLMConfig(BaseModel):
    """LLM provider configuration.

    Works with any OpenAI-compatible API.  Set ``base_url`` to point at
    alternative providers (Anthropic Claude, Ollama, vLLM, Together, etc.).
    """

    api_key: Optional[str] = Field(
        default=None, description="API key for the LLM provider"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for OpenAI-compatible API (None = default OpenAI)",
    )
    default_model: str = Field(default="gpt-4o-mini", description="Default model")
    temperature: float = Field(
        default=0.0001, ge=0.0, le=2.0, description="Default temperature"
    )
    batch_size: int = Field(default=10, ge=1, description="Default batch size")
    max_concurrent_fine_tunes: int = Field(
        default=3, ge=1, description="Max concurrent fine-tunes"
    )
    daily_fine_tune_limit: int = Field(
        default=10, ge=1, description="Daily fine-tune limit"
    )


# Backward-compatible alias
OpenAIConfig = LLMConfig


class DataConfig(BaseModel):
    """Data management configuration."""

    data_dir: Path = Field(default=Path("./data"), description="Main data directory")
    raw_data_dir: Path = Field(
        default=Path("./data/raw_data"), description="Raw data directory"
    )
    processed_data_dir: Path = Field(
        default=Path("./data/processed_data"), description="Processed data directory"
    )
    labeled_data_dir: Path = Field(
        default=Path("./data/labeled_data"), description="Labeled data directory"
    )
    cache_expiry_hours: int = Field(
        default=24, ge=0, description="Cache expiry in hours"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )
    file_path: Optional[Path] = Field(default=None, description="Log file path")


class LinnaeusConfig(BaseModel):
    """Complete Linnaeus configuration."""

    environment: str = Field(
        default="development", description="Application environment"
    )
    openai: OpenAIConfig = Field(
        default_factory=OpenAIConfig, description="OpenAI configuration"
    )
    data: DataConfig = Field(
        default_factory=DataConfig, description="Data configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    two_stage_pipeline: TwoStagePipelineConfig = Field(
        ..., description="Two-stage pipeline configuration"
    )


class ConfigLoader:
    """Configuration loader with environment variable support."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        env_path: Optional[Path] = None,
        load_env: bool = True,
    ):
        """
        Initialize configuration loader.

        Parameters
        ----------
        config_path : Optional[Path]
            Path to config.yaml file.
        env_path : Optional[Path]
            Path to .env file.
        load_env : bool
            Whether to load environment variables.
        """
        self.config_path = config_path or Path("config.yaml")
        self.env_path = env_path or Path(".env")
        self.load_env = load_env

        if self.load_env and self.env_path.exists():
            load_dotenv(self.env_path)
            logger.info(f"Loaded environment variables from {self.env_path}")

    def load_config(self) -> LinnaeusConfig:
        """
        Load configuration from YAML and environment variables.

        Returns
        -------
        LinnaeusConfig
            Validated configuration object.
        """
        # Load YAML config
        config_data = {}
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {self.config_path}")
        else:
            logger.warning(
                f"Configuration file {self.config_path} not found, using defaults"
            )

        # Override with environment variables
        config_data = self._apply_env_overrides(config_data)

        # Validate and return configuration
        try:
            config = LinnaeusConfig(**config_data)
            logger.info("Configuration validation successful")
            return config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Parameters
        ----------
        config_data : Dict[str, Any]
            Configuration data from YAML.

        Returns
        -------
        Dict[str, Any]
            Configuration with environment overrides applied.
        """
        # Accept 'llm_provider' as YAML alias for 'openai'
        if "llm_provider" in config_data and "openai" not in config_data:
            config_data["openai"] = config_data.pop("llm_provider")
        elif "llm_provider" in config_data:
            config_data.pop("llm_provider")

        # LLM / OpenAI API Key — LLM_API_KEY takes priority
        if api_key := os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
            config_data.setdefault("openai", {})["api_key"] = api_key

        # LLM base URL
        if base_url := os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"):
            config_data.setdefault("openai", {})["base_url"] = base_url

        # IPinfo Token
        if ipinfo_token := os.getenv("IPINFO_TOKEN"):
            config_data.setdefault("apis", {})["ipinfo_token"] = ipinfo_token

        # Environment
        if environment := os.getenv("ENVIRONMENT"):
            config_data["environment"] = environment

        # Logging
        if log_level := os.getenv("LOG_LEVEL"):
            config_data.setdefault("logging", {})["level"] = log_level

        if log_file := os.getenv("LOG_FILE"):
            config_data.setdefault("logging", {})["file_path"] = log_file

        # Database URL (if needed)
        if db_url := os.getenv("DATABASE_URL"):
            config_data["database_url"] = db_url

        return config_data

    def save_config(
        self, config: LinnaeusConfig, output_path: Optional[Path] = None
    ) -> None:
        """
        Save configuration to YAML file.

        Parameters
        ----------
        config : LinnaeusConfig
            Configuration to save.
        output_path : Optional[Path]
            Output path (defaults to config_path).
        """
        output_path = output_path or self.config_path

        # Convert to dict and remove sensitive data
        config_dict = config.model_dump()
        if "openai" in config_dict and "api_key" in config_dict["openai"]:
            config_dict["openai"][
                "api_key"
            ] = "# Set via OPENAI_API_KEY environment variable"

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {output_path}")


def load_two_stage_config(
    config_path: Optional[Path] = None, env_path: Optional[Path] = None
) -> TwoStagePipelineConfig:
    """
    Convenience function to load two-stage pipeline configuration.

    Parameters
    ----------
    config_path : Optional[Path]
        Path to config file.
    env_path : Optional[Path]
        Path to .env file.

    Returns
    -------
    TwoStagePipelineConfig
        Two-stage pipeline configuration.
    """
    loader = ConfigLoader(config_path=config_path, env_path=env_path)
    full_config = loader.load_config()
    return full_config.two_stage_pipeline


# Export main classes and functions
__all__ = [
    "TwoStagePipelineConfig",
    "Stage1Config",
    "Stage2Config",
    "GeneralConfig",
    "PromptsConfig",
    "LinnaeusConfig",
    "LLMConfig",
    "OpenAIConfig",
    "ConfigLoader",
    "load_two_stage_config",
]
