"""
Tests for configuration settings.
"""

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from linnaeus.config.settings import (
    Config,
    DataConfig,
    LLMConfig,
    OpenAIConfig,
    get_config,
)


class TestOpenAIConfig:
    """Test OpenAI configuration."""

    def test_valid_config(self):
        """Test valid OpenAI configuration."""
        config = OpenAIConfig(
            api_key="test_key",
            default_model="gpt-4o-mini",
            temperature=0.001,
            batch_size=10,
        )

        assert config.api_key == "test_key"
        assert config.default_model == "gpt-4o-mini"
        assert config.temperature == 0.001
        assert config.batch_size == 10

    def test_optional_fields(self):
        """Test optional fields with defaults."""
        config = OpenAIConfig()

        assert config.api_key is None
        assert config.org_id is None
        assert config.default_model == "gpt-4o-mini"
        assert config.temperature == 0.0001
        assert config.batch_size == 10


class TestDataConfig:
    """Test data configuration."""

    def test_default_paths(self):
        """Test default data paths."""
        config = DataConfig()

        assert config.data_dir == Path("./data").resolve()
        assert config.cache_expiry_hours == 24

    def test_custom_paths(self):
        """Test custom data paths."""
        custom_dir = Path("/tmp/test_data")
        config = DataConfig(data_dir=custom_dir)

        assert config.data_dir == custom_dir.resolve()


class TestConfig:
    """Test main configuration class."""

    def test_from_dict(self, sample_config):
        """Test creating config from dictionary."""
        config = Config(**sample_config)

        assert config.environment == "testing"
        assert config.openai.api_key == "test_key"
        assert config.data.cache_expiry_hours == 24

    def test_from_file_with_yaml(self, temp_dir, sample_config):
        """Test loading config from YAML file."""
        config_file = temp_dir / "test_config.yaml"

        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        config = Config.from_file(config_file)

        assert config.environment == "testing"
        assert config.openai.api_key == "test_key"

    def test_from_file_nonexistent(self):
        """Test loading config from non-existent file falls back to defaults."""
        with patch.dict(os.environ, {}, clear=True):
            # When config file doesn't exist, Config.from_file uses defaults
            # (all fields have defaults or are Optional)
            config = Config.from_file(Path("nonexistent.yaml"))
            assert config.environment == "development"
            assert config.openai.api_key is None
            assert config.openai.default_model == "gpt-4o-mini"

    def test_env_overrides(self, temp_dir):
        """Test environment variable overrides."""
        config_data = {
            "environment": "development",
            "openai": {"api_key": "file_key"},
        }

        config_file = temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Test environment variable override
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env_key"}):
            config = Config.from_file(config_file)
            assert config.openai.api_key == "env_key"

    def test_apply_env_overrides(self):
        """Test applying environment overrides."""
        config_data = {"openai": {"api_key": "original"}}

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "new_key", "BATCH_SIZE": "20", "TEMPERATURE": "0.5"},
        ):
            result = Config._apply_env_overrides(config_data)

            assert result["openai"]["api_key"] == "new_key"
            assert result["openai"]["batch_size"] == 20
            assert result["openai"]["temperature"] == 0.5


class TestGetConfig:
    """Test global configuration getter."""

    def test_singleton_behavior(self, mock_config):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_with_custom_path(self, temp_dir):
        """Test get_config with custom configuration path."""
        config_data = {"environment": "custom", "openai": {"api_key": "custom_key"}}

        config_file = temp_dir / "custom_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Clear global config cache
        import linnaeus.config.settings

        linnaeus.config.settings._config = None

        config = get_config(config_file)
        assert config.environment == "custom"
        assert config.openai.api_key == "custom_key"


class TestLLMConfig:
    """Test LLMConfig and backward compatibility."""

    def test_base_url_defaults_to_none(self):
        """Test that base_url defaults to None."""
        config = LLMConfig()
        assert config.base_url is None

    def test_base_url_can_be_set(self):
        """Test that base_url can be set."""
        config = LLMConfig(base_url="http://localhost:11434/v1")
        assert config.base_url == "http://localhost:11434/v1"

    def test_backward_compat_alias(self):
        """Test that OpenAIConfig is LLMConfig."""
        assert OpenAIConfig is LLMConfig

    def test_openai_config_still_works(self):
        """Test that OpenAIConfig alias still constructs correctly."""
        config = OpenAIConfig(api_key="test", base_url="http://example.com/v1")
        assert config.api_key == "test"
        assert config.base_url == "http://example.com/v1"

    def test_env_override_base_url(self):
        """Test LLM_BASE_URL env var populates config."""
        config_data = {"openai": {"api_key": "k"}}
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://llm-url/v1"}, clear=True):
            result = Config._apply_env_overrides(config_data)
            assert result["openai"]["base_url"] == "http://llm-url/v1"

    def test_env_override_llm_api_key(self):
        """Test LLM_API_KEY takes priority over OPENAI_API_KEY."""
        config_data = {"openai": {}}
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "llm-key", "OPENAI_API_KEY": "oai-key"},
        ):
            result = Config._apply_env_overrides(config_data)
            assert result["openai"]["api_key"] == "llm-key"

    def test_llm_provider_yaml_alias(self):
        """Test that 'llm_provider' YAML key is accepted as alias for 'openai'."""
        config_data = {"llm_provider": {"api_key": "test", "base_url": "http://x/v1"}}
        with patch.dict(os.environ, {}, clear=True):
            result = Config._apply_env_overrides(config_data)
            assert "openai" in result
            assert result["openai"]["api_key"] == "test"
            assert result["openai"]["base_url"] == "http://x/v1"
            assert "llm_provider" not in result
