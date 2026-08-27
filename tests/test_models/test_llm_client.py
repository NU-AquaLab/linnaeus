"""Tests for the LLM client factory."""

import os
from unittest.mock import patch

from linneaus.models.llm.client import (
    _resolve_api_key,
    _resolve_base_url,
    create_async_client,
    create_client,
    create_client_from_config,
)


class TestResolveApiKey:
    """Test API key resolution logic."""

    def test_explicit_arg_wins(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "env", "OPENAI_API_KEY": "oai"}):
            assert _resolve_api_key("explicit") == "explicit"

    def test_llm_api_key_over_openai(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "llm", "OPENAI_API_KEY": "oai"}):
            assert _resolve_api_key() == "llm"

    def test_fallback_to_openai_api_key(self):
        env = {"OPENAI_API_KEY": "oai"}
        with patch.dict(os.environ, env, clear=True):
            assert _resolve_api_key() == "oai"

    def test_none_when_nothing_set(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _resolve_api_key() is None


class TestResolveBaseUrl:
    """Test base URL resolution logic."""

    def test_explicit_arg_wins(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://env"}):
            assert _resolve_base_url("http://explicit") == "http://explicit"

    def test_llm_base_url_over_openai(self):
        with patch.dict(
            os.environ,
            {"LLM_BASE_URL": "http://llm", "OPENAI_BASE_URL": "http://oai"},
        ):
            assert _resolve_base_url() == "http://llm"

    def test_fallback_to_openai_base_url(self):
        env = {"OPENAI_BASE_URL": "http://oai"}
        with patch.dict(os.environ, env, clear=True):
            assert _resolve_base_url() == "http://oai"

    def test_none_when_nothing_set(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _resolve_base_url() is None


class TestCreateClient:
    """Test synchronous client creation."""

    def test_default_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = create_client()
            assert client.api_key == "test-key"

    def test_custom_base_url(self):
        client = create_client(api_key="k", base_url="http://localhost:11434/v1")
        assert "localhost" in str(client.base_url)

    def test_llm_api_key_priority(self):
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "llm-key", "OPENAI_API_KEY": "oai-key"},
        ):
            client = create_client()
            assert client.api_key == "llm-key"


class TestCreateAsyncClient:
    """Test asynchronous client creation."""

    def test_creates_async_client(self):
        from openai import AsyncOpenAI

        client = create_async_client(api_key="k")
        assert isinstance(client, AsyncOpenAI)

    def test_custom_base_url(self):
        client = create_async_client(
            api_key="k", base_url="https://api.anthropic.com/v1/"
        )
        assert "anthropic" in str(client.base_url)


class TestCreateFromConfig:
    """Test client creation from LLMConfig."""

    def test_create_from_config(self):
        from linneaus.config.settings import LLMConfig

        cfg = LLMConfig(api_key="cfg-key", base_url="http://localhost:8000/v1")
        client = create_client_from_config(cfg)
        assert client.api_key == "cfg-key"
        assert "localhost" in str(client.base_url)
