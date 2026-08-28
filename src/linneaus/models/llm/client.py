"""
Client factory for OpenAI-compatible LLM providers.

All LLM client construction goes through this module, enabling transparent
support for any OpenAI-compatible API (Anthropic Claude, Ollama, vLLM,
Together, Fireworks, Groq, etc.) via ``base_url``.
"""

import os
from typing import TYPE_CHECKING, Optional

from openai import AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from linnaeus.config.settings import LLMConfig


def _resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolve API key: explicit arg > LLM_API_KEY > OPENAI_API_KEY."""
    return api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")


def _resolve_base_url(base_url: Optional[str] = None) -> Optional[str]:
    """Resolve base URL: explicit arg > LLM_BASE_URL > OPENAI_BASE_URL."""
    return base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def create_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    org_id: Optional[str] = None,
) -> OpenAI:
    """Create a synchronous OpenAI-compatible client.

    Parameters
    ----------
    api_key : str, optional
        API key. Falls back to ``LLM_API_KEY`` then ``OPENAI_API_KEY`` env vars.
    base_url : str, optional
        Base URL for the API. Falls back to ``LLM_BASE_URL`` then
        ``OPENAI_BASE_URL`` env vars.  Examples::

            https://api.anthropic.com/v1/   # Claude
            http://localhost:11434/v1        # Ollama
            https://api.together.xyz/v1      # Together AI

    org_id : str, optional
        Organization ID (OpenAI-specific).
    """
    resolved_key = _resolve_api_key(api_key)
    resolved_url = _resolve_base_url(base_url)

    kwargs: dict = {}
    if resolved_key:
        kwargs["api_key"] = resolved_key
    if resolved_url:
        kwargs["base_url"] = resolved_url
    if org_id:
        kwargs["organization"] = org_id

    return OpenAI(**kwargs)


def create_async_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    org_id: Optional[str] = None,
) -> AsyncOpenAI:
    """Create an asynchronous OpenAI-compatible client.

    Same resolution logic as :func:`create_client`.
    """
    resolved_key = _resolve_api_key(api_key)
    resolved_url = _resolve_base_url(base_url)

    kwargs: dict = {}
    if resolved_key:
        kwargs["api_key"] = resolved_key
    if resolved_url:
        kwargs["base_url"] = resolved_url
    if org_id:
        kwargs["organization"] = org_id

    return AsyncOpenAI(**kwargs)


def create_client_from_config(config: "LLMConfig") -> OpenAI:
    """Create a synchronous client from a :class:`LLMConfig` object."""
    return create_client(
        api_key=config.api_key,
        base_url=config.base_url,
        org_id=config.org_id,
    )


def create_async_client_from_config(config: "LLMConfig") -> AsyncOpenAI:
    """Create an asynchronous client from a :class:`LLMConfig` object."""
    return create_async_client(
        api_key=config.api_key,
        base_url=config.base_url,
        org_id=config.org_id,
    )
