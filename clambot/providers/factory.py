"""Provider factory — creates the right LLMProvider for a given config + model.

Routing logic:
  1. Parse model prefix → look up provider config section.
  2. If provider is ``custom`` → :class:`CustomProvider` (``openai/`` prefix).
  3. Otherwise → :class:`LiteLLMProvider`.
"""

from __future__ import annotations

from typing import Any

from clambot.config.schema import ClamBotConfig, ProviderConfig
from clambot.providers.base import LLMProvider
from clambot.providers.custom_provider import CustomProvider
from clambot.providers.litellm_provider import LiteLLMProvider
from clambot.providers.openai_codex_provider import OpenAICodexProvider
from clambot.providers.registry import find_provider_for_model


def create_provider(config: ClamBotConfig, model: str | None = None) -> LLMProvider:
    """Create an :class:`LLMProvider` for the given *config* and *model*.

    If *model* is ``None`` the default model from ``config.agents.defaults``
    is used.  The model string prefix determines which provider config section
    supplies the API key, base URL, and headers.

    Returns:
        A concrete provider instance satisfying :class:`LLMProvider`.
    """
    model = model or config.agents.defaults.model
    if not model:
        raise ValueError(
            "No model configured. Run: uv run clambot provider connect <provider>"
        )
    provider_name = find_provider_for_model(model)

    if provider_name:
        provider_cfg = _get_provider_config(config.providers, provider_name)
    else:
        # No recognised prefix — fall back to empty config.
        provider_cfg = ProviderConfig()

    # OpenAI Codex → OAuth-based Codex Responses API.
    if provider_name == "openai_codex":
        from clambot.providers.openai_codex_provider import DEFAULT_CODEX_URL

        return OpenAICodexProvider(
            default_model=model,
            api_url=provider_cfg.api_base or DEFAULT_CODEX_URL,
            ssl_fallback_insecure=config.security.ssl_fallback_insecure,
        )

    # Custom provider → delegate with openai/ prefix.
    if provider_name == "custom":
        return CustomProvider(
            model=model,
            api_key=provider_cfg.api_key or None,
            api_base=provider_cfg.api_base,
            extra_headers=provider_cfg.extra_headers,
        )

    # Default: LiteLLM for all standard providers.
    return LiteLLMProvider(
        model=model,
        api_key=provider_cfg.api_key or None,
        api_base=provider_cfg.api_base,
        extra_headers=provider_cfg.extra_headers,
    )


def _get_provider_config(providers: Any, name: str) -> ProviderConfig:
    """Retrieve the ``ProviderConfig`` for a given provider *name*."""
    return getattr(providers, name, ProviderConfig())
