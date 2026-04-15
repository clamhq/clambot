"""Tests for clambot.providers — Phase 3 Provider Layer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clambot.config.schema import ClamBotConfig
from clambot.providers.base import LLMProvider, LLMResponse
from clambot.providers.custom_provider import CustomProvider
from clambot.providers.factory import create_provider
from clambot.providers.litellm_provider import LiteLLMProvider
from clambot.providers.registry import PROVIDER_PREFIXES, find_provider_for_model

# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def test_frozen(self) -> None:
        """LLMResponse instances are immutable."""
        resp = LLMResponse(content="hello", usage={"total_tokens": 10})
        with pytest.raises(AttributeError):
            resp.content = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        """Usage defaults to None when not supplied."""
        resp = LLMResponse(content="ok")
        assert resp.usage is None

    def test_equality(self) -> None:
        """Two responses with the same data are equal (frozen dataclass)."""
        a = LLMResponse(content="hi", usage={"total_tokens": 5})
        b = LLMResponse(content="hi", usage={"total_tokens": 5})
        assert a == b


# ---------------------------------------------------------------------------
# LLMProvider Protocol
# ---------------------------------------------------------------------------


class TestProviderProtocol:
    def test_litellm_provider_satisfies_protocol(self) -> None:
        p = LiteLLMProvider(model="test")
        assert isinstance(p, LLMProvider)

    def test_custom_provider_satisfies_protocol(self) -> None:
        p = CustomProvider(model="test")
        assert isinstance(p, LLMProvider)


# ---------------------------------------------------------------------------
# Registry — find_provider_for_model
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_find_openrouter(self) -> None:
        assert find_provider_for_model("openrouter/google/gemini-2.0-flash-001") == "openrouter"

    def test_find_anthropic(self) -> None:
        assert find_provider_for_model("anthropic/claude-sonnet-4-20250514") == "anthropic"

    def test_find_ollama(self) -> None:
        assert find_provider_for_model("ollama/llama3") == "ollama"

    def test_find_ollama_chat(self) -> None:
        assert find_provider_for_model("ollama_chat/llama3") == "ollama"

    def test_find_deepseek(self) -> None:
        assert find_provider_for_model("deepseek/deepseek-chat") == "deepseek"

    def test_find_gemini(self) -> None:
        assert find_provider_for_model("gemini/gemini-pro") == "gemini"

    def test_find_groq(self) -> None:
        assert find_provider_for_model("groq/llama3-8b-8192") == "groq"

    def test_find_openai(self) -> None:
        assert find_provider_for_model("openai/gpt-4") == "openai"

    def test_find_custom(self) -> None:
        assert find_provider_for_model("custom/my-model") == "custom"

    def test_no_prefix_returns_none(self) -> None:
        assert find_provider_for_model("bare-model-name") is None

    def test_unknown_prefix_returns_none(self) -> None:
        assert find_provider_for_model("unknown/model") is None

    def test_all_config_providers_have_prefix(self) -> None:
        """Every provider in ClamBotConfig.providers has a matching prefix entry."""
        config_providers = {
            "custom",
            "anthropic",
            "openai",
            "openrouter",
            "ollama",
            "deepseek",
            "groq",
            "gemini",
            "openai_codex",
        }
        registry_targets = set(PROVIDER_PREFIXES.values())
        assert config_providers == registry_targets


# ---------------------------------------------------------------------------
# Factory — create_provider
# ---------------------------------------------------------------------------


class TestFactory:
    def test_routes_anthropic(self) -> None:
        config = ClamBotConfig()
        config.agents.defaults.model = "anthropic/claude-sonnet-4-20250514"
        p = create_provider(config)
        assert isinstance(p, LiteLLMProvider)
        assert p.model == "anthropic/claude-sonnet-4-20250514"

    def test_routes_custom(self) -> None:
        config = ClamBotConfig()
        config.providers.custom.api_base = "http://localhost:8000/v1"
        p = create_provider(config, model="custom/my-model")
        assert isinstance(p, CustomProvider)

    def test_uses_default_model_from_config(self) -> None:
        # model default is now "" — must be set explicitly before calling create_provider
        config = ClamBotConfig()
        config.agents.defaults.model = "anthropic/claude-sonnet-4-20250514"
        p = create_provider(config)
        assert isinstance(p, LiteLLMProvider)
        assert p.model == config.agents.defaults.model

    def test_routes_openrouter(self) -> None:
        config = ClamBotConfig()
        config.providers.openrouter.api_key = "sk-or-test-key"
        p = create_provider(config, model="openrouter/google/gemini-2.0-flash-001")
        assert isinstance(p, LiteLLMProvider)
        assert p.api_key == "sk-or-test-key"

    def test_provider_config_fields_passed(self) -> None:
        config = ClamBotConfig()
        config.providers.anthropic.api_key = "sk-ant-test"
        config.providers.anthropic.extra_headers = {"X-Custom": "value"}
        p = create_provider(config, model="anthropic/claude-sonnet-4-5")
        assert isinstance(p, LiteLLMProvider)
        assert p.api_key == "sk-ant-test"
        assert p.extra_headers == {"X-Custom": "value"}

    def test_unknown_prefix_returns_litellm(self) -> None:
        """Models with unrecognised prefixes still get a LiteLLMProvider."""
        config = ClamBotConfig()
        p = create_provider(config, model="unknown/model-x")
        assert isinstance(p, LiteLLMProvider)
        assert p.model == "unknown/model-x"

    def test_no_api_key_is_none(self) -> None:
        """Empty api_key in config is passed as None to provider."""
        config = ClamBotConfig()
        p = create_provider(config, model="anthropic/claude-3-haiku")
        assert isinstance(p, LiteLLMProvider)
        assert p.api_key is None

    def test_routes_openai_codex(self) -> None:
        """openai-codex/ model creates an OpenAICodexProvider."""
        from clambot.providers.openai_codex_provider import OpenAICodexProvider

        config = ClamBotConfig()
        p = create_provider(config, model="openai-codex/gpt-5.1-codex")
        assert isinstance(p, OpenAICodexProvider)
        assert p.default_model == "openai-codex/gpt-5.1-codex"


# ---------------------------------------------------------------------------
# LiteLLMProvider — acomplete
# ---------------------------------------------------------------------------


def _mock_litellm_response(
    content: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a mock object mimicking a LiteLLM response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


@pytest.mark.asyncio
class TestLiteLLMProviderAcomplete:
    async def test_returns_llm_response(self) -> None:
        provider = LiteLLMProvider(model="test-model", api_key="test-key")
        mock_resp = _mock_litellm_response("Hello, world!")

        with patch(
            "clambot.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await provider.acomplete([{"role": "user", "content": "Hi"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello, world!"
        assert result.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    async def test_handles_error_gracefully(self) -> None:
        provider = LiteLLMProvider(model="test-model")

        with patch(
            "clambot.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("Connection failed"),
        ):
            result = await provider.acomplete([{"role": "user", "content": "Hi"}])

        assert isinstance(result, LLMResponse)
        assert "Error calling LLM" in result.content
        assert "Connection failed" in result.content

    async def test_forwards_max_tokens_and_temperature(self) -> None:
        provider = LiteLLMProvider(model="test-model", api_key="key")
        mock_resp = _mock_litellm_response()

        with patch(
            "clambot.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await provider.acomplete(
                [{"role": "user", "content": "Hi"}],
                max_tokens=1024,
                temperature=0.0,
            )

        kw = mock_call.call_args[1]
        assert kw["max_tokens"] == 1024
        assert kw["temperature"] == 0.0

    async def test_clamps_max_tokens_to_one(self) -> None:
        provider = LiteLLMProvider(model="test-model")
        mock_resp = _mock_litellm_response()

        with patch(
            "clambot.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await provider.acomplete(
                [{"role": "user", "content": "Hi"}],
                max_tokens=-5,
            )

        assert mock_call.call_args[1]["max_tokens"] == 1

    async def test_passes_api_key_and_base(self) -> None:
        provider = LiteLLMProvider(model="m", api_key="sk-123", api_base="http://localhost:8000")
        mock_resp = _mock_litellm_response()

        with patch(
            "clambot.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await provider.acomplete([{"role": "user", "content": "Hi"}])

        kw = mock_call.call_args[1]
        assert kw["api_key"] == "sk-123"
        assert kw["api_base"] == "http://localhost:8000"

    async def test_passes_extra_headers(self) -> None:
        provider = LiteLLMProvider(model="m", extra_headers={"X-App": "clambot"})
        mock_resp = _mock_litellm_response()

        with patch(
            "clambot.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await provider.acomplete([{"role": "user", "content": "Hi"}])

        assert mock_call.call_args[1]["extra_headers"] == {"X-App": "clambot"}

    async def test_null_usage_returns_none(self) -> None:
        provider = LiteLLMProvider(model="m")
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ok"
        resp.usage = None

        with patch(
            "clambot.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=resp,
        ):
            result = await provider.acomplete([{"role": "user", "content": "Hi"}])

        assert result.usage is None

    async def test_model_kwarg_overrides_default(self) -> None:
        provider = LiteLLMProvider(model="default-model")
        mock_resp = _mock_litellm_response()

        with patch(
            "clambot.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await provider.acomplete(
                [{"role": "user", "content": "Hi"}],
                model="override-model",
            )

        assert mock_call.call_args[1]["model"] == "override-model"


# ---------------------------------------------------------------------------
# CustomProvider — delegates to LiteLLMProvider with openai/ prefix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCustomProvider:
    async def test_adds_openai_prefix(self) -> None:
        provider = CustomProvider(model="my-local-model")
        mock_resp = _mock_litellm_response("custom response")

        with patch(
            "clambot.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            result = await provider.acomplete([{"role": "user", "content": "Hi"}])

        assert result.content == "custom response"
        assert mock_call.call_args[1]["model"] == "openai/my-local-model"

    async def test_does_not_double_prefix(self) -> None:
        provider = CustomProvider(model="openai/already-prefixed")
        mock_resp = _mock_litellm_response()

        with patch(
            "clambot.providers.litellm_provider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await provider.acomplete([{"role": "user", "content": "Hi"}])

        assert mock_call.call_args[1]["model"] == "openai/already-prefixed"


# ---------------------------------------------------------------------------
# OpenAICodexProvider
# ---------------------------------------------------------------------------

from clambot.providers.openai_codex_provider import (  # noqa: E402
    OpenAICodexProvider,
    _apply_model_prefix,
    _convert_messages,
    _friendly_error,
    _models_url_for_api,
    _pick_most_advanced_codex_model,
    _should_auto_discover_default_model,
    _strip_model_prefix,
)


class TestOpenAICodexProvider:
    # ------------------------------------------------------------------
    # Protocol conformance
    # ------------------------------------------------------------------

    def test_satisfies_protocol(self) -> None:
        """OpenAICodexProvider satisfies the LLMProvider protocol."""
        p = OpenAICodexProvider()
        assert isinstance(p, LLMProvider)

    # ------------------------------------------------------------------
    # _strip_model_prefix
    # ------------------------------------------------------------------

    def test_strip_model_prefix_dash(self) -> None:
        """openai-codex/ prefix is removed."""
        assert _strip_model_prefix("openai-codex/gpt-5.1-codex") == "gpt-5.1-codex"

    def test_strip_model_prefix_underscore(self) -> None:
        """openai_codex/ prefix is removed."""
        assert _strip_model_prefix("openai_codex/model") == "model"

    def test_strip_model_prefix_bare(self) -> None:
        """Model without a recognised prefix is returned unchanged."""
        assert _strip_model_prefix("bare-model") == "bare-model"

    # ------------------------------------------------------------------
    # Model discovery helpers
    # ------------------------------------------------------------------

    def test_models_url_for_api_default_responses_url(self) -> None:
        """Models endpoint is derived from responses endpoint."""
        assert _models_url_for_api("https://chatgpt.com/backend-api/codex/responses") == (
            "https://chatgpt.com/backend-api/codex/models"
        )

    def test_should_auto_discover_default_model_legacy(self) -> None:
        """Legacy default model enables auto discovery."""
        assert _should_auto_discover_default_model("openai-codex/gpt-5.1-codex")

    def test_should_not_auto_discover_pinned_model(self) -> None:
        """Explicitly pinned models are not auto-upgraded."""
        assert not _should_auto_discover_default_model("openai-codex/gpt-5.2-codex")

    def test_apply_model_prefix_reuses_reference_style(self) -> None:
        """Discovered model keeps caller's prefix style."""
        assert (
            _apply_model_prefix("openai_codex/gpt-5.1-codex", "gpt-5.3-codex")
            == "openai_codex/gpt-5.3-codex"
        )
        assert (
            _apply_model_prefix("openai-codex/gpt-5.1-codex", "gpt-5.3-codex")
            == "openai-codex/gpt-5.3-codex"
        )

    def test_pick_most_advanced_codex_model_prefers_highest_version(self) -> None:
        """Heuristic picks highest gpt-<version>-codex model."""
        models = [
            "gpt-5.2",
            "gpt-5.3-codex-spark",
            "gpt-5.3-codex",
            "gpt-4.1-codex",
        ]
        assert _pick_most_advanced_codex_model(models) == "gpt-5.3-codex"

    def test_pick_most_advanced_codex_model_falls_back_to_highest_gpt(self) -> None:
        """If no codex model exists, use highest gpt model."""
        models = ["gpt-4.1", "gpt-5.2", "codex-auto-review"]
        assert _pick_most_advanced_codex_model(models) == "gpt-5.2"

    # ------------------------------------------------------------------
    # _convert_messages
    # ------------------------------------------------------------------

    def test_convert_messages_system(self) -> None:
        """System message is extracted as system_prompt; not added to input_items."""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        system_prompt, input_items = _convert_messages(messages)
        assert system_prompt == "You are a helpful assistant."
        assert input_items == []

    def test_convert_messages_user(self) -> None:
        """User message is converted to input_items with input_text type."""
        messages = [{"role": "user", "content": "Hello!"}]
        system_prompt, input_items = _convert_messages(messages)
        assert system_prompt == ""
        assert len(input_items) == 1
        item = input_items[0]
        assert item["role"] == "user"
        assert item["content"][0]["type"] == "input_text"
        assert item["content"][0]["text"] == "Hello!"

    def test_convert_messages_assistant(self) -> None:
        """Assistant message is converted to input_items with output_text type."""
        messages = [{"role": "assistant", "content": "Hi there!"}]
        system_prompt, input_items = _convert_messages(messages)
        assert system_prompt == ""
        assert len(input_items) == 1
        item = input_items[0]
        assert item["type"] == "message"
        assert item["role"] == "assistant"
        assert item["content"][0]["type"] == "output_text"
        assert item["content"][0]["text"] == "Hi there!"
        assert item["status"] == "completed"

    # ------------------------------------------------------------------
    # _friendly_error
    # ------------------------------------------------------------------

    def test_friendly_error_429(self) -> None:
        """429 status returns a quota/rate-limit message."""
        msg = _friendly_error(429, "Too Many Requests")
        assert "quota" in msg.lower() or "rate limit" in msg.lower()

    def test_friendly_error_other(self) -> None:
        """Non-429 status returns HTTP <code>: <raw> format."""
        assert _friendly_error(500, "bad") == "HTTP 500: bad"

    # ------------------------------------------------------------------
    # acomplete — success path
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_acomplete_success(self) -> None:
        """acomplete returns LLMResponse with content from _request_codex."""
        mock_token = MagicMock()
        mock_token.access = "test-token"
        mock_token.account_id = "acc-123"
        mock_token.refresh = "r"
        mock_token.expires = 9999999999

        provider = OpenAICodexProvider()

        with (
            patch(
                "asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_token,
            ),
            patch(
                "clambot.providers.openai_codex_provider._request_codex_models",
                new_callable=AsyncMock,
                return_value=["gpt-5.3-codex", "gpt-5.2"],
            ) as mock_model_discovery,
            patch(
                "clambot.providers.openai_codex_provider._request_codex",
                new_callable=AsyncMock,
                return_value=("Hello from Codex", []),
            ) as mock_request,
        ):
            result = await provider.acomplete([{"role": "user", "content": "Say hello"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from Codex"
        assert result.usage is None
        assert mock_model_discovery.await_count == 1
        assert mock_request.await_args.args[2]["model"] == "gpt-5.3-codex"

    @pytest.mark.asyncio
    async def test_explicit_model_skips_auto_discovery(self) -> None:
        """Explicit per-call model should bypass default auto-discovery."""
        mock_token = MagicMock()
        mock_token.access = "test-token"
        mock_token.account_id = "acc-123"

        provider = OpenAICodexProvider()

        with (
            patch(
                "asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_token,
            ),
            patch(
                "clambot.providers.openai_codex_provider._request_codex_models",
                new_callable=AsyncMock,
                return_value=["gpt-5.3-codex"],
            ) as mock_model_discovery,
            patch(
                "clambot.providers.openai_codex_provider._request_codex",
                new_callable=AsyncMock,
                return_value=("ok", []),
            ) as mock_request,
        ):
            await provider.acomplete(
                [{"role": "user", "content": "Say hello"}],
                model="openai-codex/gpt-4.1-codex",
            )

        assert mock_model_discovery.await_count == 0
        assert mock_request.await_args.args[2]["model"] == "gpt-4.1-codex"

    # ------------------------------------------------------------------
    # acomplete — error path
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_acomplete_error_handling(self) -> None:
        """acomplete catches exceptions and returns an error LLMResponse."""
        mock_token = MagicMock()
        mock_token.access = "test-token"
        mock_token.account_id = "acc-123"

        provider = OpenAICodexProvider()

        with (
            patch(
                "asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_token,
            ),
            patch(
                "clambot.providers.openai_codex_provider._request_codex_models",
                new_callable=AsyncMock,
                return_value=["gpt-5.3-codex"],
            ),
            patch(
                "clambot.providers.openai_codex_provider._request_codex",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API down"),
            ),
        ):
            result = await provider.acomplete([{"role": "user", "content": "Hi"}])

        assert isinstance(result, LLMResponse)
        assert "Error calling Codex" in result.content
        assert "API down" in result.content

    # ------------------------------------------------------------------
    # acomplete — tool schema conversion
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tools_converted_to_codex_flat_format(self) -> None:
        """Tools in Chat Completions format are converted to Codex flat format."""
        mock_token = MagicMock()
        mock_token.access = "test-token"
        mock_token.account_id = "acc-123"

        provider = OpenAICodexProvider()
        captured_body: dict = {}

        async def capture_request(url, headers, body, verify):
            captured_body.update(body)
            return ("ok", [])

        with (
            patch(
                "asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_token,
            ),
            patch(
                "clambot.providers.openai_codex_provider._request_codex_models",
                new_callable=AsyncMock,
                return_value=["gpt-5.3-codex"],
            ),
            patch(
                "clambot.providers.openai_codex_provider._request_codex",
                new_callable=AsyncMock,
                side_effect=capture_request,
            ),
        ):
            await provider.acomplete(
                [{"role": "user", "content": "list jobs"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "cron",
                            "description": "Manage cron jobs",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

        # Codex format: name at top level, no nested "function" key
        assert "tools" in captured_body
        tool = captured_body["tools"][0]
        assert tool["name"] == "cron"
        assert tool["description"] == "Manage cron jobs"
        assert "function" not in tool
