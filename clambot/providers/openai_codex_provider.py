"""OpenAI Codex Responses provider — OAuth-based access to the Codex API.

Uses ``oauth_cli_kit`` for token management (login, refresh, storage) and
streams SSE responses from the ChatGPT Codex backend API.

Model strings should use the ``openai-codex/`` prefix, e.g.:
    ``openai-codex/gpt-5.1-codex``

When using the legacy default model, the provider now discovers available
Codex models and heuristically picks the most advanced ``gpt-<max>-codex``
variant from the account-visible catalog.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from clambot.providers.base import LLMResponse
from clambot.utils.constants import USER_AGENT

logger = logging.getLogger(__name__)

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_CODEX_MODELS_URL = "https://chatgpt.com/backend-api/codex/models"
DEFAULT_CODEX_CLIENT_VERSION = "0.0.0"
DEFAULT_ORIGINATOR = "clambot"
LEGACY_DEFAULT_CODEX_MODEL = "openai-codex/gpt-5.1-codex"

_GPT_CODEX_RE = re.compile(r"^gpt-(\d+(?:\.\d+)*)-codex(?:$|[-_].*)", re.IGNORECASE)
_GPT_RE = re.compile(r"^gpt-(\d+(?:\.\d+)*)", re.IGNORECASE)


class OpenAICodexProvider:
    """LLM provider for the OpenAI Codex Responses API (OAuth flow).

    Satisfies the :class:`LLMProvider` Protocol via ``acomplete``.

    Unlike the LiteLLM-backed providers, this one manages its own HTTP
    transport and authentication via ``oauth_cli_kit``.

    Parameters:
        default_model: Default model identifier (e.g.
                       ``openai-codex/gpt-5.1-codex``).  Overridable via
                       ``providers.openai_codex`` config or the
                       ``agents.defaults.model`` config key.
        api_url: Codex Responses API endpoint.  Defaults to
                 ``DEFAULT_CODEX_URL``.  Override via
                 ``providers.openai_codex.api_base`` in config.
        ssl_fallback_insecure: When ``True``, retry with ``verify=False``
            on SSL certificate errors.  Defaults to ``False`` (strict).
    """

    def __init__(
        self,
        default_model: str = LEGACY_DEFAULT_CODEX_MODEL,
        api_url: str = DEFAULT_CODEX_URL,
        *,
        ssl_fallback_insecure: bool = False,
    ) -> None:
        self.default_model = default_model
        self.api_url = api_url
        self._ssl_fallback_insecure = ssl_fallback_insecure
        self._model_resolution_lock = asyncio.Lock()
        self._resolved_default_model: str | None = None
        self._attempted_default_model_resolution = False

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request to the Codex Responses API.

        Recognised ``kwargs``: ``model``, ``max_tokens``, ``temperature``.
        """
        from oauth_cli_kit import OPENAI_CODEX_PROVIDER, get_token

        requested_model = kwargs.pop("model", None)
        tools = kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)  # always "auto" for Codex

        system_prompt, input_items = _convert_messages(messages)

        token = await asyncio.to_thread(
            get_token,
            provider=OPENAI_CODEX_PROVIDER,
        )
        headers = _build_headers(token.account_id, token.access)

        if requested_model is None:
            model = await self._resolve_default_model(headers)
        else:
            model = requested_model

        body: dict[str, Any] = {
            "model": _strip_model_prefix(model),
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": _prompt_cache_key(messages),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        # Convert tool schemas from Chat Completions format (nested
        # "function" key) to the Codex Responses API flat format.
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": t.get("function", {}).get("name", ""),
                    "description": t.get("function", {}).get("description", ""),
                    "parameters": t.get("function", {}).get("parameters", {}),
                }
                for t in tools
            ]

        try:
            try:
                content, tool_calls = await _request_codex(
                    self.api_url,
                    headers,
                    body,
                    verify=True,
                )
            except Exception as exc:
                if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
                    raise
                if not self._ssl_fallback_insecure:
                    logger.warning(
                        "SSL verification failed for Codex API; insecure fallback disabled",
                    )
                    raise
                logger.warning(
                    "SSL verification failed for Codex API; retrying with verify=False",
                )
                content, tool_calls = await _request_codex(
                    self.api_url,
                    headers,
                    body,
                    verify=False,
                )
            return LLMResponse(
                content=content,
                tool_calls=tool_calls if tool_calls else None,
            )
        except Exception as exc:
            return LLMResponse(content=f"Error calling Codex: {exc}")

    async def _resolve_default_model(self, headers: dict[str, str]) -> str:
        """Resolve the best default Codex model once per provider instance.

        Only auto-resolves when the configured default model is a known legacy
        placeholder (e.g. ``gpt-5.1-codex``), preserving explicit user-pinned
        models.
        """
        if not _should_auto_discover_default_model(self.default_model):
            return self.default_model

        if self._resolved_default_model is not None:
            return self._resolved_default_model
        if self._attempted_default_model_resolution:
            return self.default_model

        async with self._model_resolution_lock:
            if self._resolved_default_model is not None:
                return self._resolved_default_model
            if self._attempted_default_model_resolution:
                return self.default_model

            self._attempted_default_model_resolution = True
            models_url = _models_url_for_api(self.api_url)

            try:
                try:
                    available = await _request_codex_models(
                        models_url,
                        headers,
                        verify=True,
                    )
                except Exception as exc:
                    if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
                        raise
                    if not self._ssl_fallback_insecure:
                        logger.warning(
                            "SSL verification failed for Codex models endpoint; "
                            "insecure fallback disabled",
                        )
                        raise
                    logger.warning(
                        "SSL verification failed for Codex models endpoint; retrying "
                        "with verify=False",
                    )
                    available = await _request_codex_models(
                        models_url,
                        headers,
                        verify=False,
                    )

                discovered = _pick_most_advanced_codex_model(available)
                if discovered:
                    resolved = _apply_model_prefix(self.default_model, discovered)
                    self._resolved_default_model = resolved
                    logger.info(
                        "Resolved best Codex model: %s (from %d candidates)",
                        discovered,
                        len(available),
                    )
            except Exception as exc:
                logger.debug("Failed to auto-resolve Codex model: %s", exc)

            return self._resolved_default_model or self.default_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_model_prefix(model: str) -> str:
    """Remove the ``openai-codex/`` or ``openai_codex/`` routing prefix."""
    for prefix in ("openai-codex/", "openai_codex/"):
        if model.startswith(prefix):
            return model.split("/", 1)[1]
    return model


def _build_headers(account_id: str | None, token: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": USER_AGENT,
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    if account_id:
        headers["chatgpt-account-id"] = account_id
    return headers


def _models_url_for_api(api_url: str) -> str:
    """Derive the codex models URL from the configured responses URL."""
    if api_url.endswith("/responses"):
        return f"{api_url.rsplit('/responses', 1)[0]}/models"
    return DEFAULT_CODEX_MODELS_URL


def _should_auto_discover_default_model(model: str) -> bool:
    """Whether a configured default model should be auto-resolved.

    We only auto-resolve for known legacy placeholders so explicit model
    pinning keeps working as expected.
    """
    stripped = _strip_model_prefix(model).strip().lower()
    return stripped in {
        "gpt-5.1-codex",
        "auto",
        "latest",
        "gpt-max-codex",
        "gpt-<max>-codex",
    }


def _apply_model_prefix(reference_model: str, discovered_model: str) -> str:
    """Apply the same provider prefix style as *reference_model*."""
    if reference_model.startswith("openai_codex/"):
        return f"openai_codex/{discovered_model}"
    if reference_model.startswith("openai-codex/"):
        return f"openai-codex/{discovered_model}"
    return discovered_model


def _parse_version_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split("."))


def _normalized_version_key(parts: tuple[int, ...], *, width: int = 4) -> tuple[int, ...]:
    """Normalize variable-length version tuples for consistent comparison."""
    if len(parts) >= width:
        return parts[:width]
    return parts + (0,) * (width - len(parts))


def _pick_most_advanced_codex_model(models: list[str]) -> str | None:
    """Pick the most advanced Codex model from discovered model names.

    Heuristic:
      1) Prefer ``gpt-<version>-codex`` family (highest version wins).
      2) Within same version, prefer the plain ``...-codex`` variant over
         suffixed variants such as ``...-codex-spark``.
      3) If no codex-suffixed model exists, fall back to highest ``gpt-*``.
    """
    codex_candidates: list[tuple[tuple[int, ...], int, str]] = []
    gpt_candidates: list[tuple[tuple[int, ...], str]] = []

    for raw in models:
        model = _strip_model_prefix(raw).strip()
        if not model:
            continue

        codex_match = _GPT_CODEX_RE.match(model)
        if codex_match:
            version = _normalized_version_key(_parse_version_tuple(codex_match.group(1)))
            plain_bonus = 1 if model.lower().endswith("-codex") else 0
            codex_candidates.append((version, plain_bonus, model))

        gpt_match = _GPT_RE.match(model)
        if gpt_match:
            version = _normalized_version_key(_parse_version_tuple(gpt_match.group(1)))
            gpt_candidates.append((version, model))

    if codex_candidates:
        codex_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return codex_candidates[0][2]

    if gpt_candidates:
        gpt_candidates.sort(key=lambda item: item[0], reverse=True)
        return gpt_candidates[0][1]

    return None


async def _request_codex_models(
    url: str,
    headers: dict[str, str],
    verify: bool,
) -> list[str]:
    """Fetch available Codex models for the authenticated account."""
    request_headers = dict(headers)
    request_headers["accept"] = "application/json"

    async with httpx.AsyncClient(timeout=20.0, verify=verify) as client:
        response = await client.get(
            url,
            headers=request_headers,
            params={"client_version": DEFAULT_CODEX_CLIENT_VERSION},
        )

    if response.status_code != 200:
        text = response.text
        raise RuntimeError(_friendly_error(response.status_code, text))

    payload = response.json()
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return []

    discovered: list[str] = []
    for item in raw_models:
        if isinstance(item, str) and item:
            discovered.append(item)
            continue

        if isinstance(item, dict):
            slug = item.get("slug") or item.get("id") or item.get("model")
            if isinstance(slug, str) and slug:
                discovered.append(slug)

    return discovered


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    verify: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """Make a streaming POST to the Codex API.

    Returns:
        ``(text_content, tool_calls)`` tuple.
    """
    async with httpx.AsyncClient(timeout=120.0, verify=verify) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise RuntimeError(
                    _friendly_error(response.status_code, text.decode("utf-8", "ignore")),
                )
            return await _consume_sse(response)


# ---------------------------------------------------------------------------
# Message conversion (OpenAI chat format → Codex Responses format)
# ---------------------------------------------------------------------------


def _convert_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert standard chat messages to Codex Responses API format.

    Returns:
        ``(system_prompt, input_items)`` tuple.
    """
    system_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
            continue

        if role == "user":
            input_items.append(_convert_user_message(content))
            continue

        if role == "assistant":
            if isinstance(content, str) and content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                        "status": "completed",
                        "id": f"msg_{idx}",
                    }
                )
            # Handle tool calls embedded in assistant messages.
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = _split_tool_call_id(tool_call.get("id"))
                call_id = call_id or f"call_{idx}"
                item_id = item_id or f"fc_{idx}"
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        if role == "tool":
            call_id, _ = _split_tool_call_id(msg.get("tool_call_id"))
            output_text = (
                content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )
            continue

    system_prompt = "\n\n".join(system_parts)
    return system_prompt, input_items


def _convert_user_message(content: Any) -> dict[str, Any]:
    """Convert a user message content to Codex input format."""
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append(
                        {
                            "type": "input_image",
                            "image_url": url,
                            "detail": "auto",
                        }
                    )
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def _split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    """Split a compound ``call_id|item_id`` string."""
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None


def _prompt_cache_key(messages: list[dict[str, Any]]) -> str:
    """Deterministic hash for prompt caching."""
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


async def _iter_sse(
    response: httpx.Response,
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield parsed JSON events from an SSE stream."""
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                data_lines = [ln[5:].strip() for ln in buffer if ln.startswith("data:")]
                buffer = []
                if not data_lines:
                    continue
                data = "\n".join(data_lines).strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except Exception:
                    continue
            continue
        buffer.append(line)


async def _consume_sse(
    response: httpx.Response,
) -> tuple[str, list[dict[str, Any]]]:
    """Consume SSE events and return text content + tool calls."""
    content = ""

    # Track function calls as they stream in:
    # call_id → {"id": ..., "name": ..., "arguments": ""}
    active_calls: dict[str, dict[str, str]] = {}
    completed_calls: list[dict[str, Any]] = []

    async for event in _iter_sse(response):
        event_type = event.get("type")

        if event_type == "response.output_text.delta":
            content += event.get("delta") or ""

        # Function call started
        elif event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id") or item.get("id") or ""
                active_calls[call_id] = {
                    "id": call_id,
                    "name": item.get("name", ""),
                    "arguments": "",
                }

        # Function call arguments streaming
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id") or event.get("item_id") or ""
            if call_id in active_calls:
                active_calls[call_id]["arguments"] += event.get("delta") or ""

        # Function call completed
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id") or item.get("id") or ""
                entry = active_calls.pop(call_id, None)
                name = item.get("name") or (entry["name"] if entry else "")
                arguments = item.get("arguments") or (entry["arguments"] if entry else "{}")
                # Use compound id so tool-result routing works
                item_id = item.get("id") or ""
                compound_id = f"{call_id}|{item_id}" if item_id else call_id
                completed_calls.append(
                    {
                        "id": compound_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    }
                )

        elif event_type in {"error", "response.failed"}:
            error_msg = event.get("message") or event.get("error") or "Codex response failed"
            raise RuntimeError(str(error_msg))

    return content, completed_calls


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


def _friendly_error(status_code: int, raw: str) -> str:
    if status_code == 429:
        return "ChatGPT usage quota exceeded or rate limit triggered. Please try again later."
    return f"HTTP {status_code}: {raw}"
