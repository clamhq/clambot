"""Tools — Built-in tool implementations.

This module provides the :data:`BUILTIN_TOOLS` tuple containing factory
descriptors for every built-in tool, the :class:`BuiltinToolRegistry` for
runtime dispatch, and the :class:`BuiltinTool` ABC for implementing new tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from clambot.tools.base import BuiltinTool, ToolApprovalOption

if TYPE_CHECKING:
    from pathlib import Path

    from clambot.config.schema import ClamBotConfig
    from clambot.tools.secrets.store import SecretStore

# Re-export individual tool classes for convenience.
from clambot.tools.cron import CronTool
from clambot.tools.echo import EchoTool
from clambot.tools.filesystem import FilesystemTool
from clambot.tools.http import HttpRequestTool
from clambot.tools.memory import MemoryRecallTool, MemorySearchHistoryTool
from clambot.tools.pdf import PdfReaderTool
from clambot.tools.registry import BuiltinToolRegistry
from clambot.tools.secrets import SecretsAddTool
from clambot.tools.transcribe import TranscribeTool
from clambot.tools.web import WebFetchTool, WebSearchTool

__all__ = [
    "ToolApprovalOption",
    "BuiltinTool",
    "BuiltinToolRegistry",
    "BUILTIN_TOOLS",
    "build_tool_registry",
    # Individual tool classes
    "FilesystemTool",
    "HttpRequestTool",
    "WebFetchTool",
    "WebSearchTool",
    "CronTool",
    "SecretsAddTool",
    "MemoryRecallTool",
    "MemorySearchHistoryTool",
    "EchoTool",
    "PdfReaderTool",
    "TranscribeTool",
]


# ---------------------------------------------------------------------------
# BUILTIN_TOOLS — names of all tools included in the default surface.
# The echo tool is intentionally *excluded* from this tuple; it is a debug
# utility that must be registered explicitly.
# ---------------------------------------------------------------------------

BUILTIN_TOOLS: tuple[str, ...] = (
    "fs",
    "http_request",
    "web_search",
    "web_fetch",
    "cron",
    "secrets_add",
    "memory_recall",
    "memory_search_history",
    "pdf_reader",
    "transcribe",
)


def build_tool_registry(
    *,
    workspace: Path,
    config: ClamBotConfig | None = None,
    secret_store: SecretStore | None = None,
    available_tools: list[str] | None = None,
    disabled_tools: list[str] | None = None,
) -> BuiltinToolRegistry:
    """Construct a :class:`BuiltinToolRegistry` with all configured tools.

    Tool selection priority (first match wins):
        1. ``available_tools`` — strict allowlist.  Only these tools are
           registered; new built-in tools are NOT auto-added.  Use this
           when you need a locked-down, non-extendable tool surface.
        2. ``disabled_tools`` — denylist.  Start with all
           :data:`BUILTIN_TOOLS` and remove the listed names.  New tools
           are automatically available.
        3. Neither set — all :data:`BUILTIN_TOOLS` are registered.

    Args:
        workspace: Root workspace path (used by ``fs`` and memory tools).
        config: Full ClamBot config (used for ``FilesystemToolConfig``).
        secret_store: An initialised :class:`SecretStore` (used by
            ``http_request`` and ``secrets_add``).
        available_tools: Strict allowlist — only these tools are enabled.
            Overrides ``disabled_tools`` when set.
        disabled_tools: Tool names to exclude from the default set.

    Returns:
        A fully-populated :class:`BuiltinToolRegistry`.
    """
    from pathlib import Path as _Path  # local to avoid top-level cycles

    from clambot.config.schema import ClamBotConfig as _Config

    ws = _Path(workspace) if not isinstance(workspace, _Path) else workspace
    cfg = config or _Config()
    ss = secret_store

    # Resolve tool names — priority: available_tools > disabled_tools > all
    if available_tools is not None:
        names = set(available_tools)
    elif disabled_tools:
        names = set(BUILTIN_TOOLS) - set(disabled_tools)
    else:
        names = set(BUILTIN_TOOLS)

    registry = BuiltinToolRegistry()

    # — Filesystem --------------------------------------------------------
    if "fs" in names:
        registry.register(FilesystemTool(workspace=ws, config=cfg.tools.filesystem))

    # — HTTP request ------------------------------------------------------
    if "http_request" in names:
        registry.register(
            HttpRequestTool(
                secret_store=ss,
                ssl_fallback_insecure=cfg.security.ssl_fallback_insecure,
            )
        )

    # — Web search --------------------------------------------------------
    if "web_search" in names:
        registry.register(WebSearchTool())

    # — Web fetch ---------------------------------------------------------
    if "web_fetch" in names:
        registry.register(
            WebFetchTool(
                ssl_fallback_insecure=cfg.security.ssl_fallback_insecure,
            )
        )

    # — Cron (stub until Phase 11) ----------------------------------------
    if "cron" in names:
        registry.register(CronTool())

    # — Secrets add -------------------------------------------------------
    if "secrets_add" in names and ss is not None:
        registry.register(SecretsAddTool(secret_store=ss))

    # — Memory recall -----------------------------------------------------
    if "memory_recall" in names:
        registry.register(MemoryRecallTool(workspace=ws))

    # — Memory search history ---------------------------------------------
    if "memory_search_history" in names:
        registry.register(MemorySearchHistoryTool(workspace=ws))

    # — PDF reader --------------------------------------------------------
    if "pdf_reader" in names:
        registry.register(PdfReaderTool(workspace=ws))

    # — Transcribe --------------------------------------------------------
    if "transcribe" in names:
        registry.register(TranscribeTool(config=cfg.tools.transcribe, secret_store=ss))

    # — Echo (debug only — not in BUILTIN_TOOLS) --------------------------
    if "echo" in names:
        registry.register(EchoTool())

    return registry
