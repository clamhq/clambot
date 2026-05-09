"""WebSearchTool — built-in tool for DuckDuckGo web search.

Searches the web for text, news, images, or videos and returns structured
results (title/url/snippet-style records from DuckDuckGo search backends).
The search dependency is installed lazily on first use.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Any

from clambot.tools.base import BuiltinTool, ToolApprovalOption

__all__ = ["WebSearchTool"]


def _resolve_ddgs_class() -> Any | None:
    """Return the DDGS class from installed search packages, if available."""
    try:
        from duckduckgo_search import DDGS  # type: ignore[import-untyped]

        return DDGS
    except ImportError:
        pass

    try:
        from ddgs import DDGS  # type: ignore[import-untyped]

        return DDGS
    except ImportError:
        return None


def _ensure_duckduckgo_search() -> Any:
    """Import DuckDuckGo search backend, installing it dynamically if needed."""
    ddgs_class = _resolve_ddgs_class()
    if ddgs_class is not None:
        return ddgs_class

    install_targets = ("duckduckgo-search", "ddgs")
    for package_name in install_targets:
        installed = False
        for cmd in (
            ["uv", "pip", "install", package_name],
            [sys.executable, "-m", "pip", "install", package_name],
        ):
            try:
                subprocess.check_call(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                installed = True
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        if not installed:
            continue

        importlib.invalidate_caches()
        ddgs_class = _resolve_ddgs_class()
        if ddgs_class is not None:
            return ddgs_class

    raise RuntimeError(
        "Failed to install duckduckgo-search. "
        "Please install it manually: uv pip install duckduckgo-search"
    )


def _normalize_search_type(value: Any) -> str:
    """Normalize search type to one of web/news/images/videos."""
    search_type = str(value or "web").strip().lower()
    if search_type in {"web", "news", "images", "videos"}:
        return search_type
    return "web"


def _normalize_safe_search(value: Any) -> str:
    """Normalize safe search mode to on/moderate/off."""
    mode = str(value or "moderate").strip().lower()
    if mode in {"on", "moderate", "off"}:
        return mode
    return "moderate"


def _normalize_time_range(value: Any) -> str | None:
    """Normalize timelimit to d/w/m/y or None."""
    if value is None:
        return None
    time_range = str(value).strip().lower()
    if time_range in {"d", "w", "m", "y"}:
        return time_range
    return None


def _normalize_max_results(value: Any) -> int:
    """Normalize and clamp max results to [1, 50]."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 10
    return max(1, min(n, 50))


def _coerce_results(results: Any, *, max_results: int) -> list[dict[str, Any]]:
    """Convert arbitrary backend result iterables to a bounded dict list."""
    if results is None:
        return []
    if isinstance(results, dict):
        return [dict(results)]

    normalized: list[dict[str, Any]] = []
    for item in results:
        if len(normalized) >= max_results:
            break
        if isinstance(item, dict):
            normalized.append(dict(item))
        else:
            normalized.append({"value": str(item)})
    return normalized


class WebSearchTool(BuiltinTool):
    """Built-in tool that performs DuckDuckGo-powered web searches."""

    @property
    def name(self) -> str:
        """Registered tool name: ``"web_search"``."""
        return "web_search"

    @property
    def description(self) -> str:
        """Human-readable description surfaced to the LLM."""
        return "Search the web for pages, news, images, or videos."

    @property
    def usage_instructions(self) -> list[str]:
        """Prompt guidance for generation-time web_search usage."""
        return [
            "Use for topic-based discovery when you need sources/URLs but do not have a URL yet.",
            "Prefer web_search first for research, current events, and fact-checking; then use web_fetch on selected links.",
            "Pass query and optional search_type (web/news/images/videos), max_results, time_range, region, safe_search.",
            "For known URLs, skip web_search and call web_fetch directly.",
            "Return concise top hits from result.results (title/url/snippet), not the entire raw object.",
        ]

    @property
    def schema(self) -> dict[str, Any]:
        """JSON Schema for the ``web_search`` tool parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text.",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["web", "news", "images", "videos"],
                    "default": "web",
                    "description": "Which DuckDuckGo vertical to search.",
                },
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of results to return.",
                },
                "time_range": {
                    "type": "string",
                    "enum": ["d", "w", "m", "y"],
                    "description": "Optional recency filter: day/week/month/year.",
                },
                "region": {
                    "type": "string",
                    "default": "wt-wt",
                    "description": "Region code (for example: us-en, uk-en, wt-wt).",
                },
                "safe_search": {
                    "type": "string",
                    "enum": ["on", "moderate", "off"],
                    "default": "moderate",
                    "description": "Safe-search filtering mode.",
                },
            },
            "required": ["query"],
        }

    @property
    def returns(self) -> dict[str, Any]:
        """Return value schema for ``web_search``."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Original search query."},
                "search_type": {
                    "type": "string",
                    "description": "Resolved search type used for the request.",
                },
                "count": {"type": "integer", "description": "Number of returned results."},
                "results": {
                    "type": "array",
                    "description": "Search results from the selected DuckDuckGo vertical.",
                    "items": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "error": {"type": "string", "description": "Present only on failure."},
            },
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a search query and return structured results."""
        query = str(args.get("query", "")).strip()
        search_type = _normalize_search_type(args.get("search_type", "web"))
        max_results = _normalize_max_results(args.get("max_results", 10))
        time_range = _normalize_time_range(args.get("time_range"))
        region = str(args.get("region", "wt-wt") or "wt-wt").strip()
        safe_search = _normalize_safe_search(args.get("safe_search", "moderate"))

        if not query:
            return {
                "query": query,
                "search_type": search_type,
                "count": 0,
                "results": [],
                "error": "Missing required parameter: query",
            }

        try:
            ddgs_class = _ensure_duckduckgo_search()
            ddgs = ddgs_class(timeout=20)

            if search_type == "news":
                raw_results = ddgs.news(
                    query,
                    region=region,
                    safesearch=safe_search,
                    timelimit=time_range,
                    max_results=max_results,
                )
            elif search_type == "images":
                raw_results = ddgs.images(
                    query,
                    region=region,
                    safesearch=safe_search,
                    timelimit=time_range,
                    max_results=max_results,
                )
            elif search_type == "videos":
                raw_results = ddgs.videos(
                    query,
                    region=region,
                    safesearch=safe_search,
                    timelimit=time_range,
                    max_results=max_results,
                )
            else:
                raw_results = ddgs.text(
                    query,
                    region=region,
                    safesearch=safe_search,
                    timelimit=time_range,
                    max_results=max_results,
                )

            results = _coerce_results(raw_results, max_results=max_results)
            return {
                "query": query,
                "search_type": search_type,
                "count": len(results),
                "results": results,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "query": query,
                "search_type": search_type,
                "count": 0,
                "results": [],
                "error": str(exc),
            }

    def get_approval_options(self, args: dict[str, Any]) -> list[ToolApprovalOption]:
        """Return an empty list — web_search is read-only and needs no approval."""
        return []
