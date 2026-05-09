"""Tests for the web_search built-in tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from clambot.tools.web.search import WebSearchTool


class TestWebSearchTool:
    """Tests for WebSearchTool behavior and metadata."""

    def test_usage_instructions_present(self) -> None:
        """Tool exposes usage instructions for prompt injection."""
        tool = WebSearchTool()
        instructions = tool.usage_instructions
        assert instructions
        assert any("query" in item for item in instructions)
        assert any("web_fetch" in item for item in instructions)

    def test_missing_query_returns_error(self) -> None:
        """Missing query returns a structured error payload."""
        tool = WebSearchTool()
        result = tool.execute({})
        assert result["count"] == 0
        assert result["results"] == []
        assert "error" in result

    def test_web_search_success(self) -> None:
        """search_type=web calls DDGS.text and returns normalized output."""
        tool = WebSearchTool()
        captured: dict[str, object] = {}

        class StubDDGS:
            def __init__(self, timeout: int = 0) -> None:
                captured["timeout"] = timeout

            def text(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return [{"title": "Result A", "href": "https://example.com", "body": "Snippet"}]

        with patch("clambot.tools.web.search._ensure_duckduckgo_search", return_value=StubDDGS):
            result = tool.execute(
                {
                    "query": "python testing",
                    "search_type": "web",
                    "max_results": 5,
                    "time_range": "w",
                    "region": "us-en",
                    "safe_search": "off",
                }
            )

        assert captured["timeout"] == 20
        assert captured["args"] == ("python testing",)
        assert captured["kwargs"] == {
            "region": "us-en",
            "safesearch": "off",
            "timelimit": "w",
            "max_results": 5,
        }
        assert result["search_type"] == "web"
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Result A"

    def test_execute_error_is_returned(self) -> None:
        """Backend exceptions are converted to structured error responses."""
        tool = WebSearchTool()

        class FailingDDGS:
            def __init__(self, timeout: int = 0) -> None:
                _ = timeout

            def text(self, *args, **kwargs):
                _ = args
                _ = kwargs
                raise RuntimeError("search backend unavailable")

        with patch("clambot.tools.web.search._ensure_duckduckgo_search", return_value=FailingDDGS):
            result = tool.execute({"query": "hello"})

        assert result["count"] == 0
        assert result["results"] == []
        assert "backend" in result["error"]


class TestEnsureDuckDuckGoImport:
    """Tests for the lazy dependency import path."""

    def test_ensure_uses_installed_module(self) -> None:
        """When duckduckgo_search is importable, installer is not called."""
        fake_module = MagicMock()
        fake_module.DDGS = object()

        with (
            patch("clambot.tools.web.search._resolve_ddgs_class", return_value=fake_module.DDGS),
            patch("subprocess.check_call") as mock_subprocess,
        ):
            from clambot.tools.web.search import _ensure_duckduckgo_search

            result = _ensure_duckduckgo_search()

        assert result is fake_module.DDGS
        mock_subprocess.assert_not_called()
