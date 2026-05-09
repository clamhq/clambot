"""Web tools — built-in tools for web fetch and web search.

Re-exports :class:`WebFetchTool` and :class:`WebSearchTool` as the primary
public symbols for this sub-package.
"""

from __future__ import annotations

from clambot.tools.web.fetch import WebFetchTool
from clambot.tools.web.search import WebSearchTool

__all__ = [
    "WebFetchTool",
    "WebSearchTool",
]
