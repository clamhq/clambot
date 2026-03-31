"""Clam selector — two-stage routing: pre-selection + LLM-backed selection.

Stage 1: Normalize request → exact match against catalog ``source_request``
          → skip LLM entirely.
Stage 2: LLM call with selector prompt → parse JSON decision.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from clambot.providers.base import LLMProvider
from clambot.utils.text import strip_markdown_fences

from .clams import ClamSummary
from .request_normalization import normalize_request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Selection result
# ---------------------------------------------------------------------------


@dataclass
class SelectionResult:
    """Result from the clam selection process."""

    decision: str  # "chat", "select_existing", "generate_new"
    clam_id: str | None = None
    reason: str = ""
    chat_response: str = ""
    inputs: dict[str, Any] | None = None  # extracted inputs for select_existing


# ---------------------------------------------------------------------------
# Selector prompt
# ---------------------------------------------------------------------------

SELECTOR_SYSTEM_PROMPT = """\
You are a routing agent for ClamBot. Your job is to decide how to handle the user's request.

ClamBot can generate and execute JavaScript code ("clams") that use tools.

All available tools:
{available_tools}

You have three options:
1. "chat" — The user is making casual conversation (greetings, thanks, \
small talk), asking personal/memory questions (e.g. "what's my name?", \
"what did I tell you?", "do you remember...?"), OR the request can be \
fully handled by the agent's knowledge and long-term memory without \
running code. The agent has a long-term memory with user preferences \
and facts — questions about the user or past conversations are ALWAYS chat.
2. "select_existing" — An existing clam can fulfill this request. **ALWAYS prefer this over \
generate_new when an existing clam's description matches the task**, even if the specific \
values differ. Extract the new input values from the user's message.
3. "generate_new" — No existing clam can handle this request and it requires \
code execution (computation, API calls, data processing, cron management, \
web fetching, file operations, etc.).

Available clams:
{clam_catalog}

**IMPORTANT**:
- Questions about the user (name, preferences, past conversations) → ALWAYS "chat".
- If an existing clam's description matches the user's intent, ALWAYS use "select_existing".
- Any request that requires tool use (cron, web_fetch, http_request, fs, etc.) → \
"select_existing" or "generate_new". NEVER route tool operations to "chat".
- Only choose "chat" for pure conversation that needs no code or tool execution.

Respond with a JSON object (no markdown fences):
{{"decision": "chat"|"select_existing"|"generate_new", "clam_id": "<name or null>", "reason": "<brief reason>", "chat_response": "<response if chat>", "inputs": {{}}}}

When decision is "select_existing", populate "inputs" with values extracted from the \
user's message, matching the clam's input keys.
"""

REPAIR_PROMPT = """\
Your previous response was not valid JSON. Here is what you returned:
{bad_response}

Please respond with ONLY a valid JSON object matching this schema:
{{"decision": "chat"|"select_existing"|"generate_new", "clam_id": "<name or null>", "reason": "<brief reason>", "chat_response": "<response if decision is chat, else empty string>", "inputs": {{}}}}
"""


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


class ProviderBackedClamSelector:
    """Two-stage clam selector using pre-selection + LLM routing.

    Stage 1 (pre-selection): Normalize the request and check for exact
    matches against clam catalog ``source_request`` values.

    Stage 2 (LLM routing): Call a cheap/fast selector model to decide
    between chat, select_existing, and generate_new.
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        retries: int = 1,
    ) -> None:
        self._provider = provider
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._retries = retries

    async def select(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        link_context: str = "",
        clam_catalog: list[ClamSummary] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> SelectionResult:
        """Select the appropriate action for a user message.

        Args:
            message: The user's message.
            history: Conversation history (LLM format).
            system_prompt: The system prompt (for context).
            link_context: Pre-fetched link context.
            clam_catalog: List of available clam summaries.

        Returns:
            A SelectionResult with the routing decision.
        """
        catalog = clam_catalog or []

        # ── Stage 1: Pre-selection (exact match) ──────────────────
        normalized = normalize_request(message)
        for clam in catalog:
            if clam.source_request and normalize_request(clam.source_request) == normalized:
                logger.debug("Pre-selection match: %s → %s", message, clam.name)
                return SelectionResult(
                    decision="select_existing",
                    clam_id=clam.name,
                    reason="Pre-selection exact match",
                )

        # ── Stage 2: LLM routing ────────────────────────────────
        catalog_text = self._format_catalog(catalog)
        tools_text = self._format_tools(available_tools or [])
        selector_prompt = SELECTOR_SYSTEM_PROMPT.format(
            clam_catalog=catalog_text,
            available_tools=tools_text,
        )

        messages = [{"role": "system", "content": selector_prompt}]

        # Add recent history for context (last few turns)
        if history:
            for turn in history[-4:]:
                messages.append(turn)

        messages.append({"role": "user", "content": message})

        # Try with retries
        for attempt in range(1 + self._retries):
            try:
                response = await self._provider.acomplete(
                    messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )

                result = self._parse_response(response.content)
                if result is not None:
                    return result

                # Bad JSON — retry with repair prompt
                if attempt < self._retries:
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append(
                        {
                            "role": "user",
                            "content": REPAIR_PROMPT.format(bad_response=response.content),
                        }
                    )
                    continue

            except Exception as exc:
                logger.warning("Selector LLM call failed (attempt %d): %s", attempt, exc)
                if attempt >= self._retries:
                    break

        # Fallback: generate new
        return SelectionResult(
            decision="generate_new",
            reason="Selector failed — defaulting to generate_new",
        )

    def _parse_response(self, content: str) -> SelectionResult | None:
        """Parse the LLM's JSON response into a SelectionResult."""
        # Strip markdown fences if present
        text = strip_markdown_fences(content)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        decision = data.get("decision", "")
        if decision not in ("chat", "select_existing", "generate_new"):
            return None

        raw_inputs = data.get("inputs")
        inputs = raw_inputs if isinstance(raw_inputs, dict) else None

        return SelectionResult(
            decision=decision,
            clam_id=data.get("clam_id") or None,
            reason=data.get("reason", ""),
            chat_response=data.get("chat_response", ""),
            inputs=inputs,
        )

    @staticmethod
    def _format_tools(tools: list[dict[str, Any]]) -> str:
        """Format tool schemas into a concise capability summary."""
        if not tools:
            return "(no tools available)"
        lines: list[str] = []
        for tool_schema in tools:
            func = tool_schema.get("function", tool_schema)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def _format_catalog(catalog: list[ClamSummary]) -> str:
        """Format clam catalog for the selector prompt."""
        if not catalog:
            return "(no existing clams available)"

        lines: list[str] = []
        for clam in catalog:
            tools = ", ".join(clam.declared_tools) if clam.declared_tools else "none"
            desc = clam.description or "no description"
            inputs_str = json.dumps(clam.inputs) if clam.inputs else "none"
            usage = f"used {clam.usage_count}x" if clam.usage_count else "never used"
            lines.append(f"- {clam.name}: {desc} (tools: {tools}, inputs: {inputs_str}, {usage})")

        return "\n".join(lines)
