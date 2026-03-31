"""Agent loop — full pipeline: request → select → generate → execute → analyze → response.

Orchestrates the complete agent turn lifecycle including clam selection,
generation, WASM execution, post-runtime analysis, and the self-fix loop.

Multi-step requests (e.g. "check my credits and schedule it every morning")
are decomposed by a lightweight LLM planner into ordered sub-tasks that
execute sequentially, threading context (e.g. ``clam_id``) between steps.
"""

from __future__ import annotations

import json as _json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clambot.utils.text import strip_markdown_fences

from .analysis_trace import AnalysisTraceBuilder
from .chat_mode import ChatModeFallbackResponder
from .clams import Clam, ClamRegistry
from .runtime_backend_amla_sandbox import RuntimeResult
from .context import ContextBuilder
from .error_detail_context import build_error_detail_context
from .final_response import select_final_response
from .generation_grounding import apply_grounding_rules
from .post_runtime_analysis import PostRuntimeAnalysisDecision
from .progress import ProgressState
from .protocols import (
    AnalyzerProtocol,
    GeneratorProtocol,
    RuntimeProtocol,
    ToolRegistryProtocol,
)
from .selector import ProviderBackedClamSelector, SelectionResult
from .workspace_clam_writer import WorkspaceClamPersistenceWriter

# ---------------------------------------------------------------------------
# Task planning prompt
# ---------------------------------------------------------------------------

TASK_PLANNING_PROMPT = """\
Analyze the user request and return a JSON array of action objects.

Action types:
- "execute" — run code (fetch data, compute, call APIs).
- "transform" — apply an LLM-native text transformation (translate, \
  summarize, reformat, analyze) to the previous step's output. \
  No code is run — the LLM handles it directly.
- "schedule" — create a cron/reminder job.

Rules:
- Plain questions or one-off tasks → single "execute" action.
- Data-fetch + text transformation → "execute" first (fetch only), then \
  "transform" with an "instruction" field. The clam MUST NOT attempt \
  translation, summarization, or text analysis — that is the transform \
  step's job.
- Simple reminders (no data needed) → single "schedule" action. Do NOT \
  add a separate confirmation step.
- Data-fetch + scheduling → "execute" first, then "schedule" referencing \
  {previous_clam_id}. No extra confirmation step needed.

"schedule" objects MUST include:
- Exactly ONE of: "at_minutes_from_now" (int), "every_seconds" (int), \
  or "cron_expr" (string).
- "message", "delete_after_run", and optionally "clam_id".
- For one-shot reminders ("in X minutes"), ALWAYS use "at_minutes_from_now" \
  with "delete_after_run": true. NEVER use "every_seconds" for one-shot tasks.

Return ONLY a JSON array — no markdown, no explanation.

Examples:
"What's the weather?" → [{"action":"execute","task":"What's the weather?"}]
"Fetch example.com and translate to Russian" → [{"action":"execute","task":"Fetch page https://example.com/"},{"action":"transform","instruction":"Translate the content to Russian"}]
"Summarize this article: https://..." → [{"action":"execute","task":"Fetch page https://..."},{"action":"transform","instruction":"Summarize the article"}]
"Remind me to read a book in 5 min" → [{"action":"schedule","message":"Reminder: read a book","at_minutes_from_now":5,"delete_after_run":true}]
"Check my credits every morning at 9" → [{"action":"execute","task":"Check remaining OpenRouter credit balance"},{"action":"schedule","clam_id":"{previous_clam_id}","message":"Check OpenRouter credits","cron_expr":"0 9 * * *"}]
"Remind me my credits in 5 min" → [{"action":"execute","task":"Check remaining OpenRouter credit balance"},{"action":"schedule","clam_id":"{previous_clam_id}","message":"Check OpenRouter credits","at_minutes_from_now":5,"delete_after_run":true}]
"""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secret error detection
# ---------------------------------------------------------------------------

_SECRET_NOT_FOUND_RE = re.compile(r"Secret '([^']+)' not found")


def _detect_missing_secrets(runtime_result: Any) -> list[str]:
    """Detect missing secret names from a RuntimeResult.

    Checks two sources:
    1. Pre-flight secret check error (structured code in run_log)
    2. Runtime tool error pattern ``Secret '<name>' not found`` in the
       error string, output, or tool call results.

    Returns:
        List of missing secret names, empty if no secret error detected.
    """
    missing: list[str] = []

    # Source 1: structured pre-flight error
    run_log = getattr(runtime_result, "run_log", None) or {}
    error_detail = run_log.get("error", {})
    if isinstance(error_detail, dict):
        code = error_detail.get("code", "")
        if code == "pre_runtime_secret_requirements_unresolved":
            detail = error_detail.get("detail", {})
            return detail.get("missing_secrets", [])

    # Source 2: pattern match in error/output/tool_calls
    haystack_parts: list[str] = []
    error_str = getattr(runtime_result, "error", "") or ""
    output_str = getattr(runtime_result, "output", "") or ""
    haystack_parts.append(error_str)
    haystack_parts.append(output_str)

    # Also scan tool call results for the error pattern
    tool_calls = getattr(runtime_result, "tool_calls", []) or []
    for tc in tool_calls:
        result_val = tc.get("result", "") if isinstance(tc, dict) else ""
        if isinstance(result_val, dict):
            haystack_parts.append(str(result_val.get("error", "")))
        else:
            haystack_parts.append(str(result_val))

    haystack = "\n".join(haystack_parts)
    for match in _SECRET_NOT_FOUND_RE.finditer(haystack):
        name = match.group(1)
        if name not in missing:
            missing.append(name)

    return missing


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Result from a full agent turn."""

    content: str = ""
    status: str = "completed"  # completed, failed, chat, secret_pending
    selection_reason: str = ""
    runtime_result: Any = None
    analysis_result: Any = None
    clam_name: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    missing_secrets: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Full agent pipeline orchestrator.

    Coordinates the complete flow:
        1. Fetch link context
        2. Recall memory
        3. Build system prompt
        4. Select action (chat / select_existing / generate_new)
        5. Execute (chat response or clam execution)
        6. Post-runtime analysis (ACCEPT / SELF_FIX / REJECT)
        7. Self-fix loop (max 3 attempts)
        8. Promote accepted clams
    """

    def __init__(
        self,
        selector: ProviderBackedClamSelector,
        generator: GeneratorProtocol,
        runtime: RuntimeProtocol,
        analyzer: AnalyzerProtocol,
        tool_registry: ToolRegistryProtocol | None,
        context_builder: ContextBuilder,
        clam_registry: ClamRegistry,
        memory_workspace: Path,
        link_context_builder: Any | None = None,
        chat_responder: ChatModeFallbackResponder | None = None,
        config: Any | None = None,
    ) -> None:
        self._selector = selector
        self._generator = generator
        self._runtime = runtime
        self._analyzer = analyzer
        self._tool_registry = tool_registry
        self._context_builder = context_builder
        self._clam_registry = clam_registry
        self._workspace = Path(memory_workspace)
        self._link_context = link_context_builder
        self._chat_responder = chat_responder
        self._config = config

        self._writer = WorkspaceClamPersistenceWriter(self._workspace)
        self._cron_service: Any | None = None

        # Extract config
        defaults = getattr(config, "agents", None)
        if defaults:
            defaults = getattr(defaults, "defaults", None)
        self._max_self_fix = getattr(defaults, "max_self_fix_attempts", 3) if defaults else 3

    def set_cron_service(self, cron_service: Any) -> None:
        """Inject the cron service for direct scheduling by the agent."""
        self._cron_service = cron_service

    async def process_turn(
        self,
        message: str,
        session_key: str = "",
        history: list[dict[str, Any]] | None = None,
        config: Any | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentResult:
        """Process a full agent turn.

        Args:
            message: The user's message.
            session_key: Session identifier.
            history: Conversation history (LLM message format).
            config: Optional config override.
            on_event: Optional callback for progress events.

        Returns:
            An AgentResult with the response content and metadata.
        """
        events: list[dict[str, Any]] = []

        def emit(event_type: str, data: dict[str, Any] | None = None) -> None:
            event = {"type": event_type, **(data or {})}
            events.append(event)
            if on_event:
                on_event(event)

        # Reset turn-scoped approval grants so approvals given during
        # this request carry across tool calls and self-fix retries but
        # do not leak into the next user request.
        if hasattr(self._runtime, "begin_turn"):
            self._runtime.begin_turn()

        emit("progress", {"state": ProgressState.DISCOVERING.value})

        # ── 1. Link context ───────────────────────────────────────
        link_context = ""
        if self._link_context:
            try:
                link_context = await self._link_context.fetch(message)
            except Exception as exc:
                logger.debug("Link context fetch failed: %s", exc)

        # ── 2. Memory recall ──────────────────────────────────────
        from clambot.memory.store import memory_recall

        memory = memory_recall(self._workspace)

        # ── 3. System prompt ──────────────────────────────────────
        docs = self._context_builder.load_workspace_docs()
        clam_catalog = self._clam_registry.get_catalog()
        tool_schemas = self._tool_registry.get_schemas() if self._tool_registry else []

        system_prompt = self._context_builder.build_system_prompt(
            docs=docs,
            memory=memory,
            tools=tool_schemas,
            clam_catalog=clam_catalog,
            link_context=link_context,
        )

        # ── 4. Execute (select → generate/load → run → analyze) ───
        return await self._execute_single_task(
            message=message,
            history=history,
            system_prompt=system_prompt,
            link_context=link_context,
            on_event=on_event,
            events=events,
        )

    # ------------------------------------------------------------------
    # Task planning
    # ------------------------------------------------------------------

    async def _plan_tasks(self, message: str) -> list[dict[str, Any]]:
        """Use the selector LLM to produce a structured action plan.

        Returns a list of action dicts.  Each dict has an ``"action"``
        key — ``"execute"`` for clam generation/selection, or
        ``"schedule"`` for direct cron job creation.

        Falls back to ``[{"action": "execute", "task": message}]`` on
        any parse failure.
        """
        fallback: list[dict[str, Any]] = [{"action": "execute", "task": message}]
        try:
            response = await self._selector._provider.acomplete(
                [
                    {"role": "system", "content": TASK_PLANNING_PROMPT},
                    {"role": "user", "content": message},
                ],
                max_tokens=400,
            )
            # Strip markdown fences if present
            text = strip_markdown_fences(response.content)
            actions = _json.loads(text)
            if isinstance(actions, list) and len(actions) > 0:
                # Normalise: accept plain strings as execute actions
                normalised: list[dict[str, Any]] = []
                for item in actions:
                    if isinstance(item, str):
                        normalised.append({"action": "execute", "task": item})
                    elif isinstance(item, dict) and "action" in item:
                        normalised.append(item)
                if normalised:
                    return normalised
        except Exception as exc:
            logger.debug("Task planning failed, using single task: %s", exc)
        return fallback

    # ------------------------------------------------------------------
    # Multi-step execution
    # ------------------------------------------------------------------

    async def _execute_planned_tasks(
        self,
        plan: list[dict[str, Any]],
        history: list[dict[str, Any]] | None,
        system_prompt: str,
        link_context: str,
        on_event: Callable[[dict[str, Any]], None] | None,
        events: list[dict[str, Any]],
    ) -> AgentResult:
        """Execute an ordered list of action dicts.

        ``"execute"`` actions run through the normal clam pipeline.
        ``"transform"`` actions pass the previous output through the
        LLM for translation, summarization, etc. — no clam generated.
        ``"schedule"`` actions call ``cron_service.add_job()`` directly.
        """
        context: dict[str, str] = {}
        results: list[AgentResult] = []

        for step in plan:
            action = step.get("action", "execute")

            if action == "schedule":
                result = await self._handle_schedule_action(step, context)
            elif action == "transform":
                result = await self._handle_transform_action(step, context)
            else:
                task_msg = str(step.get("task", ""))
                # Inject context
                for key, value in context.items():
                    task_msg = task_msg.replace(f"{{{key}}}", value)

                result = await self._execute_single_task(
                    message=task_msg,
                    history=history,
                    system_prompt=system_prompt,
                    link_context=link_context,
                    on_event=on_event,
                    events=events,
                    _is_subtask=True,
                )

            # Accumulate context for next steps
            if result.clam_name:
                context["previous_clam_id"] = result.clam_name
            if result.content:
                context["previous_result"] = result.content

            results.append(result)

        # Combine outputs
        combined = "\n\n".join(r.content for r in results if r.content)
        last = results[-1] if results else AgentResult(status="failed")
        return AgentResult(
            content=combined,
            status=last.status,
            selection_reason=last.selection_reason,
            clam_name=last.clam_name,
            events=events,
        )

    async def _handle_schedule_action(
        self,
        step: dict[str, Any],
        context: dict[str, str],
    ) -> AgentResult:
        """Handle a ``"schedule"`` action by calling cron_service directly."""
        if self._cron_service is None:
            return AgentResult(
                content="Scheduling is not available.",
                status="failed",
            )

        from clambot.cron.schedule import parse_schedule

        message = step.get("message", "Scheduled task")
        clam_id = step.get("clam_id")
        # Resolve context placeholders in clam_id
        if clam_id:
            for key, value in context.items():
                clam_id = clam_id.replace(f"{{{key}}}", value)

        delete_after_run = step.get("delete_after_run", False)

        # Build schedule args from the action dict
        schedule_args: dict[str, Any] = {}
        if "cron_expr" in step:
            schedule_args["cron_expr"] = step["cron_expr"]
        elif "every_seconds" in step:
            schedule_args["every_seconds"] = step["every_seconds"]
        elif "at_minutes_from_now" in step:
            import time

            at_ms = int((time.time() + step["at_minutes_from_now"] * 60) * 1000)
            schedule_args["at_ms"] = at_ms

        if not schedule_args:
            return AgentResult(
                content="Could not determine schedule.",
                status="failed",
            )

        try:
            schedule = parse_schedule(schedule_args)
        except ValueError as exc:
            return AgentResult(
                content=f"Invalid schedule: {exc}",
                status="failed",
            )

        # Read channel/target from conversation context
        from clambot.bus.context import current_channel, current_chat_id

        channel = current_channel.get("") or None
        target = current_chat_id.get("") or None

        job = self._cron_service.add_job(
            name=message[:40],
            schedule=schedule,
            message=message,
            deliver=bool(channel),
            channel=channel,
            target=target,
            clam_id=clam_id,
            delete_after_run=delete_after_run,
        )

        kind = "once" if delete_after_run else "recurring"
        return AgentResult(
            content=f"Scheduled ({kind}): {message} (job {job.id})",
            status="completed",
        )

    async def _handle_transform_action(
        self,
        step: dict[str, Any],
        context: dict[str, str],
    ) -> AgentResult:
        """Apply an LLM-native text transformation to the previous output.

        Handles translation, summarization, reformatting, etc. without
        generating a clam — the selector LLM does the work directly.
        """
        instruction = step.get("instruction", "Transform the text")
        previous = context.get("previous_result", "")
        if not previous:
            return AgentResult(
                content="Nothing to transform — no previous output.",
                status="failed",
            )

        response = await self._selector._provider.acomplete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a text transformation assistant. "
                        "Apply the requested transformation to the given "
                        "text. Return ONLY the transformed result — no "
                        "explanations, wrappers, or commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": (f"Instruction: {instruction}\n\nText to transform:\n{previous}"),
                },
            ],
            max_tokens=4096,
        )

        return AgentResult(
            content=response.content.strip(),
            status="completed",
        )

    # ------------------------------------------------------------------
    # Direct tool execution (agent-level tools: cron, web_fetch)
    # ------------------------------------------------------------------

    _DIRECT_TOOLS: frozenset[str] = frozenset({"cron", "web_fetch"})

    async def _execute_direct_tool_from_script(
        self,
        message: str,
        script: str,
        declared_tools: list[str],
        events: list[dict[str, Any]],
    ) -> AgentResult:
        """Execute agent-only tool calls parsed from a generated script.

        The LLM already generated the script with tool calls like
        ``await cron({action: "list"})``. We parse the arguments and
        execute the tools directly via the Python tool registry — no
        WASM sandbox needed.
        """
        if not self._tool_registry:
            return AgentResult(content="No tools available.", status="failed", events=events)

        results: list[str] = []
        for tool_name in declared_tools:
            tool = self._tool_registry.get_tool(tool_name)
            if tool is None:
                results.append(f"Error: tool '{tool_name}' not found")
                continue

            # Parse tool args from the script: await tool_name({...})
            args = _extract_tool_args_from_script(script, tool_name)

            try:
                result = tool.execute(args)
                if isinstance(result, str):
                    results.append(result)
                else:
                    results.append(_json.dumps(result, ensure_ascii=False, default=str))
            except Exception as exc:
                results.append(f"Error executing {tool_name}: {exc}")

        raw_output = "\n".join(results)

        # Use post-runtime analyzer to format the result for the user
        if self._analyzer and raw_output:
            analysis = await self._analyzer.analyze(
                message=message,
                clam={"script": "(direct tool call)", "name": "direct-tool"},
                runtime_result={"output": raw_output, "error": "", "stderr": ""},
            )
            if analysis.decision == PostRuntimeAnalysisDecision.ACCEPT and analysis.output:
                return AgentResult(content=analysis.output, status="completed", events=events)

        return AgentResult(
            content=raw_output or "No scheduled tasks.",
            status="completed",
            events=events,
        )


    # ------------------------------------------------------------------
    # Single-task execution (select → generate/load → execute → analyze)
    # ------------------------------------------------------------------

    async def _execute_single_task(
        self,
        message: str,
        history: list[dict[str, Any]] | None,
        system_prompt: str,
        link_context: str,
        on_event: Callable[[dict[str, Any]], None] | None,
        events: list[dict[str, Any]],
        _is_subtask: bool = False,
    ) -> AgentResult:
        """Run one task through the full select → execute → analyze pipeline.

        When ``_is_subtask`` is ``False`` (top-level request) and the
        selector decides ``generate_new``, the task planner is invoked
        first.  If it decomposes the request into multiple steps,
        :meth:`_execute_planned_tasks` takes over.  Sub-tasks set
        ``_is_subtask=True`` to prevent recursive planning.
        """
        clam_catalog = self._clam_registry.get_catalog()
        all_tool_schemas = self._tool_registry.get_schemas() if self._tool_registry else []
        # Selector sees ALL tools (including agent-only ones like cron)
        # so it can route simple tool queries to chat mode.
        selection = await self._selector.select(
            message=message,
            history=history,
            system_prompt=system_prompt,
            link_context=link_context,
            clam_catalog=clam_catalog,
            available_tools=all_tool_schemas,
        )

        logger.info(
            "Selector: decision=%s, clam_id=%s, reason=%s",
            selection.decision,
            selection.clam_id or "none",
            selection.reason,
        )

        # Chat mode — always use chat_responder (has full system prompt
        # with memory context) rather than the selector's quick response,
        # which lacks memory/personality context.
        if selection.decision == "chat":
            if self._chat_responder:
                # Rebuild system prompt WITHOUT generation rules — those
                # contain JSON output format instructions that cause
                # coding-oriented models to wrap chat responses in JSON.
                from clambot.memory.store import memory_recall

                chat_prompt = self._context_builder.build_system_prompt(
                    docs=self._context_builder.load_workspace_docs(),
                    memory=memory_recall(self._workspace),
                    tools=all_tool_schemas,
                    clam_catalog=clam_catalog,
                    link_context=link_context,
                    generation_mode=False,
                )
                response = await self._chat_responder.respond(
                    message,
                    history,
                    chat_prompt,
                )
            else:
                response = selection.chat_response
            return AgentResult(
                content=response or "I don't have a response for that.",
                status="chat",
                selection_reason=selection.reason,
                events=events,
            )

        # Select existing clam
        if selection.decision == "select_existing" and selection.clam_id:
            clam = self._clam_registry.load(selection.clam_id)
            if clam is None:
                logger.warning("Selected clam %s not found, generating new", selection.clam_id)
                selection = SelectionResult(
                    decision="generate_new",
                    reason=f"Selected clam '{selection.clam_id}' not found",
                )
            else:
                if selection.inputs:
                    clam.inputs = selection.inputs
                return await self._execute_and_analyze(
                    message=message,
                    clam=clam,
                    history=history,
                    system_prompt=system_prompt,
                    link_context=link_context,
                    selection=selection,
                    on_event=on_event,
                    events=events,
                    skip_analysis=False,
                )

        # ── generate_new: check for multi-step or agent-handled actions ──
        if not _is_subtask:
            plan = await self._plan_tasks(message)
            has_agent_action = any(s.get("action") in ("schedule", "transform") for s in plan)
            if len(plan) > 1 or has_agent_action:
                logger.info("Planned actions: %s", plan)
                return await self._execute_planned_tasks(
                    plan=plan,
                    history=history,
                    system_prompt=system_prompt,
                    link_context=link_context,
                    on_event=on_event,
                    events=events,
                )

        # Generate new clam (single task)
        if on_event:
            on_event({"type": "progress", "state": ProgressState.GENERATING.value})

        return await self._generate_and_execute(
            message=message,
            history=history,
            system_prompt=system_prompt,
            link_context=link_context,
            selection=selection,
            on_event=on_event,
            events=events,
        )

    async def _generate_and_execute(
        self,
        message: str,
        history: list[dict[str, Any]] | None,
        system_prompt: str,
        link_context: str,
        selection: SelectionResult,
        on_event: Callable[[dict[str, Any]], None] | None,
        events: list[dict[str, Any]],
        self_fix_context: str = "",
        attempt: int = 0,
    ) -> AgentResult:
        """Generate a new clam and execute it, with self-fix loop."""
        logger.debug(
            "[attempt %d] Generating clam for: %.100s (self_fix=%s)",
            attempt,
            message,
            bool(self_fix_context),
        )

        # Generate
        gen_result = await self._generator.generate(
            message=message,
            history=history,
            system_prompt=system_prompt,
            link_context=link_context,
            self_fix_context=self_fix_context,
        )

        logger.debug(
            "[attempt %d] Generated: script=%d chars, declared_tools=%s, error=%s",
            attempt,
            len(gen_result.script or ""),
            gen_result.declared_tools,
            gen_result.error or "none",
        )

        # Apply grounding rules
        gen_result = apply_grounding_rules(gen_result)

        # Intercept: if generated clam ONLY uses agent-only tools (e.g. cron),
        # execute directly in the loop instead of the sandbox.
        if (
            gen_result.script
            and not gen_result.error
            and gen_result.declared_tools
            and all(t in self._DIRECT_TOOLS for t in gen_result.declared_tools)
        ):
            logger.info(
                "[attempt %d] Agent-only tools %s — executing directly",
                attempt,
                gen_result.declared_tools,
            )
            return await self._execute_direct_tool_from_script(
                message=message,
                script=gen_result.script,
                declared_tools=gen_result.declared_tools,
                events=events,
            )

        if gen_result.error:
            logger.warning(
                "[attempt %d] Grounding rejected: %s | script preview: %.200s",
                attempt,
                gen_result.error,
                gen_result.script or "",
            )

        if not gen_result.script:
            return AgentResult(
                content="I couldn't generate code for that request. Could you try rephrasing it?",
                status="failed",
                selection_reason=selection.reason,
                events=events,
            )

        # Grounding rejected the script (refusal, non-code, etc.) — retry
        if gen_result.error:
            if attempt < self._max_self_fix:
                return await self._generate_and_execute(
                    message=message,
                    history=history,
                    system_prompt=system_prompt,
                    link_context=link_context,
                    selection=selection,
                    on_event=on_event,
                    events=events,
                    self_fix_context=f"GROUNDING ERROR: {gen_result.error}",
                    attempt=attempt + 1,
                )
            logger.warning(
                "Grounding failed after %d attempts: %s",
                attempt,
                gen_result.error,
            )
            return AgentResult(
                content=(
                    "Sorry, I wasn't able to fulfill that request. "
                    "Could you try rephrasing it or breaking it into simpler steps?"
                ),
                status="failed",
                selection_reason=selection.reason,
                events=events,
            )

        # Write to build directory — prefer LLM-generated description for a
        # generic name (e.g. "add-two-numbers") over slugified raw request.
        description = (gen_result.metadata or {}).get("description", "")
        clam_name = self._writer.generate_clam_name(description or message)

        clam_md = self._build_clam_md(gen_result, message)
        self._writer.write_to_build(clam_name, gen_result.script, clam_md)

        # Create Clam object
        clam = Clam(
            name=clam_name,
            script=gen_result.script,
            declared_tools=gen_result.declared_tools,
            inputs=gen_result.inputs,
            metadata=gen_result.metadata,
            language=gen_result.language,
        )

        return await self._execute_and_analyze(
            message=message,
            clam=clam,
            history=history,
            system_prompt=system_prompt,
            link_context=link_context,
            selection=selection,
            on_event=on_event,
            events=events,
            attempt=attempt,
        )

    async def _execute_and_analyze(
        self,
        message: str,
        clam: Clam,
        history: list[dict[str, Any]] | None,
        system_prompt: str,
        link_context: str,
        selection: SelectionResult,
        on_event: Callable[[dict[str, Any]], None] | None,
        events: list[dict[str, Any]],
        attempt: int = 0,
        skip_analysis: bool = False,
    ) -> AgentResult:
        """Execute a clam and run post-runtime analysis with self-fix loop."""
        trace = AnalysisTraceBuilder()

        logger.debug(
            "[attempt %d] Executing clam '%s' (tools: %s)",
            attempt,
            clam.name,
            clam.declared_tools,
        )

        # Execute
        runtime_result = await self._runtime.execute(
            clam=clam,
            inputs=clam.inputs,
            on_event=on_event,
        )

        logger.debug(
            "[attempt %d] Runtime result: output=%d chars, error=%s, timed_out=%s",
            attempt,
            len(getattr(runtime_result, "output", "") or ""),
            (getattr(runtime_result, "error", "") or "")[:200] or "none",
            getattr(runtime_result, "timed_out", False),
        )

        # ── Secret errors — surface immediately, never self-fix ──
        # Check BEFORE entering the self-fix or analysis paths.
        # Covers both pre-flight failures (error only) and runtime
        # tool errors like "Secret 'X' not found" (may appear in
        # error, output, or tool_calls).
        missing = _detect_missing_secrets(runtime_result)
        if missing:
            user_msg = (
                f"Secret{'s' if len(missing) != 1 else ''} required: "
                f"{', '.join(missing)}. "
                f"Reply with the value to provide."
            )

            if on_event is not None:
                on_event(
                    {
                        "type": "secret_pending",
                        "missing_secrets": missing,
                        "message": user_msg,
                    }
                )

            return AgentResult(
                content=user_msg,
                status="secret_pending",
                selection_reason=selection.reason,
                runtime_result=runtime_result,
                clam_name=clam.name,
                events=events,
                missing_secrets=missing,
            )

        # Check for tool-level errors embedded in output.
        # Tools like web_fetch return {"error": "..."} which the clam
        # passes through as output.  Promote these to runtime errors
        # so the self-fix loop can retry with a corrected URL/request.
        if not runtime_result.error and runtime_result.output:
            tool_error = _extract_tool_error(runtime_result.output)
            if tool_error:
                runtime_result = RuntimeResult(
                    output="",
                    error=tool_error,
                    timed_out=runtime_result.timed_out,
                    tool_calls=getattr(runtime_result, "tool_calls", []),
                    run_log=getattr(runtime_result, "run_log", {}),
                )

        # Check for pre-flight/execution errors
        if runtime_result.error and not runtime_result.output:
            # ── Other errors — attempt self-fix ──
            if attempt < self._max_self_fix:
                error_context = build_error_detail_context(
                    error=None,
                    clam=clam,
                    result=runtime_result,
                )
                trace.record(attempt, "SELF_FIX", reason=runtime_result.error)

                return await self._generate_and_execute(
                    message=message,
                    history=history,
                    system_prompt=system_prompt,
                    link_context=link_context,
                    selection=selection,
                    on_event=on_event,
                    events=events,
                    self_fix_context=error_context,
                    attempt=attempt + 1,
                )

            logger.warning(
                "Runtime failed after %d attempts for clam '%s': %s",
                attempt + 1,
                clam.name,
                runtime_result.error,
            )
            return AgentResult(
                content=(
                    "Sorry, I ran into an error while processing that. "
                    "Could you try again or rephrase your request?"
                ),
                status="failed",
                selection_reason=selection.reason,
                runtime_result=runtime_result,
                clam_name=clam.name,
                events=events,
            )

        # Skip analysis for trusted reusable clams
        if skip_analysis:
            self._clam_registry.record_usage(clam.name)
            return AgentResult(
                content=runtime_result.output or "Done.",
                status="completed",
                selection_reason=selection.reason,
                runtime_result=runtime_result,
                clam_name=clam.name,
                events=events,
            )

        # Post-runtime analysis
        analysis_result = await self._analyzer.analyze(
            message=message,
            clam=clam,
            runtime_result=runtime_result,
        )

        logger.debug(
            "[attempt %d] Analysis decision: %s, reason: %s, output preview: %.200s",
            attempt,
            analysis_result.decision.value,
            analysis_result.reason,
            analysis_result.output[:200] if analysis_result.output else "none",
        )

        trace.record(attempt, analysis_result.decision.value, reason=analysis_result.reason)

        # Handle analysis decision
        if analysis_result.decision == PostRuntimeAnalysisDecision.SELF_FIX:
            if attempt < self._max_self_fix:
                error_context = build_error_detail_context(
                    error=None,
                    clam=clam,
                    result=runtime_result,
                )
                if analysis_result.fix_instructions:
                    error_context += f"\n\nFIX INSTRUCTIONS: {analysis_result.fix_instructions}"

                return await self._generate_and_execute(
                    message=message,
                    history=history,
                    system_prompt=system_prompt,
                    link_context=link_context,
                    selection=selection,
                    on_event=on_event,
                    events=events,
                    self_fix_context=error_context,
                    attempt=attempt + 1,
                )

            logger.warning(
                "Self-fix exhausted after %d attempts for clam '%s': %s",
                attempt + 1,
                clam.name,
                analysis_result.reason,
            )
            return AgentResult(
                content=(
                    "Sorry, I wasn't able to get that working. "
                    "Could you try a different approach or simplify the request?"
                ),
                status="failed",
                selection_reason=selection.reason,
                runtime_result=runtime_result,
                analysis_result=analysis_result,
                clam_name=clam.name,
                events=events,
            )

        if analysis_result.decision == PostRuntimeAnalysisDecision.REJECT:
            return AgentResult(
                content=analysis_result.output or "Request could not be completed.",
                status="failed",
                selection_reason=selection.reason,
                runtime_result=runtime_result,
                analysis_result=analysis_result,
                clam_name=clam.name,
                events=events,
            )

        # NEED_FULL_OUTPUT — re-run analysis with untruncated output
        if analysis_result.decision == PostRuntimeAnalysisDecision.NEED_FULL_OUTPUT:
            logger.info(
                "[attempt %d] Analysis requested full output — re-running with untruncated data",
                attempt,
            )
            analysis_result = await self._analyzer.analyze(
                message=message,
                clam=clam,
                runtime_result=runtime_result,
                full_output=True,
            )
            logger.debug(
                "[attempt %d] Full-output analysis decision: %s, reason: %s",
                attempt,
                analysis_result.decision.value,
                analysis_result.reason,
            )
            trace.record(
                attempt,
                analysis_result.decision.value,
                reason=f"(full output) {analysis_result.reason}",
            )
            # After full-output re-run, handle SELF_FIX/REJECT normally;
            # fall through to ACCEPT for anything else.
            if analysis_result.decision == PostRuntimeAnalysisDecision.SELF_FIX:
                if attempt < self._max_self_fix:
                    error_context = build_error_detail_context(
                        error=None,
                        clam=clam,
                        result=runtime_result,
                    )
                    if analysis_result.fix_instructions:
                        error_context += f"\n\nFIX INSTRUCTIONS: {analysis_result.fix_instructions}"
                    return await self._generate_and_execute(
                        message=message,
                        history=history,
                        system_prompt=system_prompt,
                        link_context=link_context,
                        selection=selection,
                        on_event=on_event,
                        events=events,
                        self_fix_context=error_context,
                        attempt=attempt + 1,
                    )
                logger.warning(
                    "Self-fix exhausted (full-output) after %d attempts for clam '%s': %s",
                    attempt + 1,
                    clam.name,
                    analysis_result.reason,
                )
                return AgentResult(
                    content=(
                        "Sorry, I wasn't able to get that working. "
                        "Could you try a different approach or simplify the request?"
                    ),
                    status="failed",
                    selection_reason=selection.reason,
                    runtime_result=runtime_result,
                    analysis_result=analysis_result,
                    clam_name=clam.name,
                    events=events,
                )
            if analysis_result.decision == PostRuntimeAnalysisDecision.REJECT:
                return AgentResult(
                    content=analysis_result.output or "Request could not be completed.",
                    status="failed",
                    selection_reason=selection.reason,
                    runtime_result=runtime_result,
                    analysis_result=analysis_result,
                    clam_name=clam.name,
                    events=events,
                )

        # ACCEPT — promote clam and record usage
        self._writer.promote(clam.name)
        self._clam_registry.record_usage(clam.name)

        final = select_final_response(analysis_result, runtime_result)
        return AgentResult(
            content=final,
            status="completed",
            selection_reason=selection.reason,
            runtime_result=runtime_result,
            analysis_result=analysis_result,
            clam_name=clam.name,
            events=events,
        )

    # ------------------------------------------------------------------
    # Direct clam execution (cron / scheduled jobs)
    # ------------------------------------------------------------------

    async def execute_clam_direct(
        self,
        clam_id: str,
        inputs: dict[str, Any] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentResult:
        """Execute a promoted clam directly by registry id.

        Bypasses selection, generation, post-runtime analysis, and the
        self-fix loop.  Intended for cron / scheduled jobs where the clam
        has already been accepted and promoted.

        Args:
            clam_id: Directory name of the clam under ``clams/``.
            inputs: Optional input overrides merged onto the clam's
                    stored defaults.
            on_event: Optional progress callback.

        Returns:
            An :class:`AgentResult` with the clam output or an error.
        """
        clam = self._clam_registry.load(clam_id)
        if clam is None:
            return AgentResult(
                content=f"Scheduled clam '{clam_id}' not found in registry.",
                status="failed",
                clam_name=clam_id,
            )

        if inputs:
            clam.inputs = {**clam.inputs, **inputs}

        runtime_result = await self._runtime.execute(
            clam=clam,
            inputs=clam.inputs,
            on_event=on_event,
        )

        output = runtime_result.output or ""
        error = runtime_result.error or ""

        if error and not output:
            logger.warning("Scheduled clam '%s' failed: %s", clam_id, error)
            return AgentResult(
                content=f"The scheduled task '{clam_id}' ran into an error. Please check the logs.",
                status="failed",
                runtime_result=runtime_result,
                clam_name=clam_id,
            )

        self._clam_registry.record_usage(clam_id)
        return AgentResult(
            content=output or "Done.",
            status="completed",
            runtime_result=runtime_result,
            clam_name=clam_id,
        )

    @staticmethod
    def _build_clam_md(gen_result: Any, request: str) -> str:
        """Build CLAM.md content from a generation result."""
        import json as _json

        lines = ["---"]

        metadata = gen_result.metadata or {}
        lines.append(f'description: "{metadata.get("description", request)}"')
        lines.append(f"language: {gen_result.language}")

        if gen_result.declared_tools:
            lines.append("declared_tools:")
            for tool in gen_result.declared_tools:
                lines.append(f"  - {tool}")

        if getattr(gen_result, "inputs", None):
            lines.append(f"inputs: {_json.dumps(gen_result.inputs)}")

        if metadata.get("reusable"):
            lines.append("reusable: true")
        if metadata.get("source_request"):
            lines.append(f'source_request: "{metadata["source_request"]}"')

        lines.append("---")
        lines.append("")
        lines.append(metadata.get("description", request))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers (module-level)
# ---------------------------------------------------------------------------


def _extract_tool_args_from_script(script: str, tool_name: str) -> dict[str, Any]:
    """Extract tool call arguments from a generated JavaScript script.

    Parses patterns like ``await tool_name({key: "value", ...})`` and
    returns the args dict.  Falls back to ``{}`` if parsing fails.
    """
    import re

    # Match: await tool_name({ ... })  or  tool_name({ ... })
    pattern = rf"(?:await\s+)?{re.escape(tool_name)}\s*\(\s*(\{{[^}}]*\}})\s*\)"
    match = re.search(pattern, script, re.DOTALL)
    if not match:
        return {}

    raw = match.group(1)
    # JS object → JSON: add quotes around bare keys
    json_str = re.sub(r'(\w+)\s*:', r'"\1":', raw)
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", '"')

    try:
        return _json.loads(json_str)
    except _json.JSONDecodeError:
        return {}


def _extract_tool_error(output: str) -> str:
    """Detect tool-level errors embedded in clam output.

    Tools return structured JSON with an ``"error"`` key on failure.
    If the output is a JSON object (or line-separated JSON objects)
    containing a non-empty ``"error"`` field, return the error message.
    """
    text = output.strip()
    if not text:
        return ""

    # Try parsing as a single JSON object
    try:
        data = _json.loads(text)
        if isinstance(data, dict):
            err = data.get("error", "")
            if err and isinstance(err, str):
                return err
    except (_json.JSONDecodeError, ValueError):
        pass

    return ""
