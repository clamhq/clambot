"""Integration tests for the cron subsystem — Phase 14.

End-to-end tests verifying that cron jobs trigger agent turns and produce
outbound messages, and that the cron tool properly persists jobs.

Tests:
- Manually trigger cron job → agent runs → outbound message produced
- Add job via cron tool → job persisted to jobs.json → appears in cron list
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clambot.bus.events import InboundMessage, OutboundMessage
from clambot.bus.queue import MessageBus
from clambot.cron.service import InMemoryCronService, configure_cron_tool_runtime_sync_hook
from clambot.cron.store import load_cron_store
from clambot.cron.types import (
    CronJob,
    CronSchedule,
)
from clambot.gateway.orchestrator import GatewayOrchestrator
from clambot.session.manager import SessionManager


def _ollama_reachable() -> bool:
    """Return True only if the Ollama server is reachable and has the configured model."""
    host = os.environ.get("OLLAMA_HOST", "")
    model = os.environ.get("OLLAMA_MODEL", "")
    if not host or not model:
        return False
    try:
        import urllib.request

        url = f"{host}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read())
            available = [m.get("name", "") for m in data.get("models", [])]
            return any(model in name for name in available)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace directory structure."""
    ws = tmp_path / "workspace"
    for subdir in ("clams", "build", "sessions", "logs", "docs", "memory"):
        (ws / subdir).mkdir(parents=True, exist_ok=True)
    (ws / "memory" / "MEMORY.md").write_text("", encoding="utf-8")
    (ws / "memory" / "HISTORY.md").write_text("", encoding="utf-8")
    return ws


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Test: Manually trigger cron job → agent runs → outbound produced
# ---------------------------------------------------------------------------


class TestCronJobExecution:
    """Manually triggered cron job runs the agent and produces output."""

    @pytest.mark.asyncio
    async def test_manual_trigger_produces_outbound(self, tmp_path: Path) -> None:
        """Running a cron job manually triggers the executor callback
        and produces an outbound message via the orchestrator."""
        workspace = _make_workspace(tmp_path)
        store_path = workspace / "jobs.json"

        cron_service = InMemoryCronService(store_path=store_path, workspace=workspace)
        await cron_service.start()

        # Add a job
        schedule = CronSchedule(kind="every", every_seconds=3600)
        job = cron_service.add_job(
            name="test-job",
            schedule=schedule,
            message="What is the weather today?",
            channel="telegram",
            target="123",
        )

        # Track executor calls
        executor_calls: list[CronJob] = []

        async def mock_executor(executed_job: CronJob) -> str | None:
            executor_calls.append(executed_job)
            return "Sunny and warm!"

        cron_service.set_executor(mock_executor)

        # Manually trigger the job
        result = await cron_service.run_job(job.id)
        assert result is True

        # Executor was called
        assert len(executor_calls) == 1
        assert executor_calls[0].name == "test-job"

        # Job state was updated
        updated_jobs = cron_service.list_jobs(include_disabled=True)
        triggered_job = next((j for j in updated_jobs if j.id == job.id), None)
        assert triggered_job is not None
        assert triggered_job.state.last_run_at_ms is not None
        assert triggered_job.state.last_status == "ok"

        cron_service.stop()

    @pytest.mark.asyncio
    async def test_cron_executor_wired_to_orchestrator(self, tmp_path: Path) -> None:
        """Cron executor calls orchestrator.process_inbound_async to run
        agent turns — full wiring end-to-end."""
        workspace = _make_workspace(tmp_path)
        store_path = workspace / "jobs.json"
        bus = MessageBus()

        # Mock agent loop
        mock_agent_loop = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = "Weather: Sunny 72°F"
        mock_result.status = "completed"
        mock_agent_loop.process_turn = AsyncMock(return_value=mock_result)

        session_manager = SessionManager(workspace)

        # Build orchestrator
        orch = GatewayOrchestrator(
            bus=bus,
            session_manager=session_manager,
            approval_gate=MagicMock(),
            agent_loop=mock_agent_loop,
            workspace=workspace,
        )

        # Wire cron → orchestrator
        cron_service = InMemoryCronService(store_path=store_path, workspace=workspace)
        await cron_service.start()

        async def cron_executor(job: CronJob) -> str | None:
            inbound = InboundMessage(
                channel=job.payload.channel or "cron",
                source="cron",
                chat_id=job.payload.target or "system",
                content=job.payload.message,
                metadata={"cron_job_id": job.id},
            )
            with patch(
                "clambot.gateway.orchestrator.process_turn_with_persistence_and_execution",
                new_callable=AsyncMock,
                return_value=OutboundMessage(
                    channel=inbound.channel,
                    target=inbound.chat_id,
                    content="Weather: Sunny 72°F",
                    correlation_id=inbound.correlation_id,
                ),
            ):
                result = await orch.process_inbound_async(inbound)
            return result.content if result else None

        cron_service.set_executor(cron_executor)

        # Add and trigger a job
        schedule = CronSchedule(kind="every", every_seconds=3600)
        job = cron_service.add_job(
            name="weather-check",
            schedule=schedule,
            message="Check the weather",
            channel="telegram",
            target="123",
        )

        await cron_service.run_job(job.id)

        # Outbound bus should have the response
        outbound_messages = []
        while not bus.outbound.empty():
            outbound_messages.append(bus.outbound.get_nowait())

        # Find the agent response (filter out status updates)
        agent_response = next(
            (m for m in outbound_messages if "Weather" in (m.content or "")),
            None,
        )
        assert agent_response is not None, (
            "Expected weather response in outbound bus. "
            f"Got: {[m.content for m in outbound_messages]}"
        )

        cron_service.stop()

    @pytest.mark.asyncio
    async def test_due_job_fires_in_scheduler_loop(self, tmp_path: Path) -> None:
        """A job with next_run_at_ms in the past fires when the scheduler
        loop runs."""
        workspace = _make_workspace(tmp_path)
        store_path = workspace / "jobs.json"

        cron_service = InMemoryCronService(store_path=store_path, workspace=workspace)
        await cron_service.start()

        # Add a job that is already due (next_run_at_ms in the past)
        schedule = CronSchedule(kind="at", at_ms=_now_ms() - 1000)
        job = cron_service.add_job(
            name="overdue-job",
            schedule=schedule,
            message="Do something now!",
            delete_after_run=True,
        )

        executor_calls: list[str] = []

        async def mock_executor(executed_job: CronJob) -> str | None:
            executor_calls.append(executed_job.id)
            return "Done!"

        cron_service.set_executor(mock_executor)

        # Run the scheduler loop briefly
        task = asyncio.create_task(cron_service._run())
        await asyncio.sleep(0.2)
        cron_service.stop()

        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (TimeoutError, asyncio.CancelledError):
            pass

        # The overdue job should have fired
        assert job.id in executor_calls

        # Job was delete_after_run → should be removed
        remaining = cron_service.list_jobs(include_disabled=True)
        assert all(j.id != job.id for j in remaining)

    @pytest.mark.asyncio
    async def test_delete_after_run_works_for_recurring_jobs(self, tmp_path: Path) -> None:
        """delete_after_run removes recurring ('every') jobs after the first
        execution — not just one-shot 'at' jobs.

        Regression test for the bug where delete_after_run was only checked
        inside ``if job.schedule.kind == "at":``, causing recurring jobs
        with delete_after_run=True to fire forever.
        """
        workspace = _make_workspace(tmp_path)
        store_path = workspace / "jobs.json"

        cron_service = InMemoryCronService(store_path=store_path, workspace=workspace)
        await cron_service.start()

        # Recurring job with delete_after_run — should fire once then vanish
        schedule = CronSchedule(kind="every", every_seconds=300)
        job = cron_service.add_job(
            name="one-shot-recurring",
            schedule=schedule,
            message="Fire once then delete",
            delete_after_run=True,
        )

        executor_calls: list[str] = []

        async def mock_executor(executed_job: CronJob) -> str | None:
            executor_calls.append(executed_job.id)
            return "Done!"

        cron_service.set_executor(mock_executor)

        # Manually trigger the job (simulates the scheduler loop firing it)
        await cron_service.run_job(job.id)

        # Job should have executed
        assert job.id in executor_calls

        # Job should be GONE — not just disabled, but removed from store
        remaining = cron_service.list_jobs(include_disabled=True)
        assert all(j.id != job.id for j in remaining), (
            f"Job {job.id} should have been deleted after run, "
            f"but is still in store: {[j.id for j in remaining]}"
        )

        # Verify disk is clean
        store = load_cron_store(store_path)
        assert all(j.id != job.id for j in store.jobs)

        cron_service.stop()

    @pytest.mark.asyncio
    async def test_executor_error_recorded_in_job_state(self, tmp_path: Path) -> None:
        """If the executor raises, the error is recorded in the job state."""
        workspace = _make_workspace(tmp_path)
        store_path = workspace / "jobs.json"

        cron_service = InMemoryCronService(store_path=store_path, workspace=workspace)
        await cron_service.start()

        schedule = CronSchedule(kind="every", every_seconds=3600)
        job = cron_service.add_job(
            name="failing-job",
            schedule=schedule,
            message="This will fail",
        )

        async def failing_executor(executed_job: CronJob) -> str | None:
            raise RuntimeError("Executor exploded!")

        cron_service.set_executor(failing_executor)

        await cron_service.run_job(job.id)

        # Job state should record the error
        jobs = cron_service.list_jobs(include_disabled=True)
        failed_job = next((j for j in jobs if j.id == job.id), None)
        assert failed_job is not None
        assert failed_job.state.last_status == "error"
        assert "exploded" in (failed_job.state.last_error or "").lower()

        cron_service.stop()


# ---------------------------------------------------------------------------
# Test: Add job via cron tool → persisted → appears in list
# ---------------------------------------------------------------------------


class TestCronToolIntegration:
    """Cron tool (wired to live service) persists jobs and lists them."""

    @pytest.mark.asyncio
    async def test_add_job_via_tool_persists_to_disk(self, tmp_path: Path) -> None:
        """Adding a job via the cron tool sync hook persists it to jobs.json
        and the job appears in the service's list."""
        workspace = _make_workspace(tmp_path)
        store_path = workspace / "jobs.json"

        cron_service = InMemoryCronService(store_path=store_path, workspace=workspace)
        await cron_service.start()

        # Create a mock cron tool and wire it
        mock_cron_tool = MagicMock()
        sync_hook_holder: list = []
        mock_cron_tool.set_sync_hook = MagicMock(side_effect=lambda fn: sync_hook_holder.append(fn))

        configure_cron_tool_runtime_sync_hook(mock_cron_tool, cron_service)

        assert len(sync_hook_holder) == 1
        sync_hook = sync_hook_holder[0]

        # Add a job via the sync hook
        result = sync_hook(
            {
                "action": "add",
                "name": "daily-report",
                "message": "Generate daily report",
                "every_seconds": 86400,
            }
        )

        assert result["ok"] is True
        assert result["name"] == "daily-report"
        job_id = result["job_id"]

        # Job should be in the service's list
        jobs = cron_service.list_jobs()
        assert any(j.id == job_id for j in jobs)

        # Job should be persisted to disk
        store = load_cron_store(store_path)
        assert any(j.id == job_id for j in store.jobs)

        cron_service.stop()

    @pytest.mark.asyncio
    async def test_list_jobs_via_tool(self, tmp_path: Path) -> None:
        """Listing jobs via the cron tool sync hook returns all jobs."""
        workspace = _make_workspace(tmp_path)
        store_path = workspace / "jobs.json"

        cron_service = InMemoryCronService(store_path=store_path, workspace=workspace)
        await cron_service.start()

        # Add some jobs directly
        cron_service.add_job(
            name="job-alpha",
            schedule=CronSchedule(kind="every", every_seconds=60),
            message="Do alpha",
        )
        cron_service.add_job(
            name="job-beta",
            schedule=CronSchedule(kind="every", every_seconds=120),
            message="Do beta",
        )

        # Wire tool
        mock_cron_tool = MagicMock()
        sync_hook_holder: list = []
        mock_cron_tool.set_sync_hook = MagicMock(side_effect=lambda fn: sync_hook_holder.append(fn))
        configure_cron_tool_runtime_sync_hook(mock_cron_tool, cron_service)
        sync_hook = sync_hook_holder[0]

        # List via sync hook
        result = sync_hook({"action": "list"})

        assert "jobs" in result
        names = [j["name"] for j in result["jobs"]]
        assert "job-alpha" in names
        assert "job-beta" in names

        cron_service.stop()

    @pytest.mark.asyncio
    async def test_remove_job_via_tool(self, tmp_path: Path) -> None:
        """Removing a job via the sync hook removes it from disk."""
        workspace = _make_workspace(tmp_path)
        store_path = workspace / "jobs.json"

        cron_service = InMemoryCronService(store_path=store_path, workspace=workspace)
        await cron_service.start()

        job = cron_service.add_job(
            name="ephemeral",
            schedule=CronSchedule(kind="every", every_seconds=60),
            message="Temporary",
        )

        mock_cron_tool = MagicMock()
        sync_hook_holder: list = []
        mock_cron_tool.set_sync_hook = MagicMock(side_effect=lambda fn: sync_hook_holder.append(fn))
        configure_cron_tool_runtime_sync_hook(mock_cron_tool, cron_service)
        sync_hook = sync_hook_holder[0]

        # Remove the job
        result = sync_hook({"action": "remove", "job_id": job.id})
        assert result["ok"] is True

        # Should no longer appear in list
        jobs = cron_service.list_jobs(include_disabled=True)
        assert all(j.id != job.id for j in jobs)

        # Should not be on disk
        store = load_cron_store(store_path)
        assert all(j.id != job.id for j in store.jobs)

        cron_service.stop()

    @pytest.mark.asyncio
    async def test_add_job_with_cron_expression(self, tmp_path: Path) -> None:
        """Adding a job with a cron expression schedules it correctly."""
        workspace = _make_workspace(tmp_path)
        store_path = workspace / "jobs.json"

        cron_service = InMemoryCronService(store_path=store_path, workspace=workspace)
        await cron_service.start()

        mock_cron_tool = MagicMock()
        sync_hook_holder: list = []
        mock_cron_tool.set_sync_hook = MagicMock(side_effect=lambda fn: sync_hook_holder.append(fn))
        configure_cron_tool_runtime_sync_hook(mock_cron_tool, cron_service)
        sync_hook = sync_hook_holder[0]

        result = sync_hook(
            {
                "action": "add",
                "name": "morning-check",
                "message": "Good morning check",
                "cron": "0 9 * * *",
                "timezone": "UTC",
            }
        )

        assert result["ok"] is True
        assert result["next_run_at_ms"] is not None
        assert result["next_run_at_ms"] > _now_ms()

        cron_service.stop()


# ---------------------------------------------------------------------------
# Test: Chat responder creates / removes cron tasks via function calling
# ---------------------------------------------------------------------------


class TestCronManagementViaChatMode:
    """Integration: cron task create/remove through direct tool execution.

    Chat mode no longer has tools — cron is handled by the agent loop's
    direct tool execution path.  This test validates the cron tool
    directly (no LLM needed for tool dispatch).

    Requires:
        - Ollama running at ``OLLAMA_HOST`` (from ``.env.test``)
        - Model ``OLLAMA_MODEL`` available on the Ollama instance
    """

    @pytest.mark.skipif(
        not _ollama_reachable(),
        reason="Ollama not reachable or model not available (set OLLAMA_HOST and OLLAMA_MODEL)",
    )
    @pytest.mark.asyncio
    async def test_create_and_remove_cron_task(self, tmp_path: Path) -> None:
        """Direct cron tool execution round-trip:

        1. Add a cron task via direct tool execution → job persisted.
        2. List tasks → job visible.
        3. Remove the task → job deleted from service and disk.
        """
        from clambot.tools.cron.operations import CronTool

        workspace = _make_workspace(tmp_path)
        store_path = workspace / "cron" / "jobs.json"
        store_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Real cron service + tool ──────────────────────────
        cron_service = InMemoryCronService(
            store_path=store_path,
            workspace=workspace,
        )
        await cron_service.start()

        cron_tool = CronTool()
        configure_cron_tool_runtime_sync_hook(cron_tool, cron_service)

        # ── Step 1: create the cron task (direct tool call) ───
        add_result = cron_tool.execute({
            "action": "add",
            "message": "Check OpenRouter credits",
            "every_seconds": 3600,
        })
        assert add_result.get("ok") is True, f"Add failed: {add_result}"

        jobs = cron_service.list_jobs()
        assert len(jobs) >= 1, f"Expected at least 1 job after creation, got {len(jobs)}"
        created_job = jobs[0]
        assert created_job.schedule.every_seconds == 3600

        # ── Step 2: list tasks ────────────────────────────────
        list_result = cron_tool.execute({"action": "list"})
        assert isinstance(list_result.get("jobs"), list)
        assert len(list_result["jobs"]) >= 1

        # ── Step 3: remove the task ───────────────────────────
        for job in jobs:
            remove_result = cron_tool.execute({"action": "remove", "job_id": job.id})
            assert remove_result.get("ok") is True, f"Remove failed: {remove_result}"

        remaining = cron_service.list_jobs(include_disabled=True)
        assert len(remaining) == 0, f"Expected 0 jobs after removal, got {len(remaining)}"

        # Verify disk is clean
        store = load_cron_store(store_path)
        assert len(store.jobs) == 0

        cron_service.stop()
