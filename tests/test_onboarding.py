"""Tests for Phase 13 — Gateway Startup + CLI + Onboarding.

Covers:
- workspace/bootstrap.py — bootstrap_workspace idempotency
- workspace/onboard.py — onboard_workspace provider detection
- workspace/retention.py — prune_session_logs
- heartbeat/service.py — InMemoryHeartbeatService skip logic + actionable trigger
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# workspace/bootstrap.py — bootstrap_workspace
# ---------------------------------------------------------------------------


class TestBootstrapWorkspace:
    """Tests for bootstrap_workspace()."""

    def test_creates_all_directories(self, tmp_path: Path) -> None:
        """bootstrap_workspace creates all expected subdirectories."""
        from clambot.workspace.bootstrap import bootstrap_workspace

        ws = tmp_path / "workspace"
        bootstrap_workspace(ws)

        expected_dirs = ["clams", "build", "sessions", "logs", "docs", "memory"]
        for dirname in expected_dirs:
            assert (ws / dirname).is_dir(), f"Missing directory: {dirname}"

    def test_creates_seed_files(self, tmp_path: Path) -> None:
        """bootstrap_workspace creates MEMORY.md, HISTORY.md, HEARTBEAT.md."""
        from clambot.workspace.bootstrap import bootstrap_workspace

        ws = tmp_path / "workspace"
        bootstrap_workspace(ws)

        assert (ws / "memory" / "MEMORY.md").exists()
        assert (ws / "memory" / "HISTORY.md").exists()
        assert (ws / "memory" / "HEARTBEAT.md").exists()

    def test_idempotent_on_second_call(self, tmp_path: Path) -> None:
        """Calling bootstrap_workspace twice does not error or overwrite files."""
        from clambot.workspace.bootstrap import bootstrap_workspace

        ws = tmp_path / "workspace"
        bootstrap_workspace(ws)

        # Write custom content
        (ws / "memory" / "MEMORY.md").write_text("custom content")

        # Second call
        bootstrap_workspace(ws)

        # Custom content preserved
        assert (ws / "memory" / "MEMORY.md").read_text() == "custom content"

    def test_directories_exist_after_idempotent_call(self, tmp_path: Path) -> None:
        """All directories still exist after second call."""
        from clambot.workspace.bootstrap import bootstrap_workspace

        ws = tmp_path / "workspace"
        bootstrap_workspace(ws)
        bootstrap_workspace(ws)

        expected_dirs = ["clams", "build", "sessions", "logs", "docs", "memory"]
        for dirname in expected_dirs:
            assert (ws / dirname).is_dir()


# ---------------------------------------------------------------------------
# workspace/onboard.py — onboard_workspace
# ---------------------------------------------------------------------------


class TestOnboardWorkspace:
    """Tests for onboard_workspace()."""

    def test_generates_config_with_detected_env_keys(self, tmp_path: Path) -> None:
        """onboard_workspace detects API keys from environment and writes config."""
        from clambot.workspace.onboard import onboard_workspace

        config_path = tmp_path / "config.json"

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-123"}, clear=False):
            summary = onboard_workspace(config_path)

        assert config_path.exists()
        data = json.loads(config_path.read_text())

        assert "openrouter" in summary["configured_providers"]
        assert data["providers"]["openrouter"]["apiKey"] == "test-key-123"

    def test_idempotent_no_overwrite(self, tmp_path: Path) -> None:
        """onboard_workspace does not overwrite existing provider values."""
        from clambot.workspace.onboard import onboard_workspace

        config_path = tmp_path / "config.json"

        # Write initial config with existing key
        initial = {"providers": {"openrouter": {"apiKey": "existing-key"}}}
        config_path.write_text(json.dumps(initial))

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "new-key"}, clear=False):
            summary = onboard_workspace(config_path)

        data = json.loads(config_path.read_text())
        # Existing key should NOT be overwritten
        assert data["providers"]["openrouter"]["apiKey"] == "existing-key"

    def test_ollama_detection(self, tmp_path: Path) -> None:
        """onboard_workspace detects Ollama when probe succeeds."""
        from clambot.workspace.onboard import onboard_workspace

        config_path = tmp_path / "config.json"

        with patch("clambot.workspace.onboard._probe_ollama", return_value=["llama3:latest"]):
            with patch("clambot.workspace.onboard._select_default_model", return_value=(None, None)):
                with patch.dict(os.environ, {}, clear=False):
                    summary = onboard_workspace(config_path)

        assert summary["ollama_detected"] is True
        assert "ollama" in summary["configured_providers"]

    def test_no_providers_detected(self, tmp_path: Path) -> None:
        """onboard_workspace handles no providers gracefully."""
        from clambot.workspace.onboard import onboard_workspace

        config_path = tmp_path / "config.json"

        # Clear all provider env vars
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "OPENROUTER_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "GROQ_API_KEY",
                "GEMINI_API_KEY",
            )
        }

        with patch.dict(os.environ, env_clean, clear=True):
            with patch("clambot.workspace.onboard._probe_ollama", return_value=[]):
                summary = onboard_workspace(config_path)

        assert summary["configured_providers"] == []
        assert config_path.exists()  # Config still created with defaults


# ---------------------------------------------------------------------------
# cli/commands.py — provider connect ollama
# ---------------------------------------------------------------------------


class TestProviderConnectOllama:
    """Tests for the Ollama provider connect helpers."""

    def test_probe_ollama_models_success(self) -> None:
        """_probe_ollama_models returns model list when Ollama is reachable."""
        from clambot.cli.commands import _probe_ollama_models

        fake_response = json.dumps(
            {
                "models": [
                    {"name": "llama3:latest", "size": 4_000_000_000},
                    {"name": "codellama:7b", "size": 3_800_000_000},
                ]
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            models = _probe_ollama_models("http://localhost:11434")

        assert models is not None
        assert len(models) == 2
        assert models[0]["name"] == "llama3:latest"

    def test_probe_ollama_models_unreachable(self) -> None:
        """_probe_ollama_models returns None when Ollama is not reachable."""
        from clambot.cli.commands import _probe_ollama_models

        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError):
            models = _probe_ollama_models("http://localhost:11434")

        assert models is None

    def test_probe_ollama_models_empty(self) -> None:
        """_probe_ollama_models returns empty list when no models pulled."""
        from clambot.cli.commands import _probe_ollama_models

        fake_response = json.dumps({"models": []}).encode()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            models = _probe_ollama_models("http://localhost:11434")

        assert models == []

    def test_update_ollama_config_creates_new(self, tmp_path: Path) -> None:
        """_update_ollama_config creates config.json if it doesn't exist."""
        from clambot.cli.commands import _update_ollama_config

        config_path = tmp_path / "config.json"
        _update_ollama_config(config_path, "http://myhost:11434", "ollama/llama3")

        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["providers"]["ollama"]["apiBase"] == "http://myhost:11434"
        assert data["agents"]["defaults"]["model"] == "ollama/llama3"

    def test_update_ollama_config_preserves_existing(self, tmp_path: Path) -> None:
        """_update_ollama_config preserves existing config fields."""
        from clambot.cli.commands import _update_ollama_config

        config_path = tmp_path / "config.json"
        existing = {
            "providers": {"openrouter": {"apiKey": "my-key"}},
            "agents": {"defaults": {"maxTokens": 4096}},
        }
        config_path.write_text(json.dumps(existing))

        _update_ollama_config(config_path, "http://localhost:11434", "ollama/codellama:7b")

        data = json.loads(config_path.read_text())
        # Existing provider preserved
        assert data["providers"]["openrouter"]["apiKey"] == "my-key"
        # Ollama added
        assert data["providers"]["ollama"]["apiBase"] == "http://localhost:11434"
        # Model set, existing field preserved
        assert data["agents"]["defaults"]["model"] == "ollama/codellama:7b"
        assert data["agents"]["defaults"]["maxTokens"] == 4096

    def test_update_ollama_config_no_default_model(self, tmp_path: Path) -> None:
        """_update_ollama_config skips model when default_model is None."""
        from clambot.cli.commands import _update_ollama_config

        config_path = tmp_path / "config.json"
        existing = {"agents": {"defaults": {"model": "anthropic/claude-sonnet-4-20250514"}}}
        config_path.write_text(json.dumps(existing))

        _update_ollama_config(config_path, "http://localhost:11434", None)

        data = json.loads(config_path.read_text())
        assert data["providers"]["ollama"]["apiBase"] == "http://localhost:11434"
        # Original model preserved
        assert data["agents"]["defaults"]["model"] == "anthropic/claude-sonnet-4-20250514"

    def test_connect_ollama_unreachable_exits(self) -> None:
        """_connect_ollama raises typer.Exit when Ollama is unreachable."""
        import typer

        from clambot.cli.commands import _connect_ollama

        with patch("clambot.cli.commands._probe_ollama_models", return_value=None):
            with pytest.raises(typer.Exit):
                _connect_ollama(host="http://bad-host:11434", set_default=True)

    def test_connect_ollama_no_models_exits(self) -> None:
        """_connect_ollama raises typer.Exit when Ollama has no models."""
        import typer

        from clambot.cli.commands import _connect_ollama

        with patch("clambot.cli.commands._probe_ollama_models", return_value=[]):
            with pytest.raises(typer.Exit):
                _connect_ollama(host="http://localhost:11434", set_default=True)

    def test_connect_ollama_full_flow(self, tmp_path: Path) -> None:
        """_connect_ollama writes config when model is selected."""
        from clambot.cli.commands import _connect_ollama

        config_path = tmp_path / "config.json"
        models = [
            {"name": "llama3:latest", "size": 4_000_000_000},
            {"name": "codellama:7b", "size": 3_800_000_000},
        ]

        with (
            patch("clambot.cli.commands._probe_ollama_models", return_value=models),
            patch("questionary.select") as mock_select,
        ):
            mock_select.return_value.ask.return_value = "llama3:latest"
            _connect_ollama(
                host="http://localhost:11434", set_default=True, config_path=str(config_path)
            )

        data = json.loads(config_path.read_text())
        assert data["providers"]["ollama"]["apiBase"] == "http://localhost:11434"
        assert data["agents"]["defaults"]["model"] == "ollama/llama3:latest"


# ---------------------------------------------------------------------------
# workspace/retention.py — prune_session_logs
# ---------------------------------------------------------------------------


class TestPruneSessionLogs:
    """Tests for prune_session_logs()."""

    def test_deletes_oldest_files_when_over_limit(self, tmp_path: Path) -> None:
        """prune_session_logs deletes oldest files when count exceeds max."""
        from clambot.workspace.retention import prune_session_logs

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # Create 5 files with different mtimes
        for i in range(5):
            f = sessions_dir / f"session_{i}.jsonl"
            f.write_text(f"data-{i}")
            # Set mtime to spread them out
            mtime = time.time() - (5 - i) * 100
            os.utime(f, (mtime, mtime))

        deleted = prune_session_logs(sessions_dir, max_files=3)

        assert deleted == 2
        remaining = sorted(sessions_dir.glob("*.jsonl"))
        assert len(remaining) == 3
        # Oldest files (session_0, session_1) should be gone
        remaining_names = [f.name for f in remaining]
        assert "session_0.jsonl" not in remaining_names
        assert "session_1.jsonl" not in remaining_names

    def test_no_deletion_when_under_limit(self, tmp_path: Path) -> None:
        """prune_session_logs does nothing when file count is within limit."""
        from clambot.workspace.retention import prune_session_logs

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        for i in range(3):
            (sessions_dir / f"session_{i}.jsonl").write_text(f"data-{i}")

        deleted = prune_session_logs(sessions_dir, max_files=5)

        assert deleted == 0
        assert len(list(sessions_dir.glob("*.jsonl"))) == 3

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """prune_session_logs handles missing directory gracefully."""
        from clambot.workspace.retention import prune_session_logs

        deleted = prune_session_logs(tmp_path / "nonexistent")
        assert deleted == 0


# ---------------------------------------------------------------------------
# heartbeat/service.py — InMemoryHeartbeatService
# ---------------------------------------------------------------------------


class TestHeartbeatService:
    """Tests for heartbeat skip logic and actionable content detection."""

    def test_headings_only_not_actionable(self) -> None:
        """HEARTBEAT.md with only headings is not actionable."""
        from clambot.heartbeat.service import _is_actionable

        content = "# Heartbeat\n## Section\n### Subsection\n"
        assert _is_actionable(content) is False

    def test_empty_checkboxes_not_actionable(self) -> None:
        """HEARTBEAT.md with only empty checkboxes is not actionable."""
        from clambot.heartbeat.service import _is_actionable

        content = "# Heartbeat\n- [ ] \n- [ ] \n"
        assert _is_actionable(content) is False

    def test_whitespace_only_not_actionable(self) -> None:
        """Empty/whitespace-only content is not actionable."""
        from clambot.heartbeat.service import _is_actionable

        assert _is_actionable("") is False
        assert _is_actionable("   \n\n  \n") is False

    def test_headings_and_empty_checkboxes_not_actionable(self) -> None:
        """Mix of headings and empty checkboxes is not actionable."""
        from clambot.heartbeat.service import _is_actionable

        content = "# Heartbeat\n\n- [ ] \n\n## Tasks\n\n- [ ] \n"
        assert _is_actionable(content) is False

    def test_filled_checkbox_is_actionable(self) -> None:
        """A filled checkbox with text IS actionable."""
        from clambot.heartbeat.service import _is_actionable

        content = "# Heartbeat\n- [ ] Check the weather\n"
        assert _is_actionable(content) is True

    def test_plain_text_is_actionable(self) -> None:
        """Plain text content IS actionable."""
        from clambot.heartbeat.service import _is_actionable

        content = "Check the stock prices and report back."
        assert _is_actionable(content) is True

    @pytest.mark.asyncio
    async def test_skip_when_no_actionable_content(self, tmp_path: Path) -> None:
        """HeartbeatService does NOT trigger executor when HEARTBEAT.md has no actionable
        content."""
        from clambot.heartbeat.service import InMemoryHeartbeatService

        workspace = tmp_path / "workspace"
        (workspace / "memory").mkdir(parents=True)
        (workspace / "memory" / "HEARTBEAT.md").write_text("# Heartbeat\n- [ ] \n")

        config = MagicMock()
        config.interval = 0.05  # 50ms for fast test

        service = InMemoryHeartbeatService(config=config, workspace=workspace)
        executor = AsyncMock()
        service.set_executor(executor)

        await service.start()
        await asyncio.sleep(0.15)  # Wait for a couple of intervals
        await service.stop()

        executor.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_when_actionable_content(self, tmp_path: Path) -> None:
        """HeartbeatService triggers executor when HEARTBEAT.md has actionable content."""
        from clambot.heartbeat.service import InMemoryHeartbeatService

        workspace = tmp_path / "workspace"
        (workspace / "memory").mkdir(parents=True)
        (workspace / "memory" / "HEARTBEAT.md").write_text(
            "# Heartbeat\nCheck the weather for tomorrow\n"
        )

        config = MagicMock()
        config.interval = 0.05

        service = InMemoryHeartbeatService(config=config, workspace=workspace)
        executor = AsyncMock()
        service.set_executor(executor)

        await service.start()
        await asyncio.sleep(0.15)
        await service.stop()

        assert executor.call_count >= 1
        call_args = executor.call_args[0][0]
        assert call_args.channel == "heartbeat"
        assert "weather" in call_args.content

    @pytest.mark.asyncio
    async def test_no_executor_set_no_error(self, tmp_path: Path) -> None:
        """HeartbeatService handles missing executor gracefully."""
        from clambot.heartbeat.service import InMemoryHeartbeatService

        workspace = tmp_path / "workspace"
        (workspace / "memory").mkdir(parents=True)
        (workspace / "memory" / "HEARTBEAT.md").write_text("Do something\n")

        config = MagicMock()
        config.interval = 0.05

        service = InMemoryHeartbeatService(config=config, workspace=workspace)
        # No executor set

        await service.start()
        await asyncio.sleep(0.15)
        await service.stop()
        # Should not raise


class TestNotConfiguredHeartbeatService:
    """Tests for NotConfiguredHeartbeatService."""

    @pytest.mark.asyncio
    async def test_noop_lifecycle(self) -> None:
        """NotConfiguredHeartbeatService lifecycle methods are no-ops."""
        from clambot.heartbeat.service import NotConfiguredHeartbeatService

        service = NotConfiguredHeartbeatService()
        service.set_executor(AsyncMock())
        await service.start()
        await service.stop()
        # No errors raised


# ---------------------------------------------------------------------------
# Phase 1 — Installability verification
# ---------------------------------------------------------------------------


class TestPackageInstallability:
    """Verify the clambot package is properly installed and importable."""

    def test_clambot_package_importable(self) -> None:
        """clambot root package is importable and exposes __version__."""
        import clambot

        assert hasattr(clambot, "__version__")
        assert clambot.__version__ == "0.1.0"

    def test_core_subpackages_importable(self) -> None:
        """All core sub-packages are importable."""
        import importlib

        subpackages = [
            "clambot.agent",
            "clambot.bus",
            "clambot.channels",
            "clambot.cli",
            "clambot.config",
            "clambot.cron",
            "clambot.gateway",
            "clambot.heartbeat",
            "clambot.memory",
            "clambot.providers",
            "clambot.session",
            "clambot.tools",
            "clambot.workspace",
        ]
        for pkg in subpackages:
            mod = importlib.import_module(pkg)
            assert mod is not None, f"Failed to import {pkg}"

    def test_cli_entry_point_registered(self) -> None:
        """The 'clambot' CLI entry point is importable."""
        from clambot.cli import main

        assert callable(main)

    def test_amla_sandbox_dependency_satisfied(self) -> None:
        """amla-sandbox dependency is installed and importable."""
        import amla_sandbox

        assert amla_sandbox is not None

    def test_key_dependencies_importable(self) -> None:
        """All key runtime dependencies are importable."""
        import importlib

        deps = [
            "pydantic",
            "typer",
            "litellm",
            "questionary",
            "yaml",
            "dotenv",
        ]
        for dep in deps:
            mod = importlib.import_module(dep)
            assert mod is not None, f"Failed to import dependency: {dep}"

    def test_config_schema_loadable(self) -> None:
        """ClamBotConfig can be instantiated with defaults."""
        from clambot.config.schema import ClamBotConfig

        cfg = ClamBotConfig()
        assert cfg.agents.defaults.model == ""  # model is intentionally empty by default; must be set explicitly in config
        assert cfg.gateway.port > 0


# ---------------------------------------------------------------------------
# Phase 1 — Onboard config shape compatibility
# ---------------------------------------------------------------------------


class TestOnboardConfigShape:
    """Verify onboard_workspace generates config compatible with config.example.json."""

    @staticmethod
    def _collect_key_paths(d: dict, prefix: str = "") -> set[str]:
        """Recursively collect all key paths from a dict (e.g., 'agents.defaults.model')."""
        paths: set[str] = set()
        for key, val in d.items():
            if key.startswith("_"):
                # Skip comment keys like "_comment"
                continue
            full = f"{prefix}.{key}" if prefix else key
            paths.add(full)
            if isinstance(val, dict):
                paths.update(TestOnboardConfigShape._collect_key_paths(val, full))
        return paths

    def test_onboard_config_has_all_example_top_level_keys(self, tmp_path: Path) -> None:
        """Onboarded config contains all top-level sections from config.example.json."""
        from clambot.workspace.onboard import onboard_workspace

        config_path = tmp_path / "config.json"
        example_path = Path(__file__).resolve().parent.parent / "config.example.json"
        example = json.loads(example_path.read_text())

        # Onboard with no providers detected
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "OPENROUTER_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "GROQ_API_KEY",
                "GEMINI_API_KEY",
            )
        }
        with patch.dict(os.environ, env_clean, clear=True):
            with patch("clambot.workspace.onboard._probe_ollama", return_value=[]):
                onboard_workspace(config_path)

        generated = json.loads(config_path.read_text())

        example_top = {k for k in example if not k.startswith("_")}
        generated_top = {k for k in generated if not k.startswith("_")}

        missing = example_top - generated_top
        assert not missing, f"Onboarded config missing top-level keys: {missing}"

    def test_onboard_config_has_all_example_nested_keys(self, tmp_path: Path) -> None:
        """Onboarded config contains all nested key paths from config.example.json."""
        from clambot.workspace.onboard import onboard_workspace

        config_path = tmp_path / "config.json"
        example_path = Path(__file__).resolve().parent.parent / "config.example.json"
        example = json.loads(example_path.read_text())

        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "OPENROUTER_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "GROQ_API_KEY",
                "GEMINI_API_KEY",
            )
        }
        with patch.dict(os.environ, env_clean, clear=True):
            with patch("clambot.workspace.onboard._probe_ollama", return_value=[]):
                onboard_workspace(config_path)

        generated = json.loads(config_path.read_text())

        example_paths = self._collect_key_paths(example)
        generated_paths = self._collect_key_paths(generated)

        missing = example_paths - generated_paths
        assert not missing, f"Onboarded config missing key paths: {missing}"

    def test_onboard_config_parseable_by_schema(self, tmp_path: Path) -> None:
        """Onboarded config can be parsed back by ClamBotConfig schema."""
        from clambot.config.schema import ClamBotConfig
        from clambot.workspace.onboard import onboard_workspace

        config_path = tmp_path / "config.json"

        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "OPENROUTER_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "GROQ_API_KEY",
                "GEMINI_API_KEY",
            )
        }
        with patch.dict(os.environ, env_clean, clear=True):
            with patch("clambot.workspace.onboard._probe_ollama", return_value=[]):
                onboard_workspace(config_path)

        generated = json.loads(config_path.read_text())
        cfg = ClamBotConfig.model_validate(generated)

        # Verify key defaults match
        assert cfg.agents.defaults.workspace == "~/.clambot/workspace"
        assert cfg.agents.defaults.max_tokens > 0
        assert cfg.gateway.port > 0
        assert cfg.channels.telegram.enabled is False

    def test_example_config_parseable_by_schema(self) -> None:
        """config.example.json itself is valid per the ClamBotConfig schema."""
        from clambot.config.schema import ClamBotConfig

        example_path = Path(__file__).resolve().parent.parent / "config.example.json"
        example = json.loads(example_path.read_text())

        cfg = ClamBotConfig.model_validate(example)
        assert cfg.agents.defaults.model == ""  # model is intentionally empty by default; must be set explicitly in config
        assert cfg.gateway.port > 0

    def test_onboard_default_values_match_example(self, tmp_path: Path) -> None:
        """Key default values from onboard match config.example.json."""
        from clambot.workspace.onboard import onboard_workspace

        config_path = tmp_path / "config.json"
        example_path = Path(__file__).resolve().parent.parent / "config.example.json"
        example = json.loads(example_path.read_text())

        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "OPENROUTER_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
                "GROQ_API_KEY",
                "GEMINI_API_KEY",
            )
        }
        with patch.dict(os.environ, env_clean, clear=True):
            with patch("clambot.workspace.onboard._probe_ollama", return_value=[]):
                onboard_workspace(config_path)

        generated = json.loads(config_path.read_text())

        # Key structural defaults should match
        assert (
            generated["agents"]["defaults"]["workspace"]
            == example["agents"]["defaults"]["workspace"]
        )
        assert (
            generated["agents"]["defaults"]["maxTokens"]
            == example["agents"]["defaults"]["maxTokens"]
        )
        assert (
            generated["agents"]["defaults"]["temperature"]
            == example["agents"]["defaults"]["temperature"]
        )
        assert generated["gateway"]["port"] == example["gateway"]["port"]
        assert (
            generated["channels"]["telegram"]["enabled"]
            == example["channels"]["telegram"]["enabled"]
        )
        assert (
            generated["tools"]["filesystem"]["restrictToWorkspace"]
            == example["tools"]["filesystem"]["restrictToWorkspace"]
        )
        assert generated["cron"]["enabled"] == example["cron"]["enabled"]
