"""Tests for clambot.tools — Phase 5 Tool System."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clambot.config.schema import FilesystemToolConfig, TranscribeToolConfig
from clambot.tools.cron.operations import CronTool
from clambot.tools.echo.echo import EchoTool
from clambot.tools.filesystem.core import FilesystemTool
from clambot.tools.http.core import HttpRequestTool
from clambot.tools.memory.recall import MemoryRecallTool
from clambot.tools.memory.search import MemorySearchHistoryTool
from clambot.tools.registry import BuiltinToolRegistry
from clambot.tools.transcribe.transcribe import TranscribeTool
from clambot.tools.web.fetch import WebFetchTool
from clambot.tools.web.search import WebSearchTool
from clambot.tools.secrets.env import resolve_secret_value
from clambot.tools.secrets.store import SecretRecord, SecretStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace directory structure."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "memory").mkdir()
    return ws


def _make_mock_httpx_client(status_code: int = 200, text: str = "ok") -> MagicMock:
    """Return a MagicMock that behaves like httpx.Client used as a context manager."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = text

    mock_client = MagicMock()
    mock_client.request.return_value = mock_response

    mock_client_cls = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

    return mock_client_cls


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestBuiltinToolRegistry:
    """Tests for tool registry dispatch and lookup."""

    def test_register_and_dispatch(self) -> None:
        """Registered tool can be dispatched by name."""
        registry = BuiltinToolRegistry()
        echo = EchoTool()
        registry.register(echo)
        result = registry.dispatch("echo", {"message": "hello"})
        assert result == "hello"

    def test_dispatch_unknown_raises_value_error(self) -> None:
        """Dispatching unknown tool name raises ValueError."""
        registry = BuiltinToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            registry.dispatch("nonexistent", {})

    def test_get_tool_returns_none_for_missing(self) -> None:
        """get_tool returns None for unregistered name."""
        registry = BuiltinToolRegistry()
        assert registry.get_tool("nope") is None

    def test_get_tool_returns_tool(self) -> None:
        """get_tool returns the registered tool instance."""
        registry = BuiltinToolRegistry()
        echo = EchoTool()
        registry.register(echo)
        assert registry.get_tool("echo") is echo

    def test_get_schemas_returns_openai_format(self) -> None:
        """get_schemas returns list of OpenAI function-call format dicts."""
        registry = BuiltinToolRegistry()
        echo = EchoTool()
        registry.register(echo)
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "echo"

    def test_contains_and_len(self) -> None:
        """Registry supports __contains__ and __len__."""
        registry = BuiltinToolRegistry()
        echo = EchoTool()
        registry.register(echo)
        assert "echo" in registry
        assert len(registry) == 1
        assert "nope" not in registry

    def test_register_replaces_existing(self) -> None:
        """Re-registering a tool name silently replaces the previous entry."""
        registry = BuiltinToolRegistry()
        first = EchoTool()
        second = EchoTool()
        registry.register(first)
        registry.register(second)
        assert len(registry) == 1
        assert registry.get_tool("echo") is second

    def test_tool_names_property(self) -> None:
        """tool_names returns all registered names in insertion order."""
        registry = BuiltinToolRegistry()
        registry.register(EchoTool())
        registry.register(CronTool())
        assert registry.tool_names == ["echo", "cron"]

    def test_get_usage_instructions_includes_tool_metadata(self) -> None:
        """Registry exposes prompt usage instructions from registered tools."""
        registry = BuiltinToolRegistry()
        registry.register(WebSearchTool())
        registry.register(WebFetchTool())
        registry.register(HttpRequestTool())
        registry.register(TranscribeTool())

        usage = registry.get_usage_instructions()

        assert "web_search" in usage
        assert any("query" in item for item in usage["web_search"])
        assert "web_fetch" in usage
        assert any("content" in item for item in usage["web_fetch"])
        assert "http_request" in usage
        assert any("method" in item for item in usage["http_request"])
        assert "transcribe" in usage
        assert any("transcript" in item for item in usage["transcribe"])

    def test_build_tool_registry_usage_instructions_cover_enabled_tools(
        self,
        tmp_path: Path,
    ) -> None:
        """Default registry exposes instructions for all enabled built-in tools."""
        from clambot.tools import build_tool_registry

        ws = _make_workspace(tmp_path)
        ss = SecretStore(tmp_path / "secrets.json")
        registry = build_tool_registry(workspace=ws, secret_store=ss)

        usage = registry.get_usage_instructions()

        expected = {
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
        }
        assert expected.issubset(set(usage))
        assert any("one-shot" in item for item in usage["cron"])
        assert any("MEMORY.md" in item for item in usage["memory_recall"])
        assert any("query" in item for item in usage["memory_search_history"])
        assert any("upload/<name>.pdf" in item for item in usage["pdf_reader"])
        assert any("from_env" in item for item in usage["secrets_add"])


# ---------------------------------------------------------------------------
# Filesystem tool tests
# ---------------------------------------------------------------------------


class TestFilesystemTool:
    """Tests for the fs tool — path resolution, operations, restrictions."""

    def test_workspace_prefix_rejected(self, tmp_path: Path) -> None:
        """Paths starting with /workspace/ are rejected (VFS collision)."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        result = tool.execute({"operation": "read", "path": "/workspace/foo.txt"})
        assert "Permission denied" in result
        assert "/workspace/" in result

    def test_workspace_exact_rejected(self, tmp_path: Path) -> None:
        """Exact /workspace path is also rejected."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        result = tool.execute({"operation": "list", "path": "/workspace"})
        assert "Permission denied" in result

    def test_restrict_to_workspace_blocks_outside(self, tmp_path: Path) -> None:
        """restrict_to_workspace=True blocks paths outside workspace."""
        ws = _make_workspace(tmp_path)
        config = FilesystemToolConfig(restrict_to_workspace=True)
        tool = FilesystemTool(workspace=ws, config=config)
        result = tool.execute({"operation": "read", "path": "/etc/passwd"})
        assert "Permission denied" in result
        assert "outside" in result

    def test_restrict_to_workspace_allows_inside(self, tmp_path: Path) -> None:
        """restrict_to_workspace=True allows paths inside workspace."""
        ws = _make_workspace(tmp_path)
        (ws / "allowed.txt").write_text("inside workspace")
        config = FilesystemToolConfig(restrict_to_workspace=True)
        tool = FilesystemTool(workspace=ws, config=config)
        result = tool.execute({"operation": "read", "path": "allowed.txt"})
        assert "inside workspace" in result

    def test_relative_path_resolves_to_workspace(self, tmp_path: Path) -> None:
        """Relative paths are resolved against the workspace directory."""
        ws = _make_workspace(tmp_path)
        (ws / "hello.txt").write_text("world")
        tool = FilesystemTool(workspace=ws)
        result = tool.execute({"operation": "read", "path": "hello.txt"})
        assert "world" in result

    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        """Write then read produces consistent content."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        tool.execute({"operation": "write", "path": "test.txt", "content": "hello world"})
        result = tool.execute({"operation": "read", "path": "test.txt"})
        assert "hello world" in result

    def test_edit_find_and_replace(self, tmp_path: Path) -> None:
        """Edit operation performs find-and-replace on the first occurrence."""
        ws = _make_workspace(tmp_path)
        (ws / "edit_me.txt").write_text("Hello World, Hello Python")
        tool = FilesystemTool(workspace=ws)
        tool.execute(
            {
                "operation": "edit",
                "path": "edit_me.txt",
                "old_text": "Hello World",
                "new_text": "Hi World",
            }
        )
        content = (ws / "edit_me.txt").read_text()
        assert "Hi World" in content
        assert "Hello Python" in content  # Second occurrence untouched

    def test_edit_replaces_only_first_occurrence(self, tmp_path: Path) -> None:
        """Edit replaces only the first occurrence when old_text appears multiple times."""
        ws = _make_workspace(tmp_path)
        (ws / "multi.txt").write_text("AAA AAA AAA")
        tool = FilesystemTool(workspace=ws)
        tool.execute(
            {
                "operation": "edit",
                "path": "multi.txt",
                "old_text": "AAA",
                "new_text": "BBB",
            }
        )
        content = (ws / "multi.txt").read_text()
        assert content == "BBB AAA AAA"

    def test_edit_not_found_returns_error(self, tmp_path: Path) -> None:
        """Edit with non-matching old_text returns an error."""
        ws = _make_workspace(tmp_path)
        (ws / "no_match.txt").write_text("AAAA")
        tool = FilesystemTool(workspace=ws)
        result = tool.execute(
            {
                "operation": "edit",
                "path": "no_match.txt",
                "old_text": "ZZZZ",
                "new_text": "XXXX",
            }
        )
        assert "not found" in result.lower() or "error" in result.lower()

    def test_list_directory(self, tmp_path: Path) -> None:
        """List operation returns directory contents."""
        ws = _make_workspace(tmp_path)
        (ws / "a.txt").write_text("a")
        (ws / "subdir").mkdir()
        tool = FilesystemTool(workspace=ws)
        result = tool.execute({"operation": "list", "path": "."})
        assert "a.txt" in result
        assert "subdir" in result

    def test_read_nonexistent_returns_error(self, tmp_path: Path) -> None:
        """Reading a non-existent file returns a user-friendly error."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        result = tool.execute({"operation": "read", "path": "does_not_exist.txt"})
        assert "not found" in result.lower() or "error" in result.lower()

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """Write operation creates intermediate directories as needed."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        tool.execute(
            {
                "operation": "write",
                "path": "deep/nested/file.txt",
                "content": "nested content",
            }
        )
        assert (ws / "deep" / "nested" / "file.txt").read_text() == "nested content"

    def test_unknown_operation_returns_error(self, tmp_path: Path) -> None:
        """An unrecognised operation name returns an error string."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        result = tool.execute({"operation": "explode", "path": "."})
        assert "unknown operation" in result.lower() or "error" in result.lower()

    def test_approval_options_exclude_workspace(self, tmp_path: Path) -> None:
        """Approval options do not include workspace-level or root scope."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        options = tool.get_approval_options({"operation": "read", "path": "foo/bar.txt"})
        assert not any("workspace" in o.id for o in options)
        assert not any(o.scope == "dir:/" for o in options)

    def test_approval_options_include_exact_path(self, tmp_path: Path) -> None:
        """Approval options include the exact file path as the narrowest scope."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        options = tool.get_approval_options({"operation": "write", "path": "src/main.py"})
        assert any("src/main.py" in o.scope for o in options)

    def test_tool_name_is_fs(self, tmp_path: Path) -> None:
        """FilesystemTool has the canonical name 'fs'."""
        ws = _make_workspace(tmp_path)
        assert FilesystemTool(workspace=ws).name == "fs"


# ---------------------------------------------------------------------------
# Filesystem tool — normalize_args_for_approval tests (Phase 2)
# ---------------------------------------------------------------------------


class TestFilesystemNormalizeArgs:
    """Tests for FilesystemTool.normalize_args_for_approval."""

    def test_normalize_args_resolves_relative_path(self, tmp_path: Path) -> None:
        """Relative 'data.txt' resolves to workspace/data.txt absolute path."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        result = tool.normalize_args_for_approval({"operation": "read", "path": "data.txt"})
        expected = str((ws / "data.txt").resolve())
        assert result["path"] == expected
        assert result["operation"] == "read"

    def test_normalize_args_absolute_path_unchanged(self, tmp_path: Path) -> None:
        """/etc/hosts stays /etc/hosts (absolute paths pass through)."""
        ws = _make_workspace(tmp_path)
        config = FilesystemToolConfig(restrict_to_workspace=False)
        tool = FilesystemTool(workspace=ws, config=config)
        result = tool.normalize_args_for_approval({"operation": "read", "path": "/etc/hosts"})
        assert result["path"] == str(Path("/etc/hosts").resolve())

    def test_normalize_args_workspace_prefix_passthrough(self, tmp_path: Path) -> None:
        """/workspace/foo returns args unchanged (doesn't crash)."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        original = {"operation": "read", "path": "/workspace/foo"}
        result = tool.normalize_args_for_approval(original)
        # Should return unchanged since _resolve_path raises PermissionError
        assert result is original

    def test_normalize_args_tilde_expanded(self, tmp_path: Path) -> None:
        """~/file.txt expands to /home/.../file.txt."""
        ws = _make_workspace(tmp_path)
        config = FilesystemToolConfig(restrict_to_workspace=False)
        tool = FilesystemTool(workspace=ws, config=config)
        result = tool.normalize_args_for_approval({"operation": "read", "path": "~/file.txt"})
        assert "~" not in result["path"]
        assert result["path"].startswith("/")

    def test_builtin_tool_default_normalize_is_identity(self) -> None:
        """Base class normalize_args_for_approval returns args unchanged."""
        tool = EchoTool()
        args = {"message": "hello"}
        assert tool.normalize_args_for_approval(args) is args


# ---------------------------------------------------------------------------
# Filesystem tool — resolved paths in approval options (Phase 3)
# ---------------------------------------------------------------------------


class TestFilesystemApprovalOptions:
    """Tests for resolved paths in filesystem approval scope strings."""

    def test_approval_options_use_resolved_paths(self, tmp_path: Path) -> None:
        """Scope strings contain absolute paths, not raw relative."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        # Simulate normalized args (absolute path) as Phase 2 would produce
        abs_path = str((ws / "src" / "main.py").resolve())
        options = tool.get_approval_options({"operation": "read", "path": abs_path})
        # First option should be exact file scope with absolute path
        file_opt = options[0]
        assert file_opt.scope.startswith("file:")
        assert file_opt.scope == f"file:{abs_path}"
        # All scopes should use absolute paths (no relative "src/main.py")
        for opt in options:
            scope_path = opt.scope.split(":", 1)[1] if ":" in opt.scope else ""
            if scope_path:
                assert scope_path.startswith("/"), f"Scope path should be absolute: {opt.scope}"

    def test_approval_options_two_tiers(self, tmp_path: Path) -> None:
        """Returns [file, dir] in order — no workspace option."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(workspace=ws)
        abs_path = str((ws / "data" / "file.txt").resolve())
        options = tool.get_approval_options({"operation": "read", "path": abs_path})
        assert len(options) == 2
        # Tier 1: exact file
        assert options[0].scope.startswith("file:")
        # Tier 2: parent directory
        assert options[1].scope.startswith("dir:")
        assert options[1].scope == f"dir:{str(Path(abs_path).parent)}"

    def test_approval_options_root_path_only_file(self, tmp_path: Path) -> None:
        """Path at '/' has only the exact file option (parent '/' is skipped)."""
        ws = _make_workspace(tmp_path)
        tool = FilesystemTool(
            workspace=ws, config=FilesystemToolConfig(restrict_to_workspace=False)
        )
        options = tool.get_approval_options({"operation": "list", "path": "/"})
        assert len(options) == 1
        assert options[0].scope == "file:/"


# ---------------------------------------------------------------------------
# HTTP request tool tests
# ---------------------------------------------------------------------------


class TestHttpRequestTool:
    """Tests for http_request tool — auth injection and redaction."""

    def test_bearer_secret_injection(self) -> None:
        """bearer_secret auth injects Authorization header from secret store."""
        store = MagicMock()
        store.get.return_value = "my-secret-token"
        tool = HttpRequestTool(secret_store=store)

        mock_client_cls = _make_mock_httpx_client(status_code=200, text="ok")

        with patch("clambot.tools.http.core.httpx.Client", mock_client_cls):
            result = tool.execute(
                {
                    "method": "GET",
                    "url": "https://api.example.com/data",
                    "auth": {"type": "bearer_secret", "name": "my_key"},
                }
            )

        # Secret was looked up by name
        store.get.assert_called_once_with("my_key")

        # Request was made with the injected Authorization header
        mock_client = mock_client_cls.return_value.__enter__.return_value
        call_kwargs = mock_client.request.call_args.kwargs
        assert call_kwargs["headers"].get("Authorization") == "Bearer my-secret-token"

        # Result reflects the mocked response
        assert result["ok"] is True
        assert result["status_code"] == 200

    def test_bearer_secret_value_not_in_result(self) -> None:
        """The raw secret value never appears in the returned result dict."""
        store = MagicMock()
        store.get.return_value = "super-secret-value-xyz"
        tool = HttpRequestTool(secret_store=store)

        mock_client_cls = _make_mock_httpx_client(status_code=200, text="response body")

        with patch("clambot.tools.http.core.httpx.Client", mock_client_cls):
            result = tool.execute(
                {
                    "method": "GET",
                    "url": "https://api.example.com/",
                    "auth": {"type": "bearer_secret", "name": "my_key"},
                }
            )

        result_str = json.dumps(result)
        assert "super-secret-value-xyz" not in result_str

    def test_authorization_header_conflict(self) -> None:
        """Setting both Authorization header and auth field returns error."""
        tool = HttpRequestTool()
        result = tool.execute(
            {
                "method": "GET",
                "url": "https://example.com",
                "headers": {"Authorization": "Bearer old-token"},
                "auth": {"type": "bearer_secret", "name": "my_key"},
            }
        )
        assert result["ok"] is False
        assert "authorization_header_conflicts_with_auth" in result["error"]

    def test_authorization_header_conflict_case_insensitive(self) -> None:
        """Authorization header conflict check is case-insensitive."""
        tool = HttpRequestTool()
        result = tool.execute(
            {
                "method": "GET",
                "url": "https://example.com",
                "headers": {"authorization": "Bearer old-token"},
                "auth": {"type": "bearer_secret", "name": "my_key"},
            }
        )
        assert result["ok"] is False
        assert "authorization_header_conflicts_with_auth" in result["error"]

    def test_missing_secret_returns_error(self) -> None:
        """bearer_secret with missing secret returns error, not exception."""
        store = MagicMock()
        store.get.return_value = None
        tool = HttpRequestTool(secret_store=store)
        result = tool.execute(
            {
                "method": "GET",
                "url": "https://example.com",
                "auth": {"type": "bearer_secret", "name": "missing"},
            }
        )
        assert result["ok"] is False
        assert "not found" in result["error"].lower()

    def test_no_secret_store_returns_error(self) -> None:
        """bearer_secret without secret store returns error."""
        tool = HttpRequestTool(secret_store=None)
        result = tool.execute(
            {
                "method": "GET",
                "url": "https://example.com",
                "auth": {"type": "bearer_secret", "name": "key"},
            }
        )
        assert result["ok"] is False
        assert "not configured" in result["error"].lower() or "store" in result["error"].lower()

    def test_successful_get_request(self) -> None:
        """A plain GET request returns ok=True and the response body."""
        tool = HttpRequestTool()
        mock_client_cls = _make_mock_httpx_client(status_code=200, text="hello")

        with patch("clambot.tools.http.core.httpx.Client", mock_client_cls):
            result = tool.execute({"method": "GET", "url": "https://example.com"})

        assert result["ok"] is True
        assert result["status_code"] == 200
        assert result["content"] == "hello"

    def test_4xx_response_sets_ok_false(self) -> None:
        """A 4xx response sets ok=False in the result."""
        tool = HttpRequestTool()
        mock_client_cls = _make_mock_httpx_client(status_code=404, text="not found")

        with patch("clambot.tools.http.core.httpx.Client", mock_client_cls):
            result = tool.execute({"method": "GET", "url": "https://example.com/missing"})

        assert result["ok"] is False
        assert result["status_code"] == 404

    def test_approval_options_include_host(self) -> None:
        """Approval options include host-level scope."""
        tool = HttpRequestTool()
        options = tool.get_approval_options(
            {
                "method": "GET",
                "url": "https://api.example.com/data",
            }
        )
        assert any("example.com" in o.scope for o in options)

    def test_tool_name_is_http_request(self) -> None:
        """HttpRequestTool has the canonical name 'http_request'."""
        assert HttpRequestTool().name == "http_request"


# ---------------------------------------------------------------------------
# SecretStore tests
# ---------------------------------------------------------------------------


class TestSecretStore:
    """Tests for SecretStore — persistence, atomic writes, permissions."""

    def test_save_and_get(self, tmp_path: Path) -> None:
        """Saved secret can be retrieved by name."""
        store = SecretStore(tmp_path / "secrets.json")
        store.save("MY_KEY", "my-value", description="Test key")
        assert store.get("MY_KEY") == "my-value"

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Getting a non-existent secret returns None."""
        store = SecretStore(tmp_path / "secrets.json")
        assert store.get("NOPE") is None

    def test_list_returns_records(self, tmp_path: Path) -> None:
        """list() returns SecretRecord objects with all fields."""
        store = SecretStore(tmp_path / "secrets.json")
        store.save("A", "val-a", description="Key A")
        store.save("B", "val-b")
        records = store.list()
        assert "A" in records
        assert "B" in records
        assert records["A"].value == "val-a"
        assert records["A"].description == "Key A"
        assert records["A"].created_at != ""

    def test_list_returns_secret_record_instances(self, tmp_path: Path) -> None:
        """list() values are SecretRecord dataclass instances."""
        store = SecretStore(tmp_path / "secrets.json")
        store.save("X", "x-value")
        records = store.list()
        assert isinstance(records["X"], SecretRecord)

    def test_update_preserves_created_at(self, tmp_path: Path) -> None:
        """Updating a secret preserves the original created_at timestamp."""
        store = SecretStore(tmp_path / "secrets.json")
        store.save("KEY", "v1")
        created = store.list()["KEY"].created_at
        store.save("KEY", "v2")
        assert store.get("KEY") == "v2"
        assert store.list()["KEY"].created_at == created

    def test_update_refreshes_updated_at(self, tmp_path: Path) -> None:
        """Updating a secret refreshes the updated_at timestamp."""
        store = SecretStore(tmp_path / "secrets.json")
        store.save("KEY", "v1")
        first_updated = store.list()["KEY"].updated_at
        store.save("KEY", "v2")
        second_updated = store.list()["KEY"].updated_at
        # updated_at must change (or at minimum not regress)
        assert second_updated >= first_updated

    def test_atomic_write_survives_crash(self, tmp_path: Path) -> None:
        """Store file is not corrupted if write is interrupted mid-rename."""
        store_path = tmp_path / "secrets.json"
        store = SecretStore(store_path)
        store.save("ORIGINAL", "value")

        # Simulate crash by patching os.rename to raise before the swap
        with patch("clambot.tools.secrets.store.os.rename", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                store.save("NEW", "should-fail")

        # Original data must still be intact — reload from disk
        store2 = SecretStore(store_path)
        assert store2.get("ORIGINAL") == "value"
        assert store2.get("NEW") is None

    def test_file_permissions(self, tmp_path: Path) -> None:
        """Store file is created with restrictive 0600 permissions."""
        store_path = tmp_path / "secrets.json"
        SecretStore(store_path)
        if os.name != "nt":  # Skip permission check on Windows
            mode = oct(store_path.stat().st_mode & 0o777)
            assert mode == "0o600"

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        """Secrets saved by one instance are visible to a fresh instance."""
        path = tmp_path / "secrets.json"
        store1 = SecretStore(path)
        store1.save("PERSIST_KEY", "persist-value")

        store2 = SecretStore(path)
        assert store2.get("PERSIST_KEY") == "persist-value"

    def test_multiple_secrets_independent(self, tmp_path: Path) -> None:
        """Multiple secrets coexist without overwriting each other."""
        store = SecretStore(tmp_path / "secrets.json")
        store.save("KEY_A", "value-a")
        store.save("KEY_B", "value-b")
        store.save("KEY_C", "value-c")
        assert store.get("KEY_A") == "value-a"
        assert store.get("KEY_B") == "value-b"
        assert store.get("KEY_C") == "value-c"


# ---------------------------------------------------------------------------
# Secret resolution tests
# ---------------------------------------------------------------------------


class TestResolveSecretValue:
    """Tests for resolve_secret_value — 6-step resolution chain."""

    def test_explicit_value(self, tmp_path: Path) -> None:
        """Step 1: Explicit 'value' in args is returned directly."""
        store = SecretStore(tmp_path / "secrets.json")
        result = resolve_secret_value("key", {"value": "explicit"}, store)
        assert result == "explicit"

    def test_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Step 2: 'from_env' reads from environment variable."""
        store = SecretStore(tmp_path / "secrets.json")
        monkeypatch.setenv("MY_ENV_SECRET", "env-value")
        result = resolve_secret_value("key", {"from_env": "MY_ENV_SECRET"}, store)
        assert result == "env-value"

    def test_from_secret_store(self, tmp_path: Path) -> None:
        """Step 3: Falls back to SecretStore lookup by name."""
        store = SecretStore(tmp_path / "secrets.json")
        store.save("my_key", "stored-value")
        result = resolve_secret_value("my_key", {}, store)
        assert result == "stored-value"

    def test_from_os_environ(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Step 5: Falls back to os.environ with the exact secret name."""
        store = SecretStore(tmp_path / "secrets.json")
        monkeypatch.setenv("DIRECT_KEY", "direct-value")
        result = resolve_secret_value("DIRECT_KEY", {}, store)
        assert result == "direct-value"

    def test_raises_input_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Step 6: Raises RuntimeError when no resolution path succeeds."""
        store = SecretStore(tmp_path / "secrets.json")
        # Ensure the key is absent from the environment
        monkeypatch.delenv("TOTALLY_MISSING_XYZ_123", raising=False)
        with pytest.raises(RuntimeError, match="input_unavailable"):
            resolve_secret_value("TOTALLY_MISSING_XYZ_123", {}, store)

    def test_resolution_priority_explicit_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit value wins over all other resolution methods."""
        store = SecretStore(tmp_path / "secrets.json")
        store.save("key", "store-value")
        monkeypatch.setenv("key", "env-value")
        result = resolve_secret_value("key", {"value": "explicit"}, store)
        assert result == "explicit"

    def test_resolution_priority_from_env_over_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env wins over SecretStore when both are available."""
        store = SecretStore(tmp_path / "secrets.json")
        store.save("MY_SECRET", "store-value")
        monkeypatch.setenv("MY_ENV_VAR", "env-wins")
        result = resolve_secret_value("MY_SECRET", {"from_env": "MY_ENV_VAR"}, store)
        assert result == "env-wins"

    def test_from_env_missing_var_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env with a missing env var falls through to SecretStore."""
        store = SecretStore(tmp_path / "secrets.json")
        store.save("fallback_key", "store-fallback")
        monkeypatch.delenv("MISSING_ENV_VAR", raising=False)
        result = resolve_secret_value("fallback_key", {"from_env": "MISSING_ENV_VAR"}, store)
        assert result == "store-fallback"


# ---------------------------------------------------------------------------
# Memory tool tests
# ---------------------------------------------------------------------------


class TestMemoryTools:
    """Tests for memory_recall and memory_search_history tools."""

    def test_memory_recall_reads_file(self, tmp_path: Path) -> None:
        """memory_recall returns MEMORY.md content."""
        ws = _make_workspace(tmp_path)
        (ws / "memory" / "MEMORY.md").write_text("# Memory\n\nI like cats.")
        tool = MemoryRecallTool(workspace=ws)
        result = tool.execute({})
        assert "I like cats" in result

    def test_memory_recall_missing_file(self, tmp_path: Path) -> None:
        """memory_recall returns empty string when MEMORY.md doesn't exist."""
        ws = _make_workspace(tmp_path)
        tool = MemoryRecallTool(workspace=ws)
        result = tool.execute({})
        assert result == ""

    def test_memory_recall_full_content(self, tmp_path: Path) -> None:
        """memory_recall returns the complete file content unchanged."""
        ws = _make_workspace(tmp_path)
        content = "# Memory\n\nFact 1\nFact 2\nFact 3"
        (ws / "memory" / "MEMORY.md").write_text(content)
        tool = MemoryRecallTool(workspace=ws)
        assert tool.execute({}) == content

    def test_memory_recall_no_approval_needed(self, tmp_path: Path) -> None:
        """memory_recall is read-only and returns no approval options."""
        ws = _make_workspace(tmp_path)
        tool = MemoryRecallTool(workspace=ws)
        assert tool.get_approval_options({}) == []

    def test_memory_recall_tool_name(self, tmp_path: Path) -> None:
        """MemoryRecallTool has the canonical name 'memory_recall'."""
        ws = _make_workspace(tmp_path)
        assert MemoryRecallTool(workspace=ws).name == "memory_recall"

    def test_memory_search_finds_matches(self, tmp_path: Path) -> None:
        """memory_search_history finds entries containing the query string."""
        ws = _make_workspace(tmp_path)
        (ws / "memory" / "HISTORY.md").write_text(
            "User asked about weather\n---\nUser discussed Python\n---\nUser asked about cats"
        )
        tool = MemorySearchHistoryTool(workspace=ws)
        result = tool.execute({"query": "asked"})
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("asked" in entry for entry in result)

    def test_memory_search_case_insensitive(self, tmp_path: Path) -> None:
        """memory_search_history performs case-insensitive matching."""
        ws = _make_workspace(tmp_path)
        (ws / "memory" / "HISTORY.md").write_text(
            "User talked about PYTHON\n---\nUser discussed java"
        )
        tool = MemorySearchHistoryTool(workspace=ws)
        result = tool.execute({"query": "python"})
        assert isinstance(result, list)
        assert len(result) == 1

    def test_memory_search_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """memory_search_history returns empty list when HISTORY.md doesn't exist."""
        ws = _make_workspace(tmp_path)
        tool = MemorySearchHistoryTool(workspace=ws)
        result = tool.execute({"query": "anything"})
        assert result == []

    def test_memory_search_no_matches_returns_empty(self, tmp_path: Path) -> None:
        """memory_search_history returns empty list when no entries match."""
        ws = _make_workspace(tmp_path)
        (ws / "memory" / "HISTORY.md").write_text("Entry about dogs\n---\nEntry about cats")
        tool = MemorySearchHistoryTool(workspace=ws)
        result = tool.execute({"query": "elephants"})
        assert result == []

    def test_memory_search_respects_limit(self, tmp_path: Path) -> None:
        """memory_search_history respects the limit parameter."""
        ws = _make_workspace(tmp_path)
        entries = "\n---\n".join(f"Entry {i} about topic" for i in range(10))
        (ws / "memory" / "HISTORY.md").write_text(entries)
        tool = MemorySearchHistoryTool(workspace=ws)
        result = tool.execute({"query": "topic", "limit": 3})
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_memory_search_tool_name(self, tmp_path: Path) -> None:
        """MemorySearchHistoryTool has the canonical name 'memory_search_history'."""
        ws = _make_workspace(tmp_path)
        assert MemorySearchHistoryTool(workspace=ws).name == "memory_search_history"


# ---------------------------------------------------------------------------
# Echo tool tests
# ---------------------------------------------------------------------------


class TestEchoTool:
    """Tests for the echo debug tool."""

    def test_echo_returns_message(self) -> None:
        """Echo tool returns the input message verbatim."""
        tool = EchoTool()
        assert tool.execute({"message": "hello"}) == "hello"

    def test_echo_returns_empty_string_for_missing_key(self) -> None:
        """Echo tool returns empty string when 'message' key is absent."""
        tool = EchoTool()
        assert tool.execute({}) == ""

    def test_echo_name(self) -> None:
        """Echo tool has name 'echo'."""
        assert EchoTool().name == "echo"

    def test_echo_schema_requires_message(self) -> None:
        """Echo tool schema requires 'message' field."""
        schema = EchoTool().schema
        assert "message" in schema["properties"]
        assert "message" in schema.get("required", [])

    def test_echo_no_approval_options(self) -> None:
        """Echo tool returns no approval options."""
        assert EchoTool().get_approval_options({"message": "hi"}) == []

    def test_echo_to_schema_openai_format(self) -> None:
        """Echo tool's to_schema() returns valid OpenAI function-call format."""
        schema = EchoTool().to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert "parameters" in schema["function"]

    def test_echo_preserves_whitespace(self) -> None:
        """Echo tool preserves leading/trailing whitespace in the message."""
        tool = EchoTool()
        msg = "  spaced out  "
        assert tool.execute({"message": msg}) == msg


# ---------------------------------------------------------------------------
# Cron tool tests (stub)
# ---------------------------------------------------------------------------


class TestCronTool:
    """Tests for the cron tool stub."""

    def test_cron_name(self) -> None:
        """Cron tool has name 'cron'."""
        assert CronTool().name == "cron"

    def test_cron_list_returns_stub(self) -> None:
        """Cron list returns a stub response when no service is configured."""
        tool = CronTool()
        result = tool.execute({"action": "list"})
        # Stub returns a dict with a 'jobs' key
        assert isinstance(result, dict)
        assert "jobs" in result

    def test_cron_add_returns_stub(self) -> None:
        """Cron add returns a stub not-configured response."""
        tool = CronTool()
        result = tool.execute({"action": "add", "message": "test"})
        assert isinstance(result, dict)
        assert result.get("ok") is False

    def test_cron_remove_returns_stub(self) -> None:
        """Cron remove returns a stub not-configured response."""
        tool = CronTool()
        result = tool.execute({"action": "remove", "job_id": "abc"})
        assert isinstance(result, dict)
        assert result.get("ok") is False

    def test_cron_unknown_action_returns_error(self) -> None:
        """Cron tool returns an error dict for an unknown action."""
        tool = CronTool()
        result = tool.execute({"action": "explode"})
        assert isinstance(result, dict)
        assert result.get("ok") is False

    def test_cron_approval_for_add(self) -> None:
        """Cron add operation has approval options."""
        tool = CronTool()
        options = tool.get_approval_options({"action": "add"})
        assert len(options) > 0

    def test_cron_approval_for_remove(self) -> None:
        """Cron remove operation has approval options."""
        tool = CronTool()
        options = tool.get_approval_options({"action": "remove", "job_id": "x"})
        assert len(options) > 0

    def test_cron_list_no_approval(self) -> None:
        """Cron list operation has no approval options (read-only)."""
        tool = CronTool()
        options = tool.get_approval_options({"action": "list"})
        assert len(options) == 0

    def test_cron_sync_hook_called_when_set(self) -> None:
        """When a sync_hook is injected, execute delegates to it."""
        tool = CronTool()
        hook = MagicMock(return_value={"ok": True, "jobs": ["job1"]})
        tool.set_sync_hook(hook)
        result = tool.execute({"action": "list"})
        hook.assert_called_once_with({"action": "list"})
        assert result == {"ok": True, "jobs": ["job1"]}

    def test_cron_approval_options_include_wildcard(self) -> None:
        """Cron approval options include a wildcard scope for any cron operation."""
        tool = CronTool()
        options = tool.get_approval_options({"action": "add"})
        assert any(o.scope == "cron:*" for o in options)


# ---------------------------------------------------------------------------
# BUILTIN_TOOLS and build_tool_registry tests
# ---------------------------------------------------------------------------


class TestBuiltinToolsIntegration:
    """Tests for BUILTIN_TOOLS tuple and build_tool_registry factory."""

    def test_builtin_tools_tuple(self) -> None:
        """BUILTIN_TOOLS contains expected tool names."""
        from clambot.tools import BUILTIN_TOOLS

        assert "fs" in BUILTIN_TOOLS
        assert "http_request" in BUILTIN_TOOLS
        assert "web_search" in BUILTIN_TOOLS
        assert "web_fetch" in BUILTIN_TOOLS
        assert "cron" in BUILTIN_TOOLS
        assert "secrets_add" in BUILTIN_TOOLS
        assert "memory_recall" in BUILTIN_TOOLS
        assert "memory_search_history" in BUILTIN_TOOLS
        # Echo is excluded from the default surface
        assert "echo" not in BUILTIN_TOOLS

    def test_build_tool_registry_registers_default_tools(self, tmp_path: Path) -> None:
        """build_tool_registry creates a registry with all default tools."""
        from clambot.tools import build_tool_registry

        ws = _make_workspace(tmp_path)
        ss = SecretStore(tmp_path / "secrets.json")
        registry = build_tool_registry(workspace=ws, secret_store=ss)

        assert "fs" in registry
        assert "http_request" in registry
        assert "web_search" in registry
        assert "web_fetch" in registry
        assert "cron" in registry
        assert "memory_recall" in registry
        assert "memory_search_history" in registry

    def test_build_tool_registry_excludes_echo(self, tmp_path: Path) -> None:
        """build_tool_registry does not register echo by default."""
        from clambot.tools import build_tool_registry

        ws = _make_workspace(tmp_path)
        registry = build_tool_registry(workspace=ws)
        assert "echo" not in registry

    def test_build_tool_registry_echo_via_available_tools(self, tmp_path: Path) -> None:
        """build_tool_registry registers echo when explicitly listed in available_tools."""
        from clambot.tools import build_tool_registry

        ws = _make_workspace(tmp_path)
        registry = build_tool_registry(workspace=ws, available_tools=["echo"])
        assert "echo" in registry

    def test_build_tool_registry_secrets_add_requires_store(self, tmp_path: Path) -> None:
        """secrets_add is only registered when a secret_store is provided."""
        from clambot.tools import build_tool_registry

        ws = _make_workspace(tmp_path)
        # Without a secret store, secrets_add should not be registered
        registry_no_store = build_tool_registry(workspace=ws, secret_store=None)
        assert "secrets_add" not in registry_no_store

        # With a secret store, it should be registered
        ss = SecretStore(tmp_path / "secrets.json")
        registry_with_store = build_tool_registry(workspace=ws, secret_store=ss)
        assert "secrets_add" in registry_with_store

    def test_build_tool_registry_dispatches_fs(self, tmp_path: Path) -> None:
        """Registry built by build_tool_registry can dispatch the fs tool."""
        from clambot.tools import build_tool_registry

        ws = _make_workspace(tmp_path)
        (ws / "probe.txt").write_text("probe content")
        registry = build_tool_registry(workspace=ws)
        result = registry.dispatch("fs", {"operation": "read", "path": "probe.txt"})
        assert "probe content" in result

    def test_transcribe_in_builtin_tools(self) -> None:
        """'transcribe' is in BUILTIN_TOOLS."""
        from clambot.tools import BUILTIN_TOOLS

        assert "transcribe" in BUILTIN_TOOLS

    def test_build_tool_registry_includes_transcribe(self, tmp_path: Path) -> None:
        """Default registry contains 'transcribe'."""
        from clambot.tools import build_tool_registry

        ws = _make_workspace(tmp_path)
        registry = build_tool_registry(workspace=ws)
        assert "transcribe" in registry

    def test_transcribe_config_defaults(self) -> None:
        """TranscribeToolConfig() has expected defaults."""
        cfg = TranscribeToolConfig()
        assert cfg.max_duration_seconds == 7200
        assert cfg.chunk_duration_seconds == 600
        assert cfg.audio_format == "mp3"
        assert cfg.whisper_model == "whisper-large-v3"
        assert cfg.whisper_api_url == "https://api.groq.com/openai/v1/audio/transcriptions"
        assert cfg.whisper_api_style == "openai"
        assert cfg.whisper_request_timeout_seconds == 600.0
