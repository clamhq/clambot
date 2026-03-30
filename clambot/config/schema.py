"""ClamBot configuration schema.

All models use camelCase JSON aliases (for config.json serialization) but
expose snake_case Python attributes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class Base(BaseModel):
    """Shared base for all ClamBot config models."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ---------------------------------------------------------------------------
# Agent-level models
# ---------------------------------------------------------------------------


class AgentDefaults(Base):
    workspace: str = "~/.clambot/workspace"
    model: str = ""
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 20
    max_self_fix_attempts: int = 3

    # Strict allowlist — when set, ONLY these tools are registered.
    # New built-in tools are NOT auto-added.  Use this to lock down
    # the tool surface to a fixed set.  Omit to use the default
    # (all BUILTIN_TOOLS minus disabled_tools).
    available_tools: list[str] | None = Field(default=None)

    # Denylist — all BUILTIN_TOOLS are enabled by default; only tools
    # listed here are excluded.  New tools added to the codebase are
    # automatically available without config changes.  Ignored when
    # available_tools is set.
    disabled_tools: list[str] = Field(default_factory=list)


class CompactionConfig(Base):
    enabled: bool = True
    target_ratio: float = 0.5
    reserve_tokens: int = 2000
    summary_max_tokens: int = 500
    keep_recent_turns: int = 4
    max_auto_compactions_per_turn: int = 1


class MemoryPromptBudgetConfig(Base):
    max_tokens: int = 4000
    min_tokens: int = 0
    reserve_tokens: int = 500
    max_context_ratio: float = 0.3
    durable_facts_ratio: float = 0.5


class SelectorConfig(Base):
    provider: str = ""
    model: str = ""
    retries: int = 1
    max_tokens: int = 1024
    temperature: float = 0.0


class LinkContextConfig(Base):
    enabled: bool = True
    max_links: int = 3
    max_chars_per_link: int = 8000
    explicit_links_only: bool = False
    heuristic_prefetch_enabled: bool = True
    intent_url_inference_enabled: bool = False


class ApprovalsConfig(Base):
    enabled: bool = True
    interactive: bool = True
    allow_always: bool = True
    always_grants: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Channel models
# ---------------------------------------------------------------------------


class TelegramConfig(Base):
    enabled: bool = False
    token: str = ""
    allow_from: list[str] = Field(default_factory=list)
    proxy: str | None = None
    reply_to_message: bool = False


# ---------------------------------------------------------------------------
# Infrastructure / feature-flag models
# ---------------------------------------------------------------------------


class CleanupConfig(Base):
    """Workspace cleanup settings, run on each heartbeat tick."""

    stale_clam_days: int = 30
    orphan_build_hours: int = 1
    upload_retention_days: int = 30
    cron_log_max_lines: int = 5000
    prune_disabled_cron: bool = True
    session_max_files: int = 100


class HeartbeatConfig(Base):
    enabled: bool = False
    interval: int = 1800
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)


class CronConfig(Base):
    enabled: bool = True


class FilesystemToolConfig(Base):
    restrict_to_workspace: bool = False
    max_read_bytes: int = 1_048_576  # 1 MB
    max_write_bytes: int = 1_048_576  # 1 MB


class TranscribeToolConfig(Base):
    """Configuration for the transcribe tool.

    Controls audio download limits, chunking behaviour, Whisper model
    selection, and the Whisper API endpoint for the built-in ``transcribe``
    tool.

    ``whisper_api_url`` defaults to the Groq Whisper endpoint but can be
    overridden to point at any OpenAI-compatible transcription API.
    """

    max_duration_seconds: int = 7200
    chunk_duration_seconds: int = 600
    audio_format: str = "mp3"
    whisper_model: str = "whisper-large-v3"
    whisper_api_url: str = "https://api.groq.com/openai/v1/audio/transcriptions"


# ---------------------------------------------------------------------------
# Provider / model models
# ---------------------------------------------------------------------------


class ProviderConfig(Base):
    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None


class LLMModelConfig(Base):
    """Per-model overrides (named LLMModelConfig to avoid clash with Pydantic's ModelConfig)."""

    max_context_size: int = 100_000


# ---------------------------------------------------------------------------
# Aggregate / container models
# ---------------------------------------------------------------------------


class ProvidersConfig(Base):
    custom: ProviderConfig = Field(default_factory=ProviderConfig)
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    ollama: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    openai_codex: ProviderConfig = Field(default_factory=ProviderConfig)


class AgentsConfig(Base):
    defaults: AgentDefaults = Field(default_factory=AgentDefaults)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    memory_prompt_budget: MemoryPromptBudgetConfig = Field(default_factory=MemoryPromptBudgetConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    link_context: LinkContextConfig = Field(default_factory=LinkContextConfig)
    approvals: ApprovalsConfig = Field(default_factory=ApprovalsConfig)
    models: dict[str, LLMModelConfig] = Field(default_factory=dict)


class ChannelsConfig(Base):
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


class ToolsConfig(Base):
    filesystem: FilesystemToolConfig = Field(default_factory=FilesystemToolConfig)
    transcribe: TranscribeToolConfig = Field(default_factory=TranscribeToolConfig)


class GatewayConfig(Base):
    host: str = "127.0.0.1"
    port: int = 18790


class SecurityConfig(Base):
    """Security-related settings.

    ``ssl_fallback_insecure`` controls whether HTTP tools and providers are
    allowed to retry with ``verify=False`` when an SSL certificate error
    occurs.  Defaults to ``False`` (strict — SSL errors are **not**
    silently bypassed).  Set to ``True`` only in sandboxed or proxy
    environments where CA certificates are unavailable.
    """

    ssl_fallback_insecure: bool = False


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------


class ClamBotConfig(Base):
    """Root configuration model for ClamBot."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)
    cron: CronConfig = Field(default_factory=CronConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
