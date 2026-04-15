"""CLI commands — Typer application for ClamBot.

All commands are registered on the top-level ``app`` Typer instance.
Entry point: ``clambot`` (configured in pyproject.toml).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer

app = typer.Typer(
    name="clambot",
    help="ClamBot — AI agent with tool execution in WASM sandbox.",
    no_args_is_help=True,
)

# Sub-commands
cron_app = typer.Typer(name="cron", help="Cron job management.")
channels_app = typer.Typer(name="channels", help="Channel management.")
provider_app = typer.Typer(name="provider", help="Provider management.")

app.add_typer(cron_app, name="cron")
app.add_typer(channels_app, name="channels")
app.add_typer(provider_app, name="provider")


def _setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level."""
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: str | None = None):  # type: ignore[no-untyped-def]
    """Load config from path, handling import lazily."""
    from clambot.config.loader import load_config

    return load_config(config_path)


# ---------------------------------------------------------------------------
# clambot agent
# ---------------------------------------------------------------------------


@app.command()
def agent(
    message: str | None = typer.Option(None, "-m", "--message", help="Single message to send."),
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    logs: bool = typer.Option(False, "--logs", help="Show runtime logs during chat."),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume the previous CLI session."),
) -> None:
    """Run a single agent turn or start interactive REPL."""
    _setup_logging(verbose=verbose, quiet=not logs and not verbose)

    from clambot.agent.bootstrap import build_provider_backed_agent_loop_from_config
    from clambot.agent.turn_execution import process_turn_with_persistence_and_execution
    from clambot.async_runner import run_sync
    from clambot.bus.events import InboundMessage
    from clambot.config.loader import load_config
    from clambot.providers.factory import create_provider
    from clambot.session.manager import SessionManager
    from clambot.tools import build_tool_registry
    from clambot.tools.secrets.store import SecretStore
    from clambot.workspace.bootstrap import bootstrap_workspace

    cfg = load_config(config)
    workspace = Path(cfg.agents.defaults.workspace).expanduser()
    bootstrap_workspace(workspace)

    secrets_dir = workspace / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    secret_store = SecretStore(secrets_dir / "secrets.json")
    tool_registry = build_tool_registry(
        workspace=workspace,
        config=cfg,
        secret_store=secret_store,
    )
    provider = create_provider(cfg)
    agent_loop = build_provider_backed_agent_loop_from_config(
        config=cfg,
        tool_registry=tool_registry,
        workspace=workspace,
    )
    session_manager = SessionManager(workspace)

    # Session ID: fresh per run by default, --resume continues the last one.
    if resume:
        session_chat_id = "agent"
    else:
        import uuid

        session_chat_id = f"agent-{uuid.uuid4().hex[:8]}"

    def _cli_secret_prompt(missing: list[str]) -> dict[str, str]:
        """Prompt the user for missing secrets via getpass."""
        import getpass

        results: dict[str, str] = {}
        for name in missing:
            try:
                value = getpass.getpass(f"Secret '{name}' required. Enter value: ")
                if value:
                    results[name] = value
            except (EOFError, KeyboardInterrupt):
                break
        return results

    async def run_turn(text: str) -> str:
        inbound = InboundMessage(
            channel="cli",
            source="user",
            chat_id=session_chat_id,
            content=text,
        )
        outbound = await process_turn_with_persistence_and_execution(
            inbound=inbound,
            agent_loop=agent_loop,
            session_manager=session_manager,
            config=cfg,
            provider=provider,
            workspace=workspace,
            secret_prompt_callback=_cli_secret_prompt,
            secret_store=secret_store,
        )
        return outbound.content if outbound else ""

    if message:
        # Single-turn mode
        result = run_sync(run_turn(message))
        typer.echo(result)
    else:
        # Interactive REPL with persistent history
        import atexit
        import readline

        history_path = workspace / ".agent_history"
        try:
            readline.read_history_file(history_path)
        except FileNotFoundError:
            pass
        readline.set_history_length(1000)
        atexit.register(readline.write_history_file, history_path)

        typer.echo("ClamBot agent REPL. Type 'exit' or Ctrl+C to quit.\n")
        try:
            while True:
                try:
                    text = input("you> ").strip()
                except EOFError:
                    break
                if not text or text.lower() in ("exit", "quit"):
                    break
                result = run_sync(run_turn(text))
                typer.echo(f"\nbot> {result}\n")
        except KeyboardInterrupt:
            typer.echo("\nBye!")


# ---------------------------------------------------------------------------
# clambot gateway
# ---------------------------------------------------------------------------


@app.command()
def gateway(
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Start the ClamBot gateway daemon."""
    _setup_logging(verbose)
    from clambot.config.loader import load_config
    from clambot.gateway.main import gateway_main

    cfg = load_config(config)
    asyncio.run(gateway_main(config=cfg, config_path=config))


# ---------------------------------------------------------------------------
# clambot onboard
# ---------------------------------------------------------------------------


@app.command()
def onboard(
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Initialize workspace and auto-detect providers."""
    _setup_logging(verbose)

    from clambot.config.loader import resolve_config_path
    from clambot.workspace.bootstrap import bootstrap_workspace
    from clambot.workspace.onboard import onboard_workspace

    config_path = resolve_config_path(config)

    # Bootstrap workspace
    from clambot.config.schema import ClamBotConfig

    workspace = Path(ClamBotConfig().agents.defaults.workspace).expanduser()
    bootstrap_workspace(workspace)
    typer.echo(f"Workspace bootstrapped at {workspace}")

    # Onboard
    summary = onboard_workspace(config_path)
    if summary["configured_providers"]:
        typer.echo(f"Providers configured: {', '.join(summary['configured_providers'])}")
    else:
        typer.echo("No new providers detected.")


# ---------------------------------------------------------------------------
# clambot status
# ---------------------------------------------------------------------------


@app.command()
def status(
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
) -> None:
    """Show provider readiness and model alignment."""
    from clambot.config.loader import load_config

    cfg = load_config(config)

    typer.echo("ClamBot Status\n" + "=" * 40)
    typer.echo(f"Workspace: {cfg.agents.defaults.workspace}")
    typer.echo(f"Default model: {cfg.agents.defaults.model}")
    typer.echo(f"Selector model: {cfg.agents.selector.model}")
    typer.echo(f"Telegram: {'enabled' if cfg.channels.telegram.enabled else 'disabled'}")
    typer.echo(f"Heartbeat: {'enabled' if cfg.heartbeat.enabled else 'disabled'}")
    typer.echo(f"Cron: {'enabled' if cfg.cron.enabled else 'disabled'}")

    # Show configured providers
    typer.echo("\nProviders:")
    providers = cfg.providers
    for name in (
        "anthropic",
        "openai",
        "openrouter",
        "ollama",
        "deepseek",
        "groq",
        "gemini",
        "custom",
    ):
        pcfg = getattr(providers, name, None)
        if pcfg and (pcfg.api_key or pcfg.api_base):
            marker = "configured" if pcfg.api_key else f"base={pcfg.api_base}"
            typer.echo(f"  {name}: {marker}")
        else:
            typer.echo(f"  {name}: not configured")


# ---------------------------------------------------------------------------
# clambot cron subcommands
# ---------------------------------------------------------------------------


@cron_app.command("list")
def cron_list(
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
) -> None:
    """List all cron jobs."""
    from clambot.config.loader import load_config
    from clambot.cron.store import load_cron_store

    cfg = load_config(config)
    workspace = Path(cfg.agents.defaults.workspace).expanduser()
    cron_path = workspace / "cron" / "jobs.json"

    store = load_cron_store(cron_path)
    if not store.jobs:
        typer.echo("No cron jobs.")
        return

    typer.echo(
        f"{'ID':<38} {'Name':<25} {'Schedule':<20} {'Enabled':<8} {'Last Status':<12} {'Next Run'}"
    )
    typer.echo("-" * 120)
    for job in store.jobs:
        schedule_str = job.schedule.kind
        if job.schedule.kind == "every":
            schedule_str = f"every {job.schedule.every_seconds}s"
        elif job.schedule.kind == "cron":
            schedule_str = job.schedule.cron_expr or "?"
        elif job.schedule.kind == "at":
            schedule_str = f"at {job.schedule.at_ms}"

        typer.echo(
            f"{job.id:<38} {job.name:<25} {schedule_str:<20} "
            f"{'yes' if job.enabled else 'no':<8} "
            f"{job.state.last_status or '-':<12} "
            f"{job.state.next_run_at_ms or '-'}"
        )


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", help="Job name."),
    message: str = typer.Option(..., "--message", help="Message to send."),
    cron_expr: str | None = typer.Option(None, "--cron", help="Cron expression (5-field)."),
    timezone: str = typer.Option("UTC", "--timezone", help="Timezone for cron."),
    every: str | None = typer.Option(None, "--every", help="Interval (e.g. 60s, 5m, 2h, 1d)."),
    at: str | None = typer.Option(None, "--at", help="One-time run at ISO8601 datetime."),
    deliver: bool = typer.Option(True, "--deliver/--no-deliver", help="Deliver to channel."),
    channel: str | None = typer.Option(None, "--channel", help="Target channel."),
    target: str | None = typer.Option(None, "--target", help="Target chat_id."),
    delete_after_run: bool = typer.Option(
        False, "--delete-after-run", help="Delete after execution."
    ),
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
) -> None:
    """Add a new cron job."""
    from clambot.config.loader import load_config
    from clambot.cron.schedule import (
        parse_duration_to_seconds,
        parse_iso8601_to_epoch_ms,
        parse_schedule,
    )
    from clambot.cron.service import InMemoryCronService

    cfg = load_config(config)
    workspace = Path(cfg.agents.defaults.workspace).expanduser()
    cron_path = workspace / "cron" / "jobs.json"
    cron_path.parent.mkdir(parents=True, exist_ok=True)

    # Build schedule args — keys must match what parse_schedule expects
    schedule_args: dict = {}
    if cron_expr:
        schedule_args["cron_expr"] = cron_expr
        schedule_args["timezone"] = timezone
    elif every:
        schedule_args["every_seconds"] = parse_duration_to_seconds(every)
    elif at:
        schedule_args["at_ms"] = parse_iso8601_to_epoch_ms(at)
    else:
        typer.echo("Error: must specify --cron, --every, or --at", err=True)
        raise typer.Exit(1)

    try:
        schedule = parse_schedule(schedule_args)
    except ValueError as exc:
        typer.echo(f"Invalid schedule: {exc}", err=True)
        raise typer.Exit(1) from exc

    service = InMemoryCronService(store_path=cron_path, workspace=workspace)

    async def _add() -> None:
        await service.start()
        job = service.add_job(
            name=name,
            schedule=schedule,
            message=message,
            deliver=deliver,
            channel=channel,
            target=target,
            delete_after_run=delete_after_run,
        )
        typer.echo(f"Job added: {job.id} ({job.name})")

    asyncio.run(_add())


@cron_app.command("remove")
def cron_remove(
    job_id: str = typer.Argument(..., help="Job ID to remove."),
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
) -> None:
    """Remove a cron job."""
    from clambot.config.loader import load_config
    from clambot.cron.service import InMemoryCronService

    cfg = load_config(config)
    workspace = Path(cfg.agents.defaults.workspace).expanduser()
    cron_path = workspace / "cron" / "jobs.json"

    service = InMemoryCronService(store_path=cron_path, workspace=workspace)

    async def _remove() -> None:
        await service.start()
        removed = service.remove_job(job_id)
        if removed:
            typer.echo(f"Job '{job_id}' removed.")
        else:
            typer.echo(f"Job '{job_id}' not found.", err=True)

    asyncio.run(_remove())


@cron_app.command("enable")
def cron_enable(
    job_id: str = typer.Argument(..., help="Job ID to enable."),
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
) -> None:
    """Enable a cron job."""
    from clambot.config.loader import load_config
    from clambot.cron.service import InMemoryCronService

    cfg = load_config(config)
    workspace = Path(cfg.agents.defaults.workspace).expanduser()
    cron_path = workspace / "cron" / "jobs.json"

    service = InMemoryCronService(store_path=cron_path, workspace=workspace)

    async def _enable() -> None:
        await service.start()
        job = service.enable_job(job_id)
        if job:
            typer.echo(f"Job '{job_id}' enabled.")
        else:
            typer.echo(f"Job '{job_id}' not found.", err=True)

    asyncio.run(_enable())


@cron_app.command("disable")
def cron_disable(
    job_id: str = typer.Argument(..., help="Job ID to disable."),
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
) -> None:
    """Disable a cron job."""
    from clambot.config.loader import load_config
    from clambot.cron.service import InMemoryCronService

    cfg = load_config(config)
    workspace = Path(cfg.agents.defaults.workspace).expanduser()
    cron_path = workspace / "cron" / "jobs.json"

    service = InMemoryCronService(store_path=cron_path, workspace=workspace)

    async def _disable() -> None:
        await service.start()
        job = service.disable_job(job_id)
        if job:
            typer.echo(f"Job '{job_id}' disabled.")
        else:
            typer.echo(f"Job '{job_id}' not found.", err=True)

    asyncio.run(_disable())


@cron_app.command("run")
def cron_run(
    job_id: str = typer.Argument(..., help="Job ID to trigger manually."),
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
) -> None:
    """Manually trigger a cron job."""
    from clambot.config.loader import load_config
    from clambot.cron.store import load_cron_store

    cfg = load_config(config)
    workspace = Path(cfg.agents.defaults.workspace).expanduser()
    cron_path = workspace / "cron" / "jobs.json"

    store = load_cron_store(cron_path)
    job = next((j for j in store.jobs if j.id == job_id), None)
    if job is None:
        typer.echo(f"Job '{job_id}' not found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Triggering job '{job.name}' ({job.id})...")
    typer.echo(f"Message: {job.payload.message}")
    typer.echo("(Manual trigger — use 'clambot gateway' for full execution)")


# ---------------------------------------------------------------------------
# clambot channels connect telegram
# ---------------------------------------------------------------------------


@channels_app.command("connect")
def channels_connect(
    channel: str = typer.Argument(
        "telegram", help="Channel to connect (only 'telegram' supported)."
    ),
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
) -> None:
    """Connect a channel (interactive setup)."""
    if channel != "telegram":
        typer.echo(f"Unsupported channel: {channel}. Only 'telegram' is supported.", err=True)
        raise typer.Exit(1)

    from clambot.config.loader import resolve_config_path

    config_path = resolve_config_path(config)

    typer.echo("Telegram Channel Setup")
    typer.echo("=" * 40)
    typer.echo("1. Message @BotFather on Telegram")
    typer.echo("2. Create a new bot with /newbot")
    typer.echo("3. Copy the API token\n")

    token = typer.prompt("Enter your bot token")
    if not token:
        typer.echo("No token provided.", err=True)
        raise typer.Exit(1)

    typer.echo("\nStarting temporary bot to detect your user ID...")
    typer.echo("Send any message to your bot on Telegram.\n")

    try:
        user_id = asyncio.run(_detect_telegram_user(token))
    except Exception as exc:
        typer.echo(f"Failed to detect user: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not user_id:
        typer.echo("No message received. Try again.", err=True)
        raise typer.Exit(1)

    # Update config
    _update_telegram_config(config_path, token, str(user_id))
    typer.echo(f"\nTelegram configured! User ID: {user_id}")
    typer.echo(f"Config updated at {config_path}")


async def _detect_telegram_user(token: str, timeout: float = 60.0) -> str | None:
    """Start a temporary bot and wait for the first message to detect user_id."""
    try:
        from telegram import Bot

        bot = Bot(token=token)
        deadline = asyncio.get_event_loop().time() + timeout

        # Clear pending updates
        async with bot:
            updates = await bot.get_updates(offset=-1, timeout=1)
            offset = updates[-1].update_id + 1 if updates else 0

            while asyncio.get_event_loop().time() < deadline:
                updates = await bot.get_updates(offset=offset, timeout=5)
                for update in updates:
                    offset = update.update_id + 1
                    if update.message and update.message.from_user:
                        user = update.message.from_user
                        print(f"Detected user: {user.first_name} (ID: {user.id})")
                        return str(user.id)

    except ImportError:
        print("python-telegram-bot is not installed.")
    except Exception as exc:
        print(f"Error: {exc}")

    return None


def _update_telegram_config(config_path: Path, token: str, user_id: str) -> None:
    """Update config.json with Telegram settings."""
    config_path = Path(config_path)
    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    channels = existing.setdefault("channels", {})
    telegram = channels.setdefault("telegram", {})
    telegram["enabled"] = True
    telegram["token"] = token

    allow_from = telegram.get("allowFrom", [])
    if user_id not in allow_from:
        allow_from.append(user_id)
    telegram["allowFrom"] = allow_from

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# clambot provider connect ollama
# ---------------------------------------------------------------------------


_API_KEY_PROVIDERS = ("openrouter", "anthropic", "openai", "deepseek", "gemini", "groq")


@provider_app.command("connect")
def provider_connect(
    provider: str = typer.Argument(..., help="Provider to connect (e.g. 'openrouter', 'ollama')."),
    host: str | None = typer.Option(None, "--host", help="Provider host URL."),
    set_default: bool = typer.Option(
        True, "--set-default/--no-set-default", help="Set the selected model as default."
    ),
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
) -> None:
    """Connect to a provider (interactive setup)."""
    if provider == "ollama":
        _connect_ollama(host=host, set_default=set_default, config_path=config)
    elif provider in _API_KEY_PROVIDERS:
        _connect_api_key_provider(provider, set_default=set_default, config_path=config)
    else:
        supported = ", ".join([*_API_KEY_PROVIDERS, "ollama"])
        typer.echo(f"Unknown provider: {provider}. Supported: {supported}", err=True)
        raise typer.Exit(1)


def _connect_api_key_provider(
    provider: str,
    *,
    set_default: bool = True,
    config_path: str | None = None,
) -> None:
    """Interactive setup for API-key-based providers.

    Steps:
    1. Prompt for API key (masked input)
    2. Verify the key with a lightweight LLM call
    3. Let the user pick a model
    4. Write provider config and optionally set as default model
    """
    import getpass

    import questionary

    from clambot.config.loader import resolve_config_path
    from clambot.workspace.onboard import _PROVIDER_MODELS, _SELECTOR_MODELS

    display = provider.replace("_", " ").title()
    typer.echo(f"{display} Provider Setup")
    typer.echo("=" * 40)

    # ── Step 1: API key ───────────────────────────────────────────
    api_key = getpass.getpass(f"Enter your {display} API key: ").strip()
    if not api_key:
        typer.echo("No API key provided.", err=True)
        raise typer.Exit(1)

    # ── Step 2: Model selection ───────────────────────────────────
    models = _PROVIDER_MODELS.get(provider, [])
    selected_model: str | None = None
    if set_default and models:
        choices = [questionary.Choice(title=label, value=value) for label, value in models]
        selected_model = questionary.select(
            "Select a default model:",
            choices=choices,
        ).ask()
        if not selected_model:
            typer.echo("No model selected.", err=True)
            raise typer.Exit(1)
        typer.echo(f"\nSelected: {selected_model}")

    # ── Step 4: Write config ──────────────────────────────────────
    resolved_path = resolve_config_path(config_path)
    _update_api_key_config(
        resolved_path, provider, api_key, selected_model,
        selector_model=_SELECTOR_MODELS.get(provider),
    )

    typer.echo(f"\n{display} configured! ✅")
    if selected_model:
        typer.echo(f"Default model: {selected_model}")
    typer.echo(f"Config updated: {resolved_path}")


def _update_api_key_config(
    config_path: Path,
    provider: str,
    api_key: str,
    default_model: str | None = None,
    selector_model: str | None = None,
) -> None:
    """Write API key provider settings to config.json."""
    config_path = Path(config_path)
    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Set provider API key
    providers = existing.setdefault("providers", {})
    prov_cfg = providers.setdefault(provider, {})
    prov_cfg["apiKey"] = api_key

    # Set default model
    if default_model:
        agents = existing.setdefault("agents", {})
        defaults = agents.setdefault("defaults", {})
        defaults["model"] = default_model

    # Set selector model
    if selector_model and default_model:
        agents = existing.setdefault("agents", {})
        selector = agents.setdefault("selector", {})
        if not selector.get("model"):
            selector["model"] = selector_model

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _connect_ollama(
    *,
    host: str | None = None,
    set_default: bool = True,
    config_path: str | None = None,
) -> None:
    """Interactive Ollama connection setup.

    Steps:
    1. Prompt for host URL (default ``http://localhost:11434``)
    2. Probe ``/api/tags`` to confirm reachability
    3. List available models and let user pick one
    4. Write provider config and optionally set as default model
    """
    import questionary

    from clambot.config.loader import resolve_config_path
    from clambot.utils.constants import OLLAMA_DEFAULT_HOST

    typer.echo("Ollama Provider Setup")
    typer.echo("=" * 40)

    # ── Step 1: Host URL ──────────────────────────────────────────────
    default_host = OLLAMA_DEFAULT_HOST
    if not host:
        host = typer.prompt("Ollama host URL", default=default_host)
    host = host.rstrip("/")
    typer.echo(f"\nProbing {host} ...")

    # ── Step 2: Probe ─────────────────────────────────────────────────
    models = _probe_ollama_models(host)
    if models is None:
        typer.echo(f"Could not reach Ollama at {host}", err=True)
        typer.echo("Make sure Ollama is running: ollama serve")
        raise typer.Exit(1)

    if not models:
        typer.echo("Ollama is reachable but has no models pulled.")
        typer.echo("Pull a model first: ollama pull llama3")
        raise typer.Exit(1)

    typer.echo(f"Connected! Found {len(models)} model(s).\n")

    # ── Step 3: Model selection ───────────────────────────────────────
    # Build display labels with size info
    choices = []
    for m in models:
        name = m["name"]
        size_gb = m.get("size", 0) / (1024**3)
        label = f"{name}  ({size_gb:.1f} GB)" if size_gb > 0 else name
        choices.append(questionary.Choice(title=label, value=name))

    selected = questionary.select(
        "Select a model:",
        choices=choices,
    ).ask()

    if not selected:
        typer.echo("No model selected.", err=True)
        raise typer.Exit(1)

    # Prefix with ollama/ for LiteLLM routing
    model_string = f"ollama/{selected}" if not selected.startswith("ollama/") else selected

    typer.echo(f"\nSelected: {model_string}")

    # ── Step 4: Write config ──────────────────────────────────────────
    resolved_path = resolve_config_path(config_path)
    _update_ollama_config(resolved_path, host, model_string if set_default else None)

    typer.echo(f"\nOllama configured at {host}")
    if set_default:
        typer.echo(f"Default model set to {model_string}")
    typer.echo(f"Config updated: {resolved_path}")


def _probe_ollama_models(host: str) -> list[dict] | None:
    """Probe Ollama ``/api/tags`` and return the list of models.

    Returns:
        List of model dicts (each with ``name``, ``size``, etc.) on success,
        empty list if reachable but no models, or ``None`` if unreachable.
    """
    try:
        import urllib.error
        import urllib.request

        url = f"{host}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return None
            import json as _json

            data = _json.loads(resp.read().decode("utf-8"))
            return data.get("models", [])
    except Exception as exc:
        typer.echo(f"  Error: {exc}", err=True)
        return None


def _update_ollama_config(
    config_path: Path,
    host: str,
    default_model: str | None = None,
) -> None:
    """Write Ollama provider settings to config.json."""
    config_path = Path(config_path)
    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Set ollama provider
    providers = existing.setdefault("providers", {})
    ollama = providers.setdefault("ollama", {})
    ollama["apiBase"] = host

    # Optionally set default model
    if default_model:
        agents = existing.setdefault("agents", {})
        defaults = agents.setdefault("defaults", {})
        defaults["model"] = default_model

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# clambot provider login
# ---------------------------------------------------------------------------


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="Provider to login (e.g. 'openai-codex')."),
    set_default: bool = typer.Option(
        True, "--set-default/--no-set-default", help="Set as default model after login."
    ),
    config: str | None = typer.Option(None, "--config", help="Path to config.json."),
) -> None:
    """Login to a provider (OAuth flow)."""
    if provider == "openai-codex":
        from oauth_cli_kit import OPENAI_CODEX_PROVIDER, login_oauth_interactive

        typer.echo("OpenAI Codex OAuth Login")
        typer.echo("=" * 40)
        typer.echo("A browser window will open for authentication.\n")

        try:
            token = login_oauth_interactive(
                print_fn=typer.echo,
                prompt_fn=lambda msg: input(msg),
                provider=OPENAI_CODEX_PROVIDER,
                originator="clambot",
            )
            typer.echo(f"\nLogin successful! Account: {token.account_id or 'unknown'}")

            if set_default:
                from clambot.config.loader import resolve_config_path

                config_path = resolve_config_path(config)
                _set_default_model(config_path, "openai-codex/gpt-5.3-codex")
                typer.echo("Default model set to openai-codex/gpt-5.3-codex")
            else:
                typer.echo("Token stored — you can now use openai-codex/ models.")
        except KeyboardInterrupt:
            typer.echo("\nLogin cancelled.")
        except Exception as exc:
            typer.echo(f"\nLogin failed: {exc}", err=True)
            raise typer.Exit(1) from exc
    else:
        typer.echo(f"Unknown provider: {provider}. Supported: openai-codex")
        typer.echo("(For API key providers, use: uv run clambot provider connect <name>)")


def _set_default_model(config_path: Path, model: str) -> None:
    """Update the default model in config.json."""
    config_path = Path(config_path)
    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    agents = existing.setdefault("agents", {})
    defaults = agents.setdefault("defaults", {})
    defaults["model"] = model

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    app()
