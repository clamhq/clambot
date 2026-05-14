<div align="center">
  <h1>🐚 ClamBot: Secure AI Agent with WASM Sandbox Execution</h1>
  <p>
    <img src="https://img.shields.io/badge/version-0.1.2-blue" alt="Version">
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </p>
</div>

🐚 **ClamBot** is a security-focused personal AI assistant that runs all LLM-generated code inside a **WASM sandbox** (QuickJS inside Wasmtime) — eliminating the arbitrary code execution risks of `exec()`/`subprocess.run()` patterns common in other agent frameworks.

✨ Inspired by [OpenClaw](https://github.com/openclaw/openclaw) and [nanobot](https://github.com/HKUDS/nanobot).

> [!IMPORTANT]
> ClamBot is tested primarily with **OpenAI Codex** on **Linux** and this is the recommended provider for the best out-of-the-box reliability.

🔒 Every other agent framework runs LLM-generated code directly on your machine. ClamBot isolates it:

1. 🤖 LLM generates a JavaScript **"clam"** (named, versioned, reusable script)
2. 📦 The clam runs inside **amla-sandbox** (WASM/QuickJS) with memory isolation
3. ✅ Tool calls yield back to Python for **capability-checked, approval-gated** dispatch
4. ♻️ Successful clams are **persisted and reused** for identical future requests — zero latency, zero cost

## ✨ Key Features

🔒 **WASM Sandbox Execution** — all generated code runs in QuickJS/Wasmtime with memory isolation and no ambient network access

🛡️ **Interactive Approval Gate** — SHA-256 fingerprinted tool approvals with always-grants, turn-scoped grants, and per-tool scope options

♻️ **Clam Reuse** — successful scripts are promoted and reused for identical requests without any LLM call

🔧 **Self-Fix Loop** — up to 3 automatic retries with LLM-guided fix instructions on runtime failures

🤖 **Multi-Provider LLM** — tested path with OpenAI Codex (OAuth, recommended) plus OpenRouter, Anthropic, OpenAI, Gemini, DeepSeek, Ollama, and custom endpoints

💬 **Telegram Integration** — typing indicators, phase status messages, MarkdownV2 rendering, inline approval keyboards, file uploads

🧠 **Long-Term Memory** — MEMORY.md (durable facts auto-injected into prompts) + HISTORY.md (searchable interaction summaries)

⏰ **Cron Scheduling** — persistent timezone-aware jobs with `cron`, `every`, and `at` schedule types

💓 **Heartbeat Service** — proactive agent wakeup with task-driven execution from HEARTBEAT.md

🔑 **Host-Managed Secrets** — atomic-write store with 0600 permissions; secrets never appear in tool args, logs, or traces

🌐 **SSRF Protection** — private IP blocking on all outbound HTTP tools

📝 **Session Compaction** — automatic LLM-summarized compaction to prevent context window overflow

## ✅ Out-of-the-Box User Features

After onboarding, users can immediately:

- 💬 Chat in CLI (single-turn and interactive REPL modes)
- 📱 Use Telegram with typing indicators, approvals, status updates, and file uploads
- 🌐 Fetch web pages and call HTTP APIs
- 🎙️ Transcribe YouTube/media audio and summarize/translate the transcript
- 📄 Extract text from PDF files (including uploaded files)
- ⏰ Schedule reminders and recurring jobs (`cron`, `every`, `at`)
- 🧠 Use long-term memory recall + searchable history
- 🔑 Add/store secrets and approve tool access interactively

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Inbound Sources                                               │
│  ┌───────────┐  ┌─────────────┐  ┌───────────┐  ┌──────────┐ │
│  │ 💬 Telegram│  │ ⏰ Cron     │  │ 💓 Heartbeat│ │ 🖥️ CLI   │ │
│  └─────┬─────┘  └──────┬──────┘  └─────┬─────┘  └────┬─────┘ │
└────────┼───────────────┼──────────────┼────────────┼──────────┘
         ▼               ▼              ▼            ▼
┌────────────────────────────────────────────────────────────────┐
│  🎛️ Gateway Orchestrator                                       │
│  /approve · /secret · /new command routing                     │
└────────────────────────┬───────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  🧠 Agent Pipeline                                             │
│                                                                │
│  1. 📂 Session load + auto-compaction                          │
│  2. 🔀 Clam Selector (pre-selection → LLM routing)             │
│  3. ⚡ Clam Generator (LLM → JavaScript)                       │
│  4. 📦 WASM Runtime (QuickJS sandbox + approval-gated tools)   │
│  5. 🔍 Post-Runtime Analyzer (ACCEPT / SELF_FIX / REJECT)      │
│  6. 🧠 Background memory extraction (fire-and-forget)          │
└────────────────────────┬───────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  📤 Outbound → Telegram / CLI                                  │
└────────────────────────────────────────────────────────────────┘
```

## 📦 Install

```bash
git clone https://github.com/clamguy/clambot.git
cd clambot
```

## 🚀 Quick Start

> [!TIP]
> Recommended: **OpenAI Codex** (OAuth, tested path) via `uv run clambot provider login openai-codex`.
> API-key providers are also supported: [OpenRouter](https://openrouter.ai/keys) · [Anthropic](https://console.anthropic.com) · [OpenAI](https://platform.openai.com)

**1. 🎬 Initialize** — auto-detects configured providers (API keys and Codex OAuth) and sets up workspace:

```bash
# Recommended: OpenAI Codex (OAuth)
uv run clambot provider login openai-codex

# Initialize workspace + config
uv run clambot onboard
```

`uv run clambot onboard` scans your environment variables, probes local Ollama, and generates `~/.clambot/config.json` with everything it finds. No manual editing needed.

**2. ✅ Verify**

```bash
uv run clambot status
```

**3. 💬 Chat**

```bash
uv run clambot agent
```

That's it! You have a working sandboxed AI assistant in under a minute. 🎉

> [!NOTE]
> If you need to tweak settings later, edit `~/.clambot/config.json` — see [⚙️ Configuration](#%EF%B8%8F-configuration) below.

## 💬 Telegram

Connect ClamBot to Telegram for a full mobile experience with inline approval buttons, typing indicators, and phase status messages.

**1. 🤖 Create a bot** — Open Telegram, search `@BotFather`, send `/newbot`, follow prompts, copy the token.

**2. 🔗 Connect** — the interactive command handles everything:

```bash
uv run clambot channels connect telegram
# Enter bot token → press "Connect" in bot → user ID auto-added → done!
```

**3. 🚀 Run the gateway**

```bash
uv run clambot gateway
```

That's it — message your bot on Telegram and ClamBot responds! 🎉

<details>
<summary>📝 Manual configuration (advanced)</summary>

If you prefer to configure manually, add the following to `~/.clambot/config.json`:

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

> `allowFrom`: Leave empty to allow all users, or add user IDs/usernames to restrict access.

</details>

## 🤖 Providers

ClamBot supports multiple LLM backends through a registry-driven provider layer. Use Codex OAuth or set provider API keys, then run `uv run clambot onboard` — provider/model selection is auto-detected.

| Provider | Purpose | Setup |
|----------|---------|-------|
| `openai_codex` | ⚡ LLM (recommended, primary tested path) | `uv run clambot provider login openai-codex` |
| `openrouter` | 🌐 LLM gateway (optional) | `export OPENROUTER_API_KEY=sk-or-...` |
| `anthropic` | 🧠 LLM (Claude direct) | `export ANTHROPIC_API_KEY=sk-ant-...` |
| `openai` | 💡 LLM (GPT direct) | `export OPENAI_API_KEY=sk-...` |
| `deepseek` | 🔬 LLM (DeepSeek direct) | `export DEEPSEEK_API_KEY=...` |
| `gemini` | 💎 LLM (Gemini direct) | `export GEMINI_API_KEY=...` |
| `groq` | 🎙️ LLM + voice transcription (Whisper) | `export GROQ_API_KEY=...` |
| `ollama` | 🏠 LLM (local, any model) | `ollama serve` (auto-probed) |
| `custom` | 🔌 Any OpenAI-compatible endpoint | Config only — see below |

```bash
# Example: set up with OpenAI Codex (recommended)
uv run clambot provider login openai-codex
uv run clambot onboard    # auto-detects provider + model
uv run clambot status     # verify provider is ready ✅
uv run clambot agent      # start chatting 💬
```

<details>
<summary>⚡ <b>OpenAI Codex (OAuth)</b></summary>

Codex uses OAuth instead of API keys. Requires a ChatGPT Plus or Pro account.

```bash
# 1. Login (opens browser)
uv run clambot provider login openai-codex

# 2. Chat — model auto-configured
uv run clambot agent -m "Hello!"
```

</details>

<details>
<summary>🔌 <b>Custom Provider (Any OpenAI-compatible API)</b></summary>

Connects directly to any OpenAI-compatible endpoint — LM Studio, llama.cpp, Together AI, Fireworks, Azure OpenAI, or any self-hosted server. Add to `~/.clambot/config.json`:

```json
{
  "providers": {
    "custom": {
      "apiKey": "your-api-key",
      "apiBase": "https://api.your-provider.com/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "your-model-name"
    }
  }
}
```

> For local servers that don't require a key, set `apiKey` to any non-empty string (e.g. `"no-key"`).

</details>

<details>
<summary>🏠 <b>Ollama (local)</b></summary>

Start Ollama and let `onboard` auto-detect it:

```bash
# 1. Start Ollama
ollama serve

# 2. Onboard auto-probes Ollama and discovers available models
uv run clambot onboard

# 3. Chat
uv run clambot agent
```

</details>

## ⚙️ Configuration

Config file: `~/.clambot/config.json` (auto-generated by `uv run clambot onboard`)

📖 See [docs/configuration.md](docs/configuration.md) for the full schema reference.

### 🔒 Security

> [!TIP]
> For production deployments, set `"restrictToWorkspace": true` in your tools config to sandbox file access.

| Option | Default | Description |
|--------|---------|-------------|
| `tools.filesystem.restrictToWorkspace` | `true` | 📁 Restricts filesystem tool to the workspace directory. Prevents path traversal. |
| `security.sslFallbackInsecure` | `false` | 🔓 When `true`, HTTP tools retry with `verify=False` on SSL errors. Only for sandboxed environments. |
| `channels.telegram.allowFrom` | `[]` (allow all) | 👤 Whitelist of user IDs. Empty = allow everyone. |
| SSRF protection | Always on | 🌐 Blocks requests to `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16`, `::1`, `fc00::/7` |
| Secret redaction | Always on | 🔑 Secret values never appear in tool args, events, approval records, or logs |

### 🛡️ Tool Approvals

Every tool call from generated code goes through an approval gate:

```
🔍 Tool call arrives
├─ ✅ Check always_grants → ALLOW immediately
├─ 🔄 Check turn-scoped grants → ALLOW if same resource
└─ 🙋 Interactive prompt → Allow Once / Allow Always (scoped) / Reject
```

Configure pre-approved patterns in `~/.clambot/config.json`:
```json
{
  "agents": {
    "approvals": {
      "enabled": true,
      "interactive": true,
      "alwaysGrants": [
        {"tool": "web_fetch", "scope": "host:api.coinbase.com"},
        {"tool": "fs", "scope": "workspace"}
      ]
    }
  }
}
```

## 🧰 Built-In Tools

All tools below are available out of the box and callable from generated JavaScript clams via `await tool_name({...})`.

| Tool | Description |
|------|-------------|
| 📁 `fs` | Filesystem operations: read, write, edit, list |
| 🌐 `http_request` | Authenticated HTTP with secret-based bearer tokens |
| 🔎 `web_search` | Search the web for pages, news, images, and videos |
| 🔗 `web_fetch` | URL content fetching |
| 🎙️ `transcribe` | Download + transcribe media audio (YouTube and other yt-dlp supported sources) |
| 📄 `pdf_reader` | Extract text from PDF files |
| ⏰ `cron` | Schedule management: add, list, remove jobs |
| 🔑 `secrets_add` | Secret storage with multiple resolution sources |
| 🧠 `memory_recall` | Read MEMORY.md durable facts |
| 🔍 `memory_search_history` | Search HISTORY.md interaction summaries |
| 📢 `echo` | Debug output tool (optional) |

## 🖥️ CLI Reference

| Command | Description |
|---------|-------------|
| `uv run clambot onboard` | 🎬 Initialize config & workspace (auto-detects providers) |
| `uv run clambot agent -m "..."` | 💬 Run a single agent turn |
| `uv run clambot agent` | 🔄 Interactive chat mode (REPL) |
| `uv run clambot gateway` | 🚀 Start the gateway (Telegram + cron + heartbeat) |
| `uv run clambot status` | ✅ Show provider readiness |
| `uv run clambot provider login openai-codex` | 🔑 OAuth login for Codex |
| `uv run clambot channels connect telegram` | 💬 Interactive Telegram setup |
| `uv run clambot cron list` | 📋 List scheduled jobs |
| `uv run clambot cron add --name "daily" --message "Hello" --cron "0 9 * * *"` | ➕ Add a cron job |
| `uv run clambot cron remove <job_id>` | ❌ Remove a cron job |

Interactive mode exits: `exit`, `quit`, `/exit`, `/quit`, `:q`, or `Ctrl+D`.

## 📁 Project Structure

```
clambot/
├── agent/             # 🧠 Core agent logic (loop, selector, generator, runtime, approvals)
│   ├── loop.py        #    Agent pipeline orchestration
│   ├── selector.py    #    Two-stage clam routing (pre-selection + LLM)
│   ├── generator.py   #    LLM-based JavaScript generation
│   ├── runtime.py     #    WASM execution wrapper + timeout/cancellation
│   ├── approvals.py   #    Capability-gated approval gate
│   └── tools/         #    Built-in tool implementations
├── bus/               # 🚌 Async message routing (inbound + outbound queues)
├── channels/          # 💬 Chat channel integrations (Telegram)
├── cli/               # 🖥️ Typer CLI commands
├── config/            # ⚙️ Config schema (Pydantic) + loader
├── cron/              # ⏰ Persistent timezone-aware job scheduling
├── gateway/           # 🎛️ Gateway orchestrator (connects all subsystems)
├── heartbeat/         # 💓 Proactive scheduled agent wakeup
├── memory/            # 🧠 Long-term memory (MEMORY.md + HISTORY.md)
├── providers/         # 🤖 LLM provider layer (LiteLLM, Codex, custom)
├── session/           # 💬 Conversation session management (JSONL)
├── tools/             # 🧰 Built-in tool implementations
├── utils/             # 🔧 Shared utilities (tracked tasks, text processing)
└── workspace/         # 📂 Workspace bootstrap + onboarding
```

## 🔬 How It Works

### 🐚 The Clam Lifecycle

```
User request: "What is the price of BTC?"
│
├─ ♻️ Pre-selection: exact match against existing clams? → YES → reuse (zero LLM cost)
│                                                         → NO  ↓
├─ 🔀 Selector LLM: generate_new / select_existing / chat
│
├─ ⚡ Generator LLM → JavaScript clam:
│   async function run(args) {
│     const res = await http_request({
│       method: "GET",
│       url: "https://api.coinbase.com/v2/prices/BTC-USD/spot"
│     });
│     return JSON.parse(res.content).data;
│   }
│
├─ 📦 WASM Sandbox executes clam
│   └─ http_request → 🛡️ Approval Gate → Python host dispatch → result
│
├─ 🔍 Post-Runtime Analyzer: ACCEPT → promote to clams/ for future reuse
│                             SELF_FIX → retry with fix instructions (up to 3×)
│                             REJECT → return error
│
└─ 📤 Response delivered → 🧠 background memory extraction (fire-and-forget)
```

### 📦 WASM Sandbox Model

All LLM-generated code runs inside amla-sandbox:

- 🏗️ **QuickJS** JavaScript engine compiled to **WebAssembly** via **Wasmtime**
- 🔒 **Memory isolation** — sandbox cannot access host memory
- 🚫 **No ambient network** — all HTTP goes through approved tool calls
- ✅ **Capability-gated tools** — each tool call yields to Python for approval
- ⏱️ **Timeout + cancellation** — configurable limits with graceful shutdown

## 📚 Documentation

| File | Contents |
|------|----------|
| [docs/architecture.md](docs/architecture.md) | 🏗️ System architecture, data flow, concurrency model |
| [docs/features.md](docs/features.md) | ✨ All features with implementation details |
| [docs/modules.md](docs/modules.md) | 📦 Complete module list with descriptions |
| [docs/tech-stack.md](docs/tech-stack.md) | 🔧 Dependencies, versions, external services |
| [docs/configuration.md](docs/configuration.md) | ⚙️ Config schema, environment variables, workspace layout |
| [docs/sandbox.md](docs/sandbox.md) | 📦 WASM execution model, sandbox limitations |
| [docs/telegram-ux.md](docs/telegram-ux.md) | 💬 Telegram integration, UX flows |
| [docs/cron.md](docs/cron.md) | ⏰ Cron scheduling, job lifecycle |

## 🤝 Contributing

PRs welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for dev setup, testing, and code conventions. 🤗

```bash
# Dev setup
uv venv && uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -x -v

# Lint
ruff check . && ruff format --check .
```

## 📄 License

[MIT](./LICENSE) — ClamBot Contributors 2026

<p align="center">
  <sub>🐚 ClamBot is for educational, research, and technical exchange purposes.</sub>
</p>
