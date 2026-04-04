"""Context builder — assembles system prompts for LLM calls.

Composes the system prompt from:
  - Agent identity and instructions
  - Workspace docs (``workspace/docs/*.md``)
  - Memory content (MEMORY.md)
  - Clam catalog
  - Tool schemas
  - Generation rules
  - Link context
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds system prompts for the agent's LLM calls.

    Assembles multiple context sources into a coherent system prompt
    with budget management for memory injection.
    """

    def __init__(
        self,
        workspace: Path | None = None,
        memory_budget_config: Any | None = None,
        model_context_size: int = 100_000,
    ) -> None:
        self._workspace = Path(workspace) if workspace else None
        self._memory_budget = memory_budget_config
        self._model_context_size = model_context_size

    def build_system_prompt(
        self,
        docs: str = "",
        memory: str = "",
        tools: list[dict[str, Any]] | None = None,
        clam_catalog: list[Any] | None = None,
        link_context: str = "",
        generation_mode: bool = True,
    ) -> str:
        """Build the complete system prompt.

        Args:
            docs: Concatenated workspace docs content.
            memory: MEMORY.md content.
            tools: Tool schema list from the registry.
            clam_catalog: List of ClamSummary objects.
            link_context: Pre-fetched link context.
            generation_mode: Whether to include generation rules.

        Returns:
            The assembled system prompt string.
        """
        sections: list[str] = []

        # Identity
        sections.append(self._build_identity())

        # Workspace docs
        if docs:
            sections.append(f"## Workspace Documentation\n\n{docs}")

        # Memory injection (with budget)
        memory_section = self._build_memory_section(memory)
        if memory_section:
            sections.append(memory_section)

        # Clam catalog
        catalog_section = self._build_catalog_section(clam_catalog)
        if catalog_section:
            sections.append(catalog_section)

        # Tool schemas
        tool_section = self._build_tool_section(tools)
        if tool_section:
            sections.append(tool_section)

        # Generation rules
        if generation_mode:
            sections.append(self._build_generation_rules())

        # Link context
        if link_context:
            sections.append(f"## Pre-fetched Link Context\n\n{link_context}")

        return "\n\n---\n\n".join(sections)

    def load_workspace_docs(self) -> str:
        """Load all docs from ``workspace/docs/*.md``."""
        if not self._workspace:
            return ""

        docs_dir = self._workspace / "docs"
        if not docs_dir.exists():
            return ""

        parts: list[str] = []
        for md_file in sorted(docs_dir.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                if content.strip():
                    parts.append(f"### {md_file.stem}\n\n{content}")
            except Exception:
                continue

        return "\n\n".join(parts)

    def _build_identity(self) -> str:
        """Build the agent identity section."""
        return (
            "## Identity\n\n"
            "You are ClamBot, an AI agent that fulfills user requests by generating "
            "and executing JavaScript code (called 'clams') in a secure WASM sandbox. "
            "You have access to built-in tools that clams can call.\n\n"
            "When a user makes a request:\n"
            "- If it's conversational (greeting, question, etc.), respond directly\n"
            "- If it requires action, generate a JavaScript clam to accomplish it\n"
            "- If a suitable clam already exists, reuse it\n\n"
            "### Uploaded Files\n\n"
            "Users may upload files via chat. Uploaded files are saved to the "
            "`upload/` directory inside the workspace. When the message contains a "
            "file upload annotation, read the file using the `fs` tool:\n"
            '`await fs({operation: "read", path: "upload/<filename>"})`\n'
            "Then process the file contents as requested by the user."
        )

    def _build_memory_section(self, memory: str) -> str:
        """Build the memory injection section with full budget enforcement.

        Applies all MemoryPromptBudgetConfig fields:
          - ``max_tokens``: hard cap on memory tokens
          - ``min_tokens``: skip injection if content is below this threshold
          - ``reserve_tokens``: deducted from the effective budget
          - ``max_context_ratio``: memory cannot exceed this fraction of the
            model's total context window

        Token estimation uses ``len(text) / 4`` as a rough approximation.
        """
        if not memory or not memory.strip():
            return ""

        # --- Resolve budget parameters ---
        max_tokens = 4000
        min_tokens = 0
        reserve_tokens = 0
        max_context_ratio = 1.0

        if self._memory_budget:
            _max_tokens = getattr(self._memory_budget, "max_tokens", max_tokens)
            if isinstance(_max_tokens, (int, float)):
                max_tokens = int(_max_tokens)

            _min_tokens = getattr(self._memory_budget, "min_tokens", min_tokens)
            if isinstance(_min_tokens, (int, float)):
                min_tokens = int(_min_tokens)

            _reserve_tokens = getattr(self._memory_budget, "reserve_tokens", reserve_tokens)
            if isinstance(_reserve_tokens, (int, float)):
                reserve_tokens = int(_reserve_tokens)

            _max_context_ratio = getattr(
                self._memory_budget, "max_context_ratio", max_context_ratio
            )
            if isinstance(_max_context_ratio, (int, float)):
                max_context_ratio = float(_max_context_ratio)

        # --- Calculate effective budget ---
        budget_from_config = max_tokens
        budget_from_ratio = int(max_context_ratio * self._model_context_size)
        effective_budget = min(budget_from_config, budget_from_ratio) - reserve_tokens
        effective_budget = max(effective_budget, 0)  # never negative

        # --- Skip injection if content is too small to be useful ---
        estimated_tokens = len(memory) / 4
        if estimated_tokens < min_tokens:
            return ""

        # --- Truncate if over effective budget ---
        max_chars = effective_budget * 4
        if len(memory) > max_chars:
            memory = memory[:max_chars] + "\n\n[Memory truncated due to budget]"

        return f"## Long-Term Memory\n\n{memory}"

    def _build_catalog_section(self, catalog: list[Any] | None) -> str:
        """Build the clam catalog section."""
        if not catalog:
            return ""

        lines = ["## Available Clams\n"]
        for clam in catalog:
            name = getattr(clam, "name", str(clam))
            desc = getattr(clam, "description", "")
            tools = getattr(clam, "declared_tools", [])
            tools_str = ", ".join(tools) if tools else "none"
            usage_count = getattr(clam, "usage_count", 0)
            usage = f", used {usage_count}x" if usage_count else ""
            lines.append(f"- **{name}**: {desc} (tools: {tools_str}{usage})")

        return "\n".join(lines)

    def _build_tool_section(self, tools: list[dict[str, Any]] | None) -> str:
        """Build the tool schemas section.

        Renders each tool's name, description, parameters, and — when
        provided — the return value schema so the LLM knows the response
        shape without guessing.
        """
        if not tools:
            return ""

        import json

        lines = ["## Available Tools\n"]
        for tool_schema in tools:
            func = tool_schema.get("function", tool_schema)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            returns = func.get("returns")

            lines.append(f"### {name}")
            lines.append(f"{desc}\n")
            lines.append(f"Parameters: ```json\n{json.dumps(params, indent=2)}\n```\n")
            if returns:
                lines.append(f"Returns: ```json\n{json.dumps(returns, indent=2)}\n```\n")

        return "\n".join(lines)

    @staticmethod
    def _build_generation_rules() -> str:
        """Build the clam generation rules block."""
        return """\
## Clam Generation Rules

When generating a clam, follow these rules strictly:

1. **JavaScript ONLY** — No shell scripts, Python, or other languages
2. **Object argument syntax** — Call tools with: `await tool({param: value})`
3. **Tool names use underscores** — e.g., `http_request`, not `http.request`
4. **Only declared tools** — Only call tools that are declared in the clam's tool list
5. **No modules, no packages, no npm** — The sandbox is a minimal QuickJS WASM runtime. \
There is NO `require()`, NO `import`, NO `fetch()`, NO Node.js APIs, and NO way to install \
npm packages. Do NOT attempt to use or reference any external libraries, modules, or packages. \
ALL functionality must come from plain JavaScript code and the built-in tools listed below.
6. **No direct network** — Use `http_request` or `web_fetch` tools instead
7. **File paths** — Use relative or absolute host paths; `/workspace/` prefix is FORBIDDEN
8. **COMPUTE, don't hardcode** — Scripts MUST compute results via code logic, NEVER hardcode \
pre-computed answers. The script runs in a sandbox, so it must actually perform the computation.
9. **DATA ONLY — no LLM tasks in code** — Clams must ONLY fetch/compute data. \
Do not use external APIs for translation, summarization, text analysis, or reformatting \
unless the user explicitly requests a specific service. These transformations are applied \
automatically in a post-processing step. Just return the raw data.
10. **Parameterize with `run(args)`** — Extract all user-supplied values into `"inputs"` and \
write an `async function run(args)` that destructures them. The runtime injects `args` \
automatically and calls `run(args)`. This makes clams reusable with different inputs.
11. **Return values** — Return a string (used as-is) or an object (serialized to JSON). \
When calling tools that return structured objects (like `web_fetch`, `http_request`), \
extract and return the meaningful content (e.g., `response.content`) rather than the \
full raw response object. The user wants the useful result, not JSON metadata.
12. **Reusability** — Set `metadata.reusable: true` and `metadata.source_request` for \
repeatable requests
13. **CANNOT access host OS** — The sandbox has NO access to the host operating system. \
It CANNOT run shell commands, manage processes, install software, open/close windows, \
interact with the desktop, manage system services, or use OS-specific APIs (Win32, \
PowerShell, bash, etc.). If the user asks for any of these, respond with: \
`{"script": "return 'I cannot perform OS-level operations (close windows, manage processes, install software). I can help with data processing, web requests, file operations, and scheduling.';", "declared_tools": [], "inputs": {}, "metadata": {"description": "Explains sandbox limitations", "reusable": false}}`

### Inputs and `run(args)` pattern

**Clams must be GENERIC.** When the request mentions specific values (strings, \
numbers, URLs, names, etc.), extract them into `"inputs"` and write the script to \
work with ANY values — not just the ones in the request. The specific values go \
into `"inputs"`, the script uses `args`.

Examples:

```
// Request: "reverse the string 'hello'"
// inputs: {"text": "hello"}
async function run(args) {
  return args.text.split("").reverse().join("");
}
```

```
// Request: "calculate 2+2"
// inputs: {"a": 2, "b": 2}
async function run(args) {
  return args.a + args.b;
}
```

```
// Request: "fetch the weather for London"
// inputs: {"city": "London"}
async function run(args) {
  const resp = await http_request({url: `https://wttr.in/${args.city}?format=j1`, method: "GET"});
  return resp.content;
}
```

```
// Request: "fetch this page https://example.com/"
// inputs: {"url": "https://example.com/"}
async function run(args) {
  const resp = await web_fetch({url: args.url});
  return resp.content;
}
```

```
// Request: "show ~/Download/test.txt"
// inputs: {"path": "~/Download/test.txt"}
async function run(args) {
  const content = await fs({operation: "read", path: args.path});
  return content;
}
```

```
// Request: "list files in ~/Documents"
// inputs: {"path": "~/Documents"}
async function run(args) {
  const listing = await fs({operation: "list", path: args.path});
  return listing;
}
```

```
// Request: "save 'hello world' to output.txt"
// inputs: {"path": "output.txt", "content": "hello world"}
async function run(args) {
  const result = await fs({operation: "write", path: args.path, content: args.content});
  return result;
}
```

```
// Request: "find largest dirs in /var"
// inputs: {"path": "/var"}
async function run(args) {
  const usage = await fs({operation: "disk_usage", path: args.path});
  return usage;
}
```

```
// Request: "read the uploaded PDF"
// inputs: {"path": "upload/report.pdf"}
async function run(args) {
  const result = await pdf_reader({path: args.path});
  return result.text;
}
```

```
// Request: "summarize pages 1-3 of document.pdf"
// inputs: {"path": "upload/document.pdf", "pages": "1-3"}
async function run(args) {
  const result = await pdf_reader({path: args.path, pages: args.pages});
  return result.text;
}
```

The runtime will prepend `const args = {"a": 2, "b": 2};` and append \
`return await run(args);` so the function is invoked automatically.

For requests with NO extractable values (e.g., "what time is it"), \
use inline code with `"inputs": {}`:

```
return new Date().toISOString();
```

Output format — respond with a JSON object:
```
{
  "script": "<javascript code>",
  "declared_tools": ["tool1", "tool2"],
  "inputs": {"key": "value"},
  "metadata": {
    "description": "<what this clam does>",
    "reusable": true/false,
    "source_request": "<original request if reusable>"
  }
}
```"""
