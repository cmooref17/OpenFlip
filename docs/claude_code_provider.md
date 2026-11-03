# claude_code provider

Runs an agent on the **real Claude Code binary** through the Claude Agent SDK
instead of imitating Claude Code over raw HTTP. Claude Code holds the login
(whatever `claude auth status` on the box reports); OpenFlip never reads,
stores or refreshes a credential.

## Enable

```jsonc
// agents/<id>/agent.json
{ "provider": "claude_code", "model": "anthropic/claude-opus-5-5" }
```

```jsonc
// config.json (all optional)
"claude_code": {
  "cli_path": "",               // a specific `claude` binary; default = the SDK's bundled one
  "idle_disconnect_s": 600,     // drop a conversation's Claude Code process after this much idle
  "max_buffer_size": 52428800,  // bytes for one CLI message (large tool results)
  "migrate_history_chars": 60000 // how much mirrored history to replay into a NEW session
},
"models": { "claude-opus-5-5": { "provider": "claude_code", "effort": "high" } }
```

Requirements: `claude-agent-sdk==0.2.177` (pinned in `requirements.txt`, bundles
Claude Code 2.1.317) and a logged-in `claude` for the user the gateway runs as.
Under launchd, if the Keychain is unreadable, put a `claude setup-token` value in
the LaunchAgent's env as `CLAUDE_CODE_OAUTH_TOKEN`.

Switching an agent back: set `provider` to `anthropic` and restart. Histories
are shared (same JSONL mirror); the Claude Code session is simply left behind.

## How it maps

| OpenFlip concept | claude_code provider |
|---|---|
| Conversation | one Claude Code session, resumed by id (`.meta.json` → `claude_code.session_id`); one `ClaudeSDKClient` per live conversation, disconnected when idle |
| System prompt | SOUL/AGENT/TOOLS + shared files + memory index, exactly as runtime assembles `conv.system_message`, sent as a custom prompt with `snapshot: false`; a change (hot reload, `/model`, `/effort`) reconnects with `resume` |
| Tools | the `@tool` registry served as an in-process MCP server `openflip` (`mcp__openflip__<name>`); every call runs through `tool_executor.execute_tool_calls` (ACL, lock, dry-run, posting, feedback) — Claude Code's built-in tools are off |
| Tool loop | Claude Code's; `chat()` returns the final text with `tool_calls=[]`, runtime's loop runs once per turn |
| History | `conv.messages` + JSONL are a **mirror** (user text, assistant rounds with `tool_calls`, tool results) for the dashboard, `/status`, `/undo` and migration; prompts = everything appended since the last turn (`[FRAMEWORK]` notes, nudges, `inject_context` plants) |
| Intermediate text | text that accompanied a tool call posts immediately (same as the in-loop intra-round post); the final text posts via runtime |
| Images | inbound attachments go in as image content blocks on the user message |
| `/compact` | sends Claude Code's own `/compact` before the next prompt; auto-compaction is Claude Code's (PreCompact hook flags it so the usual notice + `[FRAMEWORK]` note fire) |
| `/undo` | runtime calls `on_history_cut()`; the next turn starts a NEW session seeded with the surviving mirror (`migrate_history_chars`) |
| `/reset` | deletes mirror + meta, drops the session |
| `/stop` | task cancellation → `client.interrupt()` |
| `/model`, `/effort`, `/session set` | model, effort, memory, activity_status, error_display, error_message (context window / compaction trigger / max_tokens are Claude Code's) |
| Memory recall | selector = one-shot `query()` on Claude Code (`memory_recall.model`, default `sonnet`) |
| Usage | `/usage` ledger row per turn from the result's usage (+ `model_usage`, `total_cost_usd` in `raw_usage`); `/status` context = last request's input |
| Operator soft-inject | PreToolUse hook denies further tool calls once a message is queued (batch interrupt) |
| `web_search` tool | backend `anthropic`/`auto` runs as a one-shot Claude Code query using its built-in WebSearch (no direct Messages call from openflip); SearXNG backend unchanged |

Not available on this provider: `/uncompact`, sticky `tool_choice` forcing,
request dumps (`OPENFLIP_REQUEST_DUMP`; use `OTEL_LOG_RAW_API_BODIES=file:<dir>`
in the LaunchAgent env instead), the `claude_code` *tool* (redundant: the agent
is Claude Code).

## Spike results (2026-11-03, laptop, bundled CLI 2.1.317, haiku)

`tests/spike_claude_code.py` against a real session, output in
`tests/spike_claude_code.out.json` (git-ignored):

| Check | Result |
|---|---|
| connect | 0.8 s (resume 0.2 s); child process ~220 MB RSS |
| custom system prompt, no built-ins, MCP tool call | works; `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop` hooks fire; Stop `block` makes the model continue |
| PreToolUse deny with reason | tool result carries the reason, model reported BLOCKED |
| raw stream events | `message_start/content_block_*/message_delta/message_stop` delivered per API round |
| `get_context_usage()` | works (1M window reported on haiku) |
| `/compact` as a prompt | works, `compact_boundary` system message, ~9.5 s |
| resume by session id | memory intact, same id; `set_model("sonnet")` live switch works |
| image content block | works ("Red") |
| one-shot selector (`query()`, max_turns=1) | 2.4 s |

## The `anthropic` provider beside it

The old provider stays selectable and now has two auth modes per agent
(`anthropic_auth` in agent.json, or `integrations.anthropic.auth` in config.json):
`api_key` is a plain API client (x-api-key, openflip user agent, no Claude Code
headers or billing block); `oauth` is the historical path, unchanged. Memory
recall and `web_search` follow the same mode. See SETUP.md §10.

## Tests

* `tests/test_claude_code_provider.py` — fake SDK client; no process, no network.
* `tests/smoke_claude_code_live.py [model]` — real Claude Code session end to
  end (tool turn → resume → `/compact` → image → selector). Needs a login.
* `tests/spike_claude_code.py` — raw SDK feature spike; rerun after any SDK upgrade.

On a machine running inside a Claude Code desktop session, run the live scripts
with a clean environment (`env -i HOME=$HOME PATH=$PATH USER=$USER …`), otherwise
the subprocess inherits that session's `ANTHROPIC_BASE_URL`.
