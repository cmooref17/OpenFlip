<!-- MANUAL.md — operator-facing reference for openflip.
     NOT auto-loaded into agent system prompts. Agents fetch this file
     on demand via read_file when the operator asks them to configure,
     diagnose, or explain how openflip works. Keep it accurate; the
     framework changes faster than memory does. -->

# openflip operator's manual

You are an agent on openflip, a multi-agent bot framework. The operator
relies on you to know how this framework works. This file is your
reference — fetch it (`read_file("agents/_shared/MANUAL.md")`) when you
need to look up a field, a tool, an ACL shape, a recovery recipe, or a
diagnostic move.

Behavioral rules live in `_shared/FRAMEWORK.md` (auto-loaded). This file
is reference, not rules.

---

# 1. Agents in one paragraph

An agent is a directory under `agents/<id>/` plus a Discord (or iMessage)
token. The framework reads `agent.json` to discover the agent, assembles
a system prompt from `system_files`, opens a transport, and routes
inbound messages through `pipeline.should_respond` → model → tool calls
→ reply. Conversations persist per-channel as JSONL. Memory persists
per-agent as Markdown plus an embedding index. Every agent runs as one
process inside the openflip systemd unit.

---

# 2. The agent directory

```
agents/<id>/
├── agent.json            # Required. Config, tools, ACLs, channels. Hot-reloads.
├── SOUL.md               # Character / persona. Auto-injected by default (first in system_files).
├── AGENT.md              # Per-agent extension of _shared/FRAMEWORK.md. Auto-created empty, auto-injected by default. Empty = no-op.
├── TOOLS.md              # Per-agent extension of _shared/TOOLS.md. Auto-created empty, auto-injected by default. Empty = no-op.
├── REMINDER.md           # End-of-payload nudge. Uncached. Paid every turn. Soft-warned >2000 chars.
├── HEARTBEAT.md          # (optional) Prompt loaded when a `heartbeat:true` cron job fires.
├── MEMORY.md             # Core memory. NOT auto-loaded — accessed via memory tools.
├── conversations/        # Per-channel <conv_id>.jsonl + <conv_id>.meta.json. Auto-managed.
├── memory/               # Daily logs (YYYY-MM-DD.md) + index.json (embeddings).
└── chain_state.json      # In-flight inter-agent chain IDs. Persisted across restarts.
```

Auto-load decisions are driven by `system_files` in `agent.json`. Files
not listed there don't enter the system prompt no matter what they
contain. The project-level `~/.openflip/CLAUDE.md` is *always*
injected first regardless of `system_files` (see `_load_system_files` in
`openflip/agent.py`).

Files under `agents/_shared/` (the SHARED tier — identical bytes for
every agent):

- `FRAMEWORK.md` — universal behavioral rules. Listed in every agent's
  `system_files`. Per-agent extension: each agent's own `AGENT.md`.
- `TOOLS.md` — universal tool hygiene. Same. Per-agent extension: each
  agent's own `TOOLS.md`.
- `MANUAL.md` — this file. NOT in `system_files`. Read on demand. No
  per-agent extension.

There is NO shared `AGENT.md` — `_shared/` holds FRAMEWORK.md, TOOLS.md,
MANUAL.md only. `AGENT.md` is per-agent by definition (the personal
counterpart to the shared FRAMEWORK.md).

Directories whose name starts with `_` (like `_shared/`) are not
discovered as agents.

## Personal vs shared files (the two-tier model)

Two system-prompt files have a shared base AND a per-agent extension:

- **Shared tier** — `_shared/FRAMEWORK.md` (rules) + `_shared/TOOLS.md`
  (tool hygiene). Same bytes for every agent.
- **Personal tier** — each agent's own `AGENT.md` (extends FRAMEWORK) +
  `TOOLS.md` (extends shared TOOLS). The personal file is ADDITIVE —
  it never replaces the shared one.

Both personal files are auto-created empty (0 bytes) on new-agent
bootstrap and auto-injected by default via `system_files`. An empty
(0-byte / whitespace-only) file contributes NOTHING to the prompt — no
blank section, no stray separator (the loader does `.read().strip()` and
appends only if truthy, in `_load_system_files`). To use one, just write
content to it; no `system_files` surgery needed.

Injection order (the default for every agent):

```
SOUL.md → _shared/FRAMEWORK.md → AGENT.md → _shared/TOOLS.md → TOOLS.md
```

`AGENT.md` lands right after `_shared/FRAMEWORK.md`; personal `TOOLS.md`
lands right after `_shared/TOOLS.md`.

---

# 3. `agent.json` — every field

Hot reload semantics: edits to `agent.json` and any file listed in
`system_files` are hash-detected on every turn (`Agent.reload_if_changed`).
The next message picks them up automatically. Code changes under
`openflip/` need a framework restart. The `/reload` slash command also
forces an explicit re-read.

## Identity

- `id` — stable string. Should match the directory name. Used in logs,
  in `talk_to_agent` routing, and as a key in `RUNNERS`.
- `display_name` — human label used in commands and panels. Defaults to
  `id`.

## Owner vs admin (config.json, not agent.json)

Two privilege tiers, both keyed by Discord user ID. These live in
`config.json` (machine-local, gitignored — a repo pull never overwrites
them), not in any agent.json.

- `owner_id` — single supreme operator. `integrations.discord.owner_id`
  canonical, legacy top-level `owner_id` honored. `is_owner(id)` gates the
  dangerous stuff: `run_command`, `restart_gateway`, `claude_code`,
  `/reset`, `/say`, `/restart`, `/stop`, and the other sensitive slash
  commands. Exactly one owner.
- `admin_ids` — list of elevated users. `integrations.discord.admin_ids`
  canonical, legacy top-level `admin_ids` honored. `is_admin(id)` returns
  true for the owner (always implicitly an admin) PLUS everyone in the
  list. Admins get: the ACL bypass (not blocked on normal tools), plus
  `/grant`, `/revoke`, `/reload`. Admins do NOT get the dangerous tier —
  that stays `is_owner`-only. Empty/missing list = admins are just the
  owner. Set IDs here to add admins:
  ```json
  "integrations": {"discord": {"owner_id": 139..., "admin_ids": [111, 222]}}
  ```
- **`is_admin` also gates `fetch_url`'s internal-address SSRF guard.**
  Owner/admins can `fetch_url` private/internal/loopback/link-local/
  metadata addresses; everyone else is refused those (public URLs stay
  open to all). The check is transport-aware — Discord matches the numeric
  speaker, handle-based transports (iMessage) match the handle — and is
  resolved inside the tool via the same Session-derived (transport,
  speaker_id, handle) the runtime ACL uses.

## Provider + model

- `provider` — `"anthropic"`, `"openai"`, or `"ollama"`. Default
  `"ollama"`. Picks which conversation class wraps the agent (routing
  lives in `openflip/providers.py`): `AnthropicConversation` for direct
  API calls through the operator's Claude OAuth, `OpenAIConversation`
  for direct API-key calls to OpenAI's Chat Completions endpoint, or
  `DiscordConversation` (Ollama-backed) for local models. All three are
  NATIVE providers — the agent's own turns run on that API.
- `model` — provider-specific model name. **For Anthropic, the working
  format is `anthropic/<model-id>[-1m]`** — note the `anthropic/` provider
  prefix (stripped by `_normalize_model` before the API call) and the
  optional `-1m` suffix. Current model IDs: `claude-opus-4-8` (newest,
  released 2026-05-28, 1M-context-capable), `claude-opus-4-7`,
  `claude-sonnet-4-6`. Append `-1m` to opt into the 1M-token context beta
  (`context-1m-2025-08-07`) — this is encoded as a name suffix in the
  field but sent as an `anthropic-beta` header, not a different model id.
  So a 1M Opus 4.8 agent reads exactly:
  `"model": "anthropic/claude-opus-4-8-1m"`. **For OpenAI, the same
  convention: `openai/<model-id>`** (prefix stripped before the API
  call; a bare model id also works). Any Chat-Completions model id is
  valid — gpt-series (`gpt-5.1`, `gpt-4o`), o-series reasoning models
  (`o4-mini`), codex models (`gpt-5.1-codex`) — whatever the configured
  API key can access. There is no hardcoded list; declare each model you
  use in `config.json`'s `models.<id>` block so its real
  `context_window` is known (unlisted openai models default to a
  conservative 128k). An empty `model` field on an openai agent falls
  back to `integrations.openai.default_model` (default `gpt-5.1`).
  For Ollama: a bare tag like
  `qwen3.5:cloud` (no provider prefix). Changing the value takes effect on
  the next `/reload` or restart — agent.json's bytes are part of the
  reload fingerprint, so the conversation rebuilds against the new model
  without a full restart.
- `think` — Anthropic only. `true`/`false`/`null` toggles the
  extended-thinking budget. Default `null` = framework picks based on
  model. Ignored by the openai provider (use the per-model `effort`
  knob below for reasoning models).
- Reasoning **`effort`** is NOT an agent.json field — it's a model
  capability, so it lives per-model in `config.json` under
  `models.<id>.effort`, right next to `compaction_trigger`. A
  per-conversation override can be set with `/effort` (owner-only,
  Anthropic-only), which wins over the model config. See "What's
  the current model / context window?" below.

### The "openai" provider — setup + what's supported

Auth is a plain API key — no OAuth, no token refresh. Configure it in
`config.json` (or set the standard `OPENAI_API_KEY` environment variable
as a fallback):

```json
"integrations": {
  "openai": {
    "api_key": "sk-...",
    "base_url": "https://api.openai.com",   // optional; for proxies/gateways
    "default_model": "gpt-5.1"              // optional; used when agent.model is empty
  }
}
```

Then in the agent's `agent.json`: `"provider": "openai"`,
`"model": "openai/gpt-5.1"`. Declare the model in `config.json`'s
`models` block (`{"gpt-5.1": {"provider": "openai", "context_window":
400000}}`) so the picker lists it and the context window is right.

Supported (same UX as an anthropic agent): streaming, full tool calling
(every framework tool works unchanged), JSONL conversation persistence +
`/reset`, image attachments (vision via base64 data URLs), REMINDER.md
injection, `/status` (context + cache-read stats), usage-ledger
recording, malformed-tool-call retry, mid-turn soft-inject.

NOT supported (deliberate differences — these commands answer
"Anthropic-only"):
- **No server-side compaction** → `/compact` and `/uncompact` don't
  apply. Context is bounded by a local pre-flight trim EVERY turn
  (oldest messages dropped once the estimated input nears the window),
  like the ollama provider. Nothing is summarized — trimmed history
  stays on disk but leaves the model's context.
- **No `/effort` session override** — only the per-model
  `models.<id>.effort` config knob (see below; openai vocabulary is
  `minimal`/`low`/`medium`/`high`).
- **No explicit prompt-cache control** — OpenAI caches prompt prefixes
  automatically; cache reads show up in `/status` and the ledger as
  `cache_read` tokens, and `cache_creation` is always 0.
- **No `-1m`-style beta suffixes** and no `think` toggle.

## System prompt

- `system_files` — ordered list. Concatenated with `\n\n`. Entries
  starting with `_shared/` resolve under `agents/_shared/`; bare names
  resolve relative to the agent's directory. Template variables
  `{agent_id}`, `{agent_dir}`, `{display_name}` are substituted at read
  time. Missing files are silently skipped, and empty (0-byte /
  whitespace-only) files contribute nothing — no blank section, no stray
  separator. New agents bootstrap with the five-file default below;
  `AGENT.md` and the personal `TOOLS.md` are auto-created empty and
  included by default (empty = no-op until you fill them). See "Personal
  vs shared files" in §2.

  ```json
  "system_files": ["SOUL.md", "_shared/FRAMEWORK.md", "AGENT.md", "_shared/TOOLS.md", "TOOLS.md"]
  ```

## Channels

- `channels.respond_in` — `"all"` (every channel the bot can see),
  `"mentions_only"` (only when @mentioned), `"channels_only"` (none
  unless overridden by `always_respond_channel_ids`). Default
  `"mentions_only"`. DMs always pass through regardless of this field
  (subject to `dm_allowlist_user_ids`).
- `channels.ignore_channel_ids` — list[int]. Hard ignore. Beats
  everything else, including `always_respond_channel_ids`.
- `channels.always_respond_channel_ids` — list[int]. Always answer in
  these channels, regardless of `respond_in`.
- `channels.dm_allowlist_user_ids` — list[int]. When populated, only
  these users get DM responses (the bot owner is always allowed
  implicitly). Empty/missing = anyone can DM (legacy behavior).
- `respond_to_bots` — bool, default `false`. Set `true` to let one agent
  respond to messages authored by another bot (used for multi-bot
  Discord testing).

## Behavior

- `tool_response_mode` — only one effective value is checked at
  runtime: `"media_only"`, which is ALSO the default when the field is
  unset (`Agent.from_dict` falls back to `"media_only"`; the global
  default in `main.py` is the same). In this mode your assistant text is
  suppressed on turns that produced attachments (the attachment speaks
  for itself, no redundant "here's the image" caption). To make text AND
  attachments both post, you must set it to some OTHER explicit string
  (e.g. `"caption"`, `"text"`) — any non-`"media_only"` value disables
  the suppression. Leaving it out does NOT do that; unset == suppressed.
  Captions on attachments are handled in `tool_executor.py` and aren't
  gated by this field.
- `memory_enabled` — bool, default `true`. Toggles whether the five
  memory tools are auto-injected regardless of `allowed_tools`. Set
  `false` to fully strip an agent of memory (rare — usually you want
  this on).

## Tools

- `allowed_tools` — list of ACL objects. See section 5 for the full
  shape. Tools not listed here are auto-injected by the framework with
  empty `auth` (denied by default; owner sees them anyway via the
  Discord owner-bypass; broadens via `/grant`).

## Path ACLs

These govern `read_file`, `write_file`, `edit_file`, `delete_file`,
`list_files`.

- `allowed_read_paths` — directories the agent may read. Default: the
  agent's own directory only. Use `["*"]` for unrestricted.
- `allowed_write_paths` — same shape. Default: agent's own directory.
- `denied_paths` — always blocked. Wins over allow lists, checked first,
  applies to everyone. Flat list only (no per-user form). Use for
  secret/config paths the agent should never touch.

### Per-user path scope (opt-in)

`allowed_read_paths` / `allowed_write_paths` each accept EITHER the flat
list above OR a dict. A flat list applies to everyone and is unchanged —
existing agents are byte-for-byte identical, so you only get the new
behavior by opting in to the dict form. The dict form is **TRANSPORT-KEYED,
structurally identical to a tool's `auth` block** (see "Tool auth model"
below): the outer keys are transports (`discord`, `imessage`, …), and
inside each transport block you use the SAME `users` / `all_users`
vocabulary as a tool ACL. There is ONE way to express "who gets what"
across the whole project. It lets ONE agent give the owner full scope and
a coworker a narrow sandbox on the SAME bot, with per-transport identities:

```json
"allowed_read_paths": {
  "discord": {
    "users":     { "100000000000000000": ["*"],
                   "207001234567890123": ["/work/sandbox", "/work/shared"] },
    "all_users": ["/work/sandbox"]
  },
  "imessage": {
    "users":     { "coworker@example.com": ["/work/sandbox"] },
    "all_users": []
  }
}
```

Resolution, per request, against the current speaker:

1. **`<transport>`** — the current speaker's transport block is selected
   first, exactly like a tool ACL's `auth.<transport>`. No block for this
   transport → `[]` (no access beyond the default-deny / read fallback
   below). This is the structural fix: the outer key is the transport, just
   like tool `auth`.
2. **`<transport>.users.<id>`** — within that block, used if the speaker's
   id matches a key. Discord: the numeric user id as a string
   (`"100000000000000000"`). iMessage: the handle, lowercased/trimmed
   (`"coworker@example.com"`). Same match semantics as a tool ACL's `users`
   dimension.
3. **`<transport>.all_users`** — else, everyone in that transport not
   matched above (the baseline).
4. A missing branch (no transport block, no matching user, no `all_users`)
   resolves to `[]`, which falls into the normal default-deny logic (empty
   write list = deny; empty read list = the agent-dir + `/tmp` read
   fallback). Never fail-open.

**The owner is NOT special here.** There is no magic `owner` key — to give
the owner a particular scope, list the owner's plain id under the relevant
transport's `users`, exactly as you would in a tool ACL (tool owner-bypass
was removed; path ACLs mirror that). An owner whose id is not under `users`
falls to `all_users` like anyone else.

Paths differ from tool ACLs in only one necessary way: a tool ACL is a
yes/no predicate, but a path ACL must return a SET of dirs per audience —
so a path `users` is a dict-of-lists rather than a flat list, and there is
no `exclude` / roles / channels dimension in v1 (deny is `denied_paths`,
and simply not granting a `users` / `all_users` entry already withholds
access). The transport-keying and the `users` / `all_users` keys are
identical to tool auth.

> **TYPE TRAP — same name, different type, do NOT cross.** The keys are the
> same as tool auth but the *types* are not. In a **tool** ACL `all_users` is
> a **BOOL** (`true`/`false`) and `users` is a **flat list** of ids. In a
> **path** ACL `all_users` is a **LIST OF PATHS** and `users` is a
> **DICT** of id→paths. Pasting the wrong schema's shape (e.g. `all_users:
> true` into a path block, or `all_users: ["/dir"]` into a tool block) is
> treated as misconfiguration and fails **closed** with a logged warning —
> the path side never crashes on `list(true)`, and the tool side never
> silently grants the tool to everyone. Match the type to the schema.

`"*"`, the realpath/separator boundary, and `denied_paths` (still flat,
still checked first, still wins) behave exactly as for flat lists — a
per-user `"*"` allows-all for THAT user only. The web editor now accepts
the dict form too (parity with `agent.json`): it validates list-or-dict for
the two path fields and its `"*"` widen-guard walks the dict's path lists,
so a `"*"` buried inside `users`/`all_users` is still caught. `denied_paths`
stays list-only in both editors.

## Slash commands

- `agent_specific_commands` — list of slash-command names that exist
  ONLY on this agent's bot. Used to opt into commands gated by name in
  `commands.py` (e.g. one agent gets `/some_special_cmd` while siblings
  don't). Empty list (default) = standard command set only.

## Transport

- `transport` — `"discord"` (default) or `"imessage"`. Selects which
  Transport class `AgentRunner` instantiates. Per-transport config lives
  in `config.json` under `integrations.<transport>.agents.<id>`. iMessage
  agents need `handle`, optional `imsg_path`, optional `allowlist_chats`
  there. macOS-only.

## Ollama-only

- `ollama_options` (alias: `options`) — passed to Ollama's chat API.
  `temperature`, `num_ctx`, `num_predict`, `repeat_penalty`, etc.
  Ignored when `provider == "anthropic"` or `"openai"`. `num_predict: -1`
  is rejected by cloud Ollama — keep it positive.

## Example minimal agent.json

```json
{
  "id": "myagent",
  "display_name": "myagent",
  "provider": "anthropic",
  "model": "anthropic/claude-opus-4-8-1m",
  "system_files": ["SOUL.md", "_shared/FRAMEWORK.md", "AGENT.md", "_shared/TOOLS.md", "TOOLS.md"],
  "channels": {"respond_in": "mentions_only"},
  "tool_response_mode": "media_only",
  "allowed_tools": [
    {"name": "save_memory", "auth": {"discord": {"all_users": true}}},
    {"name": "web_search",  "auth": {"discord": {"users": [100000000000000000]}}}
  ]
}
```

## Complete field reference (every configurable agent.json key)

Quick-scan menu of every operator-settable `agent.json` key, its type,
and its as-coded default (traced in `openflip/agent.py`). The prose
subsections above explain the non-obvious ones; this table is the index.

Privilege keys `owner_id` and `admin_ids` are NOT agent.json fields —
they live in `config.json`. See "Owner vs admin (config.json, not
agent.json)" above.

Dotted keys (`channels.*`, `dream.*`, `proactive.*`) are sub-keys of a
nested object. `respond_to_bots`, `memory_enabled`, etc. are top-level.

| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | string | directory name | Stable id; should match the dir. Used in logs, routing, `RUNNERS`. |
| `display_name` | string | `id` | Human label in commands/panels. |
| `provider` | string | `"ollama"` | `"anthropic"`, `"openai"`, or `"ollama"`. Picks the conversation class. |
| `model` | string | `""` | Provider-specific tag. Anthropic: `anthropic/<id>[-1m]` (e.g. `anthropic/claude-opus-4-8-1m`); the `anthropic/` prefix is required and the `-1m` suffix flips on the 1M-context beta header. Ollama: bare tag, no prefix. |
| `think` | bool \| null | `null` | Anthropic only. Toggles extended-thinking budget; `null` = framework decides. |
| `system_files` | list[string] | `[]` | Ordered; concatenated with `\n\n`. `_shared/` prefix resolves to shared dir. New agents bootstrap with the 5-file default (SOUL → FRAMEWORK → AGENT.md → shared TOOLS → personal TOOLS.md); `AGENT.md` + personal `TOOLS.md` auto-created and injected by default, empty = no-op. |
| `ollama_options` (alias `options`) | object | `{}` | Passed to Ollama chat API. Ignored when `provider=="anthropic"` or `"openai"`. |
| `allowed_tools` | list[object] | `[]` | ACL entries (see §5). Unlisted tools auto-injected with blank (deny) auth. |
| `tool_response_mode` | string | `"media_only"` | `"media_only"` suppresses text on attachment turns; any other value disables that. |
| `respond_to_bots` | bool | `false` | Top-level. `true` lets the agent reply to other bots. |
| `memory_enabled` | bool | `true` | Auto-injects the memory tools regardless of `allowed_tools`. |
| `channels.respond_in` | string | `"mentions_only"` | `"all"` \| `"mentions_only"` \| `"channels_only"`. DMs bypass this. |
| `channels.ignore_channel_ids` | list[int] | `[]` | Hard ignore; beats `always_respond_channel_ids`. |
| `channels.always_respond_channel_ids` | list[int] | `[]` | Always answer here regardless of `respond_in`. |
| `channels.dm_allowlist_user_ids` | list[int] | `[]` | When populated, only these users (+ owner) get DM replies. |
| `channels.guild_whitelist` | list[int] | `[]` | When populated, only these guild IDs are answered. DMs unaffected. |
| `channels.respond_guilds` | list[int] | `[]` | Nine-list routing: respond-if-mentioned (guild tier). |
| `channels.respond_channels` | list[int] | `[]` | Nine-list routing: respond-if-mentioned (channel tier). |
| `channels.respond_categories` | list[int] | `[]` | Nine-list routing: respond-if-mentioned (category tier). |
| `channels.respond_no_mention_guilds` | list[int] | `[]` | Nine-list: respond always, no mention needed (guild tier). |
| `channels.respond_no_mention_channels` | list[int] | `[]` | Nine-list: respond always (channel tier). |
| `channels.respond_no_mention_categories` | list[int] | `[]` | Nine-list: respond always (category tier). |
| `channels.ignore_guilds` | list[int] | `[]` | Nine-list: hard deny (guild tier). Highest precedence. |
| `channels.ignore_channels` | list[int] | `[]` | Nine-list: hard deny (channel tier). |
| `channels.ignore_categories` | list[int] | `[]` | Nine-list: hard deny (category tier). |
| `dream.enabled` | bool | `false` | When `true`, auto-dream fires (live — see §10). Manual `/dream` works regardless. |
| `dream.min_idle_minutes` | int | `120` | Idle threshold for auto-dream — the channel must have been quiet at least this long before the operator's next turn. |
| `dream.max_memory_chars` | int | `25000` | Cap on memory text fed to a dream pass. |
| `proactive.enabled` | bool | `false` | KAIROS. When `true`, `main._sync_kairos_jobs()` creates the tick cron job (live — see §10). |
| `proactive.interval_minutes` | int | `30` | Tick interval (must be > 0). |
| `proactive.quiet_hours` | object \| null | `null` | e.g. `{"start":"23:00","end":"08:00","timezone":"US/Mountain"}`. |
| `proactive.channel_id` | int | `0` | Channel the tick anchors / posts to. |
| `allowed_read_paths` | list[string] OR dict | agent's own dir | Dirs the agent may read. `["*"]` = unrestricted. Dict form is transport-keyed (`{"discord": {"users", "all_users"}}`), mirroring tool `auth` = per-user scope — see "Per-user path scope". |
| `allowed_write_paths` | list[string] OR dict | agent's own dir | Dirs the agent may write. `["*"]` = unrestricted. Dict form = transport-keyed per-user scope. |
| `denied_paths` | list[string] | `[]` | Always blocked; wins over allow lists; checked first; flat list only (no per-user form). |
| `agent_specific_commands` | list[string] | `[]` | Slash commands registered only on this agent's bot. |
| `transport` | string | `"discord"` | `"discord"` or `"imessage"`. Single-transport selector. |
| `transports` | list[string] | `[]` | Multi-transport list (e.g. `["discord","imessage"]`). Empty = use `transport`. |

---

# 4. Tool inventory

Every tool currently registered in the framework. Args listed are the
AI-facing signature only; owner-locked parameters (model names, step
counts, dimensions) live in `/toolset` and `data/tool_settings.json`
and are not exposed to the model. All tools are async.

ACL gating: every tool is checked against the speaker's `allowed_tools`
entry before invocation. The owner ID is always allowed on Discord (see
`_check_acl` in `openflip/acl.py`). Tools also tagged
`silent_to_discord` send their text to the model only, not the channel.

## Memory

- **`save_memory(text: str)`** — append `text` to today's daily log
  (`agents/<id>/memory/YYYY-MM-DD.md`) with a timestamp and update the
  embedding index. Auto-injected when `memory_enabled: true`.
- **`update_core_memory(content: str)`** — overwrite `MEMORY.md` with
  `content`. Read the file first; you are responsible for preserving
  what should stay.
- **`search_memory(query: str)`** — cosine-similarity search across
  `MEMORY.md` + daily logs using the embedding model in `config.json`.
  Top-5 results above a 0.3 threshold.
- **`read_memory(file: str = "")`** — empty arg reads `MEMORY.md`. A
  date (`"2026-05-21"`) reads that day's log.
- **`list_memory_files()`** — list daily logs with dates and sizes.

## Files (path-ACL gated)

- **`read_file(path: str)`** — read a text file. Path can be absolute
  or relative to the agent's directory. Subject to `allowed_read_paths`
  + `denied_paths`.
- **`write_file(path: str, content: str)`** — CREATE-ONLY. Refuses to
  overwrite an existing file. Max 8000 bytes. Atomic via tmp+rename.
  Subject to `allowed_write_paths` + `denied_paths`.
- **`edit_file(path: str, old_string: str, new_string: str)`** —
  replace exactly one occurrence of `old_string`. Match must be
  byte-for-byte; whitespace counts. Atomic. After every edit, grep the
  file to confirm — `edit_file` can silently miss when the old_string
  drifts.
- **`list_files(path: str = ".")`** — list directory contents.
- **`delete_file(path: str)`** — delete a file. Snapshots before
  deletion (recoverable via `restore_snapshot`).

## Web

- **`web_search(query: str)`** — search via local SearXNG. Silent to
  Discord (result text goes only to the model).
- **`fetch_url(url: str)`** — fetch a URL. HTML auto-stripped to
  readable text. ~100KB cap. Browser headers. Silent to Discord.
  **SSRF-guarded:** private/internal/link-local/loopback/reserved and
  cloud-metadata (169.254.169.254) addresses are refused for
  non-owner/non-admin callers. The host is DNS-resolved and EVERY
  resolved address is checked (defeats DNS-rebinding-to-localhost), and
  the check re-runs on each redirect hop (a public URL can't 302 you to
  an internal IP). Owner/admins retain full access, so the owner can still
  fetch localhost services (flipflaskapp, etc.). Public URLs are
  unchanged for everyone.
- **`download_url_to_tmp` (internal helper, not a model tool)** — fetches
  model/user-controlled URLs for edit/upscale/animate/audio_separate. Enforces
  the SAME internal-host SSRF guard as `fetch_url` (private/internal/loopback/
  link-local/metadata refused) but UNCONDITIONALLY — no owner bypass, since it
  has no caller identity. Streams with a 50 MB size cap (no unbounded read) and
  a 120s timeout. ComfyUI's own calls go through `comfy_host()`, not here, so
  refusing localhost is safe. The ComfyUI HTTP calls have per-call timeouts:
  submit/upload/view 120s; the `/history` poll 30s per-request inside the
  bounded 600s loop.

## System

- **`run_command(command: str, timeout: int = 30)`** — `/bin/sh -c
  <command>`. Timeout 1–120s, default 30. Output capped at 50k chars.
  Silent to Discord. ACL-gated tightly; usually owner-only.
- **`restart_gateway(reason: str, continuation: str = "", force:
  bool = False)`** — restart the openflip systemd unit. Preflight
  checks: no peer agent mid-turn, no queued inbound, no human-spoken
  message in the last 5s, no syntax errors in `openflip/`, all
  `agent.json` files valid. Pass `force=True` ONLY when the operator
  asked for it. The optional `continuation` fires as a synthetic turn
  after restart so you can resume what you were doing. Persists
  in-flight conversation state first.

## Discord

- **`send_message(text: str, channel_id: int = 0)`** — post to the
  current channel (default) or a different channel if `channel_id` is
  set. Auto-splits >1900 chars. Routes through Transport.send.
- **`delete_message(message_id: int = 0, channel_id: int = 0,
  with_attachments: bool = False)`** — delete a Discord message by ID,
  or find-and-delete the bot's most recent message if both IDs are 0.
  Cross-channel deletion is owner-only.
- **`fetch_discord_message(url: str)`** — fetch a Discord message URL
  or a Discord CDN image URL. Returns author/content/attachment URLs.
  Image attachments are downloaded and queued for vision on the next
  model iteration.

## Multi-agent

- **`talk_to_agent(agent_id: str, message: str, channel_id: int = 0,
  session_id: str = "")`**
  — fire a synthetic turn at another running agent. Fire-and-forget.
  Framed as `"<your_id>: <message>"` to the recipient.
  **`session_id` is the canonical target conversation key** — pass the
  transport-prefixed id (e.g. `"discord:12345"`, `"imessage:you@example.com"`,
  `"internal:email-support"`) and it is used DIRECTLY: all the
  return-channel guessing is bypassed. `channel_id` is the
  deprecated-but-supported bare-int fallback, used only when `session_id`
  is empty. When BOTH are omitted, the return channel auto-resolves via:
  (1) caller's current channel if recipient can access, (2) recipient's DM
  with the originating human, (3) caller's channel. Chain depth cap: 20.
  Generates a fresh chain_id per dispatch; a reply on a superseded chain_id is delivered with a `[FRAMEWORK]` late-reply prefix (it used to be silently dropped).
- **`end_chain()`** — explicitly end a chain-terminator turn without
  dispatching anywhere. Use when silence is correct (recipient agent
  decides nothing further needs to go back to the originator).
- **`inject_context(agent_id: str, channel_id: int = 0, text: str = "",
  session_id: str = "")`** —
  silently plant context into ANOTHER agent's conversation history for a
  specific conversation. **`session_id` is the canonical conversation
  key** — pass the transport-prefixed id (e.g. `"discord:12345"`,
  `"imessage:you@example.com"`, `"internal:email-support"`) and it is used DIRECTLY as
  the conversation key (no int() coercion, no prefix guessing).
  `channel_id` is the deprecated-but-supported bare-int fallback, used only
  when `session_id` is empty (the transport prefix is then inferred from
  the target runner). The text lands as a user-role message tagged
  `[INJECTED CONTEXT]: <text>` so the target's NEXT turn in that
  conversation sees it as background — it is NOT posted to Discord and does
  NOT trigger a turn now. Use to pre-load a fact/reminder into a peer
  before its next reply. Routes through the live in-memory conversation
  when the target runner is active (so it's visible on the very next turn),
  falls back to a JSONL append when the target isn't running (picked up on
  its next load). MID-TURN CAVEAT: do NOT inject into an agent that is
  actively mid-reply (between a tool_use and its tool_result) — it can
  corrupt the in-flight message sequence. Inject when the target is idle
  between turns. Owner-only by default (auto-injected blank ACL → owner
  bypass). A `/inject_context` slash command does the same thing for the
  operator.
- **Known later-pass item:** the `send_message` / `send_file` /
  `delete_message` trio still takes a bare-int `channel_id` only and has
  NOT yet been migrated to the canonical `session_id` arg. Until that pass
  lands, those three remain `channel_id`-keyed.

## Cron

- **`add_cron_job(name, prompt, cron="", every_seconds=0,
  mode="reminder", timezone="", channel_id=0, session_id="",
  tool_grants=None)`** —
  schedule a recurring
  job for yourself. Either `cron` (expression like `"0 9 * * 5"`) OR
  `every_seconds` — never both. Modes: `"reminder"` (final text auto-
  posts to the anchor conversation), `"data_collection"` (silent — you
  can still call `send_message`), `"mixed"` (silent, reserved).
  **Anchor:** `session_id` (transport-prefixed, e.g. `"imessage:you@example.com"` or
  `"discord:12345"`) is PREFERRED — it pins the fired turn to the exact
  conversation. A bare `channel_id` is ambiguous on agents that run more
  than one transport (iMessage + Discord), so the turn can land in the
  WRONG conversation; pass the prefixed `session_id` to avoid that. When
  you omit `session_id`, it defaults to the current conversation's
  prefixed id. `channel_id` defaults to the current channel and is still
  stored for back-compat; you need one of the two at fire time or the
  job is skipped with a warning (even in `data_collection`).
  Timezone is IANA (`"US/Eastern"`).
  Returns the job_id. Job creator is stamped on the job — the
  synthetic turn fires as the creator's user, not as owner.
  **`tool_grants`** (optional list of tool names) authorizes this job's
  synthetic turn to call those tools REGARDLESS of per-user auth — it is
  stored on the job as `toolGrants`. Cron turns have no human speaker
  (`speaker_id=0`), so a tool gated to specific human users would
  normally fail; grant it here instead. See "Per-session tool grants"
  under the auth section below for the exact semantics and limits.
- **`list_cron_jobs(agent_id="", include_all_agents=False)`** —
  defaults to your own jobs.
- **`cancel_cron_job(job_id: str)`** — delete the job. No archive.

Jobs live in `cron/jobs.json` (project root, not per-agent). Scheduler
ticks every 5s. `lastRunMs` persists across restarts so jobs don't
re-fire on boot.

## Snapshots

- **`list_snapshots(path: str)`** — newest-first list of saved
  snapshots for a file with timestamps and sizes.
- **`restore_snapshot(path, timestamp="", index=-1)`** — restore by
  exact timestamp or by 0-based index (0 = newest). Current state is
  snapshotted before the restore so you can undo.

## Code delegation

- **`claude_code(task: str, timeout: int = 600)`** — delegate a task
  to a Claude Code subprocess (`--print --dangerously-skip-permissions`)
  rooted at the openflip repo. Timeout 30–1800s, default 600. Useful
  for "go read 30 files and report" tasks that would burn your own
  context. Owner-only via ACL by default.

## Media (optional extras — only present if symlinked from openflip-extras)

- **`generate_image(prompt, model=None, lora=None, lora_strength=1.0,
  negative_prompt=None)`** — txt2img via ComfyUI. Owner-locked params:
  width, height, steps, cfg, sampler, batch_size, default model.
- **`edit_image(image_url, instruction)`** — img2img via
  Qwen-Image-Edit-Plus + optional 4-step Lightning LoRA.
- **`upscale_image(image_url)`** — 4× ESRGAN upscale. Bound-down before
  upscale to stay within VRAM.
- **`generate_video(prompt)`** — txt2video via Wan 2.2 T2V.
- **`animate_image(image_url, prompt)`** — img2video via Wan 2.2 I2V.
  Output dims snap to multiples of 16 within configured pixel budget.
- **`extract_audio_track(audio_url, target)`** — Demucs htdemucs_ft;
  keeps the named stem (`vocals`, `drums`, `bass`, `other`). Rejects
  audio over 600s.
- **`remove_audio_track(audio_url, target)`** — Demucs; same model;
  removes the named stem (instrumental / karaoke).
- **`generate_tts(model_id, text)`** — TTS via Gradio. `model_id` is
  `creator-modelname`; profile loaded from `data/tts_models/<id>/profile.pt`.
  Optional `language` hint via toolset.

---

# 5. The ACL shape (`allowed_tools`)

Each entry is an object with `name`, `auth`, and optional
`visibility_when_denied`. Bare strings are rejected at load time — the
framework hard-fails with a pointer at the offending entry.

```json
{
  "name": "send_message",
  "auth": {
    "discord": {
      "all_users": false,
      "users": [100000000000000000],
      "roles": [],
      "channels": [],
      "exclude": {"users": [], "roles": [], "channels": []}
    },
    "imessage": {
      "all_users": false,
      "users": ["+15551234567"],
      "exclude": {"users": []}
    }
  }
}
```

## Rules

1. **`auth.<transport>` missing → blocked on that transport.** No
   `auth.imessage` block = no iMessage caller can use the tool. `auth:
   {}` blocks everywhere.
2. **`all_users: true`** grants every speaker on that transport.
   Subject to `exclude` and to `roles`/`channels` if set. Short-circuits
   the `users` list.
3. **AND within one auth block.** `users`, `roles`, `channels` are
   AND-ed. Empty list / missing field = no restriction on that
   dimension. An auth block with no dimensions at all (no `all_users`,
   no users/roles/channels) is a *deny*, not a silent allow — same
   reason bare-string entries were removed.
4. **OR across same-name entries.** Multiple entries with the same
   `name` are independent rules; any one passing makes the tool
   callable.
5. **`exclude` always wins.** Matching anything in
   `exclude.{users,roles,channels}` blocks regardless of inclusions.
6. **Admin bypass on Discord.** Any admin (the `owner_id` plus everyone
   in `admin_ids`) passes every ACL on the Discord transport. iMessage has
   no admin bypass (no canonical handle mapping yet). This means an
   auto-injected blank entry surfaces the tool for all admins. NOTE: the
   ACL bypass is `is_admin`, but the genuinely dangerous capabilities
   (`run_command`, `restart_gateway`, `claude_code`, `/reset`, `/say`,
   `/restart`, `/stop`) are gated on `is_owner` separately and are NOT
   widened by admin status — see "Owner vs admin" in §3.
7. **`roles` and `channels` are Discord-only.** Belong inside
   `auth.discord` only.
8. **User-ID types are transport-native.** Discord = int IDs.
   iMessage = handle strings (`"+15551234567"`, `"name@example.com"`).
   The evaluator compares against the speaker's transport-native
   identity directly.

### Per-session tool grants (`tool_grants` / `toolGrants`)

Everything above is per-user/per-transport auth and is unchanged. On top
of it there is one additive allow-path for **sessions that have no human
speaker** — synthetic cron turns, which run with `speaker_id=0` and would
fail any human-gated ACL.

- **Where it lives.** The `Session` carries a `tool_grants: list[str]`.
  For cron jobs it comes from the job's `toolGrants` field, set via
  `add_cron_job(..., tool_grants=[...])`. The scheduler builds the
  synthetic session with those grants; the turn's tool evaluation honors
  them.
- **What it does.** A tool whose name is in `tool_grants` is callable for
  that session even if **no** auth block matches the (absent) speaker. It
  is OR-ed onto the normal check — `callable = human_ACL_passes OR
  name_in_tool_grants`.
- **What it does NOT do.** It is purely additive. It can NEVER override
  an `exclude.{users,roles,channels}` deny, and it can NEVER authorize a
  tool that isn't already present in the agent's `allowed_tools` — the
  evaluator only iterates names the agent was configured with, so a grant
  for an unknown/never-given tool is silently ignored (it can't conjure a
  tool). It confers tool-call authorization ONLY — never owner/admin.
- **Humans are unaffected.** `tool_grants` only ever appears on synthetic
  sessions. Human turns carry an empty list, so their access is exactly
  what the auth blocks above say.

**Worked example.** Schedule a daily email-summary job that may read mail
and send a message even though no human auth block grants those tools:

```
add_cron_job(
  name="morning email summary",
  prompt="Summarize my unread email and DM me the digest.",
  cron="0 7 * * *",
  mode="data_collection",
  tool_grants=["read_email", "send_message"],
)
```

At fire time the synthetic turn (speaker_id=0) can call `read_email` and
`send_message` — because both names are in `tool_grants` AND both are
present in the agent's `allowed_tools`. A human messaging the same agent
is still gated by the normal `auth.<transport>` blocks for those tools;
nothing about their access changed. If `read_email` were NOT in the
agent's `allowed_tools`, listing it in `tool_grants` would do nothing.

## Common shapes

```jsonc
// Open to every Discord user:
{"name": "generate_image", "auth": {"discord": {"all_users": true}}}

// Owner-only on Discord (redundant — owner bypass — but explicit):
{"name": "edit_file", "auth": {"discord": {"users": [100000000000000000]}}}

// Two Discord users plus a role:
{"name": "web_search", "auth": {"discord": {"users": [111, 222], "roles": [333]}}}

// Discord all-users EXCEPT one griefer:
{"name": "generate_video", "auth": {"discord": {"all_users": true, "exclude": {"users": [999]}}}}

// Channel-locked:
{"name": "delete_message", "auth": {"discord": {"channels": [123, 456], "all_users": true}}}

// Cross-transport — Discord owner + iMessage handle:
{"name": "send_message", "auth": {
  "discord":  {"users": [100000000000000000]},
  "imessage": {"users": ["+15551234567"]}
}}
```

## Optional fields

- **`visibility_when_denied`** — `"hidden"` (default) or `"known"`.
  `"known"` surfaces the tool to the model with a note "exists but you
  can't use it" so you can decline politely instead of pretending the
  tool doesn't exist.

## Editing via slash commands

- `/grant <tool> @user` — adds `@user` to `auth.discord.users`. Creates
  the entry if missing.
- `/revoke <tool> @user` — removes.
- `/agents` — interactive panel; toggling a tool on adds it with
  Discord owner-only auth by default. Broaden with `/grant`.

## Auto-injection

Every tool registered in `TOOL_REGISTRY` gets a blank-auth entry
appended at agent-load if missing from `allowed_tools`. The blank entry
denies everyone on every transport — but the Discord owner bypass means
the operator still sees and can call every framework tool without
listing it. New tools shipped in the framework appear automatically in
`/agents` and `/grant` autocomplete without per-agent config changes.

When the agent saves its config, only entries with a non-empty `auth`
are serialized. Blank auto-injected entries don't pollute on-disk
`agent.json`.

---

# 6. Memory

Two tiers, both per-agent (other agents cannot read yours).

## Core memory — `MEMORY.md`

One file. Lasting facts: preferences, decisions, anchors, things that
are TRUE across days. Not auto-loaded into the prompt — you read it
with `read_memory()`. Updated via `update_core_memory(content)` — pass
the full new contents; you are responsible for preserving prior facts.

Promote a fact to core when: it's mentioned more than once, the
operator states a lasting preference, or future-you would need it after
a context wipe.

## Daily logs — `memory/YYYY-MM-DD.md`

Append-only. Today's events. `save_memory(text)` adds a timestamped
line; the framework auto-creates today's file if missing. Search across
both tiers with `search_memory(query)`. List dates with
`list_memory_files()`.

Memory is FACTS. Behavioral rules go in `SOUL.md` / `AGENT.md` /
`REMINDER.md` / `_shared/FRAMEWORK.md`; tool-specific notes go in your
personal `TOOLS.md` (extends `_shared/TOOLS.md`).

## Embeddings

`memory/index.json` stores one vector per memory file plus per-line
vectors for daily logs. The embedding model is configured in
`config.json` as `embedding_model` (default `nomic-embed-text` via
Ollama's `/api/embed`). Cosine similarity, top-5, threshold 0.3.

---

# 7. Channels & transports

`pipeline.should_respond` is the single decision point. There are TWO
routing systems and which one governs depends on whether ANY nine-list
field is set — read this whole section before touching an agent's
channel config.

**THE LOAD-BEARING GOTCHA:** the moment an agent sets ANY ONE of the
nine routing lists (`respond_guilds`, `respond_channels`,
`respond_categories`, `respond_no_mention_guilds`,
`respond_no_mention_channels`, `respond_no_mention_categories`,
`ignore_guilds`, `ignore_channels`, `ignore_categories`), the agent
flips off the legacy `respond_in` path entirely and becomes an
**explicit allowlist** — anything not matched by a tier is IGNORED
(tier h returns False). `respond_in` is then dead except as the
all-lists-empty fallback. So adding one channel to `respond_channels`
silently changes behavior everywhere else in that agent's reach.

Evaluation order (from `pipeline.should_respond`, current):

1. Sender is the bot itself → ignore.
2. Sender is a bot AND `respond_to_bots:false` → ignore.
3. `guild_whitelist` set AND guild not in it → ignore (legacy hard
   pre-gate, composes with the tiers below).
4. **Tier (c) IGNORE** — channel/guild/category in any `ignore_*` list
   OR legacy `ignore_channel_ids` → ignore. Wins over everything.
5. **Tier (d) NO-MENTION** — channel/guild/category in any
   `respond_no_mention_*` list OR legacy `always_respond_channel_ids`
   → respond regardless of mention.
6. **Tier (e) RESPOND-IF-MENTIONED** — channel/guild/category in
   `respond_guilds` / `respond_channels` / `respond_categories`
   → respond **only if @mentioned**.
7. **Tier (f) DM path** — DMs have no guild/category so they skip c-e.
   If `dm_allowlist_user_ids` is populated, only the owner + listed
   users pass; otherwise anyone can DM.
8. **Tier (g) LEGACY FALLBACK** — reached only when NO nine-list field
   is set (`any_new_routing` False): `respond_in == "all"` → respond;
   `"mentions_only"` → respond iff @mentioned; `"channels_only"` →
   ignore (channel must be in `always_respond_channel_ids` to override,
   which already returned True in tier d).
9. **Tier (h)** — new-style config set but nothing matched → ignore.

Tier precedence is the key mental model: a guild in `respond_guilds`
makes the WHOLE guild respond-if-mentioned; a single channel of that
guild in `respond_no_mention_channels` upgrades just that channel to
no-mention; a channel in any `ignore_*` list hard-kills it. Mention-only
scoping to one channel = put that channel in `respond_channels`. "Whole
server when mentioned" = put the guild in `respond_guilds` (you do NOT
also need the channel listed).

## Silence sentinel (`STAY_SILENT`)

An agent woken on every channel message (e.g. `channels_only` +
`always_respond_channel_ids`) is forced to run a turn for messages that
aren't for it. To let it genuinely say nothing, the runtime recognizes a
silence sentinel: if the agent's final reply text, stripped, equals
**exactly** `STAY_SILENT` (case-sensitive, nothing else), `runtime._run_turn`
blanks `final_text` so the normal "nothing to post" path runs and NOTHING
reaches the channel — the literal token is never posted. Defined as
`STAY_SILENT_SENTINEL` in `openflip/runtime.py`; detection sits right where
`final_text` is first resolved (~line 2435) so every downstream post-site
(inter-agent route, silent-drop branches, main auto-post) sees the already-
empty value. The terminal-result contract is also told the empty output was
intentional, so it won't fire its "no visible reply" warning. Suppression is
**exact-match only**: a reply that merely contains the word inside a sentence
posts normally. A suppressed turn logs `STAY_SILENT: suppressing channel post
for agent=<id>` and still saves the assistant turn (the bare token) to the
JSONL history, so the agent remembers it chose silence. Agent-facing guidance
lives in `_shared/FRAMEWORK.md` ("Group channels — know when to speak").

## Transports

- **Discord** (default). `nextcord.Bot` per agent. Token from
  `api_config.json` (gitignored) keyed by agent id.
- **iMessage** (macOS-only, experimental). Transport class at `openflip/transports/imessage.py`. `_handle_inbound` is wired in `runtime.py:411` — real implementation, not a stub. Per-transport config in `config.json` under `integrations.imessage.agents.<id>` — keys: `handle`, `imsg_path` (default `~/.local/bin/imsg`), `allowlist_chats` (list of chat rowids), `respond_in` (inherits from agent.json). Requires Full Disk Access on chat.db and Automation permissions for Messages.app. Not currently in active use. Note: the imessage.py module docstring still has a stale "INTEGRATION GAP" note from before the wiring landed — ignore it.

Multi-transport per agent is not currently supported — `transport:`
selects one. iMessage agents need Full Disk Access (chat.db) and
Automation permissions for Messages.app on macOS.

---

# 8. Conversations on disk

```
agents/<id>/conversations/
├── <conversation_id>.jsonl         # Append-only message log. Source of truth.
├── <conversation_id>.meta.json     # Anthropic-only sidecar (compaction block, last_usage).
├── <conversation_id>.jsonl.pre_reset_<ts>.bak.jsonl     # Made by /reset.
├── <conversation_id>.jsonl.compaction_<ts>.bak.jsonl    # Made when compaction fires.
└── <conversation_id>.jsonl.pre_uncompact_<ts>.bak.jsonl # Made by /uncompact.
```

`<conversation_id>` is `discord:<channel_id>`, `imessage:<handle>` (1:1) /
`imessage:<chat_id>` (group), or `linked:<canonical>` for identity-linked
conversations (see below).

## Cross-transport identity links (`identity_links`)

One person talking to the same agent on Discord AND iMessage normally gets
two separate conversation histories (`discord:<dm_channel>` and
`imessage:<handle>`). The top-level `identity_links` map in `config.json`
merges them into ONE shared history:

```json
"identity_links": {
  "discord:139243578504249344": "flip",
  "imessage:+15551234567": "flip"
}
```

- **Keys** are `<transport>:<native_id>` — the Discord **user id** (not a
  channel id) or the raw iMessage **handle** (phone/email). Keys are
  case-folded on load, so handle casing doesn't matter.
- **Values** are an arbitrary canonical string. Every identity mapped to the
  same value shares one conversation.

**How it works.** At session construction, if the speaker's
`<transport>:<id>` is in the map, the session's `conversation_id` is
rewritten to `linked:<canonical>` — so history lives in
`conversations/linked:<canonical>.jsonl` and the runtime's in-memory
conversation state (live conversation object, active-turn slot, soft-inject
buffer) is keyed by that same string on every transport. Both transports see
each other's turns live, no restart needed. Turns from the two transports
serialize against each other like two messages in one channel (they ARE one
conversation).

**1:1 only.** The rewrite applies to DMs / 1:1 chats. Guild channels and
iMessage group chats are shared spaces keyed by channel — they are never
rewritten, even when a linked person speaks in them.

**Unlinked users are untouched.** Anyone not in the map behaves exactly as
before — same conversation ids, same files, same in-memory keying.

**Routing vs auth — the security line.** A link rewrites conversation
ROUTING only (which history a session resolves to). It NEVER confers
privilege across transports:

- Replies post to the transport the message arrived on. A linked person
  writing on Discord gets the answer on Discord; writing on iMessage gets it
  on iMessage. The link never redirects outbound traffic.
- ACL/owner/admin evaluation is recomputed every turn from the session's
  native transport + identity (Discord user id, iMessage handle) and never
  consults `identity_links`. Being owner on Discord does NOT make the linked
  iMessage handle owner — grant each transport identity its own privileges
  explicitly (`integrations.<transport>.owner_id` / `admin_ids` /
  per-tool `auth` blocks). The iMessage `allowlist_senders` gate also still
  applies independently.

**Adding a link.**
1. Add both `<transport>:<id>` entries to `identity_links` in `config.json`
   pointing at the same canonical value.
2. Restart the gateway (config.json is read at startup).
3. New messages from either transport now land in
   `linked:<canonical>.jsonl`. Existing per-transport histories are NOT
   auto-merged — they stay on disk under their old names. To carry one
   forward, stop the gateway and rename/concatenate the old `.jsonl` into
   `linked:<canonical>.jsonl` first.

**Tools and commands.** `/reset`, `/compact`, `/uncompact`, `/status`,
`/stop` and the text-command mirrors operate on the linked conversation when
fired from a linked DM on either transport (one `/reset` wipes the shared
history). `inject_context` / `talk_to_agent` accept
`session_id: "linked:<canonical>"` to target the shared history directly;
for `talk_to_agent` the visible reply still needs a real channel to land on
(inter-agent auto-route handles that), since a linked conversation spans two
native channels and has no single posting target of its own.

## JSONL shape

One JSON object per line. Roles: `user`, `assistant`, `tool`.

```jsonl
{"role": "user", "content": "...", "ts": 1714934422.5}
{"role": "assistant", "content": "...", "ts": 1714934423.0}
{"role": "tool", "content": "...", "ts": 1714934423.5}
```

The runtime appends new messages each turn (no full-file rewrites).
History stays full on disk even as the in-memory list trims defensively
for context-window protection.

## /reset

Owner-only slash command. Backs up the live `.jsonl` to
`.pre_reset_<ts>.bak.jsonl`, then deletes both `.jsonl` and
`.meta.json`. Retention: 5 backups per channel; older pruned on every
`/reset`. To restore, copy a `.pre_reset_*.bak.jsonl` over the live
file and `pop` the in-memory conversation by restarting or by
`/uncompact` (which pops the cache).

## /compact and /uncompact (Anthropic only)

`/compact` sets `conv.force_compact_next = True` and fires a synthetic
turn so Anthropic emits a fresh compaction block on the next request.
On success: `compaction_block` lands in `.meta.json`, the live `.jsonl`
is archived to `.compaction_<ts>.bak.jsonl`, and the live `.jsonl` is
rewritten with only the post-compaction tail.

`/uncompact` finds the most recent `.compaction_<ts>.bak.jsonl`, copies
the current live state to `.pre_uncompact_<ts>.bak.jsonl` (in case the
operator changes their mind again), restores the backup over the live
file, strips `compaction_block` from `.meta.json`, and pops the
in-memory conversation so the next message reloads from disk.

Ollama and OpenAI have no equivalent (neither has server-side
compaction). Both commands return "Anthropic-only" if the provider
doesn't match.

## Context-window protection

`_trim_to_fit_window` in `anthropic_conversation.py` only fires
defensively — if the assembled request would otherwise exceed the
model's window. Anthropic's server-side auto-compaction handles
overflow in normal operation.

The openai provider has no server-side compaction, so its
`_trim_to_fit_window` (in `openai_conversation.py`) runs pre-flight on
EVERY turn, like the ollama provider — oldest messages drop from the
model's context once the estimated input nears the window. Full history
stays on disk regardless.

---

# 9. Mid-turn interrupts

## Turn dispatch (per-session concurrency)

Every turn — real Discord/iMessage messages and synthetic turns (cron,
`talk_to_agent`, restart continuation) — flows through one inbound queue
drained by a single worker. The worker does NOT run them one-at-a-time
globally: it dispatches each item into its own task and immediately pulls
the next, so:

- **Different channels/sessions run CONCURRENTLY.** A turn in DM A no
  longer waits behind an in-flight turn in channel B; person A and person
  B are served in parallel. (Previously all turns on an agent serialized
  behind a single in-flight turn.)
- **The same channel stays SERIALIZED.** `_active_turns[channel_id]`
  marks the in-flight turn for a channel. A second inbound for a channel
  that already has a live turn never runs concurrently with it — it
  either soft-injects (default) or hard-interrupts (`/stop`), exactly as
  below. Same-session serialization is required because both turns would
  otherwise race the one shared conversation object + its `.jsonl` file.

A global cap (8 concurrent turns per agent) backstops a burst of many
distinct channels; per-channel serialization already bounds normal load
to the number of distinct active channels.

**Slot reconciliation is worker-owned.** Each dispatched turn runs under a
supervisor task that serializes behind any prior same-channel turn still
winding down. When that task ends, the `_active_turns[channel_id]` slot is
reconciled — re-pointed at a still-live predecessor or popped — by a
done-callback the worker attaches, NOT by code inside the turn task. This
matters for a rapid double-`/stop`: a supervisor can be cancelled *before
it ever starts running*, which skips any `finally` it might carry, so
reconciliation can't live there or it would silently leak and let the next
same-channel turn run concurrently with a still-cleaning-up predecessor
(torn history / Anthropic 400). The worker also re-resolves the slot
synchronously at dispatch time, so same-channel serialization holds even
in that cancel-before-start window. Net effect for you: same-session turns
NEVER overlap, even under fast repeated interrupts — you can `/stop`
aggressively without corrupting conversation history.

**Convention — fire-and-forget tasks must surface their exceptions.** A
bare `asyncio.create_task(coro)` on a long-lived coroutine swallows any
exception the coroutine raises (it only reappears as an unretrieved-task
warning at GC, which never reaches `log.txt`) — the "bot looks dead with
no error" failure mode. Every long-lived `create_task` (the cron
scheduler, the restart-sentinel processor, auto-route / soft-inject
follow-up dispatches) attaches `utils.log_task_exception` as a
done-callback and is given a descriptive `name=` so a death logs a named
traceback instead of vanishing. The cron scheduler additionally wraps its
per-tick body in a supervising try/except so one bad tick logs and the
loop continues (it lets `CancelledError` propagate so shutdown still
works). If you add a long-lived background task, attach the callback.

Two modes for a same-channel mid-turn message: soft (default) and hard.

## Soft inject (default)

When the operator messages while you're mid-turn, the framework appends
the message to `_pending_inject[channel_id]`. At the next tool-result
boundary (or at end of turn if no tool fires), the buffer drains as
user-role messages with this prefix:

```
[FRAMEWORK]: The operator sent this message while you were mid-turn.
You MUST address it in your very next reply — before any tool-result
confirmation, before continuing what you were doing, before anything
else. ...

Operator's message: <text>
```

You address it FIRST in your next reply, then resume what you were
doing (or pivot if the message redirects you).

Your reply to a drained operator message is ALWAYS made visible — even
in `media_only` mode on a turn that also produced image/video
attachments. Normally `media_only` suppresses plain text on an
attachment turn (so you don't narrate every pic), but that suppression
is overridden when a human soft-inject drained during the turn: the
operator spoke, so your answer to them must reach the channel. A plain
attachment turn with no drained message still suppresses chatter text as
before.

### Mid-batch checkpoint

When you fire multiple tool calls in one block (e.g. 4 `generate_image`
calls), the executor checks for pending operator messages between each
call. If the operator messaged mid-batch, the remaining calls are
skipped — they are NOT executed. The last completed tool's result
carries a `[BATCH INTERRUPTED]` note explaining how many calls were
skipped. You must acknowledge the interruption and address the
operator's message before deciding whether to re-issue the skipped
calls.

## Hard interrupt

Triggered by:
- Operator typing a message starting with `/stop` in the channel.
- Operator firing the `/stop` slash command.

Effect: cancels the active `_run_turn` task (`task.cancel()`), wipes
`_pending_inject[channel_id]`, and drops the queued soft-injects. The
`/stop` message itself becomes the next inbound. If it includes a
follow-up instruction (`/stop actually do X instead`), that fires as a
fresh turn.

`_active_turns[channel_id]` tracks the in-flight task so the cancel can
find it.

---

# 10. Cron jobs

State file: `cron/jobs.json` at the project root (not per-agent).

```json
{
  "version": 1,
  "jobs": [
    {
      "id": "<uuid>",
      "agentId": "<agent>",
      "name": "Weekly report",
      "enabled": true,
      "mode": "reminder",
      "schedule": {"kind": "cron", "expression": "0 9 * * 5", "timezone": "US/Eastern"},
      "payload": {"message": "Time for the weekly summary..."},
      "sessionId": "discord:1234567890",
      "channelId": 1234567890,
      "createdBySpeakerId": 100000000000000000,
      "toolGrants": ["read_email", "send_message"],
      "lastRunMs": 0
    }
  ]
}
```

`sessionId` (transport-prefixed: `"discord:<id>"` / `"imessage:<chat_id>"`)
is the preferred anchor and is written automatically on new jobs.
`channelId` is the legacy bare-int form, kept for back-compat. See the
anchor note under Modes for why the prefix matters on multi-transport
agents.

`toolGrants` (optional list of tool names) authorizes this job's synthetic
turn (`speaker_id=0`) to call those tools regardless of per-user auth. Set
it via `add_cron_job(..., tool_grants=[...])`; omitted when empty. It is an
additive allow-path only — it never overrides an `exclude` deny and can't
authorize a tool absent from the agent's `allowed_tools`. See "Per-session
tool grants" under the auth Rules above.

## Schedule kinds

- `"interval"` — fires every `seconds` seconds. Needs `seconds > 0`.
- `"cron"` — standard cron expression. Needs `expression`. Optional
  `timezone` (IANA name, defaults to UTC). Backed by `croniter`.

Validation runs at job-add and at scheduler-start. Bad expressions
surface immediately; the scheduler still starts but bad jobs won't
fire.

## Modes

- `"reminder"` — fires a synthetic turn; final reply text auto-posts to
  `channelId`.
- `"data_collection"` — fires silent. No auto-post. You can still call
  `send_message` explicitly inside the turn.
- `"mixed"` — reserved for accumulate-then-summarize pattern. Behaves
  like `data_collection` for now.

Every job needs an anchor — even silent jobs use it to attach the
synthetic turn to a conversation file. Prefer `sessionId`, the
transport-prefixed id (`"imessage:you@example.com"`, `"discord:12345"`): it routes the
turn to the exact conversation. The legacy bare `channelId` still works,
but on an agent with more than one transport a bare id is ambiguous —
the scheduler can't tell `discord:1` from `imessage:1` and falls back to
the first transport (logging a WARNING that names the job), which may
misroute. Set `sessionId` to be safe. A job with neither is skipped at
fire time with a warning.

## Heartbeats

A legacy alternate form: set `payload.heartbeat: true` and the
scheduler reads the prompt from `agents/<id>/HEARTBEAT.md` instead of
`payload.message`. Empty/missing file = job skipped with a warning.
HEARTBEAT.md is not auto-loaded into the normal system message — it's
only the prompt for heartbeat-mode cron jobs.

## KAIROS (proactive ticks)

STATUS: live and proven end-to-end (2026-06-02). A real proactive tick
fired, woke the agent, and it acted + send_message'd the operator. Quiet
hours, kill switch, and per-agent enable all confirmed working.

KAIROS is a proactive cron job MODE (rides the existing cron scheduler,
not a parallel loop). On fire the agent gets a synthetic
`<tick>HH:MM Weekday</tick>` turn prepended to its HEARTBEAT.md prompt,
looks around (logs, repo, TODO, pending work) and DECIDES whether to
act or sleep. Doing nothing is valid and expected — the prompt forbids
inventing busywork.

- Always fires `auto_post_final_text=False` + `silent=True`. Agent must
  call `send_message` explicitly to surface anything.
  `originator_visibility="kairos"`.
- The "should I act?" judgment lives in HEARTBEAT.md, NOT framework code.

### Enabling (per-agent, default OFF)

In `agent.json`:
```
"proactive": {
  "enabled": true,
  "interval_minutes": 30,
  "quiet_hours": {"start":"23:00","end":"08:00","timezone":"US/Mountain"},
  "channel_id": <id>
}
```
No `proactive` block (or `enabled:false`) = no kairos job; agent behaves
exactly as before. `main._sync_kairos_jobs()` creates/updates the job on
startup for agents with `proactive.enabled=true`, removes a stale one
when disabled. Default interval 30 min.

### Cost gates (checked in `_due()` BEFORE any model call)

- Quiet hours — not due during the window. No tick, no cost.
- Global kill switch — env `OPENFLIP_DISABLE_KAIROS=1` disables all
  kairos across all agents.
- Per-job `enabled` — still respected.

### Other behavior

- Idle-tick pruning — a tick with NO tool calls and NO send_message has
  its tick message + empty reply pruned from in-memory conversation so
  idle ticks don't inflate cost.
- Stop-hook exemption — `originator_visibility="kairos"` is exempt from
  `promise_without_action`. Prompt also instructs emitting NO text when
  idle.
- Retry-heuristic location — the four in-loop turn-retry heuristics
  (action-promise, peer-prose, empty-reply, and the `promise_without_action`
  stop-hook invocation) were extracted out of `runtime._run_turn` into
  `openflip/turn_retries.py` (audit §4 item 1, 2026-06-07). The decision
  logic — phrase lists, peer-prose line scan, nudge text, env kill switches
  (`OPENFLIP_DISABLE_ACTION_PROMISE_RETRY` / `_PEER_PROSE_RETRY` /
  `_EMPTY_RETRY`), and the `stop_hooks.evaluate_stop_hooks` wrapper — lives
  there now; `runtime.py` still owns the one-shot flags, the `conv.messages`
  nudge append, the sticky `tool_choice=any` override, and the `continue`.
  Pure code motion, no behavior change.
- No self-retrigger — kairos send_message posts as the bot;
  `should_respond()` filters the agent's own id. Must not reset another
  agent's cooldown.

## DREAM (auto memory consolidation)

STATUS: live. Auto-fire is wired (`openflip/dream_autofire.py` +
end-of-turn hook in `runtime._run_turn`). Unlike kairos, a dream is
INVISIBLE — it rewrites `MEMORY.md` in the background and NEVER messages
the operator (kairos pinged and got killed; dream does not).

DREAM is NOT a cron job. The check is event-driven: it runs at the end of
a cleanly-completed, top-level, operator-driven turn. Synthetic /
subagent / chain turns are excluded, so the dream's own (synthetic) turn
can never recursively trigger another dream. When all gates pass it fires
a `/dream`-equivalent SILENT synthetic turn — reusing the existing
`dream()` tool + 4-phase consolidation prompt — which calls
`update_core_memory()` and stops. Manual `/dream` is unchanged and
ignores `dream.enabled`.

### Enabling (per-agent, default OFF)

In `agent.json`:
```
"dream": {
  "enabled": true,
  "min_idle_minutes": 120,
  "max_memory_chars": 25000
}
```
No `dream` block (or `enabled:false`) = no auto-dream; the agent behaves
exactly as before and only manual `/dream` consolidates. The block is
hot-reloaded with the rest of `agent.json` — no restart needed to flip it
on. `max_memory_chars` is the same cap the consolidation prompt already
uses.

### Gates (checked cheapest-first, bail on first failure)

1. `dream.enabled` is true.
2. IDLE — the gap PRECEDING this turn ≥ `min_idle_minutes`. Measured
   per-channel as time since the previous operator turn, so a dream fires
   when the operator returns after an idle stretch (the only boundary an
   event-driven check can see — during true idleness no turns fire).
   Never dreams mid-conversation.
3. Scan throttle — once past the cheap gates, skip the filesystem stats if
   we scanned this channel < ~10 min ago.
4. COOLDOWN — last dream ≥ 24h ago (mtime of `memory/.dream_marker`).
5. NEW MATERIAL — at least one `memory/YYYY-MM-DD.md` daily log is newer
   than the marker. No new memory = nothing to consolidate = skip.
6. LOCK — atomically acquire `memory/.dream.lock` (O_CREAT|O_EXCL). If a
   pass already holds it, bail.

### Race + lock (the load-bearing bit)

Two turns ending near-simultaneously must NOT both fire a dream. The lock
and the marker are SEPARATE files on purpose: the marker persists across
days (so it can't double as an O_EXCL lock — an existing file always
fails O_EXCL), and the lock is the true mutex. `os.open(O_CREAT|O_EXCL)`
is atomic — of N racers exactly one creates the lock; the rest get
`FileExistsError` and bail without firing. The winner touches the marker
to `now()` INSIDE the locked section, so the 24h cooldown gate blocks
every later turn even after the lock is released. The lock is held only
for the brief enqueue + marker-touch (the consolidation runs async, NOT
under the lock) and is removed in a `finally` so a crash can't wedge it;
a lock older than 30 min is treated as a crashed pass and stolen.

### Visibility

Always fires `auto_post_final_text=False` + `silent=True`,
`originator_visibility="dream"`. Nothing reaches Discord; the only
artifact is the rewritten `MEMORY.md`. The fire is logged
(`dream_autofire` event) for audit.

## Speaker attribution (security)

Each job records `createdBySpeakerId` at creation time. When the job
fires, the synthetic turn runs as that user. Legacy jobs without the
field fall back to owner (back-compat); operator can delete and
recreate them for strict attribution.

### Attribution vs privilege (synthetic turns)

`run_synthetic_turn` keeps two distinct notions of "who" separate:

- **Attribution** (whose name history/logs/return-routing read as):
  the passed `speaker_id` if non-zero, else `owner_id` as a safe
  fallback. Framework-originated turns (cron / heartbeat /
  restart-sentinel) pass no speaker, so they attribute to the owner —
  this is fine, it just keeps history readable.
- **Privilege** (`owner=True` → the full owner toolset): granted ONLY
  when an **explicitly-passed** `speaker_id` equals `owner_id`. It is
  NOT derived from the attribution fallback.

Why the split: a framework turn passes `speaker_id=0`, which the
attribution fallback resolves to `owner_id`. If privilege were derived
from that, every cron/heartbeat/restart turn would silently get the
owner toolset and bypass `Session.tool_grants` (the additive mechanism
built for exactly these trusted-but-not-owner turns). So:

- **`speaker_id=0`** (cron/heartbeat/restart_sentinel) → `owner=False`;
  any tools a framework turn needs come from `Session.tool_grants` (see
  §10 cron `toolGrants`), never from owner privilege.
- **explicit `speaker_id == owner_id`** (real owner-initiated synthetic
  turns — `/dream`, `/compact`, `/stop` all thread
  `interaction.user.id`) → `owner=True`, full owner toolset, as before.

The restart-sentinel continuation runs as `owner=False`; it posts via
`auto_post_final_text` (no owner-gated tool needed) and threads the
originator's handle so handle-based ACLs still resolve.

## Synthetic turn semantics

Cron firing does NOT auto-post anything in `data_collection`/`mixed`
mode — if you want the operator to see your work, call `send_message`
yourself inside the turn. `reminder` mode auto-posts the final text;
intermediate tool output still flows through `tool_response_mode`.

---

# 11. Multi-agent

## Inter-agent communication

`talk_to_agent(agent_id, message, channel_id=0)` — fire-and-forget.
The recipient sees `"<your_id>: <message>"` as a synthetic user turn.

## Visibility

- **Operator-initiated chains** (chain root = a human in a channel):
  the recipient's final reply auto-routes back to the operator's
  channel as if you said it.
- **Agent-initiated chains** (chain root = cron, heartbeat, or another
  silent context): chains run silent unless the agent explicitly calls
  `send_message`. Chain root is tagged via `originator_visibility`.

## Chain mechanics

- **Depth cap.** Hard limit at 20 hops. `talk_to_agent` refuses at the
  cap.
- **Chain IDs.** Each `talk_to_agent` dispatch generates a fresh UUID
  stored at `_current_chain_to[recipient_id]` and persisted to
  `chain_state.json`. Replies on stale chain_ids drop silently.
- **End-of-chain.** `end_chain()` is the explicit "nothing further to
  return" terminator. Otherwise: a normal text reply from the recipient
  routes back through the chain.

## Discovering peers

`list_files("agents/")` — every non-underscore subdir is a peer agent
id. Each has its own `agent.json`, `SOUL.md`, conversation, memory.

## Isolation

You cannot read another agent's:
- `MEMORY.md` (path ACL).
- Conversation `.jsonl` files (path ACL).
- `agent.json` (allowed_read_paths default to your own dir).

The exception: `_shared/` files are readable by every agent that lists
them.

## Robustness (timeouts)

- **Auto-route return-channel resolution is timeout-bounded.** When a
  recipient's reply auto-routes back to the operator, the originator's
  bot resolves a DM channel via `fetch_user` + `create_dm`. Both calls
  are wrapped in `asyncio.wait_for(..., timeout=10.0)` — a hung Discord
  API can no longer stall the post-turn loop; on timeout the auto-route
  post is skipped (logged) rather than hanging.
- **Ollama panel calls are timeout-bounded.** The `/models` and `/model`
  panels' `ollama_list` / `ollama_unload` calls cap at 30s; a model
  **pull** caps at 1800s (30 min) since it downloads weights. On timeout
  the panel surfaces a short error to the interaction instead of leaving
  a spinning, never-resolving Discord View callback.

---

# 12. Hot reload vs restart

## Auto-reload (no restart)

`Agent.reload_if_changed()` runs on every turn. Hash-based fingerprint
over `agent.json` + every file in `system_files` + project `CLAUDE.md`.
If any byte differs, the agent reloads and the next turn picks up:

- `agent.json` field changes (tools, channels, model… everything).
- `SOUL.md` / `AGENT.md` / personal `TOOLS.md` / `REMINDER.md` edits.
- `_shared/FRAMEWORK.md` and `_shared/TOOLS.md` edits.
- New files added to `system_files`.

The `/reload` slash command forces it explicitly.

## Restart needed

- Any Python file under `openflip/`.
- New tools added to `TOOL_REGISTRY` (because the auto-inject runs at
  agent-load).
- OAuth credential file (`~/.claude/.credentials.json`) — module-level
  refresh state caches across the process.
- Transport class swap (changing `transport: "discord"` →
  `"imessage"` mid-run).
- `api_config.json` token changes.

Use `restart_gateway(reason, continuation=, force=False)` to do this
safely. The preflight blocks unsafe restarts; pass `force=True` only
if the operator asked for it.

## Restart-sentinel marker hardening

The sentinel file is HMAC-verified, but its sibling `.tool_result.json`
marker is **unsigned**. On startup, `restart_sentinel` now validates the
marker's `marker_conv_id` against the canonical `transport:id` shape
(`^[A-Za-z0-9_]+:[A-Za-z0-9_.+@-]+$`) and rejects any `/`, `\`, `..`, or
control char **before** it reaches the `<id>.jsonl` path joins. A corrupt
or malicious marker (e.g. `../../etc/foo`) is treated as a stale marker —
the tool_result injection is skipped and the marker is still deleted.
Defense in depth against path traversal via the unsigned marker.

---

# 13. OAuth & credentials (Anthropic provider)

## Where creds live

`~/.claude/.credentials.json`. Owned by the OS user running the
process. Contains:

```json
{
  "claudeAiOauth": {
    "accessToken": "...",
    "refreshToken": "...",
    "expiresAt": 1717000000000
  }
}
```

`expiresAt` is unix milliseconds.

## Refresh path

`_refresh_oauth_token` in `anthropic_conversation.py` exchanges the
refresh token at `https://platform.claude.com/v1/oauth/token`. Headers
and body must match Claude Code's bundled axios shape — Anthropic's
endpoint is Cloudflare-fronted and rejects bare requests:

- `Content-Type: application/json`
- `Accept: application/json, text/plain, */*`
- `User-Agent: axios/1.7.7`
- Body includes `scope: "user:profile user:inference
  user:sessions:claude_code user:mcp_servers"` (must echo original
  grant scopes).
- **Do NOT send `Accept-Encoding: gzip`** — urllib doesn't decompress;
  parse blows up but the server still rotated the refresh token
  server-side, invalidating the next attempt. This was the 2026-05-28
  outage shape.

## Concurrency

Multiple agents in the same process refresh through a single asyncio
Future (`_REFRESH_INFLIGHT`). Cross-process coordination uses a file
lock at `~/.claude/.oauth_refresh.lock` (fcntl flock). Other openflip
*and* Claude Code itself contend on the same lock. Stale-window is 10s
(`_REFRESH_LOCK_STALE_S`) — if a process crashed mid-refresh, the next
caller waits at most that long before treating the lock as abandoned.
Retry logic: max 5 attempts with ~1s + jitter sleep between them.

After a 429 from the refresh endpoint, the module backs off for 60s
(`_REFRESH_BACKOFF_AFTER_429_S`) — additional callers in that window
short-circuit to "no token" instead of hammering.

## Surface error

When the refresh path fails, requests raise/return strings starting
with "OAuth token unavailable" or HTTP 401/429 from
`api.anthropic.com/v1/messages`. The framework treats these as
non-content errors and does NOT append them to conversation history
(see `is_framework_error` on `AnthropicAIChatMessage`).

## Request shape for /v1/messages

- `User-Agent` must match a Claude Code-shaped UA (`_DEFAULT_USER_AGENT`).
- `anthropic-version` header.
- `anthropic-beta` header with required flags:
  `claude-code-20250219,oauth-2025-04-20,extended-cache-ttl-2025-04-11,compact-2026-01-12`
  (add `context-1m-2025-08-07` for `-1m` models).
- First `system` block is the billing block (`cc_version=...;
  cc_entrypoint=sdk-cli; cch=...`) — required for Claude Code
  subscription routing on sonnet/opus. Without it, those models 429
  with the third-party harness tier rate limit.
- Cache breakpoints: one on the system block (1h TTL), one on the
  rolling tail (1h TTL). REMINDER.md injection moves the tail
  breakpoint up to `-3` to keep REMINDER edits uncached.
- `output_config.effort` — present only when an effort level resolves to
  a valid value (`low`/`medium`/`high`/`xhigh`/`max`); otherwise the key
  is absent and the API uses its default (`high`). Resolved by
  `AnthropicConversation._effort_level` with precedence **session override
  (`/effort`, persisted in `.meta.json`) > per-model `config.json`
  `models.<id>.effort` (via `config_global.get_effort`) > absent**. Set in
  both the streaming and `_chat_legacy` body builders. See `effort` and
  `/effort` under "What's the current model / context window?".

---

# 14. Self-modification recipes

## Give yourself / another agent a new tool

1. Confirm the tool exists in `TOOL_REGISTRY` (read
   `openflip/tools/__init__.py` for the list, or `list_files
   openflip/tools/`).
2. `read_file agents/<id>/agent.json`.
3. Add an entry to `allowed_tools`:
   ```json
   {"name": "web_search", "auth": {"discord": {"users": [<id1>, <id2>]}}}
   ```
   Or `/grant <tool> @user` for the owner-only-then-broaden path.
4. Save the file. Hot reload picks it up on the next turn.

## Hardening notes (audit 2026-06-07)

A pass of small correctness/security hardening, all live:
- **Web:** `MAX_CONTENT_LENGTH = 8 MB` caps authed POST bodies (413 before buffering). `is_configured()` fails CLOSED on a corrupt `auth.json` (never reopens `/setup`). `/login` is per-IP rate-limited.
- **ACL:** `run_command` / `claude_code` / `restart_gateway` / `restart_flask_app` each enforce `current_caller_is_owner()` internally — admins do NOT get them even if the ACL admin-bypass would otherwise pass. `inject_context` is owner-gated on the tool itself. A raised admin-check now logs instead of silently denying.
- **Timeouts:** ComfyUI submit/upload/view = 120s, history poll = 30s/request inside the 600s loop; `download_url_to_tmp` = 120s + 50 MB cap + unconditional SSRF guard; claude_code reap, ollama UI (list/unload 30s, pull 1800s), and the inter-agent auto-route `fetch_user`/`create_dm` (10s) are all bounded.
- **Secrets:** OAuth-refresh failures log response KEYS not the payload (no refresh_token leak); request/response debug dumps are `0600` in a `0700` dir.
- **Crash-safety:** cron persists `lastRunMs` per-fire (no re-fire storm); snapshot writes are atomic (tmp + `os.replace`).

## Tool output must never leak operator-local paths (path redaction)

Tools return a `ToolResult`; its `model_feedback` / `text` / `ToolResult.fail(...)`
strings reach the LLM and (via the model's reply) potentially a NON-OWNER user.
An absolute local path leaks the operator's OS username, home-dir layout, and
worst case a token/key embedded in a path or traceback. This actually happened
(an agent surfaced `~/...` to a user).

**The protection is a single choke point, not per-tool discipline:**
`tool_executor.build_model_feedback()` and the `text`-post path both run
`utils.redact_paths()` over tool output for any NON-owner caller (the owner is
exempt so their own diagnostics keep real paths). `_caller_is_owner_safe()`
fails CLOSED — unknown/error context → treated as non-owner → redaction ON.
So even a brand-new tool that carelessly dumps `~/...` is auto-scrubbed
for non-owners. `redact_paths` replaces home/project paths with `<path>/<basename>`
and deliberately leaves CDN/HTTP URLs and unrelated system paths (`/usr/...`)
intact so legitimate output (image re-edit URLs, fetched pages) isn't corrupted.

**When you write a tool, still do the right thing at the source:**
- Use `utils.safe_path_display(p)` to tell the model which file you touched —
  it returns the repo-relative path inside the project, else just the basename.
  Never f-string a raw absolute path into `model_feedback`.
- Scrub exception text in error returns: `ToolResult.fail(f"...: {redact_paths(str(e))}")`
  — a raw `{e}` often embeds a full traceback path.
- Never echo `api_config.json`, env vars, tokens, or file CONTENTS that could
  hold a secret into tool output.

This is a hard convention: tool output is untrusted-user-visible; treat absolute
local paths and secrets as leaks.

## Restrict who can DM an agent

```json
"channels": {
  "respond_in": "mentions_only",
  "dm_allowlist_user_ids": [100000000000000000, 111111111111111111]
}
```

Owner is implicitly always allowed. Leave the array out (or empty) to
restore "anyone can DM."

## Change which channels an agent responds in

```json
"channels": {
  "respond_in": "channels_only",
  "always_respond_channel_ids": [1234567890, 9876543210],
  "ignore_channel_ids": [5555555555]
}
```

`respond_in:"channels_only"` + `always_respond_channel_ids` makes the
agent silent everywhere else.

## Schedule a recurring reminder

```
add_cron_job(
  name="Friday standup",
  prompt="Post the weekly progress summary based on the last 7 days of memory.",
  cron="0 9 * * 5",
  timezone="US/Mountain",
  mode="reminder",
  session_id="discord:1234567890",   # or channel_id=1234567890 (legacy)
)
```

Pass the transport-prefixed `session_id` so the reminder fires into the
right conversation — this matters on agents that run more than one
transport (iMessage + Discord). Omit it and it defaults to the current
conversation. To do it silently for your own bookkeeping, use
`mode="data_collection"` (you still need an anchor — `session_id` or
`channel_id` — but the result won't post).

## Add a permanent rule to all agents

Edit `agents/_shared/FRAMEWORK.md`. Audience test: would EVERY agent
need this? If yes, shared. If no, per-agent.

Always back up first:

```
run_command("cp agents/_shared/FRAMEWORK.md agents/_shared/FRAMEWORK.md.pre_<reason>_$(date +%s).bak")
```

To roll back: find the `.pre_*.bak` files (`list_files
agents/_shared/`), pick the newest, restore via
`run_command("cp <backup> agents/_shared/FRAMEWORK.md")`.

## Add a per-agent supplement

`AGENT.md` and personal `TOOLS.md` already exist (auto-created empty) and
are already in `system_files`, injected by default — no `system_files`
surgery needed. Just write the content:

1. For operational/behavioral supplements → `edit_file agents/<id>/AGENT.md`
   (or `write_file` if still empty). Extends `_shared/FRAMEWORK.md`;
   injected right after it.
2. For tool-specific notes → `edit_file agents/<id>/TOOLS.md`. Extends
   `_shared/TOOLS.md`; injected right after it (additive, never replaces).

The default injection order is:
`SOUL.md → _shared/FRAMEWORK.md → AGENT.md → _shared/TOOLS.md → TOOLS.md`.
An empty personal file is a no-op, so a not-yet-written one costs nothing.

## Update core memory

```
update_core_memory("""
# Operator
...current MEMORY.md contents with edits...
""")
```

Read first (`read_memory()`), edit the string, write the whole file
back. There is no partial-update tool — the whole file is replaced.

## Trim REMINDER.md

`REMINDER.md` is paid every turn (uncached, end of payload). Soft-warn
at ~2000 chars. Keep tight: behavioral drift you keep slipping on. If
the nudge no longer fires for you, remove it — every line is paid
forever.

## Restore a conversation from backup

```
list_files agents/<id>/conversations/        # find the .bak.jsonl you want
run_command("cp agents/<id>/conversations/discord:NNN.jsonl.pre_reset_TS.bak.jsonl agents/<id>/conversations/discord:NNN.jsonl")
restart_gateway("restoring conversation backup")
```

The in-memory conversation pops on restart. On the next message in that
channel, the restored file loads.

## Roll back a FRAMEWORK.md edit

The framework keeps `.pre_slim_<ts>.bak` and `.bak.rvp` style backups
around recent edits in `agents/_shared/`. Find them with `list_files
agents/_shared/`, pick the one before the bad edit, restore with `cp`,
hot-reload via the next message (or `/reload`).

---

# 15. Diagnostics

Source of truth for everything that happened: `journalctl --user -u
openflip --since "10 minutes ago"`.

## "Why didn't your last reply post?"

1. `journalctl --user -u openflip --since "5 minutes ago"` — look for:
   - `TERMINAL CONTRACT FAILED: turn ended without operator-visible
     output` (terminal-result-contract failure). The log line carries
     the cause: `(reason=...)` means the model genuinely returned no
     text and no tool dispatched; `(provider_error: ...)` means the
     provider (Anthropic) returned a non-200 — rate limit (429),
     overload (529), auth, or 400. **Provider errors are now surfaced
     to the operator verbatim** ("⚠️ The model provider returned an
     error: ...") instead of the old generic "known bug" catch-all, so
     a 429/529 reads as a temporary rate limit/overload, not a bug. The
     genuinely-empty case posts "⚠️ The model returned an empty
     response (reason: ...). Try `/reset` if it persists." When the
     in-loop framework-error branch already posted the error to the
     channel, the terminal contract stays silent (no double-post).
   - `OAuth token unavailable` (refresh failed; see section 13).
   - `MalformedRequestError` (request body failed pre-flight
     validation; Anthropic 400 short-circuited).
   - `inbound worker: turn raised <exception>` (turn crashed; see
     stack trace).
2. `tail -5 agents/<id>/conversations/<conv_id>.jsonl` — confirm
   whether the assistant message actually landed in history.
3. If the assistant message is in history but didn't post, check
   `_split_for_discord` chunking and Discord 429 / network issues in
   the log.

## "Why is OAuth refresh failing?"

1. `journalctl --user -u openflip | grep -i oauth` — last refresh
   attempt's status.
2. `stat ~/.claude/.credentials.json` — mtime should be recent if a
   refresh succeeded.
3. Common causes (in order of likelihood):
   - 429 cooldown active (60s after last 429). Wait it out.
   - Headers mismatched — verify `_refresh_oauth_token` matches
     section 13's shape (UA, Accept, scope, no Accept-Encoding).
   - Refresh token rotated server-side but never saved locally —
     caused by Accept-Encoding bug. Re-auth via Claude Code CLI.
   - File lock held by another process — `lsof
     ~/.claude/.oauth_refresh.lock`.

## "Why did the agent forget something?"

Possible causes:
1. `/reset` was run. Check `.pre_reset_*.bak.jsonl` files in
   `conversations/`.
2. `/compact` archived old history into a compaction summary. Check
   `.compaction_*.bak.jsonl`.
3. The fact was never saved. Memory is opt-in via `save_memory` /
   `update_core_memory`. Conversation history is per-channel — facts
   from one channel don't leak to another. To make a fact survive
   `/reset` or cross-channel, it has to be in `MEMORY.md` or a daily
   log.
4. Anthropic auto-compaction summarized it. Check `.meta.json` for
   `compaction_block`. `/uncompact` restores.

## "How big is my conversation?"

```
wc -l agents/<id>/conversations/<conv_id>.jsonl
du -h agents/<id>/conversations/<conv_id>.jsonl
```

For token estimates on Anthropic agents, read `.meta.json`'s
`last_usage` — it has the last turn's reported input/output/cache
counts.

## "Did my edit_file actually apply?"

```
read_file <path>                                  # confirm visually
grep -n "<new content snippet>" <path>            # grep for the added text
```

`edit_file` can silently miss when `old_string` doesn't match
byte-for-byte. Always verify.

## "What tools can I actually call right now?"

In Discord: `/help` (visible to the speaker). In a tool call: every
function in your local scope IS callable for the current speaker —
denied tools don't get function references injected.

## "What's the current model / context window?"

Per-agent `model` and `think` live in the agent's `agent.json`.
Per-**model** knobs live in `config.json` under `models.<id>` (keyed by the
bare model id — the `-1m` suffix is part of the key, so a 1m-context model keys on
`claude-opus-4-8-1m`, NOT `claude-opus-4-8`):

- `context_window` — token window; feeds `get_model_context_window`.
- `compaction_trigger` — overrides the `window - reserve` default; floored at
  Anthropic's 50k minimum. See `get_compaction_trigger`.
- `effort` — reasoning depth. It's a **model capability, not an agent
  trait**, so it lives here next to `compaction_trigger`, read via
  `config_global.get_effort`. For Anthropic models it's sent as
  `output_config.effort` on every `/v1/messages` request; valid values (the
  ONLY five accepted): `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`.
  For OpenAI models it's sent as `reasoning_effort` on the Chat Completions
  request; valid values: `"minimal"`, `"low"`, `"medium"`, `"high"` — and it
  must ONLY be set on reasoning-capable models (o-series / gpt-5 family);
  other OpenAI models 400 on the parameter. **Omitting it — or setting
  anything else — sends no effort field at all, so the API falls back to its
  default (Anthropic: `"high"`).** Junk values (wrong type, typo, unknown
  level) resolve to "absent". `get_effort` also returns None for the ollama
  provider, so an Ollama model never builds an effort field.
  **Model-gating caveat:** `xhigh` is opus-4-7/opus-4-8 only and `max` is
  opus-4.6+ only. Anthropic may **400** on a level the configured model
  doesn't support (it does not silently clamp), so only set a level the model
  actually supports. `claude-opus-4-8-1m` is set to `"xhigh"`.
  **Precedence:** a per-conversation session override (set with `/effort`)
  WINS over this model config, which in turn wins over the API default. So the
  effective level is `session override > models.<id>.effort > "high"`.

### `/effort` — per-conversation override (owner-only, Anthropic-only)

`/effort <level>` sets a session-level effort override for THIS conversation
that beats the model config above. Choices: `low`/`medium`/`high`/`xhigh`/`max`,
plus `default` to CLEAR the override and fall back to the model config. The
override persists per-conversation in the `.meta.json` sidecar (key
`effort_override`, written only when set — meta stays byte-identical for
conversations that never touched `/effort`), so it survives restarts. It takes
effect on the next real turn (no synthetic turn fired). Same model-gating
caveat applies: don't set `xhigh`/`max` on a model that doesn't support it.

`/status` slash command shows runtime state.

### `/usage` — aggregated API usage (owner-only)

`/usage [window] [group_by]` shows aggregated model-API usage (Anthropic +
OpenAI turns) over a time window, read from the usage ledger (below).
Owner-only, ephemeral response.

- `window` — `24h` (default) / `7d` / `30d` / `all`.
- `group_by` — `agent` (default) / `channel` / `user` / `model`.

Output is a monospace table: a header line with the window plus grand totals
(turns, total tokens, total estimated cost in USD), then a per-group breakdown
sorted by estimated cost descending — group value, turns, humanized token count
(e.g. `2.1M` / `800k`), and `$X.XX`. Capped at the top 25 groups to stay under
Discord's 2000-char limit (a `… N more group(s) not shown` line is appended when
truncated). An empty window replies `No usage recorded in the last <window> yet.`

### Usage ledger

Every completed Anthropic or OpenAI turn appends ONE raw row to an append-only SQLite
ledger at `data/usage_ledger.db` (WAL mode; gitignored). The row carries the
full token breakdown (input / output / cache-read / cache-creation), a computed
total, an estimated USD cost, the outcome (`ok` / `compaction` / `error`), and
the entire usage dict verbatim in `raw_usage` so no future counter is ever lost.
Nothing is aggregated at write-time — `/usage` and any ad-hoc query are read-side
aggregations over these rows.

- **Module:** `openflip/usage_ledger.py` (`record_usage`, `query_usage`,
  `aggregate`, `totals`, `purge_older_than`). The write is bulletproof: a failure
  recording usage is logged and swallowed, never breaking a turn.
- **Retention:** self-cleaning — rows older than 30 days are purged opportunistically
  on write, throttled to at most once per process-hour. No cron needed.
- **Pricing:** `MODEL_PRICING` in `usage_ledger.py` maps a model-name substring
  (`opus` / `sonnet` / `haiku`) to per-1M-token rates. These are **hand-maintained**
  — update the table when Anthropic changes prices. An unknown model records
  `est_cost_usd = 0.0` (it never guesses a price).

---

## Inbound trigger endpoint

`POST /trigger/<agent_id>` lets an EXTERNAL script (cron on another box, a
webhook, a file/email/RSS watcher) wake a *running* agent with a synthetic
turn — the openflip-native successor to OpenClaw's inbound webhook. The
external script does its own cheap check (zero LLM cost) and only POSTs when
there is real work; the agent then acts with its own pre-approved tools.

It is served by the webapp (`openflip/web/app.py`), NOT a browser route: it
takes a bearer token, not a login session, and returns JSON errors (never a
`/login` redirect).

### Auth — bearer token

The endpoint is gated by a single machine-to-machine bearer token.

- The token file lives at `openflip/web/data/trigger_token` (mode `0600`,
  single line of plaintext).
- It is minted **once at webapp startup** if absent — the request path only
  ever *reads* it, never creates it. On boot the log prints its location.
- Read it to hand to your poller:
  ```
  cat openflip/web/data/trigger_token
  ```
- Send it as `Authorization: Bearer <token>`. The check is constant-time and
  **fails closed**: if the file is missing, empty, or unreadable, *every*
  caller is rejected. Rotate it by stopping the webapp, deleting the file, and
  restarting (a new token mints on boot).

### Opting an agent in — the `trigger` block in `agent.json`

OFF by default. A missing block means the endpoint returns `403` for that
agent. To enable:

```json
"trigger": {
  "enabled": true,
  "allowed_tools": ["generate_image", "send_message"],
  "session": "discord:123456789012345678",
  "rate_limit_per_minute": 6
}
```

- `enabled` — master switch. `false`/missing → `403`.
- `allowed_tools` — the **server-side** allow-path: the ONLY source of extra
  tool grants for a trigger turn. Intersected against the framework's real
  tools and a hard denylist of dangerous/admin tools (shell, framework
  restart, a full coding agent, filesystem mutation/exfiltration, cron
  persistence, cross-agent injection, message deletion), so a typo or an
  over-broad list here can never grant more than the framework considers safe
  for an unattended caller.
- `session` — the **server-side** fallback, transport-prefixed target the turn
  lands in (e.g. `"discord:<channel_id>"`, `"imessage:<id>"`). Used when the
  caller does NOT supply a `session` in the body. The transport prefix must
  name a transport this agent actually listens on. Hard-validated against path
  traversal before it touches the filesystem. (A caller MAY override this
  per-request — see the security model below — but only through the identical
  validation.)
- `rate_limit_per_minute` — per-agent sliding-60s inbound cap (default `6`).
  Over the limit → `429`.

**`allowed_tools` applies on EVERY transport, Discord included (fixed
2026-06-05).** The trigger turn's grants ride on the synthetic `Session`, and
`run_synthetic_turn` now honors that Session as the source of truth on all three
transport branches (headless, iMessage/non-Discord, Discord). Non-Discord
transports use the Session directly. For Discord agents the runtime no longer
flattens it to a bare channel id: when the session's id is a real Discord
channel it resolves that channel and wraps it in a `_SessionChannel` that carries
the passed Session through to the turn (the nextcord channel can't hold it
itself — `__slots__`); when the session's id has no real channel behind it the
turn runs against it via the `TransportChannel` shim. Either way the session's
`tool_grants` AND its `conversation_id` reach the turn, so a Discord trigger gets
the curated `allowed_tools` extras and is loaded/persisted against its own
conversation_id instead of collapsing onto the agent's default session. The same
runtime fix closes the equivalent cron tool_grants/Discord gap (cron synthetic
turns build their target the same way, via a `Session`).

**Channel resolution is centralized in `_resolve_synthetic_channel` (refactored
2026-06-07).** `run_synthetic_turn` no longer inlines the four-way resolution
(headless, iMessage/non-Discord, Discord+Session, Discord legacy raw-int); it
calls `self._resolve_synthetic_channel(...)`, which returns the channel-like
object (or `None` only when the legacy raw-int path hits a *genuinely missing*
channel, in which case the turn is abandoned as before). The `_SessionChannel`
and `TransportChannel` shim usage is unchanged. The one behavior change: both the
Discord+Session path and the legacy raw-int path now share the **shim-fallback**
when the 15s `fetch_channel` call *stalls* (times out). Previously the Session
path warned and fell through to the shim while the legacy-int path logged an
ERROR and `return`ed — silently dropping the turn. Now a transient channel-fetch
stall never drops a turn on either path: the legacy-int stall builds a
Discord-keyed `Session` (conversation_id `discord:<id>`, matching the session
`_run_turn` would otherwise synthesize, so history stays on the same key) and
runs against the `TransportChannel` shim, which re-resolves the channel at send
time.

### Security model — caller supplies prompt + context + an optional session

The HTTP body is `{"prompt": "...", "context"?: "...", "session"?: "..."}`.
`prompt` is required; `prompt` and `context` are capped at 8000 chars and
concatenated into the turn.

The caller can **never** name tools — grants come exclusively from the agent's
server-side `trigger.allowed_tools`. That hole (privilege escalation by naming
a powerful tool) is closed *structurally*: the caller's input never reaches the
grant computation.

The caller MAY supply an optional `session` selector so a poller can route each
item into its own conversation (e.g. one session per email thread). This does
NOT reopen the path-traversal hole: a caller-supplied `session` is passed
through the **same** hard validation as the server-side config session
(`_validate_trigger_session`) before it touches the filesystem — bounded to 256
chars, rejected for any `/`, `\`, `..`, or control char, required to be exactly
one `transport:id` pair where the transport is one this agent actually listens
on and the id matches the strict id whitelist (digits-only for Discord). The
caller's string is never trusted raw, and the `_conversation_io` path-hardening
backstop still applies. If the body omits `session` (or sends an empty string),
the turn falls back to the server-side `trigger.session`. An invalid
caller-supplied session is rejected with `400 "invalid session"`.

The turn also runs as a synthetic non-owner, non-admin speaker, so it cannot
pick up owner/admin-gated tools by identity — only the curated server-side
grants (plus whatever the agent already exposes to all users) apply.

The boundary, therefore, is the `trigger.allowed_tools` you configure: a
valid-token caller fully controls the agent's *reasoning* for that turn, so
grant only tools whose worst-case misuse you accept. Do not grant tools that
mutate the filesystem, run code, restart services, or message other agents
unless you mean it.

### Responses

- `202 {"status":"accepted","agent":"<id>"}` — queued (fire-and-forget; the
  turn runs asynchronously and its result is NOT returned to the caller).
- `401` — missing/invalid bearer token.
- `403` — agent has no enabled trigger config, or its configured session is
  missing/invalid.
- `400` — body is not a JSON object, `prompt` is missing/oversized, or a
  caller-supplied `session` is oversized or fails validation (`"invalid
  session"`).
- `404` — agent is not currently running.
- `429` — per-agent rate limit exceeded.

### Worked example

```bash
TOKEN=$(cat openflip/web/data/trigger_token)
curl -sS -X POST http://127.0.0.1:1750/trigger/myagent \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "A new file landed in the inbox. Summarize it for the channel.",
        "context": "filename: report-2026-06-05.pdf, 3 pages",
        "session": "discord:123456789012345678"
      }'
# -> {"status":"accepted","agent":"myagent"}
# "session" is optional — omit it to fall back to the agent's configured
# trigger.session. When supplied it is hard-validated (same rules as config).
```

The webapp binds to `127.0.0.1` by default — reach it from another host only
by setting `OPENFLIP_WEB_HOST=0.0.0.0` (or, preferred, fronting the loopback
port with nginx/caddy running TLS).

### Identity-scoped tool grants — caller asserts WHO, the server decides WHAT

Everything above is the **flat** mode: one shared bearer, one `allowed_tools`
list, every inbound gets the same toolset. Identity-scoped grants add a second
mode on top of it. The caller asserts an **opaque identity** — a `"id"` string
in the body like `"alice@gmail.com"`, `"1234"`, `"custom-thing-7"` — and the
**server** maps that identity to a set of tools (`id → labels → tools`). A
**per-hook bearer token** proves *which identities a given hook is allowed to
assert*, so a leaked email-hook token can't claim a Discord owner's id to widen
its grants. The caller still names **zero tools** — HOLE 1 stays structurally
closed; the only new thing it contributes is an opaque id, which is a dict key
the server looks up, never a capability. This mode engages **only** when
`trigger.tokens` is configured; with no `tokens`, the endpoint behaves exactly
as the flat mode above (and any `id` in the body is ignored entirely).

### Minting a per-hook token secret

Each hook gets its own secret, separate from (and instead of) the shared
`trigger_token`. Same machine-to-machine trust model as that file — plaintext,
read back verbatim, compared constant-time, fail-closed — just one secret per
hook instead of one per webapp.

- Generate a secret:
  ```
  python -c "import secrets;print(secrets.token_urlsafe(32))"
  ```
- It goes in `openflip/web/data/trigger_token_secrets.json`, mode `0600`, owned
  by the service user. The file is a JSON map **namespaced per agent**:
  ```json
  {
    "agent-b": {
      "email-hook":   "Yc8...long-random-secret...",
      "discord-hook": "Qa1...another-secret..."
    },
    "some-other-agent": {
      "email-hook":   "totally-different-secret-never-shared-with-agent-b..."
    }
  }
  ```
  The outer key is the `agent_id`; the inner keys are **token names** that must
  match the names you declare in that agent's `trigger.tokens` (below). Two
  agents may both define an `email-hook` token with *different* secrets — a
  secret is scoped to exactly one agent and can never authenticate to another.
- It is **hand-written by the operator**, exactly like the shared
  `trigger_token`: the request path only ever *reads* it, never creates,
  chmods, or rotates it. Minting and locking down the store is an operator
  action. If the file is missing, empty, corrupt, or has no block for the
  requested agent, that agent's multi-token auth fails closed (every caller
  `401`s). The reader **warns** (in the log) if the file is group/other-readable
  but still reads it — you own the `chmod 0600`.
- Rotate by editing this file and the hook in lockstep (no overlap window).

Secrets live **out of `agent.json`** on purpose: `agent.json` is world-readable
framework config that gets hot-reloaded, diffed, and re-serialized — a
long-lived bearer secret does not belong there. `agent.json` holds only the
*binding* (which ids a named token may assert), never the secret.

### The identity-scoped `trigger` block in `agent.json`

The flat block still works unchanged. To turn on identity scoping, add three
keys — `identities`, `labels`, `tokens` — alongside the existing ones:

```json
"trigger": {
  "enabled": true,
  "session": "discord:123456789012345678",
  "rate_limit_per_minute": 6,

  "allowed_tools": ["web_search"],

  "identities": {
    "alice@gmail.com": ["messenger"],
    "1234":            ["readonly"],
    "custom-thing-7":  ["messenger", "readonly"]
  },

  "labels": {
    "messenger": ["send_message", "generate_image"],
    "readonly":  ["web_search"]
  },

  "tokens": {
    "email-hook": {
      "allowed_ids":         ["alice@gmail.com", "bob@gmail.com"],
      "allowed_id_patterns": ["*@trusted-corp.com"]
    },
    "discord-hook": {
      "allowed_ids": ["*"]
    }
  }
}
```

- `enabled` / `session` / `rate_limit_per_minute` — unchanged from flat mode.
- `allowed_tools` — now the **default / anonymous bucket**: the grants used when
  a request authenticates with a per-hook token but asserts **no** `id` (or its
  asserted id has no identity mapping and you are *not* in identity mode for
  that turn). Set it to `[]` for "no tools unless you assert a known id", or to a
  read-only baseline. It is still intersected against real tools and the
  denylist, same as always.
- `identities` — maps an **opaque caller id → a list of label names**. The
  server does not interpret what *kind* of id it is; it's a dict key. Unknown
  ids resolve to no tools (fail-closed).
- `labels` — maps a **label name → a list of tool names**. This is the
  server-owned capability map. An id with multiple labels gets the **union** of
  their tools.
- `tokens` — maps a **token name → which ids that token may assert**. The name
  must have a matching secret in `trigger_token_secrets.json[<agent_id>]`.
  - `allowed_ids` — exact-match id list. The literal `"*"` means "any id".
  - `allowed_id_patterns` — `fnmatch`-style **full-string** globs, case-sensitive
    (e.g. `"*@trusted-corp.com"`). See the anchor warning below.
  - A token with no matching secret (or a secret with no matching binding here)
    can't authenticate — the two must intersect.

### How a request resolves to tools

Multi-token mode (`trigger.tokens` present) resolves in this strict order; any
step failing closed yields **no identity-scoped tools** (never a silent
fall-up):

1. **Identify the token.** The presented `Bearer` is compared (constant-time, no
   early return) against every usable secret. No match → `401`. Match → the
   request is now *bound to that token name*.
2. **May this token assert this id?** If the body carries an `id`, it is
   seatbelt-validated (rejects `/`, `\`, `..`, control chars, anything outside
   the strict charset/length cap) → `400 "invalid id"` on violation. Then **the
   spine**: the identified token must be permitted to assert that id via its
   `allowed_ids` / `allowed_id_patterns`. Not permitted → `403`, *before* any
   label lookup.
3. **id → labels → tools.** The id's label names are looked up in `identities`,
   each label expanded via `labels`, and the results **unioned** (de-duped) into
   the requested tool set.
4. **Intersect real tools.** The requested set is reduced to tools the agent
   actually exposes (`grants ⊆ runner.agent.allowed_tools`).
5. **Subtract the denylist.** The canonical `DANGEROUS_TOOL_NAMES` set
   (`openflip/_constants.py`, aliased here as `_TRIGGER_FORBIDDEN_TOOLS`: shell,
   restart, coding agent, fs mutation/exfil, cron, cross-agent injection,
   message deletion) is removed regardless of how the set was built. Dropped
   tools are logged. This is the single source of truth — the web config editor
   reuses the same set to refuse ADDING any of these to an agent that lacks
   them.

Key properties:

- **Replace, not additive.** When an id is asserted under a token, its resolved
  union *replaces* the flat `allowed_tools` bucket — it is not added to it.
- **Misconfig and unmapped both fail closed.** If `identities`/`labels` are
  missing or only half-configured, OR the asserted id has no mapping, the turn
  resolves to **`[]` (no tools)** — it does **not** fall back to the flat
  `allowed_tools` bucket. (If you want everyone on one flat toolset under
  multi-token auth, simply send **no** `id`.)
- The flat bucket is used only when **no `id` is asserted** (multi-token,
  anonymous) or in single-bearer back-compat mode.

### The hook's request

The hook sends its **own** per-hook secret and adds one field, `id`. It still
names **zero tools** — the server does all capability mapping:

```bash
SECRET=$(python -c "import json;print(json.load(open('openflip/web/data/trigger_token_secrets.json'))['agent-b']['email-hook'])")
curl -sS -X POST http://127.0.0.1:1750/trigger/agent-b \
  -H "Authorization: Bearer $SECRET" \
  -H "Content-Type: application/json" \
  -d '{
        "prompt":  "New email landed. Summarize it for the channel.",
        "context": "from: alice@gmail.com, subject: Q3 numbers",
        "id":      "alice@gmail.com"
      }'
# -> {"status":"accepted","agent":"agent-b"}
# The body names NO tools. The server maps id="alice@gmail.com" -> ["messenger"]
# -> ["send_message","generate_image"], then intersects real tools + subtracts
# the denylist. "session" is still optional (same hard validation as flat mode).
```

**What the calling script changes vs. flat mode:** one header value (the shared
`trigger_token` → this hook's own `email-hook` secret) and one new body field
(`"id": <sender>`). Nothing else.

### Security rules an AI must respect (multi-token mode)

- **Anchor the `@` in patterns.** `allowed_id_patterns` are **full-string**
  globs, NOT domain matchers. Write `"*@example.com"`, **never** `"*example.com"` — a
  pattern missing the `@` is matched against the *entire* id and also matches
  hostile lookalikes like `evilexample.com` and `attacker@notexample.com`, letting a
  token assert ids you never intended.
- **Secrets never go in `agent.json`.** Bindings (which ids a token may assert)
  live in `agent.json`; secrets live only in `trigger_token_secrets.json` (mode
  `0600`). Never paste a secret into config or surface `trigger.tokens` secrets
  in any web view.
- **Fail closed, always.** Missing/half-configured `identities`/`labels`, an
  unmapped id, a token not permitted to assert the id, a missing secret block —
  every one resolves to no tools (or `401`/`403`), never to a wider grant.
- **HOLE 1 stays closed.** The body's only accepted fields are `prompt`,
  `context`, `session`, `id`. The caller never contributes a tool name; grants
  come exclusively from server-side `identities`/`labels`/`allowed_tools`. Do
  not add a tool field to the body for any reason.
- **Cross-agent caveat.** Secrets are namespaced by `agent_id`, so a token named
  `email-hook` under agent A cannot authenticate to agent B — *unless an
  operator hand-copies the same secret string into both agents' blocks*. Use a
  distinct secret per agent even when the token name is reused.

### Back-compat

Absent `identities`, `labels`, **and** `tokens`, behavior is **byte-identical**
to the flat mode documented above: single shared bearer (`trigger_token`), flat
`allowed_tools` grants, and any `id` in the body is **ignored entirely** — not
read, not validated, no new `400`. A known-good flat trigger turn (including one
that sends a stray `id`) diffs identical before and after the feature ships.
