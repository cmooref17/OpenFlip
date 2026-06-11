# openflip — Setup

A multi-agent Discord bot framework. One process spawns N independent Discord
bots, one per agent directory. Each agent has its own personality, model
(Ollama or Anthropic), tools, and conversation history.

This doc gets you from a clean machine to a running bot.

---

## 1. Get the code

```bash
git clone https://github.com/cmooref17/OpenFlip.git
cd OpenFlip
```

## 2. Python + virtualenv

You need Python 3.11 or newer. Create a venv inside the project:

```bash
python3 -m venv .venv
```

Activate it. The command depends on your shell:

| Shell | Activation command |
|---|---|
| bash / zsh (Linux, macOS default) | `source .venv/bin/activate` |
| fish (macOS or Linux fish users) | `source .venv/bin/activate.fish` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd.exe | `.venv\Scripts\activate.bat` |

You'll know it worked when your prompt prefix shows `(.venv)`.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. First run (generates blank config)

```bash
sh start.sh        # Windows: start.bat — see docs/WINDOWS.md
```

This will print:
```
Created default config.json — edit owner_id and service hosts, then restart.
Created empty api_config.json — add Discord bot tokens under 'tokens', then restart.
openflip starting (owner_id=0)
Discovered agents: []
No agents running. Fill in api_config.json tokens and restart, or enable an agent.
```

Stop it with Ctrl-C. Two new files now exist in the project root:
`config.json` and `api_config.json`. You'll fill them in next.

## 5. Get your Discord owner ID

You need YOUR Discord user ID (a long integer, not your username). To get it:

1. Open Discord → User Settings → Advanced → enable **Developer Mode**.
2. Right-click your own name anywhere → **Copy User ID**.

That number goes in `config.json` as `owner_id`. Without it, owner-only
commands (like `/restart_gateway` or admin actions) won't work for you.

## 6. Fill in `config.json`

Open `config.json` and set `integrations.discord.owner_id` to your Discord user ID:

```json
{
  "integrations": {
    "discord": {
      "owner_id": 123456789012345678,
      "tokens": {}
    }
  },
  "ollama_host": "http://localhost:11434",
  "data_dir": "data",
  "tmp_dir": "data/tmp",
  "output_dir": "data/output",
  "embedding_model": "nomic-embed-text",
  "dry_run_tools": false,
  "display_name_map": {},
  "compaction_reserve_tokens": 20000,
  "models": {}
}
```

Every per-transport configuration (identity, bot tokens, anything else
transport-specific) lives under `integrations.<transport_name>.*`. When a
future iMessage / Slack / etc. transport ships, its config sits alongside
Discord under the same `integrations` key.

You can ignore `ollama_host` and `embedding_model` if you're using the
Anthropic provider — they only apply to Ollama-based agents.

## 7. Create your first agent

Decide an agent ID (the directory name — keep it lowercase, no spaces,
e.g. `example_agent`). Then:

```bash
mkdir -p agents/example_agent
```

### 7a. Write `agents/example_agent/agent.json`

Minimum Anthropic agent (recommended for first-run; needs an Anthropic API key):

```json
{
  "id": "example_agent",
  "display_name": "Example Agent",
  "model": "anthropic/claude-sonnet-4-5",
  "provider": "anthropic",
  "system_files": ["SOUL.md"],
  "memory_enabled": true,
  "tool_response_mode": "text_then_media",
  "channels": {
    "respond_in": "mentions_only",
    "ignore_channel_ids": [],
    "always_respond_channel_ids": []
  },
  "allowed_tools": [
    "save_memory",
    "search_memory",
    "read_memory",
    "list_memory_files",
    "update_core_memory",
    "web_search",
    "fetch_url",
    "read_file",
    "write_file",
    "edit_file",
    "list_files",
    "delete_file",
    "run_command",
    "send_message"
  ],
  "allowed_read_paths": ["agents/example_agent"],
  "allowed_write_paths": ["agents/example_agent"]
}
```

Minimum Ollama agent (needs a local Ollama server with the model pulled):

```json
{
  "id": "example_agent",
  "display_name": "Example Agent",
  "model": "qwen2.5:14b",
  "provider": "ollama",
  "system_files": ["SOUL.md"],
  "memory_enabled": true,
  "tool_response_mode": "text_then_media",
  "channels": {"respond_in": "mentions_only"},
  "allowed_tools": [
    "save_memory", "search_memory", "read_memory",
    "web_search", "fetch_url", "read_file", "write_file",
    "list_files", "run_command", "send_message"
  ],
  "allowed_read_paths": ["agents/example_agent"],
  "allowed_write_paths": ["agents/example_agent"]
}
```

### 7b. Write `agents/example_agent/SOUL.md`

This is the agent's personality. Anything you want. Example:

```markdown
You are Example Agent — a calm, precise, helpful assistant. You speak directly
and don't pad your replies. When asked technical questions, you give
exact answers and cite where you'd verify them.

You're allowed to push back when something looks wrong. You don't agree
just to be agreeable.
```

## 8. Make a Discord bot for the agent

1. Go to https://discord.com/developers/applications.
2. **New Application** → name it whatever you want (this is the bot's display name).
3. Left sidebar → **Bot** → **Add Bot** → **Reset Token** → copy the token.
   - **Keep this token secret. Anyone with it controls the bot.**
4. Under **Privileged Gateway Intents**, enable:
   - **Message Content Intent**
   - **Server Members Intent**
5. Left sidebar → **OAuth2** → **URL Generator** →
   - Scopes: `bot`, `applications.commands`
   - Bot Permissions: `Send Messages`, `Read Message History`, `Attach Files`, `Use Slash Commands`, `Read Messages/View Channels`
6. Copy the generated URL, paste it in a browser, invite the bot to your server.

## 9. Add the bot token to `config.json`

Open `config.json` again. The bot token goes under
`integrations.discord.tokens`, keyed by your agent's directory name from
step 7:

```json
{
  "integrations": {
    "discord": {
      "owner_id": 123456789012345678,
      "tokens": {
        "example_agent": "MTAxxxxxxxxxxxxxxxxxxxxx.xxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxx"
      }
    }
  },
  ...
}
```

The KEY (`example_agent` above) must exactly match your agent's directory name from
step 7 (since the folder is `agents/example_agent/`).

For multiple agents, add more entries to the tokens map:
```json
"tokens": {
  "example_agent": "first_bot_token...",
  "another_agent":  "second_bot_token..."
}
```

**Note for users migrating from older openflip versions:** if your install
still has a top-level `owner_id` field or a separate `api_config.json` file,
the framework will auto-migrate them under `integrations.discord.*` on the
next startup. `api_config.json` gets renamed to `api_config.json.bak`.

## 10. Anthropic API key (if using `provider: "anthropic"`)

Get an API key from https://console.anthropic.com/.

Set it in your shell BEFORE running openflip:

```bash
export ANTHROPIC_API_KEY=sk-ant-xxx...
```

To persist: add that line to `~/.bashrc`, `~/.zshrc`, or `~/.config/fish/config.fish`.

## 11. Start it for real

```bash
sh start.sh        # Windows: start.bat — see docs/WINDOWS.md
```

You should see:
```
openflip starting (owner_id=YOUR_ID)
Discovered agents: ['example_agent']
[example_agent] Online as YourBotName (12345...)
```

If your bot is offline in your Discord server, the issue is one of:
- Bot token wrong → re-copy from the developer portal
- Bot intents not enabled → re-check step 8
- Bot not invited to the server → re-do the OAuth2 URL invite

## 12. Talk to it

In your Discord server, in a channel the bot can see:

```
@Example Agent hello
```

Or DM the bot directly — DMs always respond.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'nextcord'`** — venv isn't active, or
`pip install -r requirements.txt` didn't finish. Re-activate the venv (step 2)
and re-install.

**`Agent example_agent: no token`** — your `api_config.json` key doesn't match the
agent directory name. Both must be exactly `example_agent` (or whatever you named yours).

**Bot shows online but doesn't respond** — check:
- Channel permissions (the bot needs read + send on the channel)
- `channels.respond_in` in `agent.json` is `"all"` (responds everywhere) or
  `"mentions_only"` (only when @mentioned)
- Bot has "Message Content Intent" enabled in the developer portal

**Anthropic 401** — `ANTHROPIC_API_KEY` isn't set in the shell that started openflip.

---

## What's next

* `agents/<id>/MEMORY.md` is the agent's core memory file. Write things you
  want the agent to know permanently here.
* `agents/<id>/memory/YYYY-MM-DD.md` are daily logs. Created automatically
  when the agent calls `save_memory`.
* `/toolset` slash command lets you tune per-tool settings without restarting.
* `/reset` clears a channel's conversation history.
* `/reload` reloads an agent's config + system files without restarting.
