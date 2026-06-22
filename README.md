# OpenFlip

A self-hosted multi-agent assistant framework. You define agents as folders of
plain-text config and personality files; OpenFlip runs them as always-on chat
bots that can use tools, remember things, talk to each other, and act on a
schedule.

## What it does

- **Chat transports** — each agent is a Discord bot; iMessage is also supported
  on macOS. One agent can listen on multiple transports at once.
- **Model providers** — Anthropic API (Claude) or a local Ollama server,
  per-agent and per-model configurable (reasoning effort, context, etc.).
- **Tools** — file access, web search/fetch, sending messages, and a drop-in
  `openflip/tools/` plugin system, all gated by per-agent/per-user ACLs.
- **Inter-agent dispatch** — agents can message each other (`talk_to_agent`)
  and relay answers back to whoever asked.
- **Scheduling** — cron jobs per agent, plus optional proactive ticks (KAIROS)
  and end-of-turn memory consolidation (dreams).
- **Web panel** — login-gated dashboard for conversations, usage graphs, agent
  config editing, and token-authed webhook triggers (`POST /trigger/<agent>`).
- **Operational safety** — restart sentinel, atomic conversation persistence,
  secret scrubbing of tool output, per-session tool grants.

## Extending OpenFlip (without losing your work on `git pull`)

**NEVER edit git-tracked framework files to add your own models, agents, tools,
or transports.** OpenFlip is a live repo — a `git pull` overwrites tracked files,
silently clobbering any customization you wedged into them. (This warning is for
you *and* for any AI coding assistant you point at the repo — they reach for
`openflip/main.py` by default; redirect them here.)

Every kind of extension has a **gitignored** home that survives every update.
Put your customization there instead:

| You want to add a… | Put it in… | Why it's safe |
|---|---|---|
| **Model** | a `models.<bare-id>` block in `config.json` | `config.json` is gitignored → survives pull |
| **Agent** | a new `agents/<id>/` dir + its token in `config.json` | personal agent dirs are gitignored by default → survives pull |
| **Local tool** | a `@tool` `.py` file dropped in `openflip/tools/` | non-core tool files are gitignored + auto-loaded → survives pull |
| **Transport** | a `.py` file in `transports_local/` (`TRANSPORT_NAME` + `build(agent)`) | dir is gitignored + auto-discovered → survives pull (see [transports_local/README.md](transports_local/README.md)) |

None of these require touching a single tracked file. Step-by-step recipes for
each are in **[agents/_shared/MANUAL.md](agents/_shared/MANUAL.md)** §14.

**The one exception — contributing to the framework itself:** making a tool
*first-class* (shipped to everyone in the public repo) does require editing
tracked files — a `!`-allowlist line in `.gitignore` plus a static import in
`openflip/tools/__init__.py`. Those are version-controlled and *can* conflict on
pull, which is expected for a real contribution. For anything personal, use the
gitignored paths above.

## Requirements

- Python 3.11+
- A Discord bot token per agent (free — see SETUP.md §8)
- An Anthropic API key, or a local [Ollama](https://ollama.com) server

## Quick start

```bash
git clone https://github.com/cmooref17/OpenFlip.git
cd OpenFlip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m openflip.main   # first run generates a blank config.json
# fill in config.json + create your first agent, then:
./start.sh                # Linux/macOS — on Windows use start.bat (see docs/WINDOWS.md)
```

The full walkthrough — config fields, creating an agent (`agent.json` +
`SOUL.md`), Discord bot setup, troubleshooting — is in **[SETUP.md](SETUP.md)**.

## Documentation

- **[SETUP.md](SETUP.md)** — install and first-agent walkthrough
- **[agents/_shared/MANUAL.md](agents/_shared/MANUAL.md)** — the operator
  manual: every config field, tool, ACL form, and subsystem
- **[agents/_shared/FRAMEWORK.md](agents/_shared/FRAMEWORK.md)** — the
  framework guide injected into every agent's context
- **[docs/WINDOWS.md](docs/WINDOWS.md)** — running openflip on Windows:
  setup, credentials, launch via `start.bat`, platform differences
