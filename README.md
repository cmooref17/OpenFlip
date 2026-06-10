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
./start.sh
```

The full walkthrough — config fields, creating an agent (`agent.json` +
`SOUL.md`), Discord bot setup, troubleshooting — is in **[SETUP.md](SETUP.md)**.

## Documentation

- **[SETUP.md](SETUP.md)** — install and first-agent walkthrough
- **[agents/_shared/MANUAL.md](agents/_shared/MANUAL.md)** — the operator
  manual: every config field, tool, ACL form, and subsystem
- **[agents/_shared/FRAMEWORK.md](agents/_shared/FRAMEWORK.md)** — the
  framework guide injected into every agent's context
