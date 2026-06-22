# Local transport plugins

Drop your own messaging transports here. This directory is the **only**
supported way to add a third-party transport to openflip.

## Why here and not `main.py`?

openflip's built-in transports are registered in a dict named
`_TRANSPORT_BUILDERS` inside `openflip/main.py`. That file is **git-tracked** —
a `git pull` may overwrite it, and any edit you make there gets clobbered.

This directory (`transports_local/`) is **gitignored**, so anything you drop in
it survives every `git pull`. At startup openflip scans this directory, imports
each plugin, and **merges** it into the transport registry alongside the
built-ins. You never touch a tracked file.

> The discovery directory defaults to `transports_local/` at the repo root. To
> load plugins from somewhere else entirely, set the environment variable
> `OPENFLIP_TRANSPORTS_DIR` to an absolute path (or a path relative to the repo
> root) before launching.

## The plugin contract

Each plugin is a single `.py` file in this directory that exposes exactly two
module-level symbols:

```python
TRANSPORT_NAME = "yourname"          # str — the name agents reference in config

def build(agent):                    # build(agent) -> Transport | None
    return YourTransport(...)         # a Transport instance, or None to skip
```

- `TRANSPORT_NAME` — the string an agent puts in its `agent.json`
  (`"transports": ["yourname"]`). Must not collide with a built-in name
  (`discord`, `imessage`, `internal`, `external`) — if it does, the built-in
  wins and your plugin is ignored with a warning.
- `build(agent)` — called once per agent that requests this transport. Return a
  `Transport` instance, or `None` to skip building it for that agent (e.g. the
  agent has no config for it). The returned object must satisfy the same
  Transport protocol as the built-ins.

## Rules

- **Filenames starting with `_` are skipped** (so `_helpers.py` won't be loaded
  as a plugin). The `.template` example below is also skipped.
- **Crash-safe:** if your module throws on import, or is missing one of the two
  required symbols, it's logged via `print_ts` and skipped — it never takes
  down startup. Watch `log.txt` for the warning.
- **Built-ins can't be shadowed:** reusing `discord`/`imessage`/`internal`/
  `external` keeps the built-in and ignores your plugin.

## Working examples

The built-in transports are the canonical reference implementations — read them:

- `openflip/transports/discord.py` — a full real-world transport.
- `openflip/transports/null.py` — the minimal no-op transport (smallest
  complete example of the protocol).
- `openflip/transports/external.py` — an HTTPS/token-gated transport.

## Get started

Copy `example_transport.py.template` to `my_transport.py` in this directory,
fill in the two symbols and your Transport class, then add
`"transports": ["yourname"]` to an agent's `agent.json`.
