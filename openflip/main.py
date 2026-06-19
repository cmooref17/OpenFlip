"""Entry point: load configs, walk agents/, spawn AgentRunner for each enabled agent."""
from __future__ import annotations
import asyncio
import os
import signal
import sys
from pathlib import Path

from .agent import Agent
from .config_global import load_config, get_config
from .persistence import is_enabled, load_state, save_state
from .registry import RUNNERS, ALL_AGENTS, TOKENS
from .runtime import AgentRunner
from . import cron as cron_module
from . import restart_sentinel as _restart_sentinel
from .utils import print_ts, load_json, save_json, COLOR_YELLOW, COLOR_RED, COLOR_GREEN, COLOR_END, set_log_path, project_root, log_task_exception


_DEFAULT_AGENT_JSON = {
    "id": "",
    "display_name": "",
    "model": "anthropic/claude-sonnet-4-6",
    "provider": "anthropic",
    # Shared framework files are loaded for every agent; SOUL.md is the
    # per-agent character. Personal AGENT.md (agent-specific operational
    # supplement) loads right after SOUL.md; personal TOOLS.md loads LAST so
    # per-agent tool notes are additive to the shared _shared/TOOLS.md hygiene.
    # Both personal files are auto-created blank on discovery and contribute
    # nothing to the prompt until filled in (empty files are skipped).
    "system_files": ["SOUL.md", "_shared/FRAMEWORK.md", "AGENT.md", "_shared/TOOLS.md", "TOOLS.md"],
    # ollama_options are only honored when provider == "ollama"; the default
    # provider is anthropic (above), which ignores them. New ollama agents
    # can add this block manually if they want to tune sampling.
    "tool_response_mode": "media_only",
    "respond_to_bots": False,
    "channels": {
        "respond_in": "mentions_only",
        "ignore_channel_ids": [],
        "always_respond_channel_ids": []
    },
    "allowed_tools": []
}


_DEFAULT_SOUL_TEMPLATE = """# {display_name}

<!-- This is your SOUL.md — your personality and character. The framework
     auto-loads `_shared/FRAMEWORK.md` (operational rules) and
     `_shared/TOOLS.md` (tool hygiene), so you already know how openflip
     works and how to use tools. This file is just for who you ARE.

     Fill in each section below. The more specific you are, the less the
     agent drifts toward generic chatbot voice. Vague = bland. -->

## Identity

You are {display_name}. Replace this with one paragraph that says who you
are: name, age/role if relevant, your relationship to the operator, what
makes you recognizable. Be concrete about backstory and stance — "Casey,
the operator's long-time collaborator; treats them as a peer" beats
"a helpful assistant".

## Voice

How do you talk? Sentence length, vocabulary level, formality, slang,
emoji use, capitalization quirks, pauses, ellipses, asterisk-actions.
Give 2-3 example replies showing the texture of your speech.

## Quirks & defaults

What's distinctively *you*? Things you always do, things you never do,
running jokes, references you reach for, topics that animate you,
topics that bore you. The dials that aren't "personality traits" but
*style* — they're what readers will pick up on first.

## Boundaries

What's out of bounds for this character? Hard nos, things that would
break tone, requests that should produce a polite refusal. Be specific
— not "be safe", but "won't roleplay sexual content; redirects gently".

## Triggers & emotional state

How do you react to specific situations? Praise, criticism, being
ignored, being given a hard problem, the operator being upset. Real
characters have predictable emotional reactions; flat assistants don't.
"""


_DEFAULT_REMINDER_TEMPLATE = """Respond in accordance with your SOUL and shared framework files.

Edit this file to add short, high-priority reminders the model sees at the
end of every payload (after cached history, before each new user message).
Position-at-end means highest model attention — useful for things you keep
forgetting in the moment. Keep it tight: paid every turn, soft-warned over
~500 tokens / ~2000 chars. Delete or blank the file to disable.
"""


def _agents_dir() -> Path:
    return Path(project_root()) / "agents"


_DEFAULT_CONFIG_TEMPLATE = {
    "integrations": {
        "discord": {
            "owner_id": 0,
            "tokens": {}
        }
    },
    "ollama_host": "http://localhost:11434",
    "data_dir": "data",
    "tmp_dir": "data/tmp",
    "output_dir": "data/output",
    "embedding_model": "nomic-embed-text",
    "dry_run_tools": False,
    "display_name_map": {},
    "compaction_reserve_tokens": 20000,
    "models": {}
}


def bootstrap_files() -> None:
    """Create missing required files/dirs on first run, and migrate legacy
    config shapes forward.

    Idempotent: existing files are never overwritten in-place destructively.
    Migration rules:
      * Pre-2026-05-15 deployments had `owner_id` as a flat top-level key in
        config.json and Discord bot tokens in a separate `api_config.json`.
        On startup we move both under `integrations.discord.*` and remove
        the old keys. `api_config.json` is renamed to `api_config.json.bak`.
      * New canonical shape: `integrations.<name>.owner_id` + `.tokens`.
    """
    root = project_root()

    first_run = False

    # config.json — canonical operator config.
    cfg_path = os.path.join(root, "config.json")
    if not os.path.exists(cfg_path):
        save_json(cfg_path, _DEFAULT_CONFIG_TEMPLATE)
        first_run = True
        print_ts(f"{COLOR_YELLOW}Created default config.json — set integrations.discord.owner_id to your Discord user ID.{COLOR_END}")

    # Migrate legacy shape: top-level owner_id + api_config.json -> integrations.discord.*
    try:
        cfg = load_json(cfg_path, default={}) or {}
        migrated = False

        integrations = cfg.get("integrations")
        if not isinstance(integrations, dict):
            integrations = {}

        discord_block = integrations.get("discord")
        if not isinstance(discord_block, dict):
            discord_block = {}

        # Hoist top-level owner_id into integrations.discord.owner_id if not already set.
        legacy_owner = cfg.get("owner_id")
        if legacy_owner and not discord_block.get("owner_id"):
            discord_block["owner_id"] = int(legacy_owner)
            migrated = True
        if "owner_id" in cfg:
            cfg.pop("owner_id", None)
            migrated = True

        # Hoist api_config.json tokens into integrations.discord.tokens.
        api_path = os.path.join(root, "api_config.json")
        if os.path.exists(api_path):
            api_data = load_json(api_path, default={}) or {}
            api_tokens = api_data.get("tokens") or {}
            if isinstance(api_tokens, dict) and api_tokens:
                existing = discord_block.get("tokens") or {}
                if not isinstance(existing, dict):
                    existing = {}
                # api_config.json wins on key conflicts (it's the historical source).
                for k, v in api_tokens.items():
                    if k and v:
                        existing[k] = v
                discord_block["tokens"] = existing
                migrated = True
            # Always retire the legacy file once we've read it.
            backup_path = api_path + ".bak"
            try:
                os.replace(api_path, backup_path)
                print_ts(f"{COLOR_YELLOW}Migrated api_config.json -> integrations.discord.tokens in config.json. Old file moved to {backup_path}.{COLOR_END}")
            except OSError as e:
                print_ts(f"{COLOR_YELLOW}Could not rename api_config.json ({e}); leaving it in place.{COLOR_END}")

        if discord_block:
            integrations["discord"] = discord_block
        if integrations:
            cfg["integrations"] = integrations

        if migrated:
            save_json(cfg_path, cfg)
            print_ts(f"{COLOR_YELLOW}Migrated config.json to integrations.<name>.* shape.{COLOR_END}")
    except Exception as e:
        print_ts(f"{COLOR_YELLOW}config migration check failed (continuing with current shape): {e}{COLOR_END}")

    # agents/ — discovery walks this. Empty is fine; framework runs with no bots.
    agents_path = os.path.join(root, "agents")
    if not os.path.isdir(agents_path):
        os.makedirs(agents_path, exist_ok=True)
        print_ts(f"Created empty agents/ directory.")

    if first_run:
        print_ts(f"{COLOR_YELLOW}First-run setup detected. See SETUP.md for the full walkthrough.{COLOR_END}")

    # cron/ — cron scheduler creates jobs.json lazily, but the dir must exist.
    cron_path = os.path.join(root, "cron")
    if not os.path.isdir(cron_path):
        os.makedirs(cron_path, exist_ok=True)


def ensure_personal_files(agent_dir) -> bool:
    """Create blank personal AGENT.md + TOOLS.md if missing, and register them.

    Two responsibilities, in order:
      1. Create blank AGENT.md / TOOLS.md (0 bytes) if absent. The exists()
         guard means an existing file (blank or filled) is never touched.
      2. For files JUST created this call, if they aren't already in the
         agent's agent.json `system_files` list, insert them at the canonical
         position so they actually get injected. A file that already existed
         is left alone — we only patch the list for files we just made, so we
         never resurrect a file the operator deliberately removed from the
         list (their removal stands as long as the file is on disk).

    Blank files contribute zero prompt noise (the loader skips whitespace-only
    files), so registering an empty stub is safe — it just wires up injection
    for once the operator fills it in.

    Canonical system_files order:
        SOUL.md, _shared/FRAMEWORK.md, AGENT.md, _shared/TOOLS.md, TOOLS.md
    so AGENT.md slots right after _shared/FRAMEWORK.md and TOOLS.md appends to
    the end.

    Called from discover_agents() (startup) AND from the /reload path so a
    missing stub gets backfilled + registered on either entry point. Returns
    True if it created anything (so the reload path knows to re-read).
    """
    from pathlib import Path
    agent_dir = Path(agent_dir)
    just_created: list[str] = []
    for personal in ("AGENT.md", "TOOLS.md"):
        personal_path = agent_dir / personal
        if not personal_path.exists():
            personal_path.write_text("", encoding="utf-8")
            print_ts(f"Ensured blank {personal} for agent '{agent_dir.name}'")
            just_created.append(personal)

    if not just_created:
        return False

    # Register newly-created files in agent.json's system_files if missing.
    agent_json = agent_dir / "agent.json"
    if agent_json.exists():
        try:
            data = load_json(str(agent_json))
            sf = list(data.get("system_files") or [])
            changed = False
            for personal in just_created:
                if personal in sf:
                    continue
                if personal == "AGENT.md":
                    # Insert right after _shared/FRAMEWORK.md if present,
                    # else after SOUL.md, else append.
                    if "_shared/FRAMEWORK.md" in sf:
                        sf.insert(sf.index("_shared/FRAMEWORK.md") + 1, personal)
                    elif "SOUL.md" in sf:
                        sf.insert(sf.index("SOUL.md") + 1, personal)
                    else:
                        sf.append(personal)
                else:  # TOOLS.md — personal tool notes go last (additive).
                    sf.append(personal)
                changed = True
                print_ts(f"Registered {personal} in system_files for agent '{agent_dir.name}'")
            if changed:
                data["system_files"] = sf
                save_json(str(agent_json), data)
        except Exception as e:
            print_ts(f"{COLOR_RED}Failed to register personal files for '{agent_dir.name}': {e}{COLOR_END}", error=True)

    return True


def discover_agents() -> dict[str, Agent]:
    out: dict[str, Agent] = {}
    d = _agents_dir()
    if not d.exists():
        return out
    for agent_dir in sorted(d.iterdir()):
        if not agent_dir.is_dir():
            continue
        # Skip framework/shared dirs (anything starting with `_`). These hold
        # files that get injected into every agent — they are not agents.
        if agent_dir.name.startswith("_"):
            continue
        agent_json = agent_dir / "agent.json"
        agent_id = agent_dir.name
        if not agent_json.exists():
            # Generate a default agent.json for new agent directories.
            display_name = agent_id
            defaults = {**_DEFAULT_AGENT_JSON, "id": agent_id, "display_name": display_name}
            save_json(str(agent_json), defaults)
            # Also drop a SOUL.md stub so the agent has a character file to
            # edit. Without this a new directory boots with nothing to say.
            soul_path = agent_dir / "SOUL.md"
            if not soul_path.exists():
                soul_path.write_text(
                    _DEFAULT_SOUL_TEMPLATE.format(display_name=display_name),
                    encoding="utf-8",
                )
            # And a REMINDER.md stub. Optional surface, injected at end-of-
            # payload right before each new user turn. Empty/missing = no-op,
            # so the default text is itself the operator hint about what
            # the file is for.
            reminder_path = agent_dir / "REMINDER.md"
            if not reminder_path.exists():
                reminder_path.write_text(
                    _DEFAULT_REMINDER_TEMPLATE,
                    encoding="utf-8",
                )
            print_ts(f"Generated default agent.json + SOUL.md + REMINDER.md for '{agent_id}' — fill in {soul_path}")
        # Ensure blank personal AGENT.md + TOOLS.md for EVERY agent (not just
        # brand-new ones). Same helper is called on /reload, so a missing stub
        # is backfilled on either path.
        ensure_personal_files(agent_dir)
        try:
            a = Agent.from_file(str(agent_json))
            out[a.id] = a
        except Exception as e:
            print_ts(f"{COLOR_RED}Failed to load agent {agent_json}: {e}{COLOR_END}", error=True)
    return out


def load_tokens() -> dict[str, str]:
    """Return Discord bot tokens for all agents.

    Canonical source: `integrations.discord.tokens` in config.json. The
    config_global helper handles backward-compat lookup of the legacy
    `api_config.json` file when present.
    """
    from .config_global import get_integration_tokens
    return get_integration_tokens("discord")


def _build_discord_transport(agent: Agent) -> "DiscordTransport | None":
    """Build a DiscordTransport for an agent. Returns None if no token."""
    from .transports.discord import DiscordTransport
    token = TOKENS.get(agent.id)
    if not token:
        print_ts(
            f"{COLOR_YELLOW}No Discord token for agent '{agent.id}'; "
            f"skipping discord transport. Add it to "
            f"integrations.discord.tokens.{agent.id} in config.json.{COLOR_END}"
        )
        return None
    return DiscordTransport(token)


def _build_imessage_transport(agent: Agent) -> "IMessageTransport | None":
    """Build an IMessageTransport for an agent. Returns None on failure."""
    from .config_global import get_config
    cfg = get_config()
    imsg_cfg = (
        (cfg.get("integrations") or {})
        .get("imessage", {})
        .get("agents", {})
        .get(agent.id, {})
    )
    handle = imsg_cfg.get("handle")
    if not handle:
        print_ts(
            f"{COLOR_YELLOW}No iMessage handle for agent '{agent.id}'; "
            f"skipping imessage transport. Add "
            f"integrations.imessage.agents.{agent.id}.handle "
            f"in config.json.{COLOR_END}"
        )
        return None
    from .transports.imessage import IMessageTransport
    import os as _os
    try:
        return IMessageTransport(
            handle=handle,
            imsg_path=_os.path.expanduser(imsg_cfg.get("imsg_path", "~/.local/bin/imsg")),
            allowlist_chats=imsg_cfg.get("allowlist_chats"),
            allowlist_senders=imsg_cfg.get("allowlist_senders"),
        )
    except RuntimeError as e:
        print_ts(
            f"{COLOR_RED}IMessageTransport init failed for '{agent.id}': {e}{COLOR_END}",
            error=True,
        )
        return None


def _build_null_transport(agent: Agent) -> "NullTransport":
    """Build a NullTransport for a headless agent. Always succeeds — there's
    nothing to connect. Lets an agent with `"transports": ["internal"]` boot
    and land in RUNNERS so talk_to_agent can dispatch to it, while satisfying
    the framework's "transport is always present" invariant."""
    from .transports.null import NullTransport
    return NullTransport()


def _build_external_transport(agent: Agent) -> "ExternalTransport | None":
    """Build an ExternalTransport (authenticated HTTPS ingress) for an agent.

    Config resolution, most-specific first:
      1. agent.json `external` block (carried on the Agent dataclass so it
         survives save() round-trips).
      2. config.json `integrations.external.agents.<id>`.
      3. Built-in defaults (port 1780, cert dir / token file under the repo
         root, 120s request timeout).
    Returns None on init failure.
    """
    from .config_global import get_config
    cfg = get_config()
    global_cfg = (
        (cfg.get("integrations") or {})
        .get("external", {})
        .get("agents", {})
        .get(agent.id, {})
    )
    # agent.json block wins over the global config fallback, key by key.
    ext = dict(global_cfg)
    ext.update(agent.external or {})

    port = int(ext.get("port", 1780) or 1780)
    bind_host = str(ext.get("bind_host", "0.0.0.0") or "0.0.0.0")
    cert_dir = str(ext.get("cert_dir", "") or "")
    cert_path = str(ext.get("cert_path", "") or "")
    key_path = str(ext.get("key_path", "") or "")
    token_path = str(ext.get("token_path", "") or "")
    request_timeout = float(ext.get("request_timeout", 120.0) or 120.0)

    from .transports.external import ExternalTransport
    try:
        return ExternalTransport(
            port=port,
            bind_host=bind_host,
            cert_dir=cert_dir,
            cert_path=cert_path,
            key_path=key_path,
            token_path=token_path,
            request_timeout=request_timeout,
        )
    except Exception as e:
        print_ts(
            f"{COLOR_RED}ExternalTransport init failed for '{agent.id}': {e}{COLOR_END}",
            error=True,
        )
        return None


_TRANSPORT_BUILDERS = {
    "discord": _build_discord_transport,
    "imessage": _build_imessage_transport,
    "internal": _build_null_transport,
    "external": _build_external_transport,
}


async def start_runner(agent: Agent) -> bool:
    """Create and start a runner for an agent. Returns True if launched.

    Picks the messaging transport(s) based on agent config:
      - agent.transports (list) — multi-transport: builds one Transport per
        entry and all run concurrently in the same AgentRunner.
      - agent.transport (string, legacy) — single transport fallback when
        agent.transports is empty. 'discord' (default) or 'imessage'.

    Per-transport config lives in config.json under
    integrations.<transport>.agents.<id> (imessage) or
    integrations.discord.tokens.<id> (discord).
    """
    if agent.id in RUNNERS:
        return True

    # Resolve the list of transport names to build. Multi-transport config
    # takes priority; fall back to the legacy single-transport field.
    transport_names = list(agent.transports) if agent.transports else [
        getattr(agent, "transport", "discord") or "discord"
    ]

    transport_objects = []
    for t_name in transport_names:
        builder = _TRANSPORT_BUILDERS.get(t_name)
        if builder is None:
            print_ts(
                f"{COLOR_YELLOW}Unknown transport '{t_name}' for agent "
                f"'{agent.id}'; skipping.{COLOR_END}"
            )
            continue
        t_obj = builder(agent)
        if t_obj is not None:
            transport_objects.append(t_obj)

    if not transport_objects:
        print_ts(
            f"{COLOR_YELLOW}No transports could be built for agent "
            f"'{agent.id}'; skipping.{COLOR_END}"
        )
        return False

    # Token is the Discord bot token — needed by AgentRunner for the legacy
    # self.token field. Pass the Discord token if we have one, else empty.
    token = TOKENS.get(agent.id, "")
    runner = AgentRunner(agent, token, transports=transport_objects)

    RUNNERS[agent.id] = runner
    asyncio.create_task(runner.start())
    return True


async def stop_runner(agent_id: str) -> bool:
    runner = RUNNERS.pop(agent_id, None)
    if not runner:
        return False
    await runner.stop()
    return True


def _sync_kairos_jobs(agents: dict[str, Agent]) -> None:
    """Create/update/remove KAIROS cron jobs based on each agent's proactive config.

    Called once at startup before the cron scheduler starts. Idempotent —
    existing kairos jobs are updated to match the agent's current config;
    stale jobs for agents that no longer have proactive.enabled are removed.
    """
    from .cron import _load_jobs, _save_jobs
    data = _load_jobs()
    jobs: list[dict] = data.get("jobs") or []
    # Index existing kairos jobs by agent_id for fast lookup.
    kairos_by_agent: dict[str, int] = {}
    for i, job in enumerate(jobs):
        if (job.get("mode") or "").strip().lower() == "kairos":
            aid = job.get("agentId", "")
            if aid:
                kairos_by_agent[aid] = i
    dirty = False
    for aid, agent in agents.items():
        proactive = agent.proactive or {}
        enabled = bool(proactive.get("enabled", False))
        if enabled:
            interval_min = int(proactive.get("interval_minutes", 30) or 30)
            channel_id = int(proactive.get("channel_id", 0) or 0)
            quiet_hours = proactive.get("quiet_hours")
            if not channel_id:
                print_ts(
                    f"{COLOR_YELLOW}kairos: agent '{aid}' has proactive.enabled=true "
                    f"but no channel_id; skipping job creation{COLOR_END}"
                )
                continue
            job_id = f"kairos-{aid}"
            new_job = {
                "id": job_id,
                "agentId": aid,
                "name": f"KAIROS proactive tick for {aid}",
                "enabled": True,
                "mode": "kairos",
                "schedule": {
                    "kind": "interval",
                    "seconds": interval_min * 60,
                },
                "channelId": channel_id,
                "payload": {
                    "heartbeat": True,
                    "timeoutSeconds": 300,
                },
                "lastRunMs": 0,
            }
            if quiet_hours and isinstance(quiet_hours, dict):
                new_job["quiet_hours"] = quiet_hours
            if aid in kairos_by_agent:
                idx = kairos_by_agent[aid]
                # Preserve lastRunMs from existing job so we don't re-fire
                # immediately on config-only changes.
                new_job["lastRunMs"] = jobs[idx].get("lastRunMs", 0)
                if jobs[idx] != new_job:
                    jobs[idx] = new_job
                    dirty = True
                    print_ts(f"{COLOR_GREEN}kairos: updated job for agent '{aid}'{COLOR_END}")
            else:
                jobs.append(new_job)
                dirty = True
                print_ts(f"{COLOR_GREEN}kairos: created job for agent '{aid}'{COLOR_END}")
        else:
            # proactive disabled/absent — remove any stale kairos job.
            if aid in kairos_by_agent:
                idx = kairos_by_agent[aid]
                removed = jobs.pop(idx)
                dirty = True
                print_ts(f"{COLOR_YELLOW}kairos: removed stale job for agent '{aid}'{COLOR_END}")
                # Re-index after pop (indices shifted).
                kairos_by_agent = {
                    a: (j if j < idx else j - 1)
                    for a, j in kairos_by_agent.items()
                    if a != aid
                }
    if dirty:
        data["jobs"] = jobs
        _save_jobs(data)


async def main():
    set_log_path(os.path.join(project_root(), "log.txt"))
    # Bootstrap missing files/dirs BEFORE load_config so a fresh clone
    # doesn't crash on missing config.json. Idempotent — won't touch
    # files that already exist.
    bootstrap_files()
    load_config()
    cfg = get_config()
    from .config_global import get_owner_id
    print_ts(f"{COLOR_GREEN}openflip starting (discord owner_id={get_owner_id('discord')}){COLOR_END}")

    # ollama_api is vendored at openflip/ollama_api/. Imported as a relative
    # subpackage. config.json's `ollama_api_dir` is retained for backward
    # compat but ignored — the in-tree vendor always wins.
    from . import ollama_api
    ollama_api.set_host(cfg.get("ollama_host", "http://localhost:11434"))

    # In-place mutation so commands.py (which imports the same dicts from
    # registry) sees the populated values. Rebinding would NOT propagate.
    ALL_AGENTS.clear(); ALL_AGENTS.update(discover_agents())
    TOKENS.clear(); TOKENS.update(load_tokens())

    # Resolve empty model fields to the ollama default (only for ollama agents).
    default_model = ollama_api.config.get("default_model", "")
    for agent in ALL_AGENTS.values():
        if not agent.model and default_model and agent.provider == "ollama":
            agent.model = default_model
    state = load_state()
    # Auto-add new agents to state with enabled=true.
    dirty = False
    for aid in ALL_AGENTS:
        if aid not in state:
            state[aid] = {"enabled": True}
            dirty = True
    if dirty:
        save_state(state)

    print_ts(f"Discovered agents: {list(ALL_AGENTS.keys())}")

    for aid, agent in ALL_AGENTS.items():
        if is_enabled(aid):
            ok = await start_runner(agent)
            if ok:
                prov_info = f"provider={agent.provider}, model={agent.model}"
                print_ts(f"Launched agent '{aid}' ({prov_info})", agent=aid)
        else:
            print_ts(f"Agent '{aid}' is disabled; skipping.", agent=aid)

    if not RUNNERS:
        print_ts(f"{COLOR_YELLOW}No agents running. Add Discord bot tokens to integrations.discord.tokens in config.json, then restart. See SETUP.md.{COLOR_END}")

    # Sync KAIROS proactive cron jobs from agent.proactive config.
    # Creates/updates/removes kairos jobs in cron/jobs.json before the
    # scheduler starts, so the first tick picks up the correct state.
    try:
        _sync_kairos_jobs(ALL_AGENTS)
    except Exception as _kairos_err:
        print_ts(f"{COLOR_YELLOW}kairos job sync failed (continuing): {_kairos_err}{COLOR_END}")

    # Process any restart sentinels left by a previous gateway restart.
    _sentinel_task = asyncio.create_task(
        _restart_sentinel.process_pending(), name="restart_sentinel_processor",
    )
    _sentinel_task.add_done_callback(log_task_exception)

    # Start the cron / heartbeat scheduler.
    _cron_task = asyncio.create_task(
        cron_module.run_scheduler(), name="cron_scheduler",
    )
    _cron_task.add_done_callback(log_task_exception)

    # Graceful shutdown on SIGTERM / SIGINT. Without this, discord.py +
    # aiohttp sessions never close and systemd SIGKILLs us at 90s.
    # MUST register before start_async() — hypercorn calls add_signal_handler
    # when its shutdown_trigger is None (last-write-wins), so we pass our
    # shutdown_event.wait to keep it coordinated.
    shutdown_event = asyncio.Event()

    def _request_shutdown() -> None:
        if not shutdown_event.is_set():
            print_ts(f"{COLOR_YELLOW}Shutdown signal received; stopping runners…{COLOR_END}")
            shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except (NotImplementedError, RuntimeError):
            # add_signal_handler is unsupported on Windows; SIGINT still
            # raises KeyboardInterrupt at the asyncio.run level.
            pass

    # Start the management webapp (Quart) in-process on BIND_HOST:BIND_PORT.
    # In-process = direct access to RUNNERS / tool_settings / conversation
    # caches, so config changes take effect without restart. Pass our
    # shutdown_event.wait as hypercorn's shutdown_trigger so hypercorn
    # coordinates instead of stealing SIGTERM (see comment above).
    try:
        from . import web as _web
        asyncio.create_task(_web.app.start_async(shutdown_trigger=shutdown_event.wait))
        print_ts(f"{COLOR_GREEN}management webapp mounted on http://{_web.app.BIND_HOST}:{_web.app.BIND_PORT}{COLOR_END}")
    except Exception as _web_err:
        print_ts(f"{COLOR_YELLOW}failed to start management webapp (continuing without it): {_web_err}{COLOR_END}")

    await shutdown_event.wait()

    # Stop every running agent in parallel. stop_runner pops from RUNNERS
    # and awaits transport.stop() + cancels the runner task.
    stop_tasks = [stop_runner(aid) for aid in list(RUNNERS.keys())]
    if stop_tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*stop_tasks, return_exceptions=True), timeout=10.0)
        except asyncio.TimeoutError:
            print_ts(f"{COLOR_YELLOW}Runner stop took >10s; exiting anyway.{COLOR_END}")

    # Cancel every remaining task on the loop. asyncio.run() waits for all
    # tasks to complete before closing the loop, so background tasks
    # (cron scheduler, sentinel processor, agent runner reconnect loops,
    # hypercorn worker) keep the loop alive after main() returns. We
    # cancel them explicitly with a short await window — anything that
    # doesn't honor cancellation is abandoned. Without this, the loop
    # sits open until systemd's SIGKILL at TimeoutStopSec=90s.
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    for t in pending:
        t.cancel()
    if pending:
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            # Name the stragglers so future regressions are debuggable
            # without re-instrumenting. Each task's name comes from
            # asyncio.create_task(coro, name=...) or defaults to
            # 'Task-N'; the coro repr identifies which coroutine.
            stuck = [t for t in pending if not t.done()]
            names = ", ".join(
                f"{t.get_name()}({(t.get_coro().__qualname__ if hasattr(t.get_coro(), '__qualname__') else '?')})"
                for t in stuck
            )
            print_ts(
                f"{COLOR_YELLOW}{len(stuck)} background task(s) didn't cancel in 5s; "
                f"exiting anyway. Stuck: {names}{COLOR_END}"
            )
    print_ts(f"{COLOR_GREEN}openflip shutdown complete.{COLOR_END}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_ts("Shutting down.")
