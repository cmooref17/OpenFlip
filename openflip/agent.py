"""Agent definition. Loaded from agents/<id>/agent.json. The framework reads these fields
generically — there is no agent-specific code anywhere in the framework."""
from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass, field
from typing import Any
from .utils import load_json, save_json, project_root, print_ts


# Filenames in `system_files` that begin with this prefix resolve to the
# project-wide shared agent files (agents/_shared/) instead of the per-agent
# directory. Lets one canonical FRAMEWORK.md or TOOLS.md feed every agent.
_SHARED_PREFIX = "_shared/"


# The only provider values the framework actually routes on: "anthropic" and
# "openai" are routed via openflip/providers.py to AnthropicConversation /
# OpenAIConversation; everything else (including "ollama", the default) falls
# through to DiscordConversation/ollama.
# A typo like "antropic" or the stale "claude-cli" would therefore route
# SILENTLY to ollama — _valid_provider warns on an unknown value instead so the
# mistake surfaces at load instead of as mysterious wrong-provider behavior.
_VALID_PROVIDERS = frozenset({"ollama", "anthropic", "openai"})


def _valid_provider(raw: Any, *, agent_id: str) -> str:
    """Normalize the provider field. Unknown/blank → 'ollama' (the default
    route) with a loud warning, never a crash — a bad provider string should
    surface, not silently mis-route or brick agent load."""
    p = str(raw or "ollama").strip()
    if p in _VALID_PROVIDERS:
        return p
    print_ts(
        f"[agent] agent {agent_id!r}: unknown provider {raw!r} — falling back to "
        f"'ollama'. Valid providers: {sorted(_VALID_PROVIDERS)}."
    )
    return "ollama"


# Default "dream" (memory-consolidation) config. The block is OFF by default
# and fully optional — a missing block parses to these values, so every
# existing agent.json stays backward-compatible. `enabled` gates AUTO-fire
# only; the manual /dream command works regardless. Auto-fire is implemented
# in openflip/dream_autofire.py (end-of-turn hook in runtime._run_turn); this
# just carries the config it reads.
_DREAM_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "min_idle_minutes": 120,
    "max_memory_chars": 25000,
}


# Default "proactive" (KAIROS) config. OFF by default. When enabled, main.py
# auto-creates a kairos cron job that fires a synthetic <tick> turn at the
# configured interval. The agent decides whether to act or stay silent.
_PROACTIVE_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "interval_minutes": 30,
    "quiet_hours": None,  # e.g. {"start":"23:00","end":"08:00","timezone":"US/Mountain"}
    "channel_id": 0,
}


# Default "trigger" (inbound wake endpoint) config. OFF by default and fully
# optional — a missing block parses to these values, so every existing
# agent.json stays backward-compatible. When `enabled` is true, the webapp's
# POST /trigger/<id> endpoint may wake this agent with a synthetic turn.
# SECURITY: `allowed_tools` is the ONLY source of extra tool grants for a
# trigger turn — the HTTP caller can NOT name tools. `session` is the ONLY
# session a trigger turn may land in — the caller can NOT supply a session
# path. Both live server-side here precisely so an external caller can't
# escalate. See agents/_shared/MANUAL.md and openflip/web/app.py.
_TRIGGER_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "allowed_tools": [],          # server-side allow-path; caller can't add to it
    "session": "",                # transport-prefixed, e.g. "imessage:1"; caller can't override
    "rate_limit_per_minute": 6,   # per-agent inbound cap; 429 over the limit
    # --- identity-scoped grants (all OFF/empty by default → byte-identical to old configs) ---
    "identities": {},             # opaque caller id -> [label names]; caller asserts id only
    "labels": {},                 # label name -> [tool names]; the server-owned capability map
    "tokens": {},                 # token NAME -> {allowed_ids, allowed_id_patterns}; secret lives in store
}


def _parse_trigger(raw: Any) -> dict[str, Any]:
    """Coerce an optional `trigger` block into a normalized dict.

    None / non-dict / missing keys all fall back to _TRIGGER_DEFAULTS. Bad
    types are dropped rather than crashing agent load. `allowed_tools` is
    filtered to non-empty strings; the endpoint further intersects it against
    the agent's real tools and a hard denylist of dangerous/admin tools, so a
    typo or an over-broad list here can never grant more than the framework
    considers safe for an unattended caller.
    """
    out = dict(_TRIGGER_DEFAULTS)
    out["allowed_tools"] = []
    out["identities"] = {}
    out["labels"] = {}
    out["tokens"] = {}
    if not isinstance(raw, dict):
        return out
    out["enabled"] = bool(raw.get("enabled", False))
    tools = raw.get("allowed_tools")
    if isinstance(tools, list):
        out["allowed_tools"] = [str(t).strip() for t in tools if isinstance(t, str) and str(t).strip()]
    sess = raw.get("session")
    if isinstance(sess, str):
        out["session"] = sess.strip()
    try:
        rl = int(raw.get("rate_limit_per_minute", out["rate_limit_per_minute"]))
        if rl > 0:
            out["rate_limit_per_minute"] = rl
    except (ValueError, TypeError):
        pass
    # --- identity-scoped grants. All optional; bad shapes are dropped, never crash load.
    #     The endpoint re-validates everything downstream (id regex, real-tool
    #     intersect, denylist subtract), so a malformed map here can never grant
    #     more than the framework already considers safe.
    idents = raw.get("identities")
    if isinstance(idents, dict):
        clean_idents: dict[str, list[str]] = {}
        for ident_id, label_names in idents.items():
            if not isinstance(ident_id, str) or not ident_id.strip():
                continue
            if not isinstance(label_names, list):
                continue
            names = [str(n).strip() for n in label_names if isinstance(n, str) and str(n).strip()]
            if names:
                clean_idents[ident_id.strip()] = names
        out["identities"] = clean_idents
    labels = raw.get("labels")
    if isinstance(labels, dict):
        clean_labels: dict[str, list[str]] = {}
        for label_name, tool_names in labels.items():
            if not isinstance(label_name, str) or not label_name.strip():
                continue
            if not isinstance(tool_names, list):
                continue
            tnames = [str(t).strip() for t in tool_names if isinstance(t, str) and str(t).strip()]
            if tnames:
                clean_labels[label_name.strip()] = tnames
        out["labels"] = clean_labels
    tokens = raw.get("tokens")
    if isinstance(tokens, dict):
        clean_tokens: dict[str, dict] = {}
        for tname, entry in tokens.items():
            if not isinstance(tname, str) or not tname.strip() or not isinstance(entry, dict):
                continue
            allowed_ids = [str(i).strip() for i in entry.get("allowed_ids", [])
                           if isinstance(i, str) and str(i).strip()] \
                          if isinstance(entry.get("allowed_ids"), list) else []
            allowed_pats = [str(p).strip() for p in entry.get("allowed_id_patterns", [])
                            if isinstance(p, str) and str(p).strip()] \
                           if isinstance(entry.get("allowed_id_patterns"), list) else []
            clean_tokens[tname.strip()] = {
                "allowed_ids": allowed_ids,
                "allowed_id_patterns": allowed_pats,
            }
        out["tokens"] = clean_tokens
    return out


def _parse_proactive(raw: Any) -> dict[str, Any]:
    """Coerce an optional `proactive` block into a normalized dict.

    None / non-dict / missing keys all fall back to _PROACTIVE_DEFAULTS.
    """
    out = dict(_PROACTIVE_DEFAULTS)
    if not isinstance(raw, dict):
        return out
    out["enabled"] = bool(raw.get("enabled", out["enabled"]))
    try:
        val = int(raw.get("interval_minutes", out["interval_minutes"]))
        if val > 0:
            out["interval_minutes"] = val
    except (ValueError, TypeError):
        pass
    qh = raw.get("quiet_hours")
    if isinstance(qh, dict):
        out["quiet_hours"] = qh
    ch = raw.get("channel_id")
    if ch:
        try:
            out["channel_id"] = int(ch)
        except (ValueError, TypeError):
            pass
    return out


def _parse_dream(raw: Any) -> dict[str, Any]:
    """Coerce an optional `dream` block into a normalized dict with safe
    defaults. None / non-dict / missing keys all fall back to _DREAM_DEFAULTS.
    Bad numeric values are clamped to the default rather than crashing load.
    """
    out = dict(_DREAM_DEFAULTS)
    if not isinstance(raw, dict):
        return out
    out["enabled"] = bool(raw.get("enabled", out["enabled"]))
    for key in ("min_idle_minutes", "max_memory_chars"):
        try:
            val = int(raw.get(key, out[key]))
            if val > 0:
                out[key] = val
        except (ValueError, TypeError):
            continue
    return out


def _int_set(raw) -> set[int]:
    """Coerce an optional JSON list into a set of ints. None/empty → empty set.

    Used for the nine-list channel/category/guild routing config. Non-int-
    coercible entries are skipped so a malformed config can't crash agent load.
    """
    out: set[int] = set()
    for v in (raw or []):
        try:
            out.add(int(v))
        except (ValueError, TypeError):
            continue
    return out


@dataclass
class TransportAuth:
    """One transport's slice of a tool's auth block.

    All fields are optional. Empty/missing means "no restriction on that
    dimension" (within a single TransportAuth, inclusion dimensions are AND-ed).
    `roles` and `channels` are only meaningful on transports that have those
    concepts (Discord); on other transports they exist in the schema but are
    unused.

    Element types of `users` are transport-native: int for Discord, string
    (handle) for iMessage. The evaluator compares against the speaker's
    transport-native identity, not a normalized form.
    """
    all_users: bool = False
    users: list[Any] = field(default_factory=list)
    roles: list[int] = field(default_factory=list)
    channels: list[int] = field(default_factory=list)
    exclude_users: list[Any] = field(default_factory=list)
    exclude_roles: list[int] = field(default_factory=list)
    exclude_channels: list[int] = field(default_factory=list)


@dataclass
class ToolACL:
    """One tool entry inside `allowed_tools`.

    `auth` is a per-transport dict. A transport key missing from `auth` means
    the tool is blocked on that transport. Empty `auth` ({}) blocks the tool
    everywhere — useful for keeping a tool entry around (for inspection) while
    denying all access.

    Multiple ToolACL entries with the same `name` are OR-ed: if any rule
    passes, the tool is callable.

    `visibility_when_denied` is presentation-only — it controls whether a
    blocked tool is surfaced to the model as "exists but you can't use it"
    (`"known"`) or hidden entirely (`"hidden"`, default). Orthogonal to
    transport.
    """
    name: str
    auth: dict[str, TransportAuth] = field(default_factory=dict)
    visibility_when_denied: str = "hidden"


def _coerce_user_id(v: Any) -> Any:
    """Pass discord-style numeric IDs through as int, imessage handles as str.

    JSON only has one numeric type so a discord ID may arrive as int OR a
    string of digits. Coerce all-digit strings to int so equality matches the
    runtime's `int(sender_id)`. Non-numeric strings (iMessage handles) stay as
    strings. Booleans are forwarded as-is so we never accidentally treat
    `True` as `1` in a users list.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.lstrip("-").isdigit():
            try:
                return int(s)
            except (TypeError, ValueError):
                return s
        return s
    return v


def _parse_transport_auth(block: Any, *, agent_id: str, tool_name: str, transport: str) -> TransportAuth:
    if block is None:
        return TransportAuth()
    if not isinstance(block, dict):
        raise ValueError(
            f"agent {agent_id!r}: allowed_tools entry for {tool_name!r} has "
            f"non-object auth.{transport} block: {block!r}"
        )
    excl = block.get("exclude") or {}
    if excl and not isinstance(excl, dict):
        raise ValueError(
            f"agent {agent_id!r}: allowed_tools entry for {tool_name!r} has "
            f"non-object auth.{transport}.exclude: {excl!r}"
        )
    # NOTE: a tool ACL `all_users` is a BOOL; a path ACL `all_users` is a LIST
    # OF PATHS (see openflip/tools/files.py `_effective_allowed`). Same name,
    # different type, do not cross. Only an actual bool is honored here — a
    # non-bool (e.g. someone pasting the path idiom `all_users: ["/dir"]` into a
    # tool block) must NOT be `bool()`-coerced, because `bool([...])` is True and
    # that would silently grant the tool to EVERY user (fail-open, the dangerous
    # direction). Treat a non-bool as False and warn loudly instead.
    raw_all_users = block.get("all_users", False)
    if isinstance(raw_all_users, bool):
        all_users = raw_all_users
    else:
        all_users = False
        print_ts(
            f"[acl] agent {agent_id!r}: allowed_tools entry for {tool_name!r} "
            f"auth.{transport}.all_users must be a BOOL, got {raw_all_users!r} — "
            f"treating as false (denied). A tool ACL all_users is true/false, NOT "
            f"a list of paths (that is the path-ACL form)."
        )
    return TransportAuth(
        all_users=all_users,
        users=[_coerce_user_id(u) for u in (block.get("users") or [])],
        roles=[int(r) for r in (block.get("roles") or [])],
        channels=[int(c) for c in (block.get("channels") or [])],
        exclude_users=[_coerce_user_id(u) for u in (excl.get("users") or [])],
        exclude_roles=[int(r) for r in (excl.get("roles") or [])],
        exclude_channels=[int(c) for c in (excl.get("channels") or [])],
    )


def _parse_tool_entry(entry: Any, *, agent_id: str = "<unknown>") -> ToolACL:
    """Parse one element of `allowed_tools`.

    Hard-rejects bare strings — the silent "open to everyone" shortcut that
    pre-dated transport-aware auth. Every entry must be a dict with `name`
    and `auth`; missing `auth` means the tool is blocked everywhere.
    """
    if isinstance(entry, str):
        raise ValueError(
            f"agent {agent_id!r}: bare-string allowed_tools entry {entry!r} "
            f"is no longer supported. Replace with "
            f"`{{\"name\": \"{entry}\", \"auth\": {{\"discord\": {{\"all_users\": true}}}}}}` "
            f"(open to all discord users) or a more specific auth block. See "
            f"agents/_shared/MANUAL.md for the full auth shape."
        )
    if not isinstance(entry, dict):
        raise ValueError(
            f"agent {agent_id!r}: allowed_tools entry is not an object: {entry!r}"
        )
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"agent {agent_id!r}: allowed_tools entry missing string `name`: {entry!r}"
        )
    auth_raw = entry.get("auth")
    if auth_raw is not None and not isinstance(auth_raw, dict):
        raise ValueError(
            f"agent {agent_id!r}: allowed_tools entry for {name!r} has non-object "
            f"`auth` field: {auth_raw!r}"
        )
    auth: dict[str, TransportAuth] = {}
    for transport, block in (auth_raw or {}).items():
        if not isinstance(transport, str):
            raise ValueError(
                f"agent {agent_id!r}: allowed_tools entry for {name!r} has "
                f"non-string transport key in `auth`: {transport!r}"
            )
        auth[transport] = _parse_transport_auth(
            block, agent_id=agent_id, tool_name=name, transport=transport,
        )
    return ToolACL(
        name=name,
        auth=auth,
        visibility_when_denied=entry.get("visibility_when_denied", "hidden"),
    )


def _resolve_system_file(agent_dir: str, fname: str) -> str:
    """Resolve one entry from `system_files` to an absolute path.

    Names beginning with `_shared/` resolve under `agents/_shared/`; everything
    else resolves relative to the agent's own directory (legacy behavior).
    """
    if fname.startswith(_SHARED_PREFIX):
        rel = fname[len(_SHARED_PREFIX):]
        return os.path.join(project_root(), "agents", "_shared", rel)
    return os.path.join(agent_dir, fname)


def _apply_template_vars(content: str, *, agent_id: str, agent_dir: str, display_name: str) -> str:
    """Replace `{agent_id}`, `{agent_dir}`, `{display_name}` in shared files.

    Only useful for `_shared/` content where the same file feeds every agent
    but needs to render with that agent's identity. Per-agent files normally
    hardcode their own values, so substitution is a no-op there.
    """
    if "{" not in content:
        return content
    return (content
            .replace("{agent_id}", agent_id)
            .replace("{agent_dir}", agent_dir)
            .replace("{display_name}", display_name))


def _load_system_files(
    agent_dir: str,
    filenames: list[str],
    *,
    agent_id: str = "",
    display_name: str = "",
) -> str:
    parts = []
    # Phase 7.2 (ISSUES.md 7.2): always include the project-level CLAUDE.md
    # from openflip root as read-only context. Provides project rules and
    # architecture overview to every agent without per-agent configuration.
    # Missing file is fine — silently skipped (don't force users to have one).
    project_claude_md = os.path.join(project_root(), "CLAUDE.md")
    try:
        with open(project_claude_md, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            parts.append(_apply_template_vars(
                content,
                agent_id=agent_id,
                agent_dir=agent_dir,
                display_name=display_name,
            ))
    except (FileNotFoundError, OSError):
        pass
    for fname in filenames:
        fpath = _resolve_system_file(agent_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                parts.append(_apply_template_vars(
                    content,
                    agent_id=agent_id,
                    agent_dir=agent_dir,
                    display_name=display_name,
                ))
        except (FileNotFoundError, OSError):
            pass
    return "\n\n".join(parts)


def _get_fingerprint(path: str, agent_dir: str, system_files: list[str]) -> str:
    """SHA256 fingerprint over agent.json + all system_files (per-agent + shared).

    Used to detect when the assembled system prompt would actually differ —
    byte-accurate, unlike mtime which lies on `touch` or identical-rewrite saves.
    Cost: ~1ms for the 5–10 small text files an agent loads.

    Includes the project-level CLAUDE.md too (loaded by _load_system_files).
    Missing files contribute their path with empty content (so deleting a
    file changes the fingerprint).
    """
    h = hashlib.sha256()
    files = [path, os.path.join(project_root(), "CLAUDE.md")]
    for fname in system_files:
        files.append(_resolve_system_file(agent_dir, fname))
    for fpath in files:
        h.update(fpath.encode("utf-8"))
        h.update(b"\0")
        try:
            with open(fpath, "rb") as f:
                h.update(f.read())
        except (FileNotFoundError, OSError):
            pass
        h.update(b"\0")
    return h.hexdigest()


@dataclass
class Agent:
    id: str
    path: str
    display_name: str
    model: str
    system_message: str
    system_files: list[str]
    ollama_options: dict
    allowed_tools: list[ToolACL]
    tool_response_mode: str
    respond_in: str
    ignore_channel_ids: set[int]
    always_respond_channel_ids: set[int]
    # DM allowlist — set of user IDs allowed to DM this agent. Empty (default)
    # = ANYONE who can DM the bot gets through (current/legacy behavior). When
    # populated, only listed user IDs receive responses on the DM path. The
    # bot owner is always implicitly allowed regardless of this list — set
    # via pipeline.should_respond, not here. Useful for locking an agent to
    # owner-only DMs while still letting it speak in approved channels.
    dm_allowlist_user_ids: set[int]
    # Guild (server) whitelist — set of Discord guild IDs this agent may
    # respond in. Empty (default) = ALL guilds allowed (current/legacy
    # behavior). When populated, guild messages from guilds NOT in this set
    # are ignored. DMs are never affected (they have no guild). Checked in
    # pipeline.should_respond.
    guild_whitelist: set[int]
    # ── Nine-list channel/category/guild routing (supersedes respond_in) ──
    # All empty by default. When ALL nine are empty, should_respond falls back
    # to the legacy respond_in / ignore_channel_ids / always_respond_channel_ids
    # path, so existing agents behave byte-identically. Precedence (first match
    # wins) is implemented in pipeline.should_respond:
    #   IGNORE (hard deny) > NO-MENTION (respond always) > RESPOND (if mentioned).
    # Each tier checks guild_id, then channel_id, then category_id.
    respond_guilds: set[int]
    respond_channels: set[int]
    respond_categories: set[int]
    respond_no_mention_guilds: set[int]
    respond_no_mention_channels: set[int]
    respond_no_mention_categories: set[int]
    ignore_guilds: set[int]
    ignore_channels: set[int]
    ignore_categories: set[int]
    respond_to_bots: bool
    memory_enabled: bool = True
    # Memory-consolidation ("dream") config. OFF by default. Keys:
    # enabled (gates AUTO-fire only), min_idle_minutes, max_memory_chars.
    # Parsed via _parse_dream so a missing block is fully backward-compatible.
    # When enabled, auto-fire is driven by openflip/dream_autofire.py via an
    # end-of-turn hook in runtime._run_turn. Manual /dream + the dream() tool
    # work regardless of `enabled`.
    dream: dict = field(default_factory=lambda: dict(_DREAM_DEFAULTS))
    # Proactive (KAIROS) config. OFF by default. Keys: enabled, interval_minutes,
    # quiet_hours, channel_id. When enabled, main.py auto-creates/syncs a kairos
    # cron job for this agent.
    proactive: dict = field(default_factory=lambda: dict(_PROACTIVE_DEFAULTS))
    # Inbound trigger config. OFF by default. Keys: enabled, allowed_tools,
    # session, rate_limit_per_minute. Parsed via _parse_trigger so a missing
    # block is fully backward-compatible. Consumed by the webapp's
    # POST /trigger/<id> endpoint — see openflip/web/app.py. The grantable
    # tools and target session are fixed HERE (server-side); the HTTP caller
    # supplies only a prompt + optional context, never tools or a session.
    trigger: dict = field(default_factory=lambda: {**_TRIGGER_DEFAULTS, "allowed_tools": []})
    provider: str = "ollama"
    think: bool | None = None
    # Path ACLs accept EITHER a flat list (historical form, applies to
    # everyone) OR an opt-in dict that is TRANSPORT-KEYED, structurally
    # identical to a tool's `auth` block: {"discord": {"users": {"<id>": [...]},
    # "all_users": [...]}, "imessage": {...}}. The current speaker's transport
    # block is selected first (like acl.py's acl.auth.get(transport)), then
    # `users`/`all_users` are resolved inside it; the owner is just an id under
    # `users`, no magic owner key. The form is stored as-is with no coercion;
    # per-user resolution happens at access time in
    # tools/files.py:_effective_allowed. denied_paths stays a flat list
    # (unconditional, checked first) — per-user deny is intentionally not v1.
    allowed_read_paths: list | dict = field(default_factory=list)
    allowed_write_paths: list | dict = field(default_factory=list)
    denied_paths: list[str] = field(default_factory=list)
    # Slash commands that are agent-specific — registered ONLY on this agent's
    # bot, not on every agent's bot. Names must match the @bot.slash_command
    # name= argument inside commands.py's gated blocks. Empty list = no
    # agent-specific commands.
    agent_specific_commands: list[str] = field(default_factory=list)
    # Messaging transport — 'discord' (default) or 'imessage'. Selected at
    # AgentRunner instantiation time (see main.start_runner). Per-transport
    # config lives in config.json under integrations.<transport>.agents.<id>.
    transport: str = "discord"
    # Multi-transport: optional list of transport names to listen on
    # simultaneously (e.g. ["discord", "imessage"]). When populated, the
    # agent spawns one Transport per entry and all run concurrently in the
    # same AgentRunner. Empty list (default) = fall back to the single
    # `transport` field above (backward compat).
    transports: list[str] = field(default_factory=list)
    # Per-agent config block for the "external" HTTPS transport (port, cert/key
    # paths, token-file path, request timeout). Read by main.py's
    # _build_external_transport. Kept as an opaque dict so the transport owns
    # its own schema; carried as a real field (not a stray agent.json key) so it
    # survives save() re-serialization round-trips (save() rebuilds data from
    # known fields only and would otherwise drop an unknown top-level key).
    external: dict = field(default_factory=dict)
    _fingerprint: str = ""

    @classmethod
    def from_file(cls, path: str) -> "Agent":
        data = load_json(path)
        if not data:
            raise ValueError(f"Agent file is empty or unreadable: {path}")
        agent_dir = os.path.dirname(path)
        agent_id = data.get("id") or os.path.basename(agent_dir)
        display_name = data.get("display_name", agent_id)
        system_files = data.get("system_files", []) or []
        sys_msg = _load_system_files(
            agent_dir,
            system_files,
            agent_id=agent_id,
            display_name=display_name,
        )
        channels = data.get("channels", {}) or {}
        allowed = [_parse_tool_entry(e, agent_id=agent_id) for e in (data.get("allowed_tools") or [])]
        # Auto-inject blank-auth entries for every tool in TOOL_REGISTRY that
        # the agent.json didn't list. Empty `auth` blocks the tool by default
        # — the owner (and anyone else) sees it surfaced in /agents,
        # /grant autocomplete, and discovery tooling. Explicit > implicit:
        # every agent has a complete picture of what the framework can do.
        # If TOOL_REGISTRY is empty at load time (tests, partial init), this
        # no-ops gracefully.
        try:
            from .tools._base import TOOL_REGISTRY
            existing_names = {a.name for a in allowed}
            for tool_name in TOOL_REGISTRY:
                if tool_name not in existing_names:
                    allowed.append(ToolACL(name=tool_name))
        except Exception:
            # Tool registry not importable yet (early-init path). Skip the
            # auto-inject; the agent loads with only its explicitly-configured
            # tools. Better to load partial than fail entirely.
            pass
        # Path ACLs — default to agent's own directory if not specified.
        # Use ["*"] to allow everything. A field may be EITHER a flat list
        # (applies to everyone — historical form) OR a dict keyed by speaker
        # identity using the tool-ACL vocabulary (users/all_users — see
        # _effective_allowed). Both forms are stored verbatim with NO coercion;
        # the dict is resolved per-user only at access time, so flat-list
        # agents are byte-for-byte unchanged.
        default_paths = [agent_dir]
        read_paths = data.get("allowed_read_paths", None)
        write_paths = data.get("allowed_write_paths", None)
        if read_paths is None:
            read_paths = default_paths
        if write_paths is None:
            write_paths = default_paths
        denied = data.get("denied_paths", []) or []

        fingerprint = _get_fingerprint(path, agent_dir, system_files)

        model = data.get("model", "") or ""

        return cls(
            id=agent_id,
            path=path,
            display_name=display_name,
            model=model,
            system_message=sys_msg,
            system_files=system_files,
            ollama_options=data.get("ollama_options") or data.get("options") or {},
            allowed_tools=allowed,
            tool_response_mode=data.get("tool_response_mode", "media_only"),
            respond_in=channels.get("respond_in", "mentions_only"),
            ignore_channel_ids=set(channels.get("ignore_channel_ids", []) or []),
            always_respond_channel_ids=set(channels.get("always_respond_channel_ids", []) or []),
            dm_allowlist_user_ids=set(channels.get("dm_allowlist_user_ids", []) or []),
            guild_whitelist=set(int(g) for g in (channels.get("guild_whitelist", []) or [])),
            respond_guilds=_int_set(channels.get("respond_guilds")),
            respond_channels=_int_set(channels.get("respond_channels")),
            respond_categories=_int_set(channels.get("respond_categories")),
            respond_no_mention_guilds=_int_set(channels.get("respond_no_mention_guilds")),
            respond_no_mention_channels=_int_set(channels.get("respond_no_mention_channels")),
            respond_no_mention_categories=_int_set(channels.get("respond_no_mention_categories")),
            ignore_guilds=_int_set(channels.get("ignore_guilds")),
            ignore_channels=_int_set(channels.get("ignore_channels")),
            ignore_categories=_int_set(channels.get("ignore_categories")),
            respond_to_bots=bool(data.get("respond_to_bots", False)),
            memory_enabled=bool(data.get("memory_enabled", True)),
            dream=_parse_dream(data.get("dream")),
            proactive=_parse_proactive(data.get("proactive")),
            trigger=_parse_trigger(data.get("trigger")),
            provider=_valid_provider(data.get("provider"), agent_id=agent_id),
            think=data.get("think", None),
            allowed_read_paths=read_paths,
            allowed_write_paths=write_paths,
            denied_paths=denied,
            agent_specific_commands=list(data.get("agent_specific_commands", []) or []),
            transport=str(data.get("transport", "discord")),
            transports=list(data.get("transports", []) or []),
            external=dict(data.get("external", {}) or {}),
            _fingerprint=fingerprint,
        )

    def reload(self) -> None:
        fresh = Agent.from_file(self.path)
        for f in fresh.__dataclass_fields__:
            setattr(self, f, getattr(fresh, f))

    def reload_if_changed(self) -> bool:
        """Reload if agent.json or any system_file content actually changed.

        Hash-based, not mtime-based — so `touch` and identical-content rewrites
        don't trigger spurious rebuilds (which would bust the prompt cache for
        no real reason). Returns True if reloaded.
        """
        agent_dir = os.path.dirname(self.path)
        current = _get_fingerprint(self.path, agent_dir, self.system_files)
        if current != self._fingerprint:
            self.reload()
            return True
        return False

    def save(self) -> bool:
        data = {
            "id": self.id,
            "display_name": self.display_name,
            "model": self.model,
            "provider": self.provider,
            "system_files": self.system_files,
            "ollama_options": self.ollama_options,
            "tool_response_mode": self.tool_response_mode,
            "respond_to_bots": self.respond_to_bots,
            "memory_enabled": self.memory_enabled,
            "channels": {
                "respond_in": self.respond_in,
                "ignore_channel_ids": sorted(self.ignore_channel_ids),
                "always_respond_channel_ids": sorted(self.always_respond_channel_ids),
                # Only serialize dm_allowlist_user_ids when populated — keeps
                # existing agent.json files unchanged on disk for agents that
                # never set it (current/legacy "anyone can DM" behavior).
                **({"dm_allowlist_user_ids": sorted(self.dm_allowlist_user_ids)}
                   if self.dm_allowlist_user_ids else {}),
                # Only serialize guild_whitelist when populated — agents that
                # never set it keep their agent.json unchanged (all guilds).
                **({"guild_whitelist": sorted(self.guild_whitelist)}
                   if self.guild_whitelist else {}),
                # Nine-list routing — each serialized ONLY when populated, so
                # legacy agents that use none of them keep their agent.json
                # byte-identical on disk.
                **({"respond_guilds": sorted(self.respond_guilds)}
                   if self.respond_guilds else {}),
                **({"respond_channels": sorted(self.respond_channels)}
                   if self.respond_channels else {}),
                **({"respond_categories": sorted(self.respond_categories)}
                   if self.respond_categories else {}),
                **({"respond_no_mention_guilds": sorted(self.respond_no_mention_guilds)}
                   if self.respond_no_mention_guilds else {}),
                **({"respond_no_mention_channels": sorted(self.respond_no_mention_channels)}
                   if self.respond_no_mention_channels else {}),
                **({"respond_no_mention_categories": sorted(self.respond_no_mention_categories)}
                   if self.respond_no_mention_categories else {}),
                **({"ignore_guilds": sorted(self.ignore_guilds)}
                   if self.ignore_guilds else {}),
                **({"ignore_channels": sorted(self.ignore_channels)}
                   if self.ignore_channels else {}),
                **({"ignore_categories": sorted(self.ignore_categories)}
                   if self.ignore_categories else {}),
            },
            # Only serialize entries with non-empty auth — the framework
            # auto-injects blank entries at load time for every registered
            # tool, but we don't want to write all 30+ to disk each save
            # (that would defeat the auto-inject point — explicit empty
            # entries on disk are indistinguishable from auto-injected ones).
            "allowed_tools": [self._serialize_acl(a) for a in self.allowed_tools if a.auth],
        }
        if self.think is not None:
            data["think"] = self.think
        # Only serialize the dream block when it differs from the OFF-by-default
        # baseline, so agents that never opted in keep their agent.json
        # byte-identical on disk.
        if self.dream and self.dream != _DREAM_DEFAULTS:
            data["dream"] = dict(self.dream)
        # Only serialize the proactive block when it differs from OFF-by-default.
        if self.proactive and self.proactive != _PROACTIVE_DEFAULTS:
            data["proactive"] = dict(self.proactive)
        # Only serialize the trigger block when it differs from OFF-by-default,
        # so agents that never opted in keep their agent.json byte-identical.
        if self.trigger and self.trigger != _TRIGGER_DEFAULTS:
            data["trigger"] = dict(self.trigger)
        if self.allowed_read_paths:
            data["allowed_read_paths"] = self.allowed_read_paths
        if self.allowed_write_paths:
            data["allowed_write_paths"] = self.allowed_write_paths
        if self.denied_paths:
            data["denied_paths"] = self.denied_paths
        if self.agent_specific_commands:
            data["agent_specific_commands"] = list(self.agent_specific_commands)
        # Only serialize transport if non-default — keeps existing agent.json
        # files unchanged on disk for Discord-only deployments.
        if self.transport and self.transport != "discord":
            data["transport"] = self.transport
        # Multi-transport list — only serialize when populated. Empty list
        # means "use legacy single transport field" so existing agent.json
        # files stay byte-identical.
        if self.transports:
            data["transports"] = list(self.transports)
        # external transport config — only serialize when populated so agents
        # without the transport keep byte-identical agent.json files.
        if self.external:
            data["external"] = dict(self.external)
        return save_json(self.path, data)

    @staticmethod
    def _serialize_acl(acl: ToolACL) -> Any:
        out: dict[str, Any] = {"name": acl.name}
        auth_out: dict[str, Any] = {}
        for transport, tauth in acl.auth.items():
            block: dict[str, Any] = {}
            if tauth.all_users:
                block["all_users"] = True
            if tauth.users:
                # Users may be int (discord) or str (imessage handle). Sort
                # mixed-type lists by their stringified form so JSON output is
                # deterministic across runs.
                block["users"] = sorted(tauth.users, key=lambda v: (not isinstance(v, int), str(v)))
            if tauth.roles:
                block["roles"] = sorted(tauth.roles)
            if tauth.channels:
                block["channels"] = sorted(tauth.channels)
            excl: dict[str, Any] = {}
            if tauth.exclude_users:
                excl["users"] = sorted(tauth.exclude_users, key=lambda v: (not isinstance(v, int), str(v)))
            if tauth.exclude_roles:
                excl["roles"] = sorted(tauth.exclude_roles)
            if tauth.exclude_channels:
                excl["channels"] = sorted(tauth.exclude_channels)
            if excl:
                block["exclude"] = excl
            auth_out[transport] = block
        out["auth"] = auth_out
        if acl.visibility_when_denied != "hidden":
            out["visibility_when_denied"] = acl.visibility_when_denied
        return out
