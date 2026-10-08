"""Transport-agnostic session, attachment, and inbound-message types.

Introduced as part of the Discord-decouple refactor (TODO #2). Phase 1 wraps
nextcord types at the on_message boundary so the rest of openflip can stop
depending on Discord internals. Phase 2 extracts Transport adapters.

A Session is one conversation context. For Discord that's a channel/DM; for
other transports it's whatever the native identity unit is. The conversation
file key is `conversation_id` which is already prefixed (e.g. `"discord:12345"`).

Tools never touch nextcord. They receive `InboundMessage` and Session info via
contextvars or explicit parameters and route outbound traffic through the
Transport adapter (Phase 2).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkerOutcome:
    """Result handed back through a `run_synthetic_turn(result_future=...)`.

    Subagent workers settle their future with one of these instead of bare
    text so the spawner can judge success structurally — on tool-outcome
    COUNTS, never by matching strings in the reply. `calls` is how many tool
    calls the worker's turn processed, `errors` how many of those failed
    (ToolResult not ok — includes ACL denials), `denied` the subset that were
    ACL-denied. A turn that called no tools settles with all-zero counts.
    """
    text: str = ""
    calls: int = 0
    errors: int = 0
    denied: int = 0


@dataclass
class Session:
    """One conversation context, transport-agnostic.

    For Discord: transport="discord", transport_id=str(channel_id).
    For future transports: transport_id is whatever string identifier that
    transport uses natively.
    """
    transport: str               # "discord", "imessage", "slack", etc.
    transport_id: str            # platform-native ID as a string
    conversation_id: str         # prefixed: f"{transport}:{transport_id}"
    speaker_id: int              # always int; non-discord transports hash native ID to int
    speaker_role_ids: list[int]  # transport-native role IDs; empty for transports without roles
    is_owner: bool               # speaker == openflip owner_id
    is_dm: bool                  # 1:1 conversation vs group/channel
    display_name: str            # human-readable label for logs/UI
    guild_id: int = 0            # Discord guild (server) id; 0 for DMs and non-guild transports
    category_id: int = 0         # Discord channel category id; 0 for DMs, uncategorized channels, non-discord
    handle: str = ""             # raw sender handle for handle-based transports (iMessage email/phone);
                                 # "" for Discord. Source of truth for owner/admin auth on those
                                 # transports — never the per-process-unstable hash in speaker_id.
    tool_grants: list[str] = field(default_factory=list)
    # Tool names this session is authorized to call REGARDLESS of per-user auth.
    # Used for trusted synthetic sessions (cron jobs) that have no human speaker.
    # Empty = no extra grants; normal per-user ACL applies. A tool name here makes
    # that tool callable for this session even if no auth block matches. Additive
    # only — it never overrides an exclude deny, and can't authorize a tool that
    # isn't present in the agent's allowed_tools at all.
    tool_allowlist: Optional[list[str]] = None
    # RESTRICTIVE intersection ceiling — the exact opposite of tool_grants.
    # tool_grants WIDENS (adds callability); tool_allowlist NARROWS (removes it).
    # Semantics (fail-closed, internet-facing — see ExternalTransport):
    #   None          → no narrowing. The session gets its full per-transport ACL
    #                   ceiling unchanged. This is the value for EVERY Discord /
    #                   iMessage / cron / internal session — nothing changes for
    #                   them.
    #   [] (empty)    → narrows to NOTHING. No tools are callable (chat only).
    #                   Meaningfully distinct from None: "present but empty" is a
    #                   deliberate deny-all, not "unset".
    #   ["a", "b"]    → the callable set becomes the INTERSECTION of the ACL
    #                   ceiling and this list. A name here that the ceiling
    #                   doesn't already allow is still blocked — the list can only
    #                   restrict, never widen. It cannot conjure a tool the
    #                   transport's auth block denies.
    # Used by the `external` transport so an operator can issue a token that is
    # narrower than the agent's `auth.external` ceiling (per-token least
    # privilege). Threaded Session → build_visible_tools → evaluate_tools_for_
    # speaker, parallel to tool_grants but applied as a ceiling, not a grant.
    read_scope: Optional[list[str]] = None
    # Independent FILE-READ scope for this session, overriding the agent's
    # transport-keyed `allowed_read_paths` entirely (files.py `_effective_allowed`).
    # Semantics (fail-closed):
    #   None        → no override. The session resolves path ACLs normally (the
    #                 value for EVERY Discord / iMessage / cron / external
    #                 session — nothing changes for them).
    #   [] (empty)  → read scope is empty; downstream default-deny / read-
    #                 fallback (agent dir + system temp) applies, same as an
    #                 agent with no allowed_read_paths configured.
    #   [dir, …]    → reads are confined to exactly these dirs.
    # Writes are governed separately by `write_scope` below (a worker with a
    # read_scope but no write_scope cannot write anywhere). `denied_paths`
    # still wins (checked first in _check_access).
    # Set by `make_subagent_session` so an ephemeral worker (transport
    # "internal", whose agent may only have a "discord" path block) can read the
    # project dirs the owner scopes it to via spawn_subagents `read_paths`.
    write_scope: Optional[list[str]] = None
    # Independent FILE-WRITE scope for this session, the write-side twin of
    # `read_scope` (files.py `_effective_allowed`). Semantics (fail-closed):
    #   None        → no override on its own. For a NON-worker session (both
    #                 scopes None) path ACLs resolve normally. For a worker
    #                 (read_scope set, write_scope None) writes resolve to []
    #                 → write_file / edit_file are denied everywhere.
    #   [] (empty)  → writes denied (no write fallback exists; an empty write
    #                 allow list is a default-deny).
    #   [dir, …]    → write_file / edit_file are confined to exactly these
    #                 dirs. delete_file / restore_snapshot stay in
    #                 SUBAGENT_DENIED_TOOLS regardless of this scope.
    # `denied_paths` still wins. Set by `make_subagent_session` from the
    # spawn_subagents `write_paths` setting.

    @property
    def channel_id_int(self) -> int:
        """Best-effort int conversion of transport_id, for legacy callsites.

        Phase 1 compat helper. New code should use transport_id directly.
        Raises ValueError if transport_id isn't numeric.
        """
        return int(self.transport_id)


@dataclass
class Attachment:
    """Transport-neutral attachment. Either url OR local_path is set.

    Discord transports populate `url` from the CDN. Local-file transports
    populate `local_path`. Consumers that need bytes call `read_bytes()`
    or fetch the URL.
    """
    content_type: str
    filename: str
    url: Optional[str] = None
    local_path: Optional[str] = None


@dataclass
class InboundMessage:
    """Transport-neutral inbound message.

    Built at the transport-adapter boundary (Phase 2) or at the
    `_handle_message` boundary in runtime.py (Phase 1) by wrapping a
    `nextcord.Message`.

    `reply_to` is optional and currently only populated for Discord replies.
    """
    session: Session
    text: str
    sender_id: int
    sender_display_name: str
    is_dm: bool
    mentions_us: bool
    sender_is_bot: bool = False
    attachments: list[Attachment] = field(default_factory=list)
    reply_to: Optional["InboundMessage"] = None


def make_discord_session(
    *,
    channel_id: int,
    speaker_id: int,
    speaker_role_ids: list[int],
    is_owner: bool,
    is_dm: bool,
    display_name: str,
    guild_id: int = 0,
    category_id: int = 0,
) -> Session:
    """Build a Session for a Discord channel.

    Centralizes the prefix convention so we don't recompute
    `f"discord:{channel_id}"` everywhere. `guild_id` and `category_id` are
    0 for DMs (and `category_id` is also 0 for uncategorized channels).

    Identity links: if this is a DM and the speaker is in config.json's
    `identity_links` ("discord:<user_id>" → canonical), conversation_id is
    rewritten to the forwarded primary conversation_id (e.g. "imessage:+1555") so the conversation history is shared
    with the same person's linked sessions on other transports. DM-only:
    guild channels are a shared space keyed by channel, not by speaker —
    rewriting them would split one channel's history across speakers.
    Routing (transport/transport_id) and auth inputs (speaker_id, is_owner,
    roles) are untouched — the link affects conversation identity ONLY.
    """
    conversation_id = f"discord:{channel_id}"
    if is_dm:
        from .config_global import resolve_linked_conversation_id
        linked = resolve_linked_conversation_id("discord", speaker_id)
        if linked:
            conversation_id = linked
    return Session(
        transport="discord",
        transport_id=str(channel_id),
        conversation_id=conversation_id,
        speaker_id=speaker_id,
        speaker_role_ids=speaker_role_ids,
        is_owner=is_owner,
        is_dm=is_dm,
        display_name=display_name,
        guild_id=guild_id,
        category_id=category_id,
    )


def make_external_session(
    name: str,
    *,
    speaker_label: str = "",
    tool_allowlist: Optional[list[str]] = None,
) -> Session:
    """Build a Session for an `external` transport turn.

    Mirrors `make_internal_session` (null.py): the conversation is keyed by a
    fixed NAME chosen by the operator (the token's `session` field), NOT by
    anything the external caller can influence. All turns bound to one token
    land in `external:<name>` so a game keeps one continuous conversation.

    `name` becomes the `transport_id` verbatim (so the transport's `send()` —
    which receives `session.transport_id` — can route the captured reply back
    to the in-flight HTTP request keyed by it) and the `conversation_id` is
    `external:<name>` (so history lives in
    `agents/<id>/conversations/external:<name>.jsonl`, isolated from every
    Discord/iMessage/internal conversation). The in-memory dict key the runtime
    derives (`TransportChannel.id`) is a per-process-stable hash of `name`
    (channel_shim.py) — only ever an in-memory key; the on-disk filename is
    governed by `conversation_id`, which IS stable across restarts.

    SECURITY: `is_owner` is hard-False and `speaker_id` is a non-owner hash —
    an external caller can NEVER be the owner (acl.is_owner stays False), so
    owner-only tools/disclosure never unlock on this transport. The numeric
    `speaker_id` is only an internal keying value; `external` ACL auth is keyed
    by transport name, so a tool needs an explicit `auth.external` block to be
    callable here (fail-closed: Discord-only tool entries are invisible).

    `tool_allowlist` (the token's optional `allowed_tools`) NARROWS the
    `auth.external` ceiling for this turn — None = no narrowing (full ceiling),
    [] = no tools, [names…] = intersection with the ceiling. See
    Session.tool_allowlist for the full semantics. It can only restrict, never
    grant, so it is always safe to thread from an untrusted token.
    """
    speaker_id = abs(hash(f"external:{name}")) % (2**31)
    return Session(
        transport="external",
        transport_id=name,
        conversation_id=f"external:{name}",
        speaker_id=speaker_id,
        speaker_role_ids=[],
        is_owner=False,
        is_dm=True,
        display_name=speaker_label or f"external:{name}",
        handle="",
        tool_allowlist=tool_allowlist,
    )


def make_subagent_session(
    agent_id: str,
    task_uuid: str,
    grants: list[str],
    read_scope: Optional[list[str]] = None,
    write_scope: Optional[list[str]] = None,
) -> Session:
    """Build a Session for one ephemeral subagent/worker task.

    Subagent foundation: an
    orchestrating agent hands a scoped sub-task to a throwaway worker turn
    that runs in its OWN conversation (`internal:subagent-<task_uuid>` — its
    history lands in `agents/<id>/conversations/internal:subagent-<uuid>.jsonl`,
    isolated from every human conversation) and returns only its final text.

    Modeled on `make_external_session`:
      - `is_owner` is hard-False and `speaker_id` is a NON-OWNER hash — a
        worker can never unlock owner-only tools or disclosure.
      - `speaker_id` is derived from `task_uuid`, so two parallel workers get
        DISTINCT speaker ids and the tool_executor `_inflight` lock — keyed on
        (agent, speaker, tool) — never serializes one worker behind another.
        It is only an in-memory keying value; nothing on disk depends on it.
      - `tool_grants` == `tool_allowlist` == a copy of `grants`: the grants
        make the worker's tools callable without a human auth match (the
        session has no human), and the identical allowlist CEILS it to exactly
        that set — a worker can't reach anything outside what the caller
        handed it. The caller is responsible for intersecting `grants` down
        from its own privileges and for stripping always-denied tools.
      - `read_scope` is stored as an INDEPENDENT copy (so a later mutation of
        the caller's list can't reach into the worker's session). None = the
        worker resolves path ACLs normally; a list (possibly empty) confines
        its file reads per Session.read_scope. See the spawn_subagents
        `read_paths` setting.
      - `write_scope` is likewise an INDEPENDENT copy. None or [] = the
        worker cannot write anywhere (write_file / edit_file denied); a
        non-empty list confines its writes to exactly those dirs per
        Session.write_scope. See the spawn_subagents `write_paths` setting.
        delete_file / restore_snapshot are hard-denied for workers whatever
        this says (SUBAGENT_DENIED_TOOLS).
    """
    speaker_id = abs(hash(f"internal:subagent:{agent_id}:{task_uuid}")) % (2**31)
    return Session(
        transport="internal",
        transport_id=f"subagent-{task_uuid}",
        conversation_id=f"internal:subagent-{task_uuid}",
        speaker_id=speaker_id,
        speaker_role_ids=[],
        is_owner=False,
        is_dm=True,
        display_name=f"{agent_id} subagent {task_uuid[:8]}",
        handle="",
        tool_grants=list(grants),
        tool_allowlist=list(grants),
        read_scope=None if read_scope is None else list(read_scope),
        write_scope=None if write_scope is None else list(write_scope),
    )
