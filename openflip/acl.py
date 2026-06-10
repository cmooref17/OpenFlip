"""ACL evaluation. Per-message, decides which tools a speaker can trigger and how denied tools
should be presented to the model (hidden vs known).

The auth shape (see agents/_shared/MANUAL.md for the operator-facing version):

  Each ToolACL has a transport-keyed `auth` dict (typically `{"discord": {...},
  "imessage": {...}}`). Within one auth block, inclusion dimensions are AND-ed
  (`users` AND `roles` AND `channels` all must match if set; an empty list /
  missing field is no-restriction on that dimension). Multiple ToolACL entries
  with the same `name` are OR-ed — any rule passing means the tool is callable.
  `exclude` always wins: a sender matching anything in `exclude.{users,roles,
  channels}` is blocked regardless of inclusions.

  Owner-bypass is gone — the owner's user ID lives in `users` lists like
  anyone else. `is_owner()` still exists for slash-command gating, but it no
  longer short-circuits ACL checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .agent import Agent, ToolACL, TransportAuth


@dataclass
class ToolVisibility:
    name: str
    callable: bool        # may the model invoke it?
    known: bool           # should the model be told it exists?


def evaluate_tools_for_speaker(
    agent: Agent,
    *,
    transport: str = "discord",
    speaker_id: Any,
    speaker_role_ids: Iterable[int] = (),
    channel_id: Optional[int] = None,
    handle: str = "",
    tool_grants: list[str] | None = None,
) -> list[ToolVisibility]:
    """Return the per-tool visibility for this speaker in this channel.

    `speaker_id` is transport-native: int for Discord (user ID), string for
    iMessage (the handle, e.g. "+15551234567" or "you@example.com"). Match
    semantics: `==` against entries in `auth.<transport>.users`.

    `handle` is the raw sender handle for handle-based transports (iMessage
    email/phone). It is the source of truth for the admin bypass on those
    transports — see `_check_acl`. Empty/"" for Discord.

    Tools whose `name` appears more than once in `agent.allowed_tools` are
    OR-merged — the tool is callable if ANY entry matches. `visibility_when_
    denied` is taken from the first entry with that name (and across multiple
    entries, the "more permissive" choice — "known" beats "hidden").

    `tool_grants` is a per-session additive allow-path: a tool whose name is
    listed there is callable for this session even if no `_check_acl` entry
    passes — used for trusted synthetic sessions (cron jobs) with no human
    speaker. It is OR-ed onto the human path and never weakens it. Because we
    only iterate names already grouped from `agent.allowed_tools`, a grant for
    a tool the agent was never configured with is silently ignored (it can't
    conjure a tool). It confers tool-call authorization ONLY — never owner/admin.
    """
    role_set = set(speaker_role_ids)
    grants = set(tool_grants or ())
    grouped: dict[str, list[ToolACL]] = {}
    for acl in agent.allowed_tools:
        grouped.setdefault(acl.name, []).append(acl)
    out: list[ToolVisibility] = []
    for name, acls in grouped.items():
        callable_ok = any(
            _check_acl(
                a,
                transport=transport,
                speaker_id=speaker_id,
                role_set=role_set,
                channel_id=channel_id,
                handle=handle,
            )
            for a in acls
        )
        # Additive session grant: OR-path on top of the unchanged human path.
        # Only reachable for names present in allowed_tools (we iterate those).
        if name in grants:
            callable_ok = True
        # "known" beats "hidden" — if any entry asked to surface the tool to
        # the model even when blocked, honor it. A granted tool is callable,
        # hence known.
        known_when_denied = any(a.visibility_when_denied == "known" for a in acls)
        known = callable_ok or known_when_denied
        out.append(ToolVisibility(name=name, callable=callable_ok, known=known))
    return out


def _check_acl(
    acl: ToolACL,
    *,
    transport: str,
    speaker_id: Any,
    role_set: set[int],
    channel_id: Optional[int],
    handle: str = "",
) -> bool:
    """Evaluate one tool entry against the current speaker.

    Auth missing for this transport entirely → blocked, EXCEPT for admins:
    admins (owner + admin_ids/admin_handles) get every tool regardless of
    auth shape. Auto-injected blank entries (no auth at all) thus surface
    for admins only — non-admins see "blocked" the same as before.

    The admin tier is transport-aware (mirroring `is_admin`). Discord matches
    `speaker_id` (numeric) against `admin_ids`; handle-based transports
    (iMessage etc.) match the case-folded `handle` against
    `get_admin_handles(transport)`. An empty/missing handle on a handle-based
    transport never matches → no bypass (fail closed).

    Within one transport block: exclude wins; otherwise inclusion dimensions
    are AND-ed (with empty list = no restriction on that dimension).
    """
    # Admin bypass — owner + admins always pass every ACL. Transport-aware:
    # Discord matches the numeric speaker_id against admin_ids; handle-based
    # transports (iMessage) match the case-folded handle against the
    # configured admin_handles. Empty handle on a handle transport → no match
    # → falls through to normal ACL (fail closed). Cheap call (cached config).
    try:
        if transport == "discord":
            if is_admin(speaker_id):
                return True
        elif is_admin(speaker_id, integration=transport, handle=handle):
            return True
    except Exception as _admin_err:
        # An identity-shape mismatch here silently denies an admin. Don't let
        # it vanish — log it (lazy import keeps this hot auth module lean) so a
        # genuinely-broken admin check is debuggable instead of a silent deny.
        try:
            from .utils import print_ts, COLOR_YELLOW, COLOR_END
            print_ts(
                f"{COLOR_YELLOW}_check_acl: admin check raised for "
                f"transport={transport} speaker={speaker_id}: {_admin_err}{COLOR_END}",
                error=True,
            )
        except Exception:
            pass
    tauth = acl.auth.get(transport)
    if tauth is None:
        return False  # tool not configured for this transport
    # exclude-wins (always evaluated first)
    if speaker_id in tauth.exclude_users:
        return False
    if tauth.exclude_roles and (set(tauth.exclude_roles) & role_set):
        return False
    if channel_id is not None and channel_id in tauth.exclude_channels:
        return False
    # users dimension
    user_ok = (
        tauth.all_users
        or (not tauth.users and not tauth.all_users)  # empty list AND all_users=false → no users restriction
        or speaker_id in tauth.users
    )
    # An empty users list combined with all_users=false used to be ambiguous;
    # we treat it as "no users-dimension restriction" so that role-only or
    # channel-only ACLs work (e.g. {"roles": [...]}). all_users=true is the
    # canonical way to say "every user" — it short-circuits the users check
    # without requiring an empty list to mean the same thing.
    if not user_ok:
        return False
    # roles dimension (Discord-only in practice; empty list = no restriction)
    if tauth.roles and not (set(tauth.roles) & role_set):
        return False
    # channels dimension (Discord-only in practice; empty list = no restriction)
    if tauth.channels and channel_id not in tauth.channels:
        return False
    # If NO dimension was specified at all (no all_users, no users, no roles,
    # no channels), the entry is effectively "deny all on this transport" —
    # otherwise the operator could write `auth: {"discord": {}}` and silently
    # grant access to everyone, which is the trap the old bare-string entry
    # was. Force them to opt in explicitly via all_users:true.
    if not (tauth.all_users or tauth.users or tauth.roles or tauth.channels):
        return False
    return True


def is_owner(user_id: int, *, transport: str = "discord", handle: str = "") -> bool:
    """Standalone owner check for slash-command gating, cross-channel guards,
    and disclosure rules. NOT used inside ACL evaluation — the owner appears
    in `users` lists like any other user post-refactor.

    Transport-aware. Discord (the default) keeps the numeric path:
    `int(user_id) == get_owner_id("discord")` — byte-for-byte the original
    behavior, so every existing `is_owner(some_int)` caller is unchanged.

    Handle-based transports (iMessage, future SMS/email) pass
    `transport="imessage"` and the raw sender `handle`. iMessage identities
    are email/phone STRINGS, not numeric IDs — and the int we hash a handle
    into is per-process unstable (PYTHONHASHSEED), so it can never match a
    fixed config value. We compare the case-folded handle against the
    configured owner handle, using the SAME normalization as the iMessage
    sender allowlist (`.strip().lower()`). No configured owner handle, or no
    handle supplied → False (fail closed; never false-allow).
    """
    from .config_global import get_owner_id, get_owner_handle
    if transport != "discord":
        owner_handle = get_owner_handle(transport)
        if not owner_handle:
            return False
        return handle.strip().lower() == owner_handle
    return int(user_id) == get_owner_id("discord")


def current_caller_is_owner() -> bool:
    """Whether the CURRENT tool caller (from the contextvars set by the
    executor) is the owner. Defense-in-depth guard for genuinely dangerous
    tools (run_command / claude_code / restart_gateway / restart_flask_app):
    call this at the top of the tool so the owner check lives ON the dangerous
    operation and cannot be waved through by any upstream ACL bug (e.g. the
    admin-bypass in _check_acl, which intentionally does NOT cover these).

    Mirrors the ACL-context derivation in runtime.py: Session is the source of
    truth for transport + handle on handle-based transports (iMessage); Discord
    uses the numeric speaker_id. Fail closed: any error / unknown caller → False.
    """
    from .tool_executor import CURRENT_SPEAKER_ID, CURRENT_SESSION
    try:
        speaker_id = int(CURRENT_SPEAKER_ID.get(None) or 0)
    except Exception:
        speaker_id = 0
    transport = "discord"
    handle = ""
    try:
        sess = CURRENT_SESSION.get(None)
    except Exception:
        sess = None
    if sess is not None:
        transport = getattr(sess, "transport", "discord") or "discord"
        if transport != "discord":
            handle = getattr(sess, "handle", "") or ""
    try:
        return is_owner(speaker_id, transport=transport, handle=handle)
    except Exception:
        return False


def is_admin(user_id: int, integration: str = "discord", *, handle: str = "") -> bool:
    """Elevated-admin check — true for the owner AND anyone in `admin_ids`.

    This is the privilege tier ABOVE normal users but BELOW owner-only powers.
    Use this to gate elevated-but-not-dangerous capabilities. Keep genuinely
    dangerous powers (run_command, restart_gateway, claude_code) on
    `is_owner()` — admins must NOT get a shell or the ability to restart the
    framework. The owner is always an admin (get_admin_ids/get_admin_handles
    append the owner).

    Transport-aware, mirroring is_owner(). Discord (default) keeps the numeric
    path. Handle-based transports (`integration != "discord"`) compare the
    case-folded `handle` against `get_admin_handles(integration)` (which
    already includes the owner). Empty/unconfigured → False (fail closed).
    """
    from .config_global import get_admin_ids, get_admin_handles
    if integration != "discord":
        handles = get_admin_handles(integration)
        if not handles:
            return False
        return handle.strip().lower() in handles
    try:
        return int(user_id) in get_admin_ids(integration)
    except (TypeError, ValueError):
        return False
