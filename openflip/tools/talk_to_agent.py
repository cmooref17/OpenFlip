"""Inter-agent comms: dispatch a message to another running agent's turn loop.

Fires the recipient's `run_synthetic_turn` with the message framed as coming
from the calling agent. The recipient processes it like any other turn —
can call tools, reply with text, etc. Inter-agent traffic is SILENT on every
human transport: neither side's turns post to Discord/iMessage unless an
agent explicitly calls send_message. The operator's window into agent-to-
agent conversations is the OpenFlip dashboard (reads conversation files).

Dispatch is fire-and-forget on the caller's side: control returns immediately
after scheduling the synthetic turn. The recipient's reply does NOT come
back through this tool's result; it auto-routes back to the caller as a
follow-up (chain-terminator) synthetic turn in the conversation the caller
dispatched from (see runtime._run_turn's inter-agent auto-route block).

Intended use: one agent telling another to do something, share context,
or coordinate. Not a generic broadcast — point-to-point.

Target conversation resolution:
  1. `session_id` (explicit, canonical) — transport-prefixed conversation key,
     used directly.
  2. `channel_id` (explicit, deprecated) — bare Discord channel id, used only
     when session_id is empty.
  3. DEFAULT (both omitted) — a dedicated agent-to-agent conversation in the
     RECIPIENT's own namespace: "internal:peer-<caller>", one per sender
     (agents/<recipient>/conversations/internal:peer-<caller>.jsonl). The
     recipient processes peer traffic in its own isolated context, never
     inside a human-facing channel conversation. Headless recipients keep
     their single internal channel instead (every turn runs there by design).

The pre-2026-06 default resolved a REAL channel instead: the caller's current
channel when the recipient could access it, else the recipient's DM with the
human who triggered the caller's turn, else the caller's channel. That keyed
the recipient's side of inter-agent exchanges to operator-facing
conversations and — combined with chain-terminator return turns then being
dispatched visible — repeatedly dumped inter-agent chatter into the
operator's DMs. Removed deliberately; do not reintroduce channel guessing
here.

Security caveats:
* No throttle. A misbehaving agent could spam another with synthetic turns.
  Tool-level ACL in agent.json is the gate.
"""
from __future__ import annotations

import asyncio
import uuid

from ._base import tool, ToolResult
from ..utils import print_ts


@tool
async def talk_to_agent(agent_id: str, message: str, channel_id: int = 0, session_id: str = "") -> ToolResult:
    """Send a message to another running agent. The recipient processes it as a
    synthetic turn, framed as coming from the calling agent. By default the
    exchange is private agent-to-agent traffic: the recipient runs it in a
    dedicated peer conversation in its own namespace, and nothing is posted
    to any human channel — the recipient's reply comes back to you as a
    follow-up turn, not to the operator.

    Fire-and-forget: returns as soon as the synthetic turn is scheduled. The
    recipient's reply does NOT come back through this tool's result.

    Args:
        agent_id: The target agent's id (must be a currently running agent).
        message: The message text to send. Will be framed as '<sender>: <message>'
            so the recipient knows which agent it came from.
        session_id: OPTIONAL explicit target conversation key — the
            transport-prefixed conversation id (e.g. "discord:12345",
            "imessage:1", "internal:email-support"). When provided it is used
            DIRECTLY as the recipient's conversation for this turn. Pass it
            only when the recipient genuinely needs to process the message in
            a specific (usually human-facing) conversation's context.
        channel_id: DEPRECATED bare-int Discord channel id. Used only when
            session_id is empty. If both are omitted (the normal case), the
            recipient processes the message in its private per-peer
            conversation ("internal:peer-<your_id>") — inter-agent traffic
            never lands in a human-facing conversation by default.
    """
    from ..registry import RUNNERS
    from ..tool_executor import CURRENT_AGENT, CURRENT_CHANNEL_ID, CURRENT_SPEAKER_ID, CURRENT_TURN_DEPTH, CURRENT_SESSION, CURRENT_TURN_VISIBILITY

    sender = CURRENT_AGENT.get(None)
    sender_id = sender.id if sender else "unknown"

    if not agent_id:
        return ToolResult.fail("agent_id is required")
    if agent_id == sender_id:
        return ToolResult.fail(f"Cannot talk to yourself ({agent_id}).")
    # Coerce non-string types to str. Anthropic's tool-call parser sometimes
    # types bare-number messages (like "3" in a count-to-10 test) as int,
    # which crashed message.strip() on 2026-05-12.
    if message is not None and not isinstance(message, str):
        message = str(message)
    if not message or not message.strip():
        return ToolResult.fail("message is empty")

    # Loop prevention. The cap exists to stop runaway agent-to-agent loops
    # if either agent's logic goes off the rails. Each hop bumps depth by 1
    # (human=0 -> A->B=1 -> B->A=2 -> A->B=3 ...). Cap at 20 (~10 round-
    # trips) — low enough that runaway loops burn minimal compute before
    # the operator notices, high enough for legitimate multi-hop
    # coordination. If a real workflow hits 20, raise then — evidence-
    # driven rather than guessing. Previously 100; lowered 2026-05-22
    # after audit flagged it as too permissive.
    current_depth = CURRENT_TURN_DEPTH.get(0)
    MAX_DEPTH = 20
    if current_depth >= MAX_DEPTH:
        return ToolResult.fail(
            f"talk_to_agent depth cap reached ({current_depth} >= {MAX_DEPTH}). "
            "Report this to the owner and ask whether to continue — their next message "
            "resets the depth counter to 0 and the chain can resume from where it stopped."
        )

    target = RUNNERS.get(agent_id)
    if not target:
        return ToolResult.fail(
            f"Agent '{agent_id}' is not running. Active agents: {sorted(RUNNERS.keys())}"
        )

    # Resolve the originating human speaker once at the top so we can
    # thread it through both channel resolution AND the synthetic turn's
    # speaker attribution. Previously the speaker was hardcoded to
    # owner_id inside run_synthetic_turn, which silently spoofed the owner as
    # the speaker on every inter-agent dispatch regardless of who actually
    # triggered the upstream turn — a cross-user attribution leak. Now
    # the actual speaker_id rides through; runtime falls back to owner_id
    # only when this is 0 (no speaker context available).
    _caller_visibility = ""
    try:
        originating_speaker_id = int(CURRENT_SPEAKER_ID.get(0))
        # Chain-root visibility — what kind of channel is awaiting visible
        # output from this whole chain. Propagated to the recipient so its
        # chain-terminator turn knows whether to surface failures.
        _caller_visibility = CURRENT_TURN_VISIBILITY.get("") or ""
    except Exception:
        originating_speaker_id = 0

    # session_id is the canonical target conversation key. When supplied, build
    # a Session from it and dispatch DIRECTLY against it — bypassing every bit
    # of the channel-resolution guessing below. The transport prefix in
    # session_id is authoritative; the caller owns making it match the target's
    # transport. run_synthetic_turn already accepts a Session and keys history +
    # routing off its conversation_id / transport_id across all target types.
    _explicit_session = None
    # str()-coerce before .strip() so a non-string arg can't AttributeError.
    session_id = (str(session_id) if session_id else "").strip()
    if session_id:
        from .._conversation_io import _safe_conversation_id
        # Reuse the framework's single filesystem-safety gate (fail-closed on
        # path traversal / control chars) — this id becomes a filename.
        try:
            _safe_sid = _safe_conversation_id(session_id)
        except ValueError as _e:
            return ToolResult.fail(str(_e))
        from ..session import Session as _Session
        if ":" in _safe_sid:
            _t_name, _t_id = _safe_sid.split(":", 1)
        else:
            _t_name, _t_id = "", _safe_sid
        # CRITICAL: run_synthetic_turn keys the in-memory conversation by
        # int(Session.transport_id) — that's TransportChannel.id / the
        # `conversations` dict key — NOT by the conversation_id suffix. For
        # iMessage 1:1 DMs the suffix (the handle) does NOT equal the
        # transport_id (the numeric chat_id), so a suffix-derived transport_id
        # would key to a DUPLICATE dead object the agent never reads. If the
        # target already has a LIVE conversation with this conversation_id,
        # reuse ITS dict key as the transport_id so the synthetic turn lands in
        # that SAME live object. Otherwise the suffix-derived transport_id is an
        # acceptable fallback — a fresh conversation loads from the right
        # on-disk file, which is governed by conversation_id.
        _transport_id = _t_id
        for _k, _c in target.conversations.items():
            if getattr(_c, "conversation_id", None) == _safe_sid:
                # Only an INT dict key is a transport-native channel id worth
                # reusing for routing. Identity-linked conversations key by
                # the "linked:<canonical>" STRING — not a routable id; leave
                # the suffix-derived fallback. The synthetic turn still lands
                # in the right live object because get_conversation keys
                # linked conversations by conversation_id, not transport_id.
                if isinstance(_k, int):
                    _transport_id = str(_k)
                break
        _explicit_session = _Session(
            transport=_t_name,
            transport_id=_transport_id,
            conversation_id=_safe_sid,
            speaker_id=originating_speaker_id,
            speaker_role_ids=[],
            is_owner=False,
            is_dm=True,
            display_name=f"synthetic:{_safe_sid}",
            handle="",
        )

    target_is_headless = getattr(target, "is_headless", False)
    target_channel_id = int(channel_id) if channel_id else 0

    # Capture the caller's channel id + Session — threaded into the synthetic
    # turn as `originator_channel_id` / `originator_session` so the recipient's
    # reply auto-route can return to the caller's own conversation (the one
    # this dispatch is happening from) without any DM resolution, on any
    # transport, including non-routable internal/linked conversations.
    originator_channel_id = 0
    try:
        _caller_session = CURRENT_SESSION.get(None)
    except Exception:
        _caller_session = None
    try:
        if _caller_session is not None:
            # Raises for non-numeric ids (e.g. the caller is itself running in
            # an internal peer conversation) — fine: 0 means "no routable bare
            # channel"; the threaded `originator_session` below carries the
            # real conversation for the reply auto-route in that case.
            originator_channel_id = _caller_session.channel_id_int
        else:
            originator_channel_id = int(CURRENT_CHANNEL_ID.get(0) or 0)
    except Exception:
        originator_channel_id = 0
    if _explicit_session is not None:
        # Explicit session_id supplied — skip ALL channel resolution below;
        # the Session built above is dispatched directly.
        pass
    elif target_is_headless and not target_channel_id:
        # Headless target: no Discord/iMessage channel of its own. The target
        # runs every turn in its single internal channel (by design — don't
        # fragment a headless worker's context into per-peer conversations);
        # run_synthetic_turn's headless branch ignores this id and builds that
        # channel. This non-zero sentinel only satisfies the "no channel"
        # guard below + logging. The target's reply auto-routes BACK to us
        # (the originator) as a chain-terminator turn.
        from ..transports.null import make_internal_session
        target_channel_id = int(make_internal_session(agent_id).transport_id)
    elif not target_channel_id:
        # DEFAULT (no session_id, no channel_id): a dedicated agent-to-agent
        # conversation in the RECIPIENT's own namespace, keyed by the sender —
        # agents/<recipient>/conversations/internal:peer-<sender>.jsonl.
        # Inter-agent traffic is point-to-point and must not run inside (or
        # pollute) a human-facing conversation. The previous default resolved
        # a REAL channel here instead — the caller's current channel when the
        # recipient could reach it, else the recipient's own DM with the
        # human who triggered the caller's turn — which keyed the recipient's
        # side of the exchange to operator-facing conversations and
        # repeatedly surfaced inter-agent chatter in the operator's DMs
        # (2026-06 leak). Channel guessing removed deliberately; explicit
        # session_id/channel_id is the only way to target a real channel.
        from ..session import Session as _Session
        _peer_conv_id = f"internal:peer-{sender_id}"
        _explicit_session = _Session(
            transport="internal",
            # Non-numeric transport_id: the TransportChannel shim hashes it
            # to a per-process-stable int for the in-memory dict keys (same
            # pattern as make_internal_session). The on-disk file is keyed
            # by conversation_id, which IS stable across restarts. Being
            # non-numeric also short-circuits the Discord channel resolution
            # in _resolve_synthetic_channel straight to the session shim —
            # no wasted fetch_channel call on a fake id.
            transport_id=f"peer-{sender_id}",
            conversation_id=_peer_conv_id,
            speaker_id=originating_speaker_id,
            speaker_role_ids=[],
            is_owner=False,
            is_dm=True,
            display_name=f"peer:{sender_id}",
            handle="",
        )

    if _explicit_session is None and not target_channel_id:
        return ToolResult.fail(
            "No channel_id provided and no current channel context — pass channel_id explicitly."
        )

    # Resolve what we actually dispatch against + how we label it. With an
    # explicit session we hand run_synthetic_turn the Session (it keys history
    # off conversation_id); otherwise the legacy bare channel id.
    if _explicit_session is not None:
        _dispatch_target = _explicit_session
        _target_label = _explicit_session.conversation_id
        try:
            _events_channel_id = int(_explicit_session.transport_id)
        except (TypeError, ValueError):
            _events_channel_id = 0
    else:
        _dispatch_target = target_channel_id
        _target_label = str(target_channel_id)
        _events_channel_id = int(target_channel_id)

    framed = f"{sender_id}: {message}"

    # No Discord visibility for inter-agent traffic. The owner explicitly does
    # NOT want these messages cluttering his DMs with either agent — the
    # only intended visibility surface is the OpenFlip tab in the Flask
    # app (built separately). The synthetic turn dispatched below runs
    # silent + auto_post_final_text=False so nothing leaks to Discord.

    # Chain-ID stamping for parallel-branch detection.
    # Each talk_to_agent call generates a fresh chain_id. The caller's
    # _current_chain_to[recipient] tracker is overwritten with this new
    # chain_id, so any auto-route reply still in flight from a PREVIOUS
    # dispatch to the same recipient will mismatch on arrival and be
    # delivered with a late-reply [FRAMEWORK] tag by the check at the top
    # of _run_turn (it used to be hard-dropped there).
    # This guards the parallel-chain failure mode where an agent
    # went empty mid-test and re-dispatched — both the original
    # and the recovery chain stayed in flight, the peer answered both,
    # and the replies came back in duplicated/out-of-order pairs
    # with no way to tell which was the current chain. Now: each fresh
    # dispatch invalidates all prior chains to the same peer.
    chain_id = uuid.uuid4().hex
    caller_runner = RUNNERS.get(sender_id)
    if caller_runner is not None:
        caller_runner._current_chain_to[agent_id] = chain_id
        try:
            caller_runner._save_chain_state()
        except Exception:
            pass

    # Fire-and-forget at the caller side. The recipient processes the
    # synthetic turn; their final reply text is routed BACK to the caller
    # as another synthetic turn (see runtime._run_turn's auto-route block),
    # not posted directly to Discord. The caller then has a silent turn where
    # they can read the recipient's reply and either continue the chain (by
    # calling talk_to_agent again), surface something to a human explicitly
    # (send_message), or end with plain text — which is saved to history but
    # posted nowhere.
    try:
        asyncio.create_task(
            target.run_synthetic_turn(
                _dispatch_target,
                framed,
                # Inter-agent comms are fully invisible on Discord —
                # no auto-post to the recipient's channel, no typing
                # indicator. The reply still auto-routes back to the
                # originator via runtime's auto-route block for chain
                # continuation. the user's only window into the conversation
                # is the OpenFlip tab (reads conversation files directly).
                auto_post_final_text=False,
                silent=True,
                depth=current_depth + 1,
                originator_agent_id=sender_id,
                # chain_id carries through to the recipient's _run_turn
                # and is propagated forward when the recipient's reply
                # auto-routes back to the caller (the originator).
                chain_id=chain_id,
                # Thread the originating human speaker. Without this,
                # run_synthetic_turn falls back to owner_id and attributes
                # every inter-agent dispatch to the owner regardless of who
                # actually triggered the upstream turn — a cross-user
                # attribution leak. 0 means "no speaker context"; runtime
                # will fall back to owner_id in that case.
                speaker_id=originating_speaker_id,
                # Propagate chain-root visibility. If the CALLER's turn was
                # operator-initiated (Discord message), the recipient's
                # chain-terminator turn must surface a hard-failure message
                # to the originating channel if it goes empty. If the caller
                # was itself on a silent_agent_chain / cron / heartbeat,
                # propagate that tag — the failure should log loudly but
                # not leak into the operator's channel. Empty string falls
                # through to the conservative default (operator_channel
                # behavior). Read from contextvar set in _run_turn.
                originator_visibility=_caller_visibility,
                originator_channel_id=originator_channel_id,
                # Thread the caller's full Session so the recipient's reply
                # auto-route can return into THIS conversation even when it
                # has no routable bare channel id (internal peer sessions,
                # identity-linked conversations, iMessage handle-keyed 1:1s).
                # For plain Discord channels the int path above is used and
                # behavior is unchanged.
                originator_session=_caller_session,
            )
        )
    except Exception as e:
        return ToolResult.fail(f"Failed to schedule synthetic turn for {agent_id}: {e}")

    print_ts(
        f"talk_to_agent: {sender_id} -> {agent_id} in {_target_label}: {framed[:80]}",
        agent=sender_id,
    )
    try:
        from .. import events_log as _events_log
        _events_log.log_event(
            sender_id, "talk_to_agent",
            target=agent_id,
            channel_id=_events_channel_id,
            depth=current_depth + 1,
            chain_id=chain_id[:8],
            preview=message[:120],
        )
    except Exception:
        pass
    return ToolResult(
        model_feedback=f"Dispatched to {agent_id} in {_target_label}. Their reply will appear in that conversation; it will not return through this tool's result."
    )
