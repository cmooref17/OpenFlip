"""One AgentRunner per agent. Each owns a Transport (Discord by default)."""
from __future__ import annotations
import asyncio
import os
import tempfile
from typing import Optional

import nextcord
from nextcord.ext import commands

from .agent import Agent
from .conversation import DiscordConversation
from .anthropic_conversation import AnthropicConversation, MalformedRequestError
from .pipeline import build_user_prompt, should_respond, build_visible_tools, extract_image_attachments, build_inbound_from_discord
from .session import InboundMessage, Session
from .tool_executor import execute_tool_calls, build_model_feedback
from .tools import TOOL_REGISTRY
from .utils import print_ts, COLOR_YELLOW, COLOR_RED, COLOR_END, sanitize_outbound_text, log_task_exception
from .acl import is_owner
from .registry import RUNNERS
from .turn_retries import (
    action_promise_should_retry,
    detect_peer_prose,
    build_peer_prose_nudge,
    empty_retry_nudge,
    run_stop_hooks,
)
from . import agent_state as _agent_state
from .transport import Transport
from .transports.discord import DiscordTransport
from .discord_io import (
    safe_channel_send as _safe_channel_send,
    split_for_discord as _split_for_discord,
    safe_typing as _discord_safe_typing,
    silent_typing as _discord_silent_typing,
)


# Tools exempt from per-turn duplicate-call suppression. Most tools are
# correctly deduped by (name, args) signature — no double memory-save, no
# firing the same image gen twice. But delete_message in its
# (message_id=0, with_attachments=true) "delete my most recent bot message
# with attachment" mode is *intended* to fire repeatedly with identical
# args: each call walks one step further back through history. Deduping it
# would stop the walk after the first call. Keep this set minimal.
_DEDUPE_EXEMPT_TOOLS = {"delete_message"}


# Silence sentinel. An agent woken on every channel message (e.g. an agent
# on channels_only + always-respond) has no clean way to stay quiet — when it
# "decides" not to reply it narrates fake-silence ("stays quiet"), which still
# POSTS. To genuinely say nothing, the agent emits EXACTLY this token and
# nothing else; the runtime detects it (exact whole-message match after strip)
# and suppresses the post so NOTHING reaches the channel. A reply that merely
# mentions the word inside a sentence is NOT suppressed.
STAY_SILENT_SENTINEL = "STAY_SILENT"


class _SessionChannel:
    """A real nextcord channel paired with a caller-supplied Session.

    When run_synthetic_turn fires a Discord turn with a Session whose
    transport_id IS a real Discord channel, we still want to post to that
    real channel (so embeds/views/references/files keep working — the
    TransportChannel shim drops them) while honoring the passed Session as
    the source of truth for the conversation (its conversation_id keys
    history and the in-memory dicts).

    nextcord channel objects use __slots__, so the Session can't be attached
    to them directly. This thin proxy exposes `_session` as a real attribute
    (which `_run_turn` reads to set CURRENT_SESSION + derive conversation_id)
    and delegates every other attribute/method access to the wrapped channel.
    """

    def __init__(self, channel, session: "Session"):
        # Real instance attrs so they're found before __getattr__ delegates.
        self._real = channel
        self._session = session

    def __getattr__(self, name):
        # Only reached for names not set on the proxy (everything except
        # _real / _session) — forward to the wrapped nextcord channel.
        return getattr(self._real, name)

    def __repr__(self):
        return f"_SessionChannel(channel={self._real!r}, conversation_id={self._session.conversation_id!r})"


async def _notify_compaction_done(channel, *, was_manual: bool, elapsed_s: float | None = None) -> None:
    """Post the compaction-complete notice to a channel.

    Compaction fires two ways and the only thing that varies is the wording:
      - was_manual=True  → operator typed /compact; they already saw the
        "⚙️ Compacting conversation..." start notice, so give them an explicit
        "⚙️ Compacted conversation in Xs" completion terminator with timing.
      - was_manual=False → Anthropic compacted mid-stream unprompted; no start
        notice was posted, so this is the only signal — "⚙️ Compacted
        conversation in Xs".

    `elapsed_s`, when provided, is the wall-clock duration from the start
    notice to this call, rendered as "in X.Xs". When None (no start time
    captured) the duration suffix is omitted.

    Best-effort: a send failure here is cosmetic and must never tear the turn.
    """
    suffix = f" in {elapsed_s:.1f}s" if elapsed_s is not None else ""
    notice = f"⚙️ *Compacted conversation{suffix}*"
    try:
        await _safe_channel_send(channel, notice)
    except Exception:
        pass


class AgentRunner:
    def __init__(self, agent: Agent, token: str, transport: Optional[Transport] = None, transports: Optional[list[Transport]] = None):
        self.agent = agent
        self.token = token
        # Multi-transport support: the runner holds a list of transports.
        # Priority: explicit `transports` list > single `transport` > legacy default.
        if transports:
            self._transports: list[Transport] = list(transports)
        elif transport is not None:
            self._transports = [transport]
        else:
            self._transports = [DiscordTransport(token)]
        # Wire each transport back to us — each needs to know which runner to
        # dispatch on_message events to.
        for t in self._transports:
            if hasattr(t, "attach_runner"):
                t.attach_runner(self)
        # Per-transport task tracking for start/stop lifecycle.
        self._transport_tasks: list[asyncio.Task] = []
        self.conversations: dict[int, DiscordConversation | AnthropicConversation] = {}
        self._task: Optional[asyncio.Task] = None
        # Inbound queue: every turn (real Discord message OR synthetic turn from
        # cron / restart_gateway / talk_to_agent) goes through this. A single
        # worker drains it, but it dispatches each item into its own supervisor
        # task and loops back immediately — so turns in DIFFERENT channels/
        # sessions run CONCURRENTLY while turns in the SAME channel stay
        # SERIALIZED (see _inbound_worker / _serialized_turn).
        self._inbound_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._worker_task: Optional[asyncio.Task] = None
        # Global cap on concurrently-running turns across all channels on this
        # agent. Per-channel serialization already bounds concurrency to the
        # number of distinct active channels; this is a backstop so a burst of
        # many distinct channels can't spawn unbounded turns at once.
        self._turn_semaphore: asyncio.Semaphore = asyncio.Semaphore(8)
        # Inter-agent chain tracking: maps peer_agent_id -> chain_id of the
        # CURRENT outbound talk_to_agent dispatch to that peer. When this
        # agent calls talk_to_agent(peer, ...), a fresh chain_id is
        # generated and stored here, overwriting any previous chain to
        # that peer. When an auto-route synthetic turn arrives back at
        # this agent (carrying chain_id + auto_route_from_peer in kwargs),
        # the check at the top of _run_turn looks up
        # self._current_chain_to[auto_route_from_peer]: on mismatch the
        # turn is DELIVERED with a [FRAMEWORK] late-reply prefix (it used
        # to be dropped outright — that silently ate real answers whenever
        # a follow-up dispatch raced an in-flight reply; 2026-06-10 fix).
        # Cleared at end of a successful chain-terminator turn if no new
        # dispatch replaced it during the turn. Persisted to disk via
        # _save_chain_state/_load_chain_state so pending chains survive a
        # restart.
        self._current_chain_to: dict[str, str] = self._load_chain_state()
        # Active turn task per channel. Soft-inject path (default): if a new
        # inbound arrives while a turn here is still running, the message
        # gets appended to _pending_inject[ch_id] and surfaces as a
        # [FRAMEWORK] marker at the next tool-result boundary (or at turn
        # end if no tool fires). Hard-interrupt path (operator typed
        # `/stop` or fired the /stop slash command): the entry here gets
        # cancelled, _pending_inject is cleared, and the worker drains the
        # queue + runs the `/stop` message as a fresh turn. Reconciled
        # (re-pointed at a live predecessor or popped) by the worker's
        # `_on_turn_done` done-callback, plus a dispatch-time chase in
        # `_inbound_worker` for the pre-first-step-cancel race (B2).
        self._active_turns: dict[int, asyncio.Task] = {}
        # Per-channel soft-inject buffer. Holds operator messages typed
        # mid-turn that haven't been delivered to the model yet. Drained
        # inside _run_turn (after each tool-result append + at post-loop
        # cleanup) by appending each pending text as a user-role
        # [FRAMEWORK] marker, so the model naturally sees them on the
        # next chat() call. Wiped on hard interrupt and on /reset.
        self._pending_inject: dict[int, list[str]] = {}
        # Timestamp (ms since epoch) of the most recent INBOUND human
        # Discord message this agent received. Used by restart_gateway's
        # preflight to refuse restarting if a human just spoke in any
        # channel — they're probably mid-conversation and a restart
        # would interrupt them. Updated in _handle_message.
        self.last_human_inbound_ms: int = 0
        # Phase 2: event registration + slash commands are wired by the
        # transport's attach_runner() call above. AgentRunner no longer
        # manages the Bot directly — it delegates to self.transport.

    @property
    def transport(self) -> Transport:
        """Primary transport (first in the list).

        Back-compat: most of the codebase does `self.transport.X`. This
        returns `self._transports[0]` so existing call sites keep working.
        For multi-transport dispatch, callers that need the transport a
        specific message arrived on should thread the transport through
        from the inbound handler instead of using this property.
        """
        return self._transports[0]

    @property
    def is_headless(self) -> bool:
        """True when this agent has no real messaging surface — every transport
        is the no-op NullTransport (name == "internal").

        Headless agents only ever run synthetic turns (talk_to_agent / cron):
        they have no Discord/iMessage channel of their own and MUST NOT touch
        `self.bot` (which raises AttributeError for non-Discord transports).
        The run_synthetic_turn / typing paths branch on this to build an
        internal TransportChannel instead.
        """
        return bool(self._transports) and all(
            getattr(t, "name", "") == "internal" for t in self._transports
        )

    @property
    def bot(self):
        """Backward-compat shim: many call sites still do `self.bot.X`.

        Returns the underlying nextcord.Bot. For multi-transport agents,
        prefers the Discord transport's bot (if present), else falls back
        to the first transport's bot (which will raise AttributeError for
        non-Discord transports — intentional, those call sites need to
        migrate to `self.transport.X` in Phase 3).
        """
        for t in self._transports:
            if getattr(t, "name", "") == "discord" and hasattr(t, "bot"):
                return t.bot
        return self._transports[0].bot

    def reload_agent_config(self) -> bool:
        """Reload this agent's config + system files from disk on demand.

        Wired to the owner /reload slash command. Returns True if the on-disk
        files actually changed since last load. Re-applies the new system
        message to every live conversation so the next turn picks up the
        update. Without this, edits to SOUL.md / _shared/FRAMEWORK.md etc.
        only take effect after a process restart.
        """
        # Backfill blank personal AGENT.md/TOOLS.md first, so creating a missing
        # stub registers as a change and gets picked up by reload_if_changed().
        from .main import ensure_personal_files
        from pathlib import Path
        ensure_personal_files(Path(self.agent.path).parent)
        changed = self.agent.reload_if_changed()
        if not changed:
            return False
        for conv in self.conversations.values():
            try:
                conv.reapply_agent()
            except Exception:
                pass
        return True

    def get_conversation(self, channel_id: int, conversation_id: str) -> DiscordConversation | AnthropicConversation:
        """Get-or-create the conversation for this channel.

        `conversation_id` must be the full transport-prefixed id from the
        Session object (e.g. "discord:<id>", "imessage:<chat_id>"). No
        fallback — callers must thread the real value through from the
        session. Hardcoding a default here was the bug that produced
        cross-transport collisions and `discord:` filenames for iMessage
        conversations.

        In-memory dict key stays as the int channel_id for fast lookup,
        but the on-disk filename is governed by `conversation_id`.
        """
        if not conversation_id:
            raise ValueError(
                f"get_conversation requires conversation_id (channel_id={channel_id}). "
                f"Pass session.conversation_id from the inbound message."
            )
        c = self.conversations.get(channel_id)
        if c is None:
            if self.agent.provider == "anthropic":
                c = AnthropicConversation(conversation_id, self.agent)
            else:
                c = DiscordConversation(conversation_id, self.agent)
            c.load()
            self.conversations[channel_id] = c
        return c


    def _hard_interrupt(self, channel_id: int) -> int:
        """Hard-interrupt the active turn (if any) for `channel_id`.

        Cancels the in-flight `_run_turn` task and clears the channel's
        pending soft-inject buffer. Transport-agnostic — called by both:
          - `_handle_message` when the operator's text starts with `/stop`
          - the `/stop` Discord slash command in commands.py

        Returns the number of pending soft-inject messages that were
        dropped (useful for the caller's user-visible ack). The caller is
        responsible for enqueueing whatever follow-up turn should land
        next — this method ONLY cancels + clears.
        """
        if not channel_id:
            return 0
        active_task = self._active_turns.get(channel_id)
        pending_count = len(self._pending_inject.get(channel_id, []) or [])
        if active_task is None or active_task.done():
            # No active turn. Still wipe the pending buffer so a /stop
            # between turns doesn't leave stale queued messages behind.
            if pending_count:
                self._pending_inject.pop(channel_id, None)
            return pending_count
        print_ts(
            f"{COLOR_YELLOW}hard interrupt (/stop) on channel {channel_id} "
            f"— cancelling active turn + clearing {pending_count} "
            f"pending soft-inject(s){COLOR_END}",
            agent=self.agent.id,
        )
        self._pending_inject.pop(channel_id, None)
        active_task.cancel()
        return pending_count

    def _drain_pending_injects(self, channel_id: int, conv) -> int:
        """Soft-inject drain. Pops everything in _pending_inject[channel_id]
        and appends each message as a user-role [FRAMEWORK] marker on the
        conversation so the model sees it at its next chat() call.

        Mirrors Claude Code's mid-task soft-inject: a message typed while
        the agent is working lands at the next natural tool boundary, not
        as a hard cancel. Caller is responsible for picking the right
        moment (after tool-results appended / at post-loop cleanup) so we
        never insert a user message between a tool_use and its tool_result
        (Anthropic's API would 400 on the broken pair).

        Returns the number of messages drained.
        """
        if not channel_id:
            return 0
        pending = self._pending_inject.pop(channel_id, None)
        if not pending:
            return 0
        if self.agent.provider == "anthropic":
            from .anthropic_conversation import ChatMessage
        else:
            from openflip.ollama_api import ChatMessage
        for text in pending:
            marker = (
                f"[FRAMEWORK]: The operator sent this message while you were "
                f"mid-turn. You MUST address it in your very next reply — "
                f"before any tool-result confirmation, before continuing what "
                f"you were doing, before anything else. If it changes what you "
                f"should be doing, pivot. If it does not, acknowledge it in "
                f"one sentence and then continue. Ignoring it is the failure "
                f"mode this marker exists to prevent.\n\n"
                f"Operator's message: {text}"
            )
            conv.messages.append(ChatMessage('user', marker))
        print_ts(
            f"{COLOR_YELLOW}drained {len(pending)} soft-inject message(s) "
            f"into channel {channel_id} history{COLOR_END}",
            agent=self.agent.id,
        )
        return len(pending)

    def _chain_state_path(self):
        """Path to the persisted chain-tracker file for this agent.
        Lives alongside agent.json so it stays inside the agent's own dir."""
        import os
        return os.path.join(os.path.dirname(self.agent.path), "chain_state.json")

    def _load_chain_state(self) -> dict:
        """Load _current_chain_to from disk if present. Restart-safe so an
        in-flight chain reply landing on a fresh process is recognized
        instead of silently dropped as stale."""
        import json, os
        p = self._chain_state_path()
        if not os.path.exists(p):
            return {}
        try:
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception as e:
            print_ts(
                f"{COLOR_YELLOW}_load_chain_state: {p} unreadable, starting empty: {e}{COLOR_END}",
                agent=self.agent.id,
            )
        return {}

    def _save_chain_state(self) -> None:
        """Atomically persist _current_chain_to. Call after every mutation
        (set on dispatch, pop on consumption). tmp-then-rename so a crash
        mid-write cannot leave a corrupt file."""
        import json, os
        p = self._chain_state_path()
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._current_chain_to, f)
            os.replace(tmp, p)
        except Exception as e:
            print_ts(
                f"{COLOR_YELLOW}_save_chain_state: failed to persist {p}: {e}{COLOR_END}",
                agent=self.agent.id,
            )

    def _ensure_worker_started(self):
        """Start the inbound queue worker if it isn't already running.

        Lazy-started on the first enqueue so we don't need an async __init__
        — by the time anything enqueues, we're already inside an event loop.
        """
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._inbound_worker())

    async def _inbound_worker(self):
        """Drain the inbound queue, dispatching turns with per-session concurrency.

        Real Discord messages and synthetic turns (cron, talk_to_agent, restart
        continuation) all flow through this one queue, but the worker no longer
        awaits a turn before pulling the next item. Each item is dispatched into
        its own supervisor task (`_serialized_turn`) and the worker loops back
        IMMEDIATELY. Consequences:
        * Turns in DIFFERENT channels/sessions run CONCURRENTLY. A Discord turn
          no longer waits behind an in-flight iMessage turn; person A no longer
          waits for person B.
        * Turns in the SAME channel stay SERIALIZED. `_active_turns[channel_id]`
          is the per-channel in-flight marker; the enqueue-time guard in
          `_handle_message`/`_handle_inbound` soft-injects or hard-interrupts a
          second inbound for a channel that already has a live turn, so two
          turns for one channel almost never reach the queue concurrently. The
          one case that does — a hard interrupt cancels the live turn and
          enqueues the `/stop` follow-up while the cancelled turn is still
          winding down — is covered by the supervisor waiting for any prior
          same-channel task before it touches the shared conversation object.

        Per-turn slot reconciliation + task_done() accounting live in the
        `_on_turn_done` done-callback (attached below), NOT in the supervisor
        body — a supervisor cancelled before its first __step skips its own
        try/finally entirely, so cleanup placed there would never run (B2). A
        done-callback always fires once the task completes, started or not. The
        [CHAIN_ERROR] routing on exception stays in `_serialized_turn`.
        Exceptions never reach here — the supervisor swallows them — so one bad
        turn can't kill the worker. The agent stays receptive.
        """
        while True:
            try:
                kwargs = await self._inbound_queue.get()
            except asyncio.CancelledError:
                return
            channel = kwargs.get("channel")
            channel_id = int(getattr(channel, "id", 0) or 0)
            # Snapshot any prior same-channel turn SYNCHRONOUSLY (no await
            # between get() and the _active_turns write) so the enqueue-time
            # interrupt check always sees a live marker the instant an item
            # leaves the queue, and so the supervisor can serialize behind a
            # still-cancelling predecessor. Different channels have distinct
            # keys and never serialize against each other.
            prev_task = self._active_turns.get(channel_id) if channel_id else None
            # B2 dispatch-time chase: a supervisor cancelled BEFORE its first
            # __step never ran its body, so its slot reconciliation runs only in
            # the `_on_turn_done` done-callback — which is call_soon-SCHEDULED,
            # not yet run, at the instant we read the slot here (the cancelled
            # task's wakeup is scheduled before this worker wakeup, but its
            # done-callbacks are scheduled AFTER). So the slot can still point at
            # that now-done() task while the turn it was serializing behind is
            # alive. Resolve through any such orphaned pre-start-cancelled markers
            # NOW, synchronously, to the genuine live predecessor — otherwise this
            # turn would read a done() prev_task, skip the wait below, and run
            # `_run_turn` CONCURRENTLY with a still-cleaning-up predecessor (torn
            # history / Anthropic 400). _prev_task always points at an earlier
            # task, so the chain is a finite DAG; the _hops cap is pure paranoia.
            _hops = 0
            while (prev_task is not None and prev_task.done()
                    and not getattr(prev_task, "_started_run", False)
                    and _hops < 64):
                prev_task = getattr(prev_task, "_prev_task", None)
                _hops += 1
            turn_task = asyncio.create_task(
                self._serialized_turn(channel_id, prev_task, kwargs)
            )
            # Fields read by `_on_turn_done` and by the B2 chase above. Set
            # BEFORE registering the slot / attaching the callback and BEFORE the
            # coroutine's first __step — create_task only SCHEDULES the coro; it
            # cannot run until this worker next awaits get(), so these are in
            # place before anything observes them.
            turn_task._channel_id = channel_id
            turn_task._prev_task = prev_task
            turn_task._started_run = False
            if channel_id:
                self._active_turns[channel_id] = turn_task
            # Slot reconciliation + task_done() accounting live in this
            # done-callback, NOT the supervisor body: a supervisor cancelled
            # before its first __step skips its own try/finally entirely (B2),
            # but a done-callback ALWAYS fires once the task completes.
            turn_task.add_done_callback(self._on_turn_done)
            # Do NOT await turn_task — loop back to drain the next item so turns
            # in other channels start without waiting for this one to finish.

    def _on_turn_done(self, task: asyncio.Task):
        """Done-callback attached by `_inbound_worker` to every dispatched
        supervisor task. Fires EXACTLY ONCE when the task completes — crucially
        INCLUDING when the supervisor was cancelled before its first __step ever
        ran (B2), the path on which the coroutine's own try/finally never
        executes. Owns the two cleanup duties the supervisor body can no longer
        guarantee:

        1. Slot reconciliation (`is task` guarded so a newer same-channel turn
           that already re-registered is never stomped): if this task was
           cancelled before it started `_run_turn` (`_started_run` still False)
           and the predecessor it was serializing behind is still live, RE-POINT
           the slot at that predecessor so the next same-channel inbound
           soft-injects into / serializes behind the real running turn instead
           of reading a stale own-done slot. Otherwise pop. (The dispatch-time
           B2 chase in `_inbound_worker` covers the window where a successor is
           ALREADY being dispatched before this call_soon callback runs; this
           callback covers the no-successor case + final slot cleanup.)
        2. task_done() accounting: balance the `_inbound_queue.get()` for this
           item exactly once, even on the pre-first-step-cancel path.
        """
        ch = getattr(task, "_channel_id", 0)
        if ch and self._active_turns.get(ch) is task:
            prev = getattr(task, "_prev_task", None)
            if (not getattr(task, "_started_run", False)
                    and prev is not None and not prev.done()):
                self._active_turns[ch] = prev
            else:
                self._active_turns.pop(ch, None)
        # Balance the _inbound_queue.get() for this item. Runs exactly once per
        # dispatched item even when the supervisor was cancelled before its
        # first __step — a done-callback fires regardless of whether the
        # coroutine body (and any finally it might have held) ever ran.
        try:
            self._inbound_queue.task_done()
        except Exception:
            pass

    async def _serialized_turn(self, channel_id: int, prev_task: Optional[asyncio.Task], kwargs: dict):
        """Run one turn, serialized behind any prior same-channel turn. Spawned
        (not awaited) by `_inbound_worker` so different channels overlap; the
        same-channel wait below keeps one channel serial.

        Registered in `_active_turns[channel_id]` by the worker, so cancelling
        THIS task (via `_hard_interrupt`) propagates into `_run_turn` — we await
        the `_run_turn` coroutine directly (not as a sub-task), so its
        CancelledError cleanup branch fires exactly as the old inline-await
        worker relied on. This preserves the same-channel interrupt/cancel
        behavior unchanged.

        Per-turn slot reconciliation + task_done() accounting do NOT live in
        this body — they're owned by the worker's `_on_turn_done` done-callback.
        This body can be cancelled BEFORE its first __step (rapid double-/stop),
        in which case NONE of it runs — not even a `finally` — so cleanup placed
        here would silently leak (B2). `_started_run` is hung on the task object
        (initialised False by the worker, flipped True here) so the done-callback
        AND the worker's dispatch-time B2 chase can both observe whether this
        turn ever reached `_run_turn`.
        """
        me = asyncio.current_task()
        try:
            # Serialize behind a prior same-channel turn still winding down.
            # The hard-interrupt case: the old turn was cancelled and this one
            # enqueued, but the old turn's cancellation cleanup — which mutates
            # the SAME conversation object + JSONL file — may not have finished
            # yet. Use asyncio.wait() (not `await prev_task`) so prev's own
            # exception/cancellation is ignored here, while a cancel of THIS
            # task still propagates out normally. Different channels never share
            # a prev_task, so they never block each other.
            if prev_task is not None and not prev_task.done():
                await asyncio.wait({prev_task})
            # Global concurrency backstop across all channels (per-channel
            # serial already bounds it to #active-channels). `async with`
            # releases the permit on every exit path — including cancel — so
            # there's no try/finally release to leak through. Kept INSIDE the
            # prev_task wait so same-channel ordering is unchanged.
            async with self._turn_semaphore:
                # Hung on the task object so the worker's `_on_turn_done`
                # callback and dispatch-time B2 chase can see it. Flipped the
                # instant before `_run_turn`: a supervisor cancelled while still
                # False (during the prev_task wait or the semaphore acquire)
                # never ran the turn, so it must NOT orphan its still-running
                # predecessor — the callback/chase re-point the slot at prev_task.
                me._started_run = True
                await self._run_turn(**kwargs)
        except asyncio.CancelledError:
            # Turn was interrupted by a new inbound — expected, not an error.
            # _run_turn already wrote the interrupted marker to history in its
            # except-CancelledError branch (if it had started).
            print_ts(
                f"{COLOR_YELLOW}turn cancelled by interrupt (channel {channel_id}){COLOR_END}",
                agent=self.agent.id,
            )
        except Exception as e:
            print_ts(
                f"{COLOR_RED}inbound worker: turn raised {e}{COLOR_END}",
                error=True, agent=self.agent.id,
            )
            # S1: `_run_turn` has already returned/raised, so this conversation
            # is no longer being driven. Mark the turn no-longer-active NOW —
            # before the [CHAIN_ERROR] routing below, which awaits fetch_user
            # (up to 10s) + create_dm. If we left the slot live across that
            # window, a concurrent same-session inbound would soft-inject into
            # `_pending_inject[ch]` that nothing will ever drain (and no
            # successor turn is enqueued), losing the message. Popping here lets
            # that inbound see an empty slot and enqueue a FRESH turn instead.
            # Guarded by `me._started_run` so this only fires once `_run_turn`
            # actually ran — the not-yet-started case is owned by the worker's
            # `_on_turn_done` re-point / dispatch-time B2 chase, never this early
            # pop. `is me`-guarded so we don't stomp a newer same-channel
            # registration.
            if getattr(me, "_started_run", False) and channel_id and self._active_turns.get(channel_id) is me:
                self._active_turns.pop(channel_id, None)
            # #4 audit: if this turn was processing for an inter-agent
            # caller (originator_agent_id set), route a [CHAIN_ERROR]
            # synthetic turn back so the originator doesn't wait
            # forever for a reply that will never come. Best-effort —
            # any failure here is itself swallowed; we already logged
            # the underlying error above.
            try:
                _orig_id = kwargs.get("originator_agent_id") or ""
                if _orig_id:
                    from .registry import RUNNERS
                    _orig = RUNNERS.get(_orig_id)
                    if _orig is not None:
                        _chain_id = kwargs.get("chain_id") or ""
                        _speaker_id = int(kwargs.get("speaker_id") or 0)
                        _orig_vis = kwargs.get("originator_visibility") or ""
                        _depth = int(kwargs.get("depth") or 0)
                        # Resolve a return channel — originator's DM with
                        # the originating human, falling back to current
                        # channel id (which will likely 403 but at least
                        # surfaces the issue).
                        _ret_channel = 0
                        try:
                            if _speaker_id:
                                _user = await asyncio.wait_for(
                                    _orig.bot.fetch_user(_speaker_id), timeout=10.0,
                                )
                                if _user is not None:
                                    _dm = _user.dm_channel or await _user.create_dm()
                                    if _dm is not None:
                                        _ret_channel = int(_dm.id)
                        except Exception:
                            pass
                        if not _ret_channel:
                            _ret_channel = channel_id or 0
                        if _ret_channel:
                            _err_preview = str(e)[:200].replace("\n", " ")
                            _framed = f"{self.agent.id}: [CHAIN_ERROR] {_err_preview}"
                            _ce_task = asyncio.create_task(_orig.run_synthetic_turn(
                                _ret_channel,
                                _framed,
                                auto_post_final_text=True,
                                silent=False,
                                depth=_depth + 1,
                                originator_agent_id="",
                                chain_id=_chain_id,
                                auto_route_from_peer=self.agent.id,
                                speaker_id=_speaker_id,
                                originator_visibility=_orig_vis,
                            ), name="autoroute_chain_error")
                            _ce_task.add_done_callback(log_task_exception)
                            print_ts(
                                f"{COLOR_YELLOW}routed [CHAIN_ERROR] back to '{_orig_id}' "
                                f"chain={(_chain_id or '')[:8]}{COLOR_END}",
                                agent=self.agent.id,
                            )
                            try:
                                from . import events_log as _events_log
                                _events_log.log_event(
                                    self.agent.id, "chain_error_routed",
                                    to_originator=_orig_id,
                                    chain_id=(_chain_id or "")[:8],
                                    reason=_err_preview,
                                )
                            except Exception:
                                pass
            except Exception as _route_err:
                print_ts(
                    f"{COLOR_YELLOW}failed to route CHAIN_ERROR to caller: {_route_err}{COLOR_END}",
                    agent=self.agent.id,
                )
        # No `finally`: slot reconciliation (the B1 re-point-vs-pop decision) and
        # task_done() accounting are owned by the worker's `_on_turn_done`
        # done-callback, NOT this body. A supervisor cancelled before its first
        # __step (rapid double-/stop) skips its whole try/finally — `coro.throw`
        # into a never-started coroutine raises immediately without running any
        # of it — so anything placed in a finally here would be silently skipped,
        # leaving the slot pointing at this now-done() task and letting the next
        # same-channel turn run `_run_turn` concurrently with a still-cleaning-up
        # predecessor (B2). The done-callback fires regardless of whether this
        # body ever started, and the worker's dispatch-time chase re-resolves the
        # slot synchronously the instant a successor is dispatched, so both the
        # no-successor and racing-successor windows are covered.

    async def _handle_inbound(self, inbound: InboundMessage, *, transport: Optional[Transport] = None):
        """Transport-agnostic message handler. Called by non-Discord transport
        adapters (currently IMessageTransport via _dispatch).

        Mirrors _handle_message but starts from a pre-built InboundMessage and
        a TransportChannel shim instead of a nextcord.Message + nextcord channel.
        Discord-specific affordances (👀 reaction on captured soft-inject) are
        skipped because there's no inbound Message object to react on.

        `transport` is the Transport instance the message arrived on. Multi-
        transport agents pass this so the TransportChannel shim and text-
        command handler use the correct platform. Falls back to
        self.transport (first in list) when None.
        """
        import time
        from .transports.channel_shim import TransportChannel
        from .pipeline import build_user_prompt
        _transport = transport or self.transport
        self.last_human_inbound_ms = int(time.time() * 1000)

        # Human messages preempt queued synthetic turns (same drain logic
        # as _handle_message — see that method's comment for rationale).
        try:
            dropped = 0
            saved: list = []
            while True:
                try:
                    item = self._inbound_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                is_synthetic = (item.get("log_tag") or "").strip().startswith("[synthetic]")
                is_chain_terminator = bool(item.get("auto_route_from_peer"))
                if is_synthetic and not is_chain_terminator:
                    dropped += 1
                else:
                    saved.append(item)
                self._inbound_queue.task_done()
            for item in saved:
                await self._inbound_queue.put(item)
            if dropped:
                print_ts(
                    f"{COLOR_YELLOW}dropped {dropped} queued synthetic turn(s) — "
                    f"preempted by human message{COLOR_END}",
                    agent=self.agent.id,
                )
        except Exception as _drain_err:
            print_ts(
                f"{COLOR_YELLOW}queue preempt drain failed (continuing): {_drain_err}{COLOR_END}",
                agent=self.agent.id,
            )

        # Build the channel shim — runtime.py's queue payload + _run_turn
        # expect a channel object with .id, .send, .typing(), etc.
        channel = TransportChannel(transport=_transport, session=inbound.session)
        speaker_id = inbound.sender_id
        is_dm = inbound.is_dm
        role_ids = inbound.session.speaker_role_ids
        owner = inbound.session.is_owner

        ch_label = "DM" if is_dm else f"#{channel.name}"
        preview = (inbound.text or "")[:80].replace("\n", " ")
        print_ts(f"<- {inbound.sender_display_name} in {ch_label}: {preview}", agent=self.agent.id)

        user_text = build_user_prompt(inbound)

        # Text-prefix commands like /reset, /compact, /status, /help, /reload,
        # /restart, /uncompact. Handled cross-transport before any
        # interrupt/enqueue logic — if matched, short-circuit. /stop is
        # NOT in this set; it falls through to the hard-interrupt path below.
        ch_id = channel.id
        raw_text = (inbound.text or "").strip()
        try:
            from . import text_commands as _tcmd
            if _tcmd.is_text_command(raw_text):
                handled = await _tcmd.handle_text_command(
                    runner=self,
                    channel=channel,
                    speaker_id=speaker_id,
                    raw_text=raw_text,
                    transport=_transport,
                    session_id=getattr(inbound.session, "session_id", "") or str(ch_id),
                    session=inbound.session,
                )
                if handled:
                    return
        except Exception as _tcmd_err:
            print_ts(
                f"{COLOR_YELLOW}text_commands dispatch failed (continuing as normal message): {_tcmd_err}{COLOR_END}",
                agent=self.agent.id,
            )

        # Soft-inject vs hard-interrupt — same shape as _handle_message.
        active_task = self._active_turns.get(ch_id) if ch_id else None
        is_hard_interrupt = raw_text.lower().startswith("/stop")
        if active_task is not None and not active_task.done():
            if is_hard_interrupt:
                self._hard_interrupt(ch_id)
                # Fall through to enqueue path below.
            else:
                self._pending_inject.setdefault(ch_id, []).append(user_text)
                print_ts(
                    f"{COLOR_YELLOW}soft-inject queued on channel {ch_id} "
                    f"(pending count: {len(self._pending_inject[ch_id])}) "
                    f"— from {inbound.sender_display_name}{COLOR_END}",
                    agent=self.agent.id,
                )
                # Non-Discord transports: no inbound Message to add 👀
                # reaction to. Operator gets visibility via log only.
                return

        self._ensure_worker_started()
        await self._inbound_queue.put({
            "channel": channel,
            "speaker_id": speaker_id,
            "role_ids": role_ids,
            "owner": owner,
            "user_text": user_text,
            "discord_message": None,  # non-Discord transport
            "inbound": inbound,
            "log_tag": "",
            "depth": 0,
            "originator_visibility": "operator_channel",
        })

    async def _handle_message(self, message: nextcord.Message, *, inbound: Optional[InboundMessage] = None, transport: Optional[Transport] = None):
        """Real Discord message path. Extracts speaker info, builds the user
        prompt with attribution + attachments, and delegates to _run_turn.

        `transport` is the Transport instance the message arrived on (passed
        by DiscordTransport). Falls back to self.transport when None.

        The transport (DiscordTransport.on_message) pre-builds the
        InboundMessage and passes it in via the `inbound` kwarg — that lets
        the transport call should_respond() on the wrapped object instead
        of forcing the legacy nextcord-aware path. If `inbound` is None
        (legacy callers), we build it ourselves below.

        `message` is still the raw nextcord.Message and is kept for the
        deliberate Discord-side bridge paths (vision attachment .read(),
        fetch_discord_message tool, etc).
        """
        _transport = transport or self.transport
        # Track the most recent inbound human message ts so restart_gateway's
        # preflight can refuse to restart if a human just spoke. Without this
        # tracking, an agent can fire restart while the user is mid-conversation
        # and lose his queued reply.
        import time
        self.last_human_inbound_ms = int(time.time() * 1000)

        # Human messages preempt queued synthetic turns. Without this, a
        # message from the user can get stuck behind queued inter-agent
        # chain-terminator turns or cron-triggered work, and effects of
        # his message (especially "stop X") only land after the synthetic
        # queue drains. Drains all queued items, drops synthetic ones,
        # re-queues non-synthetic, then enqueues the human message at
        # the back. Added 2026-05-10 after the user's complaint that synthetic
        # turns kept processing after he said "stop focusing on openflip tab."
        try:
            dropped = 0
            saved: list = []
            while True:
                try:
                    item = self._inbound_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                is_synthetic = (item.get("log_tag") or "").strip().startswith("[synthetic]")
                # Chain-terminator turns (auto_route_from_peer set) carry an
                # auto-routed reply from another agent — the ONLY chance this
                # agent has to read what the peer just said. Dropping them
                # loses the chain irrecoverably. Cron/heartbeat/restart-
                # continuation synthetic turns DON'T have auto_route_from_peer
                # set, so they're safe to drop (they'll fire again later).
                # Confirmed in prod: an agent's finish-reply vanished
                # because a human message preempted the queued
                # chain-terminator, and the agent gave a stale "still working"
                # answer because his reply never reached my context.
                is_chain_terminator = bool(item.get("auto_route_from_peer"))
                if is_synthetic and not is_chain_terminator:
                    dropped += 1
                else:
                    saved.append(item)
                self._inbound_queue.task_done()
            for item in saved:
                await self._inbound_queue.put(item)
            if dropped:
                print_ts(
                    f"{COLOR_YELLOW}dropped {dropped} queued synthetic turn(s) — "
                    f"preempted by human message{COLOR_END}",
                    agent=self.agent.id,
                )
        except Exception as _drain_err:
            print_ts(
                f"{COLOR_YELLOW}queue preempt drain failed (continuing): {_drain_err}{COLOR_END}",
                agent=self.agent.id,
            )

        # InboundMessage is normally pre-built by the transport adapter (see
        # DiscordTransport.on_message) so should_respond can be called on the
        # wrapped object. If a legacy caller didn't pre-wrap, we build it
        # here as a fallback.
        if inbound is None:
            from .config_global import get_owner_id
            owner_id = get_owner_id("discord")
            inbound = build_inbound_from_discord(message, self.bot.user.id, owner_id)
        speaker_id = inbound.sender_id
        is_dm = inbound.is_dm
        role_ids = inbound.session.speaker_role_ids
        owner = inbound.session.is_owner

        ch_label = "DM" if is_dm else f"#{getattr(message.channel, 'name', message.channel.id)}"
        preview = (message.content or "")[:80].replace("\n", " ")
        print_ts(f"<- {message.author.name} in {ch_label}: {preview}", agent=self.agent.id)

        user_text = build_user_prompt(inbound)

        # Soft-inject vs hard-interrupt branching for mid-turn messages.
        #
        # Default (soft inject): a message typed while a turn is running on
        # this channel does NOT cancel the turn. It gets queued in
        # _pending_inject[ch_id] and _run_turn picks it up at the next
        # tool-result boundary (or at post-loop cleanup if no tool fires),
        # appending it as a user-role [FRAMEWORK] marker so the model sees
        # it on its next chat() call. Lets the agent finish the current
        # tool then pivot naturally.
        #
        # Hard interrupt (operator prefixes with `/stop`): falls through to
        # the original cancel-active-task path. The `/stop` message itself
        # still enqueues as a fresh turn so the operator's directive lands
        # in conversation history. Pending soft-injects for the channel
        # are wiped (the active task is dying, queued messages are stale).
        # Transport-agnostic — the same prefix check runs on any transport
        # (Discord text, iMessage, future) so operators have one mechanism
        # everywhere. The Discord-only `/stop` slash command in commands.py
        # calls the same _hard_interrupt() shared method.
        ch_id = int(getattr(message.channel, "id", 0) or 0)
        raw_text = (message.content or "").strip()

        # Text-prefix commands (/reset, /compact, /status, etc) — cross-
        # transport mirror of the slash commands. Handled BEFORE the soft-
        # inject/hard-interrupt branching so command typing doesn't get
        # queued as a soft-inject mid-turn. /stop is NOT in this set; it
        # falls through to the hard-interrupt path below.
        try:
            from . import text_commands as _tcmd
            if _tcmd.is_text_command(raw_text):
                handled = await _tcmd.handle_text_command(
                    runner=self,
                    channel=message.channel,
                    speaker_id=int(message.author.id),
                    raw_text=raw_text,
                    transport=_transport,
                    session_id=str(ch_id),
                    session=inbound.session,
                )
                if handled:
                    return
        except Exception as _tcmd_err:
            print_ts(
                f"{COLOR_YELLOW}text_commands dispatch failed (continuing as normal message): {_tcmd_err}{COLOR_END}",
                agent=self.agent.id,
            )

        active_task = self._active_turns.get(ch_id) if ch_id else None
        is_hard_interrupt = raw_text.lower().startswith("/stop")
        if active_task is not None and not active_task.done():
            if is_hard_interrupt:
                self._hard_interrupt(ch_id)
                # Falls through to the enqueue path below — `/stop` lands
                # as a fresh turn after the cancelled one cleans up.
            else:
                # Soft inject: queue the framed user_text (attribution-
                # preserved) and return WITHOUT enqueueing a new turn.
                # The active turn's drain picks it up at the next boundary.
                self._pending_inject.setdefault(ch_id, []).append(user_text)
                print_ts(
                    f"{COLOR_YELLOW}soft-inject queued on channel {ch_id} "
                    f"(pending count: {len(self._pending_inject[ch_id])}) "
                    f"— from {message.author.name}{COLOR_END}",
                    agent=self.agent.id,
                )
                # Visible confirmation that the queue accepted the message.
                # Without this, operators can't tell their mid-turn message
                # was captured (vs dropped) until the next tool boundary,
                # which discourages the feature. Guarded so a Discord-side
                # reaction failure (permissions, rate-limit) doesn't tear
                # the queue logic.
                try:
                    await message.add_reaction("👀")
                except Exception as _react_err:
                    print_ts(
                        f"{COLOR_YELLOW}soft-inject 👀 reaction failed "
                        f"(continuing): {_react_err}{COLOR_END}",
                        agent=self.agent.id,
                    )
                return

        self._ensure_worker_started()
        await self._inbound_queue.put({
            "channel": message.channel,
            "speaker_id": speaker_id,
            "role_ids": role_ids,
            "owner": owner,
            "user_text": user_text,
            "discord_message": message,
            "inbound": inbound,  # Phase 1: transport-agnostic wrapper alongside legacy refs
            "log_tag": "",
            "depth": 0,
            # Real Discord messages: the operator IS this turn's audience.
            # Any chain that fires out of this turn inherits operator_channel
            # visibility so empty/dead chains surface back to them.
            "originator_visibility": "operator_channel",
        })

    async def run_synthetic_turn(
        self,
        channel_id: int | Session,
        prompt_text: str,
        *,
        auto_post_final_text: bool = False,
        depth: int = 0,
        originator_agent_id: str = "",
        silent: bool = False,
        chain_id: str = "",
        auto_route_from_peer: str = "",
        speaker_id: int = 0,
        speaker_handle: str = "",
        originator_visibility: str = "",
        originator_channel_id: int = 0,
        force_tool_choice: dict | None = None,
    ) -> None:
        """Fire a turn for this agent without an inbound Discord message.

        Used by the cron scheduler / heartbeat system / talk_to_agent. The
        agent runs as if `speaker_id` (or owner_id as fallback) sent
        `prompt_text` in `channel_id`.

        `originator_agent_id` (when set) marks this turn as the recipient
        side of an inter-agent dispatch from that agent. `_run_turn` checks
        it after the final reply is built: if set AND under the depth cap,
        the reply is auto-routed BACK to the originator as another
        synthetic turn instead of posting to Discord here. The originator
        then has a turn where they read the reply and can decide whether
        to continue the chain.

        `speaker_id` (when non-zero) sets the attributed speaker for this
        synthetic turn. talk_to_agent threads the actual originating human
        through this so inter-agent dispatches don't silently spoof the human
        as the speaker on every chain regardless of who triggered it.
        When 0, falls back to owner_id (cron / heartbeat / restart paths).

        `originator_visibility` tags the kind of channel ultimately awaiting
        a visible result from this chain. Values:
          - "operator_channel" — human-initiated turn; empty/dead chains MUST
            surface a hard-failure message to the originating channel.
          - "silent_agent_chain" — agent-initiated chain (one agent pinged
            another spontaneously); empty/dead chains log loudly but stay
            invisible to the human.
          - "cron" — fired by the scheduler; same as silent_agent_chain for
            visibility, but tagged for events.jsonl differentiation later.
          - "heartbeat" — fired by the heartbeat system; same as cron for
            visibility, tagged separately.
          - "" (empty) — legacy / no information; treated like operator_channel
            on the conservative side (better to nag the operator than lose a
            failure silently).
        Propagated through talk_to_agent so the recipient's chain-terminator
        turn knows whether the original requester is watching.
        """
        from .config_global import get_owner_id
        owner_id = get_owner_id("discord")
        if not owner_id:
            print_ts(f"{COLOR_RED}run_synthetic_turn: no owner_id configured{COLOR_END}", error=True, agent=self.agent.id)
            return
        # ATTRIBUTION vs PRIVILEGE split (security-load-bearing — read before editing):
        #
        # `resolved_speaker_id` is for ATTRIBUTION only (history/logs/return
        # routing). Caller-provided non-zero wins; otherwise owner_id is the safe
        # fallback so framework-originated turns (cron / restart_sentinel /
        # heartbeat — no human speaker) read as the owner in history rather than
        # as a phantom user. This fallback is fine for attribution.
        #
        # It must NOT confer owner PRIVILEGE. `owner` (computed below from the
        # *explicitly-passed* speaker_id, NOT resolved_speaker_id) decides whether
        # this turn gets the full owner toolset. If we derived owner from
        # resolved_speaker_id, every framework turn — which passes no speaker_id
        # and thus resolves to owner_id — would silently get the owner toolset,
        # bypassing the additive Session.tool_grants mechanism built for exactly
        # these trusted-but-not-owner turns. So owner privilege requires an
        # explicitly-passed speaker_id equal to the owner (real owner-initiated
        # synthetic turns: /dream, /compact, /stop all thread interaction.user.id).
        # Framework turns (speaker_id=0) get owner=False and must rely on
        # Session.tool_grants for any tools they need.
        resolved_speaker_id = int(speaker_id) if speaker_id else owner_id
        is_owner_turn = bool(speaker_id) and int(speaker_id) == owner_id

        channel = await self._resolve_synthetic_channel(
            channel_id,
            speaker_id=speaker_id,
            speaker_handle=speaker_handle,
        )
        if channel is None:
            # _resolve_synthetic_channel returns None only when the legacy
            # raw-int path hit a genuine "channel not found" — abandon the
            # turn (unchanged). A transient fetch *stall* no longer lands
            # here; it now falls through to the shim inside the resolver.
            return

        ch_label = f"#{getattr(channel, 'name', channel.id)}"
        preview = prompt_text[:80].replace("\n", " ")
        orig_tag = f" (from {originator_agent_id})" if originator_agent_id else ""
        print_ts(f"<- [synthetic{orig_tag}] in {ch_label}: {preview}", agent=self.agent.id)

        self._ensure_worker_started()
        await self._inbound_queue.put({
            "channel": channel,
            "speaker_id": resolved_speaker_id,
            "role_ids": [],
            # owner PRIVILEGE comes from an explicitly-passed owner speaker_id
            # only — NOT from the attribution fallback. See the attribution-vs-
            # privilege comment where is_owner_turn is computed. Framework turns
            # (speaker_id=0) land here as owner=False and use Session.tool_grants.
            "owner": is_owner_turn,
            "user_text": prompt_text,
            "discord_message": None,
            "log_tag": "[synthetic] ",
            "auto_post_final_text": auto_post_final_text,
            "depth": depth,
            "originator_agent_id": originator_agent_id or "",
            "silent": bool(silent),
            "chain_id": chain_id or "",
            "auto_route_from_peer": auto_route_from_peer or "",
            "originator_visibility": originator_visibility or "",
            "originator_channel_id": int(originator_channel_id or 0),
            "force_tool_choice": force_tool_choice,
        })

    async def _resolve_synthetic_channel(
        self,
        channel_id: int | Session,
        *,
        speaker_id: int = 0,
        speaker_handle: str = "",
    ):
        """Resolve the channel object a synthetic turn runs against.

        Centralizes the channel resolution that run_synthetic_turn used to
        inline. Returns a channel-like object the queue worker can use, or
        None to signal the turn should be abandoned. None is returned ONLY
        for the legacy raw-int "channel genuinely not found" case — every
        other path (including a transient fetch stall) returns a usable
        channel so the turn always proceeds.

        Paths preserved exactly:
          - headless agents: TransportChannel over the NullTransport
          - non-Discord transports (iMessage, future): TransportChannel over
            the real transport, with a Session built from speaker_handle so
            ACLs match
          - Discord + caller-passed Session: _SessionChannel over the real
            nextcord channel (sub-case a), or the TransportChannel shim when
            no real channel sits behind the session / the fetch stalls
            (sub-case b)
          - Discord + legacy raw-int id: the real nextcord channel, or the
            TransportChannel shim on a fetch stall

        DIVERGENCE FIX (audit §2): a 15s channel-fetch *timeout* on the
        legacy raw-int path used to log an ERROR and abandon the turn, while
        the Session path logged a WARNING and fell through to the shim. Both
        paths now share the shim-fallback on a stall, so a transient
        fetch_channel hang never silently drops a turn. The legacy shim's
        conversation_id ("discord:<id>") matches the Discord session
        _run_turn would otherwise synthesize, so history stays on the same
        key. A genuine "channel not found" on the legacy path still abandons
        — only the stall behavior changed.
        """
        # Headless agents have no Discord/iMessage channel of their own and no
        # self.bot to resolve one with. They run every turn in a single
        # internal channel — a TransportChannel over the NullTransport. The
        # incoming channel_id is ignored (talk_to_agent passes a sentinel);
        # any visible reply auto-routes back to the originating agent, which
        # posts on its own real transport (see the auto-route block below).
        if self.is_headless:
            from .transports.channel_shim import TransportChannel
            # Honor a caller-passed Session as the source of truth for the
            # conversation (cron/trigger 'reply in a unique session', e.g.
            # internal:email-support): build the internal channel from THAT
            # session so its conversation_id keys history + the in-memory
            # dicts. Only legacy int/sentinel callers (no Session) fall back to
            # the agent's single stable internal session.
            if isinstance(channel_id, Session):
                _session_for_chan = channel_id
            else:
                from .transports.null import make_internal_session
                _session_for_chan = make_internal_session(self.agent.id)
            channel = TransportChannel(
                transport=self.transport,
                session=_session_for_chan,
            )
        elif not hasattr(self.transport, "bot"):
            # Non-headless transport without a Discord bot (iMessage and any
            # future non-Discord transport). The Discord path below would
            # AttributeError on self.bot.get_channel. Wrap the existing
            # transport + a session for channel_id in a TransportChannel —
            # same shim the headless branch uses, just over the real outbound
            # transport instead of NullTransport. Resolves the
            # restart-sentinel continuation failure for iMessage agents.
            from .transports.channel_shim import TransportChannel
            from .session import Session as _Session
            if isinstance(channel_id, Session):
                _session_for_chan = channel_id
            else:
                _t_name = getattr(self.transport, "name", "")
                _ch_int = int(channel_id)
                # Use speaker_handle when caller provided one so iMessage ACLs
                # (auth.imessage.users matching handle strings) succeed for synthetic
                # turns. Without this, the synthesized session has handle="", every
                # iMessage tool ACL fails, tools never reach the API request, and
                # the model emits its tool_use as raw JSON in chat.
                _display = speaker_handle or f"synthetic:{_ch_int}"
                _session_for_chan = _Session(
                    transport=_t_name,
                    transport_id=str(_ch_int),
                    conversation_id=f"{_t_name}:{_ch_int}",
                    speaker_id=speaker_id,
                    speaker_role_ids=[],
                    is_owner=False,
                    is_dm=True,
                    display_name=_display,
                    handle=speaker_handle or "",
                )
            channel = TransportChannel(
                transport=self.transport,
                session=_session_for_chan,
            )
        elif isinstance(channel_id, Session):
            # Discord transport, caller passed a Session. The Session — not a
            # bare channel id — is the source of truth for the conversation, so
            # its conversation_id keys history + the in-memory dicts no matter
            # which sub-case applies:
            #  (a) transport_id IS a real numeric Discord channel (restart
            #      continuations, talk_to_agent into a real channel, per-channel
            #      cron): resolve and post to that real nextcord channel, but
            #      wrap it in _SessionChannel so _run_turn still sees the passed
            #      Session (the real channel can't carry it — __slots__).
            #  (b) the conversation_id has no real Discord channel behind it
            #      (an arbitrary unique trigger / per-thread session id): run
            #      against that conversation_id via the TransportChannel shim,
            #      same as the headless / iMessage branches, so the session's
            #      history stays isolated instead of collapsing onto a real
            #      channel (or the agent's default).
            _passed_session = channel_id
            channel = None
            try:
                _cid_int = int(_passed_session.transport_id)
            except (TypeError, ValueError):
                _cid_int = None
            if _cid_int is not None:
                channel = self.bot.get_channel(_cid_int)
                if channel is None:
                    try:
                        channel = await asyncio.wait_for(
                            self.bot.fetch_channel(_cid_int),
                            timeout=15.0,
                        )
                    except asyncio.TimeoutError:
                        print_ts(f"{COLOR_YELLOW}run_synthetic_turn: fetch_channel({_cid_int}) timed out — using session shim for {_passed_session.conversation_id}{COLOR_END}", agent=self.agent.id)
                        channel = None
                    except Exception:
                        # Not a real Discord channel id — fall through to the shim.
                        channel = None
            if channel is not None:
                # Sub-case (a): real channel, preserve the passed Session.
                channel = _SessionChannel(channel, _passed_session)
            else:
                # Sub-case (b): no real Discord channel behind this session —
                # run against its conversation_id via the shim.
                from .transports.channel_shim import TransportChannel
                channel = TransportChannel(
                    transport=self.transport,
                    session=_passed_session,
                )
        else:
            # Legacy raw-int channel id (cron with channelId, restart-sentinel
            # continuations that pass a bare id).
            _ch_int = int(channel_id)
            channel = self.bot.get_channel(_ch_int)
            if channel is None:
                try:
                    channel = await asyncio.wait_for(
                        self.bot.fetch_channel(_ch_int),
                        timeout=15.0,
                    )
                except asyncio.TimeoutError:
                    # DIVERGENCE FIX (audit §2): previously logged ERROR and
                    # returned here, silently dropping the turn on a transient
                    # fetch stall — unlike the Session branch above, which
                    # warns and falls through to the shim. Unify on the shim
                    # fallback: build a Discord-keyed Session for this id and
                    # wrap it in a TransportChannel. The shim re-resolves the
                    # channel at send time, and its conversation_id
                    # ("discord:<id>") matches the session _run_turn would
                    # otherwise synthesize, so history stays on the same key.
                    print_ts(f"{COLOR_YELLOW}run_synthetic_turn: fetch_channel({_ch_int}) timed out — using session shim{COLOR_END}", agent=self.agent.id)
                    from .transports.channel_shim import TransportChannel
                    from .session import Session as _Session
                    _t_name = getattr(self.transport, "name", "")
                    channel = TransportChannel(
                        transport=self.transport,
                        session=_Session(
                            transport=_t_name,
                            transport_id=str(_ch_int),
                            conversation_id=f"{_t_name}:{_ch_int}",
                            speaker_id=speaker_id,
                            speaker_role_ids=[],
                            is_owner=False,
                            is_dm=True,
                            display_name=speaker_handle or f"synthetic:{_ch_int}",
                            handle=speaker_handle or "",
                        ),
                    )
                except Exception as e:
                    # Genuine "channel not found" (deleted channel / bad id):
                    # abandon the turn (unchanged). Only the stall path above
                    # was changed.
                    print_ts(f"{COLOR_RED}run_synthetic_turn: channel {_ch_int} not found: {e}{COLOR_END}", error=True, agent=self.agent.id)
                    return None
        return channel

    async def _run_turn(
        self,
        *,
        channel,
        speaker_id: int,
        role_ids: list,
        owner: bool,
        user_text: str,
        discord_message: Optional[nextcord.Message],
        log_tag: str,
        inbound: Optional[InboundMessage] = None,  # Phase 1: optional transport-agnostic wrapper; None for synthetic turns
        auto_post_final_text: bool = True,
        depth: int = 0,
        originator_agent_id: str = "",
        silent: bool = False,
        chain_id: str = "",
        auto_route_from_peer: str = "",
        originator_visibility: str = "",
        originator_channel_id: int = 0,
        force_tool_choice: dict | None = None,
    ) -> None:
        """Shared agent loop — calls the model, runs tools, feeds results back,
        loops until the model emits no more tool calls or hits the turn cap.

        Driven by either a real Discord message (_handle_message) or a
        synthetic turn from the cron scheduler (run_synthetic_turn). Posts
        chat replies and tool output to `channel`.
        """
        # ---- Chain-terminator detection ----
        # A chain-terminator turn is an auto-route reply from a peer with no
        # continued chain originator — i.e. the LAST hop of an inter-agent
        # round-trip, landing back here at the initiating agent. As of the
        # 2026-05-19 refactor (Option 4 in chain_terminator_architecture
        # design doc) these turns use the FULL toolset and post plain text
        # normally — the previous narrowed-3-tool / forced-tool_choice design
        # was the root cause of the silent-failure bug class. pipeline.py
        # reads this flag only to inject a [returning from peer] context
        # extension so the model knows what the previous tool result was.
        is_chain_terminator = bool(auto_route_from_peer) and not originator_agent_id

        # Tracks whether a HUMAN soft-inject was drained during this turn.
        # Set True in the three soft-inject drain blocks below. Used to gate
        # the "post inter-agent reply to current channel for human visibility"
        # block in the post-loop final-text section. Without this, an agent
        # working on a peer-triggered (talk_to_agent) task that emits plain
        # text addressed to the human (e.g. "done, shipped the fix")
        # auto-leaks into the operator's DM as unsolicited maintenance noise.
        # The operator explicitly flagged this 2026-05-23 as a flaw.
        _human_softinjected_this_turn = False

        # ---- Superseded chain-terminator: deliver tagged, never drop ----
        # If this turn is an auto-route reply (auto_route_from_peer is set)
        # whose chain_id no longer matches the tracker for that peer — a
        # newer dispatch overwrote it, or the entry was already consumed —
        # this used to RETURN here: no model call, no warning, no posting.
        # That silently ate real answers. Any follow-up dispatch to a peer
        # while their reply was in flight killed that reply on arrival
        # (4 drops in 2 days observed on one deployment, 2026-06-08..10 —
        # incl. an answer the operator was actively waiting on), and it
        # was the main reason an agent "talks to its specialists but never
        # gets back to me." The reply now goes THROUGH, prefixed with a
        # [FRAMEWORK] marker naming it late/possibly-duplicate, and the
        # model reconciles against its own history. The duplicate/fork
        # pollution the hard drop guarded against (parallel-chain failure
        # mode, 2026-05) is handled by the tag instead of by losing data.
        # The tracker is deliberately left alone on mismatch: the pending
        # newer chain still gets its normal consumption at end-of-turn.
        if auto_route_from_peer:
            expected_chain_id = self._current_chain_to.get(auto_route_from_peer, "")
            if expected_chain_id != chain_id:
                print_ts(
                    f"{COLOR_YELLOW}late auto-route delivered (tagged): from "
                    f"'{auto_route_from_peer}' chain={chain_id[:8] or '<empty>'} "
                    f"expected={expected_chain_id[:8] or '<none>'}{COLOR_END}",
                    agent=self.agent.id,
                )
                try:
                    from . import events_log as _events_log
                    _events_log.log_event(
                        self.agent.id, "chain_late_delivery",
                        from_peer=auto_route_from_peer,
                        chain_id=(chain_id or "")[:8],
                        expected=(expected_chain_id or "")[:8],
                    )
                except Exception:
                    pass
                user_text = (
                    f"[FRAMEWORK: late reply from '{auto_route_from_peer}' — it answers an "
                    f"earlier message you sent them, not your most recent one, and may "
                    f"duplicate something you already handled. If you already acted on this "
                    f"answer, ignore it. Otherwise treat it as the reply you were waiting "
                    f"for: relay or act on it NOW — the person who asked has not seen it.]\n"
                    f"{user_text}"
                )

        from .tool_executor import CURRENT_TURN_DEPTH, CURRENT_SPEAKER_ID, CURRENT_CHANNEL_ID, CURRENT_SESSION
        CURRENT_TURN_DEPTH.set(int(depth))
        # Phase 1 of discord-decouple: set CURRENT_SESSION so tools that prefer
        # Session-based routing (send_message, talk_to_agent, fetch_discord_message)
        # see the transport-agnostic wrapper. For synthetic turns where we don't
        # have an inbound (cron, restart_sentinel, chain-terminator), synthesize a
        # Session from the channel + speaker_id so tools still get something
        # consistent. Fallback to None for truly anonymous turns.
        try:
            if inbound is not None:
                CURRENT_SESSION.set(inbound.session)
            elif getattr(channel, "_session", None) is not None:
                # The channel already carries a transport-agnostic Session
                # (TransportChannel — non-Discord transports incl. the headless
                # internal channel). Use it directly so CURRENT_SESSION and the
                # conversation_id below get the right transport prefix
                # ("internal:" / "imessage:") instead of a synthesized Discord one.
                CURRENT_SESSION.set(channel._session)
            else:
                # Synthesize a discord session from the channel/speaker we have.
                from .session import make_discord_session
                _ch_id = int(getattr(channel, "id", 0) or 0)
                if _ch_id:
                    _synth_session = make_discord_session(
                        channel_id=_ch_id,
                        speaker_id=int(speaker_id),
                        speaker_role_ids=list(role_ids) if role_ids else [],
                        is_owner=bool(owner),
                        is_dm=getattr(channel, "type", None) and "dm" in str(getattr(channel, "type", "")).lower(),
                        display_name=getattr(channel, "name", None) or "synthetic",
                    )
                    CURRENT_SESSION.set(_synth_session)
        except Exception as _ctx_err:
            print_ts(f"{COLOR_YELLOW}CURRENT_SESSION set failed (continuing): {_ctx_err}{COLOR_END}", agent=self.agent.id)
        # Set speaker/channel contextvars for the whole turn so tool
        # dispatches and downstream IPC calls see the real speaker ID.
        # Default (0) breaks talk_to_agent's DM-resolution path.
        CURRENT_SPEAKER_ID.set(int(speaker_id))
        try:
            CURRENT_CHANNEL_ID.set(int(getattr(channel, "id", 0) or 0))
        except Exception:
            pass
        # Visibility classification for this turn's chain root. talk_to_agent
        # reads CURRENT_TURN_VISIBILITY when dispatching to a peer, so the
        # peer's chain-terminator turn knows whether the original requester
        # is operator-awaiting (must surface failures) or silent agent/
        # scheduler chain (logs only). Empty string falls through to the
        # conservative default in the empty-reply escalation block (treat
        # as operator_channel — better to nag than lose silently).
        try:
            from .tool_executor import CURRENT_TURN_VISIBILITY
            CURRENT_TURN_VISIBILITY.set(originator_visibility or "")
        except Exception:
            pass
        # Pre-turn self-edit hot-reload. Hash-based fingerprint over agent.json
        # + all system_files (per-agent + _shared/ + CLAUDE.md). If anything
        # changed since last load — whether from this agent's own tool call
        # last turn, another agent editing _shared/, or the owner editing files in
        # his editor — reload_if_changed rebuilds agent.system_message and
        # re-applies it to every live conversation here. Hash (not mtime)
        # means `touch` and identical-content rewrites don't bust the prompt
        # cache for no reason. Cost: ~1ms across ~6 small text files.
        try:
            self.reload_agent_config()
        except Exception as e:
            print_ts(f"{COLOR_RED}pre-turn reload check failed: {e}{COLOR_END}", error=True, agent=self.agent.id)

        # Fresh set of read paths per turn — edit_file enforces 'read first'.
        agent = self.agent

        # Resolve transport + transport-native speaker identity for ACL eval.
        # `speaker_id` is always an int in our internal plumbing; for imessage
        # it's the hashed handle. The ACL system stores `auth.imessage.users`
        # as handle strings, so we pass `session.display_name` (which IS the
        # handle for imessage sessions — see make_imessage_session) when the
        # current transport is imessage. Discord falls through with the int.
        #
        # Multi-transport: prefer the inbound's session transport over
        # self.transport.name — the inbound knows which transport the
        # message actually arrived on. self.transport is the first
        # transport (fine for single-transport agents but wrong when a
        # multi-transport agent receives on a non-primary transport).
        if inbound is not None and getattr(inbound, "session", None) is not None:
            _acl_transport = getattr(inbound.session, "transport", "discord") or "discord"
        else:
            _acl_transport = getattr(self.transport, "name", "discord") or "discord"
        _acl_speaker: object = int(speaker_id)
        _acl_handle: str = ""
        # Per-session tool grants (cron/synthetic sessions). Transport-agnostic:
        # extracted regardless of transport so a granted synthetic turn works on
        # any transport. Additive allow-path only — see Session.tool_grants.
        _acl_tool_grants: list[str] = []
        try:
            from .tool_executor import CURRENT_SESSION as _CS
            _ss = _CS.get(None)
            if _ss is not None:
                _acl_tool_grants = list(getattr(_ss, "tool_grants", None) or [])
                if _ss.transport == "imessage":
                    _acl_transport = "imessage"
                    _acl_speaker = _ss.display_name or ""
                    # Raw handle backs the admin bypass for handle-based transports
                    # (see _check_acl). Source of truth — not the unstable hash.
                    _acl_handle = getattr(_ss, "handle", "") or ""
        except Exception:
            pass
        callable_funcs, system_extension, user_preamble = build_visible_tools(
            agent,
            transport=_acl_transport,
            speaker_id=_acl_speaker,
            speaker_role_ids=role_ids,
            channel_id=channel.id,
            owner=owner,
            chain_terminator_mode=is_chain_terminator,
            handle=_acl_handle,
            tool_grants=_acl_tool_grants,
        )

        # Pull conversation_id from the session if we have one — that's the
        # only place that knows the right transport prefix ("discord:" vs
        # "imessage:" etc). Fall back to the legacy Discord default only
        # when no session is available (truly anonymous synthetic turns).
        _conv_id = ""
        if inbound is not None and getattr(inbound, "session", None) is not None:
            _conv_id = getattr(inbound.session, "conversation_id", "") or ""
        if not _conv_id:
            # Synthetic path — CURRENT_SESSION was synthesized above as Discord.
            try:
                from .tool_executor import CURRENT_SESSION as _CS
                _ss = _CS.get(None)
                if _ss is not None:
                    _conv_id = getattr(_ss, "conversation_id", "") or ""
            except Exception:
                pass
        conv = self.get_conversation(channel.id, conversation_id=_conv_id)
        # Derive `talking_with` so the openflip dashboard can render
        # who's-talking-to-whom. originator_agent_id (forward inter-agent
        # turn) or auto_route_from_peer (chain-terminator return turn)
        # both name the peer. For human turns, owner=True means the owner
        # (the only owner), and synthetic turns from cron/heartbeat have
        # neither — empty string.
        _talking_with = ""
        if originator_agent_id:
            _talking_with = originator_agent_id
        elif auto_route_from_peer:
            _talking_with = auto_route_from_peer
        elif discord_message is not None:
            # Real Discord message — generic "owner" label for the owner,
            # else the speaker's id as a generic human marker. Avoids
            # hardcoding the owner's name so the dashboard stays neutral
            # across deployments.
            _talking_with = "owner" if owner else f"user:{speaker_id}"
        try:
            _agent_state.on_turn_start(agent.id, channel, talking_with=_talking_with)
        except Exception:
            pass
        # events.jsonl: framework-wide observability feed for Activity tab.
        # turn_start fires on every turn; the Flask side filters by kind.
        try:
            from . import events_log as _events_log
            _events_log.log_event(
                agent.id, "turn_start",
                channel_id=int(getattr(channel, "id", 0) or 0),
                channel_name=getattr(channel, "name", "DM"),
                talking_with=_talking_with,
                synthetic=bool(originator_agent_id or auto_route_from_peer
                               or discord_message is None),
            )
        except Exception:
            pass

        # Image attachments → download + queue for vision.
        # Only fires for anthropic agents (ollama vision path isn't wired
        # in this framework). Filters Discord attachments to image/* and
        # downloads each into a tempfile, then appends to the
        # conversation's _pending_image_attachments list which
        # AnthropicConversation drains on its next chat() / chat_stream().
        _img_tmp_paths: list[str] = []
        if (agent.provider == "anthropic"
                and discord_message is not None
                and getattr(discord_message, "attachments", None)):
            try:
                import aiohttp, tempfile, os as _os
                from .pipeline import extract_image_attachments as _eia
                img_meta = _eia(discord_message)
                if img_meta:
                    pending = list(getattr(conv, "_pending_image_attachments", None) or [])
                    async with aiohttp.ClientSession() as _http:
                        for entry in img_meta:
                            url = entry.get("url")
                            if not url:
                                continue
                            try:
                                async with _http.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                                    if r.status != 200:
                                        continue
                                    data = await r.read()
                            except Exception as _dl_err:
                                print_ts(
                                    f"{COLOR_YELLOW}image attachment download failed: {_dl_err}{COLOR_END}",
                                    agent=agent.id,
                                )
                                continue
                            fn = entry.get("filename") or "image"
                            ct = entry.get("content_type") or "image/png"
                            try:
                                fd, path = tempfile.mkstemp(prefix="openflip_inbound_", suffix="_" + fn)
                                with _os.fdopen(fd, "wb") as f:
                                    f.write(data)
                            except Exception as _io_err:
                                print_ts(
                                    f"{COLOR_YELLOW}image temp-write failed: {_io_err}{COLOR_END}",
                                    agent=agent.id,
                                )
                                continue
                            _img_tmp_paths.append(path)
                            pending.append({"path": path, "content_type": ct, "filename": fn.lower()})
                    if pending:
                        conv._pending_image_attachments = pending
                        print_ts(
                            f"  ← inbound: queued {len(_img_tmp_paths)} image attachment(s) for vision",
                            agent=agent.id,
                        )
            except Exception as _img_err:
                print_ts(
                    f"{COLOR_YELLOW}inbound image queue failed (continuing without vision): {_img_err}{COLOR_END}",
                    agent=agent.id,
                )

        # Optional per-message system extension (e.g. "tools the speaker can't use").
        original_system = conv.system_message
        if system_extension:
            conv.system_message = original_system + "\n" + system_extension
            if conv.messages and conv.messages[0].role == 'system':
                conv.messages[0]['content'] = conv.system_message

        if self.agent.provider == "anthropic":
            from .anthropic_conversation import ChatMessage
        else:
            from openflip.ollama_api import ChatMessage
        CHAT_TIMEOUT_S = 300
        MAX_TOOL_TURNS = 100
        callable_names = {f.__name__ for f in callable_funcs}
        media_only = (agent.tool_response_mode == "media_only")
        any_attachments_this_turn = False
        any_tool_called = False
        # Tracks whether any reply-equivalent tool (send_message, end_chain)
        # has already dispatched. If so, the post-loop final-text path knows
        # the reply IS the tool call and doesn't double-emit.
        reply_equivalent_tool_fired = False
        # Terminal-result-contract tracker. Set True at every place that
        # successfully posts assistant-generated content (text or attachment)
        # to the operator's channel. Checked at the bottom of _run_turn — if
        # this stays False AND no diagnostic-equivalent posting happened,
        # the function emits a "⚠️ turn ended without visible reply"
        # diagnostic instead of letting the operator see silence. Models
        # Claude Code's `isResultSuccessful` exit check.
        _posted_assistant_text = False
        # Captured provider/framework error string for this turn. Set when the
        # in-loop framework-error branch fires (non-200 from the API: rate
        # limit / overload / auth / 400). The terminal-result contract at
        # function exit prefers this over the generic empty-reply diagnostic
        # so the operator sees the REAL provider error instead of a misleading
        # "known bug" catch-all. None = no API error captured this turn.
        _captured_framework_error: str | None = None
        # Cross-iteration tool-call dedup. A (function_name, sorted-args-json)
        # signature is added here ONLY for calls that succeeded — failed
        # calls stay retry-able. Mirrors Claude Code's `runTools` semantics
        # of permitting retry of failed tools while blocking accidental
        # double-fires of successful ones.
        called_signatures: set[tuple[str, str]] = set()

        # Add the user's message once at the start of the turn. Per-speaker
        # access notes ride on the user message (not the system prompt) to
        # keep the cached system prefix byte-stable across speaker rotations.
        if user_preamble:
            framed_user = f"{user_preamble}\n\n---\n\n{user_text}"
        else:
            framed_user = user_text
        conv.messages.append(ChatMessage('user', framed_user))

        async def _restore_system():
            if system_extension:
                conv.system_message = original_system
                if conv.messages and conv.messages[0].role == 'system':
                    conv.messages[0]['content'] = original_system
            # Clean up any image tempfiles staged for this turn. Chat() has already
            # consumed them by now (read + base64); deleting is safe regardless of
            # whether the turn ended normally, timed out, or errored.
            for _p in _img_tmp_paths:
                try:
                    os.unlink(_p)
                except Exception:
                    pass
            _img_tmp_paths.clear()

        # Discord typing indicator is UX polish — wrap so a 429 on the
        # /typing endpoint doesn't take down the whole turn. When the bot
        # is rate-limited at the REST level (e.g. fresh after a reconnect
        # storm) the typing call fails before the model is even called.
        # Before this guard, that turned every message into "Internal error:
        # 429" even though the model and tools were perfectly capable of
        # answering.
        # Typing-indicator context managers live in discord_io.py. Bind them
        # to this turn's args here so the call site below stays clean.
        def _safe_typing(ch):
            return _discord_safe_typing(ch, agent_id=agent.id)

        def _silent_typing(_ch):
            # Headless agents have no self.bot to resolve the human's DM for a
            # typing indicator. Fall back to the channel's own (no-op) typing
            # context so a silent synthetic turn doesn't crash on self.bot.
            # Headless agents AND non-Discord transports (iMessage, etc) have no
            # self.bot — reaching for it AttributeErrors. Both cases fall back to
            # the channel's own (no-op) typing context so a silent synthetic turn
            # doesn't crash before the model is ever called. Only a real Discord
            # bot gets the human-DM typing indicator.
            _has_discord_bot = any(
                getattr(t, "name", "") == "discord" and hasattr(t, "bot")
                for t in self._transports
            )
            if self.is_headless or not _has_discord_bot:
                return _safe_typing(_ch)
            return _discord_silent_typing(self.bot, int(speaker_id or 0), agent_id=agent.id)

        # Hoisted out of the `async with` so the CancelledError handler below
        # can read what the model last produced (text snippet or tool name)
        # and write a richer interrupt marker. Without this, the resume
        # marker has no context about what got cut off, and the agent has
        # nothing to decide "resume vs drop" from. 2026-05-19 fix for the
        # "I'm scared to interrupt you mid-task" failure mode.
        # Claude-Code-style query loop. Single exit gate: `needs_follow_up`
        # is True iff the assistant message contained at least one tool_use
        # block. No tool_use → loop ends, post-loop section handles final
        # text + stop_hooks + terminal-contract diagnostic. Mirrors
        # queryLoop in claude-code's src/query.ts at line 241.
        #
        # Architectural reason: openflip used to layer three inline retry
        # guards on top of the natural exit (empty_retry_attempted,
        # post_tool_empty_retry_attempted, promise_retry_attempted). Each
        # tried to force the model into firing a tool after prose. The
        # model kept inventing new phrasings to escape the regex/heuristic
        # nudges, and the band-aids decayed faster than they could be
        # added. Claude Code's loop does NOT retry — prose-without-tool is
        # a legitimate exit. Failures-to-act were intended to be caught by
        # a post-turn stop_hook layer that fires ONE follow-up synthetic
        # turn with tool_choice=any. NOTE: that stop_hook layer is NOT
        # IMPLEMENTED yet — only the terminal-contract diagnostic exists
        # (it warns but does not retry). The primary defense is the
        # FRAMEWORK.md "Action-promise STOP-TEST" rule (prompt-level
        # coupling of announce + act in the same response).
        last_ai_message = None
        turn_count = 0
        # Track soft-inject drain across finally/post-loop so the follow-up
        # turn fire-check below sees the count even after the safety-net
        # drain in `finally:` has popped _pending_inject.
        _drained_count = 0
        _drained_texts: list[str] = []
        # Turn-cumulative flag: True once ANY human soft-inject is drained at
        # ANY point this turn — per-iteration tool-dispatch drain (line ~1957),
        # finally safety-net drain (~2226), or post-loop drain (~2286). This is
        # deliberately NOT _drained_count: the per-iteration drain pops
        # _pending_inject before finally runs, so it never reaches _drained_count
        # (which therefore only ever reflects finally/post-loop drains, not the
        # post_drain_retry path). This flag overrides media_only attachment-
        # suppression at the final-text post-sites: when the operator spoke
        # mid-turn, the agent's reply MUST post even though the turn also made
        # media. Without it, that reply lands in history and never reaches Discord.
        _human_softinject_drained_this_turn = False
        try:
            async with (_silent_typing(channel) if silent else _safe_typing(channel)):
                while True:
                    turn_count += 1
                    if turn_count > MAX_TOOL_TURNS:
                        # Mirror queryLoop's max_turns return ({ reason: 'max_turns' }).
                        print_ts(f"{COLOR_YELLOW}{log_tag}Tool loop exceeded {MAX_TOOL_TURNS} turns; aborting.{COLOR_END}", agent=agent.id)
                        if auto_post_final_text:
                            try:
                                await _safe_channel_send(channel, f"⚠️ Tool loop exceeded {MAX_TOOL_TURNS} turns. Stopping. Try `/reset` and rephrase.")
                            except Exception:
                                pass
                        break

                    # Per-iteration soft-inject drain count. Set in the inner-
                    # loop drain block (after tool_results) and consumed by
                    # the produced_attachment break check. Re-initialised at
                    # the top of every iteration so it ONLY reflects drains
                    # that happened during THIS iteration's tool dispatch.
                    # The previous design used a turn-wide counter
                    # (_drained_during_turn) which fired the retry on stale
                    # state after the model had already responded to the
                    # marker in a subsequent chat() — sending the assistant
                    # message as the trailing entry, which Opus rejects
                    # with 400 "model does not support assistant message
                    # prefill" (incident 2026-05-25).
                    _drained_this_iter = 0

                    _prov = agent.provider or "ollama"
                    print_ts(
                        f"  → {_prov} chat ({len(conv.messages)} msgs, {len(callable_funcs)} tools, turn {turn_count}) {log_tag}".rstrip(),
                        agent=agent.id,
                    )

                    # Pre-flight trim removed per operator directive (2026-05-22):
                    # trim now fires only inside chat()/chat_stream() on retry-
                    # after-prompt-too-long. Anthropic's server-side auto-
                    # compaction handles normal overflow.

                    _pre_notice_fired = False
                    _compact_started_at = None
                    if getattr(conv, "force_compact_next", False) and not silent:
                        # safe_channel_send returns the Message on success,
                        # None on timeout/transport error (it logs its own
                        # failure). Gate _pre_notice_fired on actual
                        # delivery so the post-notice below can fire as a
                        # fallback when the pre-notice silently drops —
                        # without that, a single rate-limited send makes
                        # /compact invisible in the channel even though
                        # the work happened.
                        import time as _time
                        _compact_started_at = _time.monotonic()
                        _pre_sent = await _safe_channel_send(channel, "⚙️ *Compacting conversation...*")
                        _pre_notice_fired = _pre_sent is not None

                    # force_tool_choice forwarded only on the FIRST iteration
                    # of the turn (not on subsequent tool-result continuations,
                    # which would loop). Skipped for chain-terminator turns —
                    # their narrow toolset already covers the valid exits.
                    _tc_this_turn = None
                    if (force_tool_choice is not None
                            and turn_count == 1
                            and not is_chain_terminator):
                        _tc_this_turn = force_tool_choice
                    # Sticky override from action-promise retry. Consumed
                    # immediately so it only fires once.
                    if locals().get("_force_next_tc"):
                        _tc_this_turn = _force_next_tc
                        _force_next_tc = None

                    # ===== Call the model =====
                    # `chat()` consumes `chat_stream()` internally (see
                    # anthropic_conversation.chat()) and returns a complete
                    # AnthropicAIChatMessage once the stream ends. By the
                    # time this await resolves we know every tool_use block
                    # the model emitted — that's the data Claude Code's
                    # queryLoop also has at the equivalent point (after the
                    # `for await (const message of deps.callModel(...))`
                    # exhausts).
                    ai_message = await asyncio.wait_for(
                        conv.chat(tools=callable_funcs or None, think=agent.think,
                                  tool_choice=_tc_this_turn),
                        timeout=CHAT_TIMEOUT_S,
                    )
                    last_ai_message = ai_message

                    # Compaction done-notice. Fires whenever compaction
                    # actually happened this turn; the helper picks the right
                    # wording from whether the operator triggered it manually
                    # (pre-notice fired) or it happened spontaneously.
                    if getattr(conv, "compacted_this_turn", False) and not silent:
                        _elapsed = None
                        if _compact_started_at is not None:
                            import time as _time
                            _elapsed = _time.monotonic() - _compact_started_at
                        await _notify_compaction_done(
                            channel, was_manual=_pre_notice_fired, elapsed_s=_elapsed,
                        )

                    # Inject a [FRAMEWORK] note when compaction fires so the
                    # agent sees on its NEXT turn that fine-grained history
                    # was summarized away. Without this, agents fill the gap
                    # with intuition instead of search_memory / read_memory.
                    if getattr(conv, "compacted_this_turn", False):
                        try:
                            conv.messages.append(ChatMessage(
                                'user',
                                '[FRAMEWORK]: Your conversation was just compacted — '
                                'recent fine-grained history is now a summary. If you '
                                'need exact details about earlier work, use search_memory '
                                'or read_memory rather than guessing from context.'
                            ))
                            print_ts(
                                f"{COLOR_YELLOW}  compaction — injected [FRAMEWORK] note for next turn{COLOR_END}",
                                agent=agent.id,
                            )
                        except Exception:
                            pass

                    # ===== Framework error (auth/rate/transport/bad-request) =====
                    # chat_stream returns a FrameworkErrorEvent for non-200
                    # responses; chat() wraps it in an AnthropicAIChatMessage
                    # with is_framework_error=True. Mirrors Claude Code's
                    # error path in queryLoop's outer try/catch — surface
                    # the error, don't pollute history with the error string,
                    # exit cleanly.
                    if getattr(ai_message, "is_framework_error", False):
                        # Capture the real provider error for this turn so the
                        # terminal-result contract surfaces it instead of the
                        # generic empty-reply diagnostic.
                        _captured_framework_error = (
                            getattr(ai_message, "content_text", None) or
                            getattr(ai_message, "content", "") or "framework error"
                        )
                        if auto_post_final_text and not silent:
                            _err_text = (getattr(ai_message, "content_text", None) or
                                         getattr(ai_message, "content", "") or "")
                            try:
                                for chunk in _split_for_discord(_err_text):
                                    await _safe_channel_send(channel, chunk)
                                # The error WAS shown to the operator — mark the
                                # turn as having produced visible output so the
                                # terminal contract doesn't double-post it below.
                                _posted_assistant_text = True
                            except Exception:
                                pass
                        if conv.messages and conv.messages[-1].role == 'user':
                            conv.messages.pop()
                        _err_text = (getattr(ai_message, "content_text", None) or
                                     getattr(ai_message, "content", "") or "framework error")
                        _err_preview = str(_err_text)[:200].replace("\n", " ")
                        try:
                            conv.messages.append(ChatMessage(
                                'user',
                                f'[FRAMEWORK]: Previous turn failed before a reply was generated. Reason: {_err_preview}'
                            ))
                        except Exception:
                            pass
                        print_ts(
                            f"{COLOR_YELLOW}  framework error — popped trailing user, injected [FRAMEWORK] note{COLOR_END}",
                            agent=agent.id,
                        )
                        break

                    _tc = getattr(ai_message, "tool_calls", None) or []
                    _ct = (getattr(ai_message, "content_text", None) or
                           getattr(ai_message, "content", "") or "")
                    _done = ""
                    _raw = getattr(ai_message, "raw_response", None)
                    if _raw is not None:
                        _done = getattr(_raw, "done_reason", "") or ""
                    if _tc:
                        print_ts(f"  ← {_prov} replied  tool_calls={[t.function_name for t in _tc]} done={_done}", agent=agent.id)
                    elif _ct.strip():
                        print_ts(f"  ← {_prov} replied  text={_ct[:80].replace(chr(10),' ')!r} done={_done}", agent=agent.id)
                        # Action-promise retry: if the text reads like an
                        # action-commitment ("lemme look", "imma do it",
                        # etc.) but no tool_use accompanied it, retry the
                        # same turn forcing tool_choice=any so the model
                        # must emit a tool. Cap at 1 retry per turn.
                        # Kill switch: OPENFLIP_DISABLE_ACTION_PROMISE_RETRY=1.
                        # Decision logic extracted to turn_retries.py.
                        if action_promise_should_retry(
                                _ct, bool(locals().get("_force_tool_retry_used"))):
                            _force_tool_retry_used = True
                            print_ts(
                                f"{COLOR_YELLOW}{log_tag}action-promise detected without tool — retrying with tool_choice=any{COLOR_END}",
                                agent=agent.id,
                            )
                            # Sticky flag — survives the loop iteration reset.
                            # Picked up by the tool_choice assignment block
                            # at the top of the next iteration.
                            _force_next_tc = {"type": "any"}
                            continue
                        # Peer-prose leak detection. If text starts with
                        # "<peer_agent_id>: " (or "<peer_agent_id> ,"
                        # or "<peer_agent_id> —"), the model is addressing
                        # another agent in prose but did NOT fire
                        # talk_to_agent. Without intervention the text
                        # auto-routes to whoever triggered this turn
                        # (often the operator), leaking inter-agent
                        # prose into the wrong channel.
                        #
                        # Behavior: inject a [FRAMEWORK] nudge naming the
                        # detected peer and require the model to either
                        # (a) re-emit using talk_to_agent, or
                        # (b) rewrite the reply for the actual reader.
                        # Cap at 1 retry per turn to bound cost.
                        # Kill switch: OPENFLIP_DISABLE_PEER_PROSE_RETRY=1.
                        # Detection logic extracted to turn_retries.py
                        # (line-by-line scan, fenced-code skipping, kill
                        # switch + one-shot gating all inside the helper).
                        _detected_peer = detect_peer_prose(
                            _ct,
                            agent.id,
                            lambda _pid: RUNNERS.get(_pid) is not None,
                            bool(locals().get("_peer_prose_retry_used")),
                        )
                        if _detected_peer:
                            _peer_prose_retry_used = True
                            print_ts(
                                f"{COLOR_YELLOW}{log_tag}peer-prose leak detected "
                                f"(addressed '{_detected_peer}' without talk_to_agent) "
                                f"— injecting nudge and retrying{COLOR_END}",
                                agent=agent.id,
                            )
                            try:
                                if agent.provider == "anthropic":
                                    from .anthropic_conversation import ChatMessage as _CM
                                else:
                                    from openflip.ollama_api import ChatMessage as _CM
                                _nudge = build_peer_prose_nudge(_detected_peer)
                                conv.messages.append(_CM('user', _nudge))
                            except Exception as _nudge_err:
                                print_ts(
                                    f"{COLOR_YELLOW}{log_tag}peer-prose nudge "
                                    f"injection failed (continuing without retry): "
                                    f"{_nudge_err}{COLOR_END}",
                                    agent=agent.id,
                                )
                            else:
                                continue
                    else:
                        # Empty reply — no text AND no tool_use. Inject a
                        # nudge into history before retrying so the API
                        # call has different input. A bare retry would
                        # send the same body and get the same empty back.
                        print_ts(
                            f"{COLOR_YELLOW}  ← {_prov} replied  EMPTY (no text, no tool_calls) done={_done}{COLOR_END}",
                            agent=agent.id,
                        )
                        # Kill switch: OPENFLIP_DISABLE_EMPTY_RETRY=1 falls
                        # through to the normal break path (model emits empty,
                        # turn ends with the operator-visible warn from the
                        # terminal-contract diagnostic).
                        # Gate (kill switch + one-shot) + nudge text extracted
                        # to turn_retries.empty_retry_nudge.
                        nudge = empty_retry_nudge(bool(locals().get("_empty_retry_used")))
                        if nudge is not None:
                            _empty_retry_used = True
                            print_ts(
                                f"{COLOR_YELLOW}{log_tag}empty response — injecting nudge and retrying{COLOR_END}",
                                agent=agent.id,
                            )
                            try:
                                from .anthropic_conversation import ChatMessage as _CM
                            except Exception:
                                _CM = None
                            if _CM is not None:
                                conv.messages.append(_CM("user", nudge))
                                # Mark so we can pop after the retry call so
                                # the nudge doesn't pollute history forever.
                                _nudge_to_pop = True
                            continue

                    # Don't persist empty assistant responses. Verified against
                    # Claude Code source 2026-05-26: their `I5z` sanitizer
                    # writes "(no content)" to empty assistants but ONLY as
                    # a request-prep transform applied in-memory at the
                    # message-construction path — it does NOT persist to
                    # their on-disk format. We previously mutated the bare
                    # `ai_message["content"]` field and persisted it, which
                    # fed the literal "(no content)" string back to Anthropic
                    # on every subsequent turn as the canonical response
                    # pattern, creating the (no content) loop that bricked
                    # the maintainer agent 2026-05-25. the maintainer agent's review: cleaner shape for our
                    # architecture is to gate at the persistence boundary
                    # (skipping append when empty) rather than mirror their
                    # sanitizer downstream — we don't have their UI render
                    # constraint, our on-disk format is purely conversation
                    # memory. user→user history after a dropped empty is
                    # valid for Anthropic; the next real assistant turn
                    # restores alternation naturally.
                    _ai_ct = (getattr(ai_message, "content_text", None) or
                              getattr(ai_message, "content", "") or "")
                    _ai_calls = getattr(ai_message, "tool_calls", None) or []
                    if not _ai_ct.strip() and not _ai_calls:
                        print_ts(
                            f"{COLOR_YELLOW}{log_tag}empty assistant response — "
                            f"dropping from history (not persisted){COLOR_END}",
                            agent=agent.id,
                        )
                    else:
                        # Append assistant message to history (must precede the
                        # needs_follow_up break so the conversation log reflects
                        # what the model said, even on a no-tool exit).
                        conv.messages.append(ai_message)

                    # Intra-loop text posting: when the model emits text AND
                    # tool_calls in the same iteration, post the text to the
                    # operator's channel right now. This is a Discord-specific
                    # affordance; Claude Code's CLI streams text-deltas live
                    # so there's no equivalent intra-loop post.
                    if _tc and _ct.strip() and auto_post_final_text and not silent:
                        try:
                            for chunk in _split_for_discord(_ct.strip()):
                                await _safe_channel_send(channel, chunk)
                        except Exception:
                            pass

                    # ===== THE GATE: needs_follow_up =====
                    # No tool_use → loop ends. This is the structural exit
                    # that mirrors Claude Code's `if (!needsFollowUp) {
                    # return { reason: 'completed' } }` at query.ts:1062.
                    # All four ex-band-aids (empty_retry_attempted,
                    # post_tool_empty_retry_attempted, promise_retry_attempted,
                    # force_tool_choice continue) are GONE — the structural
                    # exit replaces them. Promise-without-action is handled
                    # primarily by the FRAMEWORK.md "Action-promise STOP-TEST"
                    # prompt rule, BACKSTOPPED by the stop_hook layer just
                    # below — see `openflip/stop_hooks.py` for the hook
                    # registry and the promise_without_action regex.
                    needs_follow_up = bool(_tc)
                    if not needs_follow_up:
                        # ----- Stop-hook layer -----
                        # Mirrors Claude Code's `handleStopHooks` pattern: a
                        # text-only turn (no tool_use) gets one chance to be
                        # rewritten/extended if any registered hook decides
                        # the reply is malformed. The current single hook,
                        # `promise_without_action`, catches text like
                        # "checking…" / "let me look" / "on it" that leaves
                        # the operator staring at a dangling promise.
                        #
                        # Depth-cap: `_promise_hook_used` is a one-shot flag
                        # local to THIS _run_turn invocation. We allow ONE
                        # retry per turn — same convention as the other one-
                        # shot flags above (`_empty_retry_used`,
                        # `_force_tool_retry_used`, `_peer_prose_retry_used`).
                        # A second misfire after a nudge is a deeper model
                        # failure that an infinite-retry loop would only mask.
                        if not locals().get("_promise_hook_used"):
                            try:
                                # Invocation extracted to turn_retries.run_stop_hooks
                                # (thin wrapper around stop_hooks.evaluate_stop_hooks).
                                _shr = run_stop_hooks(
                                    agent_id=agent.id,
                                    channel_id=int(getattr(channel, "id", 0) or 0),
                                    assistant_text=_ct,
                                    tool_was_called=bool(_tc),
                                    depth=int(depth),
                                    is_chain_terminator=is_chain_terminator,
                                    is_synthetic=str(log_tag or "").strip().startswith("[synthetic]"),
                                    originator_visibility=originator_visibility or "",
                                )
                            except Exception as _stop_err:
                                _shr = None
                                print_ts(
                                    f"{COLOR_YELLOW}{log_tag}stop_hook evaluation failed "
                                    f"(continuing without retry): {_stop_err}{COLOR_END}",
                                    agent=agent.id,
                                )
                            if _shr is not None and _shr.blocked:
                                _promise_hook_used = True
                                print_ts(
                                    f"{COLOR_YELLOW}{log_tag}stop_hook fired: {_shr.reason} "
                                    f"— injecting nudge + forcing tool_choice=any and retrying{COLOR_END}",
                                    agent=agent.id,
                                )
                                try:
                                    if _shr.suggested_user_message:
                                        conv.messages.append(
                                            ChatMessage('user', _shr.suggested_user_message)
                                        )
                                    # CRITICAL: force tool_choice=any on the retry so the
                                    # API mechanically REQUIRES a tool_use block in the
                                    # response. Without this, the model just reads the
                                    # nudge and emits another text-only "okay sorry"
                                    # reply — same defective shape, same operator
                                    # frustration. This mirrors claude_code's
                                    # forceToolUseRetry path (see audits/claude_code_full_port).
                                    _force_next_tc = {"type": "any"}
                                    continue
                                except Exception as _inject_err:
                                    print_ts(
                                        f"{COLOR_YELLOW}{log_tag}stop_hook nudge injection "
                                        f"failed (falling through to exit): "
                                        f"{_inject_err}{COLOR_END}",
                                        agent=agent.id,
                                    )
                        # NOTE: no post_drain_retry here.
                        #
                        # The text-only break can only fire AFTER chat() has
                        # already run with the latest history visible. Any
                        # [FRAMEWORK] marker from a prior iteration's inner-
                        # loop drain was therefore part of the input the model
                        # just responded to — the model has already addressed
                        # it (in its own judgement). Forcing another chat()
                        # iteration here would re-send the same messages list
                        # with the just-emitted assistant text as the trailing
                        # entry, which Opus rejects with 400 "model does not
                        # support assistant message prefill. conversation
                        # must end with a user message." The structural retry
                        # for "drain happened but model never got a chat()"
                        # lives ONLY in the produced_attachment branch below,
                        # where the break would otherwise exit BEFORE the
                        # next chat() runs.
                        break

                    # ===== Tool dispatch =====
                    # Cross-iteration + in-batch dedup. Successful calls are
                    # promoted to called_signatures below; failed calls stay
                    # retry-able (mirror of Claude Code's runTools semantics).
                    new_calls = []
                    new_call_sigs: list[tuple[str, str]] = []
                    batch_sigs: set[tuple[str, str]] = set()
                    seen_dup = False
                    import json as _json
                    for t in _tc:
                        try:
                            sig = (t.function_name, _json.dumps(t.args or {}, sort_keys=True, default=str))
                        except Exception:
                            sig = (t.function_name, repr(t.args or {}))
                        if t.function_name in _DEDUPE_EXEMPT_TOOLS:
                            # Intentionally repeatable with identical args
                            # (e.g. delete_message walking back through
                            # history). Skip both the dup check and sig
                            # tracking so it never gets suppressed.
                            new_calls.append(t)
                            new_call_sigs.append(sig)
                            continue
                        if sig in called_signatures or sig in batch_sigs:
                            seen_dup = True
                            print_ts(
                                f"{COLOR_YELLOW}Duplicate tool call suppressed: {t.function_name}({t.args}){COLOR_END}",
                                agent=agent.id,
                            )
                            continue
                        batch_sigs.add(sig)
                        new_calls.append(t)
                        new_call_sigs.append(sig)
                    if not new_calls:
                        # Every call in this iteration was a duplicate of a
                        # successful prior call. Routing-tool repeats (end_chain
                        # / send_message / talk_to_agent) terminate the loop —
                        # feeding "try a different approach" back to the model
                        # just makes it call the same routing tool again. See
                        # inter-agent log loop 2026-05-13. Non-routing repeats
                        # get a feedback note and a continued loop so the
                        # model can finalize without losing context.
                        if seen_dup:
                            _routing_tools = {"end_chain", "send_message", "talk_to_agent"}
                            if all(t.function_name in _routing_tools for t in _tc):
                                print_ts(
                                    f"{COLOR_YELLOW}{log_tag}routing call already fired; "
                                    f"breaking loop instead of re-prompting{COLOR_END}",
                                    agent=agent.id,
                                )
                                break
                            feedback_msg = (
                                'Note: your previous tool call(s) were duplicates of calls '
                                'already executed this turn and were skipped. Either try a '
                                'different approach or finalize your reply.'
                            )
                            conv.messages.append(ChatMessage('tool', feedback_msg))
                            continue
                        break
                    ai_message.tool_calls = new_calls

                    _ch_id_ic = int(getattr(channel, "id", 0) or 0)
                    tool_results = await execute_tool_calls(
                        agent=agent,
                        conversation=conv,
                        ai_message=ai_message,
                        callable_tool_names=callable_names,
                        channel=channel,
                        speaker_id=speaker_id,
                        discord_message=discord_message,
                        silent=silent,
                        interrupt_check=(
                            (lambda: bool(self._pending_inject.get(_ch_id_ic)))
                            if _ch_id_ic else None
                        ),
                    )

                    # Promote SUCCESSFUL call sigs into called_signatures so
                    # they can't repeat in later iterations. Failed calls
                    # stay retry-able. Mapping is 1:1 because
                    # execute_tool_calls returns one entry per processed call
                    # (including ACL-blocked / lock-held / dry-run).
                    for (_tname, _tres), _sig in zip(tool_results, new_call_sigs):
                        if _tres.ok:
                            called_signatures.add(_sig)

                    if tool_results:
                        any_tool_called = True

                    # Track reply-equivalent tools (send_message routes text
                    # to the operator, end_chain terminates the chain). The
                    # post-loop final-text path uses this to avoid double-
                    # posting when send_message already delivered the reply.
                    _REPLY_EQUIVALENT_TOOLS = {"send_message", "end_chain"}
                    for _name, _ in tool_results:
                        if _name in _REPLY_EQUIVALENT_TOOLS:
                            reply_equivalent_tool_fired = True
                            break
                    produced_attachment = False
                    for _, r in tool_results:
                        if r.attachments:
                            any_attachments_this_turn = True
                            produced_attachment = True

                    if not tool_results:
                        # All calls were ACL-blocked/locked/dry-run and yielded
                        # zero feedback messages. Same as no-tool-fired —
                        # break and let post-loop handling take over.
                        break

                    # Pair each result with its originating call so we can
                    # stamp tool_use_id onto the tool ChatMessage. The
                    # anthropic provider needs this pairing to emit a valid
                    # tool_use → tool_result round-trip on the next turn;
                    # without it, the API 400s on orphan tool_use blocks.
                    for (tname, tres), _tc_obj in zip(tool_results, new_calls):
                        feedback = build_model_feedback(tname, tres)
                        tool_msg = ChatMessage('tool', feedback)
                        tu_id = getattr(_tc_obj, "tool_use_id", "") or ""
                        if tu_id:
                            tool_msg["tool_use_id"] = tu_id
                        conv.messages.append(tool_msg)

                    # Soft-inject drain. Must happen AFTER tool_results are
                    # appended (so we don't break the tool_use→tool_result
                    # pairing Anthropic requires) and BEFORE the next chat()
                    # call (so the model sees the operator's mid-turn
                    # message at its next decision point). See
                    # _drain_pending_injects for the marker format.
                    try:
                        _ch_id_drain = int(getattr(channel, "id", 0) or 0)
                        _drained_n = self._drain_pending_injects(_ch_id_drain, conv)
                        if _drained_n > 0:
                            # Turn-cumulative: this per-iteration drain is the
                            # primary bug path (post_drain_retry below continues
                            # the loop so the model replies to the operator, but
                            # _drained_count never sees this drain). Mark it here
                            # so the reply survives media_only suppression.
                            _human_softinject_drained_this_turn = True
                            # Per-iteration counter consumed by the
                            # produced_attachment break check below. Stays
                            # local to THIS iteration so a stale value from
                            # an earlier iteration cannot mis-fire the
                            # retry after the model has already responded
                            # to the marker on the next chat() call.
                            _drained_this_iter = _drained_n
                        # If a human soft-injected during a synthetic turn
                        # (e.g. operator messaged us mid-talk_to_agent
                        # exchange), the silent-by-default visibility no
                        # longer holds — flip auto_post_final_text on so
                        # subsequent text reaches the operator's channel.
                        # Soft-injects only come from the _handle_message
                        # path (human transports), so any drain > 0 is a
                        # human speaking. See 2026-05-22 routing_bug notes.
                        if _drained_n > 0 and (not auto_post_final_text or silent):
                            auto_post_final_text = True
                            silent = False
                            _human_softinjected_this_turn = True
                            print_ts(
                                f"{COLOR_YELLOW}soft-inject from human during synthetic turn "
                                f"— flipping visibility on (auto_post=True, silent=False){COLOR_END}",
                                agent=agent.id,
                            )
                    except Exception as _drain_err:
                        print_ts(
                            f"{COLOR_YELLOW}inner-loop soft-inject drain failed (continuing): "
                            f"{_drain_err}{COLOR_END}",
                            agent=agent.id,
                        )

                    # Attachment-satisfied early exit: an attachment is a
                    # terminal artifact for the operator (Claude Code has a
                    # similar pattern where a successful tool result with
                    # attached files signals task completion). Loop continues
                    # only when no attachment was produced.
                    if produced_attachment:
                        # Post-drain retry — STRUCTURAL invariant:
                        #   fire iff a soft-inject drained THIS iteration
                        #   (after this iteration's tool dispatch) AND the
                        #   loop is about to break before the next chat()
                        #   can run.
                        #
                        # Without this, the [FRAMEWORK] marker just appended
                        # by the drain block above sits in history with the
                        # model never given a chat() iteration to respond.
                        # The post-loop synthetic follow-up also misses it
                        # (the safety drain sees _pending_inject empty by
                        # then). Result: silent drop of operator's mid-tool
                        # message.
                        #
                        # Kill switch: OPENFLIP_DISABLE_POST_DRAIN_RETRY=1
                        # disables the retry and falls through to break.
                        # The drained marker is still in history; the post-
                        # loop synthetic follow-up below will re-fire it
                        # as a fresh turn (the slower path, but safe).
                        _pdr_disabled = (
                            os.environ.get("OPENFLIP_DISABLE_POST_DRAIN_RETRY") == "1"
                        )
                        if (_drained_this_iter > 0
                                and not _pdr_disabled
                                and not locals().get("_post_drain_retry_used")
                                and not is_chain_terminator
                                and not str(log_tag or "").strip().startswith("[synthetic]")):
                            _post_drain_retry_used = True
                            print_ts(
                                f"{COLOR_YELLOW}{log_tag}post_drain_retry fired: "
                                f"drained {_drained_this_iter} message(s) this iteration — "
                                f"continuing so the model can respond (was about to "
                                f"break on attachment){COLOR_END}",
                                agent=agent.id,
                            )
                            continue
                        break
                    # Otherwise: continue loop (model gets tool results,
                    # can either emit more tool_use or stop).
        except asyncio.CancelledError:
            print_ts(
                f"{COLOR_YELLOW}{log_tag}turn interrupted by new inbound{COLOR_END}",
                agent=agent.id,
            )
            # Hard interrupt fired (operator typed `/stop`, fired the
            # /stop slash command, or some other path cancelled us).
            # Pending soft-injects for this channel are stale — the active
            # task is dying and the new `/stop` turn is about to take
            # over. Drop them so they don't bleed into the next turn's
            # history.
            try:
                _ch_id_cx = int(getattr(channel, "id", 0) or 0)
                if _ch_id_cx:
                    _dropped = len(self._pending_inject.pop(_ch_id_cx, []) or [])
                    if _dropped:
                        print_ts(
                            f"{COLOR_YELLOW}{log_tag}dropped {_dropped} pending soft-inject(s) "
                            f"on cancellation{COLOR_END}",
                            agent=agent.id,
                        )
            except Exception:
                pass
            # On user-driven cancellation (the common case — new inbound
            # arrived while we were mid-turn), we only need to synthesize
            # tool_result blocks for IN-FLIGHT tools so the API doesn't 400
            # on the next turn with orphaned tool_use. We do NOT inject a
            # separate "[Previous turn interrupted...]" marker for the
            # text-in-flight case — the operator's actual new user message
            # is about to land as the next message and IS the interruption
            # signal. Adding a redundant synthetic marker just gives the
            # model an excuse to treat the cut as a hard stop instead of
            # naturally continuing with partial work + new message visible.
            # This mirrors Claude Code's pattern at query.ts:1046-1048
            # (skip createUserInterruptionMessage when abort reason is
            # 'interrupt', i.e. user-driven).
            try:
                if last_ai_message is not None:
                    _tc = getattr(last_ai_message, "tool_calls", None) or []
                    if _tc:
                        # Tools were in flight. Synthesize a tool_result for each
                        # pending tool_use_id so the API doesn't 400 on next turn
                        # with orphaned tool_use blocks. The tool_result itself
                        # carries the interrupt context — no separate marker needed.
                        for tool_call in _tc:
                            _tu_id = getattr(tool_call, "tool_use_id", "") or ""
                            if _tu_id:
                                _interrupt_msg = ChatMessage(
                                    'tool',
                                    '[Tool interrupted by new user message before completion]',
                                )
                                _interrupt_msg["tool_use_id"] = _tu_id
                                conv.messages.append(_interrupt_msg)
                    # else: text-in-flight — no marker needed. The new user
                    # message arriving immediately after IS the marker. The
                    # model sees [my partial assistant text] + [new user msg]
                    # and continues naturally.
            except Exception as _marker_err:
                print_ts(
                    f"{COLOR_RED}{log_tag}failed to append interrupt marker: {_marker_err}{COLOR_END}",
                    error=True, agent=agent.id,
                )
            # Cleanup steps are each guarded so a nested cancellation
            # (or any error) can't tear conversation state. Best-effort.
            try:
                await _restore_system()
            except Exception as _err:
                print_ts(
                    f"{COLOR_RED}{log_tag}cleanup: _restore_system failed: {_err}{COLOR_END}",
                    error=True, agent=agent.id,
                )
            try:
                conv.save()
            except Exception as _err:
                print_ts(
                    f"{COLOR_RED}{log_tag}cleanup: conv.save failed: {_err}{COLOR_END}",
                    error=True, agent=agent.id,
                )
            try:
                _agent_state.on_turn_end(agent.id)
            except Exception:
                pass
            raise  # Let the worker see CancelledError so it knows to log + continue.
        except asyncio.TimeoutError:
            print_ts(f"{COLOR_RED}{log_tag}Chat timeout after {CHAT_TIMEOUT_S}s — model unresponsive{COLOR_END}", agent=agent.id, error=True)
            if auto_post_final_text:
                try:
                    await _safe_channel_send(channel, f"⚠️ The model didn't reply within {CHAT_TIMEOUT_S}s. Try again, or check Ollama.")
                except Exception:
                    pass
            if conv.messages and conv.messages[-1].role == 'user':
                conv.messages.pop()
            try:
                self._drain_pending_injects(int(getattr(channel, "id", 0) or 0), conv)
            except Exception:
                pass
            await _restore_system()
            conv.save()
            try:
                _agent_state.on_turn_end(agent.id)
            except Exception:
                pass
            return
        except MalformedRequestError as _mre:
            # Pre-flight validator refused to send a malformed body.
            # Surface a clear, specific message instead of the cryptic
            # Anthropic 400 we'd otherwise see. The trailing user message
            # is popped so the next turn doesn't try to re-send the same
            # bad shape.
            _problems_short = "; ".join(
                f"{p.rule} @ {p.location}" for p in _mre.problems
            ) or str(_mre)
            print_ts(
                f"{COLOR_RED}{log_tag}MalformedRequestError caught — refused to send: {_problems_short}{COLOR_END}",
                agent=agent.id, error=True,
            )
            if auto_post_final_text:
                try:
                    await _safe_channel_send(
                        channel,
                        f"⚠️ openflip caught a malformed request before sending — {_problems_short}",
                    )
                except Exception:
                    pass
            if conv.messages and conv.messages[-1].role == 'user':
                conv.messages.pop()
            try:
                self._drain_pending_injects(int(getattr(channel, "id", 0) or 0), conv)
            except Exception:
                pass
            await _restore_system()
            conv.save()
            try:
                _agent_state.on_turn_end(agent.id)
            except Exception:
                pass
            return
        except Exception as e:
            print_ts(f"{COLOR_RED}{log_tag}Chat error: {e}{COLOR_END}", agent=agent.id, error=True)
            if auto_post_final_text:
                # Suppress chat-side dump for known-noisy asyncio cleanup
                # errors — orphan-generator residue from Python 3.14's
                # stricter async cancellation. Don't paste into Discord —
                # user just wants "try again."
                emsg = str(e)
                _NOISY = (
                    "generator didn't stop after athrow",
                    "generator didn't stop after",
                )
                if any(n in emsg for n in _NOISY):
                    try:
                        await _safe_channel_send(channel, "⚠️ Hit a timeout. Try again or shorten the request.")
                    except Exception:
                        pass
                else:
                    try:
                        await _safe_channel_send(channel, f"⚠️ Internal error: {emsg[:300]}")
                    except Exception:
                        pass
            try:
                self._drain_pending_injects(int(getattr(channel, "id", 0) or 0), conv)
            except Exception:
                pass
            await _restore_system()
            conv.save()
            try:
                _agent_state.on_turn_end(agent.id)
            except Exception:
                pass
            return
        finally:
            # Safety-net soft-inject drain. The explicit drains at the
            # tool boundary (line ~1107), the post-loop exit (line ~1276),
            # and the TimeoutError/Exception branches above all handle the
            # happy paths. This finally exists to close the silent-loss
            # hole when _run_turn dies via an unexpected exception type
            # (or an exception escapes one of the except branches) before
            # any of those drains run. Without it, _pending_inject[ch_id]
            # orphans forever and the operator's mid-turn messages
            # vanish without a trace.
            #
            # Runs AFTER /stop's hard-interrupt buffer-wipe (which pops
            # the dict entry in both _hard_interrupt() and the
            # CancelledError handler above), so when an interrupt was
            # the cause, this is a no-op — the wipe is intentional and
            # this drain finds nothing.
            try:
                _ch_id_safety = int(getattr(channel, "id", 0) or 0)
                if _ch_id_safety and self._pending_inject.get(_ch_id_safety):
                    # Capture BEFORE drain pops the dict — needed by the
                    # post-loop follow-up-turn fire-check.
                    _drained_texts = list(self._pending_inject.get(_ch_id_safety, []))
                    _drained_count = self._drain_pending_injects(_ch_id_safety, conv)
                    if _drained_count > 0:
                        _human_softinject_drained_this_turn = True
                    # Human soft-injected during a synthetic turn → flip
                    # visibility so the post-block sends to the operator.
                    # See 2026-05-22 routing_bug notes.
                    if _drained_count > 0 and (not auto_post_final_text or silent):
                        auto_post_final_text = True
                        silent = False
                        _human_softinjected_this_turn = True
                        print_ts(
                            f"{COLOR_YELLOW}finally-drain: soft-inject from human during synthetic turn "
                            f"— flipping visibility on{COLOR_END}",
                            agent=agent.id,
                        )
            except Exception as _safety_err:
                print_ts(
                    f"{COLOR_YELLOW}{log_tag}finally soft-inject drain failed (continuing): "
                    f"{_safety_err}{COLOR_END}",
                    agent=agent.id,
                )
            # Safety-net save(). The except handlers above each call
            # conv.save() — but several have unguarded await points
            # (await _restore_system, await _safe_channel_send) BEFORE
            # their save() call. A CancelledError arriving at those
            # awaits escapes the handler (CancelledError is BaseException,
            # not caught by `except Exception`) and lands here with
            # messages in memory that were never flushed to disk. This
            # finally-save closes that gap. Idempotent: if an except
            # handler already called save(), _persisted_count ==
            # len(non_system) so new_count <= 0 → immediate no-op.
            try:
                conv.save()
            except Exception as _save_err:
                print_ts(
                    f"{COLOR_RED}{log_tag}finally: conv.save failed: {_save_err}{COLOR_END}",
                    error=True, agent=agent.id,
                )

        # Post-loop cleanup. Each step is independently guarded against
        # a secondary CancelledError so a cancel landing mid-cleanup can't
        # leave history in a torn state. Mirrors the in-try cleanup guards
        # in the CancelledError handler above.
        # End-of-turn soft-inject drain. Covers the no-tool-fired exit
        # (model emitted text only, loop broke at the needs_follow_up
        # gate) and any-other-break exit. The drain lands the pending
        # message(s) in conv history before conv.save() so they persist
        # and the model sees them on its next turn (or its next chat()
        # call within a chain-terminator handoff). Documented behavior:
        # if a turn produces no tool_use, soft-injects don't surface
        # until the NEXT turn — that's the price of keeping the boundary
        # natural rather than yanking the model mid-text.
        # Post-loop drain is now a no-op safety check — the `finally:`
        # safety-net above already drained whatever was in
        # _pending_inject. Kept as a belt-and-suspenders no-op in case
        # some future control path skips the finally; emits nothing if
        # there's nothing to drain.
        try:
            _ch_id_drain = int(getattr(channel, "id", 0) or 0)
            if _ch_id_drain and self._pending_inject.get(_ch_id_drain):
                _drained_texts = _drained_texts or list(self._pending_inject.get(_ch_id_drain, []))
                _post_drained = self._drain_pending_injects(_ch_id_drain, conv)
                _drained_count += _post_drained
                if _post_drained > 0:
                    _human_softinject_drained_this_turn = True
                # Human soft-injected during a synthetic turn → flip
                # visibility (see 2026-05-22 routing_bug notes).
                if _post_drained > 0 and (not auto_post_final_text or silent):
                    auto_post_final_text = True
                    silent = False
                    _human_softinjected_this_turn = True
                    print_ts(
                        f"{COLOR_YELLOW}post-loop drain: soft-inject from human during synthetic turn "
                        f"— flipping visibility on{COLOR_END}",
                        agent=agent.id,
                    )
        except Exception as _drain_err:
            print_ts(
                f"{COLOR_YELLOW}{log_tag}post-loop soft-inject drain failed (continuing): "
                f"{_drain_err}{COLOR_END}",
                agent=agent.id,
            )
        try:
            await _restore_system()
        except Exception as _err:
            print_ts(f"{COLOR_RED}{log_tag}cleanup: _restore_system failed: {_err}{COLOR_END}", error=True, agent=agent.id)
        try:
            conv.save()
        except Exception as _err:
            print_ts(f"{COLOR_RED}{log_tag}cleanup: conv.save failed: {_err}{COLOR_END}", error=True, agent=agent.id)
        # Kairos idle-tick pruning: if this was a kairos proactive turn
        # and the agent did nothing (no tool calls at all), prune the
        # tick user message + empty assistant response from the in-memory
        # conversation so idle ticks don't inflate context/cost on every
        # subsequent tick. Messages are already saved to disk (above) for
        # audit. On restart, JSONL reload brings them back but that's rare
        # and the context overhead is negligible.
        if originator_visibility == "kairos" and not any_tool_called:
            try:
                n = len(conv.messages)
                if n >= 2 and conv.messages[-1].role == "assistant" and conv.messages[-2].role == "user":
                    conv.messages.pop()  # assistant (empty/idle response)
                    conv.messages.pop()  # user (tick prompt)
                    print_ts(f"{log_tag}kairos idle tick pruned from memory", agent=agent.id)
            except Exception:
                pass
        # Soft-inject follow-up turn — RESTORED 2026-05-23.
        # 
        # Ports Claude Code's queue-drain pattern (cli.js v2.1.149 Hvq hook):
        # the moment the current turn ends, if there's anything queued, fire
        # a synthetic turn immediately so the model responds to the queued
        # message NOW instead of waiting for the next natural inbound. See
        # agents/<agent>/notes/queue_design.md
        # for the full design and source-of-truth analysis.
        #
        # Previously removed 2026-05-22 citing "/stop wrong behavior", but
        # that concern doesn't actually apply: the CancelledError branch
        # (line ~1503) pops _pending_inject before this code runs, so on
        # /stop _drained_count is 0 here and this block is a no-op. The
        # previous removal was overcautious.
        #
        # Oldest-first: when multiple messages were queued, we fire the
        # OLDEST as the new turn's prompt. The drain code already appended
        # all of them as [FRAMEWORK] markers to history, so the model sees
        # both the older ones (as markers) and the latest (as the new user
        # message) and can address them in order. If new messages arrive
        # during the follow-up turn, the follow-up's own end-of-turn
        # drain catches them and fires another follow-up — self-perpetuating
        # until the queue is empty.
        if _drained_count > 0 and _drained_texts:
            try:
                follow_up_prompt = _drained_texts[0]
                follow_up_ch_id = int(getattr(channel, "id", 0) or 0)
                follow_up_speaker = int(speaker_id or 0)
                print_ts(
                    f"{COLOR_YELLOW}{log_tag}drained {_drained_count} soft-inject(s) "
                    f"— firing follow-up turn for oldest queued msg{COLOR_END}",
                    agent=agent.id,
                )
                _fu_task = asyncio.create_task(
                    self.run_synthetic_turn(
                        follow_up_ch_id,
                        follow_up_prompt,
                        auto_post_final_text=True,
                        speaker_id=follow_up_speaker,
                        silent=False,
                        depth=0,
                    ),
                    name="softinject_followup",
                )
                _fu_task.add_done_callback(log_task_exception)
            except Exception as _fu_err:
                print_ts(
                    f"{COLOR_YELLOW}{log_tag}soft-inject follow-up fire failed (continuing): "
                    f"{_fu_err}{COLOR_END}",
                    agent=agent.id,
                )
        try:
            _agent_state.on_turn_end(agent.id)
        except Exception:
            pass
        try:
            from . import events_log as _events_log
            _events_log.log_event(agent.id, "turn_end")
        except Exception:
            pass

        # Silence sentinel state — set True below if the agent chose silence.
        # Initialized out here (not inside the `last_ai_message is not None`
        # block) so the terminal-result contract at function exit can read it
        # unconditionally without risking a NameError.
        _chose_silence = False

        # Final chat text — the model's actual answer after any tools.
        if last_ai_message is not None:
            final_calls = getattr(last_ai_message, "tool_calls", None) or []
            final_text = (getattr(last_ai_message, "content_text", None) or
                          getattr(last_ai_message, "content", "") or "").strip()
            # Framework errors (5xx, timeout, auth) are already posted to
            # the channel by the in-loop framework-error branch above
            # (line ~493). Re-posting them here doubles the message in
            # the user's DM. Skip the final-text post path entirely for those.
            if getattr(last_ai_message, "is_framework_error", False):
                final_text = ""
            # Silence sentinel — blank final_text BEFORE any post-site sees it.
            # final_text is already .strip()'d above, so this is an exact
            # whole-message, case-sensitive match: only a reply that is ONLY
            # the token suppresses. A sentence merely containing the word
            # (e.g. "I'll STAY_SILENT this round") does not match and posts
            # normally. Blanking here makes every downstream post-site (the
            # inter-agent route block, the silent-drop branches, the main
            # auto-post elif) see the already-empty value and run their
            # "nothing to post" path. The assistant turn is still saved to
            # history upstream so the agent remembers it chose silence.
            if final_text == STAY_SILENT_SENTINEL:
                _chose_silence = True
                final_text = ""
                print_ts(
                    f"{COLOR_YELLOW}{log_tag}STAY_SILENT: suppressing channel post "
                    f"for agent={agent.id} (agent chose silence){COLOR_END}",
                    agent=agent.id,
                )
            if final_text and not final_calls:
                # Inter-agent auto-route: if this turn originated from
                # another agent's talk_to_agent dispatch, the reply text
                # routes BACK to that agent as a new synthetic turn
                # instead of posting to Discord here. The originator's
                # turn handler decides whether to continue the chain (by
                # calling talk_to_agent again) or wrap up (their reply
                # posts to their own channel naturally). Depth+1 is
                # threaded through; the talk_to_agent cap on the
                # originator's side stops runaway loops.
                if originator_agent_id:
                    originator = RUNNERS.get(originator_agent_id)
                    if originator is not None:
                        framed_reply = f"{agent.id}: {final_text}"
                        # ALSO post the reply text to THIS recipient's
                        # channel so the human has visibility into agent-to-
                        # agent conversations. Without this, inter-agent
                        # exchanges are invisible to him — that was the
                        # complaint that triggered the 2026-05-10 revert
                        # of per-peer file isolation. The reply text both
                        # posts here AND auto-routes (below) — auto-route
                        # is the silent chain-continuation path, the post
                        # is the human-visibility path.
                        # Gate: only post the inter-agent reply text to the
                        # operator's channel if (a) we'd normally auto-post,
                        # AND (b) a human soft-injected this turn — i.e.
                        # the operator is actually waiting on output here.
                        # Without (b), peer-triggered maintenance work
                        # leaks into the operator's DM as unsolicited noise
                        # (the 2026-05-23 complaint). The agent can still
                        # reach the operator explicitly via send_message.
                        if (auto_post_final_text
                                and not silent
                                and not (media_only and any_attachments_this_turn
                                         and not _human_softinject_drained_this_turn)
                                and _human_softinjected_this_turn):
                            try:
                                for chunk in _split_for_discord(final_text):
                                    await _safe_channel_send(channel, chunk)
                                _posted_assistant_text = True
                            except Exception as _post_err:
                                print_ts(
                                    f"{COLOR_RED}{log_tag}Failed to post inter-agent reply: {_post_err}{COLOR_END}",
                                    error=True, agent=agent.id,
                                )
                        # Resolve a channel the ORIGINATOR can actually
                        # reach. The recipient's channel (channel.id) is
                        # often not accessible to the originator's bot
                        # (each bot has its own DMs). Without resolving,
                        # the originator's run_synthetic_turn 403s and
                        # the auto-route reply is lost. Priority: try
                        # originator's DM with the human speaker (the
                        # one whose user_id is in CURRENT_SPEAKER_ID).
                        return_channel_id = 0
                        # Transport-aware return-channel resolution:
                        # - Discord originator: fetch_user → DM channel id
                        #   (each bot has its own DM ids; recipient's channel
                        #   often isn't accessible to originator).
                        # - Non-Discord originator (iMessage, internal): the
                        #   threaded originator_channel_id IS the caller's
                        #   own session id (e.g. iMessage chat_id). Use it
                        #   directly — originator.transport.send routes by
                        #   that id natively, no DM resolution needed.
                        _originator_channel_id = int(originator_channel_id or 0)
                        # PRIORITY 1 (transport-native, every transport): the
                        # threaded originator_channel_id is the originator's OWN
                        # session id (iMessage chat_id / Discord channel /
                        # internal). run_synthetic_turn routes by it natively, no
                        # DM resolution. MUST come first — gating on
                        # hasattr(originator,"bot") wrongly sent multi-transport
                        # (Discord+iMessage) originators down the Discord path,
                        # where fetch_user(imessage_hashed_speaker_id) 404s
                        # (Unknown User 10013) and the reply was lost on iMessage.
                        if _originator_channel_id:
                            return_channel_id = _originator_channel_id
                        # PRIORITY 2 (legacy fallback): no threaded channel id —
                        # resolve the originator's Discord DM via the human
                        # speaker. Only reached for Discord originators that
                        # didn't thread a channel id.
                        elif hasattr(originator, "bot"):
                            try:
                                sp_id = int(speaker_id or 0)
                                if sp_id:
                                    user = await asyncio.wait_for(
                                        originator.bot.fetch_user(sp_id), timeout=10.0,
                                    )
                                    if user is not None:
                                        dm = user.dm_channel or await asyncio.wait_for(
                                            user.create_dm(), timeout=10.0,
                                        )
                                        if dm is not None:
                                            return_channel_id = int(dm.id)
                            except asyncio.TimeoutError:
                                print_ts(
                                    f"{COLOR_RED}inter-agent auto-route: "
                                    f"timed out resolving {originator_agent_id}'s DM "
                                    f"with speaker {speaker_id}; skipping post{COLOR_END}",
                                    error=True, agent=agent.id,
                                )
                            except Exception as _resolve_err:
                                print_ts(
                                    f"{COLOR_YELLOW}inter-agent auto-route: "
                                    f"couldn't resolve {originator_agent_id}'s DM "
                                    f"with speaker {speaker_id}: {_resolve_err}{COLOR_END}",
                                    agent=agent.id,
                                )
                        if not return_channel_id:
                            # Last-resort: try the channel where this turn
                            # was processing. Likely unreachable, but at
                            # least we'll surface the 403 in logs instead
                            # of silently dropping the reply.
                            return_channel_id = int(channel.id)
                        print_ts(
                            f"inter-agent reply routed: {agent.id} -> "
                            f"{originator_agent_id} (depth {depth} -> {depth + 1}) "
                            f"via channel {return_channel_id}",
                            agent=agent.id,
                        )
                        try:
                            _ar_task = asyncio.create_task(
                                originator.run_synthetic_turn(
                                    return_channel_id,
                                    framed_reply,
                                    # ARCHITECTURE FIX 2026-05-19:
                                    # Chain-terminator turns now behave
                                    # like normal turns — plain text posts
                                    # to the operator's DM, full toolset
                                    # available. This eliminates the
                                    # silent-failure bug class that came
                                    # from asking the model to "produce
                                    # output that gets discarded." See
                                    # chain_terminator_architecture design
                                    # doc (Option 4). Visibility tradeoff:
                                    # operator now sees the originator's
                                    # summary of inter-agent exchanges.
                                    auto_post_final_text=True,
                                    silent=False,
                                    depth=depth + 1,
                                    # Don't propagate originator_agent_id —
                                    # the originator's turn ends the auto-
                                    # route. They have to explicitly call
                                    # talk_to_agent again to extend the
                                    # chain.
                                    originator_agent_id="",
                                    # Chain-ID propagation for stale-drop
                                    # check at top of _run_turn — keeps
                                    # parallel branches under control.
                                    chain_id=chain_id,
                                    auto_route_from_peer=agent.id,
                                    # Thread the originating human speaker
                                    # back to the chain-terminator turn so
                                    # the originator's auto-route reply is
                                    # attributed to the right human, not
                                    # always to owner_id. Without this, a
                                    # non-owner user's inter-agent chain
                                    # would resolve to the owner on the way
                                    # back too.
                                    speaker_id=speaker_id,
                                    # Propagate chain-root visibility to
                                    # the originator's chain-terminator
                                    # turn. The originator was THIS turn's
                                    # caller — its visibility is what we
                                    # received via `originator_visibility`
                                    # at the top of _run_turn. If the chain
                                    # root was the operator, the originator's
                                    # chain-terminator empty path will
                                    # surface failure to them.
                                    originator_visibility=originator_visibility,
                                ),
                                name="autoroute_dispatch",
                            )
                            _ar_task.add_done_callback(log_task_exception)
                        except Exception as e:
                            print_ts(
                                f"{COLOR_RED}inter-agent auto-route failed: {e}{COLOR_END}",
                                error=True, agent=agent.id,
                            )
                    else:
                        if silent:
                            # Originator agent isn't running anymore AND
                            # the turn is silent — don't leak the reply
                            # into the channel that the framework happened
                            # to resolve to. Just drop the reply with a
                            # warning so we have a trace in logs.
                            print_ts(
                                f"{COLOR_YELLOW}inter-agent auto-route: originator "
                                f"'{originator_agent_id}' not running and turn is silent; "
                                f"dropping reply rather than leaking to channel{COLOR_END}",
                                agent=agent.id,
                            )
                        else:
                            print_ts(
                                f"{COLOR_YELLOW}inter-agent auto-route: originator "
                                f"'{originator_agent_id}' not running; posting reply normally{COLOR_END}",
                                agent=agent.id,
                            )
                            if auto_post_final_text and not (media_only and any_attachments_this_turn
                                                             and not _human_softinject_drained_this_turn):
                                try:
                                    for chunk in _split_for_discord(final_text):
                                        await _safe_channel_send(channel, chunk)
                                    _posted_assistant_text = True
                                except Exception as e:
                                    print_ts(f"{COLOR_RED}{log_tag}Failed to post message: {e}{COLOR_END}", error=True, agent=agent.id)
                elif silent:
                    # Defense-in-depth assertion: this branch should be
                    # unreachable now that chain-terminator turns force
                    # tool_choice=any with a narrowed surface (talk_to_agent /
                    # send_message / end_chain). If it fires, something slipped
                    # past the forcing — log loudly so we can diagnose.
                    # Ollama provider falls through here legitimately (no
                    # tool_choice support) — that's expected and documented.
                    _prov_tag = agent.provider or "ollama"
                    print_ts(
                        f"{COLOR_RED}ASSERTION: silent synthetic turn with no originator "
                        f"reached plain-text drop branch (provider={_prov_tag}). "
                        f"Chain-terminator forcing did not prevent plain-text reply. "
                        f"Lost text preview: {(final_text or '')[:120].replace(chr(10),' ')!r}{COLOR_END}",
                        agent=agent.id, error=True,
                    )
                elif auto_post_final_text and not (media_only and any_attachments_this_turn
                                                   and not _human_softinject_drained_this_turn):
                    try:
                        for chunk in _split_for_discord(final_text):
                            await _safe_channel_send(channel, chunk)
                        _posted_assistant_text = True
                    except Exception as e:
                        print_ts(f"{COLOR_RED}{log_tag}Failed to post message: {e}{COLOR_END}", error=True, agent=agent.id)

        # Chain consumed: clear tracker for this peer if it still holds the
        # chain_id we came in with. If the agent dispatched a NEW chain to
        # the same peer during this turn (via talk_to_agent), the tracker
        # was overwritten with the new chain_id and we leave it alone so
        # the next auto-route from that new chain gets through.
        # Without this cleanup, the tracker retains the consumed chain_id
        # forever, which means any late/replayed auto-route from the SAME
        # consumed chain would mis-match-check ok and get processed twice.
        if auto_route_from_peer and \
                self._current_chain_to.get(auto_route_from_peer) == chain_id:
            self._current_chain_to.pop(auto_route_from_peer, None)
            self._save_chain_state()
            print_ts(
                f"chain consumed: cleared tracker for peer "
                f"'{auto_route_from_peer}' chain={chain_id[:8] if chain_id else '<empty>'}",
                agent=agent.id,
            )

        # ---- Terminal-result contract (audit P0 #1 — silent-drop catch) ----
        # Sanity check at function exit: did this turn produce ANYTHING
        # visible to the operator? If silent=True or auto_post_final_text=False,
        # the turn intentionally produces no operator-visible output — skip.
        # If a tool produced an attachment and the agent is in media-only
        # mode, skip — that's a successful media turn. Otherwise: if neither
        # text nor attachment landed, AND no reply-equivalent tool fired
        # (send_message routes to the operator's channel; end_chain is
        # silent-by-design), emit a diagnostic so the operator sees SOMETHING
        # instead of total silence.
        #
        # Models Claude Code's `isResultSuccessful` check from QueryEngine:
        # every turn must produce a valid terminal artifact or surface an
        # error_during_execution. We can't fully port their structured
        # result type without refactoring the turn engine, but we can catch
        # the worst pathology (model emits text like "going" then loop falls
        # through with no Discord post).

        try:
            # _chose_silence: the agent emitted exactly STAY_SILENT, so the
            # empty output is intentional — producing no operator-visible
            # artifact is the desired outcome, not a silent-drop pathology.
            # Without this guard the contract recomputes _final_text from
            # last_ai_message (which still holds the literal token) and posts
            # a bogus "⚠️ Turn ended without a visible reply" warning, which
            # is exactly the channel noise the sentinel exists to avoid.
            _intentionally_silent = silent or not auto_post_final_text or _chose_silence
            _media_satisfied = (media_only and any_attachments_this_turn)
            _reply_via_tool = reply_equivalent_tool_fired
            if (not _intentionally_silent
                    and not _media_satisfied
                    and not _reply_via_tool
                    and not _posted_assistant_text):
                # Diagnose WHY the turn produced nothing visible.
                _diag_bits = []
                if last_ai_message is None:
                    _diag_bits.append("no_assistant_message")
                else:
                    _final_calls = getattr(last_ai_message, "tool_calls", None) or []
                    _final_text = (getattr(last_ai_message, "content_text", None) or
                                   getattr(last_ai_message, "content", "") or "").strip()
                    if not _final_text and not _final_calls:
                        _diag_bits.append("empty_assistant_message")
                    elif _final_text and not _final_calls:
                        _diag_bits.append("text_present_but_not_posted")
                    elif _final_calls:
                        _diag_bits.append("tools_called_but_no_reply_emitted")
                if not _diag_bits:
                    _diag_bits.append("unknown")
                _diag = ",".join(_diag_bits)
                # If the turn ended empty because the provider returned an
                # error (rate limit / overload / auth / 400), surface the
                # ACTUAL error — never the generic catch-all. Otherwise the
                # operator sees "known bug" when the real cause is a temporary
                # 429/529 from Anthropic.
                if _captured_framework_error:
                    _api_err = str(_captured_framework_error).strip()[:1500]
                    _user_msg = (
                        f"⚠️ The model provider returned an error: {_api_err} "
                        f"This is usually a temporary rate limit/overload — try again in a moment."
                    )
                    _log_reason = f"provider_error: {str(_captured_framework_error)[:200]!r}"
                else:
                    _user_msg = (
                        f"⚠️ The model returned an empty response (reason: {_diag}). "
                        f"Try `/reset` if it persists."
                    )
                    _log_reason = f"reason={_diag}"
                print_ts(
                    f"{COLOR_RED}{log_tag}TERMINAL CONTRACT FAILED: turn ended without "
                    f"operator-visible output ({_log_reason}){COLOR_END}",
                    agent=agent.id, error=True,
                )
                try:
                    await _safe_channel_send(channel, _user_msg)
                except Exception:
                    pass
        except Exception as _term_err:
            # The terminal check itself must never raise — that would defeat
            # its purpose. Log and move on.
            print_ts(
                f"{COLOR_YELLOW}{log_tag}terminal-contract check failed (continuing): "
                f"{_term_err}{COLOR_END}",
                agent=agent.id,
            )

        # ---- Dream auto-fire (silent background memory consolidation) ----
        # Event-driven end-of-turn check (mirrors Claude Code's autoDream). Only
        # eligible on a cleanly-completed, top-level, operator-driven turn:
        # SYNTHETIC turns (log_tag "[synthetic] " — cron/kairos/restart AND the
        # dream's OWN turn), SUBAGENT/inter-agent turns (originator_agent_id),
        # chain-terminators (auto_route_from_peer), and any depth>0 turn are all
        # excluded so a dream can never recursively trigger another dream. Only
        # the clean-completion path reaches here — CancelledError re-raises and
        # the timeout/error branches return before this point. The gate-check +
        # lock + firing all live in dream_autofire; it never posts to Discord
        # and is fully self-guarded, but we belt-and-suspenders wrap it so a
        # dream-layer bug can never tear a completed turn.
        if (not str(log_tag or "").strip().startswith("[synthetic]")
                and not originator_agent_id
                and not auto_route_from_peer
                and int(depth) == 0):
            try:
                from . import dream_autofire
                await dream_autofire.maybe_fire_dream(self, channel)
            except Exception as _dream_err:
                print_ts(
                    f"{COLOR_YELLOW}{log_tag}dream auto-fire check failed (continuing): "
                    f"{_dream_err}{COLOR_END}",
                    agent=agent.id,
                )


    async def start(self):
        """Start the agent — delegates to each transport's start lifecycle.

        Multi-transport: creates a task per transport and gathers them.
        All transports run concurrently; if one crashes the others continue.
        Single-transport agents behave identically to before.
        """
        try:
            self._transport_tasks = [
                asyncio.create_task(t.start()) for t in self._transports
            ]
            # gather() returns a cancellable _GatheringFuture (NOT a coroutine,
            # so it must not be wrapped in create_task). stop() can cancel it.
            self._task = asyncio.gather(*self._transport_tasks, return_exceptions=True)
            await self._task
        except asyncio.CancelledError:
            for t in self._transport_tasks:
                if not t.done():
                    t.cancel()
            return
        except Exception as e:
            for t in self._transport_tasks:
                if not t.done():
                    t.cancel()
            print_ts(
                f"{COLOR_RED}Agent {self.agent.id} transport failed: {e}{COLOR_END}",
                error=True, agent=self.agent.id,
            )

    async def stop(self):
        for t in self._transports:
            try:
                await t.stop()
            except Exception:
                pass
        for t in getattr(self, "_transport_tasks", []):
            if t and not t.done():
                t.cancel()
        if self._task:
            self._task.cancel()


# Discord I/O helpers (safe_channel_send, split_for_discord, safe_typing,
# silent_typing) extracted to openflip/discord_io.py and imported at the top
# of this file as _safe_channel_send / _split_for_discord / _discord_safe_typing
# / _discord_silent_typing.
