"""Runs the tool calls the model emitted, posts results back to Discord, enforces concurrency."""
from __future__ import annotations
import asyncio
import contextvars
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from .agent import Agent
from .config_global import get_config
from .conversation import DiscordConversation
from .tools import TOOL_REGISTRY, ToolResult
from .utils import print_ts, redact_paths, COLOR_YELLOW, COLOR_RED, COLOR_GREEN, COLOR_END
from . import agent_state as _agent_state

# Context variable exposing the current agent to tools (e.g. memory tools).
CURRENT_AGENT: contextvars.ContextVar[Agent] = contextvars.ContextVar("current_agent")
CURRENT_CHANNEL_ID: contextvars.ContextVar[int] = contextvars.ContextVar("current_channel_id")
CURRENT_SPEAKER_ID: contextvars.ContextVar[int] = contextvars.ContextVar("current_speaker_id")
# Transport-agnostic session for the current turn. Set by runtime._run_turn
# once Session objects exist (Phase 1 Discord-decouple). Tools that need
# to know the session (transport, transport_id, conversation_id, etc.) read
# this instead of CURRENT_CHANNEL_ID so they don't depend on Discord internals.
# None until runtime starts setting it — tools fall back to CURRENT_CHANNEL_ID
# for compat during the transition.
CURRENT_SESSION: contextvars.ContextVar = contextvars.ContextVar("current_session", default=None)
# Loop-prevention for inter-agent comms via talk_to_agent. Set in runtime
# _run_turn at the start of every turn. Human-originated turns enter at
# depth 0; A->B (talk_to_agent) hands B depth 1; B->A hands A depth 2;
# at depth 2 talk_to_agent refuses, capping the chain at one round-trip.
CURRENT_TURN_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("current_turn_depth", default=0)
# Chain-root visibility classification for the CURRENT turn. Set in runtime
# _run_turn from the queued item's originator_visibility. Values:
# "operator_channel" / "silent_agent_chain" / "cron" / "heartbeat" / "".
# talk_to_agent reads this to propagate the same visibility into the
# recipient's chain-terminator turn, so when a peer goes silent we know
# whether the original requester is a human awaiting output or an
# agent/scheduler that just needs the failure logged.
CURRENT_TURN_VISIBILITY: contextvars.ContextVar[str] = contextvars.ContextVar("current_turn_visibility", default="")
# Chain-ROOT-AGENT identity for the CURRENT turn: the agent the human DIRECTLY
# addressed at the head of this inter-agent chain. Set in runtime _run_turn from
# the queued item's chain_root_agent_id. talk_to_agent stamps it with the
# sender's own id on the FIRST hop of a chain (contextvar empty) and propagates
# it verbatim on every deeper hop, so it survives the full round-trip and lands
# back on the originating agent's chain-terminator turn. This is the load-bearing
# discriminator for whether a SILENT chain-terminator's plain text may surface:
# ONLY the root agent (chain_root_agent_id == self.agent.id) — i.e. the outermost
# hop, the one the human actually messaged — is allowed to post its return-turn
# answer. A nested middle agent (operator -> A -> B -> ...) carries root="A" while
# running as "B", so its terminator stays silent even when it was explicitly
# dispatched into a real human channel via talk_to_agent(channel_id=...). Empty
# means "not part of an inter-agent chain" (a plain human turn) — fail-closed: a
# terminator with no root identity does NOT surface.
CURRENT_CHAIN_ROOT_AGENT: contextvars.ContextVar[str] = contextvars.ContextVar("current_chain_root_agent", default="")
# Read-before-edit enforcement: the documented per-turn tracking set was
# never implemented. Instead, `edit_file` enforces correctness via the
# `old_string must appear EXACTLY ONCE` constraint — agents that didn't
# read the file don't know its exact whitespace/EOLs and the edit fails.
# This is slightly weaker than tracking (agents could guess at boilerplate
# patterns), but defensible for aligned agents. See tools/files.py for the
# constraint logic.

def current_caller() -> tuple[int, str, str]:
    """Derive the current tool caller's identity from the executor contextvars.

    Returns `(speaker_id, transport, handle)`:
      - `speaker_id` — int from `CURRENT_SPEAKER_ID`; 0 when unset/unreadable.
      - `transport`  — `CURRENT_SESSION.transport`, or "discord" when no session.
      - `handle`     — `CURRENT_SESSION.handle`, or "" when no session.

    Single source of truth for the (speaker, transport, handle) tuple that the
    cross-channel owner guards (send_file / send_message / delete_message) and
    the per-user path ACL (`_effective_allowed`) all key off — previously this
    same block was copy-pasted at each of those sites. Session is the source of
    truth for transport + handle on handle-based transports (iMessage); Discord
    uses the numeric speaker_id. Fail soft: any error → empty identity
    (0 / "discord" / ""), so callers fall through to their default-deny /
    non-owner branch rather than crashing.

    NOTE: `handle` is returned UNCONDITIONALLY (even on Discord, where it goes
    unused because the ACL key is the numeric id). The owner/admin entry points
    in acl.py (`current_caller_is_owner`) and fetch_url.py
    (`_caller_is_owner_or_admin`) derive handle ONLY for non-discord transports;
    that difference is immaterial (is_owner/is_admin ignore handle on Discord)
    but is why those entry points are intentionally NOT routed through here.
    """
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
        handle = getattr(sess, "handle", "") or ""
    return speaker_id, transport, handle


# Per-(agent_id, user_id, tool_name) lock — one in-flight job per user per tool.
_inflight: dict[tuple[str, int, str], asyncio.Lock] = defaultdict(asyncio.Lock)


async def _send_text(transport, session_id: str, channel, text: str) -> None:
    """Send a short status/notice text to the user. Phase 2: routes through
    transport.send when available; falls back to channel.send for compat.

    Every external send is wrapped in `asyncio.wait_for(..., timeout=30)`
    so a stuck Discord gateway can't block the turn indefinitely. 30s
    accommodates Discord's built-in 429 retries without false positives.
    """
    if transport and session_id:
        try:
            await asyncio.wait_for(transport.send(session_id, text), timeout=30.0)
        except asyncio.TimeoutError:
            print_ts(f"{COLOR_YELLOW}_send_text via transport timed out after 30s{COLOR_END}")
        except Exception as e:
            print_ts(f"{COLOR_YELLOW}_send_text via transport failed: {e}{COLOR_END}")
    elif channel:
        try:
            await asyncio.wait_for(channel.send(text), timeout=30.0)
        except asyncio.TimeoutError:
            print_ts(f"{COLOR_YELLOW}_send_text via channel timed out after 30s{COLOR_END}")
        except Exception:
            pass


def _lock_key(agent: Agent, user_id: int, tool_name: str) -> tuple[str, int, str]:
    return (agent.id, user_id, tool_name)


async def _invoke_tool(name: str, args: dict) -> ToolResult:
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return ToolResult.fail(f"Tool not registered: {name}")
    try:
        result = tool.func(**args)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolResult):
            return result
        return ToolResult(text=str(result) if result is not None else None)
    except Exception as e:
        return ToolResult.fail(f"Tool '{name}' raised: {e}")


async def execute_tool_calls(
    *,
    agent: Agent,
    conversation: DiscordConversation,
    ai_message,
    callable_tool_names: set[str],
    channel,
    speaker_id: int,
    session_id: str = "",
    discord_message=None,
    silent: bool = False,
    interrupt_check: Callable[[], bool] | None = None,
) -> list[tuple[str, ToolResult]]:
    """
    Run each tool call from the AI response. Posts files / text to Discord
    (subject to silent_to_discord + media_only). Returns a list of
    (tool_name, ToolResult) for the agent loop to feed back to the model
    as `role=tool` messages on the next turn.

    `channel` and `speaker_id` are required for legacy compat — they identify
    where to post output and who's invoking. Phase 2: callers should also pass
    `session_id` so outbound goes through transport.send/send_file instead of
    bare channel.send. If session_id is empty, we derive it from channel.id.

    `discord_message` is kept as an optional reference for tools that
    need raw access to attachments etc., but the executor itself no
    longer reads from it.

    Empty list = no tool calls fired (or all were blocked / dry-run / locked).
    The runtime uses that to decide "is the conversation done this turn?"
    """
    out: list[tuple[str, ToolResult]] = []
    tool_calls = getattr(ai_message, "tool_calls", None) or []
    if not tool_calls:
        return out

    CURRENT_AGENT.set(agent)
    channel_id_int = int(getattr(channel, "id", 0) or 0)
    try:
        CURRENT_CHANNEL_ID.set(channel_id_int)
    except Exception:
        pass
    try:
        CURRENT_SPEAKER_ID.set(int(speaker_id))
    except Exception:
        pass

    # Resolve session_id for transport routing. Prefer explicit arg, then
    # CURRENT_SESSION from contextvar, then fall back to channel.id string.
    _effective_session_id = session_id
    if not _effective_session_id:
        _sess = CURRENT_SESSION.get(None)
        if _sess is not None:
            _effective_session_id = getattr(_sess, "session_id", "") or getattr(_sess, "channel_id", "")
    if not _effective_session_id and channel_id_int:
        _effective_session_id = str(channel_id_int)

    # Resolve transport from registry for outbound sends.
    _transport = None
    try:
        from .registry import RUNNERS
        _runner = RUNNERS.get(agent.id)
        if _runner:
            _transport = getattr(_runner, "transport", None)
    except Exception:
        pass

    media_only = (agent.tool_response_mode == "media_only")

    for _tc_idx, tc in enumerate(tool_calls):
        name = tc.function_name
        args = tc.args or {}

        # Memory tools bypass ACL when memory_enabled is on.
        from .pipeline import MEMORY_TOOL_NAMES
        _memory_bypass = agent.memory_enabled and name in MEMORY_TOOL_NAMES
        if name not in callable_tool_names and not _memory_bypass:
            print_ts(f"{COLOR_YELLOW}Blocked tool call '{name}' (not in current ACL).{COLOR_END}", agent=agent.id)
            if not media_only and not silent:
                await _send_text(_transport, _effective_session_id, channel,
                                 f"_(I can't use `{name}` for you here.)_")
            # Tell the model so it can adjust on the next turn.
            out.append((name, ToolResult.fail(f"You don't have permission to call '{name}' for this user.")))
            continue

        key = _lock_key(agent, speaker_id, name)
        lock = _inflight[key]
        if lock.locked():
            if not silent:
                await _send_text(_transport, _effective_session_id, channel,
                                 f"_(Already working on `{name}` for you. Wait.)_")
            out.append((name, ToolResult.fail(f"Already running '{name}' for this user — try again after it finishes.")))
            continue

        async with lock:
            print_ts(f"Tool call: {name}({args})", agent=agent.id)
            # Dry-run: log what WOULD have been called and skip the actual tool.
            # Bypasses media_only suppression so the notice always posts.
            if get_config().get("dry_run_tools", False):
                print_ts(f"{COLOR_GREEN}[dry-run] suppressed call to {name}{COLOR_END}", agent=agent.id)
                try:
                    args_pretty = json.dumps(args, indent=2, ensure_ascii=False, default=str)
                except Exception:
                    args_pretty = str(args)
                msg = (
                    f"🧪 **dry-run** — would have called `{name}`\n"
                    f"```json\n{args_pretty[:1700]}\n```"
                )
                if not silent:
                    await _send_text(_transport, _effective_session_id, channel, msg)
                out.append((name, ToolResult(text=f"[dry-run] would have called {name}", model_feedback="(dry-run mode is enabled; tool was not executed)")))
                continue

            # Pre-tool notice removed — was noisy in chat and the operator
            # doesn't want it. Tool result posting still handles user-facing
            # output below.

            try:
                _agent_state.on_tool_start(agent.id, name)
            except Exception:
                pass
            try:
                from . import events_log as _events_log
                _events_log.log_event(agent.id, "tool_call", tool=name)
            except Exception:
                pass
            result = await _invoke_tool(name, args)
            try:
                _agent_state.on_tool_end(agent.id)
            except Exception:
                pass
            try:
                from . import events_log as _events_log
                _events_log.log_event(
                    agent.id, "tool_end", tool=name,
                    ok=bool(getattr(result, "ok", True)),
                )
            except Exception:
                pass
            posted_ok, posted_urls, post_fail_reason = await _post_tool_result(
                channel, result, name=name, agent=agent, silent=silent,
                transport=_transport, session_id=_effective_session_id,
            )
            result.posted_ok = posted_ok
            result.post_fail_reason = post_fail_reason or None
            if posted_urls:
                result.posted_urls = posted_urls
            out.append((name, result))

            # Mid-batch interrupt checkpoint: after each completed tool,
            # check whether the operator queued a message. If so, stop
            # dispatching remaining calls so control returns to runtime
            # and the model sees the new message on its next iteration.
            #
            # ONLY fires for read-only / cheap tools. Attachment-producing
            # tools (generate_image etc.) are NOT interrupted mid-batch:
            # if the operator asked for 4 images, they get all 4 — a queued
            # message waits for the batch to finish rather than throwing
            # away work the operator explicitly requested. The drain still
            # happens, just after the full batch.
            _produced_attachment_this_call = bool(getattr(result, "attachments", None))
            if (interrupt_check is not None
                    and interrupt_check()
                    and not _produced_attachment_this_call):
                remaining = len(tool_calls) - (_tc_idx + 1)
                if remaining > 0:
                    print_ts(
                        f"BATCH-INTERRUPT: operator message queued, "
                        f"skipping {remaining} of {len(tool_calls)} remaining tool calls",
                        agent=agent.id,
                    )
                    # Append a synthetic note to the last executed result
                    # so the model knows the batch was cut short.
                    last_name, last_result = out[-1]
                    note = (
                        f"[BATCH INTERRUPTED] The operator sent a new message. "
                        f"{remaining} remaining tool call(s) in this batch were "
                        f"skipped and were NOT executed."
                    )
                    if last_result.model_feedback is not None:
                        last_result.model_feedback += f"\n{note}"
                    else:
                        last_result.model_feedback = note
                    break

    return out


async def _post_tool_result(
    channel,
    result: ToolResult,
    *,
    name: str,
    agent: Agent,
    silent: bool = False,
    transport=None,
    session_id: str = "",
) -> tuple[Optional[bool], list[str], str]:
    """Post the tool's user-facing output to the messaging transport. Attachments
    normally always post; text output respects:
        - silent_to_discord (per-tool flag) — fully suppress text in chat
        - media_only (per-agent mode) — suppress text but allow attachments
        - silent (per-turn flag) — suppress everything, used for inter-agent
          synthetic turns where there's no human audience on this channel
    The model's view of the result is built separately in runtime via
    build_model_feedback() — this function only handles user-facing posting.

    Returns `(posted_ok, posted_urls, fail_reason)`:
      - `posted_ok` True  — every send call SUCCEEDED. Success is judged by
        the send call itself, NEVER by URL presence: non-Discord transports
        can succeed without returning a URL.
      - `posted_ok` False — posting was attempted and FAILED (fully or
        partially); `fail_reason` says why. The user has NOT seen the output.
      - `posted_ok` None  — posting wasn't attempted (silent turn, tool
        error, or nothing to post).
      - `posted_urls` — CDN/attachment URLs of attachments actually posted,
        in order. Caller stashes them on result.posted_urls so
        build_model_feedback can echo them (lets the model reference prior
        images by URL).
    The caller threads posted_ok/fail_reason onto the ToolResult so feedback
    can say "posting FAILED — the user has NOT seen them" instead of the old
    unconditional "The user can see them" (the lie that made an agent say
    "i already sent it" when nothing posted, 2026-05-29/30).
    """
    if silent:
        # Inter-agent synthetic turn — no human watching this channel for
        # this turn's output. Skip both the text and the attachments. The
        # files still exist on disk (tool wrote them); the model gets the
        # description via build_model_feedback.
        #
        # NOTE: send_file is NOT special-cased here anymore. It no longer
        # returns attachments through this auto-post path at all — it posts
        # directly through the transport itself (see tools/files.py) and
        # reports honest success/failure. So this generic suppression is
        # correct as-is for every tool that DOES return attachments.
        return (None, [], "")

    media_only = (agent.tool_response_mode == "media_only")
    tool = TOOL_REGISTRY.get(name)
    tool_silent = bool(tool and getattr(tool, "silent_to_discord", False))

    if not result.ok:
        # Tool errors do NOT auto-post to chat. The model sees them via
        # build_model_feedback and decides how to communicate.
        return (None, [], "")

    text: Optional[str] = None
    if result.text and not tool_silent and not media_only:
        text = result.text
        # Never post an operator-identifying absolute local path to the channel.
        # The owner (operating their own tools) is exempt so diagnostics stay
        # intact; everyone else gets paths scrubbed. redact_paths only touches
        # home/project paths, so legitimate text/URLs are preserved.
        if not _caller_is_owner_safe():
            text = redact_paths(text)

    attachment_paths = []
    for p in result.attachments:
        try:
            if p and Path(p).exists():
                attachment_paths.append(str(p))
        except Exception:
            pass

    if result.attachments and not attachment_paths:
        # The tool CLAIMED attachments but none exist on disk — nothing can
        # post. That's a delivery failure, not a nothing-to-do.
        print_ts(
            f"{COLOR_RED}tool result for {name}: attachment file(s) missing on "
            f"disk — nothing posted{COLOR_END}", error=True,
        )
        return (False, [], "attachment file(s) missing on disk")

    if not attachment_paths and not text:
        return (None, [], "")

    posted_urls: list[str] = []

    if transport and session_id:
        # Phase 2 path — route through Transport.
        # Send text first if present and no attachments (clean message).
        # If attachments exist, pass text as caption on the first file.
        if attachment_paths:
            sent_count = 0
            last_err = ""
            first = True
            for path in attachment_paths:
                caption = text if first else ""
                try:
                    url = await transport.send_file(session_id, path, content=caption or "")
                    sent_count += 1
                    if url:
                        posted_urls.append(url)
                except Exception as e:
                    last_err = str(e)
                    print_ts(f"{COLOR_RED}transport.send_file failed for {name}: {e}{COLOR_END}", error=True)
                first = False
            if sent_count == len(attachment_paths):
                return (True, posted_urls, "")
            failed = len(attachment_paths) - sent_count
            return (False, posted_urls,
                    f"{failed}/{len(attachment_paths)} upload(s) failed: {last_err}")
        elif text:
            try:
                await transport.send(session_id, text)
                return (True, [], "")
            except Exception as e:
                print_ts(f"{COLOR_RED}transport.send failed for {name}: {e}{COLOR_END}", error=True)
                return (False, [], str(e))
        return (None, [], "")
    else:
        # Legacy fallback — direct channel.send (Discord-specific).
        try:
            import nextcord as _nextcord
            files = [_nextcord.File(p) for p in attachment_paths]
            # 60s timeout — file uploads can take longer than text-only sends,
            # especially for video tool results. Beyond 60s assume the gateway
            # is stuck.
            sent = await asyncio.wait_for(
                channel.send(content=text, files=files or None),
                timeout=60.0,
            )
            posted_urls = [a.url for a in (sent.attachments or [])]
            return (True, posted_urls, "")
        except asyncio.TimeoutError:
            print_ts(f"{COLOR_RED}Tool result post for {name} timed out after 60s{COLOR_END}", error=True)
            return (False, [], "upload timed out after 60s")
        except Exception as e:
            print_ts(f"{COLOR_RED}Failed to post tool result for {name}: {e}{COLOR_END}", error=True)
            return (False, [], str(e))


def _caller_is_owner_safe() -> bool:
    """Owner check that never raises. Fail CLOSED (treat as non-owner) on any
    error so path redaction defaults to ON — a missing context must never
    cause a leak."""
    try:
        from .acl import current_caller_is_owner
        return current_caller_is_owner()
    except Exception:
        return False


def build_model_feedback(name: str, result: ToolResult) -> str:
    """The string the model sees as the tool's output on the next turn.

    Absolute local paths are scrubbed out for non-owner callers via
    redact_paths — this is the single choke point that protects EVERY tool's
    model_feedback / text / error string (including the generic exception
    wrapper in _invoke_tool) from leaking the operator's home dir or layout to
    a non-owner user. The owner is exempt so their own diagnostic tools
    (run_command, etc.) still surface real paths."""
    raw = _build_model_feedback_raw(name, result)
    if _caller_is_owner_safe():
        return raw
    return redact_paths(raw)


def _build_model_feedback_raw(name: str, result: ToolResult) -> str:
    """Compose the raw feedback string (pre-redaction).
    Distinct from what Discord saw — for media tools we tell the model
    'image saved' rather than dumping a file path; for text tools we feed
    the full text back so the model can summarize / answer.

    Delivery honesty: `result.posted_ok` (set by the executor from
    _post_tool_result's explicit outcome) drives what we claim about the
    user having seen the output. False → posting FAILED and the feedback
    says so; True with no URLs is still success (non-Discord transports
    return no URL); None → posting wasn't attempted (silent turn etc.).
    Never infer delivery from URL presence."""
    if not result.ok:
        return f"Tool '{name}' returned an error: {result.error}"

    # Appended to text/model_feedback payloads when the user-facing post
    # failed — the payload itself is still returned (the model needs the
    # data), but it must not believe the user saw it.
    _post_failed_note = ""
    if result.posted_ok is False:
        _post_failed_note = (
            f"\n[POSTING FAILED: {result.post_fail_reason or 'send error'} — "
            f"the user has NOT seen this tool's output. Don't claim it was "
            f"delivered; tell the user the post failed.]"
        )

    if result.model_feedback is not None:
        # Even with a custom model_feedback, append posted URLs so the model
        # can reference prior outputs by URL on follow-up turns.
        if result.posted_urls:
            urls = "\n".join(f"[attachment: {u}]" for u in result.posted_urls)
            return f"{result.model_feedback}\n{urls}{_post_failed_note}"
        return f"{result.model_feedback}{_post_failed_note}"
    if result.text:
        if result.posted_urls:
            urls = "\n".join(f"[attachment: {u}]" for u in result.posted_urls)
            return f"{result.text}\n{urls}{_post_failed_note}"
        return f"{result.text}{_post_failed_note}"
    if result.attachments:
        names = ", ".join(p.name if hasattr(p, "name") else str(p) for p in result.attachments)
        if result.posted_ok is False:
            return (f"Tool '{name}' produced {len(result.attachments)} file(s): {names}, "
                    f"but posting them to the channel FAILED "
                    f"({result.post_fail_reason or 'send error'}) — the user has NOT "
                    f"seen them. Tell the user the post failed instead of claiming "
                    f"delivery; the file(s) still exist on disk.")
        if result.posted_urls:
            urls = "\n".join(f"[attachment: {u}]" for u in result.posted_urls)
            return (f"Tool '{name}' produced {len(result.attachments)} file(s): {names}. "
                    f"The user can see them. You don't see them automatically — pass the URL "
                    f"to follow-up image_url args, or call fetch_discord_message(url) on a CDN "
                    f"URL to re-inject the image into your vision when you actually need to "
                    f"inspect it:\n{urls}")
        if result.posted_ok is None:
            # Posting wasn't attempted — silent inter-agent turn. The files
            # exist on disk but nobody in this channel has seen them.
            return (f"Tool '{name}' produced {len(result.attachments)} file(s): {names}. "
                    f"They were NOT posted to the channel (silent turn — no human "
                    f"audience); the files exist on disk.")
        # posted_ok True, no URLs: a transport that doesn't return URLs
        # (e.g. iMessage) — the send succeeded, the user really can see them.
        return f"Tool '{name}' produced {len(result.attachments)} file(s): {names}. The user can see them."
    return f"Tool '{name}' completed.{_post_failed_note}"
