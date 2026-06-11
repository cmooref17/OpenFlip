"""Restart the openflip gateway and announce the reason after restart.

Pattern (mirrors OpenClaw's restart sentinel):

1. The agent calls `restart_gateway(reason, continuation="")`.
2. This tool writes a sentinel JSON to `restart-sentinel/<uuid>.json` capturing:
   - which agent initiated the restart
   - which channel to announce in
   - the reason
   - an optional continuation prompt to fire as a synthetic turn after restart
3. The tool then triggers `systemctl --user restart openflip` (Linux) or
   `launchctl kickstart -k` (macOS). On Windows it runs OPENFLIP_RESTART_CMD
   if set, else exits cleanly for a supervisor loop (start.bat / NSSM /
   Task Scheduler) to respawn — see docs/WINDOWS.md. The current process dies.
4. On startup, `main.py` scans `restart-sentinel/*.json`. For each:
   - Posts the reason to the saved channel via the saved agent's bot.
   - If a continuation is set, fires a synthetic turn so the agent can act on it.
   - Deletes the sentinel.

This tool is high-impact (restart affects every agent in the framework). Each
agent must explicitly opt in via `allowed_tools` in `agent.json`, and the entry
should restrict access to the owner via `users: [<owner_id>]`.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid

from ._base import tool, ToolResult
from ..utils import print_ts, save_json, load_json, project_root, COLOR_YELLOW, COLOR_END


def _sentinel_dir() -> str:
    return os.path.join(project_root(), "restart-sentinel")


def _check_framework_compiles() -> list[tuple[str, str]]:
    """Run py_compile against every framework .py file. Returns a list of
    (path, error_message) for any file that fails. Empty list = all clean.

    Scans openflip/*.py and openflip/tools/*.py.
    """
    import py_compile
    import glob

    framework_root = os.path.join(project_root(), "openflip")
    if not os.path.isdir(framework_root):
        return []

    files: list[str] = []
    files.extend(glob.glob(os.path.join(framework_root, "*.py")))
    files.extend(glob.glob(os.path.join(framework_root, "tools", "*.py")))

    errors: list[tuple[str, str]] = []
    for f in files:
        try:
            py_compile.compile(f, doraise=True)
        except py_compile.PyCompileError as e:
            # The .msg attribute has the human-readable error
            errors.append((os.path.relpath(f, project_root()), str(e.msg).strip()))
        except Exception as e:
            errors.append((os.path.relpath(f, project_root()), f"{type(e).__name__}: {e}"))
    return errors


def _check_agent_configs() -> list[tuple[str, str]]:
    """Validate every agents/<id>/agent.json as parseable JSON. Returns a list
    of (path, error_message) for any file that fails. Empty list = all clean.

    Catches the case where a bad agent.json edit (typo, trailing comma,
    missing brace) would crash the framework on startup when main.py tries
    to load it. Skips directories starting with '_' (e.g. _shared/).
    """
    import json
    import glob

    agents_root = os.path.join(project_root(), "agents")
    if not os.path.isdir(agents_root):
        return []

    errors: list[tuple[str, str]] = []
    for agent_json in glob.glob(os.path.join(agents_root, "*", "agent.json")):
        # Skip _shared and any other directory that starts with underscore.
        agent_dir = os.path.basename(os.path.dirname(agent_json))
        if agent_dir.startswith("_"):
            continue
        try:
            with open(agent_json, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            errors.append((os.path.relpath(agent_json, project_root()), f"JSON parse error: {e}"))
        except OSError as e:
            errors.append((os.path.relpath(agent_json, project_root()), f"OS error: {e}"))
        except Exception as e:
            errors.append((os.path.relpath(agent_json, project_root()), f"{type(e).__name__}: {e}"))
    return errors


def _check_other_agents_busy(*, exclude_agent_id: str, stale_after_s: float = 60.0) -> list[dict]:
    """Read every other agent's live.json and return a list of those currently
    in a turn or running a tool. Stale entries (last_active_ms older than
    stale_after_s) are treated as not busy — they're either idle or crashed.

    Returns a list of {agent_id, activity, last_active_ms, age_s} dicts,
    or [] if nobody else is busy.
    """
    import time
    import os

    agents_dir = os.path.join(project_root(), "agents")
    if not os.path.isdir(agents_dir):
        return []

    busy = []
    now_ms = int(time.time() * 1000)
    for entry in os.listdir(agents_dir):
        if entry.startswith("_") or entry == exclude_agent_id:
            continue
        live_path = os.path.join(agents_dir, entry, "live.json")
        if not os.path.isfile(live_path):
            continue
        state = load_json(live_path, default={}) or {}
        is_busy = bool(state.get("is_in_turn")) or state.get("current_tool") is not None
        if not is_busy:
            continue
        last_active_ms = int(state.get("last_active_ms") or 0)
        age_s = (now_ms - last_active_ms) / 1000.0 if last_active_ms else 999999.0
        if age_s > stale_after_s:
            # Stale — agent likely crashed or hung; safe to restart over it.
            continue
        busy.append({
            "agent_id": entry,
            "activity": state.get("activity") or "(unknown)",
            "current_tool": state.get("current_tool"),
            "last_active_ms": last_active_ms,
            "age_s": age_s,
        })
    return busy


def _check_inbound_queues_pending(*, exclude_agent_id: str = "") -> list[dict]:
    """Return list of {agent_id, queue_size} for agents with queued
    inbound work (Discord messages or synthetic turns waiting to be
    processed).

    IMPORTANT — checks the CALLING agent's queue too. The in-flight item
    that triggered this restart_gateway call has already been dequeued
    by the worker before _run_turn started, so qsize at this moment
    reflects items queued AFTER the current turn began — exactly the
    items a restart would silently discard (e.g., the user's reply queued
    while the calling agent was mid-turn).

    Earlier version (2026-05-10) excluded the caller on the mistaken
    assumption that its qsize included the in-flight item. That bug
    let the user's queued message get dropped when an agent restarted on
    2026-05-11.
    """
    from ..registry import RUNNERS
    pending = []
    for aid, runner in RUNNERS.items():
        if exclude_agent_id and aid == exclude_agent_id:
            continue
        try:
            qs = runner._inbound_queue.qsize()
            if qs > 0:
                pending.append({"agent_id": aid, "queue_size": qs})
        except Exception:
            pass
    return pending


def _check_recent_human_activity(*, exclude_agent_id: str, window_s: float = 5.0) -> list[dict]:
    """Return list of {agent_id, age_s} for agents that received an
    inbound human Discord message within the last `window_s` seconds.

    Catches the case where a human just hit enter on a message but the
    agent's runtime hasn't queued it yet, OR the message is being
    processed and isn't visible in qsize() anymore. Without this, a
    restart fired within ~5s of a human sending a message can blow
    away their context mid-reply.
    """
    import time
    from ..registry import RUNNERS
    now_ms = int(time.time() * 1000)
    recent = []
    for aid, runner in RUNNERS.items():
        if aid == exclude_agent_id:
            continue
        last_ms = int(getattr(runner, "last_human_inbound_ms", 0) or 0)
        if not last_ms:
            continue
        age_s = (now_ms - last_ms) / 1000.0
        if age_s <= window_s:
            recent.append({"agent_id": aid, "age_s": age_s})
    return recent


@tool
async def restart_gateway(reason: str, continuation: str = "", force: bool = False) -> ToolResult:
    """Restart the openflip framework. After restart completes, the new process
    will post `reason` as a message in the channel where this tool was invoked,
    and (if `continuation` is provided) fire a synthetic turn with that prompt
    so the agent can act on it immediately.

    Owner-only. Restarts the entire framework — every agent goes offline
    briefly. Use sparingly.

    Preflight: refuses to restart if any OTHER agent is currently in a turn or
    running a tool. Stale activity (>60s old) is ignored — assumed crashed.
    The calling agent itself is always excluded from the check (it's by
    definition mid-tool when calling this).

    Args:
        reason: Human-readable explanation. Posted to Discord after restart.
        continuation: Optional follow-up prompt the agent should respond to
            once the gateway is back up. Leave empty for just the announcement.
        force: If True, skip the preflight busy-check and restart anyway.
            Use only for emergencies (e.g., framework is wedged).
    """
    reason = (reason or "").strip()
    if not reason:
        return ToolResult.fail("`reason` is required — say why you're restarting.")

    from ..acl import current_caller_is_owner
    if not current_caller_is_owner():
        return ToolResult.fail("restart_gateway is owner-only.")

    # Pull invocation context from the executor's contextvars.
    from ..tool_executor import CURRENT_AGENT, CURRENT_CHANNEL_ID, CURRENT_SPEAKER_ID
    try:
        agent = CURRENT_AGENT.get()
    except LookupError:
        return ToolResult.fail("Tool invoked outside an agent context.")
    try:
        channel_id = int(CURRENT_CHANNEL_ID.get())
    except LookupError:
        return ToolResult.fail("No channel context available for restart announcement.")
    try:
        speaker_id = int(CURRENT_SPEAKER_ID.get())
    except LookupError:
        speaker_id = 0

    # Preflight: refuse if any other agent is mid-work, unless force=True.
    if not force:
        busy = _check_other_agents_busy(exclude_agent_id=agent.id)
        if busy:
            lines = [
                f"- {b['agent_id']}: {b['activity']}"
                + (f" (tool: {b['current_tool']})" if b.get('current_tool') else "")
                + f" — {b['age_s']:.1f}s ago"
                for b in busy
            ]
            details = "\n".join(lines)
            return ToolResult.fail(
                f"Preflight blocked: {len(busy)} other agent(s) currently mid-work. "
                f"Wait for them to finish, or pass force=True to restart anyway.\n{details}"
            )

    # Preflight: refuse if any other agent has queued inbound work
    # (Discord messages or synthetic turns sitting in their queue).
    # A restart silently discards them — the user's queued message gets
    # dropped, agent-to-agent chains die mid-flight. Added 2026-05-10
    # after the user lost a queued message to a same-turn restart.
    if not force:
        # No exclusion — the calling agent's queue can also contain items
        # queued AFTER the current turn started (the user's reply that arrived
        # while we were processing something else). The in-flight item
        # itself was already dequeued before _run_turn started, so qsize
        # only reflects newly-queued work we'd discard on restart.
        pending = _check_inbound_queues_pending()
        if pending:
            lines = [f"- {p['agent_id']}: {p['queue_size']} item(s) queued" for p in pending]
            details = "\n".join(lines)
            return ToolResult.fail(
                f"Preflight blocked: {len(pending)} agent(s) have queued inbound work. "
                f"A restart would discard those messages. Wait for processing, "
                f"or pass force=True to restart anyway.\n{details}"
            )

    # Preflight: refuse if any other agent just received a human Discord
    # message in the last 5s. Catches the "human just hit enter, queue
    # hasn't picked it up yet" race. Window is short because anything
    # older than 5s is either being processed (busy check) or queued
    # (qsize check) — this just covers the milliseconds between Discord
    # webhook arrival and queue enqueueing.
    if not force:
        recent = _check_recent_human_activity(exclude_agent_id=agent.id, window_s=5.0)
        if recent:
            lines = [f"- {r['agent_id']}: human message {r['age_s']:.1f}s ago" for r in recent]
            details = "\n".join(lines)
            return ToolResult.fail(
                f"Preflight blocked: {len(recent)} agent(s) received a human message in the last 5s. "
                f"Wait a moment, or pass force=True to restart anyway.\n{details}"
            )

    # Compile preflight: refuse to restart if any framework code has a syntax
    # error. Without this, a bad write to runtime.py + a restart_gateway call
    # = restart-loop hell because the new process crashes on import. force=True
    # bypasses (e.g., when you genuinely need to restart over a broken state).
    if not force:
        compile_errors = _check_framework_compiles()
        if compile_errors:
            details = "\n".join(f"- {p}: {err}" for p, err in compile_errors[:5])
            more = f"\n  ...and {len(compile_errors) - 5} more" if len(compile_errors) > 5 else ""
            return ToolResult.fail(
                f"Preflight blocked: {len(compile_errors)} framework file(s) have syntax errors. "
                f"Fix them or pass force=True to restart anyway (NOT recommended — the new "
                f"process will likely crash on import).\n{details}{more}"
            )

    # Agent config preflight: refuse to restart if any agent.json is malformed.
    # The framework iterates over agent dirs on startup and parses each
    # agent.json — a bad JSON file there breaks discovery and prevents the
    # framework from coming up cleanly. Added 2026-05-19 (audit P0 #8).
    if not force:
        config_errors = _check_agent_configs()
        if config_errors:
            details = "\n".join(f"- {p}: {err}" for p, err in config_errors[:5])
            more = f"\n  ...and {len(config_errors) - 5} more" if len(config_errors) > 5 else ""
            return ToolResult.fail(
                f"Preflight blocked: {len(config_errors)} agent config file(s) are malformed. "
                f"Fix them or pass force=True to restart anyway (the framework will likely "
                f"fail to load the affected agents).\n{details}{more}"
            )

    # Build sentinel
    sid = str(uuid.uuid4())
    sentinel = {
        "version": 1,
        "id": sid,
        "ts_ms": int(time.time() * 1000),
        "agent_id": agent.id,
        "channel_id": channel_id,
        "speaker_id": speaker_id,
        "reason": reason,
        "continuation": continuation.strip() or None,
    }

    # Sign the sentinel before writing. Without this any agent/tool that
    # can drop a file into the sentinel dir could forge a continuation
    # prompt and have it fire as an owner-attributed synthetic turn on
    # the next restart (MED-3 from the security audit). The writer (us)
    # and verifier (restart_sentinel._process_one) share data/sentinel_hmac_key.
    from ..restart_sentinel import sign_payload as _sign_payload
    sentinel["signature"] = _sign_payload(sentinel)

    sentinel_dir = _sentinel_dir()
    try:
        os.makedirs(sentinel_dir, exist_ok=True)
        save_json(os.path.join(sentinel_dir, f"{sid}.json"), sentinel)
    except Exception as e:
        return ToolResult.fail(f"Failed to write restart sentinel: {e}")

    print_ts(f"{COLOR_YELLOW}restart_gateway: sentinel written ({sid}); restarting now — reason: {reason!r}{COLOR_END}", agent=agent.id)
    try:
        from .. import events_log as _events_log
        _events_log.log_event(
            agent.id, "restart",
            reason=reason[:200],
            has_continuation=bool(continuation),
        )
    except Exception:
        pass

    # Persist the in-flight conversation BEFORE systemctl SIGTERMs us. The
    # turn that's calling restart_gateway has an assistant message with a
    # tool_use block in conv.messages that conv.save() hasn't written yet —
    # because the normal save() fires at end-of-_run_turn AFTER all tools
    # complete, and this tool is going to kill the process before we
    # return to that code path. Without this save, post-restart-me loads
    # the .jsonl and sees no record of the call she made. That's the
    # resume-fear bug from 2026-05-19's audit: new instance has no
    # in-history proof of an action that already happened in the world.
    #
    # Save is best-effort — a torn write here is less bad than no write.
    try:
        from ..registry import RUNNERS
        runner = RUNNERS.get(agent.id)
        if runner and channel_id:
            # conv_key: identity-linked channels share a "linked:<canonical>"
            # dict key; unlinked ids pass through unchanged.
            conv = runner.conversations.get(runner.conv_key(int(channel_id)))
            if conv is not None:
                conv.save()
                print_ts(
                    f"restart_gateway: persisted in-flight conversation before restart",
                    agent=agent.id,
                )
                # Also stash a synthetic tool_result marker file the post-
                # restart sentinel processor will append to .jsonl. This
                # closes the orphan tool_use that would otherwise 400 the
                # next API call.
                # Resolve the real conversation_id from CURRENT_SESSION.
                # No fallback — every tool call runs inside a turn that
                # has CURRENT_SESSION set (see _run_turn). If we somehow
                # got here without one, raise loud rather than mis-write
                # the marker with the wrong transport prefix.
                from ..tool_executor import CURRENT_SESSION as _CS
                _ss = _CS.get(None)
                if _ss is None:
                    raise RuntimeError(
                        "restart_gateway marker: CURRENT_SESSION is None — "
                        "cannot resolve conversation_id without a session."
                    )
                _conv_id = getattr(_ss, "conversation_id", "") or ""
                if not _conv_id:
                    raise RuntimeError(
                        f"restart_gateway marker: session has no conversation_id "
                        f"(session={_ss!r})"
                    )
                # Capture the originator's handle so the post-restart
                # continuation can re-create an ACL-equivalent session.
                # Without it, the synthesized session has handle="" and
                # every iMessage ACL fails -> tools missing from API request
                # -> model emits tool calls as raw JSON text in chat.
                _orig_handle = getattr(_ss, "handle", "") or ""
                marker = {
                    "sentinel_id": sid,
                    "agent_id": agent.id,
                    "channel_id": int(channel_id),
                    "conversation_id": _conv_id,
                    "originator_handle": _orig_handle,
                    "reason": reason,
                    "ts": time.time(),
                }
                marker_path = os.path.join(sentinel_dir, f"{sid}.tool_result.json")
                save_json(marker_path, marker)
    except Exception as _save_err:
        print_ts(
            f"{COLOR_YELLOW}restart_gateway: pre-restart conv.save failed: {_save_err}{COLOR_END}",
            agent=agent.id,
        )

    # Defensive: kill any rogue openflip.main process that isn't us before
    # invoking systemctl. systemctl only manages processes it spawned —
    # if someone started openflip manually via ./start.sh (or an earlier
    # crash leaked a process), systemctl restart wouldn't touch it and
    # you'd end up with two openflip processes, both bots connected,
    # duplicate replies on every message. pkill -9 against
    # "openflip.main" with exclusion of our own PID nukes the rogues.
    # Systemd then restarts its own copy cleanly. Our process dies in
    # the systemctl restart a moment later.
    # POSIX-only (pgrep/xargs/kill don't exist on Windows; the Windows
    # restart path below is exit-and-respawn, which can't leak a rogue).
    if sys.platform != "win32":
        try:
            own_pid = os.getpid()
            cleanup = await asyncio.create_subprocess_shell(
                f"pgrep -f 'openflip\\.main' | grep -v '^{own_pid}$' | xargs -r kill -9",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(cleanup.communicate(), timeout=3)
            except asyncio.TimeoutError:
                pass
        except Exception as _kill_err:
            print_ts(
                f"{COLOR_YELLOW}restart_gateway: orphan-process cleanup failed (continuing): {_kill_err}{COLOR_END}",
                agent=agent.id,
            )

    # Windows: there is no systemd/launchd. Two supported restart stories,
    # in order of precedence:
    #   1. OPENFLIP_RESTART_CMD — an operator-provided shell command that
    #      restarts the service from outside (e.g. NSSM: `nssm restart
    #      openflip`). We run it and expect it to kill us.
    #   2. OPENFLIP_SUPERVISED=1 — set by start.bat's relaunch loop (or by
    #      an NSSM / Task Scheduler config with restart-on-exit). "Restart"
    #      then means: exit cleanly and let the supervisor respawn us; the
    #      sentinel announces after the relaunch.
    # With neither set, a restart would just take the framework DOWN with
    # nothing to bring it back — refuse with instructions instead.
    if sys.platform == "win32":
        _win_cmd = os.environ.get("OPENFLIP_RESTART_CMD", "").strip()
        if _win_cmd:
            try:
                await asyncio.create_subprocess_shell(
                    _win_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except Exception as e:
                try:
                    os.remove(os.path.join(sentinel_dir, f"{sid}.json"))
                except Exception:
                    pass
                return ToolResult.fail(f"Failed to invoke OPENFLIP_RESTART_CMD ({_win_cmd!r}): {e}")
            # The command is expected to terminate this process. If we're
            # still alive after 20s, it didn't do its job.
            await asyncio.sleep(20.0)
            return ToolResult.fail(
                f"OPENFLIP_RESTART_CMD ran but this process is still alive after 20s — "
                f"check that {_win_cmd!r} actually restarts openflip."
            )
        if os.environ.get("OPENFLIP_SUPERVISED", "").strip() in ("1", "true", "yes"):
            print_ts(
                f"{COLOR_YELLOW}restart_gateway: exiting for supervisor relaunch "
                f"(Windows, OPENFLIP_SUPERVISED set){COLOR_END}",
                agent=agent.id,
            )
            # Short delay so the log line + any in-flight file writes land.
            # os._exit skips cleanup the same way systemd's kill would.
            asyncio.get_running_loop().call_later(2.0, os._exit, 0)
            await asyncio.sleep(20.0)
            return ToolResult.fail("process did not exit — supervisor relaunch path failed.")
        # Neither configured — restarting would strand the framework offline.
        for _f in (f"{sid}.json", f"{sid}.tool_result.json"):
            try:
                os.remove(os.path.join(sentinel_dir, _f))
            except OSError:
                pass
        return ToolResult.fail(
            "restart_gateway is not configured on this Windows host. Either run "
            "openflip via start.bat (its relaunch loop sets OPENFLIP_SUPERVISED=1), "
            "or set OPENFLIP_RESTART_CMD to a command that restarts the service "
            "(e.g. `nssm restart openflip`). Without one of these, a restart would "
            "leave the framework offline with nothing to bring it back."
        )

    # Trigger the restart. The current process will be terminated by the
    # service manager before this returns. Platform-aware: systemd on Linux,
    # launchd on macOS (under KeepAlive the -k kill triggers an immediate respawn).
    if sys.platform == "darwin":
        _label = os.environ.get("OPENFLIP_LAUNCHD_LABEL", "ai.openflip")
        _restart_cmd = f"launchctl kickstart -k gui/{os.getuid()}/{_label}"
        _status_cmd = f"launchctl print gui/{os.getuid()}/{_label}"
    else:
        _unit = os.environ.get("OPENFLIP_SYSTEMD_UNIT", "openflip")
        _restart_cmd = f"systemctl --user restart {_unit}"
        _status_cmd = f"systemctl --user is-active {_unit}"
    try:
        proc = await asyncio.create_subprocess_shell(
            _restart_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        # We don't await communicate() because the restart will kill us.
        # Earlier versions of this tool used a fixed asyncio.sleep then
        # blindly returned "still running" if we hadn't been SIGKILLed yet,
        # producing false-alarm errors whenever graceful shutdown took
        # longer than the sleep. Real fix landed 2026-05-24: main.py now
        # has a SIGTERM handler that closes runners cleanly so shutdown
        # completes in a few seconds — but we still poll systemctl state
        # instead of guessing, so a slow shutdown never reads as a failure.
        # We poll up to 20s. With the SIGTERM handler real shutdowns finish
        # in ~5s; 20s gives generous headroom while still failing fast if a
        # regression breaks the shutdown path again. "active" means restart
        # already finished and we're alive in the NEW process (shouldn't
        # happen — we should have been killed first — but treat as success).
        # "deactivating" / "activating" / "inactive" mean the restart is in
        # progress. Anything else is a true failure.
        deadline = asyncio.get_running_loop().time() + 20.0
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(1.0)
            check = await asyncio.create_subprocess_shell(
                _status_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                stdout, _ = await asyncio.wait_for(check.communicate(), timeout=3.0)
            except asyncio.TimeoutError:
                continue
            raw = stdout.decode("utf-8", errors="replace").strip()
            if sys.platform == "darwin":
                # launchctl print emits a block; find the "state = ..." line.
                state = ""
                for _line in raw.splitlines():
                    _l = _line.strip()
                    if _l.startswith("state = "):
                        state = _l[len("state = "):].strip()
                        break
                if state == "running":
                    return ToolResult(model_feedback="Restart completed.")
                continue
            state = raw
            if state in ("deactivating", "activating", "inactive"):
                # Restart in progress; keep waiting. Our process will
                # disappear before we see "active" — that's expected.
                continue
            if state == "active":
                # We're still running on the new process side. Treat as ok.
                return ToolResult(model_feedback="Restart completed.")
            # Unknown state ("failed", "reloading", etc.) — real problem.
            return ToolResult.fail(f"service manager reports unexpected state '{state}' after restart request.")
        return ToolResult.fail(f"Restart did not complete within 20s (status cmd: {_status_cmd}).")
    except Exception as e:
        # If we get here, the subprocess didn't even start.
        try:
            os.remove(os.path.join(sentinel_dir, f"{sid}.json"))
        except Exception:
            pass
        return ToolResult.fail(f"Failed to invoke restart command '{_restart_cmd}': {e}")
