"""Startup processor for restart sentinel files.

When `tools/restart.py` triggers a gateway restart, it leaves a sentinel JSON
in `restart-sentinel/<id>.json`. On the next startup, this module reads each
sentinel and:

1. Posts the saved `reason` to the saved `channel_id` via the saved `agent_id`'s bot.
2. Optionally fires a synthetic turn with the saved `continuation` prompt.
3. Deletes the sentinel.

Failures are logged but never block. A bad sentinel is moved to `<id>.json.bad`
so it doesn't loop on every restart.
"""
from __future__ import annotations
import asyncio
import os
import re
from typing import Any, Optional

from .registry import RUNNERS
from .utils import print_ts, load_json, project_root, COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_END


# Canonical conversation_id shape: "<transport>:<id>" (e.g. "discord:12345",
# "imessage:you@example.com"). The sentinel is HMAC-verified but its sibling
# marker file is NOT, so a corrupt/malicious marker_conv_id like "../../etc/foo"
# would traverse out of conversations/ when joined into "<id>.jsonl". Validate
# against this shape before any path join. Defense in depth.
_CONV_ID_SHAPE = re.compile(r"^[A-Za-z0-9_]+:[A-Za-z0-9_.+@-]+$")


_DIR_NAME = "restart-sentinel"
_HMAC_KEY_FILENAME = "sentinel_hmac_key"


def _hmac_key_path() -> str:
    return os.path.join(project_root(), "data", _HMAC_KEY_FILENAME)


def get_or_create_hmac_key() -> bytes:
    """Return the persisted HMAC key for signing/verifying sentinel files.

    Creates one if absent. The key lives in data/sentinel_hmac_key with
    0600 perms — only the openflip process user can read it. Tools that
    write sentinels (restart_gateway) sign with this; the startup processor
    verifies. Any sentinel without a valid signature is treated as forged
    (MED-3 from the security audit).
    """
    import secrets as _secrets
    path = _hmac_key_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        pass
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read().strip()
    # Generate. 32 bytes is plenty for HMAC-SHA256.
    key = _secrets.token_bytes(32)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(key)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return key


def sign_payload(payload: dict) -> str:
    """Compute the HMAC-SHA256 over a stable serialization of `payload`.

    Returns the hex digest. Caller stamps it into the sentinel under
    `signature` before writing to disk. Verification recomputes over
    the payload MINUS the `signature` field and constant-time compares.
    """
    import hmac as _hmac
    import hashlib as _hashlib
    import json as _json
    key = get_or_create_hmac_key()
    # Stable serialization — sorted keys, no whitespace variation — so the
    # writer and verifier compute over identical bytes.
    body = _json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return _hmac.new(key, body, _hashlib.sha256).hexdigest()


def verify_signature(sentinel: dict) -> bool:
    """Verify a sentinel file's HMAC.

    Returns True only if `sentinel["signature"]` matches an HMAC computed
    over the sentinel with that field removed. Constant-time compare via
    hmac.compare_digest. Missing/bad signature = False.
    """
    import hmac as _hmac
    sig = sentinel.get("signature")
    if not sig or not isinstance(sig, str):
        return False
    payload = {k: v for k, v in sentinel.items() if k != "signature"}
    expected = sign_payload(payload)
    return _hmac.compare_digest(sig, expected)
# When openflip restarts during a Discord rate-limit window, the agent's
# AgentRunner sits in a 5-min reconnect floor before it can come online.
# A 30s timeout would give up before the first reconnect even completes,
# leaving the user with no "I'm back" announcement. Bumping this to 15min
# covers the worst-case rate-limit floor; if still not ready by then, the
# sentinel is left on disk so the next process attempt picks it up.
_BOT_READY_TIMEOUT = 900.0
_BOT_READY_POLL_S = 1.0


def _dir() -> str:
    return os.path.join(project_root(), _DIR_NAME)


async def _wait_runner_ready(runner, timeout_s: float) -> bool:
    """Wait until the runner can actually deliver outbound messages.

    Discord needs a real handshake — poll until `transport.bot.is_ready()`.
    Other transports (iMessage, internal/null) are ready synchronously when
    the runner lands in RUNNERS: nothing to wait for."""
    transport = getattr(runner, "transport", None)
    if transport is None:
        return False
    # Discord-specific handshake. Detect by the presence of `bot` rather than
    # by transport name so a hypothetical second Discord-shaped transport
    # (eg. Slack via discord.py-style bot) would still get the poll.
    if hasattr(transport, "bot"):
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            bot = getattr(transport, "bot", None)
            if bot is not None and getattr(bot, "user", None) is not None and bot.is_ready():
                return True
            await asyncio.sleep(_BOT_READY_POLL_S)
        return False
    # Non-Discord transports: ready by virtue of being attached to a running
    # runner. The transport.send() call below will surface its own errors.
    return True


async def _process_one(path: str) -> None:
    sentinel = load_json(path, default=None)
    if not isinstance(sentinel, dict):
        print_ts(f"{COLOR_YELLOW}restart-sentinel: bad sentinel at {path}; renaming{COLOR_END}")
        try:
            # os.replace, not os.rename — rename fails on Windows when the
            # .bad target already exists from an earlier pass.
            os.replace(path, path + ".bad")
        except OSError:
            pass
        return

    # HMAC verify BEFORE doing anything with this sentinel. Without this
    # any process that can write into restart-sentinel/ could forge a
    # continuation prompt and have it fire as an owner-attributed turn
    # on this restart. Unsigned / mis-signed sentinels are renamed to
    # .forged and skipped entirely.
    if not verify_signature(sentinel):
        print_ts(
            f"{COLOR_RED}restart-sentinel: signature verification FAILED for {path} — "
            f"sentinel is forged or stale (pre-HMAC). Renaming to .forged; "
            f"NOT processing.{COLOR_END}",
            error=True,
        )
        try:
            os.replace(path, path + ".forged")
        except OSError:
            pass
        return

    agent_id = sentinel.get("agent_id")
    channel_id = sentinel.get("channel_id")
    channel_transport = sentinel.get("channel_transport") or ""
    reason = sentinel.get("reason") or "Gateway restarted."
    continuation = sentinel.get("continuation") or None

    if not agent_id or not channel_id:
        print_ts(f"{COLOR_YELLOW}restart-sentinel: incomplete sentinel at {path}; deleting{COLOR_END}")
        try:
            os.remove(path)
        except OSError:
            pass
        return

    runner = RUNNERS.get(agent_id)
    if not runner:
        print_ts(f"{COLOR_YELLOW}restart-sentinel: agent '{agent_id}' not running; deleting sentinel{COLOR_END}")
        try:
            os.remove(path)
        except OSError:
            pass
        return

    print_ts(
        f"restart-sentinel: waiting up to {int(_BOT_READY_TIMEOUT)}s for agent '{agent_id}' to come online…",
        agent=agent_id,
    )
    ready = await _wait_runner_ready(runner, _BOT_READY_TIMEOUT)
    if not ready:
        print_ts(f"{COLOR_RED}restart-sentinel: agent '{agent_id}' still not ready after {int(_BOT_READY_TIMEOUT)}s; leaving sentinel for next try{COLOR_END}", error=True)
        return

    # No channel-object lookup here — the transport handles its own
    # session resolution inside transport.send(). For NullTransport (headless
    # agents), send is a documented no-op; the announce just goes nowhere,
    # which is correct for an agent with no surface.
    session_id_str = str(channel_id)

    # Inject a synthetic tool_result for the restart_gateway call before
    # firing the continuation. This closes the orphan tool_use that
    # tools/restart.py persisted to .jsonl before systemctl killed us.
    # Without this, the next API call after restart 400s with
    # "tool_use without matching tool_result." With it, post-restart-me
    # sees in her OWN context: assistant called restart_gateway →
    # tool returned "Gateway restarted (sentinel id X)." Resume-fear fix.
    marker_path = path[:-5] + ".tool_result.json"  # <id>.json -> <id>.tool_result.json
    # Conversation_id from the marker — transport-prefixed ("discord:1234",
    # "imessage:5678"). No default: if we can't determine it, the marker
    # is malformed (or stale from pre-imessage code) and we skip processing
    # rather than mis-write a "discord:" file for what might be an
    # iMessage conversation.
    marker_conv_id = ""
    marker_stale = False
    _orig_handle = ""
    if os.path.isfile(marker_path):
        try:
            marker = load_json(marker_path, default=None)
            if isinstance(marker, dict):
                conv_id = marker.get("conversation_id") or ""
                _orig_handle = marker.get("originator_handle") or ""
                if not conv_id:
                    print_ts(
                        f"{COLOR_YELLOW}restart-sentinel: marker {marker_path} "
                        f"has no conversation_id; skipping tool_result injection. "
                        f"(stale marker from pre-prefix-fix code?){COLOR_END}",
                        agent=agent_id,
                    )
                    marker_stale = True
                else:
                    marker_conv_id = conv_id
                    # The marker is UNSIGNED — validate marker_conv_id against
                    # the canonical transport:id shape before it reaches any
                    # path join below. Reject path separators, "..", control
                    # chars, or anything off-shape; treat as a stale marker so
                    # the injection is skipped and the marker still gets deleted.
                    if ("/" in marker_conv_id
                            or "\\" in marker_conv_id
                            or ".." in marker_conv_id
                            or any(ord(c) < 0x20 for c in marker_conv_id)
                            or not _CONV_ID_SHAPE.match(marker_conv_id)):
                        print_ts(
                            f"{COLOR_RED}restart-sentinel: marker {marker_path} "
                            f"has malformed conversation_id {marker_conv_id!r}; "
                            f"skipping tool_result injection (path-traversal "
                            f"guard).{COLOR_END}",
                            error=True, agent=agent_id,
                        )
                        marker_stale = True
                        # Clear so the SECOND path join in the continuation
                        # block (below) is also protected — matches the empty
                        # marker_conv_id left by the no-conversation_id case.
                        marker_conv_id = ""
            if marker_stale or not marker_conv_id:
                # Bail on tool_result injection cleanly — let the
                # outer except keep running so the marker still gets
                # deleted at end (one-shot semantics preserved).
                raise RuntimeError("marker missing conversation_id")
            # conversation_path handles the Windows filename encoding
            # (":" → "%3A") and re-validates the id — never join the raw
            # conversation_id into a filename here.
            from ._conversation_io import conversation_path as _conv_path_for
            conv_path = _conv_path_for(
                os.path.join(project_root(), "agents", agent_id),
                marker_conv_id,
            )
            # Find the most recent assistant message with a
            # restart_gateway tool_use to learn its tool_use_id.
            tool_use_id = None
            try:
                import json as _json
                with open(conv_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                    except Exception:
                        continue
                    content = obj.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if (isinstance(block, dict)
                                    and block.get("type") == "tool_use"
                                    and block.get("name") == "restart_gateway"):
                                tool_use_id = block.get("id")
                                break
                    if tool_use_id:
                        break
            except Exception as _scan_err:
                print_ts(
                    f"{COLOR_YELLOW}restart-sentinel: scan for tool_use_id failed: {_scan_err}{COLOR_END}",
                    agent=agent_id,
                )

            # Append the synthetic tool_result. Even if we couldn't
            # find the tool_use_id we still write a marker — the API
            # will tolerate an extra plain user message even if it
            # 400s an orphan tool_use.
            synth_msg = {
                "role": "tool",
                "content": (
                    f"[restart_gateway completed successfully — sentinel id "
                    f"{marker.get('sentinel_id','?')}, agent reloaded, "
                    f"all framework changes effective. This message was "
                    f"injected by the post-restart sentinel processor.]"
                ),
                "ts": __import__("time").time(),
            }
            if tool_use_id:
                synth_msg["tool_use_id"] = tool_use_id
            try:
                with open(conv_path, "a", encoding="utf-8") as f:
                    import json as _json2
                    f.write(_json2.dumps(synth_msg, ensure_ascii=False) + "\n")
                print_ts(
                    f"{COLOR_GREEN}restart-sentinel: injected synthetic restart tool_result "
                    f"(tool_use_id={tool_use_id or 'unmatched'}){COLOR_END}",
                    agent=agent_id,
                )
            except Exception as _write_err:
                print_ts(
                    f"{COLOR_YELLOW}restart-sentinel: failed to append tool_result: {_write_err}{COLOR_END}",
                    agent=agent_id,
                )
        except Exception as _marker_err:
            print_ts(
                f"{COLOR_YELLOW}restart-sentinel: tool_result marker handling failed: {_marker_err}{COLOR_END}",
                agent=agent_id,
            )
        # Always delete the marker, success or fail — one-shot semantics.
        try:
            os.remove(marker_path)
        except OSError:
            pass

    announce = f"♻️ Gateway restarted.\n**Reason:** {reason}"
    announce_ok = False
    # Route the announce through the transport that OWNS this channel, not the
    # agent's primary transport. `runner.transport` is `_transports[0]`; on a
    # multi-transport agent that may not be the transport the restart was
    # requested on, so the announce would go out the wrong pipe (a Discord
    # channel id handed to `imsg send`, silently dropped). Match by the
    # transport name the sentinel recorded; fall back to primary when no
    # transport was recorded (older sentinels) or none matches — that
    # fallback is exactly the historical behavior and is correct for
    # single-transport agents, where the one transport is always the primary.
    announce_transport = runner.transport
    if channel_transport:
        for _t in getattr(runner, "_transports", None) or [runner.transport]:
            if getattr(_t, "name", "") == channel_transport:
                announce_transport = _t
                break
    try:
        await asyncio.wait_for(
            announce_transport.send(session_id_str, announce),
            timeout=30.0,
        )
        # transport.send is documented to log-and-swallow its own delivery
        # errors rather than raise — so reaching this point means we issued
        # the send call cleanly. Surfaces would still log a failure separately.
        announce_ok = True
        print_ts(
            f"{COLOR_GREEN}restart-sentinel: announced restart via {getattr(announce_transport, 'name', '?')} "
            f"(agent={agent_id}, session={session_id_str}){COLOR_END}",
            agent=agent_id,
        )
    except asyncio.TimeoutError:
        print_ts(f"{COLOR_RED}restart-sentinel: announce timed out after 30s (agent={agent_id}){COLOR_END}", error=True, agent=agent_id)
    except Exception as e:
        print_ts(f"{COLOR_RED}restart-sentinel: failed to post announcement: {e}{COLOR_END}", error=True, agent=agent_id)

    # If the announce failed, skip the continuation. Sentinel still gets
    # deleted at end (line 183) — "tried once, dropped" semantic. Without
    # this skip, a stuck announce would still fire the continuation, which
    # might post to a channel the operator can't see, OR worse, fire a
    # synthetic turn that produces output the user never sees the announce
    # context for. Safer to skip and let next restart try again.
    if continuation and not announce_ok:
        print_ts(f"{COLOR_YELLOW}restart-sentinel: skipping continuation because announce failed (agent={agent_id}){COLOR_END}", agent=agent_id)
        continuation = ""

    if continuation:
        # If the human spoke right before the restart and the agent hadn't
        # answered yet, the conversation file's last message is a user
        # message. Firing the continuation turn in that state makes the
        # agent respond to the continuation prompt instead of the human's
        # actual question — the question gets stranded in history,
        # unanswered. Skip the continuation in that case; the human's
        # next Discord message will wake the agent up naturally with
        # their original question still at the tail of the conversation.
        # marker_conv_id can legitimately be "" here (no marker file / stale
        # marker) — conversation_path fail-closes on empty ids, so keep "" as
        # a never-exists path and let the os.path.exists guard below skip it.
        conv_path = ""
        if marker_conv_id:
            from ._conversation_io import conversation_path as _conv_path_for2
            conv_path = _conv_path_for2(
                os.path.join(project_root(), "agents", agent_id),
                marker_conv_id,
            )
        skip_continuation = False
        try:
            if os.path.exists(conv_path):
                import json as _json
                last_role = None
                with open(conv_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = _json.loads(line)
                        except Exception:
                            continue
                        role = obj.get("role")
                        if role in ("user", "assistant"):
                            last_role = role
                if last_role == "user":
                    skip_continuation = True
                    print_ts(
                        f"{COLOR_YELLOW}restart-sentinel: skipping continuation — "
                        f"unanswered user message at tail of conversation; "
                        f"next human message will wake the agent naturally{COLOR_END}",
                        agent=agent_id,
                    )
        except Exception as e:
            print_ts(
                f"{COLOR_YELLOW}restart-sentinel: couldn't check conv tail "
                f"({e}); firing continuation anyway{COLOR_END}",
                agent=agent_id,
            )

        if not skip_continuation:
            try:
                # Mark the continuation so the agent unambiguously knows
                # it came from past-self, not from the operator. Without
                # this marker, the agent reads the synthetic prompt as if
                # it's a fresh user instruction and may act on it without
                # the skepticism appropriate for self-generated text.
                # Past-self could have been wrong about state ("the fix
                # is shipped" → grep first); the marker is the prompt to
                # verify-before-acting.
                marked_continuation = (
                    "[RESTART_CONTINUATION from past-self before the restart]: "
                    + continuation
                    + "\n\n(Reminder: this prompt was written by you before the "
                    "restart fired. Treat it as past-self's instruction — verify "
                    "any factual claims it makes via tool calls before acting on "
                    "them. The operator did not send this message.)"
                )
                # Continuation turns after restart are user-facing — the user is
                # waiting for the agent to confirm it's back and what changed.
                # Without auto_post_final_text=True, the agent's reply text is
                # swallowed and only its tool calls (if any) post — resulting
                # in silent restarts even when the agent had something to say.
                #
                # Fire the continuation against the SAME conversation the
                # synthetic tool_result was just appended to (marker_conv_id),
                # not a bare int(channel_id). A bare int makes run_synthetic_turn
                # re-expand the target to "{transport}:{channel_id}", which on
                # any non-Discord transport differs from marker_conv_id — so the
                # continuation would run against a DIFFERENT conversation than the
                # one that got the tool_result, re-orphaning the tool_use (the
                # exact 400 this injection exists to prevent). Build a Session
                # whose conversation_id == marker_conv_id (mirrors cron's
                # _build_session shape), threading _orig_handle so handle-based
                # ACLs still resolve. Fall back to the bare int only when
                # marker_conv_id is empty/unparseable.
                _cont_target: object = int(channel_id)
                if marker_conv_id:
                    from .session import Session as _Session
                    _t_name, _sep, _tid = marker_conv_id.partition(":")
                    _t_name, _tid = _t_name.strip(), _tid.strip()
                    if _sep and _t_name and _tid:
                        _cont_target = _Session(
                            transport=_t_name,
                            transport_id=_tid,
                            conversation_id=marker_conv_id,
                            speaker_id=0,
                            speaker_role_ids=[],
                            is_owner=False,
                            is_dm=True,
                            display_name=_orig_handle or f"synthetic:{_tid}",
                            handle=_orig_handle or "",
                        )
                await runner.run_synthetic_turn(
                    _cont_target,
                    marked_continuation,
                    speaker_handle=_orig_handle,
                    auto_post_final_text=True,
                    # Tag the chain root: restart continuations DO post to
                    # the operator's channel (auto_post_final_text=True
                    # above). Tagged operator_channel so an empty/dead
                    # chain inside the continuation surfaces a failure
                    # to the operator instead of going silent.
                    originator_visibility="operator_channel",
                )
            except Exception as e:
                print_ts(f"{COLOR_RED}restart-sentinel: continuation failed: {e}{COLOR_END}", error=True, agent=agent_id)

    # Always delete on success path.
    try:
        os.remove(path)
    except OSError as e:
        print_ts(f"{COLOR_YELLOW}restart-sentinel: couldn't remove {path}: {e}{COLOR_END}")


async def process_pending() -> None:
    """Scan the sentinel dir and process each file. Safe to call once at startup."""
    d = _dir()
    if not os.path.isdir(d):
        return
    try:
        files = sorted(
            os.path.join(d, f) for f in os.listdir(d)
            if f.endswith(".json") and not f.endswith(".bad")
            # `.tool_result.json` files are continuation MARKERS, not signed
            # sentinels — they're read via marker_path derivation inside
            # _process_one, never on their own. Globbing them here made the
            # processor treat each as a sentinel, fail HMAC ("forged or
            # stale"), and rename it .forged — scary log noise that could also
            # clobber the marker before its real sibling sentinel consumed it.
            and not f.endswith(".tool_result.json")
        )
    except OSError as e:
        print_ts(f"{COLOR_RED}restart-sentinel: failed to list {d}: {e}{COLOR_END}", error=True)
        return
    if not files:
        return
    print_ts(f"{COLOR_GREEN}restart-sentinel: processing {len(files)} pending sentinel(s){COLOR_END}")
    for path in files:
        try:
            await _process_one(path)
        except Exception as e:
            print_ts(f"{COLOR_RED}restart-sentinel: error processing {path}: {e}{COLOR_END}", error=True)
