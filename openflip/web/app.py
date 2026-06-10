"""Quart app entry: routes + websocket + Quart-Auth wiring.

Run with:
    ~/.openflip/.lvenv/bin/python -m openflip_web

or for hot-reload during dev:
    hypercorn --reload openflip_web.app:app -b 0.0.0.0:8765
"""
from __future__ import annotations

import asyncio
import fnmatch
import hmac
import json
import os
import re
import time
from pathlib import Path

from quart import (
    Quart, render_template, request, redirect, url_for,
    websocket, jsonify, abort, flash,
)
from quart_auth import (
    AuthUser, QuartAuth, login_required, login_user, logout_user,
    current_user,
)

from . import auth as _auth
from . import openflip_data as _data
from . import interagent as _interagent
from . import watcher as _watcher
from .csrf import csrf_protect, get_or_create_csrf_token, clear_csrf_token
from .config import (
    BIND_HOST, BIND_PORT, SESSION_COOKIE, TRIGGER_TOKEN_FILE,
)
from ..utils import print_ts, COLOR_RED, COLOR_GREEN, COLOR_YELLOW, COLOR_END
from .._constants import DANGEROUS_TOOL_NAMES


app = Quart(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = _auth.get_or_create_secret_key()
app.config["QUART_AUTH_COOKIE_NAME"] = SESSION_COOKIE
app.config["QUART_AUTH_COOKIE_SECURE"] = False  # LAN-local; HTTP fine
app.config["QUART_AUTH_DURATION"] = 60 * 60 * 24 * 30  # 30 days
# Template edits should appear on next request — webapp runs in-process with
# the agent runtime so we can't easily use hypercorn --reload. Jinja's
# auto-reload checks file mtimes on every template lookup; the perf cost is
# trivial for the handful of templates we have, and the alternative is
# "template fix shipped to disk but invisible until next openflip restart"
# which already cost a debugging cycle once (2026-05-24).
app.config["TEMPLATES_AUTO_RELOAD"] = True
# Cap request body size so an authed POST can't buffer unbounded bytes into
# memory. 8 MB is generous for agent.json edits / config saves (the largest
# legitimate bodies); anything bigger is rejected with 413 before buffering.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

QuartAuth(app)


# Mint the inbound /trigger bearer token ONCE at import/startup if absent, and
# log its location so the operator can hand it to their poller. The request
# path NEVER creates it (see auth.read_trigger_token / _trigger_bearer_ok) —
# minting a credential is a startup-only privileged action. A get-OR-create on
# the request path was a flaw in the declined branch: deleting the file
# mid-flight silently rotated the secret out from under live callers.
try:
    _tok = _auth.ensure_trigger_token()
    if _tok:
        print_ts(
            f"{COLOR_GREEN}trigger: inbound API bearer token ready at "
            f"{TRIGGER_TOKEN_FILE} (mode 0600){COLOR_END}"
        )
    else:
        print_ts(
            f"{COLOR_YELLOW}trigger: could not read/create bearer token at "
            f"{TRIGGER_TOKEN_FILE}; /trigger will fail closed{COLOR_END}"
        )
except Exception as _tok_err:  # never let token setup block webapp boot
    print_ts(
        f"{COLOR_RED}trigger: token setup failed ({_tok_err}); "
        f"/trigger will fail closed{COLOR_END}", error=True
    )


# ---------- jinja helpers ----------

@app.template_filter("relative_time")
def _jinja_relative(ts):
    return _data.fmt_relative(float(ts))


@app.template_filter("fmt_ts")
def _jinja_ts(ts):
    return _data.fmt_ts(float(ts))


@app.template_filter("tojson_pretty")
def _jinja_pretty(value):
    return json.dumps(value, indent=2)


# Expose CSRF token to every template under `csrf_token`. Templates
# render it as a hidden form field on POST forms, or read it via JS
# for fetch/XHR calls and send as X-CSRF-Token header.
@app.context_processor
def _inject_csrf():
    return {"csrf_token": get_or_create_csrf_token}


# ---------- auth routes ----------

@app.route("/setup", methods=["GET", "POST"])
async def setup():
    """First-boot password setup. Locked after a credential exists."""
    if _auth.is_configured():
        return redirect(url_for("login"))
    if request.method == "POST":
        form = await request.form
        u = (form.get("username") or "").strip()
        p = form.get("password") or ""
        p2 = form.get("password_confirm") or ""
        err = None
        if not u or not p:
            err = "Username and password required."
        elif p != p2:
            err = "Passwords don't match."
        elif len(p) < 8:
            err = "Password must be at least 8 characters."
        if err:
            return await render_template("setup.html", error=err, username=u)
        _auth.set_credentials(u, p)
        return redirect(url_for("login"))
    return await render_template("setup.html")


@app.route("/login", methods=["GET", "POST"])
async def login():
    if not _auth.is_configured():
        return redirect(url_for("setup"))
    if request.method == "POST":
        client_ip = request.remote_addr or "unknown"
        if _login_rate_limited(client_ip):
            return await render_template(
                "login.html",
                error="Too many failed attempts. Wait a few minutes and try again.",
            ), 429
        form = await request.form
        u = (form.get("username") or "").strip()
        p = form.get("password") or ""
        if _auth.verify(u, p):
            login_user(AuthUser(u), remember=True)
            return redirect(url_for("agents_list"))
        _login_record_fail(client_ip)
        return await render_template("login.html", error="Invalid credentials.", username=u)
    return await render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
@csrf_protect
async def logout():
    logout_user()
    clear_csrf_token()
    return redirect(url_for("login"))


# ---------- agents list ----------

@app.route("/")
@login_required
async def agents_list():
    agents = await asyncio.to_thread(_data.list_agents)
    return await render_template("agents.html", agents=agents)


# ---------- agent detail ----------

@app.route("/agents/<agent_id>")
@login_required
async def agent_detail(agent_id):
    agent = await asyncio.to_thread(_data.get_agent, agent_id)
    if not agent:
        abort(404)
    return await render_template("agent.html", agent=agent)


@app.route("/agents/<agent_id>/config", methods=["POST"])
@login_required
@csrf_protect
async def agent_config_save(agent_id):
    body = await request.get_json()
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "Expected JSON object"}), 400
    verr = await asyncio.to_thread(_data.validate_agent_config, agent_id, body)
    if verr is not None:
        return jsonify({"ok": False, "error": verr}), 400
    ok = await asyncio.to_thread(_data.write_agent_config, agent_id, body)
    return jsonify({"ok": ok})


@app.route("/agents/<agent_id>/enabled", methods=["POST"])
@login_required
@csrf_protect
async def agent_enabled_toggle(agent_id):
    """Enable/disable an agent. Persists to agent_state.json AND applies
    live: stops the runner if disabling, starts a fresh one if enabling.
    """
    body = await request.get_json()
    if not isinstance(body, dict) or "enabled" not in body:
        return jsonify({"ok": False, "error": "Expected {enabled: bool}"}), 400
    enabled = bool(body.get("enabled"))

    # Validate agent exists.
    from ..registry import ALL_AGENTS
    agent = ALL_AGENTS.get(agent_id)
    if agent is None:
        return jsonify({"ok": False, "error": f"Unknown agent '{agent_id}'"}), 404

    # Persist state first so a partial failure on the runtime side at least
    # leaves the desired state on disk for the next process boot.
    from ..persistence import set_enabled
    persisted = await asyncio.to_thread(set_enabled, agent_id, enabled)
    if not persisted:
        return jsonify({"ok": False, "error": "Failed to persist state"}), 500

    # Apply live. Errors here are surfaced but the persisted state stands.
    from ..main import start_runner, stop_runner
    try:
        if enabled:
            await start_runner(agent)
        else:
            await stop_runner(agent_id)
    except Exception as e:
        return jsonify({"ok": True, "warning": f"Persisted but live apply failed: {e}"})

    return jsonify({"ok": True, "enabled": enabled})


@app.route("/agents/<agent_id>/files/<path:file_name>", methods=["GET", "POST"])
@login_required
async def system_file(agent_id, file_name):
    if request.method == "GET":
        text = await asyncio.to_thread(_data.read_system_file, agent_id, file_name)
        if text is None:
            abort(404)
        return jsonify({"content": text})
    # POST: state-changing write. CSRF check inline since the route is
    # GET+POST and @csrf_protect at the decorator level would also block
    # GETs. Same logic as the decorator — read submitted token from
    # form or X-CSRF-Token header and constant-time compare.
    import secrets as _secrets
    from quart import session as _session
    expected = _session.get("_csrf_token") or ""
    submitted = ""
    try:
        form = await request.form
        submitted = form.get("csrf_token", "") or ""
    except Exception:
        submitted = ""
    if not submitted:
        submitted = request.headers.get("X-CSRF-Token", "") or ""
    if not expected or not submitted or not _secrets.compare_digest(submitted, expected):
        abort(403, "CSRF token missing or invalid.")
    body = await request.get_json()
    content = (body or {}).get("content", "")
    ok = await asyncio.to_thread(_data.write_system_file, agent_id, file_name, content)
    return jsonify({"ok": ok})


# ---------- conversations ----------

@app.route("/agents/<agent_id>/conversations/<channel_id>")
@login_required
async def conversation_view(agent_id, channel_id):
    agent = await asyncio.to_thread(_data.get_agent, agent_id)
    if not agent:
        abort(404)
    msgs = await asyncio.to_thread(_data.read_conversation, agent_id, channel_id, 200, 0)
    return await render_template(
        "conversation.html",
        agent=agent,
        channel_id=channel_id,
        messages=msgs,
    )


@app.route("/agents/<agent_id>/conversations/<channel_id>/messages.json")
@login_required
async def conversation_messages(agent_id, channel_id):
    limit = request.args.get("limit", default=200, type=int)
    offset = request.args.get("offset", default=0, type=int)
    msgs = await asyncio.to_thread(
        _data.read_conversation, agent_id, channel_id, limit, offset
    )
    return jsonify({"messages": msgs})


# ---------- inter-agent threaded view ----------

@app.route("/agents/<agent_id>/interagent")
@login_required
async def interagent_index(agent_id):
    """List inter-agent links for a focal agent — pick a peer to view."""
    agent = await asyncio.to_thread(_data.get_agent, agent_id)
    if not agent:
        abort(404)
    links = await asyncio.to_thread(_interagent.find_interagent_links, agent_id)
    return await render_template(
        "interagent_index.html", agent=agent, links=links,
    )


@app.route("/agents/<agent_id>/interagent/<peer_id>/<focal_channel>")
@login_required
async def interagent_thread(agent_id, peer_id, focal_channel):
    """Render the merged timeline between focal agent and peer."""
    focal = await asyncio.to_thread(_data.get_agent, agent_id)
    peer = await asyncio.to_thread(_data.get_agent, peer_id)
    if not focal or not peer:
        abort(404)
    peer_channel = await asyncio.to_thread(
        _interagent.find_peer_channel_for, agent_id, peer_id, focal_channel
    )
    pairs = [(agent_id, focal_channel)]
    if peer_channel:
        pairs.append((peer_id, peer_channel))
    timeline = await asyncio.to_thread(_interagent.merged_timeline, pairs)
    return await render_template(
        "interagent_thread.html",
        focal=focal, peer=peer,
        focal_channel=focal_channel,
        peer_channel=peer_channel,
        timeline=timeline,
    )


# ---------- memory inspector ----------

@app.route("/agents/<agent_id>/memory")
@login_required
async def memory_view(agent_id):
    agent = await asyncio.to_thread(_data.get_agent, agent_id)
    if not agent:
        abort(404)
    mem = await asyncio.to_thread(_data.get_memory, agent_id)
    return await render_template("memory.html", agent=agent, memory=mem)


@app.route("/agents/<agent_id>/memory/daily/<date>")
@login_required
async def memory_daily(agent_id, date):
    text = await asyncio.to_thread(_data.read_daily_log, agent_id, date)
    if text is None:
        abort(404)
    return jsonify({"content": text})


# ---------- events.jsonl tail ----------

@app.route("/events")
@login_required
async def events_view():
    return await render_template("events.html")


@app.route("/events/recent.json")
@login_required
async def events_recent():
    """Returns the last 200 events from data/events.jsonl."""
    from .config import OPENFLIP_EVENTS_JSONL
    if not OPENFLIP_EVENTS_JSONL.exists():
        return jsonify({"events": []})

    def _read():
        rows = []
        try:
            with OPENFLIP_EVENTS_JSONL.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
        return rows[-200:]

    events = await asyncio.to_thread(_read)
    return jsonify({"events": events})


# ---------- usage ledger panel (read-only over data/usage_ledger.db) ----------
#
# Owner-sensitive (cost/token data across all agents + users), so it sits
# behind the SAME @login_required gate as every other authenticated page — no
# weaker path. Pure READ side over the already-shipped usage ledger: it only
# calls usage_ledger's query functions, never the write path.

# Friendly group name → ledger column, and window → since-delta. Both MIRROR
# the /usage slash command (commands.py) EXACTLY so the web and Discord views
# agree. 'all' = everything (since 0).
_USAGE_GROUP_COLS = {"agent": "agent_id", "channel": "channel_id",
                     "session": "session_id", "user": "user_id", "model": "model"}
_USAGE_WINDOW_DELTAS = {"24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}


@app.route("/usage")
@login_required
async def usage_view():
    return await render_template("usage.html")


@app.route("/usage/data.json")
@login_required
async def usage_data():
    """Aggregated usage for a window + group_by. Returns grand totals, the
    group_by breakdown rows, and a per-day time series for charting. All three
    come from usage_ledger's read functions — no SQL in the web layer."""
    from .. import usage_ledger
    window = request.args.get("window", default="24h")
    if window not in ("24h", "7d", "30d", "all"):
        window = "24h"
    group_by = request.args.get("group_by", default="session")
    col = _USAGE_GROUP_COLS.get(group_by)
    if col is None:
        group_by, col = "session", "session_id"
    now = time.time()
    since_ts = 0.0 if window == "all" else now - _USAGE_WINDOW_DELTAS.get(window, 86400)

    def _gather():
        return (
            usage_ledger.totals(since_ts),
            usage_ledger.aggregate(since_ts, group_by=col),
            usage_ledger.daily_series(since_ts),
        )

    grand, rows, series = await asyncio.to_thread(_gather)
    return jsonify({
        "window": window,
        "group_by": group_by,
        "totals": grand,
        "rows": rows,
        "series": series,
    })


# ---------- websocket: live conversation tail ----------

@app.websocket("/ws/conversations/<agent_id>/<channel_id>")
async def ws_conversation(agent_id, channel_id):
    """Subscribe to a conversation. Pushes JSON frames as new messages
    arrive in the underlying jsonl."""
    # quart-auth's auth_required works for HTTP but not WS — manual check
    if not await current_user.is_authenticated:
        return
    q = await _watcher.subscribe(agent_id, channel_id)
    try:
        await websocket.send_json({"type": "subscribed",
                                   "agent_id": agent_id,
                                   "channel_id": channel_id})
        while True:
            frame = await q.get()
            await websocket.send_json(frame)
    except asyncio.CancelledError:
        raise
    finally:
        await _watcher.unsubscribe(agent_id, channel_id, q)


# ---------- global config ----------

@app.route("/settings")
@login_required
async def settings():
    global_cfg = await asyncio.to_thread(_data.load_global_config)
    tool_settings = await asyncio.to_thread(_data.load_tool_settings)
    return await render_template(
        "settings.html",
        global_config=global_cfg,
        tool_settings=tool_settings,
    )


@app.route("/settings/tools", methods=["POST"])
@login_required
@csrf_protect
async def settings_tools_save():
    body = await request.get_json()
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "Expected object"}), 400
    ok = await asyncio.to_thread(_data.write_tool_settings, body)
    return jsonify({"ok": ok})


# ---------- inbound trigger API (token-authed, NOT a browser session) ----------
#
# POST /trigger/<agent_id> lets an EXTERNAL script (cron, webhook, file/email
# watcher) wake a running agent with a synthetic turn — the openflip-native
# successor to OpenClaw's inbound webhook. The script does its own cheap
# checking (zero LLM cost) and only POSTs when there's real work; the agent
# then acts with its own PRE-APPROVED tools.
#
# Treat every caller as hostile. The two holes that killed the declined branch
# are closed structurally here:
#   HOLE 1 (priv-esc via caller tool_grants): the caller cannot name tools at
#     all. The grantable tools come ONLY from agent.json `trigger.allowed_tools`
#     (server-side), intersected against the agent's real tools AND a hard
#     denylist of dangerous/admin tools. The turn also runs as a non-owner,
#     non-admin synthetic speaker so the ACL admin-bypass never fires.
#   HOLE 2 (path traversal via caller session_id): the caller cannot supply a
#     session at all. The target session comes ONLY from agent.json
#     `trigger.session` (server-side), and is HARD-validated (prefix must be a
#     live transport; id is whitelist-matched) before use. conversation_path
#     is independently hardened as defense-in-depth.

# Tools that must NEVER be grantable to an unattended inbound trigger, even if
# an operator mistakenly lists them in trigger.allowed_tools. This is the
# framework-wide canonical denylist (openflip/_constants.py), shared with the
# web config editor so the two untrusted-grant paths can't drift apart.
# Intersect-and-subtract, never trust the list.
_TRIGGER_FORBIDDEN_TOOLS = DANGEROUS_TOOL_NAMES

# Synthetic speaker id for trigger turns. MUST be non-zero (so
# run_synthetic_turn does NOT fall back to owner_id) and can never equal a real
# Discord snowflake or an admin id — a negative int satisfies both. This keeps
# the turn OFF the admin-bypass path in acl._check_acl, so the ONLY tools it
# can call are the curated grants below (plus whatever the agent already
# exposes to all_users — already public, not an escalation).
_TRIGGER_SPEAKER_ID = -1

# Body-size guards — bound what a single (authenticated) caller can shove into
# a turn's context. Generous but finite.
_TRIGGER_MAX_PROMPT = 8000
_TRIGGER_MAX_CONTEXT = 8000
# Defensive cap on a caller-supplied session selector before validation. The
# id regex already bounds the id to 128 chars; this bounds the whole
# "transport:id" string so a giant body field can't reach the validator.
_TRIGGER_MAX_SESSION = 256

# Per-agent inbound rate limiter. Maps agent_id -> list of monotonic hit
# timestamps within the trailing 60s window. The declined branch had NO limit,
# so an unbounded fire-and-forget loop meant unbounded LLM spend. Single event
# loop → the prune+check+append critical section runs without an await between,
# so no lock is needed.
_TRIGGER_HITS: dict[str, list[float]] = {}
_TRIGGER_WINDOW_SECONDS = 60.0

# Login brute-force throttle. Per-client-IP sliding window; only FAILED
# attempts count, so a legitimate user typing one wrong password isn't locked
# out by their own successful retry. argon2 already slows each guess; this caps
# the attempt RATE so a 0.0.0.0-exposed deployment isn't LAN-brute-forceable.
_LOGIN_FAILS: dict[str, list[float]] = {}
_LOGIN_WINDOW_SECONDS = 300.0
_LOGIN_MAX_FAILS = 10


def _login_rate_limited(client_ip: str) -> bool:
    """True if `client_ip` has exceeded the failed-login budget in the window.

    Checks WITHOUT recording — call `_login_record_fail` only on an actual
    auth failure. No await inside → atomic on the event loop.
    """
    now = time.monotonic()
    fails = _LOGIN_FAILS.get(client_ip)
    if not fails:
        return False
    cutoff = now - _LOGIN_WINDOW_SECONDS
    fails[:] = [t for t in fails if t >= cutoff]
    return len(fails) >= _LOGIN_MAX_FAILS


def _login_record_fail(client_ip: str) -> None:
    now = time.monotonic()
    fails = _LOGIN_FAILS.get(client_ip)
    if fails is None:
        fails = []
        _LOGIN_FAILS[client_ip] = fails
    cutoff = now - _LOGIN_WINDOW_SECONDS
    fails[:] = [t for t in fails if t >= cutoff]
    fails.append(now)

# id portion of a session selector: digits/handles/slugs only — no path
# separators, no whitespace, no control chars. The ".." token is rejected
# separately. ":" is the prefix separator and is partitioned off before this.
_TRIGGER_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.+@-]{1,128}$")


def _trigger_bearer_ok(auth_header: str) -> bool:
    """Constant-time check of an `Authorization: Bearer <token>` header against
    the on-disk trigger token.

    READ-ONLY on the request path: reads the token via auth.read_trigger_token
    (which never creates it) and fails closed when it's missing/empty. Returns
    False on any malformed/missing header or empty expected token.
    """
    expected = _auth.read_trigger_token()
    if not expected:
        return False  # fail-closed: no token on disk → reject everyone
    prefix = "Bearer "
    if not auth_header or not auth_header.startswith(prefix):
        return False
    presented = auth_header[len(prefix):].strip()
    if not presented:
        return False
    return hmac.compare_digest(presented, expected)


def _trigger_rate_limited(agent_id: str, limit_per_minute: int) -> bool:
    """Record one hit for `agent_id` and return True if it exceeds the limit.

    Sliding 60s window. No await inside, so the prune/check/append is atomic
    on the single event loop.
    """
    now = time.monotonic()
    hits = _TRIGGER_HITS.get(agent_id)
    if hits is None:
        hits = []
        _TRIGGER_HITS[agent_id] = hits
    cutoff = now - _TRIGGER_WINDOW_SECONDS
    # Drop timestamps outside the window (in place).
    hits[:] = [t for t in hits if t >= cutoff]
    if len(hits) >= max(1, int(limit_per_minute)):
        return True
    hits.append(now)
    return False


def _validate_trigger_session(session_sel: str, runner) -> tuple[str, str] | None:
    """HARD-validate a server-configured session selector → (transport, id).

    Defense-in-depth even though `session_sel` comes from agent.json (operator
    config, not the HTTP caller): reject anything that could escape the
    conversations/ directory or misroute to another transport. Returns None on
    any violation; the caller turns that into a 403.

    Rules: no '/', '\\', '..', NUL/control chars anywhere; exactly one ':'
    separating a non-empty prefix from a non-empty id; the prefix must be a
    transport this runner actually listens on; the id must match the strict
    whitelist (digits-only for Discord).
    """
    sel = (session_sel or "").strip()
    if not sel or "/" in sel or "\\" in sel or ".." in sel:
        return None
    if any(ord(c) < 32 for c in sel):
        return None
    transport_name, sep, tid = sel.partition(":")
    transport_name, tid = transport_name.strip(), tid.strip()
    if not sep or not transport_name or not tid:
        return None
    live = {getattr(t, "name", "") for t in (getattr(runner, "_transports", None) or [])}
    live.discard("")
    if transport_name not in live:
        return None
    if not _TRIGGER_SESSION_ID_RE.match(tid):
        return None
    if transport_name == "discord" and not tid.isdigit():
        return None
    return transport_name, tid


# id portion of a caller-asserted identity. SAME charset as
# _TRIGGER_SESSION_ID_RE (emails already fit: digits, letters, _ . + @ -), just
# a longer cap since ids aren't session selectors. Path separators and control
# chars are excluded by the charset; ".." is rejected separately below.
_TRIGGER_ID_RE = re.compile(r"^[A-Za-z0-9_.+@-]{1,256}$")


def _validate_trigger_id(raw_id: str) -> str | None:
    """Seatbelt-validate a caller-supplied identity string → the clean id, or
    None on any violation.

    Defense-in-depth: today the id is only ever used as a dict key, but we
    validate it like a path component anyway so it can NEVER become a traversal
    primitive if a future code path ever interpolates it into a filename.
    Mirrors _validate_trigger_session's rules: no '/', '\\', '..', no control
    chars, strict charset, length-capped.
    """
    s = (raw_id or "").strip()
    if not s or "/" in s or "\\" in s or ".." in s:
        return None
    if any(ord(c) < 32 for c in s):
        return None
    if not _TRIGGER_ID_RE.match(s):
        return None
    return s


def _identify_trigger_token(auth_header: str, secrets_map: dict[str, str]) -> str | None:
    """Return the NAME of the per-hook token whose secret matches the presented
    bearer, or None if none match. Multi-token replacement for the boolean
    _trigger_bearer_ok, used only when trigger.tokens is configured.

    Compares against EVERY configured secret with hmac.compare_digest and does
    NOT early-return on the first hit — so request timing doesn't leak which
    token (or how many) are configured. Empty/missing secrets never match.
    """
    prefix = "Bearer "
    if not auth_header or not auth_header.startswith(prefix):
        return None
    presented = auth_header[len(prefix):].strip()
    if not presented:
        return None
    matched: str | None = None
    for name, secret in secrets_map.items():
        if secret and hmac.compare_digest(presented, secret):
            matched = name  # keep scanning; no early return (timing hygiene)
    return matched


def _token_may_assert_id(token_entry: dict, ident_id: str) -> bool:
    """THE SPINE. True iff a token (its agent.json binding `token_entry`) is
    permitted to assert `ident_id`. Fail-closed: unknown/empty entry → False.

    A token can assert an id if the id is in its `allowed_ids` (exact match, or
    the literal "*" wildcard meaning any id), OR matches one of its
    `allowed_id_patterns` (fnmatch glob, e.g. "*@trusted-corp.com"). This is
    what stops a leaked/replayed email-hook token from claiming a Discord
    owner's id to get owner-scoped tools.
    """
    if not isinstance(token_entry, dict):
        return False
    allowed = token_entry.get("allowed_ids")
    if isinstance(allowed, list):
        if "*" in allowed:
            return True
        if ident_id in allowed:
            return True
    patterns = token_entry.get("allowed_id_patterns")
    if isinstance(patterns, list):
        for pat in patterns:
            if isinstance(pat, str) and fnmatch.fnmatchcase(ident_id, pat):
                return True
    return False


def _resolve_identity_grants(trigger_cfg: dict, ident_id: str) -> list[str]:
    """Map a validated `ident_id` → the union of tool names from its labels.

    Called ONLY in identity-intended mode (multi-token auth identified a named
    token AND an id was asserted AND `_token_may_assert_id` passed). Because the
    caller is committed to identity scoping by the time we get here, this
    function ALWAYS fails CLOSED — it never returns the flat `allowed_tools`
    bucket:

      * []    → either (a) identity mode is intended but MISCONFIGURED —
                `identities` or `labels` is missing/empty (half-configured
                agent), or (b) the maps are present but this id has no label
                mapping (unknown sender). Both cases grant NO identity-scoped
                tools. The caller does NOT fall back to the flat bucket.
      * [...] → the de-duplicated union of tools across the id's labels.

    There is intentionally NO `None` (flat-fallback) return anymore: a
    half-configured agent must not silently hand an asserted id the anonymous
    default bucket (that was the FIX-3 bug). Flat fallback happens upstream in
    step 6 only when NO id is asserted, or in single-bearer back-compat mode —
    never through this function.

    Pure lookup. The returned list is REQUESTED tools only — the caller still
    runs it through the existing intersect-real-tools + subtract-denylist
    pipeline. This function can never grant a tool that doesn't exist or is
    denylisted; it only widens the request set, which is then narrowed.
    """
    identities = trigger_cfg.get("identities")
    labels = trigger_cfg.get("labels")
    if not isinstance(identities, dict) or not identities \
       or not isinstance(labels, dict) or not labels:
        return []  # identity mode intended but misconfigured → fail closed, no tools
    label_names = identities.get(ident_id)
    if not isinstance(label_names, list) or not label_names:
        return []  # configured, but this id is unmapped → no identity grants
    out: list[str] = []
    seen: set[str] = set()
    for ln in label_names:
        tools = labels.get(ln)
        if not isinstance(tools, list):
            continue
        for t in tools:
            if isinstance(t, str) and t.strip() and t not in seen:
                seen.add(t)
                out.append(t.strip())
    return out


@app.route("/trigger/<agent_id>", methods=["POST"])
async def trigger_agent(agent_id):
    """Wake a running agent with a synthetic turn. Token-authed (bearer), NOT a
    browser session. Fire-and-forget: schedules the turn and returns 202.

    Body (JSON): {"prompt": "...", "context"?: "...", "session"?: "...",
                  "id"?: "..."} — prompt is required. The caller MAY supply an
    opaque `id` (asserting WHO it is); it is seatbelt-validated and the
    presented token must be permitted to assert it. The caller still CANNOT name
    tools — grants come only from server-side `trigger` config (identities →
    labels → tools). See agents/_shared/MANUAL.md for the full security model.

    Errors are returned as JSON directly (401/403/400/404/429) — never via the
    global 401 handler's /login redirect, which is for browsers.
    """
    # >>> NEW — step 1: auth. Two modes, selected by whether trigger.tokens is
    #     configured. Back-compat: no trigger.tokens → exactly today's single
    #     bearer. We must read the agent's trigger config BEFORE auth to pick the
    #     mode, so steps 2/3 (agent-running, trigger-enabled) move up here.
    from ..registry import RUNNERS
    runner = RUNNERS.get(agent_id)
    if runner is None:
        return jsonify({"status": "error", "error": f"Agent '{agent_id}' not running"}), 404
    trigger_cfg = getattr(runner.agent, "trigger", None) or {}
    if not trigger_cfg.get("enabled"):
        return jsonify({"status": "error", "error": f"Agent '{agent_id}' has no enabled trigger config"}), 403

    auth_header = request.headers.get("Authorization", "")
    tokens_cfg = trigger_cfg.get("tokens") if isinstance(trigger_cfg.get("tokens"), dict) else {}
    token_name: str | None = None
    if tokens_cfg:
        # MULTI-TOKEN MODE: identify which named token was presented. Only
        # tokens that are BOTH in agent.json (binding) AND in the secret store
        # (secret) are usable — intersect the two so a binding without a secret
        # (or vice-versa) can't authenticate.
        secrets_map = _auth.read_trigger_token_secrets(agent_id)
        usable = {n: s for n, s in secrets_map.items() if n in tokens_cfg}
        token_name = _identify_trigger_token(auth_header, usable)
        if token_name is None:
            return jsonify({"status": "error", "error": "Missing or invalid bearer token"}), 401
    else:
        # BACK-COMPAT: single shared bearer, byte-identical to today.
        if not _trigger_bearer_ok(auth_header):
            return jsonify({"status": "error", "error": "Missing or invalid bearer token"}), 401
    # <<< NEW

    # 4) Body + required prompt.
    body = await request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"status": "error", "error": "Expected JSON object body"}), 400
    _raw_prompt = body.get("prompt")
    if _raw_prompt is not None and not isinstance(_raw_prompt, str):
        return jsonify({"status": "error", "error": "'prompt' must be a string"}), 400
    prompt = (_raw_prompt or "").strip()
    if not prompt:
        return jsonify({"status": "error", "error": "Missing required field 'prompt'"}), 400
    if len(prompt) > _TRIGGER_MAX_PROMPT:
        return jsonify({"status": "error", "error": f"'prompt' exceeds {_TRIGGER_MAX_PROMPT} chars"}), 400
    context = (body.get("context") or "")
    if not isinstance(context, str):
        return jsonify({"status": "error", "error": "'context' must be a string"}), 400
    context = context.strip()
    if len(context) > _TRIGGER_MAX_CONTEXT:
        return jsonify({"status": "error", "error": f"'context' exceeds {_TRIGGER_MAX_CONTEXT} chars"}), 400

    # >>> NEW — step 4b: caller-asserted identity. MULTI-TOKEN MODE ONLY.
    #     In single-bearer back-compat mode (token_name is None) the `id` field
    #     is IGNORED ENTIRELY — not read, not type-checked, not validated — so
    #     that path stays byte-identical to today (a body carrying a stray `id`,
    #     even a malformed one, still 202s exactly as before). We only read /
    #     validate / bind `id` once multi-token auth has identified a named
    #     token, because only then is there a per-token binding to enforce.
    ident_id: str | None = None
    if token_name is not None:
        _raw_id = body.get("id")
        if _raw_id is not None and not isinstance(_raw_id, str):
            return jsonify({"status": "error", "error": "'id' must be a string"}), 400
        if isinstance(_raw_id, str) and _raw_id.strip():
            ident_id = _validate_trigger_id(_raw_id)
            if ident_id is None:
                return jsonify({"status": "error", "error": "invalid id"}), 400
            # THE SPINE: the presented token must be permitted to assert this id.
            if not _token_may_assert_id(tokens_cfg.get(token_name) or {}, ident_id):
                print_ts(
                    f"{COLOR_YELLOW}trigger: agent '{agent_id}' token '{token_name}' "
                    f"may not assert id '{ident_id}' — 403{COLOR_END}", agent=agent_id,
                )
                return jsonify({"status": "error", "error": "token not permitted to assert this id"}), 403
    # <<< NEW

    # 5) Resolve the target session. Precedence: caller-supplied `session` (if a
    #    non-empty string) wins, else fall back to SERVER-SIDE config. EITHER
    #    way the selector goes through the SAME hard validation
    #    (_validate_trigger_session) before it touches the filesystem — that is
    #    what keeps dynamic session selection from reopening HOLE 2 (path
    #    traversal). The caller's string is NEVER trusted raw. Tools remain
    #    server-side only (HOLE 1 stays closed; see step 6).
    _raw_session = body.get("session")
    if _raw_session is not None and not isinstance(_raw_session, str):
        return jsonify({"status": "error", "error": "'session' must be a string"}), 400
    caller_session = (_raw_session or "").strip()
    if caller_session:
        if len(caller_session) > _TRIGGER_MAX_SESSION:
            return jsonify({"status": "error", "error": f"'session' exceeds {_TRIGGER_MAX_SESSION} chars"}), 400
        resolved = _validate_trigger_session(caller_session, runner)
        if resolved is None:
            return jsonify({"status": "error", "error": "invalid session"}), 400
    else:
        resolved = _validate_trigger_session(trigger_cfg.get("session", ""), runner)
        if resolved is None:
            return jsonify({
                "status": "error",
                "error": f"Agent '{agent_id}' trigger.session is missing or invalid",
            }), 403
    transport_name, tid = resolved

    # >>> NEW — step 6: compute curated tool grants. The `requested` set comes
    #     from identity resolution when multi-token auth identified a token AND
    #     an id was asserted (and the token was permitted to assert it, checked
    #     in step 4b); otherwise from the flat allowed_tools bucket (back-compat
    #     single-bearer, OR multi-token with no id asserted).
    #     EVERYTHING AFTER `requested` IS THE ORIGINAL PIPELINE, UNCHANGED:
    #     intersect real tools, subtract the denylist. HOLE 1 stays closed —
    #     the caller still contributed only an id, never a tool name.
    configured_tool_names = {acl.name for acl in runner.agent.allowed_tools}
    requested = None
    if token_name is not None and ident_id is not None:
        # Identity mode intended. _resolve_identity_grants ALWAYS fails closed
        # here: it returns [] when identities/labels are missing/half-configured
        # (misconfigured) OR the id is unmapped, and [...] when mapped. It NEVER
        # returns the flat bucket — so a half-configured agent can't silently
        # leak the anonymous default to an asserted id (FIX 3). Because it
        # returns [] (not None), the flat-fallback branch below does NOT fire.
        requested = _resolve_identity_grants(trigger_cfg, ident_id)
        if not requested:
            print_ts(
                f"{COLOR_YELLOW}trigger: agent '{agent_id}' id '{ident_id}' "
                f"resolved to NO tools (unmapped sender, or identities/labels "
                f"missing/half-configured) — failing closed, no flat fallback"
                f"{COLOR_END}", agent=agent_id,
            )
    if requested is None:
        # Flat fallback: back-compat single-bearer, OR multi-token with no id
        # asserted. This is the agent's default/anonymous bucket. NOT reached
        # when an id was asserted under a token (that path is [] or [...] above).
        requested = trigger_cfg.get("allowed_tools") or []
    grants = [
        t for t in requested
        if t in configured_tool_names and t not in _TRIGGER_FORBIDDEN_TOOLS
    ]
    dropped = [t for t in requested if t not in grants]
    if dropped:
        print_ts(
            f"{COLOR_YELLOW}trigger: agent '{agent_id}' dropped non-grantable "
            f"tools {dropped} (unknown or denylisted){COLOR_END}", agent=agent_id,
        )
    # <<< NEW

    # 7) Rate limit (per-agent). 429 over the configured limit.
    rate = int(trigger_cfg.get("rate_limit_per_minute", 6) or 6)
    if _trigger_rate_limited(agent_id, rate):
        print_ts(
            f"{COLOR_YELLOW}trigger: agent '{agent_id}' rate-limited "
            f"({rate}/min){COLOR_END}", agent=agent_id,
        )
        return jsonify({
            "status": "error",
            "error": f"Rate limit exceeded ({rate}/min)",
        }), 429

    # 8) Build the synthetic Session (server-controlled transport+id+grants) and
    #    fire the turn fire-and-forget. The turn runs as a NON-owner, non-admin
    #    speaker so only the curated grants (plus already-public tools) apply.
    from .. import cron as _cron
    session_target = _cron._build_session(transport_name, tid, grants)
    full_prompt = f"{prompt}\n\n{context}" if context else prompt

    async def _run():
        try:
            await runner.run_synthetic_turn(
                session_target,
                full_prompt,
                speaker_id=_TRIGGER_SPEAKER_ID,
                auto_post_final_text=False,
                silent=True,
                # "trigger" tag: dead/empty chains log loudly but do NOT nag the
                # operator channel (only "operator_channel"/"" force-surface).
                originator_visibility="trigger",
            )
        except Exception as e:
            print_ts(
                f"{COLOR_RED}trigger: agent '{agent_id}' turn crashed: {e}{COLOR_END}",
                error=True, agent=agent_id,
            )

    asyncio.create_task(_run())
    print_ts(
        f"{COLOR_GREEN}trigger: accepted → agent={agent_id} "
        f"token={token_name or 'single'} id={ident_id or '-'} "
        f"session={transport_name}:{tid} grants={grants}{COLOR_END}", agent=agent_id,
    )
    return jsonify({"status": "accepted", "agent": agent_id}), 202


# ---------- root redirect for unauthed ----------

@app.errorhandler(401)
async def unauthorized(_e):
    return redirect(url_for("login"))


def run():
    """Blocking entry point for `python -m openflip.web`. Spins up a new
    event loop. Not used in production — the webapp now mounts INTO
    openflip's main event loop via start_async() below."""
    import hypercorn.asyncio
    import hypercorn.config
    cfg = hypercorn.config.Config()
    cfg.bind = [f"{BIND_HOST}:{BIND_PORT}"]
    cfg.accesslog = "-"
    asyncio.run(hypercorn.asyncio.serve(app, cfg))


async def start_async(host: str | None = None, port: int | None = None, shutdown_trigger=None):
    """Non-blocking entry point. Call from inside an existing event loop
    via `asyncio.create_task(start_async())`. Used by openflip.main to
    mount the webapp alongside the agent runners and cron scheduler in
    the same process — direct access to RUNNERS, tool_settings live
    state, conversation in-memory caches, etc.

    shutdown_trigger: an awaitable returned by a coroutine that completes
    when the process should shut down. CRITICAL: when this is None,
    hypercorn installs its own SIGTERM/SIGINT signal handlers, which
    OVERWRITES any handler set by the parent process (last-write-wins
    semantics of loop.add_signal_handler). openflip.main passes its
    shutdown_event.wait as the trigger so its own SIGTERM handler stays
    in control of the shutdown sequence.
    """
    import hypercorn.asyncio
    import hypercorn.config
    cfg = hypercorn.config.Config()
    cfg.bind = [f"{host or BIND_HOST}:{port or BIND_PORT}"]
    cfg.accesslog = "-"
    await hypercorn.asyncio.serve(app, cfg, shutdown_trigger=shutdown_trigger)


if __name__ == "__main__":
    run()
