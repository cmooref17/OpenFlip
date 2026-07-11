"""Direct Anthropic API conversation wrapper.

Uses Claude Code's OAuth tokens from `~/.claude/.credentials.json` to
authenticate directly against `https://api.anthropic.com/v1/messages`.
Requests route through the owner's Claude Code subscription (verified
2026-05-11 with extra usage disabled).

Mirrors the interface of `DiscordConversation` so `runtime.py` can swap
between providers based on `agent.provider`.

Reference: see your agent's notes/ directory for an OAuth path writeup if you maintain one.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Callable, Optional

import aiohttp

from .agent import Agent
from .utils import print_ts, COLOR_YELLOW, COLOR_RED, COLOR_END, load_json, save_json
from . import _conversation_io as _cio
from . import _request_validator
from .config_global import get_compaction_trigger, get_effort, get_max_tokens, get_model_context_window, _VALID_EFFORT_LEVELS


# Compaction trigger (input_tokens) sent on a MANUAL /compact. Anthropic's
# compact_20260112 edit requires trigger >= 50k (see config_global
# .get_compaction_trigger, which floors every trigger at this value), so 50k is
# the lowest value that won't 400 — and it makes Anthropic compact whenever the
# conversation is at least ~50k tokens, regardless of the per-model auto-trigger
# (which can be as high as 980k on 1m agents). Anthropic cannot compact a
# conversation whose input is below this floor; manual /compact on a tiny
# conversation is therefore a genuine no-op on the server side, not a bug here.
_MANUAL_COMPACT_TRIGGER = 50_000


class MalformedRequestError(Exception):
    """Raised by chat_stream when the assembled request body
    fails pre-flight validation. Carries the list of
    `RequestValidationProblem` so runtime.py can surface a clear
    user-visible message instead of the cryptic Anthropic 400 string."""
    def __init__(self, problems:list):
        self.problems = problems
        detail = "; ".join(str(p) for p in problems) or "no detail"
        super().__init__(f"malformed Anthropic request: {detail}")


# ── Compat types ──

class ChatMessage(dict):
    def __init__(self, role: str = None, content: str = None, message_dict: dict = None):
        if message_dict:
            super().__init__(message_dict)
        else:
            super().__init__()
            self["role"] = role
            self["content"] = content or ""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class AnthropicToolCall:
    def __init__(self, function_name: str, args: dict, tool_use_id: str = "",
                 function: Optional[Callable] = None):
        self._function_name = function_name
        self._args = args
        self.tool_use_id = tool_use_id
        self.function = function

    @property
    def function_name(self) -> str:
        return self._function_name

    @property
    def args(self) -> dict:
        return self._args

    @property
    def is_async(self) -> bool:
        return asyncio.iscoroutinefunction(self.function) if callable(self.function) else False

    async def invoke(self) -> Any:
        if not self.function:
            raise ValueError(f"No function found for tool call: {self._function_name}")
        result = self.function(**self._args)
        if asyncio.iscoroutine(result):
            return await result
        return result


class AnthropicAIChatMessage(ChatMessage):
    def __init__(self, content: str, tool_calls: Optional[list] = None, is_framework_error: bool = False):
        super().__init__(role="assistant", content=content)
        self.tool_calls: list[AnthropicToolCall] = tool_calls or []
        self.raw_response = None
        self.content_text = content
        self.thinking: str | None = None
        # Framework-generated error strings (401/429/400/timeout/etc) get
        # this flag so the runtime can skip appending them to conv.messages.
        # Without this, error strings pollute history as "assistant replies"
        # and the model later echoes them when reading its own history.
        self.is_framework_error: bool = is_framework_error


# Claude Code stores OAuth creds in this file on Linux AND Windows
# (%USERPROFILE%\.claude\.credentials.json — confirmed by the official
# authentication docs); macOS uses Keychain (read below). When
# CLAUDE_CONFIG_DIR is set, Claude Code relocates the file under it —
# honor that so we read the same creds the CLI refreshes.
_CREDS_PATH = os.path.join(
    os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser(os.path.join("~", ".claude")),
    ".credentials.json",
)
_KEYCHAIN_SERVICE = "Claude Code-credentials"
# On macOS, Claude Code stores credentials with account = $USER (the macOS
# username), NOT the literal string "Claude Code". An older Claude Code
# version wrote a stale entry under account="Claude Code" that's still in
# Keychain on some Macs — reading from there returns the original token
# from first install and never updates. Reading from $USER gets the live
# entry that Claude Code refreshes.
_KEYCHAIN_ACCOUNT = os.environ.get("USER") or "Claude Code"
_DEFAULT_API_BASE = "https://api.anthropic.com"
_DEFAULT_USER_AGENT = "claude-code/2.1.153"
_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_CLAUDE_CODE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_OAUTH_REFRESH_URL = "https://platform.claude.com/v1/oauth/token"
# 5 min eager refresh leeway — matches Claude Code's `mB()` function in cli.js
# (verified 2026-05-26 by reading actual source via claude_code subprocess).
# Refresh proactively before token actually expires so we don't race a 401
# retry. With the asyncio.Lock dedup in _load_oauth_access_token, multiple
# agents hitting the leeway window simultaneously coalesce into one refresh.
_REFRESH_LEEWAY_MS = 300_000

# Module-level coordination state for refresh — see _load_oauth_access_token()
# and _do_refresh_locked() below. Prevents the 429-storm when multiple agents
# (an agent + another agent + the maintainer agent + ...) all try to refresh the same token in the
# same second. Mirrors Claude Code's `refreshTokenPromises` map (one in-flight
# refresh per refresh_token; subsequent callers await the same Future).
#
# Also tracks last-429-ts so we don't hammer the endpoint while it's cooling
# down — when a refresh fails with HTTP 429, skip new refresh attempts for
# `_REFRESH_BACKOFF_AFTER_429_S` seconds and just surface the failure.
_REFRESH_LOCK: asyncio.Lock | None = None  # lazy-init on first call (event loop must exist)
_REFRESH_INFLIGHT: dict | None = None  # {"refresh_token": str, "future": asyncio.Future}
_REFRESH_LAST_429_TS: float = 0.0
_REFRESH_BACKOFF_AFTER_429_S = 60.0

# Recent-success short-circuit. If the creds file on disk was written within
# this many seconds, treat it as fresh: any caller (even one with
# force_refresh=True) returns the disk token instead of firing another network
# refresh. Catches the thundering-herd case where the first agent's refresh
# already succeeded by the time the 2nd/3rd agents reach the refresh path —
# and the case where the server returns the same access_token (so the
# fresh_access != old_access check below misses it).
_RECENT_REFRESH_SHORTCIRCUIT_S = 5.0


def _disk_creds_age_s() -> float:
    """Seconds since the OAuth creds file was last modified. Infinity if absent."""
    try:
        return time.time() - os.path.getmtime(_CREDS_PATH)
    except OSError:
        return float("inf")

# Cross-process file lock for refresh coordination — mirrors Claude Code's
# `hA6` function which acquires `.oauth_refresh.lock` via `fcntl.flock` before
# touching creds. Without this, openflip and the `claude` CLI itself (or two
# openflip processes) refresh the SAME credentials concurrently and one of
# them gets 429'd by Anthropic's OAuth endpoint, which then triggers our
# 60s cooldown and the user sees "OAuth token unavailable" for a minute.
# The asyncio.Lock above only coordinates within a single process.
#
# Lock file sits next to the creds file (~/.claude/.oauth_refresh.lock).
# Stale window 10s — if a process crashes mid-refresh and leaves the lock
# held, the next caller will wait at most 10s before forcibly breaking it.
# Retry up to 5 times with 1000ms + jitter(0..1000ms) sleep between attempts.
_REFRESH_LOCK_PATH = os.path.join(os.path.dirname(_CREDS_PATH), ".oauth_refresh.lock")
_REFRESH_LOCK_STALE_S = 10.0
_REFRESH_LOCK_MAX_TRIES = 5
_REFRESH_LOCK_BASE_SLEEP_S = 1.0


def _acquire_refresh_file_lock():
    """Acquire a cross-process exclusive lock on .oauth_refresh.lock.

    Returns the open file descriptor on success (caller must close it to
    release). Returns None if the lock couldn't be acquired within
    _REFRESH_LOCK_MAX_TRIES attempts.

    Uses fcntl.flock on POSIX (Linux + macOS) and msvcrt.locking on
    Windows, via the shared _file_lock helper. On a platform with neither
    (shouldn't exist), falls through (returns None) and we rely solely on
    the in-process asyncio.Lock + the per-refresh-token dedup.

    Stale detection: if the lock file exists AND its mtime is older than
    _REFRESH_LOCK_STALE_S, the holder probably crashed — we forcibly
    truncate/recreate it. Mirrors Claude Code's hA6 stale handling.
    """
    from . import _file_lock
    if not _file_lock.LOCKING_SUPPORTED:
        return None

    import random

    for attempt in range(_REFRESH_LOCK_MAX_TRIES):
        # Stale check before each attempt — if the lock file is old enough
        # that the holder must have crashed, blow it away so we can acquire.
        try:
            mtime = os.path.getmtime(_REFRESH_LOCK_PATH)
            if (time.time() - mtime) > _REFRESH_LOCK_STALE_S:
                try:
                    os.remove(_REFRESH_LOCK_PATH)
                    print_ts(
                        f"{COLOR_YELLOW}OAuth refresh: removed stale lock file "
                        f"(mtime {time.time() - mtime:.1f}s old){COLOR_END}",
                    )
                except OSError:
                    pass  # someone else might have removed it concurrently
        except OSError:
            pass  # file doesn't exist, normal case

        try:
            # O_CREAT | O_EXCL would race; just open RW with create and
            # rely on flock to serialize. flock is per-fd so the lock is
            # released when we close the fd (or process dies).
            fd = os.open(_REFRESH_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as e:
            print_ts(
                f"{COLOR_YELLOW}OAuth refresh: lock file open failed: {e}{COLOR_END}",
            )
            return None

        try:
            acquired = _file_lock.try_lock_excl(fd)
        except OSError as e:
            try:
                os.close(fd)
            except OSError:
                pass
            print_ts(
                f"{COLOR_YELLOW}OAuth refresh: file lock failed unexpectedly: {e}{COLOR_END}",
            )
            return None
        if acquired:
            # Got it. Touch the mtime so future stale checks know we're alive.
            try:
                os.utime(_REFRESH_LOCK_PATH, None)
            except OSError:
                pass
            return fd
        else:
            # Someone else holds it; sleep and retry.
            try:
                os.close(fd)
            except OSError:
                pass
            sleep_s = _REFRESH_LOCK_BASE_SLEEP_S + random.random()
            print_ts(
                f"OAuth refresh: cross-process lock held (attempt {attempt+1}/"
                f"{_REFRESH_LOCK_MAX_TRIES}, sleeping {sleep_s:.2f}s)"
            )
            time.sleep(sleep_s)
    print_ts(
        f"{COLOR_YELLOW}OAuth refresh: failed to acquire cross-process lock "
        f"after {_REFRESH_LOCK_MAX_TRIES} attempts{COLOR_END}",
    )
    return None


def _release_refresh_file_lock(fd) -> None:
    """Unlock + close the lock fd — the lock is auto-released on close
    (or process death) on every platform anyway; explicit unlock is tidy."""
    if fd is None:
        return
    from . import _file_lock
    _file_lock.unlock(fd)
    try:
        os.close(fd)
    except OSError:
        pass


def _refresh_with_file_lock(creds: dict) -> dict | None:
    """Acquire cross-process flock, RE-READ creds from disk (another
    process may have just refreshed), check if refresh is still needed,
    and only then hit the network. Releases lock in finally.

    This mirrors Claude Code's `hA6` function: lock → re-read → check →
    refresh → save → unlock. Without the re-read, we'd refresh
    unnecessarily after the lock-holder ahead of us already did the work.
    """
    fd = _acquire_refresh_file_lock()
    # If lock acquisition failed (Windows / fs issue / contention), fall
    # back to firing the refresh anyway — we tried our best at coordination
    # and the asyncio.Lock in the caller still prevents intra-process races.
    try:
        # Re-read creds from disk now that we hold the lock — another
        # process (claude CLI, second openflip, etc) may have just done
        # the refresh and written fresh creds. If so, return them without
        # making another network call.
        if fd is not None:
            fresh_creds = _load_oauth_creds()
            if fresh_creds:
                fresh_oauth = fresh_creds.get("claudeAiOauth") or {}
                fresh_expires = fresh_oauth.get("expiresAt", 0)
                fresh_access = fresh_oauth.get("accessToken")
                # Recent-write short-circuit: if disk was written within
                # _RECENT_REFRESH_SHORTCIRCUIT_S seconds, another caller
                # JUST did a successful refresh. Trust that work and use
                # disk creds — even if access_token didn't change (server
                # can return the same token) and even if our expires-at
                # check would otherwise fail. Without this, the 2nd/3rd
                # agents in a thundering herd fall through and fire
                # redundant refresh calls that earn 429s.
                disk_age = _disk_creds_age_s()
                if fresh_access and disk_age < _RECENT_REFRESH_SHORTCIRCUIT_S:
                    print_ts(
                        f"{COLOR_YELLOW}OAuth refresh: disk creds written "
                        f"{disk_age:.1f}s ago — skipping network call{COLOR_END}",
                    )
                    return fresh_creds
                # If disk creds are now valid (with our leeway window) AND
                # the access token changed vs what we entered with, the
                # other process did our work for us.
                old_access = (creds.get("claudeAiOauth") or {}).get("accessToken")
                now_ms = int(time.time() * 1000)
                if (fresh_access
                        and fresh_access != old_access
                        and (fresh_expires - _REFRESH_LEEWAY_MS) > now_ms):
                    print_ts(
                        f"{COLOR_YELLOW}OAuth refresh: disk creds refreshed by another "
                        f"process while we waited for lock — skipping network call{COLOR_END}",
                    )
                    return fresh_creds
                # Otherwise use the fresh-from-disk creds for the refresh
                # request so we use the latest refresh_token (it may have
                # rotated).
                creds = fresh_creds
        return _refresh_oauth_token(creds)
    finally:
        _release_refresh_file_lock(fd)


def _load_oauth_creds() -> dict | None:
    """Load Claude Code OAuth creds. On macOS, try Keychain under both the
    $USER account and the literal "Claude Code" account, in that order —
    different Claude Code installs use different conventions. Fall back to
    the file on miss. Linux/Windows read the file directly.
    """
    import sys
    if sys.platform == "darwin":
        # Try $USER first (newer Claude Code installs), then "Claude Code"
        # literal (older / some configurations). The dedupe prevents
        # running the second probe when they're equal.
        accounts: list[str] = []
        if _KEYCHAIN_ACCOUNT:
            accounts.append(_KEYCHAIN_ACCOUNT)
        if "Claude Code" not in accounts:
            accounts.append("Claude Code")
        import subprocess
        for acct in accounts:
            try:
                result = subprocess.run(
                    ["security", "find-generic-password",
                     "-s", _KEYCHAIN_SERVICE,
                     "-a", acct,
                     "-w"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return json.loads(result.stdout)
            except Exception:
                continue
    try:
        with open(_CREDS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def _save_oauth_creds(creds: dict) -> None:
    """Write refreshed creds. File for all platforms; Keychain too on darwin.

    The Keychain write-back exists because _load_oauth_creds reads Keychain
    BEFORE the file on darwin. Without writing back, a refresh updates the
    file but leaves Keychain stale — the next request reads the stale
    Keychain token, tries to refresh an already-used refresh_token, and
    Anthropic returns 400 (refresh_tokens are single-use). Loop forever.

    Linux: file-only (no Keychain read path → no write-back needed).
    """
    # File write — all platforms.
    try:
        os.makedirs(os.path.dirname(_CREDS_PATH), exist_ok=True)
        tmp_path = _CREDS_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(creds, f, indent=2)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, _CREDS_PATH)
    except Exception as e:
        print_ts(f"{COLOR_YELLOW}OAuth refresh: file persist failed ({_CREDS_PATH}): {e}{COLOR_END}", error=True)

    # Keychain write-back — darwin only. Use the same account the read
    # path tries first (_KEYCHAIN_ACCOUNT = $USER or "Claude Code"). -U
    # updates the existing item in place; -w reads the password from
    # stdin via the -w flag's value.
    import sys as _sys
    if _sys.platform == "darwin":
        try:
            import subprocess as _subprocess
            payload = json.dumps(creds)
            # security add-generic-password -U -s <service> -a <account> -w <secret>
            _subprocess.run(
                [
                    "security", "add-generic-password",
                    "-U",
                    "-s", _KEYCHAIN_SERVICE,
                    "-a", _KEYCHAIN_ACCOUNT,
                    "-w", payload,
                ],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except Exception as e:
            print_ts(
                f"{COLOR_YELLOW}OAuth refresh: keychain write-back failed "
                f"(account={_KEYCHAIN_ACCOUNT}): {e}{COLOR_END}",
                error=True,
            )


def _refresh_oauth_token(creds: dict) -> dict | None:
    """Exchange refreshToken for fresh access + refresh tokens.

    Cloudflare fronts this endpoint and rejects bare requests (error 1010)
    so we send the same User-Agent + anthropic-version openflip uses for
    the messages API. Be careful with retries — Anthropic's OAuth endpoint
    has aggressive rate-limiting and a multi-hour cooldown.
    """
    oauth = creds.get("claudeAiOauth") or {}
    refresh_token = oauth.get("refreshToken")
    if not refresh_token:
        print_ts(f"{COLOR_RED}OAuth refresh: no refreshToken in creds{COLOR_END}", error=True)
        return None
    try:
        import urllib.request
        body = json.dumps({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _CLAUDE_CODE_CLIENT_ID,
            "scope": "user:profile user:inference user:sessions:claude_code user:mcp_servers",
        }).encode("utf-8")
        # Mirror Claude Code's bundled axios request shape. Anthropic's OAuth
        # endpoint rate-limits the "claude-code/X.Y.Z" User-Agent aggressively
        # (seen after burst of failed refreshes); axios's defaults pass.
        # NOTE: deliberately omit Accept-Encoding — urllib doesn't transparently
        # decompress gzip, so requesting it would give us a body we can't
        # parse (and Anthropic would rotate the refresh token on the success
        # we couldn't read, breaking the next refresh).
        req = urllib.request.Request(
            _OAUTH_REFRESH_URL, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/plain, */*",
                "User-Agent": "axios/1.7.7",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
    except Exception as e:
        # Track 429s so the next caller backs off instead of dogpiling the
        # already-rate-limited endpoint. Don't backoff on transient network
        # errors — only when Anthropic explicitly told us to slow down.
        if "429" in str(e):
            import time
            global _REFRESH_LAST_429_TS
            _REFRESH_LAST_429_TS = time.time()
        print_ts(f"{COLOR_RED}OAuth refresh: HTTP request failed: {e}{COLOR_END}", error=True)
        return None
    new_access = data.get("access_token")
    new_refresh = data.get("refresh_token") or refresh_token
    expires_in = data.get("expires_in")
    if not new_access:
        # Log only the response KEYS, never the payload — a partial/odd
        # response can carry a refresh_token, and log.txt is append-only and
        # unrestricted. Keys alone are enough to diagnose a malformed response.
        print_ts(f"{COLOR_RED}OAuth refresh: response missing access_token; keys={sorted(data.keys()) if isinstance(data, dict) else type(data).__name__}{COLOR_END}", error=True)
        return None
    import time
    now_ms = int(time.time() * 1000)
    new_expires_at = now_ms + int((expires_in or 3600)) * 1000
    new_creds = dict(creds)
    new_oauth = dict(oauth)
    new_oauth["accessToken"] = new_access
    new_oauth["refreshToken"] = new_refresh
    new_oauth["expiresAt"] = new_expires_at
    new_creds["claudeAiOauth"] = new_oauth
    _save_oauth_creds(new_creds)
    print_ts(f"OAuth: refreshed token (expires in {(expires_in or 3600)//60}min)")
    return new_creds


async def _load_oauth_access_token(force_refresh: bool = False) -> str | None:
    """Return a valid OAuth access token, refreshing if expired.

    Refresh coordination (ASYNC dedup, 2026-05-26): when multiple agents
    call concurrently and the token needs refreshing, only ONE actual HTTP
    refresh fires. Other coroutines `await` the same `_REFRESH_INFLIGHT`
    future until it resolves, then re-read creds from disk (winner just
    rewrote). Mirrors Claude Code's `refreshTokenPromises` pattern but
    with asyncio primitives — earlier `threading.Lock`+`concurrent.Future`
    implementation didn't actually serialize coroutines (sync primitives
    in a single-threaded async runtime are no-ops for our concurrency
    model). Code review caught the gap.

    On refresh failure we return None — never the expired access token. The
    old behavior of falling back to the expired token caused the caller to
    hit a 401 from the chat API, which fired another forced-refresh on the
    same dead endpoint, looping. Returning None lets the caller surface a
    real "auth unavailable" error instead.

    After a 429 from the refresh endpoint, we back off for
    _REFRESH_BACKOFF_AFTER_429_S seconds before attempting again — caller
    gets None during the cooldown.
    """
    global _REFRESH_LOCK, _REFRESH_INFLIGHT, _REFRESH_LAST_429_TS
    # Lazy-init the asyncio.Lock against whatever event loop is running.
    # Module load happens before any loop exists; first call gets a loop.
    if _REFRESH_LOCK is None:
        _REFRESH_LOCK = asyncio.Lock()

    creds = _load_oauth_creds()
    if not creds:
        print_ts(
            f"{COLOR_YELLOW}AnthropicConversation: no OAuth creds found. "
            f"Run `claude` to log in.{COLOR_END}",
            error=True,
        )
        return None
    oauth = creds.get("claudeAiOauth") or {}
    access = oauth.get("accessToken")
    expires_at = oauth.get("expiresAt", 0)
    import time
    now_ms = int(time.time() * 1000)
    needs_refresh = force_refresh or not access or (expires_at - _REFRESH_LEEWAY_MS) < now_ms
    if not needs_refresh:
        return access

    # Recent-success short-circuit: if disk creds were written within the
    # last _RECENT_REFRESH_SHORTCIRCUIT_S seconds, another caller already
    # finished a successful refresh. Re-read disk and return that token
    # instead of firing yet another refresh — even with force_refresh=True.
    # Handles the thundering-herd: 3 agents 401 simultaneously, all call
    # force_refresh=True, dedup serializes them, winner succeeds and writes
    # disk, the losers' fresh disk-read returns the new token.
    # Also handles the transient-401 case where the on-disk token is fine
    # and the 401 was a server hiccup: we avoid burning a refresh attempt.
    disk_age = _disk_creds_age_s()
    if disk_age < _RECENT_REFRESH_SHORTCIRCUIT_S:
        fresh_creds = _load_oauth_creds()
        fresh_access = ((fresh_creds or {}).get("claudeAiOauth") or {}).get("accessToken")
        if fresh_access:
            print_ts(
                f"OAuth refresh: skipping — disk creds written {disk_age:.1f}s ago "
                f"by another caller"
            )
            return fresh_access

    # 429 cooldown: don't hammer the rate-limited endpoint. Surface the
    # existing on-disk access_token instead of None — the 401 that brought
    # us here may have been a transient API hiccup rather than a real
    # expiry, so the existing token will likely still work on the next
    # call. Falling back to None would mean ~60s of every API call failing
    # with "auth unavailable" even when the token is still valid.
    if _REFRESH_LAST_429_TS and (time.time() - _REFRESH_LAST_429_TS) < _REFRESH_BACKOFF_AFTER_429_S:
        remaining = _REFRESH_BACKOFF_AFTER_429_S - (time.time() - _REFRESH_LAST_429_TS)
        if access:
            print_ts(
                f"{COLOR_YELLOW}OAuth refresh: in 429 cooldown ({remaining:.0f}s left), "
                f"returning existing on-disk token instead of refreshing{COLOR_END}",
                error=True,
            )
            return access
        print_ts(
            f"{COLOR_YELLOW}OAuth refresh: in 429 cooldown ({remaining:.0f}s left), "
            f"no existing token to fall back to{COLOR_END}",
            error=True,
        )
        return None

    # Dedupe: serialize concurrent refreshes for the same refresh_token.
    # First caller fires the work; subsequent callers await the same Future
    # and then re-read creds from disk to get whatever the winner wrote.
    current_refresh_token = oauth.get("refreshToken")
    fut: asyncio.Future | None = None
    is_winner = False
    async with _REFRESH_LOCK:
        if (_REFRESH_INFLIGHT is not None
                and _REFRESH_INFLIGHT.get("refresh_token") == current_refresh_token
                and not _REFRESH_INFLIGHT["future"].done()):
            fut = _REFRESH_INFLIGHT["future"]
            print_ts("OAuth refresh: dedup — awaiting in-flight refresh for shared token")
        else:
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            _REFRESH_INFLIGHT = {"refresh_token": current_refresh_token, "future": fut}
            is_winner = True
            print_ts("OAuth refresh: dedup — kicking off new refresh")

    if is_winner:
        # _refresh_with_file_lock uses urllib (sync) wrapped in cross-process
        # flock. Run in thread executor so the event loop doesn't block
        # during the lock-wait + 1-15s round-trip. The file lock prevents
        # us from refreshing concurrently with the `claude` CLI or other
        # openflip processes; the asyncio.Lock above handles intra-process.
        try:
            refreshed = await asyncio.get_running_loop().run_in_executor(
                None, _refresh_with_file_lock, creds,
            )
            fut.set_result(refreshed)
            # Review ask: explicit success/fail log so the test plan can
            # verify which path the winner took. Without this, dedup log
            # shows "kicking off" but nothing about the outcome.
            if refreshed:
                print_ts("OAuth refresh: dedup winner — refresh succeeded")
            else:
                print_ts("OAuth refresh: dedup winner — refresh failed (returned None)")
        except Exception as e:
            fut.set_exception(e)
            print_ts(
                f"{COLOR_RED}OAuth refresh: dedup winner — refresh raised: {e}{COLOR_END}",
                error=True,
            )
        finally:
            async with _REFRESH_LOCK:
                # Clear the inflight slot so the next refresh cycle starts fresh.
                # Late-arrival race (caller arrives after this clears but before
                # creds are stale again) is handled by the creds-on-disk being
                # fresh — late caller's `needs_refresh` check returns False.
                if _REFRESH_INFLIGHT is not None and _REFRESH_INFLIGHT["future"] is fut:
                    _REFRESH_INFLIGHT = None
        if fut.exception() is not None:
            return None
        refreshed = fut.result()
        if refreshed:
            return (refreshed.get("claudeAiOauth") or {}).get("accessToken")
        return None

    # Loser path: wait on the winner's future, then re-read creds from disk.
    try:
        await asyncio.wait_for(fut, timeout=30.0)
    except Exception:
        # If the winner failed (or we timed out), also fail. Caller
        # surfaces auth-unavailable.
        return None
    # Re-read creds — winner just wrote a new token (or didn't, if it failed).
    # `_save_oauth_creds` runs synchronously inside `_refresh_oauth_token`
    # BEFORE `fut.set_result` fires, so when we get here the new token
    # is already on disk.
    fresh_creds = _load_oauth_creds()
    if not fresh_creds:
        return None
    return (fresh_creds.get("claudeAiOauth") or {}).get("accessToken")


def _build_tool_schemas(tools: list[Callable]) -> list[dict]:
    """Convert openflip tool callables to Anthropic Messages API tool format."""
    schemas = []
    for func in tools or []:
        spec = getattr(func, "tool_spec", None)
        if spec and isinstance(spec, dict):
            schemas.append({
                "name": spec.get("name") or func.__name__,
                "description": spec.get("description") or (func.__doc__ or "").strip()[:1000],
                "input_schema": spec.get("input_schema") or spec.get("parameters") or {
                    "type": "object", "properties": {},
                },
            })
        else:
            schemas.append({
                "name": func.__name__,
                "description": (func.__doc__ or "").strip()[:1000],
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            })
    return schemas


def _openflip_msgs_to_anthropic(messages: list, system_prompt: str) -> tuple[str, list[dict]]:
    """Convert local ChatMessage list to Anthropic API message format.

    Returns (system_prompt, anthropic_messages). System messages are pulled
    out into the top-level `system` field.

    Tool-use round-tripping: when an assistant message carries `tool_calls`
    AND the immediately following tool messages all have `tool_use_id`
    matching the calls, the assistant message is emitted as a content list
    with both text and tool_use blocks, and the tool messages collapse into
    a single user message of tool_result blocks. This is what Anthropic's
    Messages API requires for a tool round-trip to validate.

    If pairing is incomplete (e.g. the assistant message lost its
    `tool_calls` to a process restart, or a tool message lacks
    `tool_use_id`), the affected messages degrade to plain text — tool
    results become `[Previous tool result: ...]` user messages and the
    assistant message stays text-only. Lossy but doesn't 400.
    """
    api_msgs: list[dict] = []
    sys_parts: list[str] = []
    if system_prompt:
        sys_parts.append(system_prompt)

    def _msg_role(m):
        return m.get("role") if hasattr(m, "get") else getattr(m, "role", None)

    def _msg_content(m):
        return (m.get("content_text", None) if hasattr(m, "get") else None) \
            or (m.get("content", "") if hasattr(m, "get") else getattr(m, "content", "")) \
            or ""

    def _msg_tool_calls(m):
        if hasattr(m, "get"):
            tcs = m.get("tool_calls", None)
        else:
            tcs = getattr(m, "tool_calls", None)
        return tcs or []

    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        role = _msg_role(m)
        content = _msg_content(m)

        if role == "system":
            if content:
                sys_parts.append(content)
            i += 1
            continue

        if role == "tool":
            # Stray tool message (not preceded by a paired assistant tool_use).
            # Demote to user text so it stays in context but doesn't violate
            # tool_use_id pairing rules.
            api_msgs.append({
                "role": "user",
                "content": f"[Previous tool result: {content}]",
            })
            i += 1
            continue

        if role == "assistant":
            tool_calls = _msg_tool_calls(m)
            # Claude Code parity (normalizeMessage / isNotEmptyMessage): an
            # assistant message with NO tool_calls and empty/whitespace content
            # is a "(no content)" artifact — a prior empty end_turn that got
            # persisted. Anthropic rejects empty assistant content, and it
            # trips our own empty_assistant_content validator every turn,
            # eventually failing the terminal contract and silencing the agent.
            # Skip it entirely so it never reaches the API. (Messages WITH
            # tool_calls always build non-empty tool_use blocks, so this only
            # drops genuinely-empty text turns.)
            if not tool_calls and not (content or "").strip():
                i += 1
                continue
            if tool_calls:
                # Try to gather matched tool results from the immediately
                # following tool messages. Each tool message must have a
                # tool_use_id matching one of our calls; if any are missing
                # ids or the count doesn't line up, we fall back to text.
                lookahead_results: list[tuple[int, str, str]] = []  # (idx, id, content)
                j = i + 1
                while j < n and _msg_role(messages[j]) == "tool":
                    tm = messages[j]
                    tid = tm.get("tool_use_id", "") if hasattr(tm, "get") else ""
                    lookahead_results.append((j, tid, _msg_content(tm)))
                    j += 1

                call_ids = [getattr(tc, "tool_use_id", "") or "" for tc in tool_calls]
                pair_ok = (
                    len(lookahead_results) == len(tool_calls)
                    and all(rid for _, rid, _ in lookahead_results)
                    and all(cid for cid in call_ids)
                    and {rid for _, rid, _ in lookahead_results} == set(call_ids)
                )

                if pair_ok:
                    # Emit assistant message with text + tool_use blocks.
                    blocks: list[dict] = []
                    if content:
                        blocks.append({"type": "text", "text": content})
                    for tc in tool_calls:
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.tool_use_id,
                            "name": tc.function_name,
                            "input": tc.args or {},
                        })
                    api_msgs.append({"role": "assistant", "content": blocks})
                    # Emit paired tool_results as a single user message.
                    result_blocks = [
                        {
                            "type": "tool_result",
                            "tool_use_id": rid,
                            "content": rcontent,
                        }
                        for _, rid, rcontent in lookahead_results
                    ]
                    api_msgs.append({"role": "user", "content": result_blocks})
                    i = j  # skip past the paired tool messages
                    continue

                # Pairing failed — emit assistant as text-only; following tool
                # messages will be picked up by the `role == 'tool'` branch
                # above as legacy demoted text.
                api_msgs.append({"role": "assistant", "content": content})
                i += 1
                continue

            # Plain assistant text message.
            api_msgs.append({"role": "assistant", "content": content})
            i += 1
            continue

        if role == "user":
            api_msgs.append({"role": "user", "content": content})
            i += 1
            continue

        # Unknown role — skip.
        i += 1

    return "\n\n".join(p for p in sys_parts if p), api_msgs


def _inject_pending_image_attachments(api_messages: list[dict], pending: list[dict]) -> int:
    """Inject queued image attachments into the LAST user message of
    api_messages as Anthropic image blocks. Returns the count injected.

    All image-content constraint handling (size, dimensions, media type,
    animated-frame coercion) lives in `_image_validator`. This function
    only concerns itself with message-structure: where the image_blocks
    attach, and how to preserve tool_use/tool_result adjacency. The
    boundary (fetch_discord_message) already runs the validator at
    download time, so this is a thin safety net; rejections here should
    be rare.

    Anthropic vision shape:
        {"type": "image", "source": {"type": "base64",
         "media_type": "image/jpeg", "data": "<b64>"}}

    Mutates api_messages in place. Caller is responsible for clearing the
    pending list after the call.
    """
    import base64
    from ._image_validator import validate_and_normalize_image
    if not pending or not api_messages:
        return 0
    # Find the most recent user message.
    last_user_idx = None
    for i in range(len(api_messages) - 1, -1, -1):
        if api_messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return 0
    last_user = api_messages[last_user_idx]
    content = last_user.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        return 0
    image_blocks: list[dict] = []
    for entry in pending:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not path:
            continue
        declared = (entry.get("content_type") or "image/png").lower()
        normalized, mt_or_reason = validate_and_normalize_image(path, declared)
        if normalized is None:
            print_ts(
                f"{COLOR_YELLOW}image attachment ({path}) rejected at "
                f"inject-side safety net: {mt_or_reason}{COLOR_END}",
            )
            continue
        try:
            b64 = base64.b64encode(normalized).decode("ascii")
        except Exception:
            continue
        image_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mt_or_reason,
                "data": b64,
            },
        })
    if not image_blocks:
        return 0
    # tool_use/tool_result adjacency check: if the most-recent user message
    # contains a tool_result block, Anthropic requires that tool_result to
    # come FIRST in the message (immediately following its tool_use in the
    # prior assistant turn). Prepending image_blocks into the same message
    # would push the tool_result down and break the adjacency rule with a
    # 400 ("tool_use ids were found without tool_result blocks immediately
    # after").
    #
    # Fix: when the target user message carries tool_result blocks, append
    # the image attachments as a SEPARATE NEW user message AFTER it. The
    # shape becomes:
    #   assistant: [tool_use(X)]
    #   user:      [tool_result(X)]           ← adjacency preserved
    #   user:      [image_block, image_block] ← visual attachments
    #
    # When the target user message has no tool_result, prepend in place
    # (legacy behavior — matches Anthropic's "image, then question" advice).
    has_tool_result = any(
        isinstance(b, dict) and b.get("type") == "tool_result"
        for b in content
    )
    if has_tool_result:
        api_messages.insert(last_user_idx + 1, {
            "role": "user",
            "content": image_blocks,
        })
    else:
        last_user["content"] = image_blocks + content
    return len(image_blocks)


class AnthropicConversation:
    """Anthropic-direct provider used when agent.provider == 'anthropic'.

    Uses direct HTTP to /v1/messages with OAuth bearer auth. Requests route
    through the owner's Claude Code subscription.
    """

    def __init__(self, conversation_id: str, agent: Agent):
        self.conversation_id = conversation_id
        self.agent = agent
        self.model = self._normalize_model(agent.model)
        self.system_message = agent.system_message
        self.messages: list[ChatMessage] = []
        self._persisted_count = 0
        self._http_session: aiohttp.ClientSession | None = None
        # Compaction retry budget. Set to a smaller token cap by the
        # prompt-too-long 400-handler when Anthropic complains the prompt is
        # too long; reset to None once the retry succeeds. Declared here so the
        # field always exists on the instance — a typo at the read site would
        # otherwise silently default to None via getattr() and mask the bug.
        self._retry_budget: int | None = None
        # Most recent server-side compaction block returned by Anthropic.
        # Re-sent at the head of every subsequent request as an assistant
        # message; Anthropic auto-drops every message before this block on
        # arrival, so the effective input shrinks to:
        #   [system] [compaction-summary] [post-summary tail messages]
        # See `compact-2026-01-12` beta + body.context_management.
        self._compaction_block: dict | None = None
        # Latest API usage stats; populated after every chat(). /status reads
        # this so the user can see context utilization without log-grepping.
        self.last_usage: dict | None = None
        # True if the most recent chat() received a fresh compaction block
        # from Anthropic. Runtime reads this after chat() returns and posts
        # an in-channel notice so the user can see compaction happened.
        # Reset to False at the start of every chat() call.
        self.compacted_this_turn: bool = False
        # True when /compact has been invoked. Next chat() will use a low
        # trigger value (50k, the floor) so Anthropic compacts even if
        # input is well under the normal trigger. Reset to False after the
        # next chat() call regardless of whether compaction actually fired.
        self.force_compact_next: bool = False
        # True ONLY when force_compact_next was set by the MANUAL /compact path
        # (not by the auto-threshold gate or the retry-trim follow-up). When
        # set, the outgoing request sends a low compaction trigger
        # (_MANUAL_COMPACT_TRIGGER) so Anthropic compacts regardless of current
        # input size — without this the body always used the real per-model
        # trigger (e.g. 980k on 1m agents), so a manual /compact on a small
        # conversation was a silent no-op. Auto/retry compaction leaves this
        # False and keeps the real trigger. Consumed alongside force_compact_next.
        self.force_compact_trigger_override: bool = False
        # Session-level reasoning-effort override for THIS conversation. When
        # set to a valid level (low/medium/high/xhigh/max) it WINS over the
        # per-model config knob (see _effort_level precedence). None = no
        # override, fall back to the model default. Owner-set via /effort,
        # persisted in meta.json so it survives restarts.
        self.effort_override: str | None = None
        # Reset-epoch guard, set by runtime._run_turn at turn start (a closure
        # capturing the runner's conv epoch for this turn — the SAME check
        # _save_conv enforces). Returns False once /reset bumped the epoch
        # mid-turn. Checked by persist paths that write the JSONL directly,
        # below the _save_conv chokepoint (the compaction trim) — without it a
        # mid-turn compaction rewrites the file from the still-populated
        # in-memory self.messages and resurrects the history /reset just
        # wiped. None (e.g. paths outside _run_turn) = no guard, persist as
        # before.
        self._persist_guard = None

        # Per-turn model override (raw, pre-normalize form — e.g.
        # "anthropic/claude-..."). Set by a transport that wants THIS turn to
        # run on a caller-chosen model without mutating the agent's persistent
        # config. None = use self.agent.model as usual. The every-turn resync
        # (and reapply_agent) honor this via _effective_raw_model(); the
        # external transport clears it in a finally after the turn so it never
        # leaks into a subsequent turn. Only the model id is overridden —
        # reasoning-effort, 1M-context and compaction stay keyed to agent.model.
        self._model_override: str | None = None

    def set_model_override(self, raw_model: str | None) -> None:
        """Override the model used for the NEXT turn only (per-turn).

        `raw_model` is the raw, un-normalized model string as it would appear
        in agent.json (provider prefix / `-1m` suffix allowed — _normalize_model
        strips them). Pass None to clear. The override is consumed by the
        every-turn resync in chat_stream and by reapply_agent; the caller is
        responsible for clearing it (typically in a `finally`) so it does not
        bleed into later turns on the same conversation.
        """
        self._model_override = raw_model or None

    def _effective_raw_model(self) -> str:
        """Raw model string for THIS turn: the per-turn override if set,
        otherwise the agent's configured model."""
        return self._model_override or self.agent.model

    @staticmethod
    def _normalize_model(model_str: str) -> str:
        """Strip provider prefix (`anthropic/...`) and `-1m` context-window
        suffix. The `-1m` flag is encoded as a model-name suffix in the picker
        but Anthropic's API uses a beta header, not a different model id. We
        strip it here and check `_wants_1m_context()` separately when building
        the request headers.
        """
        m = model_str
        if "/" in m:
            m = m.split("/", 1)[1]
        if m.endswith("-1m"):
            m = m[:-3]
        return m

    def _wants_1m_context(self) -> bool:
        """True if the configured model name had the `-1m` suffix, meaning
        we need to add `context-1m-2025-08-07` to the anthropic-beta header.
        """
        raw = self.agent.model or ""
        return raw.endswith("-1m")

    def _effort_level(self) -> str | None:
        """Return the effective reasoning-effort level for this turn, or None.

        Precedence (highest wins):
          1. Session override: self.effort_override, when set to one of the
             five valid levels. Owner-set per-conversation via /effort and
             persisted in meta.json. A junk value is ignored (treated as None).
          2. Per-model config knob: config_global.get_effort reads
             `models.<bare>.effort` from config.json (Anthropic-only).
          3. None → the request omits output_config entirely (API default
             "high"), keeping the body byte-identical to pre-effort behavior.

        Effort is a model capability, not an agent trait, so the config knob
        lives next to compaction_trigger in config.json's models block. The
        session override is conversation-scoped, layered on top.

        Pass agent.model (raw, with the `-1m` suffix), not self.model: the
        config.json key is the full `claude-opus-4-8-1m` and get_effort's
        bare-name resolution keys on it exactly like get_compaction_trigger.
        """
        override = self.effort_override
        if isinstance(override, str) and override.strip().lower() in _VALID_EFFORT_LEVELS:
            return override.strip().lower()
        return get_effort(self.agent.model, "anthropic")

    def _agent_dir(self) -> str:
        return os.path.dirname(self.agent.path)

    def _conversation_path(self) -> str:
        return _cio.conversation_path(self._agent_dir(), self.conversation_id)

    def _meta_path(self) -> str:
        """Sidecar JSON holding non-message state — compaction block, etc.
        Lives next to the .jsonl. Cleared by clear_history."""
        return os.path.join(
            self._agent_dir(), "conversations", f"{_cio.fs_encode(self.conversation_id)}.meta.json"
        )

    def _content_extractor(self, m) -> str:
        return getattr(m, "content_text", None) or m.get("content", "") or ""

    def _archive_and_trim_after_compaction(self):
        """Called when Anthropic returns a fresh compaction block. The block
        is a summary of every message before some cut point — so keeping
        those messages in-memory or on-disk just wastes tokens on every
        future turn. This method:

          1. Copies the live .jsonl to a timestamped backup sidecar so the
             raw history is preserved (auditable, recoverable if needed).
          2. Trims self.messages to start at the most recent user message
             (the one that triggered THIS turn) — everything before is
             dropped from memory.
          3. Rewrites the live .jsonl with only the post-compaction tail.
          4. Resets _persisted_count so the next save() appends the new
             assistant reply correctly.

        Without this method, the conversation never actually shrinks even
        after compaction fires — we keep paying full token cost on every
        turn for content the summary already covers.
        """
        # Epoch guard — same semantics as runtime._save_conv. This method
        # rewrites the JSONL directly from self.messages, BELOW the _save_conv
        # chokepoint, so it must independently respect a mid-turn /reset:
        # clear_history() deletes the file but leaves self.messages populated,
        # and without this check a compaction landing on the dying turn would
        # rewrite the file from that stale copy — resurrecting the history the
        # operator just wiped.
        guard = self._persist_guard
        if guard is not None:
            try:
                allowed = bool(guard())
            except Exception:
                allowed = True
            if not allowed:
                print_ts(
                    "compaction trim: skipping — conversation was reset mid-turn "
                    "(epoch changed); not resurrecting cleared history",
                    agent=self.agent.id,
                )
                return
        import shutil

        def _role(m):
            if hasattr(m, "get"):
                return m.get("role")
            return getattr(m, "role", None)

        # Find cut point: the most recent user message in self.messages.
        # Everything before this is summarized by the compaction block.
        cut_idx = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if _role(self.messages[i]) == "user":
                cut_idx = i
                break

        if cut_idx <= 0:
            # No user message in history, or already at start — nothing to do.
            return

        src = self._conversation_path()
        # Backup live jsonl with timestamp. Even if rewrite fails, the
        # backup is the authoritative archive of pre-compaction state.
        ts = int(time.time())
        bak = src + f".compaction_{ts}.bak.jsonl"
        if os.path.isfile(src):
            try:
                shutil.copy2(src, bak)
                print_ts(f"backed up pre-compaction history to {os.path.basename(bak)}", agent=self.agent.id)
            except Exception as e:
                print_ts(
                    f"{COLOR_RED}compaction backup failed (aborting trim, keeping full history): {e}{COLOR_END}",
                    agent=self.agent.id, error=True,
                )
                return
            # Retention sweep: keep only the N most-recent compaction backups
            # per channel. Without this, backups accumulate forever.
            # Backups carry monotonic unix-second timestamps in their names,
            # so a sorted glob is reliable for age ordering.
            try:
                import glob as _glob
                _backup_keep = 5
                _all_bak = sorted(_glob.glob(src + ".compaction_*.bak.jsonl"))
                if len(_all_bak) > _backup_keep:
                    for _stale in _all_bak[:-_backup_keep]:
                        try:
                            os.remove(_stale)
                            print_ts(
                                f"pruned stale compaction backup: {os.path.basename(_stale)}",
                                agent=self.agent.id,
                            )
                        except OSError:
                            pass
            except Exception as _retain_e:
                # Cleanup failing is not a real error — backups will just
                # accumulate. Log and continue.
                print_ts(
                    f"{COLOR_YELLOW}backup retention sweep failed: {_retain_e}{COLOR_END}",
                    agent=self.agent.id,
                )

        # Trim in-memory
        kept = self.messages[cut_idx:]
        dropped_count = cut_idx
        self.messages = kept

        # Rewrite live jsonl atomically with only the kept messages.
        tmp = src + ".tmp"
        try:
            os.makedirs(os.path.dirname(src), exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                for m in self.messages:
                    f.write(json.dumps({
                        "role": _role(m),
                        "content": self._content_extractor(m) or "",
                        "ts": time.time(),
                    }, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, src)
            self._persisted_count = len(self.messages)
            print_ts(
                f"compaction trim: dropped {dropped_count} pre-compaction messages, "
                f"kept {len(self.messages)} (live jsonl rewritten)",
                agent=self.agent.id,
            )
        except Exception as e:
            print_ts(
                f"{COLOR_RED}compaction trim rewrite failed: {e}{COLOR_END}",
                agent=self.agent.id, error=True,
            )
            # Restore live jsonl from backup so we don't lose state on disk.
            try:
                shutil.copy2(bak, src)
            except Exception:
                pass

    def _save_meta(self):
        """Persist non-message state (compaction block + last_usage) so it
        survives bot restarts. Without this:
          - every restart triggers a fresh server-side compaction (paid
            token cost + one-turn cache miss)
          - /status shows 0 context until a chat() call has completed,
            because last_usage starts as None on a fresh process

        We strip any `cache_control` field from the compaction block before
        writing — that's a transient cache marker, not part of the block
        Anthropic gave us.
        """
        payload: dict = {}
        if self._compaction_block is not None:
            payload["compaction_block"] = {
                k: v for k, v in self._compaction_block.items() if k != "cache_control"
            }
        if self.last_usage is not None:
            payload["last_usage"] = self.last_usage
        # Session effort override — written ONLY when set, so meta stays
        # byte-identical for conversations that never touched /effort.
        if isinstance(self.effort_override, str) and \
                self.effort_override.strip().lower() in _VALID_EFFORT_LEVELS:
            payload["effort_override"] = self.effort_override.strip().lower()
        if not payload:
            # Nothing to persist. If a meta file already exists, it may hold a
            # now-cleared key (e.g. /effort default zeroed the only field that
            # was set) — rewrite it as empty so the stale value doesn't survive
            # a restart. If no meta file exists, leave it absent (byte-identical
            # to today for conversations that never persisted anything).
            try:
                if os.path.isfile(self._meta_path()):
                    save_json(self._meta_path(), payload)
            except OSError:
                pass
            return
        save_json(self._meta_path(), payload)

    def load(self):
        _cio.migrate_legacy_to_jsonl(
            self._agent_dir(), self.conversation_id,
            log_agent_id=self.agent.id,
        )
        msgs = _cio.read_all_messages(self._conversation_path())
        meta = load_json(self._meta_path(), default={}) or {}
        stored_block = meta.get("compaction_block")
        if isinstance(stored_block, dict) and stored_block.get("type") == "compaction":
            self._compaction_block = stored_block
            print_ts(
                f"Restored compaction block from meta sidecar",
                agent=self.agent.id,
            )
        stored_usage = meta.get("last_usage")
        if isinstance(stored_usage, dict):
            # Validate it has the keys /status expects, but tolerate
            # missing ones — old meta files won't have all fields.
            self.last_usage = stored_usage
        stored_effort = meta.get("effort_override")
        if isinstance(stored_effort, str) and stored_effort.strip().lower() in _VALID_EFFORT_LEVELS:
            self.effort_override = stored_effort.strip().lower()
        # else: bad/missing → leave self.effort_override at its __init__ None.
        if not msgs:
            return
        for entry in msgs:
            self.messages.append(ChatMessage(entry["role"], entry.get("content", "")))
        self._persisted_count = len(msgs)
        print_ts(
            f"Loaded {len(msgs)} messages from disk (anthropic-direct)",
            agent=self.agent.id,
        )

    def save(self):
        non_system = [m for m in self.messages if m.get("role") != "system"]
        new_count = len(non_system) - self._persisted_count
        if new_count <= 0:
            self._persisted_count = len(non_system)
            return
        _cio.append_messages(
            self._conversation_path(),
            non_system[-new_count:],
            content_extractor=self._content_extractor,
        )
        self._persisted_count = len(non_system)

    def _trim_to_fit_window(self) -> int:
        """Pre-flight LOCAL TRIM: drop oldest messages until estimated input
        fits in (context_window - 10k). Returns count dropped. Char/2
        estimator — rough but cheap (see `_est` below for why //2, not //4).
        Only fires on retry after a 400 (the `_retry_budget is not None` gate
        at the call sites in chat_stream) — in healthy operation
        the auto-compact gate in chat() / streaming paths requests Anthropic
        compaction at (window - 20k) BEFORE we get this close to the window, so
        this local-trim path stays as the last-line safety net for cases
        where Anthropic compaction failed to engage.

        DELIBERATE sibling divergence from DiscordConversation._trim_to_fit_window
        (conversation.py:126), in three ways — all intentional, do NOT align
        them. Each is justified by a feature this provider has that ollama
        lacks: (1) Anthropic hard-400s on overflow → //2 over-estimate (ollama
        silently truncates → //4); (2) Anthropic prompt caching → trim to
        budget*0.8 for cache-prefix stability (ollama has no caching → exact
        budget); (3) Anthropic server-side compaction → trim only on 400-retry
        (ollama has none → trims every turn). See the matching note in
        conversation.py for the ollama-side rationale.

        Preserves system_message and self._compaction_block (both sit
        outside self.messages — the compaction block is the summary of
        everything before, so dropping it would defeat the rescue).

        After trim, if a compaction block is present and the head of
        self.messages is an 'assistant' role, keeps dropping — sending
        [assistant(compaction), assistant(...)] back-to-back violates
        Anthropic's alternation rule. 'tool' role demotes to 'user' in
        the converter so it's safe.
        """
        # IMPORTANT: pass agent.model (raw, with `-1m` suffix), not self.model
        # which has been normalized. get_model_context_window keys on the raw
        # name — `claude-opus-4-7-1m` is the 1M entry, `claude-opus-4-7` is the
        # 200k entry. Using the normalized name here would silently set budget
        # to 190k for 1M-beta agents and trigger aggressive head-trimming.
        window = get_model_context_window(self.agent.model, "anthropic")
        if not window:
            return 0
        # _retry_budget is set on retry after a "prompt too long" 400 — it
        # halves the budget each retry so we trim further. None = first try
        # with the normal `window - 10k` budget.
        budget = self._retry_budget if getattr(self, "_retry_budget", None) else (window - 10_000)
        # Target we trim DOWN TO when trim fires. Leaving 20% headroom means
        # the next ~200k tokens of growth fit without re-trimming, so cache
        # stays stable across many turns instead of invalidating each turn.
        # Without this, sitting right at the budget makes every turn rotate
        # 2 messages off the head and the rolling cache marker never hits.
        # DELIBERATE divergence: DiscordConversation trims to the exact budget
        # (conversation.py) — it has no prompt caching, so this headroom would
        # buy it nothing. The 0.8 here is specifically a caching optimization.
        trim_target = int(budget * 0.8)

        def _est(s: str) -> int:
            # chars-per-token varies a lot: ~4 for English prose, ~2 for
            # code/JSON/tool-output-heavy content. Using /2 keeps us safe
            # across content mixes — slightly over-trims for chat-only
            # conversations, correctly trims for tool-heavy ones. The
            # alternative (/4) lets dense content through at 2x real tokens
            # and the request gets 400'd by Anthropic before compaction
            # can run.
            # DELIBERATE divergence: DiscordConversation uses //4
            # (conversation.py) — ollama silently truncates overflow instead of
            # 400'ing, so its under-estimate is harmless. Here a 400 is fatal,
            # so we over-estimate. Don't unify these.
            return len(s or "") // 2

        def _content_str(m) -> str:
            if hasattr(m, "get"):
                return m.get("content_text") or m.get("content", "") or ""
            return getattr(m, "content_text", None) or getattr(m, "content", "") or ""

        def _role(m) -> str:
            return m.get("role") if hasattr(m, "get") else getattr(m, "role", "")

        def _est_compaction(block) -> int:
            # Anthropic's compaction block stores `content` as either a plain
            # string OR a list of content blocks (e.g. [{"type":"text","text":...}]).
            # Handle both; a list under len() would otherwise look ~1 token.
            if not block:
                return 0
            c = block.get("content")
            if isinstance(c, str):
                return _est(c)
            if isinstance(c, list):
                tot = 0
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        tot += _est(b.get("text", ""))
                return tot
            return 0

        sys_cost = _est(self.system_message)
        comp_cost = _est_compaction(self._compaction_block)
        msg_cost = sum(_est(_content_str(m)) for m in self.messages)
        total = sys_cost + comp_cost + msg_cost

        if total <= budget:
            return 0

        original = len(self.messages)

        # Drop oldest non-system messages until we hit trim_target (80% of
        # budget) — not just until under budget. Trimming with headroom
        # means we don't have to trim again for ~200k tokens of growth,
        # which lets the cache prefix stabilize for many turns.
        #
        # FLOOR: always keep at least LAST_KEEP most recent messages so the
        # model has the current user turn (and a little context) to respond
        # to. Without this floor, a huge conversation would drop the entire
        # list including the user's latest message, yielding an empty
        # messages array and a 400 "at least one message is required" from
        # Anthropic. If keeping LAST_KEEP still exceeds the context window,
        # Anthropic will return a clearer "prompt too long" error which the
        # caller can handle (retry with a smaller budget, or auto-compact).
        LAST_KEEP = 4
        i = 0
        while total > trim_target and len(self.messages) > LAST_KEEP and i < len(self.messages) - LAST_KEEP:
            if _role(self.messages[i]) == "system":
                i += 1
                continue
            total -= _est(_content_str(self.messages[i]))
            del self.messages[i]
        if total > trim_target:
            # LAST-RESORT IN-MESSAGE TRUNCATION (2026-06-10). Reaching here
            # means whole-message trimming is exhausted and the kept tail
            # alone still exceeds the window — a single recent message is
            # bigger than the budget. "Sending anyway" used to hand this to
            # Anthropic, which hard-400s, and neither rescue the old comment
            # promised could ever run: auto-compact rides inside an ACCEPTED
            # request, and a smaller-budget retry still can't touch the
            # protected tail. Result: a permanently wedged conversation
            # (the 10.3MB tool-result dump, 2026-06-10). Now we truncate the
            # CONTENT of the largest kept messages until the estimate fits —
            # the model loses the middle of a dump it could never have read
            # anyway, and the conversation stays alive. The runtime-side
            # ingestion cap makes this near-unreachable; defense in depth.
            _TRUNC_MARKER = (
                "\n[FRAMEWORK: this message was emergency-truncated — its "
                "original content was too large to ever fit the model's "
                "context window.]"
            )
            _MIN_KEEP_CHARS = 2000
            while total > trim_target:
                _idx, _len = -1, 0
                for _j, _m in enumerate(self.messages):
                    if _role(_m) == "system":
                        continue
                    _c = _m.get("content")
                    if not isinstance(_c, str):
                        continue
                    if len(_c) > _len:
                        _idx, _len = _j, len(_c)
                if _idx < 0 or _len <= _MIN_KEEP_CHARS:
                    break  # nothing shrinkable left
                _m = self.messages[_idx]
                _old_c = _m["content"]
                # estimator is chars/2, so chars to shed ≈ overage tokens * 2
                _need = (total - trim_target) * 2
                _new_len = max(_MIN_KEEP_CHARS, len(_old_c) - _need - len(_TRUNC_MARKER))
                _new_c = _old_c[:_new_len] + _TRUNC_MARKER
                _m["content"] = _new_c
                total -= _est(_old_c)
                total += _est(_new_c)
                print_ts(
                    f"{COLOR_YELLOW}_trim_to_fit_window: emergency-truncated "
                    f"oversized message at index {_idx} "
                    f"({len(_old_c):,} -> {len(_new_c):,} chars){COLOR_END}",
                    agent=self.agent.id,
                )
            if total > trim_target:
                print_ts(
                    f"{COLOR_YELLOW}_trim_to_fit_window: still over budget after "
                    f"emergency truncation ({total} > {trim_target}); sending anyway"
                    f"{COLOR_END}",
                    agent=self.agent.id,
                )

        # With a compaction block present, the head of self.messages must not
        # be 'assistant' — that would emit [assistant(compaction), assistant(...)]
        # back-to-back and 400 on alternation. 'tool' demotes to 'user' in the
        # converter, so it's safe; only 'assistant' is the hazard.
        if self._compaction_block is not None:
            while self.messages and _role(self.messages[0]) == "assistant":
                total -= _est(_content_str(self.messages[0]))
                del self.messages[0]

        dropped = original - len(self.messages)
        # Keep persisted_count consistent with the shorter in-memory list, so
        # the next save() doesn't fall into its `new_count <= 0` early-return
        # and silently skip persisting this turn's user/assistant messages.
        self._persisted_count = max(0, self._persisted_count - dropped)
        return dropped

    def clear_history(self):
        # Pass the meta sidecar as an extra so it's deleted alongside the
        # .jsonl. Otherwise /reset would leave a stale compaction block on
        # disk; the next message would prepend a summary of the conversation
        # we just nuked, and the agent would "remember" things the user
        # asked to forget — or Anthropic would 400 on the orphan reference.
        _cio.delete_conversation_files(
            self._agent_dir(), self.conversation_id,
            extra_paths=[self._meta_path()],
            backup_tag="pre_reset",
        )
        self._compaction_block = None
        self._persisted_count = 0
        # /status should reflect "fresh conversation" after /reset, not the
        # last_usage from the conversation we just deleted.
        self.last_usage = None

    def reapply_agent(self):
        self.model = self._normalize_model(self._effective_raw_model())
        self.system_message = self.agent.system_message

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            # No `total` cap: it covers the ENTIRE streamed body, so a
            # healthy long generation (or a slow server-side compaction)
            # would be killed mid-stream and its billed tokens discarded.
            # Liveness comes from `sock_read` instead — a per-chunk
            # inactivity timeout. Anthropic's SSE stream emits regular
            # events (including pings) while alive, so 90s of silence
            # means the connection is genuinely stalled.
            timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=90)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def aclose(self):
        if self._http_session and not self._http_session.closed:
            try:
                await self._http_session.close()
            except Exception:
                pass

    async def chat(
        self,
        tools: list[Callable] = None,
        think: bool = None,
        _retry_attempt: int = 0,
        tool_choice: dict | None = None,
    ) -> AnthropicAIChatMessage:
        """Public chat() entry. Thin wrapper that consumes chat_stream()
        and returns an AnthropicAIChatMessage in the shape callers expect.

        This is step 4 of the streaming refactor — chat() used to be a
        ~600-line monolith that POSTed non-streaming and parsed the full
        response. Now chat_stream() does all the heavy lifting and emits
        StreamEvents; this wrapper accumulates them into the same final
        shape callers got before.

        Args / return shape identical to the pre-refactor chat().
        """
        # Lazy import to avoid circular issues at module load.
        from ._anthropic_stream import (
            MessageStartEvent, ContentBlockStartEvent,
            ContentBlockDeltaEvent, ContentBlockStopEvent,
            MessageDeltaEvent, MessageStopEvent, FrameworkErrorEvent,
        )

        # Build the tools_map once so we can map tool_use blocks → callable
        # references when constructing AnthropicToolCall instances.
        tools_map: dict[str, Callable] = {}
        for func in tools or []:
            tools_map[func.__name__] = func

        # Accumulators — these become the AnthropicAIChatMessage's fields
        # at the end. The streaming path delivers content in arrival order
        # via ContentBlockStopEvent; we preserve that order in the final
        # message so the model's next turn sees text/tool_use interleaved
        # exactly as it emitted them. Declared here, RESET per attempt in
        # the transient-retry loop below.
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[AnthropicToolCall] = []
        ordered_blocks: list[dict] = []  # full content list for raw_response
        framework_error: FrameworkErrorEvent | None = None
        stop_reason: str | None = None
        stop_details: dict | None = None

        # ---------- Transient-error retry loop (2026-07-06) ----------
        # Mirrors the Anthropic SDK / Claude Code resilience model: 429,
        # 5xx/529, timeouts, and connection errors are retried here with
        # exponential backoff + jitter (retry-after honored on 429) before
        # ANY error surfaces to the runtime. This wrapper buffers the whole
        # stream and nothing is user-visible until the turn ends, so a
        # retry-from-scratch after a mid-stream death is invisible — unlike
        # a live-streaming client we can safely re-request even after
        # partial content arrived (capped at ONE retry in that case to
        # bound double-billing). Structural errors (bad_request, auth)
        # never retry. Total wait is budgeted to stay well inside the
        # runtime's per-round wait_for.
        import random as _random
        _TRANSIENT_KINDS = {
            "rate_limit", "overloaded", "timeout", "transport",
            "json_decode", "sse_protocol",
        }
        _MAX_TRANSIENT_RETRIES = 3
        _RETRY_BUDGET_S = 90.0
        _waited = 0.0
        _notice_fired = False
        _transient_attempt = 0

        while True:
            text_parts = []
            thinking_parts = []
            tool_calls = []
            ordered_blocks = []
            framework_error = None
            stop_reason = None
            stop_details = None
            try:
                async for event in self.chat_stream(
                    tools=tools, think=think,
                    tool_choice=tool_choice,
                    _retry_attempt=_retry_attempt,
                ):
                    if isinstance(event, FrameworkErrorEvent):
                        framework_error = event
                        # Continue iterating to let chat_stream finish its
                        # `finally` cleanup; don't break early. chat_stream
                        # itself will `return` after FrameworkErrorEvent so
                        # the loop ends naturally.
                        continue
                    if isinstance(event, ContentBlockStopEvent):
                        blk = event.completed_block
                        btype = blk.get("type", "")
                        ordered_blocks.append(blk)
                        if btype == "text":
                            text_parts.append(blk.get("text", "") or "")
                        elif btype == "thinking":
                            thinking_parts.append(blk.get("thinking", "") or "")
                        elif btype == "tool_use":
                            name = blk.get("name", "")
                            args = blk.get("input", {}) or {}
                            tool_use_id = blk.get("id", "")
                            # Always append the tool_call. If the name is unknown,
                            # let the downstream executor surface the error to
                            # the model — dropping silently here produces an
                            # empty assistant message and a terminal-contract
                            # failure the user sees as "no reply".
                            tool_calls.append(AnthropicToolCall(
                                function_name=name,
                                args=args,
                                tool_use_id=tool_use_id,
                                function=tools_map.get(name),
                            ))
                        # compaction blocks are handled by chat_stream's
                        # side-effect path — they update self._compaction_block
                        # and self.compacted_this_turn directly.
                    elif isinstance(event, MessageDeltaEvent):
                        # Carries the terminal stop_reason (+ stop_details on a
                        # refusal). Previously dropped here, which turned a
                        # zero-text response (e.g. stop_reason="refusal") into a
                        # silent empty reply with no explanation. Capture it so
                        # the assembly step below can surface WHY the turn
                        # produced no text instead of returning a blank.
                        if event.stop_reason is not None:
                            stop_reason = event.stop_reason
                        if event.stop_details is not None:
                            stop_details = event.stop_details
                    # Other event types (MessageStart, ContentBlockStart,
                    # ContentBlockDelta, MessageStop) don't contribute to the
                    # final AnthropicAIChatMessage shape; chat_stream handles
                    # their side effects (last_usage, etc).
            except MalformedRequestError:
                # Let pre-flight validation failures propagate so runtime.py
                # can surface a clear user-visible message and route to the
                # malformed-request error path instead of the generic chat
                # error path. Don't bury it as a framework-error string.
                raise
            except Exception as _wrapper_e:
                print_ts(
                    f"{COLOR_RED}chat() wrapper consumption error: {_wrapper_e}{COLOR_END}",
                    agent=self.agent.id, error=True,
                )
                _msg = AnthropicAIChatMessage(
                    content=f"⚠️ chat() wrapper failed: {_wrapper_e}",
                    is_framework_error=True,
                )
                _msg.framework_error_kind = "transport"
                return _msg

            if framework_error is None:
                break  # success — fall through to post-processing

            _kind = framework_error.kind
            _had_partial = bool(ordered_blocks)
            _retry_cap = 1 if _had_partial else _MAX_TRANSIENT_RETRIES
            if _kind not in _TRANSIENT_KINDS or _transient_attempt >= _retry_cap:
                break  # terminal, or retries exhausted — give up below

            if _kind == "rate_limit":
                _wait = framework_error.retry_after or 5.0
            else:
                _wait = min(1.0 * (2 ** _transient_attempt), 8.0)
            _wait += _random.uniform(0, _wait * 0.2)
            if _waited + _wait > _RETRY_BUDGET_S:
                break  # would blow the budget (e.g. huge retry-after) — give up

            # Stay silent under the typing indicator for short waits; let
            # the operator know only when recovery is taking a while.
            if not _notice_fired and (_waited + _wait) > 8.0:
                _notice_fired = True
                _cb = getattr(self, "retry_notice_cb", None)
                if _cb is not None:
                    try:
                        await _cb("⏳ Anthropic's having a moment — retrying…")
                    except Exception:
                        pass
            print_ts(
                f"{COLOR_YELLOW}transient {_kind}"
                f"{f' (HTTP {framework_error.status})' if framework_error.status else ''}"
                f" — retry {_transient_attempt + 1}/{_retry_cap} in {_wait:.1f}s"
                f"{' (had partial content)' if _had_partial else ''}{COLOR_END}",
                agent=self.agent.id,
            )
            await asyncio.sleep(_wait)
            _waited += _wait
            _transient_attempt += 1

        # Stream-side framework error survived the retry loop → give up.
        if framework_error is not None:
            # Salvage a substantial pure-text partial instead of discarding
            # billed tokens: only when NO tool calls completed (a partial
            # tool exchange must not be half-acted-on) and the death was
            # transient (a bad_request's partial is untrustworthy).
            _salvage = "".join(text_parts).strip()
            if (framework_error.kind in _TRANSIENT_KINDS
                    and len(_salvage) >= 200 and not tool_calls):
                print_ts(
                    f"{COLOR_YELLOW}stream died after {len(_salvage)} chars of text — "
                    f"salvaging partial response{COLOR_END}",
                    agent=self.agent.id,
                )
                response = AnthropicAIChatMessage(
                    content=_salvage + "\n\n*(response was cut off mid-stream)*",
                    tool_calls=[],
                )
                response.raw_response = {"content": ordered_blocks}
                return response
            _msg = AnthropicAIChatMessage(
                content=framework_error.message,
                is_framework_error=True,
            )
            _msg.framework_error_kind = framework_error.kind
            return _msg

        # ----------- Malformed-tool_use detection + retry -----------
        # Mirrors the legacy chat() post-collect logic. Anthropic
        # occasionally emits a tool_use block with empty `input` despite
        # required fields. We detect that and retry with a [FRAMEWORK]
        # nudge so the model fixes it.
        if tool_calls and _retry_attempt < 2:
            malformed = []
            for tc in tool_calls:
                schema = (getattr(tc.function, "tool_spec", {}) or {}).get(
                    "input_schema", {}
                ) or {}
                required = schema.get("required") or []
                if required and not tc.args:
                    malformed.append((tc.function_name, required))
            if malformed:
                names_and_fields = "; ".join(
                    f"{n} (needs: {', '.join(r)})" for n, r in malformed
                )
                print_ts(
                    f"{COLOR_YELLOW}malformed tool_use detected ({names_and_fields}) — "
                    f"retry {_retry_attempt + 1}/2 (stream wrapper){COLOR_END}",
                    agent=self.agent.id,
                )
                nudge = (
                    "[FRAMEWORK]: Your last tool_use block had empty input but "
                    f"required fields are: {names_and_fields}. Retry the call with all "
                    "required arguments filled in. If you can't fill them, reply in text "
                    "explaining what you need instead."
                )
                self.messages.append(ChatMessage("user", nudge))
                try:
                    retry_response = await self.chat(
                        tools=tools, think=think,
                        tool_choice=tool_choice,
                        _retry_attempt=_retry_attempt + 1,
                    )
                    return retry_response
                finally:
                    if (self.messages and self.messages[-1].role == "user"
                            and "[FRAMEWORK]:" in (self.messages[-1].get("content", "") or "")):
                        self.messages.pop()

        # Assemble final AnthropicAIChatMessage. Content blocks are
        # joined in arrival order. raw_response stores the full content
        # array so downstream code that wants to introspect block order
        # (e.g. for persistence with interleaved text+tool_use) can.
        response = AnthropicAIChatMessage(
            content="".join(text_parts),
            tool_calls=tool_calls,
        )
        if thinking_parts:
            response.thinking = "".join(thinking_parts)
        response.raw_response = {"content": ordered_blocks}

        # Zero-text, zero-tool turn → would render as a silent blank reply.
        # Surface the API's stop_reason (and stop_details on a refusal) so
        # the void becomes an explanation instead of a mystery. Marked as a
        # framework error so it doesn't pollute history as an assistant turn.
        if not response.content and not tool_calls:
            if stop_reason == "refusal":
                cat = (stop_details or {}).get("category", "unspecified")
                why = (stop_details or {}).get("explanation", "").strip()
                msg = f"⚠️ Model declined (stop_reason=refusal, category={cat})."
                if why:
                    msg += f" {why}"
            elif stop_reason == "end_turn":
                # A clean empty end_turn is a legitimate "nothing to add" finish,
                # NOT an API failure — it routinely closes bookkeeping tool chains
                # (save_memory etc.) where the model already said its piece
                # earlier in the turn. Returning a framework error here posts
                # "⚠️ Empty reply" straight to the operator, pre-empting the
                # machinery that already handles this case: runtime's empty-reply
                # nudge-and-retry (one shot, so the model gets a chance to close
                # with text) and, if it stays empty, the terminal contract's
                # _clean_empty_end_turn suppression (Claude Code parity). Defer
                # to them instead.
                print_ts(
                    f"{COLOR_YELLOW}empty reply (stop_reason=end_turn) — deferring "
                    f"to runtime nudge-and-retry / quiet-finish{COLOR_END}",
                    agent=self.agent.id,
                )
                return response
            elif stop_reason:
                msg = (
                    f"⚠️ Empty reply (stop_reason={stop_reason}). No text or "
                    f"tool_use was returned."
                )
            else:
                # No content, no tool_use, AND no stop_reason — a transient
                # API glitch, not a model decision. Return the plain empty
                # response so runtime.py's empty-reply nudge-and-retry path
                # handles it (one-shot per turn). Returning a framework
                # error here short-circuits that path and posts the warning
                # straight to Discord — the 2026-06-10 regression.
                print_ts(
                    f"{COLOR_YELLOW}empty reply with no stop_reason — "
                    f"deferring to runtime nudge-and-retry{COLOR_END}",
                    agent=self.agent.id,
                )
                return response
            print_ts(
                f"{COLOR_RED}{msg}{COLOR_END}",
                agent=self.agent.id, error=True,
            )
            return AnthropicAIChatMessage(content=msg, is_framework_error=True)

        return response

    def _auto_compact_due(self) -> bool:
        """True if the last measured `total_input` already exceeds this
        model's compaction trigger. Single source of truth for the
        auto-compaction decision — used both by chat_stream() to SET
        `force_compact_next` mid-turn and by will_compact_this_turn() so
        the runtime start-notice agrees with what chat() will actually do.
        """
        if not isinstance(self.last_usage, dict):
            return False
        _last_total = int(self.last_usage.get("total_input") or 0)
        _cap = get_compaction_trigger(self.agent.model, "anthropic")
        return _last_total > _cap > 0

    def will_compact_this_turn(self) -> bool:
        """True if this turn will request Anthropic compaction — either the
        flag is already set (manual /compact, or a prior retry-trim) or the
        auto-compaction threshold is already exceeded. runtime.py reads this
        BEFORE chat() so the "⚙️ Compacting conversation..." start notice
        fires for auto-compaction too, not just manual /compact.
        """
        return bool(self.force_compact_next) or self._auto_compact_due()

    async def chat_stream(
        self,
        tools: list[Callable] = None,
        think: bool = None,
        tool_choice: dict | None = None,
        _retry_attempt: int = 0,
    ):
        """Streaming variant of chat() — yields StreamEvent objects as the
        Anthropic SSE stream arrives.

        This is the structural fix for the "model said it without doing it"
        failure mode: ContentBlockStopEvent for tool_use blocks fires the
        moment the model commits a tool call, before the model is done
        speaking. The runtime consumer dispatches the tool right there,
        eliminating the gap where the model can emit prose-then-end with
        no tool call.

        Yields:
            StreamEvent subclasses from _anthropic_stream — see that
            module's docstring for the full taxonomy. Always terminates
            with either MessageStopEvent (clean) or FrameworkErrorEvent
            (any error). Consumers must handle FrameworkErrorEvent (post
            to user / break / etc).

        Side effects matching chat():
            - self.compacted_this_turn set to True if a compaction
              content_block arrives (detected on ContentBlockStopEvent).
            - self._compaction_block populated + persisted via _save_meta().
            - self.last_usage updated + persisted on MessageDeltaEvent.
            - self.force_compact_next consumed (cleared) after the request.
            - Pre-flight _trim_to_fit_window() runs same as chat().

        Args mirror chat(). _retry_attempt is internal for the 401-refresh
        retry path (currently only the auth path retries; bad-request /
        timeout don't retry from chat_stream — caller can retry by
        re-invoking).
        """
        from ._anthropic_stream import (
            stream_sse_events,
            MessageStartEvent, ContentBlockStartEvent,
            ContentBlockDeltaEvent, ContentBlockStopEvent,
            MessageDeltaEvent, MessageStopEvent, FrameworkErrorEvent,
        )

        # Reset per-turn flags only on the first attempt — retries shouldn't
        # clobber a compaction-block we may have just received pre-retry.
        if _retry_attempt == 0:
            self.compacted_this_turn = False

        access_token = await _load_oauth_access_token()
        if not access_token:
            yield FrameworkErrorEvent(
                message=f"⚠️ Anthropic OAuth token unavailable — check {_CREDS_PATH}.",
                kind="auth",
            )
            return

        # See chat() — trim only fires on retry-after-overflow.
        # Auto-compact gate: if the last turn's measured total_input was
        # already past (window - 20k), request Anthropic compaction THIS
        # turn so the request does not grow further into the over-window
        # tier (Anthropic charges premium rates above the nominal window
        # on opus models; capping prevents that overage cost). Uses the
        # last actually-measured count from self.last_usage rather than
        # estimating, so we do not over-trigger.
        if not self.force_compact_next and self._auto_compact_due():
            _last_total = int(self.last_usage.get("total_input") or 0)
            _cap = get_compaction_trigger(self.agent.model, "anthropic")
            self.force_compact_next = True
            print_ts(
                f"{COLOR_YELLOW}auto-compact: last total_input {_last_total:,} > "
                f"trigger {_cap:,}; requesting Anthropic compaction this turn"
                f"{COLOR_END}",
                agent=self.agent.id,
            )
        # DELIBERATE divergence (trigger timing): the local trim fires ONLY on
        # a 400-retry (gated on `_retry_budget is not None`), never pre-flight.
        # Anthropic's server-side compaction bounds context in healthy
        # operation, so local trim is just the last-resort backstop. The ollama
        # sibling (conversation.py) has no server compaction and so trims every
        # turn instead. Intentional — see _trim_to_fit_window docstring.
        dropped = self._trim_to_fit_window() if self._retry_budget is not None else 0
        if dropped:
            print_ts(
                f"{COLOR_YELLOW}retry trim: dropped {dropped} oldest message(s) to fit smaller budget{COLOR_END}",
                agent=self.agent.id,
            )
            # Trim drops content without summarizing — follow with compaction.
            self.force_compact_next = True

        system_prompt, api_messages = _openflip_msgs_to_anthropic(
            self.messages, self.system_message or ""
        )

        # Inject queued image attachments — same as chat() path. See
        # _inject_pending_image_attachments() for shape. Pop the queue so
        # a failed request doesn't double-attach on retry.
        _pending_imgs = getattr(self, "_pending_image_attachments", None) or []
        if _pending_imgs:
            self._pending_image_attachments = []
            _injected = _inject_pending_image_attachments(api_messages, _pending_imgs)
            if _injected:
                print_ts(
                    f"  → injected {_injected} image attachment(s) into request (stream)",
                    agent=self.agent.id,
                )

        # If we have a stored compaction block from a previous turn, prepend
        # it to the messages so Anthropic sees the canonical summary.
        if self._compaction_block is not None:
            cb = self._compaction_block
            if isinstance(cb, dict) and cb.get("type") == "compaction":
                # The compaction block is itself a list of content blocks,
                # so we wrap it in a single-block assistant message and
                # prepend.
                api_messages = [
                    {"role": "assistant", "content": [cb]},
                ] + api_messages

        # REMINDER.md per-turn injection removed (2026-07-06, operator
        # decision): the mechanic was inconsistent and accreted hyper-specific
        # content paid uncached every turn. Standing guidance belongs in the
        # cached system files (AGENT.md / SOUL.md); agents that still have a
        # REMINDER.md deliver it via system_files instead.

        # Convert openflip tools to Anthropic JSON schema. Same call as
        # chat() — use the canonical helper that takes the list and
        # returns a list, not a per-tool builder.
        tool_schemas = _build_tool_schemas(tools or [])
        tools_map: dict[str, Callable] = {}
        for func in tools or []:
            tools_map[func.__name__] = func

        # Re-sync the model from the (possibly hot-reloaded) agent every turn
        # so `/model` and direct agent.json edits take effect on the NEXT turn
        # with no restart. self.model is otherwise cached at construction and
        # would go stale relative to agent.model after a reload_if_changed().
        # A per-turn override (set via set_model_override) wins for this turn
        # only — _effective_raw_model() returns it when present, else
        # agent.model.
        self.model = self._normalize_model(self._effective_raw_model())
        body = {
            "model": self.model,
            # Per-model output token cap → config.json models.<bare>.max_tokens.
            # Pass agent.model raw (with any `-1m` suffix); get_max_tokens does
            # the bare-name normalization, exactly like get_effort /
            # get_compaction_trigger. Falls back to 64000 when unset/invalid.
            "max_tokens": get_max_tokens(self.agent.model, "anthropic"),
            "messages": api_messages,
            "stream": True,
        }
        # Per-model reasoning-effort knob → Anthropic output_config.effort.
        # Omitted entirely when unset/invalid (API default "high"). Kept in
        # sync with the streaming path above. See config_global.get_effort.
        _effort = self._effort_level()
        if _effort:
            body["output_config"] = {"effort": _effort}

        # Server-side compaction opt-in. Manual /compact (force_compact_trigger_
        # override) sends the low _MANUAL_COMPACT_TRIGGER so Anthropic compacts
        # regardless of current input size; auto-compact and retry-trim leave the
        # override off and keep the real per-model trigger (only compact once
        # input actually exceeds it). Previously this always used the real
        # trigger, so a manual /compact on a sub-threshold conversation never
        # compacted — the documented no-op bug.
        if self.force_compact_next:
            _compact_trigger = (
                _MANUAL_COMPACT_TRIGGER if self.force_compact_trigger_override
                else get_compaction_trigger(self.agent.model, "anthropic")
            )
            body["context_management"] = {
                "edits": [{
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": _compact_trigger},
                }]
            }

        # Billing block (claude-cli routing) + cached system prompt.
        _BILLING_BLOCK = {
            "type": "text",
            "text": "x-anthropic-billing-header: cc_version=2.1.142; cc_entrypoint=sdk-cli; cch=00000;",
        }
        system_blocks = [_BILLING_BLOCK]
        if system_prompt:
            system_blocks.append({
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            })
        else:
            system_blocks[0]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        body["system"] = system_blocks

        if tool_schemas:
            body["tools"] = tool_schemas
            if tool_choice is not None:
                body["tool_choice"] = tool_choice

        # Extend caching past system prompt to conversation history: the
        # breakpoint lands on the newest user message so the whole history
        # prefix (including this turn's input) is a cache read next turn.
        if api_messages:
            last_user_idx = None
            scan_end = len(api_messages) - 1
            for i in range(scan_end, -1, -1):
                if api_messages[i].get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx is not None:
                last_user = api_messages[last_user_idx]
                if isinstance(last_user.get("content"), list) and last_user["content"]:
                    last_user["content"][-1]["cache_control"] = {
                        "type": "ephemeral", "ttl": "1h",
                    }
                elif isinstance(last_user.get("content"), str):
                    last_user["content"] = [
                        {
                            "type": "text",
                            "text": last_user["content"],
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }
                    ]

        # Headers — must mirror chat()'s beta flag set. The streaming path
        # previously omitted `claude-code-20250219`, `oauth-2025-04-20`, and
        # `compact-2026-01-12`, which caused 400s whenever `force_compact_next`
        # was True because `compact_20260112` requires the `compact-2026-01-12`
        # beta to be enabled. Keep this list in sync with chat()'s _beta_flags.
        _beta_flags = [
            "claude-code-20250219",
            "oauth-2025-04-20",
            "extended-cache-ttl-2025-04-11",
            "compact-2026-01-12",
        ]
        if self._wants_1m_context():
            _beta_flags.append("context-1m-2025-08-07")
        headers = {
            "authorization": f"Bearer {access_token}",
            "anthropic-version": _DEFAULT_ANTHROPIC_VERSION,
            "anthropic-beta": ",".join(_beta_flags),
            "User-Agent": _DEFAULT_USER_AGENT,
            "content-type": "application/json",
        }

        session = await self._ensure_http_session()

        # Request dump (same as chat()) — minimal vs full controlled by env.
        if os.environ.get("OPENFLIP_REQUEST_DUMP") == "1" or os.environ.get("OPENFLIP_REQUEST_DUMP_FULL") == "1":
            try:
                import time as _time
                _stamp = f"{self.agent.id}_{int(_time.time() * 1000)}"
                from .utils import project_root as _pr
                _dump_dir = os.path.join(_pr(), "data", "request_dumps")
                os.makedirs(_dump_dir, exist_ok=True)
                _dump_path = os.path.join(_dump_dir, f"{_stamp}.req.json")
                _safe_headers = {
                    k: v for k, v in headers.items()
                    if k.lower() not in ("authorization", "x-api-key")
                }
                if os.environ.get("OPENFLIP_REQUEST_DUMP_FULL") == "1":
                    _body_summary = body
                else:
                    _body_summary = {}
                    for _k in ("model", "max_tokens", "stream"):
                        if _k in body:
                            _body_summary[_k] = body[_k]
                    if api_messages:
                        _body_summary["last_message"] = api_messages[-1]
                    if self._compaction_block is not None:
                        _body_summary["has_compaction_block"] = True
                _payload = {
                    "url": f"{_DEFAULT_API_BASE}/v1/messages",
                    "headers": _safe_headers,
                    "body_summary": _body_summary,
                }
                with open(_dump_path, "w") as _df:
                    _json.dump(_payload, _df, indent=2)
                self._last_req_dump_stamp = _stamp
            except Exception as _dump_e:
                print_ts(f"{COLOR_YELLOW}request_dump failed: {_dump_e}{COLOR_END}", agent=self.agent.id)

        # Pre-flight request validation. Catches structural bugs (orphan
        # tool_use, oversized images, bad media types, etc.) before the
        # bytes leave the process. Fail-severity raises; warn-severity
        # logs and proceeds. See openflip/_request_validator.py.
        #
        # Kill switch: OPENFLIP_DISABLE_REQUEST_VALIDATOR=1 bypasses the
        # entire validator (no warns, no fails). Use if the validator
        # starts false-positiveing on legitimate traffic and there's
        # no time to fix it properly. Anthropic's own 400 still surfaces
        # via the normal error path.
        if os.environ.get("OPENFLIP_DISABLE_REQUEST_VALIDATOR") != "1":
            _vresult = _request_validator.validate_anthropic_request(body)
            for _vp in _vresult.warns():
                print_ts(
                    f"{COLOR_YELLOW}request validator WARN: {_vp}{COLOR_END}",
                    agent=self.agent.id,
                )
            if not _vresult.ok:
                for _vp in _vresult.fails():
                    print_ts(
                        f"{COLOR_RED}request validator FAIL: {_vp}{COLOR_END}",
                        agent=self.agent.id, error=True,
                    )
                raise MalformedRequestError(_vresult.fails())

        # POST + stream
        try:
            async with session.post(
                f"{_DEFAULT_API_BASE}/v1/messages",
                json=body,
                headers=headers,
            ) as resp:
                status = resp.status

                if status != 200:
                    # Read full error body for messaging.
                    text = await resp.text()
                    if status == 401:
                        # Try one refresh + recursive retry.
                        if _retry_attempt == 0:
                            print_ts(
                                f"{COLOR_YELLOW}Anthropic 401 (stream) — forcing token refresh and retrying{COLOR_END}",
                                agent=self.agent.id,
                            )
                            refreshed = await _load_oauth_access_token(force_refresh=True)
                            if refreshed and refreshed != access_token:
                                async for ev in self.chat_stream(
                                    tools=tools, think=think,
                                    tool_choice=tool_choice,
                                    _retry_attempt=_retry_attempt + 1,
                                ):
                                    yield ev
                                return
                        yield FrameworkErrorEvent(
                            message="⚠️ Anthropic OAuth token rejected (401). Try `claude` to re-login.",
                            kind="auth",
                        )
                        return
                    if status == 429:
                        # retry-after is authoritative — chat()'s transient
                        # retry loop waits it out (earlier retries are
                        # guaranteed to fail).
                        _ra = 0.0
                        try:
                            _ra = float(resp.headers.get("retry-after", "") or 0)
                        except (TypeError, ValueError):
                            _ra = 0.0
                        yield FrameworkErrorEvent(
                            message="⚠️ Anthropic rate limit (429). Subscription quota — wait and retry.",
                            kind="rate_limit",
                            status=429,
                            retry_after=_ra,
                        )
                        return
                    # 400 recovery paths — mirror chat()'s behavior.
                    # Recovery 1: stale compaction block in our stored
                    # state — Anthropic doesn't recognize it. Clear and
                    # retry once.
                    if (
                        status == 400
                        and _retry_attempt == 0
                        and self._compaction_block is not None
                        and "compact" in text.lower()
                    ):
                        print_ts(
                            f"{COLOR_YELLOW}clearing stale compaction block "
                            f"and retrying once (stream){COLOR_END}",
                            agent=self.agent.id,
                        )
                        self._compaction_block = None
                        try:
                            os.remove(self._meta_path())
                        except OSError:
                            pass
                        async for ev in self.chat_stream(
                            tools=tools, think=think,
                            tool_choice=tool_choice,
                            _retry_attempt=1,
                        ):
                            yield ev
                        return
                    # Recovery 2: prompt too long. Halve budget, retry
                    # up to 3 times. Each retry trims more aggressively.
                    if (
                        status == 400
                        and _retry_attempt < 3
                        and "prompt is too long" in text.lower()
                    ):
                        window = get_model_context_window(self.agent.model, "anthropic")
                        new_budget = max(window // (2 ** (_retry_attempt + 1)), 50_000)
                        print_ts(
                            f"{COLOR_YELLOW}prompt too long — retry "
                            f"{_retry_attempt + 1}/3 with budget {new_budget:,} (stream){COLOR_END}",
                            agent=self.agent.id,
                        )
                        self._retry_budget = new_budget
                        try:
                            async for ev in self.chat_stream(
                                tools=tools, think=think,
                                tool_choice=tool_choice,
                                _retry_attempt=_retry_attempt + 1,
                            ):
                                yield ev
                            return
                        finally:
                            self._retry_budget = None
                    # Transient server-side errors (529 overloaded, 5xx,
                    # 408) — nothing is wrong with the request; chat()'s
                    # retry loop backs off and re-sends.
                    if status in (408, 500, 502, 503, 504, 529):
                        print_ts(
                            f"{COLOR_YELLOW}Anthropic API {status} (transient): "
                            f"{text[:300]}{COLOR_END}",
                            agent=self.agent.id,
                        )
                        snippet = text[:200].replace("\n", " ")
                        yield FrameworkErrorEvent(
                            message=f"⚠️ Anthropic API {status} (server busy): {snippet}",
                            kind="overloaded",
                            status=status,
                        )
                        return
                    # Other non-200 (400/403/404/413/422 …) — structurally
                    # unsendable; terminal, never retried.
                    print_ts(
                        f"{COLOR_RED}Anthropic API {status}: {text[:600]}{COLOR_END}",
                        agent=self.agent.id, error=True,
                    )
                    snippet = text[:200].replace("\n", " ")
                    yield FrameworkErrorEvent(
                        message=f"⚠️ Anthropic API {status}: {snippet}",
                        kind="bad_request",
                        status=status,
                    )
                    return

                # Status 200 — consume the SSE stream and forward events.
                # Track stop_reason + final usage for /status persistence.
                _final_usage: dict = {}
                _final_stop_reason: str | None = None

                # try/finally so partial usage + force_compact_next
                # cleanup runs even if the stream dies mid-way. Otherwise
                # we leak force_compact_next=True into the next turn AND
                # lose partial usage data we already paid for. Caught
                # in spot-review.
                _stream_failed = False
                try:
                    async for ev in stream_sse_events(resp.content):
                        # Side-effect handling for specific event types,
                        # mirroring what chat() does post-collect.
                        if isinstance(ev, MessageStartEvent):
                            # Capture initial input_tokens — output is 0 here.
                            _final_usage = dict(ev.usage)
                        elif isinstance(ev, ContentBlockStopEvent):
                            # Compaction block detection: matches existing
                            # behavior in chat() at the post-collect stage.
                            blk = ev.completed_block
                            if blk.get("type") == "compaction":
                                self._compaction_block = {
                                    "type": "compaction",
                                    "content": blk.get("content", ""),
                                }
                                _c = self._compaction_block["content"]
                                summary_chars = len(_c) if isinstance(_c, (str, list)) else 0
                                print_ts(
                                    f"received compaction block ({summary_chars} chars/blocks of summary)",
                                    agent=self.agent.id,
                                )
                                self._save_meta()
                                self.compacted_this_turn = True
                                # _archive_and_trim_after_compaction runs
                                # AFTER the stream completes — see finally.
                        elif isinstance(ev, MessageDeltaEvent):
                            # Final usage + stop_reason
                            if ev.usage:
                                for k, v in ev.usage.items():
                                    _final_usage[k] = v
                            if ev.stop_reason is not None:
                                _final_stop_reason = ev.stop_reason
                        yield ev
                except Exception as _stream_e:
                    # Stream died mid-way. Mark failure but still run
                    # cleanup in `finally`.
                    _stream_failed = True
                    print_ts(
                        f"{COLOR_RED}Stream consumption error: {_stream_e}{COLOR_END}",
                        agent=self.agent.id, error=True,
                    )
                    yield FrameworkErrorEvent(
                        message=f"⚠️ Stream error: {_stream_e}",
                        kind="transport",
                    )
                finally:
                    # Persist partial-or-final usage if we got any. This
                    # also runs on stream failure — we paid for the
                    # tokens, the data has value for /status.
                    if _final_usage:
                        in_tok = _final_usage.get("input_tokens", 0)
                        out_tok = _final_usage.get("output_tokens", 0)
                        cache_create = _final_usage.get("cache_creation_input_tokens", 0)
                        cache_read = _final_usage.get("cache_read_input_tokens", 0)
                        self.last_usage = {
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                            "cache_creation_input_tokens": cache_create,
                            "cache_read_input_tokens": cache_read,
                            "total_input": in_tok + cache_create + cache_read,
                            "ts": time.time(),
                            # Mark partial when stream died mid-way so
                            # /status can flag "incomplete usage data."
                            "partial": _stream_failed,
                        }
                        try:
                            self._save_meta()
                        except Exception:
                            pass
                        print_ts(
                            f"anthropic usage: in={in_tok} out={out_tok} "
                            f"cache_create={cache_create} cache_read={cache_read} "
                            f"(stream{', partial' if _stream_failed else ''})",
                            agent=self.agent.id,
                        )

                        # Append a raw usage row to the durable ledger (one
                        # per completed turn). The conversation_id is the
                        # transport-prefixed key ("discord:<id>", "imessage:
                        # <chat_id>") — split it into transport + channel_id;
                        # it also serves as the canonical session key. The
                        # conversation object has no speaker identity, so
                        # user_id/user_handle are left None (nice-to-have).
                        # Wrapped defensively in addition to record_usage's
                        # own internal guard — a ledger write must NEVER break
                        # a turn.
                        try:
                            from . import usage_ledger
                            _conv_id = getattr(self, "conversation_id", "") or ""
                            if ":" in _conv_id:
                                _transport, _channel_id = _conv_id.split(":", 1)
                            else:
                                _transport, _channel_id = None, (_conv_id or None)
                            _outcome = ("error" if _stream_failed
                                        else ("compaction" if self.compacted_this_turn
                                              else "ok"))
                            usage_ledger.record_usage(
                                agent_id=self.agent.id,
                                transport=_transport,
                                channel_id=_channel_id,
                                session_id=_conv_id or None,
                                user_id=None,
                                user_handle=None,
                                model=self.model,
                                usage=_final_usage,
                                outcome=_outcome,
                            )
                        except Exception as _led_e:
                            print_ts(
                                f"usage_ledger record failed (continuing): {_led_e}",
                                agent=self.agent.id, error=True,
                            )

                    # Compaction trim — only when compaction succeeded
                    # this turn AND stream didn't fail. A partial stream
                    # that started a compaction but never finished it
                    # would leave _compaction_block in an inconsistent
                    # state; safer to skip trim.
                    if self.compacted_this_turn and not _stream_failed:
                        try:
                            self._archive_and_trim_after_compaction()
                        except Exception as _trim_e:
                            print_ts(
                                f"{COLOR_RED}compaction trim failed (continuing): {_trim_e}{COLOR_END}",
                                agent=self.agent.id, error=True,
                            )

                    # Consume force_compact_next (and the manual-trigger
                    # override) regardless of stream outcome. If we don't clear
                    # these, every future turn will fire the costly server-side
                    # compaction beta — and keep using the low manual trigger —
                    # even though the user only asked for it once.
                    if self.force_compact_next:
                        self.force_compact_next = False
                    self.force_compact_trigger_override = False

                # If stream failed, terminate the generator AFTER the
                # finally cleanup has run.
                if _stream_failed:
                    return

        except asyncio.TimeoutError:
            yield FrameworkErrorEvent(
                message="⚠️ Anthropic API stalled (no data for 90s).",
                kind="timeout",
            )
            return
        except Exception as e:
            print_ts(
                f"{COLOR_RED}chat_stream exception: {e}{COLOR_END}",
                agent=self.agent.id, error=True,
            )
            yield FrameworkErrorEvent(
                message=f"⚠️ Anthropic API error: {e}",
                kind="transport",
            )
            return
