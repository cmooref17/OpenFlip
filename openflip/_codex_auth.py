"""ChatGPT/Codex subscription OAuth credential manager.

Reads the token file `codex login` maintains at `$CODEX_HOME/auth.json`
(default `~/.codex/auth.json`) and keeps the access token fresh so the
"openai" provider can run on a ChatGPT subscription instead of a metered
API key. Protocol mirrored from simonw/llm-openai-via-codex (itself
extracted from the official openai/codex CLI):

  - auth.json shape: {"auth_mode": "chatgpt", "tokens": {"access_token",
    "refresh_token", "id_token", "account_id"}, "last_refresh": <ISO8601>}.
    Only auth_mode == "chatgpt" is supported.
  - Expiry comes from the access_token JWT's `exp` claim; refresh fires
    when now >= exp - 30s (_REFRESH_SKEW_S).
  - Refresh: POST https://auth.openai.com/oauth/token with the Codex
    CLI's client_id. A response `error` of refresh_token_expired /
    refresh_token_reused / refresh_token_invalidated means the creds are
    dead — the user must run `codex login` again.
  - Write-back is atomic (tmp + os.replace) with 0o600 on the result.

Refresh coordination mirrors anthropic_conversation's discipline: an
in-process asyncio.Lock + per-refresh-token in-flight Future dedup, plus
a cross-process file lock (with stale-lock breaking) next to auth.json
so openflip and the `codex` CLI (or a second openflip process) never
refresh the same single-use refresh_token concurrently. The lock uses
fcntl.flock on POSIX and msvcrt.locking on Windows (see _file_lock.py).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from datetime import datetime, timezone

from .utils import print_ts, COLOR_YELLOW, COLOR_RED, COLOR_END
from .config_global import get_codex_home

# Codex CLI's public OAuth client id + token endpoint (from the reference;
# constants of the official codex tooling, not secrets).
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_REFRESH_URL = "https://auth.openai.com/oauth/token"
_REFRESH_SKEW_S = 30
# `error` values that mean the refresh_token itself is dead — re-login only.
_DEAD_TOKEN_ERRORS = {
    "refresh_token_expired",
    "refresh_token_reused",
    "refresh_token_invalidated",
}

# In-process refresh coordination (mirrors anthropic_conversation):
# lazy-init asyncio.Lock + one in-flight Future per refresh_token so a
# thundering herd of agents coalesces into a single network refresh.
_REFRESH_LOCK: asyncio.Lock | None = None
_REFRESH_INFLIGHT: dict | None = None  # {"refresh_token": str, "future": asyncio.Future}

# If disk creds were written within this many seconds, another caller just
# refreshed successfully — return the disk token instead of refreshing again
# (even on force_refresh; catches herd losers AND transient-401 re-checks).
_RECENT_REFRESH_SHORTCIRCUIT_S = 5.0

# Dead-creds memo: once the refresh endpoint says the token is unrecoverable,
# fail fast with the same message until auth.json's mtime changes (i.e. the
# user ran `codex login` again) instead of hammering the endpoint every turn.
_DEAD: dict | None = None  # {"reason": str, "mtime": float}

# Cross-process file lock (fcntl flock) — same stale/retry policy as the
# anthropic provider's .oauth_refresh.lock. Sits next to auth.json.
_LOCK_STALE_S = 10.0
_LOCK_MAX_TRIES = 5
_LOCK_BASE_SLEEP_S = 1.0


class CodexAuthError(Exception):
    """Auth failure with a user-facing message (surfaced as a framework
    error by the provider — never silently swallowed)."""


def codex_auth_path() -> str:
    return os.path.join(get_codex_home(), "auth.json")


def _lock_path() -> str:
    return os.path.join(get_codex_home(), ".openflip_codex_refresh.lock")


def codex_creds_exist() -> bool:
    """True when usable subscription creds exist: auth.json is present,
    parses, and has auth_mode == "chatgpt". Drives auth-path precedence —
    an api-key-mode auth.json must NOT hijack the OPENAI_API_KEY fallback.
    """
    data = _load_auth()
    return bool(data) and data.get("auth_mode") == "chatgpt"


def _load_auth() -> dict | None:
    try:
        with open(codex_auth_path()) as f:
            return json.load(f)
    except Exception:
        return None


def _auth_mtime() -> float:
    try:
        return os.path.getmtime(codex_auth_path())
    except OSError:
        return 0.0


def _save_auth(data: dict) -> None:
    """Atomic write-back: tmp + os.replace, then chmod 0o600 (the file
    holds live OAuth tokens)."""
    path = codex_auth_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)
    os.chmod(path, 0o600)


def _jwt_exp(token: str) -> float | None:
    """Read the `exp` claim from a JWT without verifying it: base64url-decode
    the middle segment (padded to a multiple of 4) and parse the JSON."""
    try:
        seg = token.split(".")[1]
        seg += "=" * (-len(seg) % 4)
        payload = json.loads(base64.urlsafe_b64decode(seg))
        exp = payload.get("exp")
        return float(exp) if exp else None
    except Exception:
        return None


def _needs_refresh(data: dict) -> bool:
    access = ((data.get("tokens") or {}).get("access_token")) or ""
    if not access:
        return True
    exp = _jwt_exp(access)
    if exp is None:
        # Unreadable exp — treat as expired rather than risk sending a dead
        # token (the 401 path would force a refresh anyway).
        return True
    return time.time() >= (exp - _REFRESH_SKEW_S)


# ── Cross-process file lock (fcntl on POSIX, msvcrt on Windows) ──

def _acquire_refresh_file_lock():
    """Returns the open lock fd, or None if unavailable (fs issue /
    contention / platform with no locking backend). Stale locks older than
    _LOCK_STALE_S are forcibly broken — the holder must have crashed
    mid-refresh."""
    from . import _file_lock
    if not _file_lock.LOCKING_SUPPORTED:
        return None

    import random

    lock_path = _lock_path()
    for attempt in range(_LOCK_MAX_TRIES):
        try:
            mtime = os.path.getmtime(lock_path)
            if (time.time() - mtime) > _LOCK_STALE_S:
                try:
                    os.remove(lock_path)
                    print_ts(
                        f"{COLOR_YELLOW}Codex refresh: removed stale lock file "
                        f"(mtime {time.time() - mtime:.1f}s old){COLOR_END}",
                    )
                except OSError:
                    pass  # concurrent removal is fine
        except OSError:
            pass  # lock file absent — normal

        try:
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as e:
            print_ts(f"{COLOR_YELLOW}Codex refresh: lock file open failed: {e}{COLOR_END}")
            return None

        try:
            acquired = _file_lock.try_lock_excl(fd)
        except OSError as e:
            try:
                os.close(fd)
            except OSError:
                pass
            print_ts(f"{COLOR_YELLOW}Codex refresh: file lock failed unexpectedly: {e}{COLOR_END}")
            return None
        if acquired:
            try:
                os.utime(lock_path, None)  # fresh mtime = alive for stale checks
            except OSError:
                pass
            return fd
        else:
            try:
                os.close(fd)
            except OSError:
                pass
            sleep_s = _LOCK_BASE_SLEEP_S + random.random()
            print_ts(
                f"Codex refresh: cross-process lock held (attempt {attempt + 1}/"
                f"{_LOCK_MAX_TRIES}, sleeping {sleep_s:.2f}s)"
            )
            time.sleep(sleep_s)
    print_ts(
        f"{COLOR_YELLOW}Codex refresh: failed to acquire cross-process lock "
        f"after {_LOCK_MAX_TRIES} attempts{COLOR_END}",
    )
    return None


def _release_refresh_file_lock(fd) -> None:
    if fd is None:
        return
    from . import _file_lock
    _file_lock.unlock(fd)
    try:
        os.close(fd)
    except OSError:
        pass


def _refresh_sync(data: dict) -> dict:
    """Blocking token refresh (run in an executor). Returns the updated
    auth.json dict after persisting it. Raises CodexAuthError on failure;
    dead-token errors also set the _DEAD memo so subsequent turns fail fast
    until the user re-logs-in."""
    global _DEAD
    tokens = dict(data.get("tokens") or {})
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise CodexAuthError(
            "Codex auth.json has no refresh_token — run `codex login` again."
        )

    import urllib.error
    import urllib.request

    body = json.dumps({
        "client_id": _CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }).encode("utf-8")
    req = urllib.request.Request(
        _REFRESH_URL, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        err_code = ""
        try:
            err_body = json.loads(e.read().decode("utf-8", errors="replace"))
            err = err_body.get("error")
            # `error` is usually a bare string; tolerate the object form.
            err_code = err if isinstance(err, str) else \
                (err or {}).get("code") or (err or {}).get("type") or ""
        except Exception:
            pass
        if err_code in _DEAD_TOKEN_ERRORS:
            reason = (
                f"Codex refresh token rejected ({err_code}) — the subscription "
                f"login is dead. Run `codex login` again."
            )
            _DEAD = {"reason": reason, "mtime": _auth_mtime()}
            print_ts(f"{COLOR_RED}{reason}{COLOR_END}", error=True)
            raise CodexAuthError(reason)
        raise CodexAuthError(
            f"Codex token refresh failed (HTTP {e.code}"
            f"{', error=' + err_code if err_code else ''})."
        )
    except Exception as e:
        raise CodexAuthError(f"Codex token refresh failed: {e}")

    if not payload.get("access_token"):
        # Log only the KEYS — a partial response can still carry tokens and
        # log.txt is append-only and unrestricted.
        print_ts(
            f"{COLOR_RED}Codex refresh: response missing access_token; "
            f"keys={sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__}"
            f"{COLOR_END}",
            error=True,
        )
        raise CodexAuthError("Codex token refresh returned no access_token.")

    # Update each token the response carried; absent fields keep old values
    # (the endpoint may omit refresh_token when it doesn't rotate).
    for key in ("access_token", "id_token", "refresh_token"):
        if payload.get(key):
            tokens[key] = payload[key]
    new_data = dict(data)
    new_data["tokens"] = tokens
    new_data["last_refresh"] = datetime.now(timezone.utc).isoformat()
    _save_auth(new_data)
    exp = _jwt_exp(tokens.get("access_token") or "")
    mins = int((exp - time.time()) // 60) if exp else "?"
    print_ts(f"Codex OAuth: refreshed token (expires in {mins}min)")
    return new_data


def _refresh_with_file_lock(data: dict) -> dict:
    """Acquire the cross-process flock, RE-READ auth.json (another process —
    the codex CLI, a second openflip — may have just refreshed), and only hit
    the network if a refresh is still needed. Lock released in finally."""
    fd = _acquire_refresh_file_lock()
    # On lock failure (Windows / contention) refresh anyway — the in-process
    # asyncio.Lock dedup still prevents intra-process races.
    try:
        if fd is not None:
            fresh = _load_auth()
            if fresh and (fresh.get("tokens") or {}).get("access_token"):
                disk_age = time.time() - _auth_mtime()
                if disk_age < _RECENT_REFRESH_SHORTCIRCUIT_S:
                    print_ts(
                        f"{COLOR_YELLOW}Codex refresh: disk creds written "
                        f"{disk_age:.1f}s ago — skipping network call{COLOR_END}",
                    )
                    return fresh
                # Skip only if the token CHANGED vs what we entered with AND
                # is unexpired — an unchanged-but-valid-looking token may be
                # exactly the one that just 401'd (force_refresh path).
                old_access = (data.get("tokens") or {}).get("access_token")
                fresh_access = (fresh.get("tokens") or {}).get("access_token")
                if fresh_access != old_access and not _needs_refresh(fresh):
                    print_ts(
                        f"{COLOR_YELLOW}Codex refresh: creds refreshed by another "
                        f"process while we waited for lock — skipping network call{COLOR_END}",
                    )
                    return fresh
                # Use the fresh-from-disk creds so we send the LATEST
                # refresh_token — it may have rotated (they're single-use).
                data = fresh
        return _refresh_sync(data)
    finally:
        _release_refresh_file_lock(fd)


async def borrow_codex_key(force_refresh: bool = False) -> tuple[str, str]:
    """Return (access_token, account_id) for subscription requests,
    refreshing first when the JWT is within 30s of expiry (or on
    force_refresh — the 401-retry path). Raises CodexAuthError with a
    user-facing message on any failure; never returns a known-expired token.
    """
    global _REFRESH_LOCK, _REFRESH_INFLIGHT, _DEAD
    if _REFRESH_LOCK is None:
        # Lazy-init: module import happens before any event loop exists.
        _REFRESH_LOCK = asyncio.Lock()

    data = _load_auth()
    if not data:
        raise CodexAuthError(
            f"No Codex credentials at {codex_auth_path()} — run `codex login` "
            f"with your ChatGPT subscription (or set CODEX_HOME)."
        )
    if data.get("auth_mode") != "chatgpt":
        raise CodexAuthError(
            f"Codex auth.json has auth_mode={data.get('auth_mode')!r}; only "
            f"\"chatgpt\" (subscription OAuth) is supported. Run `codex login` "
            f"with the ChatGPT-subscription flow."
        )

    # Dead-creds fast-fail: the refresh endpoint already told us this
    # refresh_token is unrecoverable. A changed auth.json mtime means the
    # user re-logged-in, so clear the memo and proceed.
    if _DEAD is not None:
        if _auth_mtime() != _DEAD.get("mtime"):
            _DEAD = None
        else:
            raise CodexAuthError(_DEAD["reason"])

    tokens = data.get("tokens") or {}
    access = tokens.get("access_token") or ""
    account_id = str(tokens.get("account_id") or "")

    if not force_refresh and not _needs_refresh(data):
        return access, account_id

    # Recent-success short-circuit: another caller just refreshed and wrote
    # disk — trust that work (applies even on force_refresh; the 401 that
    # brought us here was likely already handled or transient).
    disk_age = time.time() - _auth_mtime()
    if disk_age < _RECENT_REFRESH_SHORTCIRCUIT_S:
        fresh = _load_auth() or {}
        fresh_tokens = fresh.get("tokens") or {}
        if fresh_tokens.get("access_token"):
            print_ts(
                f"Codex refresh: skipping — disk creds written {disk_age:.1f}s "
                f"ago by another caller"
            )
            return fresh_tokens["access_token"], str(fresh_tokens.get("account_id") or "")

    # Dedup: one network refresh per refresh_token; concurrent callers await
    # the winner's Future and then re-read disk.
    current_refresh_token = tokens.get("refresh_token")
    fut: asyncio.Future | None = None
    is_winner = False
    async with _REFRESH_LOCK:
        if (_REFRESH_INFLIGHT is not None
                and _REFRESH_INFLIGHT.get("refresh_token") == current_refresh_token
                and not _REFRESH_INFLIGHT["future"].done()):
            fut = _REFRESH_INFLIGHT["future"]
            print_ts("Codex refresh: dedup — awaiting in-flight refresh for shared token")
        else:
            fut = asyncio.get_running_loop().create_future()
            _REFRESH_INFLIGHT = {"refresh_token": current_refresh_token, "future": fut}
            is_winner = True
            print_ts("Codex refresh: dedup — kicking off new refresh")

    if is_winner:
        # _refresh_with_file_lock blocks (urllib + lock-wait) — run in the
        # thread executor so the event loop stays live.
        try:
            refreshed = await asyncio.get_running_loop().run_in_executor(
                None, _refresh_with_file_lock, data,
            )
            fut.set_result(refreshed)
        except Exception as e:
            fut.set_exception(e)
        finally:
            async with _REFRESH_LOCK:
                if _REFRESH_INFLIGHT is not None and _REFRESH_INFLIGHT["future"] is fut:
                    _REFRESH_INFLIGHT = None
        exc = fut.exception()
        if exc is not None:
            if isinstance(exc, CodexAuthError):
                raise exc
            raise CodexAuthError(f"Codex token refresh failed: {exc}")
        new_tokens = (fut.result() or {}).get("tokens") or {}
        new_access = new_tokens.get("access_token")
        if not new_access:
            raise CodexAuthError("Codex token refresh produced no access_token.")
        return new_access, str(new_tokens.get("account_id") or account_id)

    # Loser path: await the winner, then re-read disk (the winner persists
    # BEFORE resolving the future, so disk is already fresh here).
    try:
        await asyncio.wait_for(fut, timeout=45.0)
    except CodexAuthError:
        raise
    except asyncio.TimeoutError:
        raise CodexAuthError("Codex token refresh timed out waiting for in-flight refresh.")
    except Exception as e:
        raise CodexAuthError(f"Codex token refresh failed: {e}")
    fresh = _load_auth() or {}
    fresh_tokens = fresh.get("tokens") or {}
    fresh_access = fresh_tokens.get("access_token")
    if not fresh_access:
        raise CodexAuthError("Codex token refresh failed (no token on disk after refresh).")
    return fresh_access, str(fresh_tokens.get("account_id") or account_id)
