"""Cross-platform, cross-process advisory file locking on an open fd.

POSIX (Linux/macOS): `fcntl.flock` — the exact mechanism Claude Code itself
uses for `.oauth_refresh.lock`, so openflip and the `claude` CLI contend on
the same lock correctly. Windows: `msvcrt.locking` over the first byte of
the file — mandatory-ish region lock, released on unlock or process death.

Both auth modules (`anthropic_conversation`, `_codex_auth`) route their
refresh-lock acquire/release through here so the stale-break + retry policy
lives in ONE place per caller while the OS-specific syscall lives here.

Callers hold the lock by keeping the fd open; closing the fd (or dying)
releases it on every platform.
"""
from __future__ import annotations

import os

try:
    import fcntl as _fcntl  # POSIX
    _BACKEND = "fcntl"
except ImportError:
    try:
        import msvcrt as _msvcrt  # Windows
        _BACKEND = "msvcrt"
    except ImportError:
        _BACKEND = ""

# False only on exotic platforms with neither fcntl nor msvcrt — callers
# should skip cross-process coordination entirely (in-process asyncio.Lock
# dedup still applies) rather than burn retries on a lock that can't exist.
LOCKING_SUPPORTED = bool(_BACKEND)


def try_lock_excl(fd: int) -> bool:
    """Non-blocking exclusive lock on `fd`.

    Returns True when acquired, False when another process holds it.
    Raises OSError on unexpected failure (bad fd, fs that doesn't support
    locking, etc.) so callers can log-and-bail rather than spin.
    """
    import errno
    if _BACKEND == "fcntl":
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            return True
        except OSError as e:
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                return False
            raise
    if _BACKEND == "msvcrt":
        # Lock byte 0 (locking a region beyond EOF is legal on Windows, so
        # a zero-byte lock file works). Region must match at unlock time.
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
            return True
        except OSError as e:
            # Contention surfaces as EACCES (PermissionError) or EDEADLK.
            if e.errno in (errno.EACCES, errno.EDEADLK):
                return False
            raise
    return False


def unlock(fd: int) -> None:
    """Release the lock taken by try_lock_excl. Never raises — the caller
    is about to close the fd anyway, which releases on every platform."""
    try:
        if _BACKEND == "fcntl":
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        elif _BACKEND == "msvcrt":
            os.lseek(fd, 0, os.SEEK_SET)
            _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
