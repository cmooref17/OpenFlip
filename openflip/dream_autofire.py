"""Dream auto-fire — the trigger layer for silent memory consolidation.

The dream() tool (tools/dream_tool.py) + the /dream command already do the
CONSOLIDATION work. This module is ONLY the auto-trigger that decides WHEN a
dream fires on its own — it changes nothing about what a dream DOES.

Design (mirrors Claude Code's autoDream): the check is event-driven, fired at
the end of a cleanly-completed, top-level, operator-driven turn (see the hook
in runtime._run_turn). It is NOT a clock/cron — that's what made kairos
annoying. Synthetic / subagent / chain turns are excluded by the caller so the
dream's own synthetic turn can never recursively trigger another dream.

Gates are checked cheapest-first and bail on the first failure:
    1. dream.enabled is true for the agent             (one dict read)
    2. IDLE   — the gap PRECEDING this turn ≥ min_idle_minutes (timestamp cmp)
    3. throttle — skip the filesystem stats if we scanned < SCAN_BACKOFF ago
    4. COOLDOWN — last dream ≥ 24h ago                 (one stat: marker mtime)
    5. NEW MATERIAL — a daily log is newer than the marker (stat: log mtimes)
    6. LOCK   — atomically acquire the dream lock; bail if another pass holds it

Only if everything passes do we fire a /dream-equivalent SILENT synthetic turn.
A dream NEVER posts to Discord (auto_post_final_text=False, silent=True).

Files (per agent, under agents/<id>/memory/):
    .dream_marker  — mtime == "last time a dream fired". Drives the COOLDOWN
                     and NEW-MATERIAL gates. Touched to now() at fire time.
    .dream.lock    — existence == "a dream pass is in the critical section".
                     Created with O_CREAT|O_EXCL (truly atomic), removed in a
                     finally. This is what makes two turns ending near-
                     simultaneously unable to BOTH fire a dream.

RACE NOTE (the load-bearing bit): the marker and the lock are SEPARATE files
on purpose. The marker must persist across days (so it can't double as an
O_EXCL lock — an existing file always fails O_EXCL). The lock is the true
mutex: of two racing turns, only one's os.open(O_EXCL) succeeds; the loser
gets FileExistsError and bails WITHOUT firing. The winner updates the marker
to now() inside the locked section, so any later turn that already passed the
cooldown read against the stale marker still can't fire — it loses the lock,
and the next turn after that reads the fresh marker and fails cooldown.
"""
from __future__ import annotations

import os
import re
import time

from .utils import print_ts, COLOR_YELLOW, COLOR_END


# Back off this long between filesystem gate-checks per channel. Once the cheap
# idle gate passes we stat the memory dir; without this, a returning operator
# firing several messages in a row would re-stat on every turn.
_SCAN_BACKOFF_S = 600          # ~10 min, matches Claude Code's autoDream backoff
_COOLDOWN_S = 24 * 60 * 60     # don't dream more than once a day
# A dream pass only HOLDS the lock for the few ms it takes to enqueue the
# synthetic turn + touch the marker (the actual consolidation runs async and
# does NOT hold the lock). Anything older than this is a crashed/killed pass —
# steal it so a hard kill can't wedge dreaming forever.
_STALE_LOCK_S = 30 * 60

_MARKER_NAME = ".dream_marker"
_LOCK_NAME = ".dream.lock"

_DAILY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")


# ── runner-attached state (lazily created so runtime.py needs no __init__ edit) ──

def _state(runner, attr: str) -> dict:
    d = getattr(runner, attr, None)
    if d is None:
        d = {}
        setattr(runner, attr, d)
    return d


# ── filesystem helpers ──────────────────────────────────────────────────────

def _memory_dir(agent_dir: str) -> str:
    return os.path.join(agent_dir, "memory")


def _marker_path(agent_dir: str) -> str:
    return os.path.join(_memory_dir(agent_dir), _MARKER_NAME)


def _lock_path(agent_dir: str) -> str:
    return os.path.join(_memory_dir(agent_dir), _LOCK_NAME)


def _mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _newest_daily_log_mtime(agent_dir: str) -> float | None:
    """Newest mtime among the agent's YYYY-MM-DD.md daily logs, or None if there
    are none. The marker/lock dotfiles and index.json are excluded by the
    date-pattern filter, so they can't masquerade as "new material"."""
    mem_dir = _memory_dir(agent_dir)
    newest: float | None = None
    try:
        names = os.listdir(mem_dir)
    except OSError:
        return None
    for name in names:
        if not _DAILY_RE.match(name):
            continue
        m = _mtime(os.path.join(mem_dir, name))
        if m is not None and (newest is None or m > newest):
            newest = m
    return newest


def _touch(path: str) -> None:
    """Create the file if missing and bump its mtime to now."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8"):
            pass
        os.utime(path, None)
    except OSError as e:
        print_ts(f"{COLOR_YELLOW}dream auto-fire: failed to touch {path}: {e}{COLOR_END}")


def _acquire_lock(path: str) -> bool:
    """Atomically acquire the dream lock. Returns True iff WE created it.

    os.open with O_CREAT|O_EXCL is atomic at the OS level: of N racing callers
    exactly one succeeds, the rest get FileExistsError. A pre-existing lock
    older than _STALE_LOCK_S is treated as a crashed pass and stolen (remove +
    retry once, still atomic)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        return False
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        return True
    except FileExistsError:
        age = None
        m = _mtime(path)
        if m is not None:
            age = time.time() - m
        if age is None or age <= _STALE_LOCK_S:
            return False  # held by a live pass — bail
        # Stale lock from a crashed/killed pass — steal it.
        try:
            os.remove(path)
        except OSError:
            return False
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            print_ts(f"{COLOR_YELLOW}dream auto-fire: stole stale lock {path} "
                     f"(age {int(age)}s){COLOR_END}")
            return True
        except OSError:
            return False
    except OSError:
        return False


def _release_lock(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ── firing ──────────────────────────────────────────────────────────────────

async def _fire_dream(runner, channel) -> None:
    """Fire a /dream-equivalent SILENT synthetic turn. Reuses the existing
    dream() tool + 4-phase consolidation prompt (no duplicated logic) and
    NEVER posts to Discord (auto_post_final_text=False, silent=True).

    The turn is attributed to the owner so owner-bypass in acl makes dream()
    and update_core_memory() callable even when they aren't in the agent's
    allowed_tools — same trick /dream uses."""
    from .config_global import get_owner_id
    owner_id = get_owner_id("discord")
    ch_id = int(getattr(channel, "id", 0) or 0)
    # Same prompt as the /dream slash command — keep them in lockstep.
    prompt = (
        "It's time to dream — consolidate your long-term memory now. "
        "Call the dream() tool, then follow its 4-phase instructions: "
        "distill durable facts, convert relative dates to absolute, prune "
        "facts that were later contradicted, and write the cleaned-up "
        "result back with update_core_memory(). Do this silently — do not "
        "message anyone about it."
    )
    await runner.run_synthetic_turn(
        ch_id,
        prompt,
        auto_post_final_text=False,   # silent: never post the result to Discord
        silent=True,
        speaker_id=int(owner_id),     # owner attribution → owner-bypass for dream()
        originator_visibility="dream",
    )


# ── entry point (called from runtime._run_turn end-of-turn hook) ─────────────

async def maybe_fire_dream(runner, channel) -> None:
    """Run the gate-check at the end of an eligible turn and fire a dream if all
    gates pass. Fully self-guarded — never raises into the turn. The caller has
    already established that this is a cleanly-completed, NON-synthetic,
    NON-subagent, top-level turn (so a dream's own synthetic turn can't recurse
    back into here)."""
    agent = runner.agent

    # Gate 1: enabled (cheapest — one dict read).
    cfg = getattr(agent, "dream", None) or {}
    if not cfg.get("enabled"):
        return

    ch_id = int(getattr(channel, "id", 0) or 0)
    if not ch_id:
        return
    now = time.time()

    # Per-channel "last operator activity" tracking. We record THIS operator
    # turn now, but compare against the PREVIOUS one so the idle gate measures
    # the gap that PRECEDED this turn — i.e. a dream fires when the operator
    # returns after an idle stretch, the only boundary an event-driven check
    # can observe (during true idleness no turns fire at all).
    activity = _state(runner, "_dream_last_activity")
    prev_activity = activity.get(ch_id)
    activity[ch_id] = now

    # Gate 2: IDLE. The first turn we ever see on a channel has no previous
    # timestamp to judge against — record it and wait for the next one.
    if prev_activity is None:
        return
    min_idle_s = int(cfg.get("min_idle_minutes", 120)) * 60
    if (now - prev_activity) < min_idle_s:
        return  # mid-conversation — don't dream

    # Gate 3: scan throttle. Past the cheap gates and about to hit the
    # filesystem; back off so repeated idle-boundary turns don't re-stat.
    scans = _state(runner, "_dream_last_scan")
    if (now - scans.get(ch_id, 0.0)) < _SCAN_BACKOFF_S:
        return
    scans[ch_id] = now

    agent_dir = os.path.dirname(agent.path)
    marker = _marker_path(agent_dir)
    marker_mtime = _mtime(marker)

    # Gate 4: COOLDOWN — at most one dream per 24h.
    if marker_mtime is not None and (now - marker_mtime) < _COOLDOWN_S:
        return

    # Gate 5: NEW MATERIAL — only dream if a daily log changed since last dream.
    newest = _newest_daily_log_mtime(agent_dir)
    if newest is None:
        return  # no daily logs at all → nothing to consolidate
    if marker_mtime is not None and newest <= marker_mtime:
        return  # nothing new since the last dream

    # Gate 6: LOCK — atomic. Of two turns racing here, exactly one wins.
    lock = _lock_path(agent_dir)
    if not _acquire_lock(lock):
        return
    try:
        await _fire_dream(runner, channel)
        # Record "last dreamt = now" INSIDE the locked section. This is the
        # 24h dedup baseline: any later turn reads this fresh mtime and fails
        # the cooldown gate, so even after the lock is released no second dream
        # fires. Touched only after a successful enqueue so an enqueue failure
        # doesn't silently burn a day.
        _touch(marker)
        print_ts(f"{COLOR_YELLOW}dream auto-fire: consolidation turn fired "
                 f"(idle {int((now - prev_activity) / 60)}m){COLOR_END}",
                 agent=agent.id)
        try:
            from . import events_log as _events_log
            _events_log.log_event(agent.id, "dream_autofire",
                                  channel_id=ch_id,
                                  idle_minutes=int((now - prev_activity) / 60))
        except Exception:
            pass
    finally:
        # Crash/exception must not wedge the lock forever.
        _release_lock(lock)
