"""Claude subscription plan limits — the same numbers Claude Code's `/usage` shows.

Reads `GET {api_base}/api/oauth/usage` with the anthropic provider's OAuth login
(the same token every anthropic agent turn uses). Returns the session (5h) and
weekly limit bars with their reset times. Used by the owner-only `/limits`
slash command. Read-only: it never changes anything on the account.
"""
from __future__ import annotations

import datetime as _dt

import aiohttp

# Human labels for the known limit buckets. Anything not listed is shown with
# its raw key, so a new bucket Anthropic adds still appears (just less pretty).
_LABELS = {
    "five_hour": "Session (5h)",
    "seven_day": "Weekly (all models)",
    "seven_day_opus": "Weekly (Opus)",
    "seven_day_sonnet": "Weekly (Sonnet)",
    "seven_day_oauth_apps": "Weekly (OAuth apps)",
    "seven_day_cowork": "Weekly (Cowork)",
}


async def fetch_plan_usage() -> dict:
    """Return the raw JSON from /api/oauth/usage. Raises RuntimeError on failure."""
    from .anthropic_conversation import (
        _load_oauth_access_token, _DEFAULT_API_BASE, _DEFAULT_ANTHROPIC_VERSION,
        _DEFAULT_USER_AGENT,
    )
    token = await _load_oauth_access_token()
    if not token:
        raise RuntimeError("no Claude OAuth login found (log in with Claude Code first)")
    headers = {
        "authorization": f"Bearer {token}",
        "anthropic-version": _DEFAULT_ANTHROPIC_VERSION,
        "anthropic-beta": "oauth-2025-04-20",
        "User-Agent": _DEFAULT_USER_AGENT,
    }
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{_DEFAULT_API_BASE}/api/oauth/usage", headers=headers,
                         timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status != 200:
                body = (await r.text())[:300]
                raise RuntimeError(f"HTTP {r.status}: {body}")
            return await r.json(content_type=None)


def _bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(100.0, pct))
    full = round(pct / 100 * width)
    return "█" * full + "░" * (width - full)


def _reset_text(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        t = _dt.datetime.fromisoformat(iso).astimezone()   # machine-local time
    except ValueError:
        return f"resets {iso}"
    now = _dt.datetime.now().astimezone()
    left = t - now
    hrs = max(0, int(left.total_seconds() // 3600))
    mins = max(0, int(left.total_seconds() % 3600 // 60))
    when = t.strftime("%a %b %-d, %-I:%M %p") if left.days >= 1 else t.strftime("%-I:%M %p")
    return f"resets {when} (in {hrs}h {mins}m)"


def format_plan_usage(data: dict) -> str:
    """Discord-ready text: one bar per non-empty limit bucket."""
    lines = ["**Claude plan usage**", "```"]
    rows = []
    for key, val in data.items():
        if not isinstance(val, dict) or "utilization" not in val:
            continue
        pct = val.get("utilization")
        if pct is None:
            continue
        label = _LABELS.get(key, key)
        locked = val.get("locked_reason")
        rows.append((label, float(pct), _reset_text(val.get("resets_at")), locked))
    if not rows:
        lines.append("no limit data returned")
    w = max([len(r[0]) for r in rows] + [10])
    for label, pct, reset, locked in rows:
        lines.append(f"{label.ljust(w)}  {_bar(pct)} {pct:5.1f}%")
        if reset:
            lines.append(f"{'':{w}}  {reset}")
        if locked:
            lines.append(f"{'':{w}}  LOCKED: {locked}")
    lines.append("```")
    extra = data.get("extra_usage") or {}
    if isinstance(extra, dict) and extra.get("is_enabled"):
        used, lim = extra.get("used_credits"), extra.get("monthly_limit")
        lines.append(f"Extra usage: on (used {used} of {lim})")
    return "\n".join(lines)
