"""Append-only SQLite ledger of per-turn Anthropic API usage.

The ledger is the product: every completed turn writes ONE raw row carrying
the full token breakdown + an estimated cost, and the entire usage dict is
stored verbatim in `raw_usage` so no field is ever lost even if Anthropic
adds new counters. Nothing is aggregated at write-time — every view (the
`/usage` slash command, ad-hoc queries) is a read-side query over these rows.

Design rules (see CLAUDE.md):
  - Writes must be bulletproof: a failure recording usage NEVER propagates up
    and breaks a turn. The whole of record_usage() is wrapped in try/except.
  - SQLite calls are synchronous and fast (one INSERT per turn, low rate);
    no event-loop concern. A per-write connection (check_same_thread=False)
    keeps concurrency trivially correct under WAL.
  - 30-day retention is self-cleaning: record_usage opportunistically purges
    old rows, throttled to at most once per process-hour.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

from .config_global import get_config
from .utils import print_ts, resolve_path


# ── Pricing ──────────────────────────────────────────────────────────────
# USD per 1,000,000 tokens. Matched by SUBSTRING against the model field
# (model names look like 'claude-opus-4-8' / 'claude-sonnet-4-6'; the
# normalized form drops the provider prefix and the -1m suffix, but we match
# loosely so either form works). First matching entry wins, so order from
# most-specific to least.
#
# Cache multipliers follow Anthropic's published rates:
#   cache_read  = 0.1x  input  (cached input is 10% of base)
#   cache_write = 1.25x input  (5-minute cache creation is 125% of base)
#
# ⚠️ RATES AS OF 2026-06-08 — these are hand-maintained. When Anthropic
# changes pricing, update this table. No match → est_cost_usd = 0.0 (we do
# NOT guess a price for an unknown model).
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Claude Opus 4.x — $15 in / $75 out per 1M.
    "opus": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
    # Claude Sonnet 4.x — $3 in / $15 out per 1M (base <=200k tier).
    "sonnet": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
    # Claude Haiku 4.5 — $1 in / $5 out per 1M.
    "haiku": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25},
}

_GROUP_BY_WHITELIST = {"agent_id", "channel_id", "session_id", "user_id", "model"}

_RETENTION_SECONDS = 30 * 86400
# Purge throttle: only actually run the DELETE at most once per this many
# seconds (process-lifetime guard). Cheap: a single int compare on the hot path.
_PURGE_INTERVAL_SECONDS = 3600
_last_purge_ts: float = 0.0

_schema_ready = False


def _db_path() -> str:
    data_dir = get_config().get("data_dir") or "./data"
    return os.path.join(resolve_path(data_dir), "usage_ledger.db")


def _connect() -> sqlite3.Connection:
    """Open a fresh connection. WAL mode makes concurrent appends safe; a
    per-write connection (vs a shared one + lock) is the simplest correct
    option at one INSERT per turn."""
    path = _db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    global _schema_ready
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            iso_ts TEXT NOT NULL,
            agent_id TEXT,
            transport TEXT,
            channel_id TEXT,
            session_id TEXT,
            user_id TEXT,
            user_handle TEXT,
            model TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_creation_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            est_cost_usd REAL DEFAULT 0.0,
            outcome TEXT,
            raw_usage TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);
        CREATE INDEX IF NOT EXISTS idx_usage_agent ON usage(agent_id);
        CREATE INDEX IF NOT EXISTS idx_usage_channel ON usage(channel_id);
        """
    )
    conn.commit()
    _schema_ready = True


def _price_for_model(model: str | None) -> dict[str, float] | None:
    if not model:
        return None
    m = model.lower()
    for key, rates in MODEL_PRICING.items():
        if key in m:
            return rates
    return None


def _estimate_cost(model: str | None, input_tokens: int, output_tokens: int,
                   cache_read: int, cache_creation: int) -> float:
    rates = _price_for_model(model)
    if rates is None:
        return 0.0
    cost = (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_read * rates["cache_read"]
        + cache_creation * rates["cache_write"]
    ) / 1_000_000.0
    return round(cost, 6)


def _maybe_purge(conn: sqlite3.Connection, now: float) -> None:
    """Opportunistic, throttled retention sweep. Runs at most once per
    _PURGE_INTERVAL_SECONDS per process — keeps the ledger self-cleaning
    without a separate cron and without DELETE-ing on every turn."""
    global _last_purge_ts
    if now - _last_purge_ts < _PURGE_INTERVAL_SECONDS:
        return
    _last_purge_ts = now
    try:
        cutoff = now - _RETENTION_SECONDS
        cur = conn.execute("DELETE FROM usage WHERE ts < ?", (cutoff,))
        conn.commit()
        if cur.rowcount:
            print_ts(f"usage_ledger: purged {cur.rowcount} rows older than 30d")
    except Exception as e:
        print_ts(f"usage_ledger: purge failed (continuing): {e}", error=True)


def record_usage(*, agent_id: str | None, transport: str | None,
                 channel_id: str | None, session_id: str | None,
                 user_id: str | None, user_handle: str | None,
                 model: str | None, usage: dict,
                 outcome: str | None = None) -> None:
    """Append one raw usage row for a completed turn.

    `usage` uses the same keys anthropic_conversation.py already reads:
    input_tokens, output_tokens, cache_read_input_tokens,
    cache_creation_input_tokens. The full dict is stored verbatim in
    raw_usage so no future counter is ever lost.

    Fully exception-safe: any failure is logged and swallowed. Recording
    usage must NEVER break a turn.
    """
    try:
        usage = usage or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
        total_tokens = input_tokens + output_tokens + cache_read + cache_creation
        est_cost = _estimate_cost(model, input_tokens, output_tokens,
                                  cache_read, cache_creation)

        now = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))

        try:
            raw = json.dumps(usage, default=str)
        except Exception:
            raw = "{}"

        conn = _connect()
        try:
            if not _schema_ready:
                _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO usage (
                    ts, iso_ts, agent_id, transport, channel_id, session_id,
                    user_id, user_handle, model, input_tokens, output_tokens,
                    cache_read_tokens, cache_creation_tokens, total_tokens,
                    est_cost_usd, outcome, raw_usage
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    now, iso,
                    str(agent_id) if agent_id is not None else None,
                    str(transport) if transport is not None else None,
                    str(channel_id) if channel_id is not None else None,
                    str(session_id) if session_id is not None else None,
                    str(user_id) if user_id is not None else None,
                    str(user_handle) if user_handle is not None else None,
                    str(model) if model is not None else None,
                    input_tokens, output_tokens, cache_read, cache_creation,
                    total_tokens, est_cost, outcome, raw,
                ),
            )
            conn.commit()
            _maybe_purge(conn, now)
        finally:
            conn.close()
    except Exception as e:
        print_ts(f"usage_ledger: record_usage failed (continuing): {e}", error=True)


def query_usage(since_ts: float, until_ts: float | None = None) -> list[dict]:
    """Return raw rows in the window [since_ts, until_ts], newest first."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            if until_ts is None:
                rows = conn.execute(
                    "SELECT * FROM usage WHERE ts >= ? ORDER BY ts DESC",
                    (since_ts,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM usage WHERE ts >= ? AND ts <= ? ORDER BY ts DESC",
                    (since_ts, until_ts),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception as e:
        print_ts(f"usage_ledger: query_usage failed: {e}", error=True)
        return []


def aggregate(since_ts: float, until_ts: float | None = None,
              group_by: str = "agent_id") -> list[dict]:
    """Aggregate usage in the window grouped by one whitelisted column.

    Returns rows {group, turns, input_tokens, output_tokens,
    cache_read_tokens, cache_creation_tokens, total_tokens, est_cost_usd}
    ordered by est_cost_usd desc. `group_by` must be one of the whitelist —
    the column name is taken from the whitelist set, never f-strung from raw
    caller input (SQL-injection safe)."""
    if group_by not in _GROUP_BY_WHITELIST:
        raise ValueError(f"group_by must be one of {sorted(_GROUP_BY_WHITELIST)}")
    col = group_by  # safe: membership-checked against the whitelist above
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            where = "ts >= ?"
            params: list = [since_ts]
            if until_ts is not None:
                where += " AND ts <= ?"
                params.append(until_ts)
            sql = (
                f"SELECT {col} AS grp, COUNT(*) AS turns, "
                "SUM(input_tokens) AS input_tokens, "
                "SUM(output_tokens) AS output_tokens, "
                "SUM(cache_read_tokens) AS cache_read_tokens, "
                "SUM(cache_creation_tokens) AS cache_creation_tokens, "
                "SUM(total_tokens) AS total_tokens, "
                "SUM(est_cost_usd) AS est_cost_usd "
                f"FROM usage WHERE {where} GROUP BY {col} "
                "ORDER BY est_cost_usd DESC"
            )
            rows = conn.execute(sql, params).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                out.append({
                    "group": d.get("grp"),
                    "turns": int(d.get("turns") or 0),
                    "input_tokens": int(d.get("input_tokens") or 0),
                    "output_tokens": int(d.get("output_tokens") or 0),
                    "cache_read_tokens": int(d.get("cache_read_tokens") or 0),
                    "cache_creation_tokens": int(d.get("cache_creation_tokens") or 0),
                    "total_tokens": int(d.get("total_tokens") or 0),
                    "est_cost_usd": float(d.get("est_cost_usd") or 0.0),
                })
            return out
        finally:
            conn.close()
    except Exception as e:
        print_ts(f"usage_ledger: aggregate failed: {e}", error=True)
        return []


def totals(since_ts: float, until_ts: float | None = None) -> dict:
    """Grand totals across the window."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            where = "ts >= ?"
            params: list = [since_ts]
            if until_ts is not None:
                where += " AND ts <= ?"
                params.append(until_ts)
            row = conn.execute(
                "SELECT COUNT(*) AS turns, "
                "COALESCE(SUM(input_tokens),0) AS input_tokens, "
                "COALESCE(SUM(output_tokens),0) AS output_tokens, "
                "COALESCE(SUM(cache_read_tokens),0) AS cache_read_tokens, "
                "COALESCE(SUM(cache_creation_tokens),0) AS cache_creation_tokens, "
                "COALESCE(SUM(total_tokens),0) AS total_tokens, "
                "COALESCE(SUM(est_cost_usd),0.0) AS est_cost_usd "
                f"FROM usage WHERE {where}",
                params,
            ).fetchone()
            d = dict(row) if row else {}
            return {
                "turns": int(d.get("turns") or 0),
                "input_tokens": int(d.get("input_tokens") or 0),
                "output_tokens": int(d.get("output_tokens") or 0),
                "cache_read_tokens": int(d.get("cache_read_tokens") or 0),
                "cache_creation_tokens": int(d.get("cache_creation_tokens") or 0),
                "total_tokens": int(d.get("total_tokens") or 0),
                "est_cost_usd": float(d.get("est_cost_usd") or 0.0),
            }
        finally:
            conn.close()
    except Exception as e:
        print_ts(f"usage_ledger: totals failed: {e}", error=True)
        return {"turns": 0, "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_creation_tokens": 0,
                "total_tokens": 0, "est_cost_usd": 0.0}


def daily_series(since_ts: float, until_ts: float | None = None) -> list[dict]:
    """Per-day time buckets over the window, oldest day first.

    Returns rows {date, turns, total_tokens, est_cost_usd} where `date` is a
    UTC 'YYYY-MM-DD' string. Buckets by calendar day in UTC — `ts` is stored as
    UTC epoch seconds (record_usage uses time.gmtime), so 'unixepoch' (which
    SQLite interprets as UTC) keeps day boundaries consistent with `iso_ts`.

    Read-only, exception-safe, parameterized — mirrors totals()/aggregate().
    The bucket expression is a fixed literal (no caller input is interpolated),
    so it carries the same SQL-injection safety as the whitelisted aggregate().
    """
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            where = "ts >= ?"
            params: list = [since_ts]
            if until_ts is not None:
                where += " AND ts <= ?"
                params.append(until_ts)
            sql = (
                "SELECT strftime('%Y-%m-%d', ts, 'unixepoch') AS day, "
                "COUNT(*) AS turns, "
                "COALESCE(SUM(total_tokens),0) AS total_tokens, "
                "COALESCE(SUM(est_cost_usd),0.0) AS est_cost_usd "
                f"FROM usage WHERE {where} GROUP BY day ORDER BY day ASC"
            )
            rows = conn.execute(sql, params).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                out.append({
                    "date": d.get("day"),
                    "turns": int(d.get("turns") or 0),
                    "total_tokens": int(d.get("total_tokens") or 0),
                    "est_cost_usd": float(d.get("est_cost_usd") or 0.0),
                })
            return out
        finally:
            conn.close()
    except Exception as e:
        print_ts(f"usage_ledger: daily_series failed: {e}", error=True)
        return []


def purge_older_than(cutoff_ts: float) -> int:
    """Delete rows older than cutoff_ts. Returns count deleted."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            cur = conn.execute("DELETE FROM usage WHERE ts < ?", (cutoff_ts,))
            conn.commit()
            return cur.rowcount or 0
        finally:
            conn.close()
    except Exception as e:
        print_ts(f"usage_ledger: purge_older_than failed: {e}", error=True)
        return 0
