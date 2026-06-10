"""Snapshot framework files before destructive writes.

Background: 2026-05-07 catastrophe permanently lost claude_conversation.py
because no recovery mechanism existed. This module gives a recovery path for
any future bad write to framework code, agent identity files, or shared
framework documents.

Snapshots land at <project_root>/.snapshots/<UTC-ts>/<relpath>. Triggered
automatically by write_file / edit_file / delete_file when the target path
matches a snapshot policy.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .utils import project_root, print_ts, safe_path_display, redact_paths, COLOR_YELLOW, COLOR_END


# Keep the N most recent snapshots for any single file path. Bounded retention
# so .snapshots/ doesn't grow unbounded over time.
RETENTION_PER_FILE = 10


def _snapshots_root() -> Path:
    return Path(project_root()) / ".snapshots"


def should_snapshot(abs_path: str) -> bool:
    """Return True if this path is in scope for snapshotting.

    In scope:
      - <root>/openflip/         framework code
      - <root>/cron/             scheduling
      - <root>/agents/_shared/   shared framework files (FRAMEWORK.md, TOOLS.md)
      - <root>/agents/<id>/{SOUL.md, AGENT.md, MEMORY.md, agent.json}  identity

    Out of scope:
      - conversations/, memory/index.json, live.json, state.json (runtime)
      - data/ (generated outputs)
      - anything outside project_root
    """
    try:
        abs_path = os.path.realpath(abs_path)
        root = os.path.realpath(project_root())
    except Exception:
        return False

    if not abs_path.startswith(root + os.sep):
        return False

    rel = os.path.relpath(abs_path, root)
    parts = rel.split(os.sep)
    if not parts:
        return False

    top = parts[0]
    if top == "openflip":
        return True
    if top == "cron":
        return True
    if top == "agents":
        if len(parts) < 2:
            return False
        if parts[1] == "_shared":
            return True
        # agents/<id>/<file>: only the identity files at top level of agent dir
        if len(parts) == 3:
            fname = parts[2]
            if fname in ("SOUL.md", "AGENT.md", "MEMORY.md", "agent.json"):
                return True
        return False
    return False


def snapshot_file(abs_path: str, *, content_bytes: Optional[bytes] = None) -> Optional[Path]:
    """Snapshot a file before a destructive operation.

    If `content_bytes` is provided, those bytes are written as the snapshot
    (use this when the caller already read the file - saves a re-read).
    Otherwise reads from disk.

    Returns the snapshot path on success, None on no-op or failure. Failures
    are logged but never raised - the caller MUST proceed with the original
    write regardless. Not having a snapshot is bad; blocking the write because
    snapshot failed is worse.
    """
    abs_path = os.path.realpath(abs_path)
    if not should_snapshot(abs_path):
        return None

    # Nothing to snapshot if file doesn't exist (e.g., first-time write_file)
    if content_bytes is None and not os.path.isfile(abs_path):
        return None

    try:
        root = os.path.realpath(project_root())
        rel = os.path.relpath(abs_path, root)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap_dir = _snapshots_root() / ts / os.path.dirname(rel)
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / os.path.basename(rel)
        # Atomic write: a snapshot is a recovery artifact, so a crash mid-write
        # must never leave a torn/truncated file. Write to a sibling .tmp then
        # os.replace (atomic on the same filesystem).
        _tmp_path = snap_path.with_name(snap_path.name + ".tmp")
        if content_bytes is not None:
            _tmp_path.write_bytes(content_bytes)
        else:
            shutil.copy2(abs_path, _tmp_path)
        os.replace(_tmp_path, snap_path)
        try:
            _enforce_retention(rel)
        except Exception as e:
            print_ts(f"{COLOR_YELLOW}snapshot retention sweep failed for {rel}: {e}{COLOR_END}", error=True)
        return snap_path
    except Exception as e:
        print_ts(f"{COLOR_YELLOW}snapshot failed for {abs_path}: {e}{COLOR_END}", error=True)
        return None


def _list_snapshots_for_relpath(rel_path: str) -> list[tuple[str, Path]]:
    """Return [(timestamp_str, snapshot_path), ...] for a relative path."""
    root = _snapshots_root()
    if not root.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for ts_dir in root.iterdir():
        if not ts_dir.is_dir():
            continue
        candidate = ts_dir / rel_path
        if candidate.is_file():
            out.append((ts_dir.name, candidate))
    return out


def _enforce_retention(rel_path: str) -> None:
    """Keep only the RETENTION_PER_FILE most recent snapshots for this rel_path."""
    snaps = _list_snapshots_for_relpath(rel_path)
    if len(snaps) <= RETENTION_PER_FILE:
        return
    snaps.sort(key=lambda p: p[0], reverse=True)
    for _ts_str, snap_path in snaps[RETENTION_PER_FILE:]:
        try:
            snap_path.unlink()
            # Clean up empty parent dirs up to .snapshots root
            parent = snap_path.parent
            snaps_root = _snapshots_root().resolve()
            while parent != snaps_root and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        except Exception:
            pass


def list_snapshots(abs_path: str) -> list[dict]:
    """Public-facing: list snapshots for an absolute path.

    Returns newest-first list of {timestamp, snapshot_path, size_bytes}.
    """
    try:
        abs_path = os.path.realpath(abs_path)
        root = os.path.realpath(project_root())
        rel = os.path.relpath(abs_path, root)
    except Exception:
        return []
    snaps = _list_snapshots_for_relpath(rel)
    snaps.sort(key=lambda p: p[0], reverse=True)
    out: list[dict] = []
    for ts_str, snap_path in snaps:
        try:
            size = snap_path.stat().st_size
        except Exception:
            size = 0
        out.append({
            "timestamp": ts_str,
            "snapshot_path": str(snap_path),
            "size_bytes": size,
        })
    return out


def restore_from_snapshot(
    abs_path: str,
    *,
    timestamp: Optional[str] = None,
    index: Optional[int] = None,
) -> tuple[bool, str]:
    """Restore a file from a snapshot.

    Provide exactly one of timestamp (exact match) or index (0 = newest).
    Before overwriting, snapshots the current state too - so restore is
    itself reversible.

    Returns (ok, message).
    """
    if (timestamp is None) == (index is None):
        return False, "Provide exactly one of timestamp or index."
    snaps = list_snapshots(abs_path)
    if not snaps:
        return False, f"No snapshots found for {safe_path_display(abs_path)}"
    chosen = None
    if timestamp is not None:
        for s in snaps:
            if s["timestamp"] == timestamp:
                chosen = s
                break
        if chosen is None:
            avail = ", ".join(s["timestamp"] for s in snaps)
            return False, f"No snapshot with timestamp {timestamp}. Available: {avail}"
    else:
        if index < 0 or index >= len(snaps):
            return False, f"index {index} out of range (have {len(snaps)} snapshots)"
        chosen = snaps[index]
    try:
        # Snapshot current state before overwrite, so restore is reversible.
        snapshot_file(abs_path)
        shutil.copy2(chosen["snapshot_path"], abs_path)
        return True, f"Restored {safe_path_display(abs_path)} from snapshot {chosen['timestamp']}"
    except Exception as e:
        return False, f"Restore failed: {redact_paths(str(e))}"
