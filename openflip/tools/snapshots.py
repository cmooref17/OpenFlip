"""Tools for inspecting and restoring file snapshots.

Snapshots are taken automatically by edit_file / delete_file when the target
path is in scope (framework code, agent identity files, shared framework
docs). See openflip/snapshots.py for the policy.

These tools let an agent (or operator-via-agent) list and restore snapshots.
"""
from __future__ import annotations

from ._base import tool, ToolResult
from ..snapshots import list_snapshots as _list_snapshots, restore_from_snapshot
from ..utils import safe_path_display
from .files import _resolve, _check_access


@tool
async def list_snapshots(path: str) -> ToolResult:
    """List available snapshots for a file.

    Returns a newest-first list of snapshots with their timestamps and sizes.
    If the file isn't in snapshot scope, or no snapshots exist, returns an
    empty list.

    Args:
        path: Absolute path to the file.
    """
    if not path:
        return ToolResult.fail("path is required")
    # Same path ACL as files.py read — an agent shouldn't list snapshots of
    # files outside its read scope (another agent's SOUL.md, etc.).
    err = _check_access(_resolve(path), "read")
    if err:
        return ToolResult.fail(err)
    snaps = _list_snapshots(path)
    if not snaps:
        return ToolResult(model_feedback=f"No snapshots found for {path}.")
    lines = [f"{len(snaps)} snapshot(s) for {path} (newest first):"]
    for i, s in enumerate(snaps):
        lines.append(f"  [{i}] {s['timestamp']}  {s['size_bytes']} bytes  {safe_path_display(s['snapshot_path'])}")
    return ToolResult(model_feedback="\n".join(lines))


@tool
async def restore_snapshot(path: str, timestamp: str = "", index: int = -1) -> ToolResult:
    """Restore a file from one of its snapshots.

    Provide exactly one of `timestamp` (exact match like 20260509T143000Z) or
    `index` (0 = most recent snapshot, 1 = previous, etc). Before restoring,
    the CURRENT state is itself snapshotted - so a bad restore is reversible.

    Args:
        path: Absolute path to the file to restore.
        timestamp: Exact snapshot timestamp string. Use list_snapshots first to
            see available timestamps. Pass empty string to use index instead.
        index: 0-based index into newest-first snapshot list. Use -1 to skip
            and use timestamp instead.
    """
    if not path:
        return ToolResult.fail("path is required")
    # restore overwrites the live file — gate on WRITE scope, same as
    # edit_file/delete_file. Without this an agent could revert framework
    # code or another agent's files outside its allowed_write_paths.
    err = _check_access(_resolve(path), "write")
    if err:
        return ToolResult.fail(err)
    ts = timestamp.strip() or None
    idx = index if index >= 0 else None
    if (ts is None) == (idx is None):
        return ToolResult.fail(
            "Provide exactly one of timestamp (non-empty string) OR index (>= 0). "
            "Use list_snapshots(path) first to see available snapshots."
        )
    ok, msg = restore_from_snapshot(path, timestamp=ts, index=idx)
    if not ok:
        return ToolResult.fail(msg)
    return ToolResult(model_feedback=msg)
