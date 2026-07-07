"""General-purpose file tools — read, write, edit, list, and delete files.

Path access is controlled by agent.json:
  allowed_read_paths   — directories the agent can read (default: agent dir)
  allowed_write_paths  — directories the agent can write (default: agent dir)
  denied_paths         — always blocked (overrides allow lists)

Set ["*"] to allow everything.

Design notes (post-2026-05-07 catastrophe):
  * write_file is CREATE-ONLY. It refuses to overwrite an existing file.
    The unsafe "blind whole-file overwrite" is no longer representable.
  * edit_file is the modification tool — it takes (path, old_string, new_string)
    and replaces exactly one occurrence of old_string. old_string must appear
    EXACTLY ONCE in the file or the call fails. This means an edit must always
    target specific text the model has actually seen.
  * To genuinely replace a file, delete_file then write_file. That's two
    deliberate steps, not a one-shot accident.
  * Both write_file and edit_file write atomically (tempfile + os.replace) so
    a crash mid-write leaves either the old file or the new file, never a
    half-written one.
"""
from __future__ import annotations

import os
import tempfile

from ._base import tool, ToolResult
from ..snapshots import snapshot_file
from ..utils import project_root, safe_path_display, redact_paths, print_ts


def _get_agent():
    from ..tool_executor import CURRENT_AGENT
    agent = CURRENT_AGENT.get(None)
    if not agent:
        raise RuntimeError("No agent context available")
    return agent


def _resolve(path: str) -> str:
    """Resolve path. Absolute paths used as-is. Relative paths resolve from
    the agent dir first; if that misses, fall back to the project root so
    repo-relative paths like ``agents/_shared/MANUAL.md`` also work."""
    agent_dir = os.path.dirname(_get_agent().path)
    if os.path.isabs(path):
        return os.path.realpath(path)
    agent_rel = os.path.realpath(os.path.join(agent_dir, path))
    if os.path.exists(agent_rel):
        return agent_rel
    root_rel = os.path.realpath(os.path.join(project_root(), path))
    if os.path.exists(root_rel):
        return root_rel
    # Neither exists — return the agent-relative path (preserves prior
    # behavior for create/write callers that check non-existence).
    return agent_rel


def _is_within(child: str, parent: str) -> bool:
    """True if child path is the same as or nested inside parent.

    Both paths are realpath'd before comparison so symlinks don't bypass the
    boundary. Uses an explicit separator check so an allowlist of `/foo` does
    NOT match `/foobar` — `startswith` alone is wrong for path containment.
    normcase makes the comparison case-insensitive on Windows (where
    C:\\Secret and c:\\SECRET are the same directory — without it a deny
    list entry could be bypassed by changing case). Identity on POSIX.
    """
    parent = os.path.normcase(os.path.realpath(parent))
    child = os.path.normcase(os.path.realpath(child))
    return child == parent or child.startswith(parent + os.sep)


def _effective_allowed(agent, mode: str) -> list:
    """Resolve the effective path allow-list for the CURRENT speaker.

    Back-compat (the #1 constraint): when the configured field is a flat list
    (the historical form), it applies to everyone and is returned UNCHANGED —
    `raw or []` is byte-identical to the old `getattr(agent, attr, []) or []`.

    Opt-in per-user form: when the field is a dict, it is TRANSPORT-KEYED,
    structurally identical to a tool's `auth` block (`openflip/acl.py`). The
    outer keys are transports (`discord`, `imessage`, …); the CURRENT speaker's
    transport block is selected first, exactly like `_check_acl`'s
    `acl.auth.get(transport)`. Inside that block the SAME vocabulary as tool
    ACLs (`users` / `all_users`) is used, so there is ONE way to express "who
    gets what" across the project:

        <transport>            → the per-transport block. No block for the
                                 current transport → `[]` (default-deny /
                                 read-fallback downstream; never fail-open).
        <transport>.users.<id> → per-user override list. Discord: numeric id
                                 stringified (`"100000000000000000"`); iMessage:
                                 handle normalized with `.strip().lower()` (the
                                 same normalization acl uses). The owner is JUST
                                 an id here, exactly like tool ACLs — there is no
                                 magic `owner` key (tool owner-bypass was
                                 removed; this mirrors that).
        <transport>.all_users  → the baseline list for everyone in that
                                 transport not matched by `users`.

    Paths differ from tools in exactly one necessary way: a tool ACL is a
    yes/no predicate, but a path ACL must return a SET of dirs per matched
    audience — so a path `users` is a dict-of-lists rather than tools' flat
    list, but the transport-keying and the `users`/`all_users` keys mirror
    tools exactly.

    TYPE TRAP: a tool ACL `all_users` is a BOOL and `users` is a flat LIST;
    here in a path ACL `all_users` is a LIST OF PATHS and `users` is a
    DICT-OF-LISTS. Same names, different types — do NOT cross. A bool/list
    pasted from the wrong schema is treated as misconfiguration and fails
    closed (returns []) with a logged warning, never `list(True)`.

    Resolution against the current speaker (the same `CURRENT_SESSION` /
    `CURRENT_SPEAKER_ID` contextvars `send_file` already reads): pick the
    speaker's transport block; if absent → `[]`. Else, if the speaker's
    id/handle matches a `users` key → that list; else `all_users` (a
    present-but-empty `all_users` and an absent one both collapse to the same
    fallback); else `[]`, which feeds the existing default-deny / read-fallback
    logic in `_check_access` — never fail-open.

    Path ACLs intentionally carry NO `exclude` and no role/channel dimensions
    in v1: deny is `denied_paths` (flat, unconditional, checked first), and not
    granting a `users`/`all_users` entry already withholds access.
    """
    raw = agent.allowed_read_paths if mode == "read" else agent.allowed_write_paths
    if not isinstance(raw, dict):
        return raw or []  # flat list (or None) — applies to everyone, unchanged

    # Per-user resolution. Identity is derived exactly like send_file's
    # cross-channel owner guard: speaker_id from CURRENT_SPEAKER_ID, transport +
    # handle from CURRENT_SESSION (Session is the source of truth on handle-based
    # transports). Any lookup error → empty identity → falls to all_users/deny.
    from ..tool_executor import current_caller
    speaker_id, transport, handle = current_caller()

    # Transport-keyed: select the current speaker's transport block first, the
    # SAME shape and lookup as _check_acl's `acl.auth.get(transport)`. No block
    # for this transport → deny (empty list → default-deny / read fallback).
    block = raw.get(transport)
    if not isinstance(block, dict):
        return []

    # Within the block, mirror tool ACLs' `users` match: Discord id stringified,
    # iMessage handle normalized. The owner has no special branch — if the
    # operator wants the owner scoped, they list the owner's id under `users`
    # like any other id.
    # NOTE: a path ACL `all_users` is a LIST OF PATHS and `users` is a
    # DICT-OF-LISTS; in a TOOL ACL the SAME names mean a BOOL and a flat list
    # respectively (see openflip/agent.py `_parse_transport_auth`). Same name,
    # different type, do not cross. If someone pastes the tool idiom here
    # (`all_users: true`, or `users: [...]`) the value is misconfigured for a
    # path block — fail CLOSED (return []) and warn loudly rather than crash
    # with `list(True)` (TypeError) or silently mis-resolve.
    agent_id = getattr(agent, "id", "<unknown>")
    key = str(speaker_id) if transport == "discord" else handle.strip().lower()
    users = block.get("users", {})
    if users and not isinstance(users, dict):
        print_ts(
            f"[acl] agent {agent_id!r}: {mode} path ACL auth.{transport}.users "
            f"must be a DICT of id→paths, got {users!r} — denying. A path ACL "
            f"users is a dict-of-lists, NOT the flat list it is in a tool block."
        )
        return []
    if isinstance(users, dict) and key in users:
        return list(users[key] or [])
    av = block.get("all_users", [])
    if av and not isinstance(av, (list, tuple)):
        print_ts(
            f"[acl] agent {agent_id!r}: {mode} path ACL auth.{transport}.all_users "
            f"must be a LIST OF PATHS, got {av!r} — denying. A path ACL all_users "
            f"is a list of paths, NOT the bool it is in a tool block."
        )
        return []
    return list(av or [])


def _check_access(full_path: str, mode: str) -> str | None:
    """Check if the resolved path is allowed. Returns error string or None.

    Phase 3.1 (ISSUES.md SAFE-3): an EMPTY allow list is no longer treated
    as "no restriction" (fail-open). Empty write allow list now denies by
    default; empty read allow list falls back to a safe default (the agent
    dir plus the system temp dir — /tmp on POSIX, %TEMP% on Windows).
    Wildcards (`"*"`) still allow everything explicitly.

    Phase 3.2 (ISSUES.md SAFE-5): framework code paths are denied for
    WRITE regardless of agent allow lists — even `["*"]` does not override.
    Read access is still allowed (agents need to read framework code to
    propose changes). Framework writes happen via the operator directly,
    never via an agent's file tools.

    Always check denied_paths first — those are unconditional.
    """
    agent = _get_agent()
    denied = list(getattr(agent, "denied_paths", []) or [])
    # Framework code write deny was removed 2026-05-11 — the agent IS the
    # openflip maintainer and needs to edit framework code through normal
    # tools. allowed_write_paths in agent.json is the authoritative gate.
    for d in denied:
        if _is_within(full_path, d):
            return f"Access denied: {safe_path_display(full_path)}"

    # Per-user resolution lives in _effective_allowed; everything downstream
    # ("*" allow-all, empty-list default-deny + read fallback, the _is_within
    # boundary loop) is unchanged. For a flat-list agent this returns the same
    # list the old `getattr(...) or []` did — byte-identical back-compat.
    allowed = _effective_allowed(agent, mode)

    if "*" in allowed:
        return None

    if not allowed:
        if mode == "read":
            # Read fallback: own agent dir + /tmp. Anything else needs
            # explicit configuration. This is more conservative than the
            # old fail-open but still lets a freshly-installed agent do
            # basic work without being completely blind.
            agent_dir = os.path.dirname(agent.path)
            for fallback in (agent_dir, tempfile.gettempdir()):
                if _is_within(full_path, fallback):
                    return None
            return (
                f"Access denied: {safe_path_display(full_path)} (no allowed_read_paths "
                f"configured; default read scope is agent dir + the system temp dir)"
            )
        return (
            f"Access denied: {safe_path_display(full_path)} (no allowed_write_paths configured; "
            f"writes require an explicit allow list)"
        )

    for a in allowed:
        if _is_within(full_path, a):
            return None
    return f"Access denied: {safe_path_display(full_path)}"


def _atomic_write_bytes(full_path: str, content: bytes) -> None:
    """Write bytes to full_path atomically: tempfile in same dir, then os.replace.

    Either the old file remains intact or the new content is fully present.
    Never a torn half-written file. Caller is responsible for ensuring the
    parent directory exists and access checks have already passed.

    Operates on bytes (not str) so callers control the on-disk line-ending
    bytes explicitly. Avoids the silent CRLF→LF rewrite that bit text-mode
    write when editing files saved under non-Linux conventions.
    """
    parent = os.path.dirname(full_path) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix="." + os.path.basename(full_path) + ".",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            # fsync before close so contents hit disk before the rename. Without
            # this, a kernel crash between write and replace can leave an empty
            # tmp file on disk even after replace returned successfully.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, full_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


@tool
async def read_file(path: str) -> ToolResult:
    """Read a text file. Absolute paths or relative to your agent folder.

    Args:
        path: Path to the file.
    """
    full = _resolve(path)
    err = _check_access(full, "read")
    if err:
        return ToolResult.fail(err)
    if not os.path.exists(full):
        return ToolResult.fail(f"File not found: {path}")
    if os.path.isdir(full):
        return ToolResult.fail(f"'{path}' is a directory, not a file. Use list_files.")
    try:
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResult(model_feedback=content)
    except Exception as e:
        return ToolResult.fail(f"Failed to read: {redact_paths(str(e))}")


@tool
async def send_file(path: str, caption: str = "", channel_id: int = 0) -> ToolResult:
    """Attach an existing local file and post it to a Discord channel.

    Use this to DELIVER a file that already exists on disk — e.g. an audio
    stem you separated, an image you saved, a document. The other media tools
    (generate_image, extract_audio_track, etc.) auto-post their OWN output;
    send_file is for files that are just sitting on disk with no way to reach
    the user. send_message posts TEXT only and cannot attach a file — this is
    the tool for files.

    Args:
        path: Path to the file to attach. Absolute, or relative to your agent folder.
        caption: Optional text to post alongside the file.
        channel_id: Optional. The Discord channel ID to post to. Defaults to
            the channel that triggered the current turn. Pass an explicit ID
            to cross-post somewhere else — owner-only, mirrors send_message.
    """
    full = _resolve(path)
    err = _check_access(full, "read")
    if err:
        return ToolResult.fail(err)
    if not os.path.exists(full):
        return ToolResult.fail(f"File not found: {path}")
    if os.path.isdir(full):
        return ToolResult.fail(f"'{path}' is a directory, not a file.")

    # send_file is a DELIVERY tool: its entire job is "this file reaches the
    # channel." So it posts DIRECTLY through the transport and reports the
    # truth — success only if the transport actually accepted the upload.
    # It does NOT return attachments through the executor's auto-post path,
    # because that path swallows post failures (transport.send_file returns
    # None on 413 / unresolved channel) and then build_model_feedback tells
    # the model "the user can see them" regardless — the lie that made
    # an agent say "i already sent it" when nothing posted (2026-05-29/30).
    # Mirrors send_message.py: one path, one honest error semantic.
    from ..tool_executor import (
        CURRENT_AGENT, CURRENT_CHANNEL_ID, CURRENT_SESSION,
    )
    try:
        agent = CURRENT_AGENT.get()
    except LookupError:
        return ToolResult.fail("Tool invoked outside an agent context.")

    # Resolve the current turn's channel (used both as the default target and
    # for the cross-channel owner guard).
    current_channel_id = 0
    try:
        session = CURRENT_SESSION.get(None)
        if session is not None and session.transport == "discord":
            current_channel_id = session.channel_id_int
    except Exception:
        pass
    if not current_channel_id:
        try:
            current_channel_id = int(CURRENT_CHANNEL_ID.get())
        except LookupError:
            current_channel_id = 0

    # No explicit channel → deliver to the current turn's channel.
    explicit_channel = int(channel_id) if channel_id else 0
    target_channel = explicit_channel or current_channel_id
    if not target_channel:
        return ToolResult.fail("No channel_id provided and no current channel in context.")

    # Cross-channel posting (target ≠ current) is owner-only (mirrors
    # send_message's MED-1 guard).
    if explicit_channel and explicit_channel != current_channel_id:
        from ..acl import is_owner as _is_owner
        from ..tool_executor import current_caller
        # Transport-aware: Discord → numeric path (unchanged); iMessage →
        # compare the raw handle. Absent session → discord/"" → numeric path.
        speaker_id, _tname, _handle = current_caller()
        if not _is_owner(speaker_id, transport=_tname, handle=_handle):
            return ToolResult.fail(
                f"send_file: cross-channel posting (target {explicit_channel} ≠ "
                f"current {current_channel_id}) is restricted to the owner."
            )

    from ..registry import RUNNERS
    runner = RUNNERS.get(agent.id)
    if not runner:
        return ToolResult.fail(f"No running agent for '{agent.id}'.")
    transport = getattr(runner, "transport", None)
    if transport is None:
        return ToolResult.fail("No transport available to post the file.")

    session_id = str(target_channel)
    try:
        url = await transport.send_file(session_id, full, content=(caption or ""))
    except Exception as e:
        return ToolResult.fail(f"send_file: posting to channel {target_channel} failed: {redact_paths(str(e))}")
    if not url:
        return ToolResult.fail(
            f"send_file: the file was generated but DELIVERY FAILED — transport "
            f"accepted no attachment for channel {target_channel} (channel unreachable, "
            f"or upload rejected, e.g. file too large). The file was NOT posted; do not "
            f"claim you sent it."
        )
    return ToolResult(model_feedback=f"Posted {os.path.basename(full)} to channel {target_channel}. URL: {url}")


@tool
async def write_file(path: str, content: str) -> ToolResult:
    """Create a new file with the given content. CREATE-ONLY — refuses to overwrite an existing file.

    To modify an existing file, use edit_file. To genuinely replace a file with new
    contents, delete_file first then write_file. There is no one-shot full-file
    overwrite tool by design (see openflip ISSUES.md, Phase 5 refactor).

    Creates parent directories as needed. Writes atomically via tempfile + rename,
    so a crash mid-write cannot leave a torn file.

    Args:
        path: Path to the file. Must NOT already exist.
        content: The text content to write.
    """
    # Cap content size to prevent the 'huge content arg leaks as message.txt'
    # failure mode (incident 2026-05-08). For larger files, build via
    # run_command with a heredoc so the content never lives in the tool args.
    _MAX_BYTES = 8000
    encoded_len = len(content.encode('utf-8', errors='replace'))
    if encoded_len > _MAX_BYTES:
        return ToolResult.fail(
            f'write_file content is {encoded_len} bytes; cap is {_MAX_BYTES}. '
            f"For larger files, use run_command with a heredoc: "
            f"run_command(\"cat > <path> <<'EOF'\\n<file lines>\\nEOF\") — "
            f"that keeps the content out of tool args entirely and avoids the "
            f"large-arg leak path."
        )

    # Phase 5.3 content sanity guard: refuse content that is overwhelmingly
    # model-scaffolding. The May-7 catastrophe class is a model dumping its
    # internal tool-call / thinking blocks into a write_file payload that
    # then clobbers framework code. A handful of tag occurrences in legit
    # docs (FRAMEWORK.md, ISSUES.md, this file) is fine; a write where most
    # of the body is literal tag bodies is almost certainly the model
    # mistaking output channels.
    import re as _re
    _scaffold_patterns = [
        '<tool_call\\b[^>]*>.*?</tool_call>',
        '<function_calls\\b[^>]*>.*?</function_calls>',
        '<thinking\\b[^>]*>.*?</thinking>',
        '<system-reminder\\b[^>]*>.*?</system-reminder>',
    ]
    _scaffold_bytes = 0
    for _pat in _scaffold_patterns:
        for _m in _re.finditer(_pat, content, _re.DOTALL):
            _scaffold_bytes += len(_m.group(0).encode('utf-8', errors='replace'))
    if encoded_len > 0 and _scaffold_bytes * 2 > encoded_len:
        return ToolResult.fail(
            f"write_file refused: {_scaffold_bytes} of {encoded_len} bytes "
            f"look like model-scaffolding tags "
            "(tool_call / function_calls / thinking / system-reminder). "
            "That's more than half the payload — looks like a model-output "
            "dump rather than a real file write. If you really meant this "
            "content (e.g. a doc that quotes the tags), build the file via "
            "run_command heredoc instead so it doesn't pass through tool args."
        )
    full = _resolve(path)
    err = _check_access(full, "write")
    if err:
        return ToolResult.fail(err)
    if os.path.exists(full):
        return ToolResult.fail(
            f"File already exists: {path}. write_file is create-only. "
            f"Use edit_file to modify it, or delete_file first if you really intend to replace it entirely."
        )
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        _atomic_write_bytes(full, content.encode("utf-8"))
        return ToolResult(model_feedback=f"Written: {safe_path_display(full)}")
    except Exception as e:
        return ToolResult.fail(f"Failed to write: {redact_paths(str(e))}")


@tool
async def edit_file(path: str, old_string: str, new_string: str) -> ToolResult:
    """Modify an existing file by replacing exactly one occurrence of old_string with new_string.

    REQUIREMENTS:
      * The file must already exist (use write_file to create new files).
      * old_string must appear EXACTLY ONCE in the file. If it appears zero times,
        the call fails. If it appears multiple times, the call fails — include more
        surrounding context (a unique line, indentation, surrounding text) to make
        the match unique.
      * old_string must match the file's text byte-for-byte: whitespace, indentation,
        line endings, everything. Read the file first to see the exact bytes.
      * old_string and new_string must differ.

    To make multiple edits, call edit_file multiple times. To replace an entire file,
    delete_file first then write_file (two deliberate steps).

    Writes atomically (tempfile + rename) — a crash mid-write cannot corrupt the file.

    Args:
        path: Path to the existing file.
        old_string: The exact text to replace. Must occur exactly once in the file.
        new_string: The replacement text.
    """
    full = _resolve(path)
    err = _check_access(full, "write")
    if err:
        return ToolResult.fail(err)
    if not os.path.exists(full):
        return ToolResult.fail(
            f"File not found: {path}. edit_file modifies existing files only — use write_file to create."
        )
    if os.path.isdir(full):
        return ToolResult.fail(f"'{path}' is a directory, not a file.")
    if old_string == new_string:
        return ToolResult.fail("old_string and new_string are identical — nothing to change.")
    if not old_string:
        return ToolResult.fail("old_string cannot be empty. Provide the exact text to replace.")

    try:
        with open(full, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        return ToolResult.fail(f"Failed to read for edit: {redact_paths(str(e))}")

    # Snapshot the current state before any destructive change, in case the
    # edit produces a regression we need to roll back. Reuses raw_bytes we
    # just read - no extra disk hit. Failure is non-fatal (logged inside).
    snapshot_file(full, content_bytes=raw_bytes)

    # Preserve the file’s line-ending convention through the edit. Detect CRLF
    # vs LF on the source bytes, then normalize old_string and new_string to
    # match before doing the byte-level replace. The model typically supplies
    # \n-only strings (because read_file returns text-mode content) which would
    # never match a CRLF file without this conversion.
    file_has_crlf = b"\r\n" in raw_bytes
    file_eol = b"\r\n" if file_has_crlf else b"\n"

    def _to_file_eol(s: str) -> bytes:
        b = s.encode("utf-8")
        # Normalize any CRLF in the supplied string back to LF first, then convert
        # to the file’s convention. Avoids \r\r\n double-encoding when the model
        # already supplied CRLF in the args.
        b = b.replace(b"\r\n", b"\n").replace(b"\n", file_eol)
        return b

    old_bytes = _to_file_eol(old_string)
    new_bytes = _to_file_eol(new_string)

    count = raw_bytes.count(old_bytes)
    if count == 0:
        return ToolResult.fail(
            f"old_string not found in {path}. Read the file first to confirm the exact text "
            f"(including whitespace and indentation)."
        )
    if count > 1:
        return ToolResult.fail(
            f"old_string matches {count} places in {path}. Add surrounding context to make it unique."
        )

    new_raw = raw_bytes.replace(old_bytes, new_bytes, 1)
    try:
        _atomic_write_bytes(full, new_raw)
        size_delta = len(new_raw) - len(raw_bytes)
        sign = "+" if size_delta >= 0 else ""
        return ToolResult(model_feedback=f"Edited: {safe_path_display(full)} ({sign}{size_delta} bytes)")
    except Exception as e:
        return ToolResult.fail(f"Failed to write edit: {redact_paths(str(e))}")


@tool
async def list_files(path: str = ".") -> ToolResult:
    """List files and directories at a path.

    Args:
        path: Directory path (default: agent folder).
    """
    full = _resolve(path)
    err = _check_access(full, "read")
    if err:
        return ToolResult.fail(err)
    if not os.path.isdir(full):
        return ToolResult.fail(f"Not a directory: {path}")
    try:
        entries = sorted(os.listdir(full))
    except Exception as e:
        return ToolResult.fail(f"Failed to list: {redact_paths(str(e))}")

    lines = []
    for name in entries:
        fp = os.path.join(full, name)
        try:
            if os.path.isdir(fp):
                lines.append(f"  {name}/")
            elif os.path.islink(fp) and not os.path.exists(fp):
                # Broken symlink — target gone. Listed but marked.
                lines.append(f"  {name}  (broken symlink)")
            else:
                size = os.path.getsize(fp)
                lines.append(f"  {name}  ({size} bytes)")
        except Exception as _e:
            # Per-entry failure (permission denied, race, dead symlink etc.).
            # Don't let one bad entry kill the whole listing — that error
            # bubbles back to the model and tends to produce an empty
            # follow-up turn.
            lines.append(f"  {name}  (error: {type(_e).__name__})")
    return ToolResult(model_feedback="\n".join(lines) if lines else "(empty directory)")


@tool
async def delete_file(path: str) -> ToolResult:
    """Permanently delete a file (a snapshot is taken first so restore_snapshot can undo it). Requires write access to the path.

    Args:
        path: Path to the file to delete.
    """
    full = _resolve(path)
    err = _check_access(full, "write")
    if err:
        return ToolResult.fail(err)
    if not os.path.exists(full):
        return ToolResult.fail(f"File not found: {path}")
    if os.path.isdir(full):
        return ToolResult.fail(f"'{path}' is a directory, not a file.")
    # Snapshot before deletion so it's recoverable. Failure non-fatal.
    snapshot_file(full)
    try:
        os.remove(full)
        return ToolResult(model_feedback=f"Deleted: {safe_path_display(full)}")
    except Exception as e:
        return ToolResult.fail(f"Failed to delete: {redact_paths(str(e))}")
