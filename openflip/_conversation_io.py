"""Shared JSONL conversation persistence helpers.

`DiscordConversation` (ollama) and `AnthropicConversation` (anthropic OAuth)
both persist messages as one JSON object per line. The differences between
them are limited to:

  * Whether they keep a metadata sidecar (anthropic does — for the stored
    compaction block + last_usage; ollama doesn't).
  * Which `ChatMessage` class wraps the loaded entries.
  * What the `content` field of each persisted message is named on the
    in-memory message object (ollama: `content`; anthropic: `content_text`
    falls back to `content`).

These differences are the *only* reason to keep two classes. Everything
that touches files lives here as plain functions so neither class has to
re-implement migration, JSONL reading, or atomic appending.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Optional

from .utils import load_json, print_ts


# A conversation_id is the transport-prefixed key (e.g. "discord:12345",
# "imessage:you@example.com", "internal:myagent"). It becomes a FILENAME
# under conversations/, so any path-control character in it would let an
# attacker escape the directory and read/clobber arbitrary files. The legit
# prefix separator is ":"; legit id portions are digits, handles (with "@",
# "+", "."), and agent slugs. Path separators, NUL, control chars, and the
# parent-dir token ".." never appear in a real conversation_id. We reject
# them here as a last line of defense — callers (e.g. the inbound /trigger
# endpoint) validate too, but this guards EVERY filesystem callsite at once.
_UNSAFE_CONV_ID = re.compile(r"[\x00-\x1f/\\]")


def _safe_conversation_id(conversation_id: str) -> str:
    """Return conversation_id unchanged if filesystem-safe, else raise.

    Fail-closed: an empty, traversing, or control-char-bearing id raises
    ValueError rather than being silently rewritten — a bad id is a bug or
    an attack, never something to paper over with a guessed sanitized name.
    """
    cid = str(conversation_id)
    if not cid or ".." in cid or _UNSAFE_CONV_ID.search(cid):
        raise ValueError(f"unsafe conversation_id (path traversal blocked): {conversation_id!r}")
    return cid


# Windows forbids ":" in filenames — NTFS treats it as the alternate-data-
# stream separator, so opening "discord:123.jsonl" would silently write to
# a stream named "123.jsonl" hanging off a zero-byte file called "discord",
# collapsing every conversation into invisible streams of one file. The
# conversation_id keeps its canonical "transport:id" shape everywhere in
# memory and in URLs; ONLY the on-disk filename swaps ":" for "%3A" (the
# URL-encoding of ":" — unambiguous because "%" never appears in a real
# conversation_id). POSIX filenames are unchanged, so existing Linux/macOS
# deployments keep their files byte-for-byte.
_FS_COLON = "%3A"


def fs_encode(conversation_id: str) -> str:
    """Conversation id → on-disk filename stem. Identity on POSIX; on
    Windows the ":" separator is encoded as "%3A". Safe to call on glob
    patterns too ("*:123" → "*%3A123"). Raises on unsafe ids — same
    fail-closed contract as conversation_path."""
    cid = _safe_conversation_id(conversation_id)
    if os.name == "nt":
        return cid.replace(":", _FS_COLON)
    return cid


def fs_decode(stem: str) -> str:
    """On-disk filename stem → conversation id. Inverse of fs_encode."""
    if os.name == "nt":
        return stem.replace(_FS_COLON, ":")
    return stem


def conversation_path(agent_dir: str, conversation_id: str) -> str:
    """JSONL message log for one channel."""
    return os.path.join(agent_dir, "conversations", f"{fs_encode(conversation_id)}.jsonl")


def legacy_path(agent_dir: str, conversation_id: str) -> str:
    """Pre-JSONL single-blob `.json` location. Migrated on first load."""
    return os.path.join(agent_dir, "conversations", f"{fs_encode(conversation_id)}.json")


def read_all_messages(jsonl_path: str) -> list[dict]:
    """Read every JSONL line back as a list of dicts.

    Tolerates a torn final line from a crashed save by skipping it.
    """
    if not os.path.isfile(jsonl_path):
        return []
    out: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_messages(jsonl_path: str, messages: list, *, content_extractor: Callable[[Any], str]) -> None:
    """Append messages as JSONL lines.

    `content_extractor(msg)` returns the on-disk content string for one
    message — exists because Claude messages prefer `content_text` and
    Ollama messages use `content`. Caller controls the policy.
    """
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for m in messages:
            role = m.get("role") if hasattr(m, "get") else getattr(m, "role", None)
            f.write(json.dumps({
                "role": role,
                "content": content_extractor(m) or "",
                "ts": time.time(),
            }, ensure_ascii=False) + "\n")
        # fsync so the trailing line(s) actually reach disk before the
        # context closes. Without this, a crash between write() returning
        # and the kernel flushing buffers can drop the most-recent turn.
        f.flush()
        os.fsync(f.fileno())


def migrate_legacy_to_jsonl(
    agent_dir: str,
    conversation_id: str,
    *,
    log_agent_id: Optional[str] = None,
) -> bool:
    """Convert legacy `<id>.json` blob to `<id>.jsonl` if needed. Returns True if migrated."""
    legacy = legacy_path(agent_dir, conversation_id)
    new = conversation_path(agent_dir, conversation_id)
    if os.path.isfile(new) or not os.path.isfile(legacy):
        return False
    data = load_json(legacy, default={"messages": []})
    msgs = data.get("messages", [])
    os.makedirs(os.path.dirname(new), exist_ok=True)
    with open(new, "w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
        # fsync the migrated JSONL before we delete the legacy file below.
        # Without it, a crash between the writes returning and the kernel
        # flushing buffers can lose the migrated contents AND the legacy
        # source — net data loss.
        f.flush()
        os.fsync(f.fileno())
    try:
        os.remove(legacy)
    except OSError:
        pass
    print_ts(f"Migrated {len(msgs)} messages to JSONL ({conversation_id})", agent=log_agent_id)
    return True


def delete_conversation_files(
    agent_dir: str,
    conversation_id: str,
    *,
    extra_paths: Optional[list[str]] = None,
    backup_tag: Optional[str] = None,
) -> None:
    """Remove the JSONL file, the legacy `.json`, and any extra (e.g. meta sidecar).

    If `backup_tag` is set (e.g. "pre_reset"), the JSONL is copied to
    `<jsonl>.<backup_tag>_<unix_ts>.bak.jsonl` BEFORE deletion so the
    history can be recovered. Backups are NOT made for extra_paths or
    the legacy `.json` — only the canonical JSONL.
    """
    jsonl = conversation_path(agent_dir, conversation_id)
    if backup_tag and os.path.exists(jsonl):
        try:
            import shutil
            ts = int(time.time())
            backup = f"{jsonl}.{backup_tag}_{ts}.bak.jsonl"
            shutil.copy2(jsonl, backup)
            print_ts(f"backed up conversation to {backup} before {backup_tag}")
        except Exception as _bk_err:
            print_ts(f"WARNING: pre-{backup_tag} backup failed for {jsonl}: {_bk_err}")
        # Retention sweep — keep only the 5 most-recent backups for this tag
        # per channel. Without this, backups accumulate forever. Matches the
        # pattern in anthropic_conversation.py for compaction backups.
        try:
            import glob as _glob
            _backup_keep = 5
            _all_bak = sorted(_glob.glob(f"{jsonl}.{backup_tag}_*.bak.jsonl"))
            if len(_all_bak) > _backup_keep:
                for _stale in _all_bak[:-_backup_keep]:
                    try:
                        os.remove(_stale)
                        print_ts(f"pruned stale {backup_tag} backup: {os.path.basename(_stale)}")
                    except OSError:
                        pass
        except Exception as _retain_e:
            print_ts(f"WARNING: {backup_tag} backup retention sweep failed: {_retain_e}")
    paths = [jsonl, legacy_path(agent_dir, conversation_id)]
    if extra_paths:
        paths.extend(extra_paths)
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
