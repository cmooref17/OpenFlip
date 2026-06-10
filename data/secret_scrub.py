#!/usr/bin/env python3
"""Nightly secret scrubber.

Sweeps `agents/*/conversations/*.jsonl` (including .bak compaction/reset
backups) for leaked secrets and replaces them with `[REDACTED:<kind>]`
markers. The conversation text stays intact — only the secret strings
disappear.

Why this exists: agents grep config files, env dumps, etc. as part of
normal work. When the result is captured in their context, it lands in
the .jsonl on disk. Rotating tokens every time this happens is high
friction; periodic scrubbing is the right primitive.

Safety:
  - Atomic write (tempfile + os.replace) so a crash mid-write can't
    truncate or corrupt the file.
  - `.scrub_<unix_ts>.bak` made BEFORE rewriting, retained for 7 days
    (older ones swept at start of next run). Recovery path: copy back.
  - Files mtime'd in the last 5 minutes are SKIPPED — that's an active
    conversation getting written to. We never race the live writer.
  - `--dry-run` mode: prints what would change, writes nothing.

Patterns covered:
  - discord bot tokens: `MT[A-Za-z0-9_-]{20,}` followed by `.<chunk>.<chunk>`
  - github PATs: `ghp_[A-Za-z0-9]{36}`
  - anthropic API keys: `sk-ant-[A-Za-z0-9_-]{30,}`
  - openai API keys: `sk-[A-Za-z0-9_-]{40,}`  (after anthropic so we don't double-match)
  - generic Bearer header tokens we control: `Bearer [A-Za-z0-9_.-]{40,}`

Anthropic OAuth access tokens (`sk-ant-oat...`) are caught by the anthropic
pattern. Refresh tokens too — same prefix shape.

Usage:
    python3 secret_scrub.py            # do the work
    python3 secret_scrub.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS_GLOB = str(PROJECT_ROOT / "agents" / "*" / "conversations" / "*.jsonl")
LOG_PATH = PROJECT_ROOT / "data" / "secret_scrub.log"

# Files modified within this many seconds are considered active — skip.
SKIP_RECENT_SECONDS = 300

# How many days to keep scrub backups before sweeping at next run.
BACKUP_RETENTION_DAYS = 7


# Order matters: more-specific patterns first so we don't double-match.
# (e.g. `sk-ant-...` must be tried before generic `sk-...`)
PATTERNS = [
    ("discord_token", re.compile(r"MT[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}")),
    ("github_pat",    re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{30,}")),
    ("openai_key",    re.compile(r"sk-[A-Za-z0-9_-]{40,}")),
    ("bearer_token",  re.compile(r"Bearer [A-Za-z0-9_.-]{40,}")),
]


def log(msg: str) -> None:
    """Append a timestamped line to the scrub log and print it."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def sweep_old_backups() -> int:
    """Delete .scrub_*.bak files older than BACKUP_RETENTION_DAYS. Returns count."""
    cutoff = time.time() - (BACKUP_RETENTION_DAYS * 86400)
    removed = 0
    for path in glob.glob(str(PROJECT_ROOT / "agents" / "*" / "conversations" / "*.scrub_*.bak")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


def scrub_file(path: str, dry_run: bool = False) -> dict:
    """Scrub one file. Returns a dict {kind: count} of redactions applied.

    Returns empty dict if no changes. On dry_run, just counts matches —
    no backup made, no write."""
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    redactions: dict[str, int] = {}
    scrubbed = original
    for kind, regex in PATTERNS:
        scrubbed, n = regex.subn(f"[REDACTED:{kind}]", scrubbed)
        if n > 0:
            redactions[kind] = n

    if not redactions:
        return {}

    if dry_run:
        return redactions

    # Atomic write with backup. Backup name uses the unix ts of when scrub
    # ran so concurrent runs don't clobber each other.
    backup_path = f"{path}.scrub_{int(time.time())}.bak"
    shutil.copy2(path, backup_path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(scrubbed)
    os.replace(tmp_path, path)
    return redactions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    log(f"=== secret_scrub start ({mode}) ===")

    if not args.dry_run:
        swept = sweep_old_backups()
        if swept:
            log(f"swept {swept} old scrub backup(s)")

    # Combined set across all .jsonl files (live + .bak.jsonl backups).
    # CONVERSATIONS_GLOB ends in `*.jsonl` which also matches `foo.bak.jsonl`
    # — and adding a second pattern duplicates. One glob is enough.
    files = sorted(set(glob.glob(CONVERSATIONS_GLOB)))

    now = time.time()
    total_redactions: dict[str, int] = {}
    files_touched = 0
    files_skipped_recent = 0

    for path in files:
        if ".scrub_" in path:
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if (now - mtime) < SKIP_RECENT_SECONDS:
            files_skipped_recent += 1
            continue
        try:
            result = scrub_file(path, dry_run=args.dry_run)
        except Exception as e:
            log(f"  ERROR on {path}: {e}")
            continue
        if result:
            files_touched += 1
            verb = "would redact" if args.dry_run else "redacted"
            for kind, n in result.items():
                total_redactions[kind] = total_redactions.get(kind, 0) + n
                log(f"  {verb} {n} {kind} in {path}")

    log(f"=== secret_scrub done ({mode}) ===")
    log(f"  files scanned: {len(files)}")
    log(f"  files skipped (recent): {files_skipped_recent}")
    log(f"  files touched: {files_touched}")
    if total_redactions:
        summary = ", ".join(f"{k}={n}" for k, n in sorted(total_redactions.items()))
        log(f"  redactions: {summary}")
    else:
        log("  redactions: none")

    return 0


if __name__ == "__main__":
    sys.exit(main())
