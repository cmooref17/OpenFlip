"""Tests for the Claude-Code-style memory index (openflip/memory_index.py).

Standalone runnable script (no pytest in this venv):
    .lvenv/bin/python tests/test_memory_index.py

Covers:
  (a) an old free-form MEMORY.md converts into index + topic files, backup kept,
      no content line lost, idempotent on a second call
  (b) an already-index MEMORY.md is left byte-identical (no backup, no rewrite)
  (c) upsert_topic appends to an existing topic and adds a line for a new one
  (d) upsert_topic that doesn't change the hook leaves MEMORY.md untouched
  (e) cap_index truncates at 200 lines / 25k chars and says what was cut
  (f) load_index_block gives an "Empty" block when there's no MEMORY.md
  (g) `## ` lines inside a code fence don't start a new topic
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openflip import memory_index as mi  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILS.append(label)


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


OLD = """# Operator's core memory

Some loose preamble note.

## About the operator
- Likes short replies.
- Uses **Wayland**, never X11.

## Projects (active)
- webapp on port 1717
```
## not a heading, inside a fence
```

## Empty section
"""


def test_convert_old_style():
    d = tempfile.mkdtemp()
    _write(os.path.join(d, "MEMORY.md"), OLD)
    new = mi.ensure_index(d)
    topics = mi.list_topic_files(d)
    backups = [f for f in os.listdir(d) if ".pre-index-" in f]
    joined = "\n".join(_read(os.path.join(mi.topics_dir(d), t)) for t in topics)
    lost = [l for l in OLD.splitlines()
            if l.strip() and not l.startswith("# ") and not l.startswith("## ") and l.strip() not in joined]
    check(mi.is_index(new), "(a) converted MEMORY.md is an index")
    check(len(backups) == 1 and _read(os.path.join(d, backups[0])) == OLD, "(a) original backed up verbatim")
    check(sorted(topics) == ["about_the_operator.md", "general_notes.md", "projects.md"],
          f"(a) topic files: {topics}")
    check(not lost, f"(a) no content line lost ({lost[:2]})")
    check(mi.ensure_index(d) == new, "(a) second call is a no-op")
    check("## not a heading, inside a fence" in _read(os.path.join(mi.topics_dir(d), "projects.md")),
          "(g) fenced '## ' stays inside its topic")


def test_index_untouched():
    d = tempfile.mkdtemp()
    idx = "# Memory index\n\n- [About](memory/topics/about.md) — hook\n"
    _write(os.path.join(d, "MEMORY.md"), idx)
    before = os.path.getmtime(os.path.join(d, "MEMORY.md"))
    out = mi.ensure_index(d)
    check(out == idx and os.path.getmtime(os.path.join(d, "MEMORY.md")) == before,
          "(b) index file left byte-identical")
    check(not [f for f in os.listdir(d) if ".pre-index-" in f], "(b) no backup for an index")


def test_upsert():
    d = tempfile.mkdtemp()
    mi.upsert_topic(d, "about_flip", "About the operator", "- first fact")
    p, created = mi.upsert_topic(d, "about_flip", "", "- second fact")
    body = _read(p)
    idx = _read(os.path.join(d, "MEMORY.md"))
    check(not created and "first fact" in body and "second fact" in body, "(c) append keeps both facts")
    check(idx.count("about_flip.md") == 1 and mi.is_index(idx), "(c) exactly one index line")
    _, created = mi.upsert_topic(d, "new_one", "New one", "- x")
    check(created and "new_one.md" in _read(os.path.join(d, "MEMORY.md")), "(c) new topic gets a line")

    mtime = os.path.getmtime(os.path.join(d, "MEMORY.md"))
    mi.upsert_topic(d, "new_one", "", "- y")  # hook = first line, unchanged
    check(os.path.getmtime(os.path.join(d, "MEMORY.md")) == mtime, "(d) same hook → MEMORY.md untouched")


def test_cap():
    many = "# idx\n" + "\n".join(f"- [t{i}](memory/topics/t{i}.md) — h" for i in range(300))
    out = mi.cap_index(many)
    check(len(out.splitlines()) == 202 and "101 of 301 lines were cut off" in out, "(e) 200-line cap + warning")
    wide = "# idx\n" + "\n".join(f"- [t{i}](memory/topics/t{i}.md) — " + "x" * 190 for i in range(150))
    out = mi.cap_index(wide)
    kept = out.split("\n\n> WARNING")[0]
    check(len(kept) <= mi.INDEX_MAX_CHARS and "WARNING" in out, "(e) 25k-char cap + warning")
    check(mi.cap_index("# idx\n- [a](memory/topics/a.md)") == "# idx\n- [a](memory/topics/a.md)",
          "(e) small index passes through unchanged")


def test_empty():
    d = tempfile.mkdtemp()
    check("Empty" in mi.load_index_block(d), "(f) no MEMORY.md → Empty block")


if __name__ == "__main__":
    for fn in (test_convert_old_style, test_index_untouched, test_upsert, test_cap, test_empty):
        print(fn.__name__)
        fn()
    print("\nRESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FAIL")
    sys.exit(1 if FAILS else 0)
