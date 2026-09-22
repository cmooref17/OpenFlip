"""Memory index: Claude-Code-style MEMORY.md for openflip agents.

MEMORY.md (agent root) is an INDEX, one line per topic file:
    - [Title](memory/topics/<slug>.md) — short hook
runtime loads it into the system prompt every turn (load_index_block),
capped like Claude Code 2.1.280's loader with a warning naming what was
cut. Detail lives in topic files under memory/topics/, read on demand.

A free-form MEMORY.md (old style) converts itself on first load: each
`## ` section becomes a topic file, MEMORY.md becomes the index, and the
original is backed up beside it. Nothing is deleted.
"""
from __future__ import annotations

import os
import re
import shutil
import time

from .utils import print_ts

# openflip's caps, set to match Claude Code 2.1.280's MEMORY.md loader
# (its bundled JS: DO=200 lines, N2=25000 chars). A design choice, not
# mirrored external state.
INDEX_MAX_LINES = 200
INDEX_MAX_CHARS = 25_000
HOOK_MAX_CHARS = 150
TOPICS_REL = "memory/topics"

_INDEX_LINE_RE = re.compile(r"^- \[(?P<title>[^\]]+)\]\((?P<path>[^)\s]+)\)")
_SECTION_RE = re.compile(r"^## +(?P<title>.+?)\s*$")
_SLUG_OK_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_LOCK_STALE_S = 60


def memory_md_path(agent_dir: str) -> str:
    return os.path.join(agent_dir, "MEMORY.md")


def topics_dir(agent_dir: str) -> str:
    return os.path.join(agent_dir, TOPICS_REL)


def slugify(text: str) -> str:
    """Filename slug. Parenthetical asides are dropped and the result is cut
    at a word boundary, so long headings still give short readable names."""
    text = re.sub(r"\([^)]*\)", " ", text.lower()).replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if len(s) > 40:
        cut = s.rfind("_", 0, 41)
        s = s[:cut if cut > 0 else 40]
    return s.strip("_") or "topic"


def valid_slug(slug: str) -> bool:
    return bool(_SLUG_OK_RE.match(slug or ""))


def topic_rel(slug: str) -> str:
    return f"{TOPICS_REL}/{slug}.md"


def topic_path(agent_dir: str, slug: str) -> str:
    return os.path.join(topics_dir(agent_dir), f"{slug}.md")


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def make_hook(text: str) -> str:
    """One-line index hook: first meaningful line, markdown stripped."""
    for line in text.splitlines():
        s = re.sub(r"^[\s>*#-]+", "", line).replace("**", "").replace("`", "")
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            return s if len(s) <= HOOK_MAX_CHARS else s[:HOOK_MAX_CHARS - 1].rstrip() + "…"
    return ""


def index_line(title: str, slug: str, hook: str) -> str:
    title = re.sub(r"[\[\]\n]+", " ", title).strip() or slug
    hook = re.sub(r"\s+", " ", hook or "").strip()
    return f"- [{title}]({topic_rel(slug)})" + (f" — {hook}" if hook else "")


def topic_text(slug: str, title: str, hook: str, body: str) -> str:
    desc = (hook or "").replace('"', "'")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    return (f"---\nname: {slug}\ndescription: \"{desc}\"\nmodified: {stamp}\n---\n\n"
            f"# {title}\n\n{body.strip()}\n")


def topic_body(text: str) -> str:
    """Topic file content minus frontmatter and its `# Title` line."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end >= 0:
            text = text[end + 5:]
    text = text.lstrip("\n")
    if text.startswith("# "):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    return text.strip()


def is_index(content: str) -> bool:
    """True when every non-blank line is a heading, an index line, or a comment."""
    for line in content.splitlines():
        s = line.strip()
        if s and not (s.startswith("#") or s.startswith("<!--") or _INDEX_LINE_RE.match(s)):
            return False
    return True


def cap_index(content: str) -> str:
    """Truncate like Claude Code's loader (lines first, then chars at a line
    boundary) and append a warning saying exactly what was cut."""
    trimmed = content.strip()
    lines = trimmed.split("\n")
    over_lines, over_chars = len(lines) > INDEX_MAX_LINES, len(trimmed) > INDEX_MAX_CHARS
    if not (over_lines or over_chars):
        return trimmed
    kept = "\n".join(lines[:INDEX_MAX_LINES]) if over_lines else trimmed
    if len(kept) > INDEX_MAX_CHARS:
        j = kept.rfind("\n", 0, INDEX_MAX_CHARS)
        kept = kept[:j if j > 0 else INDEX_MAX_CHARS]
    if trimmed[len(kept):len(kept) + 1] == "\n":
        n_kept = kept.count("\n") + 1
        what = f"{len(lines) - n_kept} of {len(lines)} lines were cut off, starting at line {n_kept + 1}"
    else:
        what = f"everything after the first {len(kept):,} chars was cut off"
    return (f"{kept}\n\n> WARNING: MEMORY.md is {len(lines)} lines / {len(trimmed):,} chars "
            f"(limit {INDEX_MAX_LINES} lines / {INDEX_MAX_CHARS:,} chars). Only part of it "
            f"was loaded: {what}. Keep index entries to one line under ~200 chars; move "
            f"detail into topic files.")


def _split_sections(content: str) -> tuple[str, list[tuple[str, str]]]:
    """(preamble, [(title, body), ...]) split on `## ` headings outside code fences."""
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    in_fence = False
    for line in content.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _SECTION_RE.match(line)
        if m:
            sections.append((m.group("title"), []))
        elif sections:
            sections[-1][1].append(line)
        else:
            preamble.append(line)
    return "\n".join(preamble).strip(), [(t, "\n".join(b).strip()) for t, b in sections]


def _unique_slug(agent_dir: str, base: str, taken: set[str]) -> str:
    slug, n = base, 2
    while slug in taken or os.path.exists(topic_path(agent_dir, slug)):
        slug, n = f"{base}_{n}", n + 1
    taken.add(slug)
    return slug


def convert(agent_dir: str) -> str:
    """Convert an old-style MEMORY.md into index + topic files. No-op when it's
    already an index. Backs the original up first, so nothing is lost.
    Returns the new MEMORY.md content."""
    mem = memory_md_path(agent_dir)
    content = _read(mem)
    if not content.strip() or is_index(content):
        return content
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = f"{mem}.pre-index-{stamp}.bak"
    shutil.copy2(mem, backup)

    preamble, sections = _split_sections(content)
    heading = next((l for l in preamble.splitlines() if l.startswith("# ")), "# Memory index")
    rest = "\n".join(l for l in preamble.splitlines() if not l.startswith("# ")).strip()
    items: list[tuple[str, str]] = []
    if rest:
        items.append(("General notes", rest))
    items.extend((t, b) for t, b in sections if b)
    if not items:
        items = [("General notes", content.strip())]

    taken: set[str] = set()
    lines = [heading, "",
             "<!-- One line per topic file. Detail lives in memory/topics/. "
             "Converted automatically; original saved as "
             f"{os.path.basename(backup)} -->", ""]
    for title, body in items:
        slug = _unique_slug(agent_dir, slugify(title), taken)
        hook = make_hook(body)
        _atomic_write(topic_path(agent_dir, slug), topic_text(slug, title, hook, body))
        lines.append(index_line(title, slug, hook))
    new = "\n".join(lines) + "\n"
    _atomic_write(mem, new)
    print_ts(f"memory: converted {mem} into an index of {len(items)} topic file(s); backup {backup}")
    return new


def ensure_index(agent_dir: str) -> str:
    """Return MEMORY.md as an index, converting an old-style file once.
    O_EXCL lock so two runners never convert the same agent at once."""
    content = _read(memory_md_path(agent_dir))
    if not content.strip() or is_index(content):
        return content
    lock = memory_md_path(agent_dir) + ".convert.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock) > _LOCK_STALE_S:
                os.remove(lock)
        except OSError:
            pass
        return content  # another worker is converting; next turn picks it up
    try:
        os.close(fd)
        return convert(agent_dir)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def load_index_block(agent_dir: str) -> str:
    """The system-prompt block for this agent's memory index ('' if none)."""
    try:
        content = ensure_index(agent_dir)
    except Exception as e:
        print_ts(f"memory: index conversion failed for {agent_dir}, loading as-is: {e}", error=True)
        content = _read(memory_md_path(agent_dir))
    if not content.strip():
        return ("# Memory index (MEMORY.md)\n\nEmpty. When you save a memory with "
                "save_memory(topic=...), it gets a topic file and a line here.")
    return ("# Memory index (MEMORY.md, loaded automatically every turn)\n\n"
            "Each line points to a topic file with the details. Open one with "
            "read_memory(file=\"topics/<slug>\") when it matters. Save lasting facts "
            "with save_memory(text, topic=...).\n\n" + cap_index(content))


def upsert_topic(agent_dir: str, slug: str, title: str, text: str, *, replace: bool = False) -> tuple[str, bool]:
    """Append (or with replace=True overwrite) a topic file's body and make sure
    MEMORY.md has its index line. Returns (topic_path, created)."""
    ensure_index(agent_dir)
    path = topic_path(agent_dir, slug)
    existing = _read(path)
    created = not existing
    if existing and not replace:
        body = topic_body(existing) + "\n" + text.strip()
    else:
        body = text.strip()
    if existing and not title:
        m = re.search(r"^# (.+)$", existing, re.M)
        title = m.group(1).strip() if m else slug
    title = title or slug.replace("_", " ").capitalize()
    hook = make_hook(body)
    _atomic_write(path, topic_text(slug, title, hook, body))

    mem = memory_md_path(agent_dir)
    lines = _read(mem).splitlines() or ["# Memory index", ""]
    want = index_line(title, slug, hook)
    rel = topic_rel(slug)
    for i, line in enumerate(lines):
        m = _INDEX_LINE_RE.match(line.strip())
        if m and m.group("path") == rel:
            if line == want:
                return path, created  # index unchanged: keep the prompt cache
            lines[i] = want
            break
    else:
        lines.append(want)
    _atomic_write(mem, "\n".join(lines).rstrip("\n") + "\n")
    return path, created


def list_topic_files(agent_dir: str) -> list[str]:
    d = topics_dir(agent_dir)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".md") and valid_slug(f[:-3]))
