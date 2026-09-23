"""Agent memory tools — Claude-Code-style index + topic files, plus daily logs.

Storage:
    agents/<id>/MEMORY.md          # INDEX: one line per topic file. Loaded into
                                   # the system prompt every turn (runtime.py,
                                   # openflip/memory_index.py), capped 200 lines / 25k chars.
    agents/<id>/memory/
    ├── topics/<slug>.md           # One file per subject; opened on demand
    ├── YYYY-MM-DD.md              # Daily event log
    └── index.json                 # Embedding vectors for search

save_memory(text, topic=...) appends to a topic file and keeps its index line
current; without a topic it goes to today's daily log. An old-style free-form
MEMORY.md converts itself into index + topic files on first touch (backup kept).

Embeddings via Ollama /api/embed (nomic-embed-text, 768-dim).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import time

import aiohttp

from ._base import tool, ToolResult
from ..config_global import get_config
from ..snapshots import snapshot_file
from ..utils import load_json, save_json, print_ts, http_session


# ── Helpers ───────────────────────────────────────────────────────────────

def _get_agent_dir() -> str:
    from ..tool_executor import CURRENT_AGENT
    agent = CURRENT_AGENT.get(None)
    if not agent:
        raise RuntimeError("No agent context available")
    return os.path.dirname(agent.path)


def _memory_dir(agent_dir: str) -> str:
    return os.path.join(agent_dir, "memory")


def _memory_md_path(agent_dir: str) -> str:
    return os.path.join(agent_dir, "MEMORY.md")


def _daily_file_path(agent_dir: str, date_str: str) -> str:
    return os.path.join(agent_dir, "memory", date_str + ".md")


def _index_path(agent_dir: str) -> str:
    return os.path.join(agent_dir, "memory", "index.json")


async def _get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama."""
    config = get_config()
    host = config.get("ollama_host", "http://localhost:11434")
    model = config.get("embedding_model", "nomic-embed-text")
    session = await http_session()
    async with session.post(
        f"{host}/api/embed",
        json={"model": model, "input": text},
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Ollama embed returned {resp.status}: {body[:200]}")
        data = await resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise RuntimeError("No embeddings returned from Ollama")
        return embeddings[0]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _resolve_file_arg(agent_dir: str, file_arg: str) -> str:
    """Resolve a read_memory file argument to an absolute path: MEMORY.md, a
    daily log date, or a topic file (`topics/<slug>`, `memory/topics/<slug>.md`,
    or the bare slug)."""
    from .. import memory_index as mi
    file_arg = file_arg.strip()
    if not file_arg or file_arg.upper() in ("MEMORY.MD", "MEMORY"):
        return _memory_md_path(agent_dir)
    # Strip .md suffix for normalization, then re-add
    base = file_arg.removesuffix(".md")
    if _DATE_RE.match(base):
        return _daily_file_path(agent_dir, base)
    slug = base.removeprefix("memory/").removeprefix("topics/")
    if mi.valid_slug(slug):
        return mi.topic_path(agent_dir, slug)
    return ""  # invalid


def _remove_source_entries(index: dict, source: str) -> None:
    """Remove all index entries matching a source file."""
    index["entries"] = [e for e in index.get("entries", []) if e.get("source") != source]


_MAX_CHUNK_CHARS = 2000

def _split_oversized(chunk: str) -> list[str]:
    """Split a chunk into <= _MAX_CHUNK_CHARS pieces at whitespace boundaries.
    Guards the embedding model's context window (nomic-embed-text ~8192 tokens);
    2000 chars stays safely under even in worst-case ~1 token/char content."""
    chunk = chunk.strip()
    if len(chunk) <= _MAX_CHUNK_CHARS:
        return [chunk] if chunk else []
    pieces = []
    while len(chunk) > _MAX_CHUNK_CHARS:
        cut = chunk.rfind(" ", 0, _MAX_CHUNK_CHARS)
        if cut <= 0:
            cut = _MAX_CHUNK_CHARS  # no whitespace: hard slice
        piece = chunk[:cut].strip()
        if piece:
            pieces.append(piece)
        chunk = chunk[cut:].strip()
    if chunk:
        pieces.append(chunk)
    return pieces


def _chunk_paragraphs(content: str) -> list[str]:
    """Split content into stripped, non-empty paragraph chunks (blank-line separated)."""
    return [piece for p in content.split("\n\n") if p.strip() for piece in _split_oversized(p.strip())]


_DAILY_BULLET_RE = re.compile(r"^- \[\d{2}:\d{2}\] ")
_DAILY_HEADER_RE = re.compile(r"^# \d{4}-\d{2}-\d{2}\s*$")


def _chunk_daily_log(content: str) -> list[str]:
    """Chunk a daily log the way save_memory's indexed form looks: one chunk
    per timestamped bullet with the `- [HH:MM] ` prefix stripped (so hashes
    match the chunks save_memory indexed). Non-bullet lines — hand-edited
    prose — fall back to paragraph chunks so they stay searchable; the
    `# YYYY-MM-DD` header line is skipped."""
    chunks: list[str] = []
    residual: list[str] = []
    for line in content.splitlines():
        m = _DAILY_BULLET_RE.match(line)
        if m:
            chunks.append(line[m.end():])
        elif not _DAILY_HEADER_RE.match(line):
            residual.append(line)
    chunks.extend(_chunk_paragraphs("\n".join(residual)))
    return [piece for c in chunks if c.strip() for piece in _split_oversized(c)]


# ── Migration ─────────────────────────────────────────────────────────────

def _maybe_migrate(agent_dir: str) -> None:
    """Migrate v1 memory format (entries/*.md) to v2 (daily files). Idempotent."""
    index_path = _index_path(agent_dir)
    index = load_json(index_path, default={"entries": []})

    if index.get("version") == 2:
        return

    old_entries = index.get("entries", [])
    if not old_entries:
        save_json(index_path, {"version": 2, "entries": []})
        return

    new_entries = []
    entries_dir = os.path.join(agent_dir, "memory", "entries")

    for entry in old_entries:
        # Read old entry file
        old_file = os.path.join(entries_dir, entry.get("file", ""))
        try:
            with open(old_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except (FileNotFoundError, OSError):
            content = entry.get("preview", "")

        if not content:
            continue

        # Extract date from old ID (e.g. "2026-05-05_205612")
        old_id = entry.get("id", "")
        date_str = old_id[:10] if len(old_id) >= 10 and _DATE_RE.match(old_id[:10]) else time.strftime("%Y-%m-%d")
        daily_path = _daily_file_path(agent_dir, date_str)

        os.makedirs(os.path.dirname(daily_path), exist_ok=True)
        if not os.path.exists(daily_path):
            with open(daily_path, "w", encoding="utf-8") as f:
                f.write(f"# {date_str}\n\n")

        # Extract time from old timestamp
        old_ts = entry.get("timestamp", "")
        time_part = old_ts[11:16] if len(old_ts) >= 16 else "00:00"

        with open(daily_path, "a", encoding="utf-8") as f:
            f.write(f"- [{time_part}] {content}\n")

        new_entries.append({
            "source": f"{date_str}.md",
            "chunk": content,
            "embedding": entry.get("embedding", []),
            "timestamp": entry.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
        })

    save_json(index_path, {"version": 2, "entries": new_entries})

    # Clean up old entries directory
    if os.path.isdir(entries_dir):
        try:
            shutil.rmtree(entries_dir)
            print_ts(f"Migrated {len(new_entries)} memory entries to daily files, removed entries/")
        except OSError:
            pass


# ── Tools ─────────────────────────────────────────────────────────────────

async def _index_topic_file(agent_dir: str, slug: str) -> None:
    """Re-embed one topic file into the search index (replaces its old chunks)."""
    from .. import memory_index as mi
    source = mi.topic_rel(slug)
    try:
        with open(mi.topic_path(agent_dir, slug), encoding="utf-8") as f:
            chunks = _chunk_paragraphs(f.read())
    except OSError:
        chunks = []
    index_path = _index_path(agent_dir)
    index = load_json(index_path, default={"version": 2, "entries": []})
    index.setdefault("version", 2)
    old = {e.get("hash"): e.get("embedding") for e in index.get("entries", [])
           if e.get("source") == source and e.get("hash") and e.get("embedding")}
    fresh = []
    for i, chunk in enumerate(chunks):
        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        emb = old.get(h) or await _get_embedding(chunk)
        fresh.append({"source": source, "chunk": chunk, "chunk_index": i, "embedding": emb,
                      "hash": h, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    _remove_source_entries(index, source)
    index["entries"].extend(fresh)
    save_json(index_path, index)


@tool
async def save_memory(text: str, topic: str = "", description: str = "", type: str = "", replace: bool = False) -> ToolResult:
    """Save a memory. With `topic`, it's a lasting fact: appended to that topic file (created if new) and listed in your MEMORY.md index, which loads automatically every turn. Without `topic`, it goes to today's daily log (events, one-off notes). One topic per SUBJECT (a person, a project, a standing rule), organized by subject, not by date or incident: reuse an existing topic from your index when one fits, and fix a wrong fact instead of adding a second one. For a rule or a decision, write the rule first, then a "Why:" line and a "How to apply:" line. How to write memories is in FRAMEWORK.md's Memory section.

    Args:
        text: What to remember, stated as a durable fact or rule (absolute dates, no "yesterday"). With replace=True, the topic's complete new body.
        topic: Optional. Topic slug for a lasting fact, e.g. "operator_environment". Leave empty for a daily-log entry.
        description: Optional. One line (under ~150 chars) saying what the whole topic covers. Recall picks files by it, so make it specific. Give it when creating a topic or when its scope changes; otherwise the existing one is kept.
        type: Optional. user (who the operator is, their preferences), feedback (how they want you to work: corrections AND approaches they confirmed), project (ongoing work, decisions, deadlines), or reference (where to find something).
        replace: Optional. True overwrites the topic's body with `text` instead of appending. Use it to rewrite or merge a topic.
    """
    agent_dir = _get_agent_dir()
    _maybe_migrate(agent_dir)

    if topic.strip():
        from .. import memory_index as mi
        raw = topic.strip().removesuffix(".md").removeprefix("memory/").removeprefix("topics/")
        slug = raw if mi.valid_slug(raw) else mi.slugify(raw)
        title = "" if raw == slug else raw
        mtype = (type or "").strip().lower()
        if mtype and mtype not in mi.MEMORY_TYPES:
            return ToolResult.fail(f"type must be one of {', '.join(mi.MEMORY_TYPES)} (or empty), not '{type}'.")
        entry = text.strip() if replace else f"- [{time.strftime('%Y-%m-%d')}] {text.strip()}"
        if replace and os.path.isfile(mi.topic_path(agent_dir, slug)):
            try:
                snapshot_file(mi.topic_path(agent_dir, slug))
            except Exception as e:
                print_ts(f"topic snapshot failed (proceeding with rewrite): {e}", error=True)
        _, created = mi.upsert_topic(agent_dir, slug, title, entry, replace=replace,
                                     description=description, mtype=mtype)
        note = ("new topic file + index line" if created
                else "rewritten; index line refreshed" if replace else "appended; index line refreshed")
        try:
            await _index_topic_file(agent_dir, slug)
        except Exception as e:
            print_ts(f"Topic memory saved but indexing failed: {e}", error=True)
            return ToolResult(model_feedback=f"Saved to {mi.topic_rel(slug)} ({note}); search indexing failed: {e}")
        try:
            from .. import events_log as _events_log
            from ..tool_executor import CURRENT_AGENT
            _aid = (CURRENT_AGENT.get(None).id if CURRENT_AGENT.get(None) else "")
            _events_log.log_event(_aid, "memory_write", target="topic", topic=slug, preview=text[:120])
        except Exception:
            pass
        return ToolResult(model_feedback=f"Saved to {mi.topic_rel(slug)} ({note}): {text[:100]}")

    date_str = time.strftime("%Y-%m-%d")
    daily_path = _daily_file_path(agent_dir, date_str)
    os.makedirs(os.path.dirname(daily_path), exist_ok=True)

    if not os.path.exists(daily_path):
        with open(daily_path, "w", encoding="utf-8") as f:
            f.write(f"# {date_str}\n\n")

    time_str = time.strftime("%H:%M")
    with open(daily_path, "a", encoding="utf-8") as f:
        f.write(f"- [{time_str}] {text}\n")

    # Embed and index
    index_path = _index_path(agent_dir)
    index = load_json(index_path, default={"version": 2, "entries": []})
    index.setdefault("version", 2)

    try:
        embedding = await _get_embedding(text)
        index["entries"].append({
            "source": f"{date_str}.md",
            "chunk": text,
            "embedding": embedding,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        save_json(index_path, index)
    except Exception as e:
        print_ts(f"Memory saved but indexing failed: {e}", error=True)
        return ToolResult(model_feedback=f"Saved to {date_str} log (search indexing failed: {e})")

    try:
        from .. import events_log as _events_log
        from ..tool_executor import CURRENT_AGENT
        _aid = (CURRENT_AGENT.get(None).id if CURRENT_AGENT.get(None) else "")
        _events_log.log_event(
            _aid, "memory_write",
            target="daily", date=date_str, preview=text[:120],
        )
    except Exception:
        pass
    return ToolResult(model_feedback=f"Saved to {date_str} log: {text[:100]}")


@tool
async def delete_memory(topic: str) -> ToolResult:
    """Delete a topic file and its MEMORY.md index line, e.g. after merging it into another topic or when nothing in it is still true. The file is snapshotted first, so restore_snapshot can undo it.

    Args:
        topic: The topic slug to delete, e.g. "old_incident_notes".
    """
    agent_dir = _get_agent_dir()
    from .. import memory_index as mi
    slug = topic.strip().removesuffix(".md").removeprefix("memory/").removeprefix("topics/")
    if not mi.valid_slug(slug):
        return ToolResult.fail(f"Invalid topic '{topic}'.")
    path = mi.topic_path(agent_dir, slug)
    if os.path.isfile(path):
        try:
            snapshot_file(path)
        except Exception as e:
            print_ts(f"topic snapshot failed (proceeding with delete): {e}", error=True)
    if not mi.delete_topic(agent_dir, slug):
        return ToolResult.fail(f"No topic '{slug}' found.")
    index_path = _index_path(agent_dir)
    index = load_json(index_path, default={"version": 2, "entries": []})
    _remove_source_entries(index, mi.topic_rel(slug))
    save_json(index_path, index)
    return ToolResult(model_feedback=f"Deleted topic {mi.topic_rel(slug)} and its index line.")


@tool
async def update_core_memory(content: str) -> ToolResult:
    """Replace your memory INDEX (MEMORY.md) wholesale, e.g. to reorder, retitle, or drop stale lines. It must stay an index: one line per topic file, `- [Title](memory/topics/<slug>.md) — short hook`. Put details in topic files with save_memory(text, topic=...), never here — free-form content gets auto-split into topic files on the next turn. Read it first with read_memory().

    Args:
        content: The complete new MEMORY.md index.
    """
    agent_dir = _get_agent_dir()
    _maybe_migrate(agent_dir)

    mem_path = _memory_md_path(agent_dir)

    # Snapshot the existing MEMORY.md before overwriting — this tool replaces
    # the whole file, so a bad consolidation (e.g. a dream pass that reasoned
    # over a partial view) would otherwise be unrecoverable. snapshot_file()
    # never raises by contract, but a failed backup must never block the
    # write, so guard anyway.
    if os.path.isfile(mem_path):
        try:
            snapshot_file(mem_path)
        except Exception as e:
            print_ts(f"MEMORY.md snapshot failed (proceeding with write): {e}", error=True)

    # Atomic write
    tmp_path = mem_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, mem_path)

    # Re-index MEMORY.md
    index_path = _index_path(agent_dir)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    index = load_json(index_path, default={"version": 2, "entries": []})
    index.setdefault("version", 2)

    # Snapshot existing MEMORY.md embeddings by content hash before stripping
    # them from the index. update_core_memory rewrites the entire file, but
    # most paragraphs are usually unchanged — reusing their embeddings avoids
    # re-paying Ollama for identical work on every call.
    old_by_hash: dict[str, list[float]] = {}
    for prev in index.get("entries", []):
        if prev.get("source") != "MEMORY.md":
            continue
        ph = prev.get("hash")
        emb = prev.get("embedding")
        if ph and emb:
            old_by_hash[ph] = emb

    _remove_source_entries(index, "MEMORY.md")

    if content.strip():
        paragraphs = _chunk_paragraphs(content)

        indexed = 0
        reused = 0
        for i, para in enumerate(paragraphs):
            para_hash = hashlib.sha256(para.encode("utf-8")).hexdigest()
            embedding = old_by_hash.get(para_hash)
            if embedding is not None:
                reused += 1
            else:
                try:
                    embedding = await _get_embedding(para)
                except Exception as e:
                    print_ts(f"Failed to embed MEMORY.md paragraph {i}: {e}", error=True)
                    continue
            index["entries"].append({
                "source": "MEMORY.md",
                "chunk": para,
                "chunk_index": i,
                "embedding": embedding,
                "hash": para_hash,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            indexed += 1

        save_json(index_path, index)
        try:
            from .. import events_log as _events_log
            from ..tool_executor import CURRENT_AGENT
            _aid = (CURRENT_AGENT.get(None).id if CURRENT_AGENT.get(None) else "")
            _events_log.log_event(
                _aid, "memory_write",
                target="core", chars=len(content),
                indexed=indexed, reused=reused,
            )
        except Exception:
            pass
        return ToolResult(model_feedback=f"Core memory updated ({len(content)} chars, {indexed}/{len(paragraphs)} paragraphs indexed, {reused} reused)")

    save_json(index_path, index)
    try:
        from .. import events_log as _events_log
        from ..tool_executor import CURRENT_AGENT
        _aid = (CURRENT_AGENT.get(None).id if CURRENT_AGENT.get(None) else "")
        _events_log.log_event(_aid, "memory_write", target="core", chars=0, cleared=True)
    except Exception:
        pass
    return ToolResult(model_feedback="Core memory cleared.")


@tool
async def reindex_memory() -> ToolResult:
    """Rebuild your memory search index from the files on disk (MEMORY.md + topic files + all daily logs). Use after a memory file was edited outside the memory tools (file tools, the operator, another process) so search_memory matches the files again. Unchanged chunks keep their existing embeddings.
    """
    agent_dir = _get_agent_dir()
    _maybe_migrate(agent_dir)

    index_path = _index_path(agent_dir)
    index = load_json(index_path, default={"version": 2, "entries": []})
    index.setdefault("version", 2)

    # Reuse embeddings for unchanged chunks across ALL sources (generalizes
    # update_core_memory's old_by_hash reuse). Daily entries don't store a
    # hash, so hash their chunk text here to match.
    old_by_hash: dict[str, dict] = {}
    for prev in index.get("entries", []):
        chunk = prev.get("chunk")
        if not chunk or not prev.get("embedding"):
            continue
        h = prev.get("hash") or hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        old_by_hash[h] = prev

    # Build the complete new entries list in memory first — index.json is only
    # replaced after every chunk has an embedding, so an Ollama failure leaves
    # the existing index stale but consistent rather than partially rebuilt.
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    new_entries: list[dict] = []
    pending: list[dict] = []
    reused = 0
    sources = 0

    def _stage(entry: dict, chash: str) -> None:
        nonlocal reused
        prev = old_by_hash.get(chash)
        if prev is not None:
            entry["embedding"] = prev["embedding"]
            entry["timestamp"] = prev.get("timestamp", entry["timestamp"])
            reused += 1
        else:
            pending.append(entry)
        new_entries.append(entry)

    mem_path = _memory_md_path(agent_dir)
    if os.path.isfile(mem_path):
        try:
            with open(mem_path, "r", encoding="utf-8") as f:
                core_chunks = _chunk_paragraphs(f.read())
        except OSError as e:
            return ToolResult.fail(f"Reindex aborted (existing index untouched): failed to read MEMORY.md: {e}")
        if core_chunks:
            sources += 1
        for i, chunk in enumerate(core_chunks):
            chash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            _stage({
                "source": "MEMORY.md",
                "chunk": chunk,
                "chunk_index": i,
                "embedding": None,
                "hash": chash,
                "timestamp": now,
            }, chash)

    from .. import memory_index as mi
    for tname in mi.list_topic_files(agent_dir):
        try:
            with open(os.path.join(mi.topics_dir(agent_dir), tname), "r", encoding="utf-8") as f:
                topic_chunks = _chunk_paragraphs(f.read())
        except OSError as e:
            return ToolResult.fail(f"Reindex aborted (existing index untouched): failed to read topic {tname}: {e}")
        if topic_chunks:
            sources += 1
        tsource = mi.topic_rel(tname.removesuffix(".md"))
        for i, chunk in enumerate(topic_chunks):
            chash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            _stage({
                "source": tsource,
                "chunk": chunk,
                "chunk_index": i,
                "embedding": None,
                "hash": chash,
                "timestamp": now,
            }, chash)

    mem_dir = _memory_dir(agent_dir)
    daily_names = []
    if os.path.isdir(mem_dir):
        daily_names = sorted(
            f for f in os.listdir(mem_dir)
            if f.endswith(".md") and _DATE_RE.match(f.removesuffix(".md"))
        )
    for fname in daily_names:
        try:
            with open(os.path.join(mem_dir, fname), "r", encoding="utf-8") as f:
                daily_chunks = _chunk_daily_log(f.read())
        except OSError as e:
            return ToolResult.fail(f"Reindex aborted (existing index untouched): failed to read {fname}: {e}")
        if daily_chunks:
            sources += 1
        for chunk in daily_chunks:
            _stage({
                "source": fname,
                "chunk": chunk,
                "embedding": None,
                "timestamp": now,
            }, hashlib.sha256(chunk.encode("utf-8")).hexdigest())

    embedded = 0
    for entry in pending:
        try:
            entry["embedding"] = await _get_embedding(entry["chunk"])
        except Exception as e:
            return ToolResult.fail(f"Reindex aborted (existing index untouched): embedding failed for a chunk from {entry['source']}: {e}")
        embedded += 1

    index["entries"] = new_entries
    save_json(index_path, index)

    try:
        from .. import events_log as _events_log
        from ..tool_executor import CURRENT_AGENT
        _aid = (CURRENT_AGENT.get(None).id if CURRENT_AGENT.get(None) else "")
        _events_log.log_event(
            _aid, "memory_write",
            target="reindex", entries=len(new_entries),
            reused=reused, embedded=embedded, sources=sources,
        )
    except Exception:
        pass

    return ToolResult(model_feedback=(
        f"Memory index rebuilt: {len(new_entries)} entries across {sources} source file(s) "
        f"({reused} embeddings reused from cache, {embedded} freshly embedded)."
    ))


@tool
async def search_memory(query: str) -> ToolResult:
    """Search your memories by semantic similarity. Searches your MEMORY.md index, every topic file, and all daily logs. Returns the most relevant chunks with their source files.

    Args:
        query: What to search for — a question or topic to find relevant memories about.
    """
    agent_dir = _get_agent_dir()
    _maybe_migrate(agent_dir)

    index_path = _index_path(agent_dir)
    index = load_json(index_path, default={"version": 2, "entries": []})
    entries = index.get("entries", [])

    if not entries:
        return ToolResult(model_feedback="No memories stored yet.")

    try:
        query_embedding = await _get_embedding(query)
    except Exception as e:
        return ToolResult.fail(f"Failed to generate search embedding: {e}")

    scored = []
    for entry in entries:
        emb = entry.get("embedding")
        if not emb:
            continue
        score = _cosine_similarity(query_embedding, emb)
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, entry in scored[:5]:
        if score < 0.3:
            continue
        source = entry.get("source", "unknown")
        source_label = "MEMORY.md" if source == "MEMORY.md" else source.removeprefix("memory/").removesuffix(".md")
        chunk = entry.get("chunk", "(content unavailable)")
        results.append(f"[source: {source_label}] (relevance: {score:.2f})\n{chunk}")

    if not results:
        return ToolResult(model_feedback="No relevant memories found for that query.")

    return ToolResult(
        model_feedback="Relevant memories:\n\n" + "\n\n---\n\n".join(results),
    )


@tool
async def read_memory(file: str = "") -> ToolResult:
    """Read your memory index (MEMORY.md), a topic file, or a daily log. Your index already loads every turn — use this to open the topic file a line points to.

    Args:
        file: Leave empty for MEMORY.md, pass a topic like 'topics/about_flip', or a date like '2026-05-06' for a daily log.
    """
    agent_dir = _get_agent_dir()
    _maybe_migrate(agent_dir)

    resolved = _resolve_file_arg(agent_dir, file)
    if not resolved:
        return ToolResult.fail(f"Invalid file '{file}'. Use a topic like 'topics/about_flip', a date like '2026-05-06', or leave empty for MEMORY.md.")

    if not os.path.exists(resolved):
        if resolved == _memory_md_path(agent_dir):
            return ToolResult(model_feedback="No memory index yet. save_memory(text, topic=...) creates a topic file and its index line.")
        if os.sep + "topics" + os.sep in resolved:
            return ToolResult(model_feedback=f"No topic file '{file.strip()}'. Check the line in your MEMORY.md index.")
        return ToolResult(model_feedback=f"No daily log for {file.strip().removesuffix('.md')}.")

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return ToolResult.fail(f"Failed to read memory file: {e}")

    label = "MEMORY.md" if resolved == _memory_md_path(agent_dir) else os.path.relpath(resolved, agent_dir)
    return ToolResult(model_feedback=f"--- {label} ---\n\n{content}")


@tool
async def list_memory_files() -> ToolResult:
    """List all your memory files with dates and sizes. Shows your core memory (MEMORY.md) and all daily logs.
    """
    agent_dir = _get_agent_dir()
    _maybe_migrate(agent_dir)

    lines = []
    mem_path = _memory_md_path(agent_dir)
    if os.path.exists(mem_path):
        size = os.path.getsize(mem_path)
        lines.append(f"- MEMORY.md ({_fmt_size(size)}) — index, loads every turn")

    from .. import memory_index as mi
    for tname in mi.list_topic_files(agent_dir):
        size = os.path.getsize(os.path.join(mi.topics_dir(agent_dir), tname))
        lines.append(f"- topics/{tname.removesuffix('.md')} ({_fmt_size(size)})")

    mem_dir = _memory_dir(agent_dir)
    if os.path.isdir(mem_dir):
        daily_files = sorted(
            [f for f in os.listdir(mem_dir) if f.endswith(".md")],
            reverse=True,
        )
        for fname in daily_files:
            fpath = os.path.join(mem_dir, fname)
            size = os.path.getsize(fpath)
            lines.append(f"- {fname.removesuffix('.md')} ({_fmt_size(size)})")

    if not lines:
        return ToolResult(model_feedback="No memory files yet.")

    return ToolResult(model_feedback="Memory files:\n" + "\n".join(lines))


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    return f"{n / 1024:.1f} KB"
