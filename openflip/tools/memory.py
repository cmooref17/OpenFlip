"""Agent memory tools — two-tier memory system.

Two storage tiers:
    agents/<id>/MEMORY.md          # Curated core knowledge (agent root)
    agents/<id>/memory/            # Daily logs + search index
    ├── YYYY-MM-DD.md              # Daily event log
    └── index.json                 # Embedding vectors for search

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
    """Resolve a read_memory file argument to an absolute path."""
    file_arg = file_arg.strip()
    if not file_arg or file_arg.upper() in ("MEMORY.MD", "MEMORY"):
        return _memory_md_path(agent_dir)
    # Strip .md suffix for normalization, then re-add
    base = file_arg.removesuffix(".md")
    if _DATE_RE.match(base):
        return _daily_file_path(agent_dir, base)
    return ""  # invalid


def _remove_source_entries(index: dict, source: str) -> None:
    """Remove all index entries matching a source file."""
    index["entries"] = [e for e in index.get("entries", []) if e.get("source") != source]


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

@tool
async def save_memory(text: str) -> ToolResult:
    """Save a memory to today's daily log. Use this for events, decisions, facts, preferences, or anything worth remembering. Entries are timestamped automatically.

    Args:
        text: What to remember — a clear statement of the event, fact, or decision.
    """
    agent_dir = _get_agent_dir()
    _maybe_migrate(agent_dir)

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
async def update_core_memory(content: str) -> ToolResult:
    """Replace your core memory file (MEMORY.md) with updated content. IMPORTANT: Read your current core memory first with read_memory(), then pass the complete updated version here. This overwrites the entire file.

    Args:
        content: The complete new content for MEMORY.md. Include everything you want to keep.
    """
    agent_dir = _get_agent_dir()
    _maybe_migrate(agent_dir)

    mem_path = _memory_md_path(agent_dir)

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
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

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
async def search_memory(query: str) -> ToolResult:
    """Search your memories by semantic similarity. Searches across your core memory (MEMORY.md) and all daily logs. Returns the most relevant chunks with their source files.

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
        source_label = "MEMORY.md" if source == "MEMORY.md" else source.removesuffix(".md")
        chunk = entry.get("chunk", "(content unavailable)")
        results.append(f"[source: {source_label}] (relevance: {score:.2f})\n{chunk}")

    if not results:
        return ToolResult(model_feedback="No relevant memories found for that query.")

    return ToolResult(
        model_feedback="Relevant memories:\n\n" + "\n\n---\n\n".join(results),
    )


@tool
async def read_memory(file: str = "") -> ToolResult:
    """Read your core memory (MEMORY.md) or a specific daily log. Defaults to MEMORY.md if no file specified.

    Args:
        file: Which file to read. Leave empty for MEMORY.md, or pass a date like '2026-05-06' for a daily log.
    """
    agent_dir = _get_agent_dir()
    _maybe_migrate(agent_dir)

    resolved = _resolve_file_arg(agent_dir, file)
    if not resolved:
        return ToolResult.fail(f"Invalid file '{file}'. Use a date like '2026-05-06' or leave empty for MEMORY.md.")

    if not os.path.exists(resolved):
        if resolved == _memory_md_path(agent_dir):
            return ToolResult(model_feedback="No core memory file yet. Use update_core_memory to create one, or save_memory to start logging.")
        return ToolResult(model_feedback=f"No daily log for {file.strip().removesuffix('.md')}.")

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return ToolResult.fail(f"Failed to read memory file: {e}")

    label = "MEMORY.md" if resolved == _memory_md_path(agent_dir) else os.path.basename(resolved)
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
        lines.append(f"- MEMORY.md ({_fmt_size(size)}) — core memory")

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
