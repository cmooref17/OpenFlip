"""Selector model call for memory recall (see memory_recall.py).

One-shot, non-streaming Anthropic Messages call using the same OAuth token,
headers and billing block as the chat provider. Kept separate so the pure
recall logic stays testable without network."""
from __future__ import annotations

import os

from .memory_recall_prompt import SELECTOR_SYSTEM, SELECTOR_MAX_TOKENS, SELECTOR_TIMEOUT_S


def recall_enabled() -> bool:
    """Kill switch OPENFLIP_DISABLE_MEMORY_RECALL=1, or config.json
    `memory_recall.enabled: false`. Default on."""
    if os.environ.get("OPENFLIP_DISABLE_MEMORY_RECALL") == "1":
        return False
    from .config_global import get_config
    rc = (get_config() or {}).get("memory_recall") or {}
    return rc.get("enabled", True) is not False


def selector_model() -> str:
    """config.json `memory_recall.model`, else the newest anthropic sonnet in
    config's `models` block (CC uses the default Sonnet). '' = unavailable."""
    from .config_global import get_config
    cfg = get_config() or {}
    explicit = str((cfg.get("memory_recall") or {}).get("model") or "").strip()
    if explicit:
        return explicit.split("/", 1)[1] if "/" in explicit else explicit
    models = cfg.get("models") or {}
    sonnets = sorted(
        (n for n, meta in models.items()
         if (meta or {}).get("provider") == "anthropic" and "sonnet" in n and not n.endswith("-1m")),
        reverse=True,
    )
    return sonnets[0] if sonnets else ""


async def call_selector(model: str, candidates_text: str, query: str) -> str:
    """Return the selector's raw text answer. Raises on any failure (the
    caller swallows and logs)."""
    import aiohttp
    from .anthropic_conversation import (
        _load_oauth_access_token, _DEFAULT_API_BASE, _DEFAULT_ANTHROPIC_VERSION,
        _DEFAULT_USER_AGENT, _DETECTED_CC_VERSION,
    )
    from .utils import http_session
    token = await _load_oauth_access_token()
    if not token:
        raise RuntimeError("OAuth token unavailable")
    body = {
        "model": model,
        "max_tokens": SELECTOR_MAX_TOKENS,
        "system": [
            {"type": "text", "text": (f"x-anthropic-billing-header: cc_version={_DETECTED_CC_VERSION}; "
                                      f"cc_entrypoint=sdk-cli; cch=00000;")},
            {"type": "text", "text": SELECTOR_SYSTEM},
        ],
        "messages": [
            {"role": "user", "content": f"Available memory files:\n{candidates_text}"},
            {"role": "assistant", "content": "Understood. Send the query."},
            {"role": "user", "content": (f"Select memories relevant to:\n{query}\n\n"
                                         "Answer with JSON only: {\"selected_memories\": [\"<filename>\", ...]}")},
        ],
    }
    headers = {
        "authorization": f"Bearer {token}",
        "anthropic-version": _DEFAULT_ANTHROPIC_VERSION,
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
        "User-Agent": _DEFAULT_USER_AGENT,
        "content-type": "application/json",
    }
    session = await http_session()
    async with session.post(f"{_DEFAULT_API_BASE}/v1/messages", json=body, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=SELECTOR_TIMEOUT_S)) as resp:
        data = await resp.json(content_type=None)
        if resp.status != 200:
            raise RuntimeError(f"selector HTTP {resp.status}: {str(data)[:200]}")
    if data.get("stop_reason") == "max_tokens":
        raise RuntimeError("selector output truncated at max_tokens")
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
