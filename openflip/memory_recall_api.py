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


def selector_model(provider: str = "") -> str:
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
         if (meta or {}).get("provider") in ("anthropic", "claude_code")
         and "sonnet" in n and not n.endswith("-1m")),
        reverse=True,
    )
    if sonnets:
        return sonnets[0]
    # claude_code can run on Claude Code's own default without a config entry.
    return "sonnet" if provider == "claude_code" else ""


async def call_selector(model: str, candidates_text: str, query: str, *, provider: str = "",
                        anthropic_auth: str = "") -> str:
    """Return the selector's raw text answer. Raises on any failure (the
    caller swallows and logs). On the claude_code provider the selector is a
    one-shot Claude Code query (no OAuth handling here)."""
    if provider == "claude_code":
        from .claude_code_conversation import call_selector as _cc_selector
        return await _cc_selector(model, candidates_text, query)
    import aiohttp
    from .anthropic_conversation import _load_oauth_access_token, _DEFAULT_API_BASE, build_auth
    from .utils import http_session
    if anthropic_auth == "api_key":
        from .config_global import get_anthropic_api_key
        key = get_anthropic_api_key()
        if not key:
            raise RuntimeError("Anthropic API key not configured")
        headers, prefix_blocks = build_auth("api_key", api_key=key)
    else:
        token = await _load_oauth_access_token()
        if not token:
            raise RuntimeError("OAuth token unavailable")
        headers, prefix_blocks = build_auth("oauth", access_token=token)
    body = {
        "model": model,
        "max_tokens": SELECTOR_MAX_TOKENS,
        "system": list(prefix_blocks) + [
            {"type": "text", "text": SELECTOR_SYSTEM},
        ],
        "messages": [
            {"role": "user", "content": f"Available memory files:\n{candidates_text}"},
            {"role": "assistant", "content": "Understood. Send the query."},
            {"role": "user", "content": (f"Select memories relevant to:\n{query}\n\n"
                                         "Answer with JSON only: {\"selected_memories\": [\"<filename>\", ...]}")},
        ],
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
