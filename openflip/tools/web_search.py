"""Web search via a local SearXNG instance (JSON API).

Hits the SearXNG /search endpoint with format=json directly — no wrapper
service needed. SearXNG must have 'json' in its search.formats setting.

Owner controls (via /toolset):
  count, categories, language, time_range, engines.
  host comes from config.json (searxng_host).
"""
from __future__ import annotations
import aiohttp

from ._base import tool, ToolResult
from ..config_global import get_config
from .. import tool_settings as ts
from ..utils import print_ts, http_session, COLOR_YELLOW, COLOR_END


ts.register("web_search", [
    ts.SettingSchema("count", "int", 8,
        "How many results to return to the model.", min=1, max=25),
    ts.SettingSchema("categories", "choice", "general",
        "SearXNG category to query. Empty = no category filter (recommended if your SearXNG has no engines tagged for the chosen category).",
        choices=["", "general", "news", "images", "videos", "files", "it", "science", "social media", "music", "map"]),
    ts.SettingSchema("language", "str", "auto",
        "Language code (e.g. 'en', 'auto'). 'auto' lets SearXNG infer."),
    ts.SettingSchema("time_range", "choice", "",
        "Restrict to recent results. Empty = no constraint.",
        choices=["", "day", "month", "year"]),
    ts.SettingSchema("engines", "str", "",
        "Comma-separated SearXNG engines. Empty = all enabled engines."),
])


def _searxng_host() -> str:
    return get_config().get("searxng_host", "http://127.0.0.1:8888").rstrip("/")


@tool
async def web_search(query: str) -> ToolResult:
    """Search the web for current information — news, facts, how-to, definitions, recent events, anything you don't already know or aren't sure about. Returns a list of result titles, URLs, and snippets the user can read. Use whenever the user asks something that needs up-to-date or external knowledge.

    Args:
        query: What to search for. Plain natural language is fine.
    """
    q = (query or "").strip()
    if not q:
        return ToolResult.fail("Empty search query.")

    count = ts.get("web_search", "count")
    params = {
        "q": q,
        "format": "json",
        "language": ts.get("web_search", "language"),
    }
    cat = ts.get("web_search", "categories")
    if cat:
        params["categories"] = cat
    tr = ts.get("web_search", "time_range")
    if tr:
        params["time_range"] = tr
    eng = ts.get("web_search", "engines")
    if eng:
        params["engines"] = eng

    url = f"{_searxng_host()}/search"
    print_ts(f"{COLOR_YELLOW}web_search: {q!r} (count={count}, cat={params.get('categories') or '(none)'}){COLOR_END}")
    try:
        session = await http_session()
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                hint = ""
                if resp.status == 403 or "html" in (resp.content_type or ""):
                    hint = (
                        " — SearXNG may not have JSON format enabled. "
                        "Add 'json' to search.formats in your SearXNG settings.yml."
                    )
                return ToolResult.fail(f"SearXNG returned HTTP {resp.status}: {body}{hint}")
            data = await resp.json(content_type=None)
    except Exception as e:
        return ToolResult.fail(f"Search request failed: {e}")

    if not isinstance(data, dict) or "results" not in data:
        return ToolResult.fail(
            "SearXNG returned unexpected response (no 'results' key). "
            "Ensure JSON format is enabled: add 'json' to search.formats in SearXNG settings.yml."
        )

    results = data["results"][:count]
    if not results:
        return ToolResult(text=f"No results for {q!r}.")

    # Format for the model: numbered list with title, url, snippet. Keep it
    # short — one ToolResult.text block, not multi-attachment. The model
    # consumes this directly and either summarizes or quotes URLs back.
    lines = [f"Search results for {q!r}:"]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip() or "(no title)"
        u = (r.get("url") or "").strip()
        snippet = " ".join((r.get("content") or "").split())  # collapse whitespace
        if len(snippet) > 350:
            snippet = snippet[:347] + "…"
        lines.append(f"{i}. {title}")
        if u:
            lines.append(f"   {u}")
        if snippet:
            lines.append(f"   {snippet}")
    return ToolResult(text="\n".join(lines))


# Don't dump raw search hits into Discord — the model's summary is the answer.
# Model still sees the full text via the agent loop's role=tool feedback.
from ._base import TOOL_REGISTRY as _R
if "web_search" in _R:
    _R["web_search"].silent_to_discord = True
