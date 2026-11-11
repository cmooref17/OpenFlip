"""tool_search: load deferred tools' full definitions on demand (Claude Code's ToolSearch).

Deferred tools are sent to the API with `defer_loading: true`, so only their
names are in the model's context (listed in the system prompt). Calling this
returns `tool_reference` blocks; the Anthropic API expands them into the full
definitions, after which the tools are callable like any other. See
openflip/tool_deferral.py and the Anthropic "tool search tool" docs (custom
client-side implementation).
"""
from __future__ import annotations

from ._base import tool, ToolResult


@tool
async def tool_search(query: str, max_results: int = 5) -> ToolResult:
    """Fetches full schema definitions for deferred tools so they can be called.

    Deferred tools are listed by name in your system prompt. Until fetched, only the name is known: there is no parameter schema, so calling the tool fails. When any instruction, reminder, or other tool's description names a deferred tool, fetch it with query "select:<name>" before calling it. Batch every tool you expect to need into ONE call. Once a tool's schema is returned, it stays callable for the rest of the conversation.

    Args:
        query: "select:name1,name2" to fetch exact tools by name; or keywords ("image edit") to search names and descriptions; or "+word rest" to require a word in the name.
        max_results: Max matches for a keyword search (default 5).
    """
    from ..tool_deferral import search_deferred
    names, note = search_deferred(query, max_results=max_results)
    if not names:
        return ToolResult(model_feedback=note or f"No deferred tools matched {query!r}.")
    return ToolResult(model_feedback=note or "", tool_references=names)
