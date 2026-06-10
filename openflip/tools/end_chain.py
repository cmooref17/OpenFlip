"""Routing no-op: explicitly end a chain-terminator turn without dispatching to anyone."""
from ._base import tool, ToolResult

@tool
async def end_chain() -> ToolResult:
    """End the chain without sending anything. Use when there's nothing to relay to the human or the peer.

    This tool exists for chain-terminator turns where silence is the correct response. Calling it explicitly
    is required when no message is needed — you can't end a chain-terminator turn without calling something.
    """
    return ToolResult(model_feedback="Chain ended.")
