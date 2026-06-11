"""SSE consumer for Anthropic /v1/messages with stream=true.

Parses Anthropic's server-sent events and accumulates them into the same
final structures the non-streaming response produces:
  - text_parts: list[str]
  - thinking_parts: list[str]
  - tool_calls: list[AnthropicToolCall]
  - usage: dict with input_tokens / output_tokens / cache_*
  - compaction_block: dict or None
  - stop_reason: str
  - final_obj: dict shaped like the non-streaming JSON response (for callers
    that still want to inspect raw content/usage)

This mirrors what claude code does in src/services/api/claude.ts around
line 1980-2300 ("case 'message_start' / 'content_block_start' / ...").

The motivation: when Anthropic streams a tool_use block, we KNOW the model
committed to a tool call by the time we see content_block_stop with type=
tool_use. With non-streaming, we only see tool_use after the whole response
arrives. Streaming makes the "said it without doing it" failure structurally
impossible in the downstream executor — though step 1 of the port (this
module) just consumes the stream and returns the same final result; the
deeper "dispatch tool as soon as content_block_stop arrives" change is a
later step that touches runtime.py.

Wire format reference:
  https://docs.anthropic.com/en/api/messages-streaming
"""
from __future__ import annotations
import json
from typing import AsyncIterator, Any


async def parse_sse_stream(resp_content: AsyncIterator[bytes]) -> dict:
    """Consume an SSE response body and return a dict shaped like the
    non-streaming /v1/messages JSON response.

    Returns:
        dict with keys: content (list of blocks), usage, stop_reason,
        and any compaction block surfaced. Shape matches what the existing
        non-streaming parser at anthropic_conversation.py expects from
        json.loads(text).
    """
    # Per-index accumulators. Indexes are 0-based block positions in the
    # final content array.
    content_blocks: dict[int, dict] = {}
    # Tool-use blocks accumulate their input as a JSON string fragment-by-
    # fragment in input_json_delta events; we re-parse at content_block_stop.
    tool_input_buffers: dict[int, str] = {}

    usage: dict = {}
    stop_reason: str | None = None
    final_message: dict | None = None

    # SSE line buffer. Anthropic frames events as:
    #   event: <name>
    #   data: <json>
    #   <blank line>
    current_event: str = ""
    current_data: list[str] = []

    # aiohttp may yield chunks that split mid-line (e.g. partway through
    # 'data: {...}'). We accumulate a byte buffer and only emit complete
    # lines (terminated by \n). Anything past the last \n stays buffered
    # for the next chunk.
    line_buffer: str = ""

    async for raw_chunk in resp_content:
        # aiohttp gives us bytes; SSE is utf-8 text lines.
        text = raw_chunk.decode("utf-8", errors="replace")
        line_buffer += text
        # Pull out complete lines; keep any incomplete trailing fragment.
        if "\n" not in line_buffer:
            continue
        parts = line_buffer.split("\n")
        # Last element is the incomplete trailing fragment (may be empty
        # if the chunk ended cleanly on a newline).
        line_buffer = parts[-1]
        complete_lines = parts[:-1]
        for raw_line in complete_lines:
            # Strip trailing CR for CRLF SSE servers.
            line = raw_line.rstrip("\r")
            if line == "":
                # Blank line = end of event. Dispatch.
                if current_event and current_data:
                    data_str = "\n".join(current_data)
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        current_event = ""
                        current_data = []
                        continue
                    _dispatch_event(
                        current_event, data,
                        content_blocks=content_blocks,
                        tool_input_buffers=tool_input_buffers,
                        usage_ref=usage,
                    )
                    if current_event == "message_delta":
                        # stop_reason and usage updates land here.
                        if isinstance(data.get("delta"), dict):
                            stop_reason = data["delta"].get("stop_reason") or stop_reason
                        if isinstance(data.get("usage"), dict):
                            for k, v in data["usage"].items():
                                usage[k] = v
                    elif current_event == "message_start":
                        final_message = data.get("message") or {}
                        if isinstance(final_message.get("usage"), dict):
                            for k, v in final_message["usage"].items():
                                usage[k] = v
                current_event = ""
                current_data = []
            elif line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                current_data.append(line[5:].lstrip())
            # Ignore comments (lines starting with ":") and other fields.

    # Finalize tool_use blocks: convert accumulated input strings to dicts.
    for idx, block in content_blocks.items():
        if block.get("type") == "tool_use":
            raw_input = tool_input_buffers.get(idx, "")
            if raw_input:
                try:
                    block["input"] = json.loads(raw_input)
                except json.JSONDecodeError:
                    # Model emitted invalid JSON — leave empty so the
                    # downstream malformed-tool_use detector catches it.
                    block["input"] = {}
            else:
                block["input"] = {}

    # Assemble final dict in the shape the existing non-streaming parser
    # expects from json.loads(text). content list is ordered by index.
    ordered_content = [content_blocks[i] for i in sorted(content_blocks.keys())]
    result = {
        "content": ordered_content,
        "usage": usage,
    }
    if stop_reason is not None:
        result["stop_reason"] = stop_reason
    if final_message:
        for k in ("id", "model", "role", "type"):
            if k in final_message:
                result[k] = final_message[k]
    return result


def _dispatch_event(
    event: str,
    data: dict,
    *,
    content_blocks: dict[int, dict],
    tool_input_buffers: dict[int, str],
    usage_ref: dict,
) -> None:
    """Apply one SSE event to the accumulators. No I/O, pure mutation."""
    if event == "content_block_start":
        idx = data.get("index", 0)
        block = dict(data.get("content_block") or {})
        btype = block.get("type")
        # Reset any per-type starting state. Mirrors claude code's switch
        # statement at claude.ts:1995.
        if btype == "text":
            # Anthropic occasionally emits initial text in the content_block_start
            # AND a duplicate via the first content_block_delta — drop the
            # start text and rely on deltas to avoid double-counting.
            block["text"] = ""
        elif btype == "thinking":
            block["thinking"] = ""
            block["signature"] = block.get("signature", "")
        elif btype == "tool_use":
            block["input"] = {}  # accumulated separately in tool_input_buffers
            tool_input_buffers[idx] = ""
        content_blocks[idx] = block
        return
    if event == "content_block_delta":
        idx = data.get("index", 0)
        delta = data.get("delta") or {}
        dtype = delta.get("type")
        block = content_blocks.get(idx)
        if block is None:
            # Server bug or out-of-order event. Skip rather than crash.
            return
        if dtype == "text_delta":
            block["text"] = (block.get("text") or "") + (delta.get("text") or "")
        elif dtype == "thinking_delta":
            block["thinking"] = (block.get("thinking") or "") + (delta.get("thinking") or "")
        elif dtype == "signature_delta":
            block["signature"] = delta.get("signature") or block.get("signature", "")
        elif dtype == "input_json_delta":
            tool_input_buffers[idx] = tool_input_buffers.get(idx, "") + (delta.get("partial_json") or "")
        # citations_delta and others: ignore (we don't surface them).
        return
    if event == "content_block_stop":
        # We finalize tool_use input at the end (after the full stream),
        # not here, because partial_json deltas may still arrive in some
        # edge cases. This matches claude code's pattern of mutating the
        # block in place and reading the final value once the stream ends.
        return
    if event == "message_delta":
        # Caller handles stop_reason and usage merge.
        return
    if event == "message_start":
        # Caller handles initial usage.
        return
    if event == "message_stop":
        return
    # Unknown event: ignore.


# ============================================================================
# StreamEvent yield-mode pipeline — step 1/2 of the streaming tool dispatch
# refactor (see streaming_tool_dispatch design notes in repo history).
# ============================================================================
#
# The existing `parse_sse_stream()` above is COLLECT-mode: it consumes the
# whole stream and returns aggregated dict shaped like the non-streaming
# response. That's what the current `chat()` uses today.
#
# Below, `stream_sse_events()` is YIELD-mode: it's an async generator that
# emits StreamEvent objects as they arrive. This is what enables the
# structural fix — runtime.py can dispatch a tool the moment its
# ContentBlockStopEvent fires, BEFORE the model is done speaking.
#
# Both live side-by-side until step 4 of the refactor makes `chat()` a
# wrapper around the yielding path.

from dataclasses import dataclass


@dataclass(frozen=True)
class MessageStartEvent:
    """First event of a stream. `usage.output_tokens` is 0 here; real
    counts arrive in MessageDeltaEvent at the end."""
    model: str
    usage: dict


@dataclass(frozen=True)
class ContentBlockStartEvent:
    """A new content block began streaming. `block_type` is text /
    thinking / tool_use / compaction. `partial_block` is the initial
    shape (e.g. tool name + id for tool_use)."""
    index: int
    block_type: str
    partial_block: dict


@dataclass(frozen=True)
class ContentBlockDeltaEvent:
    """Incremental update. Exposed for callers that want token-by-token
    UX; most consumers can ignore and wait for ContentBlockStopEvent."""
    index: int
    delta: dict


@dataclass(frozen=True)
class ContentBlockStopEvent:
    """A content block finished. **THIS IS THE TOOL DISPATCH TRIGGER**
    when completed_block['type'] == 'tool_use'. Consumer should fire the
    tool right here, before the next event arrives."""
    index: int
    completed_block: dict


@dataclass(frozen=True)
class MessageDeltaEvent:
    """Final stop_reason + usage update. Arrives AFTER all
    ContentBlockStopEvents. Consumer updates last_usage here.

    stop_details carries the structured reason when stop_reason is
    "refusal" (category + explanation) — the API populates it on a
    policy/ToS decline, which otherwise produces zero content blocks
    and looks like a silent blank reply. Default None for normal stops."""
    stop_reason: str | None
    usage: dict
    stop_details: dict | None = None


@dataclass(frozen=True)
class MessageStopEvent:
    """End of successful stream. No payload."""


@dataclass(frozen=True)
class FrameworkErrorEvent:
    """Terminal error mid-stream. `kind` is one of: auth, rate_limit,
    bad_request, timeout, transport, json_decode, sse_protocol.
    Consumer should stop iterating; no MessageStopEvent follows."""
    message: str
    kind: str


async def stream_sse_events(resp_content) -> "AsyncIterator":  # type: ignore
    """Async generator yielding StreamEvent objects as SSE events arrive.

    Args:
        resp_content: aiohttp StreamReader-like object that yields bytes
            via `async for chunk in resp_content`. Same shape as what
            `parse_sse_stream` (collect-mode) accepts.

    Yields:
        StreamEvent subclasses. Always terminates with one of:
        - MessageStopEvent (clean end)
        - FrameworkErrorEvent (error)

    Cancel-safety: if the consumer stops iterating (e.g. via outer
    asyncio.CancelledError), the generator's `async for` over
    resp_content will propagate the cancellation. No accumulated state
    needs cleanup — all state lives in the generator's local frame and
    dies with it.

    Mirrors the structure of `parse_sse_stream` above but YIELDS at each
    SSE event boundary instead of accumulating to a final dict. Tool input
    finalization moves from end-of-stream to ContentBlockStopEvent
    emission (the key behavioral change).
    """
    # Per-index accumulators. Lifetime is the duration of this stream.
    content_blocks: dict[int, dict] = {}
    tool_input_buffers: dict[int, str] = {}

    # SSE framing state (same as collect-mode).
    current_event: str = ""
    current_data: list[str] = []
    line_buffer: str = ""

    # Diagnostic counters. Used to detect "Anthropic returned 200 + opened
    # the stream + closed it without sending anything meaningful" — the
    # exact failure mode that produces silent empty replies. Without these,
    # the stream just exits cleanly on connection close and the caller
    # has no idea whether Anthropic sent a complete response or hung up.
    saw_message_start = False
    saw_message_stop = False
    content_events_count = 0

    try:
        async for raw_chunk in resp_content:
            text = raw_chunk.decode("utf-8", errors="replace")
            line_buffer += text
            if "\n" not in line_buffer:
                continue
            parts = line_buffer.split("\n")
            line_buffer = parts[-1]
            complete_lines = parts[:-1]
            for raw_line in complete_lines:
                line = raw_line.rstrip("\r")
                if line == "":
                    # End of one SSE event — dispatch it.
                    if current_event and current_data:
                        data_str = "\n".join(current_data)
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            yield FrameworkErrorEvent(
                                message=f"SSE JSON decode error on event {current_event!r}",
                                kind="json_decode",
                            )
                            current_event = ""
                            current_data = []
                            continue
                        # Track diagnostic state BEFORE yielding so we can
                        # detect abnormal stream termination after the loop.
                        if current_event == "message_start":
                            saw_message_start = True
                        elif current_event == "message_stop":
                            saw_message_stop = True
                        elif current_event in ("content_block_start",
                                                "content_block_delta",
                                                "content_block_stop"):
                            content_events_count += 1
                        # Yield the appropriate StreamEvent for this SSE event.
                        async for ev in _yield_for_sse_event(
                            current_event, data,
                            content_blocks=content_blocks,
                            tool_input_buffers=tool_input_buffers,
                        ):
                            yield ev
                    current_event = ""
                    current_data = []
                elif line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    current_data.append(line[5:].lstrip())
                # Ignore comments / other fields.
    except Exception as e:
        yield FrameworkErrorEvent(
            message=f"Stream transport error: {e}",
            kind="transport",
        )
        return

    # Stream closed cleanly (no transport exception). Detect abnormal
    # termination: if we got message_start but no message_stop AND no
    # content events between them, Anthropic accepted the request, opened
    # the stream, and closed it without sending content. This is the
    # silent-empty-response failure mode that produced the 21:59:37
    # incident — multiple agents hit it simultaneously, suggesting
    # an upstream Anthropic degradation. Surface it instead of swallowing.
    if saw_message_start and not saw_message_stop and content_events_count == 0:
        yield FrameworkErrorEvent(
            message=(
                "⚠️ Anthropic stream closed after message_start with no "
                "content and no message_stop. Likely upstream degradation; "
                "the API returned 200 but produced nothing. Retry in a "
                "moment or `/reset` if it persists."
            ),
            kind="empty_stream",
        )


async def _yield_for_sse_event(
    event: str,
    data: dict,
    *,
    content_blocks: dict[int, dict],
    tool_input_buffers: dict[int, str],
):
    """Convert one SSE event into zero-or-more StreamEvents.

    This is the core mapping. `content_block_stop` is the interesting one:
    we finalize the tool_use input JSON here (was deferred to end-of-stream
    in collect-mode) so the yielded ContentBlockStopEvent carries the
    completed block ready for immediate dispatch.
    """
    if event == "message_start":
        msg = data.get("message") or {}
        usage = dict(msg.get("usage") or {})
        yield MessageStartEvent(model=msg.get("model", ""), usage=usage)
        return

    if event == "content_block_start":
        idx = data.get("index", 0)
        block = dict(data.get("content_block") or {})
        btype = block.get("type", "")
        # Reset per-type starting state (same logic as collect-mode parser).
        if btype == "text":
            block["text"] = ""
        elif btype == "thinking":
            block["thinking"] = ""
            block["signature"] = block.get("signature", "")
        elif btype == "tool_use":
            block["input"] = {}
            tool_input_buffers[idx] = ""
        content_blocks[idx] = block
        yield ContentBlockStartEvent(
            index=idx, block_type=btype, partial_block=dict(block),
        )
        return

    if event == "content_block_delta":
        idx = data.get("index", 0)
        delta = data.get("delta") or {}
        dtype = delta.get("type", "")
        block = content_blocks.get(idx)
        if block is None:
            # Out-of-order or server bug. Skip silently — matches
            # collect-mode behavior.
            return
        if dtype == "text_delta":
            block["text"] = (block.get("text") or "") + (delta.get("text") or "")
        elif dtype == "thinking_delta":
            block["thinking"] = (block.get("thinking") or "") + (delta.get("thinking") or "")
        elif dtype == "signature_delta":
            block["signature"] = delta.get("signature") or block.get("signature", "")
        elif dtype == "input_json_delta":
            tool_input_buffers[idx] = (
                tool_input_buffers.get(idx, "") + (delta.get("partial_json") or "")
            )
        # citations_delta etc: ignore.
        yield ContentBlockDeltaEvent(index=idx, delta=dict(delta))
        return

    if event == "content_block_stop":
        idx = data.get("index", 0)
        block = content_blocks.get(idx)
        if block is None:
            return
        # KEY DIFFERENCE FROM COLLECT-MODE: finalize tool_use input JSON
        # right here, so the ContentBlockStopEvent carries the fully-
        # assembled block ready for the consumer to dispatch.
        if block.get("type") == "tool_use":
            raw_input = tool_input_buffers.get(idx, "")
            if raw_input:
                try:
                    block["input"] = json.loads(raw_input)
                except json.JSONDecodeError:
                    # Malformed tool input — leave empty, downstream
                    # malformed-tool-use detector handles it.
                    block["input"] = {}
            else:
                block["input"] = {}
        yield ContentBlockStopEvent(index=idx, completed_block=dict(block))
        return

    if event == "message_delta":
        delta = data.get("delta") or {}
        stop_reason = delta.get("stop_reason")
        stop_details = delta.get("stop_details")  # populated on refusal: {type, category, explanation}
        usage = dict(data.get("usage") or {})
        yield MessageDeltaEvent(
            stop_reason=stop_reason, usage=usage, stop_details=stop_details
        )
        return

    if event == "message_stop":
        yield MessageStopEvent()
        return

    # Unknown event: ignore. Anthropic may add new event types over time
    # and we don't want to break on forward compatibility.
