"""Direct OpenAI API conversation wrapper.

Native third provider: an agent with `"provider": "openai"` runs its own
turns against OpenAI. NOT a delegation tool — this is the agent's brain,
exactly like AnthropicConversation is for Claude agents.

TWO auth/API paths, chosen per turn by credential precedence:
  1. ChatGPT/Codex subscription OAuth (PREFERRED — exists when `codex login`
     has written `$CODEX_HOME/auth.json` with auth_mode=="chatgpt"): the
     **Responses API** at `https://chatgpt.com/backend-api/codex/responses`
     — `input` items, `function_call`/`function_call_output` items,
     `response.output_text.delta` / `response.output_item.done` /
     `response.completed` SSE events. Token borrow/refresh lives in
     `_codex_auth.py` (protocol mirrored from simonw/llm-openai-via-codex).
  2. Plain API key (FALLBACK — config_global.get_openai_api_key): the
     standard Chat Completions API (`/v1/chat/completions`, messages array,
     tool_calls deltas). The original path, kept verbatim.
Both translate into the same StreamEvent taxonomy, so everything downstream
(runtime, /status, ledger) is identical regardless of path.

Mirrors the PUBLIC interface of `AnthropicConversation` so `runtime.py`
treats the two polymorphically:
  - same constructor (conversation_id, agent)
  - chat() returns the same assistant-message shape (content / content_text /
    tool_calls / is_framework_error / raw_response)
  - chat_stream() yields the same StreamEvent dataclasses from
    `_anthropic_stream` (they're provider-neutral; OpenAI's SSE chunks are
    translated into that taxonomy here)
  - same JSONL persistence via `_conversation_io` (load/save/clear_history)
  - same `last_usage` dict shape (+ meta sidecar) so /status works unchanged

Deliberate differences from the anthropic sibling (document, don't "fix"):
  - NO server-side compaction (OpenAI has no equivalent of Anthropic's
    compact beta). Context is bounded by a pre-flight local trim EVERY turn,
    like the ollama provider — see _trim_to_fit_window.
  - NO /effort session override and no force_compact_next attr — the
    owner commands gate on hasattr() and correctly report "Anthropic-only".
  - Prompt caching is automatic on OpenAI (prefix-based, no cache_control
    markers). Cache reads surface as prompt_tokens_details.cached_tokens and
    are mapped into last_usage["cache_read_input_tokens"].
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator, Callable

import aiohttp

from .agent import Agent
from .utils import print_ts, COLOR_YELLOW, COLOR_RED, COLOR_END, load_json, save_json
from . import _conversation_io as _cio
from ._codex_auth import CodexAuthError, borrow_codex_key, codex_creds_exist
from .config_global import (
    get_config,
    get_effort,
    get_model_context_window,
    get_openai_api_key,
    get_openai_base_url,
    get_openai_default_model,
)
# These three are provider-neutral despite living in anthropic_conversation
# (dict-backed message, tool-call carrier, assistant-reply wrapper). Reused
# verbatim so runtime.py's attribute contract (function_name / args /
# tool_use_id / is_framework_error / content_text) is identical across
# providers — forking them would just invite drift.
from .anthropic_conversation import (
    ChatMessage,
    AnthropicToolCall as ToolCall,
    AnthropicAIChatMessage as AIChatMessage,
)
from ._anthropic_stream import (
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    FrameworkErrorEvent,
)


# OpenAI finish_reason → the stop_reason vocabulary the framework already
# understands (Anthropic's). chat()'s empty-reply diagnostics key on these.
_FINISH_TO_STOP_REASON = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}

# Subscription-mode (Codex OAuth) request base — the ChatGPT backend's
# Responses API, NOT api.openai.com. Verified protocol constant from the
# llm-openai-via-codex reference; get_openai_base_url() does not apply here.
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Reasoning-effort vocabulary on the codex backend differs from Chat
# Completions: low/medium/high/xhigh (no "minimal"). Validated here rather
# than through get_effort() so a config of "xhigh" works in subscription
# mode without loosening the Chat-Completions validation (where xhigh 400s).
_CODEX_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh")


def _codex_effort(model_name: str) -> str | None:
    """Per-model `models.<bare>.effort` for subscription mode → the
    `reasoning.effort` request field, or None to omit. "minimal" (valid for
    Chat Completions, absent from the codex vocabulary) maps to "low" so one
    config value stays usable across both paths."""
    bare = model_name.split("/", 1)[1] if "/" in model_name else model_name
    entry = ((get_config().get("models") or {}).get(bare)) or {}
    level = entry.get("effort")
    if isinstance(level, str):
        lv = level.strip().lower()
        if lv == "minimal":
            return "low"
        if lv in _CODEX_EFFORT_LEVELS:
            return lv
    return None


def _build_openai_tool_schemas(tools: list[Callable]) -> list[dict]:
    """Convert openflip tool callables to OpenAI function-calling format.

    Same source data as the anthropic builder (`tool_spec` set by the @tool
    decorator: name / description / input_schema), different envelope:
    {"type": "function", "function": {name, description, parameters}}.
    """
    schemas = []
    for func in tools or []:
        spec = getattr(func, "tool_spec", None)
        if spec and isinstance(spec, dict):
            schemas.append({
                "type": "function",
                "function": {
                    "name": spec.get("name") or func.__name__,
                    "description": spec.get("description") or (func.__doc__ or "").strip()[:1000],
                    "parameters": spec.get("input_schema") or spec.get("parameters") or {
                        "type": "object", "properties": {},
                    },
                },
            })
        else:
            schemas.append({
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": (func.__doc__ or "").strip()[:1000],
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                },
            })
    return schemas


def _translate_tool_choice(tool_choice: dict | None) -> Any:
    """Map the framework's Anthropic-shaped tool_choice to OpenAI's.

    runtime.py forces {"type": "any"} on action-promise retries and may pass
    {"type": "tool", "name": X} for chain-terminator turns. Translating here
    keeps the runtime provider-agnostic.
    """
    if not isinstance(tool_choice, dict):
        return None
    kind = tool_choice.get("type")
    if kind == "any":
        return "required"
    if kind == "auto":
        return "auto"
    if kind == "none":
        return "none"
    if kind == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return None


def _build_responses_tool_schemas(tools: list[Callable]) -> list[dict]:
    """Convert openflip tool callables to the Responses API's FLAT function
    format: {"type": "function", "name", "description", "parameters",
    "strict": False} — no nested "function" envelope (that's the
    Chat-Completions shape). Same source data as the other builders."""
    schemas = []
    for func in tools or []:
        spec = getattr(func, "tool_spec", None)
        if spec and isinstance(spec, dict):
            schemas.append({
                "type": "function",
                "name": spec.get("name") or func.__name__,
                "description": spec.get("description") or (func.__doc__ or "").strip()[:1000],
                "parameters": spec.get("input_schema") or spec.get("parameters") or {
                    "type": "object", "properties": {},
                },
                "strict": False,
            })
        else:
            schemas.append({
                "type": "function",
                "name": func.__name__,
                "description": (func.__doc__ or "").strip()[:1000],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
                "strict": False,
            })
    return schemas


def _translate_tool_choice_responses(tool_choice: dict | None) -> Any:
    """Responses-API sibling of _translate_tool_choice: same string modes,
    but a forced specific tool is flat {"type": "function", "name": X}."""
    if not isinstance(tool_choice, dict):
        return None
    kind = tool_choice.get("type")
    if kind == "any":
        return "required"
    if kind == "auto":
        return "auto"
    if kind == "none":
        return "none"
    if kind == "tool" and tool_choice.get("name"):
        return {"type": "function", "name": tool_choice["name"]}
    return None


def _openflip_msgs_to_openai(messages: list, system_prompt: str) -> list[dict]:
    """Convert the local ChatMessage list to OpenAI Chat Completions format.

    The system prompt becomes the leading {"role": "system"} message; any
    mid-history system messages are folded into it (mirrors the anthropic
    converter's sys_parts behavior).

    Tool-use round-tripping: an assistant message carrying `tool_calls`
    whose immediately-following tool messages all have matching
    `tool_use_id`s becomes an OpenAI assistant message with a `tool_calls`
    array, followed by one {"role": "tool", "tool_call_id": ...} message per
    result — the shape OpenAI requires for a valid round-trip.

    If pairing is incomplete (restart dropped the in-memory tool_calls, or a
    tool message lacks its id), the affected messages degrade to plain text:
    tool results become `[Previous tool result: ...]` user messages and the
    assistant message stays text-only. Lossy but never 400s — identical
    degradation policy to the anthropic converter.
    """
    api_msgs: list[dict] = []
    sys_parts: list[str] = []
    if system_prompt:
        sys_parts.append(system_prompt)

    def _msg_role(m):
        return m.get("role") if hasattr(m, "get") else getattr(m, "role", None)

    def _msg_content(m):
        return (m.get("content_text", None) if hasattr(m, "get") else None) \
            or (m.get("content", "") if hasattr(m, "get") else getattr(m, "content", "")) \
            or ""

    def _msg_tool_calls(m):
        if hasattr(m, "get"):
            tcs = m.get("tool_calls", None)
        else:
            tcs = getattr(m, "tool_calls", None)
        return tcs or []

    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        role = _msg_role(m)
        content = _msg_content(m)

        if role == "system":
            if content:
                sys_parts.append(content)
            i += 1
            continue

        if role == "tool":
            # Stray tool message (not consumed by the paired-assistant branch
            # below). Demote to user text so it stays in context without
            # violating OpenAI's tool_call_id pairing rules.
            api_msgs.append({
                "role": "user",
                "content": f"[Previous tool result: {content}]",
            })
            i += 1
            continue

        if role == "assistant":
            tool_calls = _msg_tool_calls(m)
            if tool_calls:
                lookahead_results: list[tuple[int, str, str]] = []  # (idx, id, content)
                j = i + 1
                while j < n and _msg_role(messages[j]) == "tool":
                    tm = messages[j]
                    tid = tm.get("tool_use_id", "") if hasattr(tm, "get") else ""
                    lookahead_results.append((j, tid, _msg_content(tm)))
                    j += 1

                call_ids = [getattr(tc, "tool_use_id", "") or "" for tc in tool_calls]
                pair_ok = (
                    len(lookahead_results) == len(tool_calls)
                    and all(rid for _, rid, _ in lookahead_results)
                    and all(cid for cid in call_ids)
                    and {rid for _, rid, _ in lookahead_results} == set(call_ids)
                )

                if pair_ok:
                    api_msgs.append({
                        "role": "assistant",
                        # OpenAI allows null content alongside tool_calls.
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": tc.tool_use_id,
                                "type": "function",
                                "function": {
                                    "name": tc.function_name,
                                    "arguments": json.dumps(tc.args or {}, ensure_ascii=False),
                                },
                            }
                            for tc in tool_calls
                        ],
                    })
                    for _, rid, rcontent in lookahead_results:
                        api_msgs.append({
                            "role": "tool",
                            "tool_call_id": rid,
                            "content": rcontent,
                        })
                    i = j  # skip past the paired tool messages
                    continue

                # Pairing failed — emit assistant as text-only; following
                # tool messages hit the `role == 'tool'` demotion branch.
                api_msgs.append({"role": "assistant", "content": content})
                i += 1
                continue

            api_msgs.append({"role": "assistant", "content": content})
            i += 1
            continue

        if role == "user":
            api_msgs.append({"role": "user", "content": content})
            i += 1
            continue

        # Unknown role — skip.
        i += 1

    system_text = "\n\n".join(p for p in sys_parts if p)
    if system_text:
        api_msgs.insert(0, {"role": "system", "content": system_text})
    return api_msgs


def _openflip_msgs_to_responses_input(
    messages: list, system_prompt: str,
) -> tuple[str, list[dict]]:
    """Convert the local ChatMessage list to Responses-API form for the codex
    subscription path. Returns (instructions, input_items).

    The system prompt does NOT become a message — the Responses API takes it
    as the top-level `instructions` field; mid-history system messages are
    folded into it (same sys_parts behavior as the Chat-Completions
    converter).

    Tool-use round-tripping (the Responses item vocabulary): an assistant
    message with paired tool calls becomes an optional assistant message item
    (its text), then one {"type": "function_call", "call_id", "name",
    "arguments"} item per call, then one {"type": "function_call_output",
    "call_id", "output"} item per result.

    If pairing is incomplete, degradation is IDENTICAL in policy to the
    Chat-Completions converter: tool results demote to `[Previous tool
    result: ...]` user items and the assistant message stays text-only.
    Lossy but never 400s.
    """
    items: list[dict] = []
    sys_parts: list[str] = []
    if system_prompt:
        sys_parts.append(system_prompt)

    def _msg_role(m):
        return m.get("role") if hasattr(m, "get") else getattr(m, "role", None)

    def _msg_content(m):
        return (m.get("content_text", None) if hasattr(m, "get") else None) \
            or (m.get("content", "") if hasattr(m, "get") else getattr(m, "content", "")) \
            or ""

    def _msg_tool_calls(m):
        if hasattr(m, "get"):
            tcs = m.get("tool_calls", None)
        else:
            tcs = getattr(m, "tool_calls", None)
        return tcs or []

    def _user_item(text: str) -> dict:
        return {"role": "user", "content": [{"type": "input_text", "text": text}]}

    def _assistant_item(text: str) -> dict:
        return {"role": "assistant", "content": [{"type": "output_text", "text": text}]}

    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        role = _msg_role(m)
        content = _msg_content(m)

        if role == "system":
            if content:
                sys_parts.append(content)
            i += 1
            continue

        if role == "tool":
            # Stray tool message (not consumed by the paired-assistant branch
            # below) — demote to user text, same policy as the CC converter.
            items.append(_user_item(f"[Previous tool result: {content}]"))
            i += 1
            continue

        if role == "assistant":
            tool_calls = _msg_tool_calls(m)
            if tool_calls:
                lookahead_results: list[tuple[int, str, str]] = []  # (idx, id, content)
                j = i + 1
                while j < n and _msg_role(messages[j]) == "tool":
                    tm = messages[j]
                    tid = tm.get("tool_use_id", "") if hasattr(tm, "get") else ""
                    lookahead_results.append((j, tid, _msg_content(tm)))
                    j += 1

                call_ids = [getattr(tc, "tool_use_id", "") or "" for tc in tool_calls]
                pair_ok = (
                    len(lookahead_results) == len(tool_calls)
                    and all(rid for _, rid, _ in lookahead_results)
                    and all(cid for cid in call_ids)
                    and {rid for _, rid, _ in lookahead_results} == set(call_ids)
                )

                if pair_ok:
                    if content:
                        items.append(_assistant_item(content))
                    for tc in tool_calls:
                        items.append({
                            "type": "function_call",
                            "call_id": tc.tool_use_id,
                            "name": tc.function_name,
                            "arguments": json.dumps(tc.args or {}, ensure_ascii=False),
                        })
                    for _, rid, rcontent in lookahead_results:
                        items.append({
                            "type": "function_call_output",
                            "call_id": rid,
                            "output": rcontent,
                        })
                    i = j  # skip past the paired tool messages
                    continue

                # Pairing failed — text-only assistant; following tool
                # messages hit the demotion branch above.
                if content:
                    items.append(_assistant_item(content))
                i += 1
                continue

            # Empty assistant text carries nothing — skip the item rather
            # than send a zero-length output_text part.
            if content:
                items.append(_assistant_item(content))
            i += 1
            continue

        if role == "user":
            items.append(_user_item(content))
            i += 1
            continue

        # Unknown role — skip.
        i += 1

    return "\n\n".join(p for p in sys_parts if p), items


def _inject_pending_image_attachments_responses(
    input_items: list[dict], pending: list[dict],
) -> int:
    """Responses-API sibling of _inject_pending_image_attachments: queued
    images become {"type": "input_image", "image_url": "data:..."} parts
    prepended to the LAST user message item's content list. Returns the
    count injected. Validation shared via `_image_validator`."""
    import base64
    from ._image_validator import validate_and_normalize_image
    if not pending or not input_items:
        return 0
    last_user = None
    for item in reversed(input_items):
        if item.get("role") == "user":
            last_user = item
            break
    if last_user is None:
        return 0
    content = last_user.get("content")
    if not isinstance(content, list):
        return 0
    image_parts: list[dict] = []
    for entry in pending:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not path:
            continue
        declared = (entry.get("content_type") or "image/png").lower()
        normalized, mt_or_reason = validate_and_normalize_image(path, declared)
        if normalized is None:
            print_ts(
                f"{COLOR_YELLOW}image attachment ({path}) rejected at "
                f"inject-side safety net: {mt_or_reason}{COLOR_END}",
            )
            continue
        try:
            b64 = base64.b64encode(normalized).decode("ascii")
        except Exception:
            continue
        image_parts.append({
            "type": "input_image",
            "image_url": f"data:{mt_or_reason};base64,{b64}",
        })
    if not image_parts:
        return 0
    # Image-then-text ordering, matching the other injectors.
    last_user["content"] = image_parts + content
    return len(image_parts)


def _inject_pending_image_attachments(api_messages: list[dict], pending: list[dict]) -> int:
    """Inject queued image attachments into the LAST user message as OpenAI
    image_url content parts (base64 data URLs). Returns the count injected.

    Validation/normalization is shared with the anthropic path via
    `_image_validator`. OpenAI vision shape:
        {"type": "image_url", "image_url": {"url": "data:<mt>;base64,<b64>"}}

    No tool_result-adjacency hazard here: tool results map to role:"tool"
    messages (never user messages), so the last user message is always safe
    to extend. Mutates api_messages in place; caller clears the queue.
    """
    import base64
    from ._image_validator import validate_and_normalize_image
    if not pending or not api_messages:
        return 0
    last_user = None
    for msg in reversed(api_messages):
        if msg.get("role") == "user":
            last_user = msg
            break
    if last_user is None:
        return 0
    content = last_user.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        return 0
    image_parts: list[dict] = []
    for entry in pending:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not path:
            continue
        declared = (entry.get("content_type") or "image/png").lower()
        normalized, mt_or_reason = validate_and_normalize_image(path, declared)
        if normalized is None:
            print_ts(
                f"{COLOR_YELLOW}image attachment ({path}) rejected at "
                f"inject-side safety net: {mt_or_reason}{COLOR_END}",
            )
            continue
        try:
            b64 = base64.b64encode(normalized).decode("ascii")
        except Exception:
            continue
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mt_or_reason};base64,{b64}"},
        })
    if not image_parts:
        return 0
    # Image-then-text ordering, matching the anthropic injector's prepend.
    last_user["content"] = image_parts + content
    return len(image_parts)


async def _stream_openai_events(resp_content) -> AsyncIterator:
    """Async generator translating OpenAI Chat Completions SSE chunks into
    the framework's StreamEvent taxonomy (`_anthropic_stream` dataclasses).

    OpenAI frames every chunk as `data: <json>` lines terminated by
    `data: [DONE]` — there are no named `event:` lines. Text deltas arrive
    in choices[0].delta.content; tool calls accumulate fragment-by-fragment
    in choices[0].delta.tool_calls (keyed by their own `index`, arguments as
    a growing JSON string); the final usage chunk (stream_options.
    include_usage) has an empty choices array.

    Block-index convention for the yielded events: text is block 0; tool
    call k is block k+1. ContentBlockStopEvents fire when finish_reason
    arrives (OpenAI signals nothing per-block), carrying completed blocks in
    the exact shapes the chat() accumulator expects ({"type":"text","text"}
    / {"type":"tool_use","id","name","input"}).
    """
    text_acc = ""
    refusal_acc = ""
    saw_text_start = False
    # tool index (OpenAI's) -> {"id": str, "name": str, "arguments": str}
    tool_acc: dict[int, dict] = {}
    started_tool_indexes: set[int] = set()
    finish_reason: str | None = None
    usage: dict = {}
    sent_message_start = False

    line_buffer = ""
    try:
        async for raw_chunk in resp_content:
            text = raw_chunk.decode("utf-8", errors="replace")
            line_buffer += text
            if "\n" not in line_buffer:
                continue
            parts = line_buffer.split("\n")
            line_buffer = parts[-1]
            for raw_line in parts[:-1]:
                line = raw_line.rstrip("\r")
                if not line.startswith("data:"):
                    # Blank separators / comments / other SSE fields: ignore.
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue  # terminal sentinel; finalization happens below
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    yield FrameworkErrorEvent(
                        message="SSE JSON decode error on OpenAI chunk",
                        kind="json_decode",
                    )
                    continue

                if not sent_message_start:
                    sent_message_start = True
                    yield MessageStartEvent(model=chunk.get("model", ""), usage={})

                if isinstance(chunk.get("usage"), dict):
                    usage = dict(chunk["usage"])

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                delta = choice.get("delta") or {}

                piece = delta.get("content")
                if piece:
                    if not saw_text_start:
                        saw_text_start = True
                        yield ContentBlockStartEvent(
                            index=0, block_type="text",
                            partial_block={"type": "text", "text": ""},
                        )
                    text_acc += piece
                    yield ContentBlockDeltaEvent(
                        index=0, delta={"type": "text_delta", "text": piece},
                    )

                # Refusal text (structured-refusal field). Folded into the
                # text block at finalization so the operator sees WHY.
                rpiece = delta.get("refusal")
                if rpiece:
                    refusal_acc += rpiece

                for tc_delta in (delta.get("tool_calls") or []):
                    if not isinstance(tc_delta, dict):
                        continue
                    t_idx = tc_delta.get("index", 0)
                    slot = tool_acc.setdefault(
                        t_idx, {"id": "", "name": "", "arguments": ""})
                    if tc_delta.get("id"):
                        slot["id"] = tc_delta["id"]
                    fn = tc_delta.get("function") or {}
                    if fn.get("name"):
                        # The name arrives whole in the first fragment for a
                        # given index; later fragments carry arguments only.
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                    if t_idx not in started_tool_indexes:
                        started_tool_indexes.add(t_idx)
                        yield ContentBlockStartEvent(
                            index=t_idx + 1, block_type="tool_use",
                            partial_block={
                                "type": "tool_use",
                                "id": slot["id"],
                                "name": slot["name"],
                                "input": {},
                            },
                        )

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    except Exception as e:
        yield FrameworkErrorEvent(
            message=f"Stream transport error: {e}",
            kind="transport",
        )
        return

    # ── Finalization: emit completed blocks + terminal events ──
    final_text = text_acc
    if refusal_acc:
        final_text = (final_text + ("\n" if final_text else "")
                      + f"[Model refusal]: {refusal_acc}")
    if final_text:
        yield ContentBlockStopEvent(
            index=0, completed_block={"type": "text", "text": final_text},
        )
    for t_idx in sorted(tool_acc.keys()):
        slot = tool_acc[t_idx]
        raw_args = slot.get("arguments") or ""
        try:
            parsed = json.loads(raw_args) if raw_args else {}
            if not isinstance(parsed, dict):
                parsed = {}
        except json.JSONDecodeError:
            # Malformed arguments — leave empty; the downstream
            # malformed-tool_use detector in chat() handles it.
            parsed = {}
        yield ContentBlockStopEvent(
            index=t_idx + 1,
            completed_block={
                "type": "tool_use",
                "id": slot.get("id") or f"call_{t_idx}",
                "name": slot.get("name") or "",
                "input": parsed,
            },
        )

    stop_reason = _FINISH_TO_STOP_REASON.get(finish_reason or "", finish_reason)
    yield MessageDeltaEvent(stop_reason=stop_reason, usage=dict(usage))

    if not sent_message_start:
        # Connection closed before ANY chunk arrived — the OpenAI-side
        # equivalent of Anthropic's silent-empty-stream degradation.
        yield FrameworkErrorEvent(
            message=(
                "⚠️ OpenAI stream closed without sending any data. Likely "
                "upstream degradation; retry in a moment."
            ),
            kind="empty_stream",
        )
        return
    yield MessageStopEvent()


async def _stream_codex_events(resp_content) -> AsyncIterator:
    """Async generator translating Responses-API SSE events (codex
    subscription backend) into the framework's StreamEvent taxonomy.

    Responses SSE frames carry the event name BOTH as an `event:` line and
    as the data object's `type` field — we key off the data `type` (more
    robust than tracking `event:` lines across chunk boundaries). Events
    handled (the shapes from the llm-openai-via-codex reference):
      - response.output_text.delta   → text delta (`delta` field, block 0)
      - response.output_item.added   → tool-block start when item.type ==
                                       "function_call"
      - response.output_item.done    → completed tool block (item carries
                                       call_id / name / arguments-JSON-string)
      - response.completed           → usage (response.usage.input_tokens /
                                       output_tokens / input_tokens_details.
                                       cached_tokens) + stop_reason
      - response.failed / error      → FrameworkErrorEvent
    Everything else (created, in_progress, content_part.*, delta echoes of
    items we finalize on .done) is ignored.

    Block-index convention matches _stream_openai_events: text is block 0;
    tool call k is block k+1. Usage is normalized to the Chat-Completions
    key shape (prompt_tokens / completion_tokens / prompt_tokens_details.
    cached_tokens) so the caller's last_usage/ledger code is shared.
    """
    text_acc = ""
    saw_text_start = False
    sent_message_start = False
    tool_block_by_key: dict[str, int] = {}  # item id/call_id → block index
    next_tool_block = 1
    any_tool = False
    usage: dict = {}
    stop_reason: str | None = None
    failed_msg: str | None = None

    line_buffer = ""
    try:
        async for raw_chunk in resp_content:
            text = raw_chunk.decode("utf-8", errors="replace")
            line_buffer += text
            if "\n" not in line_buffer:
                continue
            parts = line_buffer.split("\n")
            line_buffer = parts[-1]
            for raw_line in parts[:-1]:
                line = raw_line.rstrip("\r")
                if not line.startswith("data:"):
                    # `event:` lines / blank separators / comments: ignore.
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    yield FrameworkErrorEvent(
                        message="SSE JSON decode error on codex chunk",
                        kind="json_decode",
                    )
                    continue

                etype = chunk.get("type", "")

                if not sent_message_start:
                    sent_message_start = True
                    model = ((chunk.get("response") or {}).get("model")) or ""
                    yield MessageStartEvent(model=model, usage={})

                if etype == "response.output_text.delta":
                    piece = chunk.get("delta") or ""
                    if piece:
                        if not saw_text_start:
                            saw_text_start = True
                            yield ContentBlockStartEvent(
                                index=0, block_type="text",
                                partial_block={"type": "text", "text": ""},
                            )
                        text_acc += piece
                        yield ContentBlockDeltaEvent(
                            index=0, delta={"type": "text_delta", "text": piece},
                        )
                    continue

                if etype == "response.output_item.added":
                    item = chunk.get("item") or {}
                    if item.get("type") == "function_call":
                        key = item.get("id") or item.get("call_id") \
                            or f"out_{chunk.get('output_index', next_tool_block)}"
                        idx = next_tool_block
                        next_tool_block += 1
                        tool_block_by_key[key] = idx
                        yield ContentBlockStartEvent(
                            index=idx, block_type="tool_use",
                            partial_block={
                                "type": "tool_use",
                                "id": item.get("call_id") or "",
                                "name": item.get("name") or "",
                                "input": {},
                            },
                        )
                    continue

                if etype == "response.output_item.done":
                    item = chunk.get("item") or {}
                    if item.get("type") != "function_call":
                        continue  # message items finalize via text_acc below
                    any_tool = True
                    key = item.get("id") or item.get("call_id") \
                        or f"out_{chunk.get('output_index', '')}"
                    idx = tool_block_by_key.get(key)
                    if idx is None:
                        # .done without a preceding .added — synthesize the
                        # start so consumers always see paired events.
                        idx = next_tool_block
                        next_tool_block += 1
                        yield ContentBlockStartEvent(
                            index=idx, block_type="tool_use",
                            partial_block={
                                "type": "tool_use",
                                "id": item.get("call_id") or "",
                                "name": item.get("name") or "",
                                "input": {},
                            },
                        )
                    raw_args = item.get("arguments") or ""
                    try:
                        parsed = json.loads(raw_args) if raw_args else {}
                        if not isinstance(parsed, dict):
                            parsed = {}
                    except json.JSONDecodeError:
                        # Malformed arguments — leave empty; chat()'s
                        # malformed-tool_use detector handles it.
                        parsed = {}
                    yield ContentBlockStopEvent(
                        index=idx,
                        completed_block={
                            "type": "tool_use",
                            "id": item.get("call_id") or item.get("id") or f"call_{idx}",
                            "name": item.get("name") or "",
                            "input": parsed,
                        },
                    )
                    continue

                if etype == "response.completed":
                    response = chunk.get("response") or {}
                    u = response.get("usage") or {}
                    in_t = int(u.get("input_tokens", 0) or 0)
                    out_t = int(u.get("output_tokens", 0) or 0)
                    cached = int(((u.get("input_tokens_details") or {})
                                  .get("cached_tokens", 0)) or 0)
                    usage = {
                        "prompt_tokens": in_t,
                        "completion_tokens": out_t,
                        "prompt_tokens_details": {"cached_tokens": cached},
                        "responses_raw": dict(u),
                    }
                    inc_reason = ((response.get("incomplete_details") or {})
                                  .get("reason") or "")
                    if response.get("status") == "incomplete" and inc_reason == "max_output_tokens":
                        stop_reason = "max_tokens"
                    continue

                if etype in ("response.failed", "error"):
                    response = chunk.get("response") or {}
                    err = response.get("error") or chunk.get("error") or {}
                    failed_msg = (err.get("message") if isinstance(err, dict) else str(err)) \
                        or "response.failed with no error detail"
                    continue
    except Exception as e:
        yield FrameworkErrorEvent(
            message=f"Stream transport error: {e}",
            kind="transport",
        )
        return

    if failed_msg:
        yield FrameworkErrorEvent(
            message=f"⚠️ OpenAI (codex) response failed: {failed_msg[:300]}",
            kind="bad_request",
        )
        return

    # ── Finalization: text block + terminal events ──
    if text_acc:
        yield ContentBlockStopEvent(
            index=0, completed_block={"type": "text", "text": text_acc},
        )
    if stop_reason is None:
        stop_reason = "tool_use" if any_tool else "end_turn"
    yield MessageDeltaEvent(stop_reason=stop_reason, usage=dict(usage))

    if not sent_message_start:
        yield FrameworkErrorEvent(
            message=(
                "⚠️ OpenAI (codex) stream closed without sending any data. "
                "Likely upstream degradation; retry in a moment."
            ),
            kind="empty_stream",
        )
        return
    yield MessageStopEvent()


class OpenAIConversation:
    """OpenAI-direct provider used when agent.provider == 'openai'.

    Auth precedence per turn: Codex subscription OAuth (Responses API at
    chatgpt.com/backend-api/codex) when `codex login` creds exist, else
    plain API key against {base}/v1/chat/completions. Both stream SSE and
    yield the same StreamEvent taxonomy.
    """

    def __init__(self, conversation_id: str, agent: Agent):
        self.conversation_id = conversation_id
        self.agent = agent
        self.model = self._normalize_model(agent.model)
        self.system_message = agent.system_message
        self.messages: list[ChatMessage] = []
        self._persisted_count = 0
        self._http_session: aiohttp.ClientSession | None = None
        # Shrunken token budget set by the context-overflow 400 retry path;
        # None = normal (window - 10k) budget. Same mechanism as the
        # anthropic sibling's prompt-too-long recovery.
        self._retry_budget: int | None = None
        # Latest API usage stats; populated after every chat(). /status reads
        # this. Same key shape as the other providers' last_usage.
        self.last_usage: dict | None = None
        # NOTE deliberately ABSENT attributes (vs AnthropicConversation):
        # force_compact_next / compacted_this_turn / effort_override.
        # /compact and /effort gate on hasattr() and correctly report
        # "Anthropic-only"; runtime reads them via getattr(..., False).

    @staticmethod
    def _normalize_model(model_str: str) -> str:
        """Strip the `openai/` provider prefix (picker form). Falls back to
        the configured default model when the agent's model field is empty."""
        m = model_str or ""
        if "/" in m:
            m = m.split("/", 1)[1]
        return m or get_openai_default_model()

    def _agent_dir(self) -> str:
        return os.path.dirname(self.agent.path)

    def _conversation_path(self) -> str:
        return _cio.conversation_path(self._agent_dir(), self.conversation_id)

    def _meta_path(self) -> str:
        """Sidecar JSON holding non-message state. For this provider that is
        only last_usage (no compaction block — OpenAI has no server-side
        compaction). Kept so /status survives restarts. Cleared by
        clear_history."""
        return os.path.join(
            self._agent_dir(), "conversations", f"{_cio.fs_encode(self.conversation_id)}.meta.json"
        )

    def _content_extractor(self, m) -> str:
        return getattr(m, "content_text", None) or m.get("content", "") or ""

    def _save_meta(self):
        payload: dict = {}
        if self.last_usage is not None:
            payload["last_usage"] = self.last_usage
        if not payload:
            try:
                if os.path.isfile(self._meta_path()):
                    save_json(self._meta_path(), payload)
            except OSError:
                pass
            return
        save_json(self._meta_path(), payload)

    def load(self):
        _cio.migrate_legacy_to_jsonl(
            self._agent_dir(), self.conversation_id,
            log_agent_id=self.agent.id,
        )
        msgs = _cio.read_all_messages(self._conversation_path())
        meta = load_json(self._meta_path(), default={}) or {}
        stored_usage = meta.get("last_usage")
        if isinstance(stored_usage, dict):
            self.last_usage = stored_usage
        if not msgs:
            return
        for entry in msgs:
            self.messages.append(ChatMessage(entry["role"], entry.get("content", "")))
        self._persisted_count = len(msgs)
        print_ts(
            f"Loaded {len(msgs)} messages from disk (openai-direct)",
            agent=self.agent.id,
        )

    def save(self):
        non_system = [m for m in self.messages if m.get("role") != "system"]
        new_count = len(non_system) - self._persisted_count
        if new_count <= 0:
            self._persisted_count = len(non_system)
            return
        _cio.append_messages(
            self._conversation_path(),
            non_system[-new_count:],
            content_extractor=self._content_extractor,
        )
        self._persisted_count = len(non_system)

    def clear_history(self):
        _cio.delete_conversation_files(
            self._agent_dir(), self.conversation_id,
            extra_paths=[self._meta_path()],
            backup_tag="pre_reset",
        )
        self._persisted_count = 0
        self.last_usage = None

    def reapply_agent(self):
        self.model = self._normalize_model(self.agent.model)
        self.system_message = self.agent.system_message

    def _max_output_tokens(self) -> int | None:
        """Optional per-model completion cap → `max_completion_tokens`.

        Omitted entirely when unset — OpenAI then allows up to the model's
        own maximum, which is the safe default across models with very
        different caps. Configure `models.<bare>.max_output_tokens` in
        config.json to bound it.
        """
        bare = self.model
        entry = ((get_config().get("models") or {}).get(bare)) or {}
        val = entry.get("max_output_tokens")
        if isinstance(val, int) and val > 0:
            return val
        return None

    def _trim_to_fit_window(self) -> int:
        """Pre-flight LOCAL TRIM: drop oldest messages until estimated input
        fits the budget. Returns count dropped.

        DELIBERATE divergences (vs the anthropic + ollama siblings — each
        justified, do NOT align):
          - Fires EVERY turn pre-flight (like ollama, unlike anthropic):
            OpenAI has no server-side compaction, so this trim IS the
            primary context-bounding mechanism.
          - //2 estimator (like anthropic, unlike ollama): OpenAI hard-400s
            on context overflow (context_length_exceeded), so over-estimate
            and trim conservatively.
          - Trims down to budget*0.8 (like anthropic): OpenAI's automatic
            prefix caching benefits from a stable head; headroom means we
            don't rotate the head every turn.

        `_retry_budget` (set by the 400 context-overflow retry path) wins
        over the normal `window - 10k` budget.
        """
        window = get_model_context_window(self.agent.model, "openai")
        if not window:
            return 0
        budget = self._retry_budget if self._retry_budget else (window - 10_000)
        if budget <= 0:
            return 0
        trim_target = int(budget * 0.8)

        def _est(s: str) -> int:
            return len(s or "") // 2

        def _content_str(m) -> str:
            if hasattr(m, "get"):
                return m.get("content_text") or m.get("content", "") or ""
            return getattr(m, "content_text", None) or getattr(m, "content", "") or ""

        def _role(m) -> str:
            return m.get("role") if hasattr(m, "get") else getattr(m, "role", "")

        sys_cost = _est(self.system_message)
        total = sys_cost + sum(_est(_content_str(m)) for m in self.messages)
        if total <= budget:
            return 0

        original = len(self.messages)
        LAST_KEEP = 4
        i = 0
        while total > trim_target and len(self.messages) > LAST_KEEP and i < len(self.messages) - LAST_KEEP:
            if _role(self.messages[i]) == "system":
                i += 1
                continue
            total -= _est(_content_str(self.messages[i]))
            del self.messages[i]

        dropped = original - len(self.messages)
        # Keep persisted_count consistent with the shorter in-memory list so
        # the next save() still detects this turn's new messages.
        self._persisted_count = max(0, self._persisted_count - dropped)
        return dropped

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=300, connect=30)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def aclose(self):
        if self._http_session and not self._http_session.closed:
            try:
                await self._http_session.close()
            except Exception:
                pass

    async def chat(
        self,
        tools: list[Callable] = None,
        think: bool = None,
        _retry_attempt: int = 0,
        tool_choice: dict | None = None,
    ) -> AIChatMessage:
        """Public chat() entry — consumes chat_stream() and returns the same
        assistant-message shape AnthropicConversation.chat() returns, so
        runtime._run_turn handles both providers identically."""
        tools_map: dict[str, Callable] = {}
        for func in tools or []:
            tools_map[func.__name__] = func

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        ordered_blocks: list[dict] = []
        framework_error: FrameworkErrorEvent | None = None
        stop_reason: str | None = None

        try:
            async for event in self.chat_stream(
                tools=tools, think=think,
                tool_choice=tool_choice,
                _retry_attempt=_retry_attempt,
            ):
                if isinstance(event, FrameworkErrorEvent):
                    framework_error = event
                    continue
                if isinstance(event, ContentBlockStopEvent):
                    blk = event.completed_block
                    btype = blk.get("type", "")
                    ordered_blocks.append(blk)
                    if btype == "text":
                        text_parts.append(blk.get("text", "") or "")
                    elif btype == "tool_use":
                        name = blk.get("name", "")
                        tool_calls.append(ToolCall(
                            function_name=name,
                            args=blk.get("input", {}) or {},
                            tool_use_id=blk.get("id", ""),
                            function=tools_map.get(name),
                        ))
                elif isinstance(event, MessageDeltaEvent):
                    if event.stop_reason is not None:
                        stop_reason = event.stop_reason
        except Exception as _wrapper_e:
            print_ts(
                f"{COLOR_RED}chat() wrapper consumption error: {_wrapper_e}{COLOR_END}",
                agent=self.agent.id, error=True,
            )
            return AIChatMessage(
                content=f"⚠️ chat() wrapper failed: {_wrapper_e}",
                is_framework_error=True,
            )

        if framework_error is not None:
            return AIChatMessage(
                content=framework_error.message,
                is_framework_error=True,
            )

        # Malformed-tool_use detection + retry — same policy as the
        # anthropic wrapper: a tool call with required fields but empty args
        # gets one [FRAMEWORK]-nudged retry.
        if tool_calls and _retry_attempt < 2:
            malformed = []
            for tc in tool_calls:
                schema = (getattr(tc.function, "tool_spec", {}) or {}).get(
                    "input_schema", {}
                ) or {}
                required = schema.get("required") or []
                if required and not tc.args:
                    malformed.append((tc.function_name, required))
            if malformed:
                names_and_fields = "; ".join(
                    f"{n} (needs: {', '.join(r)})" for n, r in malformed
                )
                print_ts(
                    f"{COLOR_YELLOW}malformed tool_use detected ({names_and_fields}) — "
                    f"retry {_retry_attempt + 1}/2 (openai stream wrapper){COLOR_END}",
                    agent=self.agent.id,
                )
                nudge = (
                    "[FRAMEWORK]: Your last tool call had empty arguments but "
                    f"required fields are: {names_and_fields}. Retry the call with all "
                    "required arguments filled in. If you can't fill them, reply in text "
                    "explaining what you need instead."
                )
                self.messages.append(ChatMessage("user", nudge))
                try:
                    return await self.chat(
                        tools=tools, think=think,
                        tool_choice=tool_choice,
                        _retry_attempt=_retry_attempt + 1,
                    )
                finally:
                    if (self.messages and self.messages[-1].role == "user"
                            and "[FRAMEWORK]:" in (self.messages[-1].get("content", "") or "")):
                        self.messages.pop()

        response = AIChatMessage(
            content="".join(text_parts),
            tool_calls=tool_calls,
        )
        response.raw_response = {"content": ordered_blocks}

        if not response.content and not tool_calls:
            if stop_reason == "refusal":
                msg = "⚠️ Model declined (finish_reason=content_filter)."
            elif stop_reason:
                msg = (
                    f"⚠️ Empty reply (stop_reason={stop_reason}). No text or "
                    f"tool call was returned."
                )
            else:
                # No content, no tool call, AND no finish_reason — a transient
                # API glitch, not a model decision. Return the plain empty
                # response so runtime.py's empty-reply nudge-and-retry path
                # handles it (one-shot per turn) instead of posting a
                # framework-error warning straight to Discord.
                print_ts(
                    f"{COLOR_YELLOW}empty reply with no finish_reason — "
                    f"deferring to runtime nudge-and-retry{COLOR_END}",
                    agent=self.agent.id,
                )
                return response
            print_ts(
                f"{COLOR_RED}{msg}{COLOR_END}",
                agent=self.agent.id, error=True,
            )
            return AIChatMessage(content=msg, is_framework_error=True)

        return response

    async def chat_stream(
        self,
        tools: list[Callable] = None,
        think: bool = None,
        tool_choice: dict | None = None,
        _retry_attempt: int = 0,
    ):
        """Streaming entry — dispatches on auth precedence:
          1. Codex subscription creds present (auth.json, auth_mode chatgpt)
             → _chat_stream_codex (Responses API, subscription billing).
          2. Else API key configured → _chat_stream_apikey (Chat Completions).
          3. Else → clear auth error naming both options.
        Checked per turn, so `codex login` (or deleting auth.json) takes
        effect without a restart."""
        if codex_creds_exist():
            worker = self._chat_stream_codex(
                tools=tools, think=think,
                tool_choice=tool_choice, _retry_attempt=_retry_attempt,
            )
        elif get_openai_api_key():
            worker = self._chat_stream_apikey(
                tools=tools, think=think,
                tool_choice=tool_choice, _retry_attempt=_retry_attempt,
            )
        else:
            yield FrameworkErrorEvent(
                message=("⚠️ OpenAI auth unconfigured — either `codex login` "
                         "with a ChatGPT subscription (writes ~/.codex/auth.json; "
                         "preferred), or set integrations.openai.api_key in "
                         "config.json / the OPENAI_API_KEY environment variable."),
                kind="auth",
            )
            return
        async for ev in worker:
            yield ev

    def _record_usage(self, final_usage: dict, stream_failed: bool) -> None:
        """Shared post-stream bookkeeping for both auth paths: update
        last_usage (+ meta sidecar) and append a usage-ledger row.
        `final_usage` is in Chat-Completions key shape — the codex stream
        translator normalizes Responses usage into it."""
        prompt_tokens = int(final_usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(final_usage.get("completion_tokens", 0) or 0)
        cached = int(((final_usage.get("prompt_tokens_details") or {})
                      .get("cached_tokens", 0)) or 0)
        # Ledger/status key mapping: OpenAI's prompt_tokens INCLUDES cached
        # tokens (Anthropic's input_tokens excludes them), so split here to
        # keep total_input = input + cache_read across providers.
        self.last_usage = {
            "input_tokens": max(prompt_tokens - cached, 0),
            "output_tokens": completion_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cached,
            "total_input": prompt_tokens,
            "ts": time.time(),
            "partial": stream_failed,
        }
        try:
            self._save_meta()
        except Exception:
            pass
        print_ts(
            f"openai usage: in={prompt_tokens} out={completion_tokens} "
            f"cache_read={cached}"
            f"{' (partial)' if stream_failed else ''}",
            agent=self.agent.id,
        )
        try:
            from . import usage_ledger
            _conv_id = getattr(self, "conversation_id", "") or ""
            if ":" in _conv_id:
                _transport, _channel_id = _conv_id.split(":", 1)
            else:
                _transport, _channel_id = None, (_conv_id or None)
            # record_usage reads the anthropic-shaped keys; merge in the raw
            # OpenAI fields so raw_usage preserves everything the API returned.
            _ledger_usage = dict(self.last_usage)
            _ledger_usage["openai_raw"] = final_usage
            usage_ledger.record_usage(
                agent_id=self.agent.id,
                transport=_transport,
                channel_id=_channel_id,
                session_id=_conv_id or None,
                user_id=None,
                user_handle=None,
                model=self.model,
                usage=_ledger_usage,
                outcome=("error" if stream_failed else "ok"),
            )
        except Exception as _led_e:
            print_ts(
                f"usage_ledger record failed (continuing): {_led_e}",
                agent=self.agent.id, error=True,
            )

    async def _chat_stream_apikey(
        self,
        tools: list[Callable] = None,
        think: bool = None,
        tool_choice: dict | None = None,
        _retry_attempt: int = 0,
    ):
        """API-key fallback worker — POSTs to /v1/chat/completions with
        stream=true and yields StreamEvent objects (same taxonomy as the
        anthropic provider, translated from OpenAI's chunk format by
        _stream_openai_events).

        Side effects matching the anthropic sibling:
          - self.last_usage updated + persisted to the meta sidecar.
          - usage_ledger row recorded per completed request.
          - Pre-flight _trim_to_fit_window() — every turn (no server-side
            compaction on this provider).
        Recovery paths: 401 → terminal auth error (API keys don't refresh);
        429 → rate_limit; 400 context overflow → halve budget + retry (≤3).
        """
        api_key = get_openai_api_key()
        if not api_key:
            yield FrameworkErrorEvent(
                message=("⚠️ OpenAI API key unconfigured — set "
                         "integrations.openai.api_key in config.json or the "
                         "OPENAI_API_KEY environment variable."),
                kind="auth",
            )
            return

        dropped = self._trim_to_fit_window()
        if dropped:
            print_ts(
                f"{COLOR_YELLOW}pre-flight trim: dropped {dropped} oldest message(s) "
                f"to fit context window (openai){COLOR_END}",
                agent=self.agent.id,
            )

        api_messages = _openflip_msgs_to_openai(
            self.messages, self.system_message or ""
        )

        # Inject queued image attachments (vision). Pop the queue first so a
        # failed request doesn't double-attach on retry.
        _pending_imgs = getattr(self, "_pending_image_attachments", None) or []
        if _pending_imgs:
            self._pending_image_attachments = []
            _injected = _inject_pending_image_attachments(api_messages, _pending_imgs)
            if _injected:
                print_ts(
                    f"  → injected {_injected} image attachment(s) into request (openai)",
                    agent=self.agent.id,
                )

        # REMINDER.md per-turn injection removed (2026-07-06, operator
        # decision) — standing guidance lives in the cached system files.

        body: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
            # Final chunk carries usage totals — required for last_usage,
            # /status, and the usage ledger.
            "stream_options": {"include_usage": True},
        }
        _max_out = self._max_output_tokens()
        if _max_out:
            body["max_completion_tokens"] = _max_out
        # Per-model reasoning-effort knob → OpenAI reasoning_effort. Only set
        # this in config for reasoning-capable models (o-series / gpt-5
        # family); the API 400s on models that don't accept the parameter.
        _effort = get_effort(self.agent.model, "openai")
        if _effort:
            body["reasoning_effort"] = _effort

        tool_schemas = _build_openai_tool_schemas(tools or [])
        if tool_schemas:
            body["tools"] = tool_schemas
            _tc = _translate_tool_choice(tool_choice)
            if _tc is not None:
                body["tool_choice"] = _tc

        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

        session = await self._ensure_http_session()

        # Request dump — same env knobs as the anthropic provider.
        if os.environ.get("OPENFLIP_REQUEST_DUMP") == "1" or os.environ.get("OPENFLIP_REQUEST_DUMP_FULL") == "1":
            try:
                _stamp = f"{self.agent.id}_{int(time.time() * 1000)}"
                from .utils import project_root as _pr
                _dump_dir = os.path.join(_pr(), "data", "request_dumps")
                os.makedirs(_dump_dir, exist_ok=True)
                _dump_path = os.path.join(_dump_dir, f"{_stamp}.req.json")
                if os.environ.get("OPENFLIP_REQUEST_DUMP_FULL") == "1":
                    _body_summary = body
                else:
                    _body_summary = {k: body[k] for k in ("model", "stream") if k in body}
                    if api_messages:
                        _body_summary["last_message"] = api_messages[-1]
                with open(_dump_path, "w") as _df:
                    json.dump({
                        "url": f"{get_openai_base_url()}/v1/chat/completions",
                        "body_summary": _body_summary,
                    }, _df, indent=2)
            except Exception as _dump_e:
                print_ts(f"{COLOR_YELLOW}request_dump failed: {_dump_e}{COLOR_END}", agent=self.agent.id)

        try:
            async with session.post(
                f"{get_openai_base_url()}/v1/chat/completions",
                json=body,
                headers=headers,
            ) as resp:
                status = resp.status

                if status != 200:
                    text = await resp.text()
                    if status == 401:
                        yield FrameworkErrorEvent(
                            message=("⚠️ OpenAI API key rejected (401). Check "
                                     "integrations.openai.api_key in config.json."),
                            kind="auth",
                        )
                        return
                    if status == 429:
                        snippet = text[:200].replace("\n", " ")
                        yield FrameworkErrorEvent(
                            message=f"⚠️ OpenAI rate limit (429). {snippet}",
                            kind="rate_limit",
                        )
                        return
                    # Context overflow → shrink budget and retry (≤3), same
                    # recovery shape as the anthropic prompt-too-long path.
                    _lower = text.lower()
                    if (
                        status == 400
                        and _retry_attempt < 3
                        and ("context_length_exceeded" in _lower
                             or "maximum context length" in _lower
                             or "context window" in _lower)
                    ):
                        window = get_model_context_window(self.agent.model, "openai")
                        new_budget = max(window // (2 ** (_retry_attempt + 1)), 8_000)
                        print_ts(
                            f"{COLOR_YELLOW}context overflow — retry "
                            f"{_retry_attempt + 1}/3 with budget {new_budget:,} (openai){COLOR_END}",
                            agent=self.agent.id,
                        )
                        self._retry_budget = new_budget
                        try:
                            # Recurse into THIS worker, not the dispatcher —
                            # the retry must stay on the same auth path.
                            async for ev in self._chat_stream_apikey(
                                tools=tools, think=think,
                                tool_choice=tool_choice,
                                _retry_attempt=_retry_attempt + 1,
                            ):
                                yield ev
                            return
                        finally:
                            self._retry_budget = None
                    print_ts(
                        f"{COLOR_RED}OpenAI API {status}: {text[:600]}{COLOR_END}",
                        agent=self.agent.id, error=True,
                    )
                    snippet = text[:200].replace("\n", " ")
                    yield FrameworkErrorEvent(
                        message=f"⚠️ OpenAI API {status}: {snippet}",
                        kind="bad_request",
                    )
                    return

                # Status 200 — translate the SSE stream into StreamEvents.
                _final_usage: dict = {}
                _stream_failed = False
                try:
                    async for ev in _stream_openai_events(resp.content):
                        if isinstance(ev, FrameworkErrorEvent):
                            _stream_failed = True
                        elif isinstance(ev, MessageDeltaEvent) and ev.usage:
                            _final_usage = dict(ev.usage)
                        yield ev
                except Exception as _stream_e:
                    _stream_failed = True
                    print_ts(
                        f"{COLOR_RED}Stream consumption error: {_stream_e}{COLOR_END}",
                        agent=self.agent.id, error=True,
                    )
                    yield FrameworkErrorEvent(
                        message=f"⚠️ Stream error: {_stream_e}",
                        kind="transport",
                    )
                finally:
                    if _final_usage:
                        self._record_usage(_final_usage, _stream_failed)

                if _stream_failed:
                    return

        except asyncio.TimeoutError:
            yield FrameworkErrorEvent(
                message="⚠️ OpenAI API timed out (5 min).",
                kind="timeout",
            )
            return
        except Exception as e:
            print_ts(
                f"{COLOR_RED}chat_stream exception: {e}{COLOR_END}",
                agent=self.agent.id, error=True,
            )
            yield FrameworkErrorEvent(
                message=f"⚠️ OpenAI API error: {e}",
                kind="transport",
            )
            return

    async def _chat_stream_codex(
        self,
        tools: list[Callable] = None,
        think: bool = None,
        tool_choice: dict | None = None,
        _retry_attempt: int = 0,
    ):
        """Subscription-mode worker — POSTs to the Responses API at
        {_CODEX_BASE_URL}/responses with the Codex OAuth bearer token and
        yields StreamEvent objects (translated by _stream_codex_events).

        Auth: borrow_codex_key() refreshes on JWT expiry (30s skew) before
        the request; a 401 forces one refresh-and-retry (mirroring the
        anthropic provider's 401 path), then surfaces a terminal auth error.
        Side effects (last_usage / meta sidecar / usage ledger / pre-flight
        trim / image injection) match the api-key worker.
        Recovery paths: 429 → rate_limit; 400 context overflow → halve
        budget + retry (≤3); other non-200 → bad_request.
        """
        try:
            access_token, account_id = await borrow_codex_key()
        except CodexAuthError as e:
            yield FrameworkErrorEvent(message=f"⚠️ {e}", kind="auth")
            return

        dropped = self._trim_to_fit_window()
        if dropped:
            print_ts(
                f"{COLOR_YELLOW}pre-flight trim: dropped {dropped} oldest message(s) "
                f"to fit context window (openai-codex){COLOR_END}",
                agent=self.agent.id,
            )

        instructions, input_items = _openflip_msgs_to_responses_input(
            self.messages, self.system_message or ""
        )

        # Inject queued image attachments (vision). Pop the queue first so a
        # failed request doesn't double-attach on retry.
        _pending_imgs = getattr(self, "_pending_image_attachments", None) or []
        if _pending_imgs:
            self._pending_image_attachments = []
            _injected = _inject_pending_image_attachments_responses(
                input_items, _pending_imgs
            )
            if _injected:
                print_ts(
                    f"  → injected {_injected} image attachment(s) into request (openai-codex)",
                    agent=self.agent.id,
                )

        # REMINDER.md per-turn injection removed (2026-07-06, operator
        # decision) — standing guidance lives in the cached system files.

        body: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "stream": True,
            # Subscription requests must not be stored server-side (the
            # reference plugin always sends store=False on this backend).
            "store": False,
        }
        if instructions:
            body["instructions"] = instructions
        _max_out = self._max_output_tokens()
        if _max_out:
            body["max_output_tokens"] = _max_out
        _effort = _codex_effort(self.agent.model)
        if _effort:
            body["reasoning"] = {"effort": _effort}

        tool_schemas = _build_responses_tool_schemas(tools or [])
        if tool_schemas:
            body["tools"] = tool_schemas
            _tc = _translate_tool_choice_responses(tool_choice)
            if _tc is not None:
                body["tool_choice"] = _tc

        headers = {
            "Authorization": f"Bearer {access_token}",
            "content-type": "application/json",
        }
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id

        session = await self._ensure_http_session()

        # Request dump — same env knobs as the other providers.
        if os.environ.get("OPENFLIP_REQUEST_DUMP") == "1" or os.environ.get("OPENFLIP_REQUEST_DUMP_FULL") == "1":
            try:
                _stamp = f"{self.agent.id}_{int(time.time() * 1000)}"
                from .utils import project_root as _pr
                _dump_dir = os.path.join(_pr(), "data", "request_dumps")
                os.makedirs(_dump_dir, exist_ok=True)
                _dump_path = os.path.join(_dump_dir, f"{_stamp}.req.json")
                if os.environ.get("OPENFLIP_REQUEST_DUMP_FULL") == "1":
                    _body_summary = body
                else:
                    _body_summary = {k: body[k] for k in ("model", "stream") if k in body}
                    if input_items:
                        _body_summary["last_item"] = input_items[-1]
                with open(_dump_path, "w") as _df:
                    json.dump({
                        "url": f"{_CODEX_BASE_URL}/responses",
                        "body_summary": _body_summary,
                    }, _df, indent=2)
            except Exception as _dump_e:
                print_ts(f"{COLOR_YELLOW}request_dump failed: {_dump_e}{COLOR_END}", agent=self.agent.id)

        try:
            async with session.post(
                f"{_CODEX_BASE_URL}/responses",
                json=body,
                headers=headers,
            ) as resp:
                status = resp.status

                if status != 200:
                    text = await resp.text()
                    if status == 401:
                        if _retry_attempt == 0:
                            print_ts(
                                f"{COLOR_YELLOW}Codex 401 — forcing token refresh "
                                f"and retrying{COLOR_END}",
                                agent=self.agent.id,
                            )
                            try:
                                await borrow_codex_key(force_refresh=True)
                            except CodexAuthError as e:
                                yield FrameworkErrorEvent(message=f"⚠️ {e}", kind="auth")
                                return
                            async for ev in self._chat_stream_codex(
                                tools=tools, think=think,
                                tool_choice=tool_choice,
                                _retry_attempt=_retry_attempt + 1,
                            ):
                                yield ev
                            return
                        yield FrameworkErrorEvent(
                            message=("⚠️ Codex subscription token rejected (401) "
                                     "even after refresh. Run `codex login` again."),
                            kind="auth",
                        )
                        return
                    if status == 429:
                        snippet = text[:200].replace("\n", " ")
                        yield FrameworkErrorEvent(
                            message=(f"⚠️ OpenAI rate/usage limit (429) on the "
                                     f"ChatGPT subscription. {snippet}"),
                            kind="rate_limit",
                        )
                        return
                    # Context overflow → shrink budget and retry (≤3), same
                    # recovery shape as the api-key worker.
                    _lower = text.lower()
                    if (
                        status == 400
                        and _retry_attempt < 3
                        and ("context_length_exceeded" in _lower
                             or "maximum context length" in _lower
                             or "context window" in _lower)
                    ):
                        window = get_model_context_window(self.agent.model, "openai")
                        new_budget = max(window // (2 ** (_retry_attempt + 1)), 8_000)
                        print_ts(
                            f"{COLOR_YELLOW}context overflow — retry "
                            f"{_retry_attempt + 1}/3 with budget {new_budget:,} (openai-codex){COLOR_END}",
                            agent=self.agent.id,
                        )
                        self._retry_budget = new_budget
                        try:
                            async for ev in self._chat_stream_codex(
                                tools=tools, think=think,
                                tool_choice=tool_choice,
                                _retry_attempt=_retry_attempt + 1,
                            ):
                                yield ev
                            return
                        finally:
                            self._retry_budget = None
                    print_ts(
                        f"{COLOR_RED}OpenAI (codex) API {status}: {text[:600]}{COLOR_END}",
                        agent=self.agent.id, error=True,
                    )
                    snippet = text[:200].replace("\n", " ")
                    yield FrameworkErrorEvent(
                        message=f"⚠️ OpenAI (codex) API {status}: {snippet}",
                        kind="bad_request",
                    )
                    return

                # Status 200 — translate the Responses SSE stream.
                _final_usage: dict = {}
                _stream_failed = False
                try:
                    async for ev in _stream_codex_events(resp.content):
                        if isinstance(ev, FrameworkErrorEvent):
                            _stream_failed = True
                        elif isinstance(ev, MessageDeltaEvent) and ev.usage:
                            _final_usage = dict(ev.usage)
                        yield ev
                except Exception as _stream_e:
                    _stream_failed = True
                    print_ts(
                        f"{COLOR_RED}Stream consumption error: {_stream_e}{COLOR_END}",
                        agent=self.agent.id, error=True,
                    )
                    yield FrameworkErrorEvent(
                        message=f"⚠️ Stream error: {_stream_e}",
                        kind="transport",
                    )
                finally:
                    if _final_usage:
                        self._record_usage(_final_usage, _stream_failed)

                if _stream_failed:
                    return

        except asyncio.TimeoutError:
            yield FrameworkErrorEvent(
                message="⚠️ OpenAI (codex) API timed out (5 min).",
                kind="timeout",
            )
            return
        except Exception as e:
            print_ts(
                f"{COLOR_RED}chat_stream exception (codex): {e}{COLOR_END}",
                agent=self.agent.id, error=True,
            )
            yield FrameworkErrorEvent(
                message=f"⚠️ OpenAI (codex) API error: {e}",
                kind="transport",
            )
            return
