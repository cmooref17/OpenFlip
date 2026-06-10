"""Final-stage validator for the assembled Anthropic /v1/messages body.

Run *immediately before* `session.post(...)`. Accumulates all problems
(fail-fast is hostile when one request can have several independent
shape bugs) and returns a `RequestValidationResult`. The caller raises
`MalformedRequestError` if `result.ok` is False; warnings are logged
but do not block the send.

Why a separate module
---------------------
Anthropic 400s have a habit of being cryptic, and we've shipped three
in one night: tool_use/tool_result adjacency broken by an image inject,
oversized base64 payloads, and unsupported media types. Each was a new
feature accidentally introducing a request-shape bug. This module is
where that gets caught up front, before bytes leave the process.

Reviewer guardrails
-------------------
Several "obvious" checks are intentionally omitted because they would
false-positive on legitimate openflip traffic:

  * "first message must be user role" — compaction blocks legitimately
    appear as the first message in assistant role.
  * `body["system"]` shape checks — the billing block lives there and
    has its own shape; tool_use/tool_result pairing does not apply.
  * strict user/assistant alternation — REMINDER.md injection produces
    consecutive user messages by design.
  * "first user message must follow a preceding assistant" — restart-
    resume injects synthetic continuation user messages with no prior
    assistant turn.

The trailing-user-message rule (`trailing_message_not_user`) IS
enforced as fail-severity. Anthropic rejects bodies whose final
message is role=assistant on every Claude model currently in this
framework with HTTP 400 "model does not support assistant message
prefill. conversation must end with a user message." The post-drain
retry false-fire (incident 2026-05-25) was exactly this shape and
went straight to the operator's channel. False-positive guards:
  1. We never legitimately end on assistant — even REMINDER injection
     adds a user role marker AFTER the assistant turn.
  2. Compaction blocks appear at message[0], not message[-1].
  3. Restart-resume injects a synthetic user message; trailing slot
     is always user.
  4. Assistant prefill is a separate Anthropic feature that would
     require an opt-in code path; this framework never sets it.

Kill switches (set env var to "1" to bypass):
  * OPENFLIP_DISABLE_REQUEST_VALIDATOR — disables the entire validator.
    The validator wrapper in anthropic_conversation.py honors this.
  * (Body-size rule below is warn-severity now, not fail — no per-rule
    switch needed.)

Constants
---------
`HARD_BODY_LIMIT_BYTES` (warn-only): Anthropic's documented total
body cap is 32 MB. We do NOT fail at this threshold — the validator's
guess at the wire limit is what bricked us 2026-05-25 when a 256KB
guess was way too low. Letting our threshold pre-empt Anthropic's
actual 413 has historically been a bigger source of bugs than the
413 itself. Surface as warn so the operator gets visibility on
growing requests; let Anthropic be authoritative on the hard cap.

If you DO need to fail-close before the wire (e.g. to avoid burning
a slow oauth roundtrip on a request that obviously can't make it),
edit `_check_body_size` to re-emit fail at HARD_BODY_LIMIT_BYTES and
DOCUMENT the new threshold against current Anthropic limits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal


# ── Tunables ──────────────────────────────────────────────────────────

MAX_IMAGE_BYTES = 5 * 1024 * 1024            # per-image base64-decoded cap
ALLOWED_IMAGE_TYPES = {
    "image/png", "image/jpeg", "image/gif", "image/webp",
}
# Warn-only threshold for total body size. Anthropic's documented hard
# cap is 32MB; we warn at 28MB so growing requests get logged before
# they hit the wire, but we let Anthropic surface the real 413 instead
# of pre-emptively rejecting. See module docstring for rationale.
HARD_BODY_LIMIT_BYTES = 28 * 1024 * 1024

WARN_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024
WARN_IMAGE_COUNT = 50
WARN_BODY_SIZE = 200 * 1024


# ── Result types ──────────────────────────────────────────────────────

@dataclass
class RequestValidationProblem:
    severity:Literal["warn", "fail"]
    rule:str
    detail:str
    location:str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.rule} @ {self.location}: {self.detail}"


@dataclass
class RequestValidationResult:
    ok:bool
    problems:list[RequestValidationProblem] = field(default_factory=list)

    def fails(self) -> list[RequestValidationProblem]:
        return [p for p in self.problems if p.severity == "fail"]

    def warns(self) -> list[RequestValidationProblem]:
        return [p for p in self.problems if p.severity == "warn"]


# ── Helpers ───────────────────────────────────────────────────────────

def _b64_decoded_len(b64:str) -> int:
    """Compute decoded byte length of a base64 string without decoding it.
    Subtracts trailing `=` padding (each `=` removes one byte)."""
    if not isinstance(b64, str) or not b64:
        return 0
    padding = b64.count("=") if b64.endswith("=") else 0
    # Trim only the trailing pad chars, not the body.
    n = len(b64)
    return (n * 3) // 4 - padding


def _iter_content_blocks(msg:dict):
    """Yield (index, block) for every content block in a message.
    Tolerates `content` being a str (yields nothing — strings have no
    typed blocks to inspect)."""
    c = msg.get("content")
    if isinstance(c, list):
        for i, b in enumerate(c):
            if isinstance(b, dict):
                yield i, b


# ── FAIL-severity rules ───────────────────────────────────────────────

def _check_trailing_message_role(body:dict, problems:list) -> None:
    """The final message in body['messages'] must be role='user'.

    Anthropic rejects bodies ending on assistant on every Claude model
    currently in this framework with HTTP 400 "model does not support
    assistant message prefill. conversation must end with a user
    message." See module docstring for the four false-positive guards
    documented for this rule.
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return  # _check_messages_empty handles this
    last = msgs[-1]
    if not isinstance(last, dict):
        return
    role = last.get("role")
    if role != "user":
        problems.append(RequestValidationProblem(
            severity="fail",
            rule="anthropic_models_require_trailing_user_message",
            detail=(
                f"final message has role={role!r}; Anthropic requires the "
                f"trailing message to be role='user' (assistant prefill is "
                f"not supported by current Claude models)"
            ),
            location=f"messages[{len(msgs) - 1}]",
        ))


def _check_messages_empty(body:dict, problems:list) -> None:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or len(msgs) == 0:
        problems.append(RequestValidationProblem(
            severity="fail",
            rule="messages_empty",
            detail="body['messages'] is missing or empty",
            location="body[messages]",
        ))


def _check_tool_use_pairing(body:dict, problems:list) -> None:
    """Every assistant tool_use must be followed by a user message whose
    content contains a tool_result with the matching tool_use_id. The
    tool_result need not be the FIRST block — Anthropic accepts other
    blocks alongside.

    Conversely, every tool_result in a user message must reference a
    tool_use from the IMMEDIATELY PRECEDING assistant message.
    """
    msgs = body.get("messages") or []
    n = len(msgs)
    for i, msg in enumerate(msgs):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "assistant":
            tool_use_ids:list[tuple[int, str]] = []
            for j, blk in _iter_content_blocks(msg):
                if blk.get("type") == "tool_use":
                    tu_id = blk.get("id", "")
                    tool_use_ids.append((j, tu_id))
            if not tool_use_ids:
                continue
            # Need a next message that is user-role and contains matching
            # tool_result blocks.
            if i + 1 >= n:
                for j, tu_id in tool_use_ids:
                    problems.append(RequestValidationProblem(
                        severity="fail",
                        rule="tool_use_orphan",
                        detail=(
                            f"tool_use id={tu_id!r} has no following message "
                            f"(it is the last message in the request)"
                        ),
                        location=f"messages[{i}].content[{j}]",
                    ))
                continue
            nxt = msgs[i + 1]
            if not isinstance(nxt, dict) or nxt.get("role") != "user":
                for j, tu_id in tool_use_ids:
                    problems.append(RequestValidationProblem(
                        severity="fail",
                        rule="tool_use_orphan",
                        detail=(
                            f"tool_use id={tu_id!r} is followed by a non-user "
                            f"message (role={nxt.get('role') if isinstance(nxt, dict) else type(nxt).__name__})"
                        ),
                        location=f"messages[{i}].content[{j}]",
                    ))
                continue
            # Collect tool_result ids in the next message.
            next_result_ids = set()
            for _, blk in _iter_content_blocks(nxt):
                if blk.get("type") == "tool_result":
                    rid = blk.get("tool_use_id", "")
                    if rid:
                        next_result_ids.add(rid)
            for j, tu_id in tool_use_ids:
                if tu_id not in next_result_ids:
                    problems.append(RequestValidationProblem(
                        severity="fail",
                        rule="tool_use_orphan",
                        detail=(
                            f"tool_use id={tu_id!r} has no matching tool_result "
                            f"in the next user message (found ids: "
                            f"{sorted(next_result_ids) or 'none'})"
                        ),
                        location=f"messages[{i}].content[{j}]",
                    ))

        elif role == "user":
            # tool_result orphans: must be a tool_use in the immediately
            # preceding assistant message with the matching id.
            local_results:list[tuple[int, str]] = []
            for j, blk in _iter_content_blocks(msg):
                if blk.get("type") == "tool_result":
                    rid = blk.get("tool_use_id", "")
                    local_results.append((j, rid))
            if not local_results:
                continue
            prev = msgs[i - 1] if i > 0 else None
            prev_use_ids = set()
            if isinstance(prev, dict) and prev.get("role") == "assistant":
                for _, blk in _iter_content_blocks(prev):
                    if blk.get("type") == "tool_use":
                        uid = blk.get("id", "")
                        if uid:
                            prev_use_ids.add(uid)
            for j, rid in local_results:
                if rid not in prev_use_ids:
                    problems.append(RequestValidationProblem(
                        severity="fail",
                        rule="tool_result_orphan",
                        detail=(
                            f"tool_result tool_use_id={rid!r} has no matching "
                            f"tool_use in the immediately preceding assistant "
                            f"message (found ids: {sorted(prev_use_ids) or 'none'})"
                        ),
                        location=f"messages[{i}].content[{j}]",
                    ))


def _check_image_blocks(body:dict, problems:list) -> tuple[int, int]:
    """Validate per-image constraints and return (total_bytes, count)
    for the warn-level aggregate checks."""
    msgs = body.get("messages") or []
    total_bytes = 0
    count = 0
    for i, msg in enumerate(msgs):
        if not isinstance(msg, dict):
            continue
        for j, blk in _iter_content_blocks(msg):
            if blk.get("type") != "image":
                continue
            count += 1
            src = blk.get("source") or {}
            mt = src.get("media_type", "")
            if mt not in ALLOWED_IMAGE_TYPES:
                problems.append(RequestValidationProblem(
                    severity="fail",
                    rule="image_unsupported_media_type",
                    detail=(
                        f"media_type={mt!r} is not one of "
                        f"{sorted(ALLOWED_IMAGE_TYPES)}"
                    ),
                    location=f"messages[{i}].content[{j}]",
                ))
            b64 = src.get("data", "")
            decoded_len = _b64_decoded_len(b64)
            total_bytes += decoded_len
            if decoded_len > MAX_IMAGE_BYTES:
                problems.append(RequestValidationProblem(
                    severity="fail",
                    rule="image_oversize",
                    detail=(
                        f"image is ~{decoded_len} bytes decoded; "
                        f"cap is {MAX_IMAGE_BYTES}"
                    ),
                    location=f"messages[{i}].content[{j}]",
                ))
    return total_bytes, count


def _check_body_size(body:dict, problems:list) -> int:
    """Serialize the body to measure size. Returns the byte count.

    Emits a fail only if the body cannot be serialized at all
    (json.dumps raises) — that's a local Python bug, not a wire
    limit. Oversize is reported as warn by _check_warn_thresholds,
    NOT here, so the validator never pre-empts Anthropic's real
    413. See module docstring for rationale.
    """
    try:
        size = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError) as e:
        problems.append(RequestValidationProblem(
            severity="fail",
            rule="body_not_serializable",
            detail=f"json.dumps failed: {e}",
            location="body",
        ))
        return 0
    if size > HARD_BODY_LIMIT_BYTES:
        problems.append(RequestValidationProblem(
            severity="warn",
            rule="body_near_anthropic_cap",
            detail=(
                f"serialized body is {size} bytes; warn threshold "
                f"is {HARD_BODY_LIMIT_BYTES} (Anthropic's documented "
                f"hard cap is 32MB — the request may still succeed)"
            ),
            location="body",
        ))
    return size


# ── WARN-severity rules ───────────────────────────────────────────────

def _check_warn_thresholds(
    body:dict,
    problems:list,
    total_image_bytes:int,
    image_count:int,
    body_size:int,
) -> None:
    if total_image_bytes > WARN_TOTAL_IMAGE_BYTES:
        problems.append(RequestValidationProblem(
            severity="warn",
            rule="total_image_bytes_high",
            detail=(
                f"total image bytes ~{total_image_bytes} exceeds "
                f"{WARN_TOTAL_IMAGE_BYTES} warn threshold"
            ),
            location="body[messages][*image*]",
        ))
    if image_count > WARN_IMAGE_COUNT:
        problems.append(RequestValidationProblem(
            severity="warn",
            rule="image_count_high",
            detail=(
                f"{image_count} image blocks across all messages "
                f"(warn threshold {WARN_IMAGE_COUNT})"
            ),
            location="body[messages][*image*]",
        ))
    if body_size > WARN_BODY_SIZE:
        problems.append(RequestValidationProblem(
            severity="warn",
            rule="body_size_high",
            detail=(
                f"serialized body is {body_size} bytes "
                f"(warn threshold {WARN_BODY_SIZE})"
            ),
            location="body",
        ))


def _check_empty_assistant_content(body:dict, problems:list) -> None:
    """Warn on empty assistant messages — they're a known historical
    artifact from old framework bugs and shouldn't fail the request."""
    msgs = body.get("messages") or []
    for i, msg in enumerate(msgs):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        c = msg.get("content")
        empty = (
            c is None
            or (isinstance(c, str) and c == "")
            or (isinstance(c, list) and len(c) == 0)
        )
        if empty:
            problems.append(RequestValidationProblem(
                severity="warn",
                rule="empty_assistant_content",
                detail="assistant message has empty content",
                location=f"messages[{i}]",
            ))


# ── Entry point ───────────────────────────────────────────────────────

def validate_anthropic_request(body:dict) -> RequestValidationResult:
    """Run every rule, accumulate problems, return the result.

    Never raises — caller decides what to do with the result. `ok` is
    True iff there are zero `severity='fail'` problems.
    """
    problems:list[RequestValidationProblem] = []

    _check_messages_empty(body, problems)
    # Pairing checks rely on messages being a list; safe to skip when not.
    if isinstance(body.get("messages"), list) and body["messages"]:
        _check_trailing_message_role(body, problems)
        _check_tool_use_pairing(body, problems)
        total_img_bytes, img_count = _check_image_blocks(body, problems)
    else:
        total_img_bytes, img_count = 0, 0
    body_size = _check_body_size(body, problems)
    _check_warn_thresholds(
        body, problems, total_img_bytes, img_count, body_size,
    )
    _check_empty_assistant_content(body, problems)

    ok = not any(p.severity == "fail" for p in problems)
    return RequestValidationResult(ok=ok, problems=problems)


# ── Inline tests ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    failures:list[str] = []

    def _expect_ok(name:str, body:dict, allowed_warns:set | None = None) -> None:
        r = validate_anthropic_request(body)
        if not r.ok:
            failures.append(
                f"{name}: expected ok=True, got fails={[str(p) for p in r.fails()]}"
            )
            return
        # If allowed_warns is given, every warn must be in it (but warns
        # are not required to fire). If None, any warn set is fine.
        if allowed_warns is not None:
            unexpected = [p for p in r.warns() if p.rule not in allowed_warns]
            if unexpected:
                failures.append(
                    f"{name}: unexpected warns: {[str(p) for p in unexpected]}"
                )
                return
        print(f"  PASS  {name}")

    def _expect_fail(name:str, body:dict, rule:str) -> None:
        r = validate_anthropic_request(body)
        if r.ok:
            failures.append(f"{name}: expected fail '{rule}', got ok=True")
            return
        matched = [p for p in r.fails() if p.rule == rule]
        if not matched:
            failures.append(
                f"{name}: expected fail rule={rule!r}, got "
                f"{[str(p) for p in r.fails()]}"
            )
            return
        print(f"  PASS  {name}  ({matched[0].rule})")

    def _tiny_b64(n_bytes:int) -> str:
        """Build a base64 string that decodes to ~n_bytes."""
        import base64
        return base64.b64encode(b"x" * n_bytes).decode("ascii")

    print("validate_anthropic_request() inline tests:")

    # ── POSITIVE cases ────────────────────────────────────────────────

    # A: minimal valid
    _expect_ok("A: minimal valid", {
        "model": "claude-x",
        "messages": [{"role": "user", "content": "hi"}],
    })

    # B: tool_use → tool_result happy path
    _expect_ok("B: tool_use/tool_result matched", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "done"},
            ]},
        ],
    })

    # C: compaction block as first message (assistant role, no tool_use)
    _expect_ok("C: compaction first as assistant", {
        "model": "claude-x",
        "messages": [
            {"role": "assistant", "content": [
                {"type": "compaction", "summary": "earlier stuff"},
            ]},
            {"role": "user", "content": "continue"},
        ],
    })

    # D: billing system block in body['system']; normal tool cycles in messages
    _expect_ok("D: billing system + tool cycle", {
        "model": "claude-x",
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=00000;"},
            {"type": "text", "text": "system prompt body"},
        ],
        "messages": [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_a", "name": "f", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_a", "content": "ok"},
            ]},
        ],
    })

    # E: REMINDER between cached history and the real new user message
    _expect_ok("E: REMINDER consecutive user messages", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier reply"},
            {"role": "user", "content": "[SYSTEM REMINDER]: stay on task"},
            {"role": "user", "content": "real question"},
        ],
    })

    # F: persisted empty assistant message → WARN, not FAIL
    _expect_ok(
        "F: empty assistant content → WARN only",
        {
            "model": "claude-x",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "again"},
            ],
        },
        allowed_warns={"empty_assistant_content"},
    )
    # Sanity: that one above MUST have emitted the empty-content warn.
    _r_F = validate_anthropic_request({
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "again"},
        ],
    })
    if not any(p.rule == "empty_assistant_content" for p in _r_F.warns()):
        failures.append("F: expected empty_assistant_content WARN was not emitted")
    else:
        print("  PASS  F: empty_assistant_content WARN emitted")

    # G: image_blocks as SEPARATE user message AFTER tool_result message.
    #    The tool_use_orphan check must accept this because messages[N+1]
    #    contains the matching tool_result; images sit at messages[N+2].
    _expect_ok("G: tool_result then image-blocks-as-separate-user", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "look at this"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_g", "name": "f", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_g", "content": "ok"},
            ]},
            {"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _tiny_b64(1024),
                }},
            ]},
        ],
    })

    # ── NEGATIVE cases ────────────────────────────────────────────────

    # H: tool_use with no following message
    _expect_fail("H: tool_use is last message", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_h", "name": "f", "input": {}},
            ]},
        ],
    }, rule="tool_use_orphan")

    # I: tool_use → next message is assistant
    _expect_fail("I: tool_use followed by assistant", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_i", "name": "f", "input": {}},
            ]},
            {"role": "assistant", "content": "oops"},
        ],
    }, rule="tool_use_orphan")

    # J: tool_use → next is user but only text (no tool_result block)
    _expect_fail("J: next user has text but no tool_result", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_j", "name": "f", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "hey what about this"},
            ]},
        ],
    }, rule="tool_use_orphan")

    # K: tool_use ids don't match the tool_result id
    _expect_fail("K: tool_use id mismatch", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_k", "name": "f", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_OTHER", "content": "x"},
            ]},
        ],
    }, rule="tool_use_orphan")

    # L: tool_result in user message without preceding assistant tool_use
    _expect_fail("L: tool_result with no preceding tool_use", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "no tool here"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_l", "content": "?"},
            ]},
        ],
    }, rule="tool_result_orphan")

    # M: unsupported media type
    _expect_fail("M: image/bmp rejected", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/bmp",
                    "data": _tiny_b64(64),
                }},
            ]},
        ],
    }, rule="image_unsupported_media_type")

    # N: image > 5MB (decoded)
    _expect_fail("N: oversize image rejected", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _tiny_b64(MAX_IMAGE_BYTES + 10_000),
                }},
            ]},
        ],
    }, rule="image_oversize")

    # O: messages=[] → fail messages_empty
    _expect_fail("O: empty messages list", {
        "model": "claude-x",
        "messages": [],
    }, rule="messages_empty")

    # P: trailing assistant message → fail
    #    (regression for post_drain_retry false-fire 2026-05-25)
    _expect_fail("P: trailing assistant message rejected", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "oh you wanted me to keep going?"},
        ],
    }, rule="anthropic_models_require_trailing_user_message")

    # Q: trailing tool_use (also assistant role) → fail
    _expect_fail("Q: trailing tool_use rejected", {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_q", "name": "f", "input": {}},
            ]},
        ],
    }, rule="anthropic_models_require_trailing_user_message")

    # R: body that previously would have failed at body_too_large
    #    (28MB+ body) now passes the fail check; we just emit a warn.
    #    Build by stuffing one giant text content block.
    _big_text = "x" * (HARD_BODY_LIMIT_BYTES + 100_000)
    _expect_ok(
        "R: oversize body is warn-only (not fail)",
        {
            "model": "claude-x",
            "messages": [
                {"role": "user", "content": _big_text},
            ],
        },
        allowed_warns={"body_near_anthropic_cap", "body_size_high"},
    )

    # ── Wrap up ───────────────────────────────────────────────────────

    total = 18  # A-G (7) + H-O (8) + P-R (3) = 18
    if failures:
        print(f"\n{len(failures)} failures:")
        for line in failures:
            print(f"  - {line}")
        sys.exit(1)
    print(f"\nall {total} tests passed")
