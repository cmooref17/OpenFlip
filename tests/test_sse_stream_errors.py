"""Pins how stream_sse_events surfaces failed streams (2026-11-19 fix).

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_sse_stream_errors.py

Background: on 2026-11-19 from 17:29 every compaction request for one agent
came back as HTTP 200 with no content and no stop_reason, for ~10 minutes.
The parser dropped the API's mid-stream `error` event as an "unknown event",
and a stream that ended before message_start yielded nothing at all, so the
caller built a blank reply and the operator saw only "Compacting
conversation..." spam. Both shapes must now surface as FrameworkErrorEvent
with a kind chat() knows how to retry (or a terminal kind for bad requests).
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip._anthropic_stream import (  # noqa: E402
    stream_sse_events, FrameworkErrorEvent, MessageStartEvent,
    ContentBlockStopEvent, MessageStopEvent,
)

START = 'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":5}}}\n\n'
PING = 'event: ping\ndata: {"type":"ping"}\n\n'
TEXT = (
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"hi"}}\n\n'
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
)
DELTA = 'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
STOP = 'event: message_stop\ndata: {"type":"message_stop"}\n\n'


def err(etype: str, msg: str = "boom") -> str:
    return ('event: error\ndata: {"type":"error","error":{"type":"%s","message":"%s"}}\n\n'
            % (etype, msg))


async def _feed(body: str, chunk: int):
    raw = body.encode()
    for i in range(0, len(raw), chunk):
        yield raw[i:i + chunk]


def events(body: str, chunk: int = 7) -> list:
    async def run():
        return [e async for e in stream_sse_events(_feed(body, chunk))]
    return asyncio.run(run())


def errors(evs: list) -> list:
    return [e for e in evs if isinstance(e, FrameworkErrorEvent)]


def test_error_only_stream_surfaces_overloaded():
    errs = errors(events(err("overloaded_error", "Overloaded")))
    assert len(errs) == 1, errs
    assert errs[0].kind == "overloaded", errs[0]
    assert "overloaded_error" in errs[0].message and "Overloaded" in errs[0].message


def test_error_after_message_start_keeps_specific_kind():
    evs = events(START + PING + err("overloaded_error"))
    errs = errors(evs)
    # exactly one error, and the end-of-stream "closed after message_start"
    # check must not replace the specific overloaded kind
    assert len(errs) == 1, [e.message for e in errs]
    assert errs[0].kind == "overloaded"


def test_error_type_mapping():
    cases = {
        "rate_limit_error": "rate_limit",
        "api_error": "overloaded",
        "timeout_error": "timeout",
        "invalid_request_error": "bad_request",
        "request_too_large": "bad_request",
        "authentication_error": "auth",
        "some_new_error_type": "transport",
    }
    for etype, kind in cases.items():
        errs = errors(events(START + err(etype)))
        assert len(errs) == 1 and errs[0].kind == kind, (etype, errs)


def test_empty_200_body_surfaces_empty_stream():
    errs = errors(events(""))
    assert len(errs) == 1 and errs[0].kind == "empty_stream", errs


def test_pings_only_surfaces_empty_stream():
    errs = errors(events(PING + PING))
    assert len(errs) == 1 and errs[0].kind == "empty_stream", errs


def test_clean_stream_has_no_error():
    evs = events(START + TEXT + DELTA + STOP, chunk=5)
    assert not errors(evs), errors(evs)
    assert isinstance(evs[0], MessageStartEvent)
    assert any(isinstance(e, ContentBlockStopEvent) and e.completed_block.get("text") == "hi"
               for e in evs)
    assert isinstance(evs[-1], MessageStopEvent)


def test_truncated_stream_still_incomplete():
    errs = errors(events(START + TEXT))
    assert len(errs) == 1 and errs[0].kind == "incomplete_stream", errs


def test_empty_stream_is_retryable_in_chat():
    # chat()'s retry set lives inside the method; check the source so a
    # later edit can't silently drop empty_stream from it.
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "openflip", "anthropic_conversation.py"), encoding="utf-8").read()
    start = src.index("_TRANSIENT_KINDS = {")
    block = src[start:src.index("}", start)]
    for kind in ("empty_stream", "overloaded", "rate_limit", "timeout", "transport"):
        assert f'"{kind}"' in block, kind
    assert '"bad_request"' not in block


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    print(f"{'FAILED' if failed else 'OK'}: {failed} failed")
    sys.exit(1 if failed else 0)
