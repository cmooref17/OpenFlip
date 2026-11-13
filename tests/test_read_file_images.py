"""read_file shows image files to the model instead of failing on them.

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_read_file_images.py

Why: read_file used to open everything as UTF-8 text, so a PNG failed with
"'utf-8' codec can't decode byte 0x89" and agents concluded they "can't see
images". Pins:
  (1) a PNG lands in the conversation's vision queue as a real, valid image
  (2) a JPEG with the wrong extension is still detected (sniffed from bytes)
  (3) a huge image is downscaled to the vision cap before it is queued
  (4) text files still return their text, unchanged
  (5) a non-image binary gets a clear "binary file" error, not a codec error
  (6) with no live conversation the tool says so plainly (no silent success)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from openflip.tools import files as fl  # noqa: E402
from openflip.tools import fetch_discord_message as fdm  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILURES.append(label)


class FakeConv:
    _pending_image_attachments: list = []


def run(coro):
    return asyncio.run(coro)


def main() -> int:
    d = tempfile.mkdtemp(prefix="rf_img_")
    png = os.path.join(d, "shot.png")
    Image.new("RGB", (40, 30), (200, 20, 20)).save(png)
    jpg_wrong_ext = os.path.join(d, "photo.dat")
    Image.new("RGB", (20, 20), (0, 0, 255)).save(jpg_wrong_ext, "JPEG")
    big = os.path.join(d, "big.png")
    Image.new("RGB", (4000, 1000), (0, 200, 0)).save(big)
    txt = os.path.join(d, "notes.txt")
    open(txt, "w").write("hello world\n")
    blob = os.path.join(d, "blob.bin")
    open(blob, "wb").write(b"\x00\x89\xff\xfe" * 50)

    conv = FakeConv()
    conv._pending_image_attachments = []

    # Route the tool to our fake conversation and allow the temp dir.
    fl._check_access = lambda full, mode: None
    fl._resolve = lambda p: p
    orig_conv = fdm._conv_for_current_turn
    fdm._conv_for_current_turn = lambda runner: conv

    class _Agent:
        id = "test-agent"
    from openflip.tool_executor import CURRENT_AGENT
    from openflip.registry import RUNNERS
    tok = CURRENT_AGENT.set(_Agent())
    RUNNERS["test-agent"] = object()
    try:
        print("(1) PNG goes to vision")
        r = run(fl.read_file(png))
        check("ok", r.ok)
        check("one image queued", len(conv._pending_image_attachments) == 1)
        q = conv._pending_image_attachments[-1] if conv._pending_image_attachments else {}
        check("queued file is a valid image", bool(q) and Image.open(q["path"]).size == (40, 30))
        check("feedback tells the model to look", "look at it" in (r.model_feedback or ""))

        print("(2) JPEG with a .dat extension is still an image")
        r = run(fl.read_file(jpg_wrong_ext))
        check("ok", r.ok and len(conv._pending_image_attachments) == 2)

        print("(3) big image downscaled to the cap")
        r = run(fl.read_file(big))
        q = conv._pending_image_attachments[-1]
        check("longest side <= cap", r.ok and max(Image.open(q["path"]).size) <= fl._READ_IMAGE_MAX_EDGE)

        print("(4) text unchanged")
        r = run(fl.read_file(txt))
        check("text returned", r.ok and r.model_feedback == "hello world\n")
        check("nothing queued for text", len(conv._pending_image_attachments) == 3)

        print("(5) other binary: clear error")
        r = run(fl.read_file(blob))
        check("fails", not r.ok)
        check("says binary, not codec", "binary file" in (r.error or "") and "codec" not in (r.error or ""))

        print("(6) no conversation: says so")
        fdm._conv_for_current_turn = lambda runner: None
        r = run(fl.read_file(png))
        check("fails plainly", not r.ok and "could not be shown" in (r.error or ""))
    finally:
        fdm._conv_for_current_turn = orig_conv
        CURRENT_AGENT.reset(tok)
        RUNNERS.pop("test-agent", None)

    print("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
