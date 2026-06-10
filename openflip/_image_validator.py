"""Central image validation/normalization for Anthropic vision.

ALL image-content constraints required by the Anthropic vision API live
here. Code elsewhere (boundary downloads, inject-side safety net) just
calls `validate_and_normalize_image(path, declared_media_type)` and trusts
the returned bytes to be API-compliant. If Anthropic adds a new constraint
(e.g. animated-frame handling, color mode, byte cap), only this module
changes — downstream code doesn't know or care about the contract details.

Anthropic vision contract (per platform.claude.com/docs vision):
  - source.base64 RAW bytes <= 5 MB (NOT post-base64)
  - max dimensions 8000x8000 px
  - allowed media types: image/png, image/jpeg, image/gif, image/webp
  - animated webp/gif: only frame 0 is visioned

This module is the SINGLE PLACE these rules live.
"""
from __future__ import annotations

import io
import os


MAX_BYTES = 5 * 1024 * 1024
MAX_DIM = 8000
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
JPEG_QUALITY = 85
MAX_DOWNSCALE_ATTEMPTS = 6


def validate_and_normalize_image(
    path: str,
    declared_media_type: str | None = None,
) -> tuple[bytes, str] | tuple[None, str]:
    """Decode the file at `path`, enforce Anthropic vision constraints, and
    return normalized bytes guaranteed to satisfy every constraint.

    Returns:
        (bytes, media_type) on success — the bytes satisfy size <= 5 MB,
            dims <= 8000x8000, and media_type is one of ALLOWED_TYPES.
            Animated webp/gif are reduced to frame 0. Non-RGB modes are
            converted (palette/alpha flattened to white background).
            On any normalization the output is JPEG q=85.
        (None, reason) on rejection — `reason` is a short human-readable
            string the caller can surface to the user or log.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception as e:
        return None, f"could not read file: {e}"
    if not raw:
        return None, "file is empty"

    try:
        from PIL import Image
    except ImportError:
        return None, "PIL not available"

    try:
        im = Image.open(io.BytesIO(raw))
    except Exception as e:
        return None, f"not a decodable image: {e}"

    # Animated webp/gif: pin to frame 0. PIL exposes multi-frame images via
    # seek(); seek(0) is safe even on single-frame images.
    try:
        im.seek(0)
    except Exception:
        pass

    # Coerce mode for JPEG fallback. P (palette), RGBA, LA all break JPEG;
    # flatten to RGB on a white background.
    if im.mode not in ("RGB", "L"):
        if im.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            rgba = im.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")

    # Dimension cap.
    w, h = im.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        im = im.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.LANCZOS,
        )

    # Byte cap: re-encode JPEG q=85, halve dims when still over, capped at
    # MAX_DOWNSCALE_ATTEMPTS iterations.
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    shrunk = buf.getvalue()
    attempts = 0
    while len(shrunk) > MAX_BYTES and attempts < MAX_DOWNSCALE_ATTEMPTS:
        w, h = im.size
        im = im.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
        shrunk = buf.getvalue()
        attempts += 1
    if len(shrunk) > MAX_BYTES:
        return None, (
            f"couldn't fit {MAX_BYTES} byte cap after {attempts} downscales"
        )

    return shrunk, "image/jpeg"


if __name__ == "__main__":
    import sys
    import tempfile
    from PIL import Image

    failed = 0

    def _check(name, cond, detail=""):
        global failed
        if cond:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}: {detail}")
            failed += 1

    def _make_temp(im, fmt, **kw):
        fd, path = tempfile.mkstemp(suffix=f".{fmt.lower()}")
        os.close(fd)
        im.save(path, format=fmt, **kw)
        return path

    print("validate_and_normalize_image() inline tests:")

    # 1. small valid PNG -> passes through
    p1 = _make_temp(Image.new("RGB", (64, 64), (255, 0, 0)), "PNG")
    b1, mt1 = validate_and_normalize_image(p1, "image/png")
    _check("small PNG accepted",
           b1 is not None and len(b1) > 0,
           f"got bytes={None if b1 is None else len(b1)} mt={mt1}")
    os.unlink(p1)

    # 2. large noisy JPEG -> downscaled to fit 5MB cap
    big = Image.new("RGB", (4000, 4000))
    px = big.load()
    for x in range(0, 4000, 8):
        for y in range(0, 4000, 8):
            px[x, y] = ((x * 31) & 0xFF, (y * 71) & 0xFF, ((x + y) * 13) & 0xFF)
    p2 = _make_temp(big, "JPEG", quality=98)
    raw_size = os.path.getsize(p2)
    b2, mt2 = validate_and_normalize_image(p2, "image/jpeg")
    _check("large noisy JPEG fits 5MB",
           b2 is not None and len(b2) <= MAX_BYTES,
           f"raw={raw_size} -> shrunk={None if b2 is None else len(b2)}")
    os.unlink(p2)

    # 3. dimension overflow -> resized under MAX_DIM
    huge = Image.new("RGB", (9000, 9000), (128, 128, 128))
    p3 = _make_temp(huge, "PNG")
    b3, mt3 = validate_and_normalize_image(p3, "image/png")
    if b3 is None:
        _check("9000x9000 dim-resized", False, f"rejected: {mt3}")
    else:
        from PIL import Image as _I
        decoded = _I.open(io.BytesIO(b3))
        _check("9000x9000 dim-resized",
               max(decoded.size) <= MAX_DIM,
               f"post dims={decoded.size}")
    os.unlink(p3)

    # 4. animated webp -> first frame extracted, file accepted
    try:
        fd4, p4 = tempfile.mkstemp(suffix=".webp")
        os.close(fd4)
        frames = [
            Image.new("RGB", (200, 200), ((i * 60) % 256, 0, 0))
            for i in range(4)
        ]
        frames[0].save(
            p4, format="WEBP", save_all=True,
            append_images=frames[1:], duration=100, loop=0,
        )
        b4, mt4 = validate_and_normalize_image(p4, "image/webp")
        _check("animated webp accepted (frame 0)",
               b4 is not None and len(b4) > 0,
               f"got bytes={None if b4 is None else len(b4)} mt={mt4}")
        os.unlink(p4)
    except Exception as _e:
        _check("animated webp accepted (frame 0)", False,
               f"creation/decode error: {_e}")

    # 5. non-image binary -> rejected cleanly
    fd5, p5 = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd5, "wb") as f:
        f.write(b"this is not an image, just garbage bytes" * 50)
    b5, mt5 = validate_and_normalize_image(p5, "image/png")
    _check("non-image binary rejected",
           b5 is None,
           f"unexpectedly returned bytes (mt={mt5})")
    os.unlink(p5)

    if failed:
        print(f"\n{failed} test(s) failed")
        sys.exit(1)
    print("\nAll tests passed")
