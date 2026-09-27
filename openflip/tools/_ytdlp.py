"""Managed yt-dlp binary for download_media (no @tool here).

The newest yt-dlp release lives at <data_dir>/bin/yt-dlp and is re-fetched
when older than a day: sites (YouTube above all) break old builds within
weeks, and a stale system yt-dlp is the usual cause of HTTP 403s. Node is
passed as the JS runtime when deno is absent (yt-dlp needs one for YouTube).
"""
from __future__ import annotations
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from ..config_global import get_config
from ..utils import print_ts, http_session, resolve_path, COLOR_YELLOW, COLOR_GREEN, COLOR_END

# The plain "yt-dlp" asset is a python zipapp (runs anywhere python3 is on PATH);
# Windows can't exec that, so it gets the standalone .exe.
_ASSET = "yt-dlp.exe" if sys.platform == "win32" else "yt-dlp"
_RELEASE_URL = f"https://github.com/yt-dlp/yt-dlp/releases/latest/download/{_ASSET}"
_REFRESH_S = 86400
_lock: Optional[asyncio.Lock] = None


async def run(cmd: list[str], timeout: float) -> tuple[int, str, str]:
    """Run a command with a hard timeout; returns (rc, stdout, stderr). rc -1 = timeout."""
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"timed out after {int(timeout)}s"
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def js_args() -> list[str]:
    """yt-dlp enables deno by default; point it at node when deno isn't installed."""
    if shutil.which("deno"):
        return []
    node = shutil.which("node")
    return ["--js-runtimes", f"node:{node}"] if node else []


async def binary(custom: str = "") -> tuple[Optional[str], str]:
    """(path, "") to a working yt-dlp, or (None, error). Fetches/refreshes the managed copy when due."""
    global _lock
    custom = (custom or "").strip()
    if custom:
        return (custom, "") if os.access(custom, os.X_OK) else (None, f"ytdlp_path is not executable: {custom}")
    path = Path(resolve_path(get_config().get("data_dir", "./data"))) / "bin" / _ASSET
    _lock = _lock or asyncio.Lock()
    async with _lock:
        if path.exists() and time.time() - path.stat().st_mtime < _REFRESH_S:
            return str(path), ""
        try:
            import aiohttp
            s = await http_session()
            async with s.get(_RELEASE_URL, timeout=aiohttp.ClientTimeout(total=120)) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                data = await r.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(_ASSET + ".part")
            await asyncio.to_thread(part.write_bytes, data)
            os.chmod(part, 0o755)
            rc, out, err = await run([str(part), "--version"], 30)
            if rc != 0:
                raise RuntimeError(f"new build won't run: {err.strip()[:150]}")
            os.replace(part, path)
            print_ts(f"{COLOR_GREEN}yt-dlp updated to {out.strip()}{COLOR_END}")
            return str(path), ""
        except Exception as e:
            if path.exists():
                print_ts(f"{COLOR_YELLOW}yt-dlp refresh failed ({e}); using the existing copy{COLOR_END}")
                os.utime(path)  # don't hammer GitHub on every call; retry tomorrow
                return str(path), ""
            return None, f"couldn't fetch yt-dlp: {e}"
