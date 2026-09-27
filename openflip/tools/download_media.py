"""download_media: download a video or its audio from a link via yt-dlp.

Framework-core tool. The yt-dlp binary is auto-managed by _ytdlp.py (newest
release, refreshed daily). Needs ffmpeg on PATH (merging, audio extraction,
sections) and, for YouTube, a JS runtime (deno, or node as a fallback).
"""
from __future__ import annotations
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from ._base import tool, ToolResult
from . import _ytdlp
from .. import tool_settings as ts
from ..config_global import get_config
from ..utils import print_ts, safe_filename, resolve_path, COLOR_YELLOW, COLOR_GREEN, COLOR_END

ts.register("download_media", [
    ts.SettingSchema("max_duration_s", "int", 1800, "Refuse media longer than this (s) unless a start/end section is given", min=10, max=36000),
    ts.SettingSchema("max_filesize_mb", "int", 200, "Abort downloads bigger than this (MB)", min=1, max=4000),
    ts.SettingSchema("max_attach_mb", "int", 10, "Files up to this size are posted in chat; bigger ones stay on disk (Discord caps unboosted uploads at 10 MB)", min=1, max=500),
    ts.SettingSchema("video_max_height", "int", 720, "Pick the best video at or below this height", min=144, max=4320),
    ts.SettingSchema("audio_format", "choice", "mp3", "Container for audio_only downloads", choices=["mp3", "m4a", "opus", "wav", "flac"]),
    ts.SettingSchema("audio_quality", "int", 2, "yt-dlp --audio-quality: 0 = best VBR .. 10 = worst", min=0, max=10),
    ts.SettingSchema("timeout_s", "int", 600, "Kill a download that runs longer than this (s)", min=30, max=3600),
    ts.SettingSchema("ytdlp_path", "str", "", "yt-dlp binary. Empty = auto-managed newest release in data/bin, refreshed daily"),
])


@tool
async def download_media(url: str, audio_only: bool = False, start: Optional[float] = None,
                         end: Optional[float] = None) -> ToolResult:
    """Download a video, or just its audio, from a link: YouTube, TikTok, Twitter/X, Reddit, Instagram, SoundCloud, Twitch clips and most other video or audio sites. Use when someone asks to download, save, rip or grab a video or song from a link, or when you need the file for a next step (e.g. pass the result's path or posted URL to extract_audio_track to isolate vocals). The file is posted in chat when it's small enough.

    Args:
        url: The page link, e.g. a YouTube watch URL.
        audio_only: True for just the audio (mp3), False for the video.
        start: Optional. Keep only the part starting here, in seconds from the beginning (1:23 = 83).
        end: Optional. Keep only the part ending here, in seconds from the beginning.
    """
    u = (url or "").strip()
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.hostname:
        return ToolResult.fail("That's not an http(s) link.")
    from .fetch_url import _resolve_and_vet_host
    internal, _ = await _resolve_and_vet_host(p.hostname)
    if internal:
        return ToolResult.fail("Refusing to download from an internal/private address.")
    if start is not None and start < 0:
        return ToolResult.fail("start can't be negative.")
    if end is not None and end <= (start or 0):
        return ToolResult.fail("end must be after start.")
    g = lambda k: ts.get("download_media", k)
    if end is not None and end - (start or 0) > g("max_duration_s"):
        return ToolResult.fail(f"That section is longer than the {g('max_duration_s')}s limit.")

    if not shutil.which("ffmpeg"):
        return ToolResult.fail("download_media needs ffmpeg installed on the host (yt-dlp uses it to merge and convert).")
    ytdlp, err = await _ytdlp.binary(g("ytdlp_path"))
    if not ytdlp:
        return ToolResult.fail(err)

    out_dir = Path(resolve_path(get_config().get("output_dir", "./data/output")))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:12]
    cmd = [ytdlp, "--no-update", "--no-playlist", "--no-progress", "--no-simulate", *_ytdlp.js_args(),
           "--max-filesize", f"{g('max_filesize_mb')}M", "-o", str(out_dir / f"{tag}.%(ext)s"),
           "--print", "%(duration)s\t%(title)s", "--print", "after_move:filepath"]
    # Non-owners get site extractors only: the generic extractor follows any
    # URL found on a page, which could be pointed at an internal service.
    from ..acl import current_caller_is_owner
    if not current_caller_is_owner():
        cmd += ["--use-extractors", "default,-generic"]
    if end is None:
        # With --print, a video the filter rejects is skipped silently: exit 0
        # and nothing printed (the prints run after the filter). Caught below.
        cmd += ["--match-filter", f"!is_live & duration<=?{g('max_duration_s')}"]
    if start is not None or end is not None:
        cmd += ["--download-sections", f"*{start or 0}-{end if end is not None else 'inf'}", "--force-keyframes-at-cuts"]
    if audio_only:
        cmd += ["-x", "--audio-format", g("audio_format"), "--audio-quality", str(g("audio_quality"))]
    else:
        # H.264 + AAC plays inline everywhere (Discord, phones); AV1/VP9 often don't.
        cmd += ["-S", f"res:{g('video_max_height')},vcodec:h264,acodec:aac,ext:mp4:m4a", "--merge-output-format", "mp4"]

    print_ts(f"{COLOR_YELLOW}download_media: {u} (audio_only={audio_only}, {start}-{end}){COLOR_END}")
    rc, out, err = await _ytdlp.run(cmd + [u], g("timeout_s"))
    lines = [l for l in out.splitlines() if l.strip()]
    files = [Path(l) for l in lines if l.startswith(str(out_dir)) and Path(l).is_file()]
    if rc != 0 or not files:
        for f in out_dir.glob(f"{tag}*"):
            f.unlink(missing_ok=True)
        text = out + err
        if "does not pass filter" in text or (rc == 0 and not lines and end is None):
            return ToolResult.fail(f"It's a livestream or longer than the {g('max_duration_s')}s limit. Give a start/end to grab part of it.")
        if "larger than max-filesize" in text or "File is larger than" in text:
            return ToolResult.fail(f"It's bigger than the {g('max_filesize_mb')} MB limit. Try audio_only or a start/end section.")
        errs = [l for l in err.splitlines() if "ERROR" in l]
        return ToolResult.fail("yt-dlp failed: " + (errs[-1] if errs else (err.strip()[-300:] or f"exit {rc}, no file")))

    src = files[-1]
    dur, _, title = lines[0].partition("\t") if "\t" in lines[0] else ("?", "", "")
    nice = safe_filename(title or "download").strip("._")[:80] or "download"
    final = out_dir / f"{nice}_{tag[:6]}{src.suffix}"
    os.replace(src, final)
    mb = final.stat().st_size / 1048576
    part = f", section {start or 0}-{end if end is not None else 'end'}s" if (start is not None or end is not None) else ""
    info = f"Downloaded {'audio' if audio_only else 'video'} {title!r} ({mb:.1f} MB{part}; full length {dur}s). Local path: {final}"
    print_ts(f"{COLOR_GREEN}download_media: {final.name} {mb:.1f} MB{COLOR_END}")
    if mb <= g("max_attach_mb"):
        # Delivery status (posted URL, or a POSTING FAILED note) is appended by the executor.
        return ToolResult(attachments=[final], model_feedback=info)
    return ToolResult(model_feedback=info + f". NOT posted in chat: it's over the {g('max_attach_mb')} MB upload limit, so it's only on disk.")
