'''Fetch a Discord message by URL and return its content + attachment URLs.

For agents that get linked to a previous message (e.g. by a user dropping a
Discord message link in chat) and need to see the content / use the attached
image. The bot must have access to the channel; otherwise fetch fails loud.
'''
from __future__ import annotations

import asyncio
import os
import re
import tempfile

from ._base import tool, ToolResult
from ..utils import print_ts, COLOR_YELLOW, COLOR_END


_URL_RE = re.compile(
    r'https?://(?:www[.]|ptb[.]|canary[.])?discord(?:app)?[.]com/channels/'
    r'(@me|[0-9]+)/([0-9]+)/([0-9]+)'
)
# Discord CDN attachment URLs — agents can pass back a URL from a previous
# tool result to re-inject the image into vision in a future turn.
_CDN_URL_RE = re.compile(
    r'https?://(?:cdn|media)[.]discordapp[.](?:com|net)/attachments/[0-9]+/[0-9]+/.+',
    re.IGNORECASE,
)
_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.heic', '.heif', '.bmp')


async def _queue_image_bytes_for_vision(
    data: bytes, filename: str, content_type: str, conv
) -> tuple[bool, str | None]:
    '''Write image bytes to a temp file, validate against Anthropic's
    vision contract, and queue on the conversation on success.

    Returns (True, None) on success, (False, reason) on rejection. The
    reason is a short human-readable string suitable to surface to the
    user. Normalized bytes are written back to the temp file so the
    inject-side never has to re-encode.

    Validation is the BOUNDARY check — all image-content constraints
    live in _image_validator. The inject-side check is a safety net that
    should rarely fire because content is validated here first.
    '''
    if conv is None:
        return False, "no conversation context"
    fd, path = tempfile.mkstemp(prefix='openflip_fdm_', suffix='_' + (filename or 'img'))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
    except Exception as e:
        try:
            os.unlink(path)
        except Exception:
            pass
        return False, f"temp-write failed: {e}"

    from .._image_validator import validate_and_normalize_image
    normalized, mt_or_reason = validate_and_normalize_image(path, content_type)
    if normalized is None:
        try:
            os.unlink(path)
        except Exception:
            pass
        return False, mt_or_reason

    try:
        with open(path, 'wb') as f:
            f.write(normalized)
    except Exception as e:
        try:
            os.unlink(path)
        except Exception:
            pass
        return False, f"normalize write-back failed: {e}"

    pending = list(getattr(conv, '_pending_image_attachments', None) or [])
    pending.append({
        'path': path,
        'content_type': mt_or_reason,
        'filename': (filename or 'image').lower(),
    })
    conv._pending_image_attachments = pending
    return True, None


def _content_type_for(filename: str) -> str:
    fn = (filename or '').lower()
    if fn.endswith(('.jpg', '.jpeg')):
        return 'image/jpeg'
    if fn.endswith('.gif'):
        return 'image/gif'
    if fn.endswith('.webp'):
        return 'image/webp'
    return 'image/png'


@tool
async def fetch_discord_message(url: str) -> ToolResult:
    '''Fetch a Discord message OR a CDN image by URL.

    Two URL forms supported:
    1. Discord message URL (https://discord.com/channels/{guild}/{channel}/{message})
       — fetches the message, returns author/content/attachment URLs, and
       queues any image attachments for vision.
    2. Discord CDN attachment URL (https://cdn.discordapp.com/attachments/...)
       — downloads the image directly and queues it for vision. Use this to
       re-inspect an image you previously produced (or the user attached) in
       a NEW turn — the URL of any prior tool output can be passed back here.

    Either form queues the image for your vision on the next iteration.

    Args:
        url: A Discord message URL or a Discord CDN attachment URL.
    '''
    url_s = (url or '').strip()

    from ..tool_executor import CURRENT_AGENT, CURRENT_CHANNEL_ID
    try:
        agent = CURRENT_AGENT.get()
    except LookupError:
        return ToolResult.fail('Tool invoked outside an agent context.')

    from ..registry import RUNNERS
    runner = RUNNERS.get(agent.id)
    if not runner or not runner.bot:
        return ToolResult.fail(f'No running bot for agent {agent.id!r}.')

    # Path A: CDN attachment URL — download directly and queue for vision.
    if _CDN_URL_RE.match(url_s):
        try:
            # Prefer Session.channel_id_int when a Session is in context (Phase 1+).
            # Fall back to legacy CURRENT_CHANNEL_ID int contextvar.
            from ..tool_executor import CURRENT_SESSION
            session = CURRENT_SESSION.get(None)
            if session is not None and session.transport == "discord":
                ch_id_for_conv = session.channel_id_int
            else:
                ch_id_for_conv = int(CURRENT_CHANNEL_ID.get())
            conv = runner.conversations.get(ch_id_for_conv)
        except (LookupError, KeyError, ValueError):
            conv = None
        base = url_s.split('?', 1)[0]
        fn = (base.rsplit('/', 1)[-1] or 'image').lower()
        if not any(fn.endswith(ext) for ext in _IMG_EXTS):
            return ToolResult.fail(f'CDN URL is not an image (filename: {fn}).')
        import aiohttp
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.get(url_s, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status != 200:
                        return ToolResult.fail(f'CDN download returned HTTP {resp.status}.')
                    data = await resp.read()
        except Exception as e:
            return ToolResult.fail(f'Failed to download CDN image: {e}')
        ct = _content_type_for(fn)
        ok, reason = await _queue_image_bytes_for_vision(data, fn, ct, conv)
        if not ok:
            return ToolResult.fail(f'Downloaded but rejected: {reason}')
        print_ts(
            f'{COLOR_YELLOW}fetch_discord_message: downloaded CDN image {fn} ({len(data)} bytes), queued for vision{COLOR_END}',
            agent=agent.id,
        )
        return ToolResult(
            model_feedback=f'Downloaded CDN image ({fn}, {len(data)} bytes) and queued for vision. You will SEE it on your next iteration.',
        )

    # Path B: Discord message URL — fetch via Discord API. agent/runner
    # already resolved at the top of the function; just need to resolve
    # the channel and message IDs from the URL.
    m = _URL_RE.match(url_s)
    if not m:
        return ToolResult.fail('Not a valid Discord message or CDN URL.')
    _guild, channel_id_s, message_id_s = m.groups()
    channel_id = int(channel_id_s)
    message_id = int(message_id_s)

    channel = runner.bot.get_channel(channel_id)
    if channel is None:
        try:
            channel = await asyncio.wait_for(
                runner.bot.fetch_channel(channel_id),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            return ToolResult.fail(f'fetch_channel({channel_id}) timed out after 15s')
        except Exception as e:
            return ToolResult.fail(f'Channel {channel_id} not reachable: {e}')

    try:
        msg = await asyncio.wait_for(channel.fetch_message(message_id), timeout=15.0)
    except Exception as e:
        return ToolResult.fail(f'Could not fetch message {message_id}: {e}')

    parts = []
    author = (
        getattr(msg.author, 'display_name', None)
        or getattr(msg.author, 'name', None)
        or 'unknown'
    )
    parts.append(f'Author: {author}')
    parts.append(f'Channel: {channel_id}')
    parts.append(f'Message: {message_id}')
    if msg.content:
        parts.append(f'Content:{chr(10)}{msg.content}')
    if msg.attachments:
        urls = chr(10).join(f'[attachment: {a.url}]' for a in msg.attachments)
        parts.append(f'Attachments:{chr(10)}{urls}')
    if not msg.content and not msg.attachments:
        parts.append('(empty message — no text or attachments)')

    # Download any image attachments and queue them on the conversation so the
    # NEXT chat iteration sees them as vision content blocks. Without this the
    # agent only has the URL string — it can't actually inspect the image
    # before passing it to edit_image / animate_image.
    image_exts = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.heic', '.heif', '.bmp')
    queued = 0
    try:
        from ..tool_executor import CURRENT_CHANNEL_ID, CURRENT_SESSION
        session = CURRENT_SESSION.get(None)
        if session is not None and session.transport == "discord":
            ch_id_for_conv = session.channel_id_int
        else:
            ch_id_for_conv = int(CURRENT_CHANNEL_ID.get())
        conv = runner.conversations.get(ch_id_for_conv)
    except (LookupError, KeyError, ValueError):
        conv = None
    if conv is not None:
        for a in msg.attachments:
            ct = (getattr(a, 'content_type', None) or '').lower()
            fn = (getattr(a, 'filename', None) or '').lower()
            is_image = ct.startswith('image/') or any(fn.endswith(ext) for ext in image_exts)
            if not is_image:
                continue
            try:
                data = await a.read()
            except Exception as e:
                print_ts(f'fetch_discord_message: download failed for {fn}: {e}', agent=agent.id, error=True)
                continue
            queue_ct = ct if ct.startswith('image/') else 'image/png'
            ok, reason = await _queue_image_bytes_for_vision(
                data, fn or 'image', queue_ct, conv,
            )
            if ok:
                queued += 1
            else:
                print_ts(
                    f'fetch_discord_message: rejected attachment {fn}: {reason}',
                    agent=agent.id,
                )
        if queued:
            parts.append(
                f'(Downloaded and queued {queued} image(s) for vision — '
                f'you will SEE them on the next iteration.)'
            )
    elif conv is None:
        # conv lookup failed — session wasn't resolved. Non-fatal; vision
        # re-injection won't work but message content still returns.
        pass

    print_ts(
        f'{COLOR_YELLOW}fetch_discord_message: fetched {message_id} from channel {channel_id} (queued {queued} image(s) for vision){COLOR_END}',
        agent=agent.id,
    )
    return ToolResult(model_feedback=(chr(10) + chr(10)).join(parts))
