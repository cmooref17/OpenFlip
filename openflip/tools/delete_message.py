'''Delete a Discord message in a channel.

The bot can always delete its own messages. Deleting OTHER users' messages
requires the bot to have the 'manage messages' permission in that channel —
without it, that path will fail with a Forbidden error.

Security: like send_message, this is gated only by allowed_tools in
agent.json. Restrict via users: [...] if needed.
'''
from __future__ import annotations

import asyncio

from ._base import tool, ToolResult
from ..utils import print_ts, COLOR_YELLOW, COLOR_END


@tool
async def delete_message(
    message_id: int = 0,
    channel_id: int = 0,
    with_attachments: bool = False,
) -> ToolResult:
    '''Delete a Discord message.

    Modes:
      * If message_id is given: delete that specific message (in channel_id,
        or the current-turn channel if channel_id is omitted).
      * If message_id is 0: scan recent channel history and delete the most
        recent message authored by THIS bot. If with_attachments is True,
        only delete the most recent bot message that has at least one
        attachment (useful for undoing a bad image post).

    Args:
        message_id: Optional Discord message ID. 0 means "find and delete my
            most recent message in this channel".
        channel_id: Optional Discord channel ID. Defaults to the channel that
            triggered the current turn.
        with_attachments: When message_id is 0, only delete the most recent
            bot-authored message that has at least one attachment.
    '''
    from ..tool_executor import CURRENT_AGENT, CURRENT_CHANNEL_ID
    try:
        agent = CURRENT_AGENT.get()
    except LookupError:
        return ToolResult.fail('Tool invoked outside an agent context.')

    explicit_channel = int(channel_id) if channel_id else 0
    current_channel_id = 0
    try:
        current_channel_id = int(CURRENT_CHANNEL_ID.get())
    except LookupError:
        current_channel_id = 0
    target_channel_id = explicit_channel or current_channel_id
    if not target_channel_id:
        return ToolResult.fail('No channel_id provided and no current channel in context.')

    # Cross-channel guard (MED-6 from the security audit). Same model as
    # send_message: a non-owner user trying to delete a message in a
    # channel OTHER than where they're talking has to be the owner. Prevents
    # one user with delete_message access from griefing other channels.
    if explicit_channel and explicit_channel != current_channel_id:
        from ..acl import is_owner as _is_owner
        from ..tool_executor import current_caller
        # Transport-aware: Discord → numeric path (unchanged); iMessage →
        # compare the raw handle. Absent session → discord/"" → numeric path.
        speaker_id, _tname, _handle = current_caller()
        if not _is_owner(speaker_id, transport=_tname, handle=_handle):
            return ToolResult.fail(
                f"delete_message: cross-channel deletion (target {explicit_channel} ≠ "
                f"current {current_channel_id}) is restricted to the owner."
            )

    from ..registry import RUNNERS
    runner = RUNNERS.get(agent.id)
    if not runner or not runner.bot:
        return ToolResult.fail(f'No running bot for agent {agent.id!r}.')

    channel = runner.bot.get_channel(target_channel_id)
    if channel is None:
        try:
            channel = await asyncio.wait_for(
                runner.bot.fetch_channel(target_channel_id),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            return ToolResult.fail(f'fetch_channel({target_channel_id}) timed out after 15s')
        except Exception as e:
            return ToolResult.fail(f'Channel {target_channel_id} not reachable: {e}')

    # Path A: specific message by ID.
    if message_id:
        try:
            msg = await asyncio.wait_for(channel.fetch_message(int(message_id)), timeout=15.0)
        except Exception as e:
            return ToolResult.fail(f'Could not fetch message {message_id}: {e}')
        try:
            await asyncio.wait_for(msg.delete(), timeout=15.0)
        except asyncio.TimeoutError:
            return ToolResult.fail(f'Delete timed out after 15s')
        except Exception as e:
            return ToolResult.fail(f'Delete failed: {e}')
        print_ts(
            f'{COLOR_YELLOW}delete_message: removed message {message_id} in channel {target_channel_id}{COLOR_END}',
            agent=agent.id,
        )
        return ToolResult(
            model_feedback=f'Deleted message {message_id} from channel {target_channel_id}.',
        )

    # Path B: bot's most recent message in this channel (optionally only ones with attachments).
    bot_user = runner.bot.user
    if not bot_user:
        return ToolResult.fail('Bot user not yet ready.')
    bot_user_id = bot_user.id

    scan_limit = 50
    try:
        async for msg in channel.history(limit=scan_limit):
            if msg.author.id != bot_user_id:
                continue
            if with_attachments and not msg.attachments:
                continue
            try:
                await asyncio.wait_for(msg.delete(), timeout=15.0)
            except asyncio.TimeoutError:
                return ToolResult.fail(f'Delete timed out after 15s')
            except Exception as e:
                return ToolResult.fail(f'Delete failed: {e}')
            print_ts(
                f'{COLOR_YELLOW}delete_message: removed last bot message {msg.id} in channel {target_channel_id}{COLOR_END}',
                agent=agent.id,
            )
            descriptor = 'message with attachment' if with_attachments else 'message'
            return ToolResult(
                model_feedback=f'Deleted my last {descriptor} ({msg.id}) from channel {target_channel_id}.',
            )
    except Exception as e:
        return ToolResult.fail(f'Failed scanning channel history: {e}')

    detail = ' with attachments' if with_attachments else ''
    return ToolResult.fail(
        f'No bot-authored message{detail} found in last {scan_limit} messages of channel {target_channel_id}.',
    )
