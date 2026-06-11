"""Per-message processing. Builds the per-message tool list (gated by ACL),
calls the model, and hands tool calls off to the executor.

# System message assembly — for anyone editing prompt construction
#
# The string the model sees as its system prompt is built from layered
# sources. Knowing the order matters because Anthropic's prompt cache
# keys on the exact byte content; anything that varies turn-to-turn
# busts the cached prefix.
#
#  ┌─ agent.system_message (assembled at agent load) ────────────────┐
#  │   1. SOUL.md                  (per-agent character — stable)    │
#  │   2. AGENT.md                 (per-agent supplement, optional)  │
#  │   3. _shared/FRAMEWORK.md     (universal rules, with template   │
#  │                                substitution for {agent_id} etc) │
#  │   4. _shared/TOOLS.md         (universal tool hygiene)          │
#  └─────────────────────────────────────────────────────────────────┘
#
#  ┌─ system_extension (built per-turn by build_visible_tools) ──────┐
#  │   5. tool_rules.for_extension (when any tools — stable)         │
#  └─────────────────────────────────────────────────────────────────┘
#
# Note: memory instructions are NOT injected per-turn — they live in the
# shared FRAMEWORK.md / TOOLS.md and load once with agent.system_message.
#
# Per-speaker ACL state (blocked-tools list, settings filtered to
# callable tools) is NOT in the system prompt. It rides on the
# user-message preamble built by build_visible_tools — that way one
# speaker rotation doesn't bust the cached system prefix.
#
# Hot reload is explicit (/reload slash command). The runtime no longer
# stat()s files on every inbound message — see runtime.py for why.
"""
from __future__ import annotations
from typing import Optional

import nextcord

from .agent import Agent
from .conversation import DiscordConversation
from .acl import evaluate_tools_for_speaker, is_owner
from .tools import TOOL_REGISTRY
from .session import InboundMessage, Session, Attachment, make_discord_session
from . import tool_settings
from . import tool_rules
from .utils import print_ts, COLOR_YELLOW, COLOR_END

# ── Time-stamp injection ──────────────────────────────────────────────
#
# Each user message that arrives at the FIRST message of a new 30-minute
# wallclock bucket (per conversation) gets a `[YYYY-MM-DD HH:MM Day]`
# prefix inserted at the top of build_user_prompt. Subsequent messages
# in the same bucket get no prefix.
#
# Buckets: HH:00-HH:30 and HH:30-(HH+1):00, in TIMEZONE below.
# State is process-memory only — keyed by Session.conversation_id —
# so a restart re-stamps the first message of the next bucket, which is
# actually useful (a restart IS a context-discontinuity worth flagging).
#
# Lives here instead of runtime.py because build_user_prompt is the
# transport-neutral chokepoint every inbound message goes through.
import datetime as _dt
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _STAMP_TZ = _ZoneInfo("US/Mountain")
except Exception:
    _STAMP_TZ = None  # fall back to system local time if zoneinfo unavailable

# {conversation_id: last_stamped_bucket_key}
_stamp_state: dict[str, str] = {}


def _current_bucket_key(now: _dt.datetime) -> str:
    """Return a stable string identifying the 30-min bucket `now` falls in."""
    half = 0 if now.minute < 30 else 30
    return f"{now.year:04d}-{now.month:02d}-{now.day:02d}T{now.hour:02d}:{half:02d}"


def _maybe_stamp(conversation_id: str) -> str | None:
    """If this conversation hasn't been stamped yet in the current bucket,
    mark the bucket as stamped and return a formatted prefix string.
    Otherwise return None.
    """
    now = _dt.datetime.now(_STAMP_TZ) if _STAMP_TZ else _dt.datetime.now()
    bucket = _current_bucket_key(now)
    if _stamp_state.get(conversation_id) == bucket:
        return None
    _stamp_state[conversation_id] = bucket
    # e.g. "[2026-05-25 11:14 Monday]"
    return f"[{now.strftime('%Y-%m-%d %H:%M %A')}]"


def build_inbound_from_discord(message: nextcord.Message, bot_user_id: int, owner_id: int) -> InboundMessage:
    """Wrap a nextcord.Message in an InboundMessage at the on_message boundary.

    Phase 1 of the discord-decouple. The rest of openflip can read from
    InboundMessage / Session and not depend on nextcord types directly.

    Note: image attachments for vision are still extracted separately via
    `extract_image_attachments(message)` because they need the live
    nextcord.Attachment object to call `.read()`. Phase 2 will fold that
    into the Transport adapter.
    """
    is_dm = isinstance(message.channel, nextcord.DMChannel)
    role_ids = [r.id for r in getattr(message.author, "roles", [])] if not is_dm else []
    speaker_id = message.author.id
    display_name = (
        getattr(message.author, "display_name", None)
        or getattr(message.author, "name", None)
        or "User"
    )
    guild = getattr(message, "guild", None)
    guild_id = int(getattr(guild, "id", 0) or 0)
    # Channel category (None for DMs / uncategorized channels) → 0.
    category_id = int(getattr(message.channel, "category_id", 0) or 0)
    session = make_discord_session(
        channel_id=message.channel.id,
        speaker_id=speaker_id,
        speaker_role_ids=role_ids,
        is_owner=(speaker_id == owner_id),
        is_dm=is_dm,
        display_name=display_name,
        guild_id=guild_id,
        category_id=category_id,
    )
    attachments = [
        Attachment(
            content_type=(getattr(a, "content_type", None) or "application/octet-stream"),
            filename=(getattr(a, "filename", None) or "file"),
            url=a.url,
        )
        for a in message.attachments
    ]
    # Reply support: pull attachments from the resolved replied-to message too.
    # Don't recurse into a full InboundMessage for the parent — Phase 1 only
    # needs the surface that build_user_prompt currently uses.
    ref = getattr(message, "reference", None)
    if ref is not None:
        resolved = getattr(ref, "resolved", None)
        if resolved is not None and hasattr(resolved, "attachments"):
            for a in resolved.attachments:
                attachments.append(Attachment(
                    content_type=(getattr(a, "content_type", None) or "application/octet-stream"),
                    filename=(getattr(a, "filename", None) or "file"),
                    url=a.url,
                ))
    mentions_us = any(u.id == bot_user_id for u in message.mentions)
    return InboundMessage(
        session=session,
        text=message.content or "",
        sender_id=speaker_id,
        sender_display_name=display_name,
        sender_is_bot=bool(getattr(message.author, "bot", False)),
        is_dm=is_dm,
        mentions_us=mentions_us,
        attachments=attachments,
        reply_to=None,
    )


def build_user_prompt(inbound: InboundMessage) -> str:
    """Compose the user-message content the model will see.

    Prepends '<DisplayName>: ' to the raw content so the model has speaker
    attribution. Character-driven agents typically reference speakers by name
    in their system messages (e.g. rules about specific people), so without
    speaker context those rules never trigger.

    Display name is run through config['display_name_map'] before formatting,
    so a platform handle (like a Discord username) can be remapped to the
    name the agent's system message expects.

    Attachment URLs go on their own line for tools that accept image/audio
    URLs via the [attachment: …] convention.

    Takes an InboundMessage (transport-neutral). Discord-side wrapping happens
    at the on_message boundary in transports/discord.py.
    """
    from .config_global import get_config
    name_map = get_config().get("display_name_map") or {}

    text = inbound.text
    raw_speaker = inbound.sender_display_name or "User"
    speaker = name_map.get(raw_speaker, raw_speaker)
    formatted = f"{speaker}: {text}" if text else f"{speaker}:"
    attachment_urls = [a.url for a in inbound.attachments if a.url]
    if attachment_urls:
        urls = "\n".join(f"[attachment: {u}]" for u in attachment_urls)
        formatted = f"{formatted}\n{urls}".strip()

    # First message in a fresh 30-min bucket for this conversation gets
    # a timestamp prefix so the model knows what time it is. See the
    # _maybe_stamp helper above for the bucket logic.
    conv_id = getattr(getattr(inbound, "session", None), "conversation_id", None)
    if conv_id:
        stamp = _maybe_stamp(conv_id)
        if stamp:
            formatted = f"{stamp}\n{formatted}"

    return formatted


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".heif", ".bmp")


def extract_image_attachments(message: nextcord.Message) -> list[dict]:
    """Return image attachments from a Discord message as a list of dicts.

    Each dict has: {url, content_type, filename}. Filters to images only based on
    content_type starting with 'image/' or filename extension. Non-image attachments
    (audio, video, generic files) are NOT returned here — those still appear as
    [attachment: URL] text in build_user_prompt for tools that consume URLs.

    Used by runtime to download image bytes and feed them to the model as vision
    content blocks, so the model actually sees the picture instead of just a URL.
    """
    out: list[dict] = []
    sources = list(message.attachments)
    # Reply support: include attachments from the replied-to message when
    # cached. Uncached references are skipped (function is sync; we don't
    # block on an async fetch). This lets an agent pinged via reply see the
    # image on the original message without the user re-attaching it.
    ref = getattr(message, "reference", None)
    if ref is not None:
        resolved = getattr(ref, "resolved", None)
        if resolved is not None and hasattr(resolved, "attachments"):
            sources.extend(resolved.attachments)
    for a in sources:
        ct = (getattr(a, "content_type", None) or "").lower()
        fn = (getattr(a, "filename", None) or "").lower()
        is_image = ct.startswith("image/") or any(fn.endswith(ext) for ext in _IMAGE_EXTS)
        if is_image:
            # Default media_type if Discord didn't provide content_type. Most CDN
            # uploads do provide it; this is just a safe fallback for the JSON block.
            media_type = ct if ct.startswith("image/") else "image/png"
            out.append({
                "attachment": a,  # nextcord.Attachment — runtime calls .read() to download
                "url": a.url,
                "content_type": media_type,
                "filename": fn or "image",
            })
    return out


def should_respond(agent: Agent, inbound: InboundMessage, bot_user_id: int) -> bool:
    """Decide whether the agent should reply to this inbound.

    Transport-neutral. Takes an InboundMessage; the caller (transport
    adapter) is responsible for wrapping native messages before this call.
    """
    # ── (a) bot-self / bot-author checks — stay at the top ──────────────
    if inbound.sender_id == bot_user_id:
        return False
    if inbound.sender_is_bot and not agent.respond_to_bots:
        return False

    # ── (b) Compute the three routing ids from the inbound (any may be 0) ─
    guild_id = int(getattr(inbound.session, "guild_id", 0) or 0)
    category_id = int(getattr(inbound.session, "category_id", 0) or 0)
    try:
        channel_id = int(inbound.session.transport_id)
    except (ValueError, TypeError):
        channel_id = -1  # non-numeric transports can't match Discord-int rules anyway

    # Guild whitelist gate (legacy, predates the nine-list system). Empty
    # whitelist = all guilds allowed. DMs have no guild and are never filtered
    # here. Kept as a hard pre-gate so existing agents that rely on it don't
    # break; it composes with the nine-list tiers below.
    if agent.guild_whitelist and not inbound.is_dm and guild_id:
        if guild_id not in agent.guild_whitelist:
            return False

    # Did the agent configure ANY of the nine new routing lists? When none are
    # set we fall through to the legacy respond_in path (tier g) so current
    # agents behave byte-identically.
    any_new_routing = bool(
        agent.respond_guilds or agent.respond_channels or agent.respond_categories
        or agent.respond_no_mention_guilds or agent.respond_no_mention_channels
        or agent.respond_no_mention_categories
        or agent.ignore_guilds or agent.ignore_channels or agent.ignore_categories
    )

    # ── (c) IGNORE tier — hard deny, wins over everything ───────────────
    # New per-dimension ignore lists PLUS the legacy ignore_channel_ids, so
    # old config keeps working alongside new. (Legacy ignore_channel_ids was
    # historically checked here, before the DM path — preserved.)
    if (guild_id and guild_id in agent.ignore_guilds) \
            or (channel_id in agent.ignore_channels) \
            or (category_id and category_id in agent.ignore_categories) \
            or (channel_id in agent.ignore_channel_ids):
        return False

    # ── (d) NO-MENTION tier — respond regardless of mention ─────────────
    # New per-dimension no-mention lists PLUS the legacy always_respond_channel_ids
    # (also historically checked before the DM path — preserved).
    if (guild_id and guild_id in agent.respond_no_mention_guilds) \
            or (channel_id in agent.respond_no_mention_channels) \
            or (category_id and category_id in agent.respond_no_mention_categories) \
            or (channel_id in agent.always_respond_channel_ids):
        return True

    # ── (e) RESPOND tier — respond only if mentioned ────────────────────
    if (guild_id and guild_id in agent.respond_guilds) \
            or (channel_id in agent.respond_channels) \
            or (category_id and category_id in agent.respond_categories):
        return inbound.mentions_us

    # ── (f) DM path — unchanged. DMs have no guild/category, so tiers c-e
    # don't match them; they fall through to here. ──────────────────────
    if inbound.is_dm:
        # DM allowlist gate. If populated, restrict DMs to listed user IDs
        # (plus the bot owner — implicitly always allowed regardless of
        # configuration, so the operator can never lock themselves out of
        # their own agent). Empty allowlist = legacy "anyone can DM"
        # behavior, preserved for backward compatibility.
        if agent.dm_allowlist_user_ids:
            from .config_global import get_owner_id
            try:
                owner_id = int(get_owner_id("discord") or 0)
            except (ValueError, TypeError):
                owner_id = 0
            if inbound.sender_id != owner_id and inbound.sender_id not in agent.dm_allowlist_user_ids:
                return False
        return True

    # ── (g) BACKWARD-COMPAT FALLBACK — no new routing lists configured ──
    # Fall through to the legacy respond_in mode. Reached only for non-DM
    # messages (DMs returned in tier f).
    if not any_new_routing:
        if agent.respond_in == "all":
            return True
        if agent.respond_in == "mentions_only":
            return inbound.mentions_us
        if agent.respond_in == "channels_only":
            return False
        return inbound.mentions_us

    # ── (h) New-style config is an explicit allowlist: nothing matched ──
    return False


MEMORY_TOOL_NAMES = {"save_memory", "update_core_memory", "search_memory", "read_memory", "list_memory_files"}


def build_visible_tools(agent: Agent, *, speaker_id, speaker_role_ids, channel_id, owner: bool = False, transport: str = "discord", chain_terminator_mode: bool = False, chain_root_operator: bool = True, handle: str = "", tool_grants: list[str] | None = None):
    """Return (callable_tool_funcs, system_extension_text, user_preamble_text).

    The split is for prompt caching:
      * system_extension is stable across speakers/channels for one agent —
        memory instructions and tool-usage hygiene only. Goes into the
        system prompt and benefits from prefix caching.
      * user_preamble carries per-speaker state (which tools the speaker
        can't use, current tool settings filtered to their callable set).
        Lives on the user message instead of the system prompt so a
        speaker rotation doesn't bust the cached system prefix.

    `transport` selects which `auth.<transport>` block in each ToolACL is
    consulted. `speaker_id` is transport-native: int for discord, string
    handle for imessage. `handle` is the raw sender handle for handle-based
    transports (iMessage email/phone) — it backs the admin bypass for those
    transports (see `_check_acl`); "" for Discord. `owner` is kept for callers
    that pass it but is no longer used inside ACL evaluation — see
    `_check_acl` in acl.py.

    `tool_grants` is a per-session additive allow-path (cron/synthetic sessions
    with no human speaker). It is forwarded to `evaluate_tools_for_speaker` and
    only widens callability — it never weakens per-user ACL. See Session.tool_grants.
    """
    visibility = evaluate_tools_for_speaker(
        agent,
        transport=transport,
        speaker_id=speaker_id,
        speaker_role_ids=speaker_role_ids,
        channel_id=channel_id,
        handle=handle,
        tool_grants=tool_grants,
    )
    callable_funcs = []
    known_but_blocked = []
    seen_names: set[str] = set()
    for v in visibility:
        tool = TOOL_REGISTRY.get(v.name)
        if not tool:
            continue
        seen_names.add(v.name)
        if v.callable:
            callable_funcs.append(tool.func)
        elif v.known:
            known_but_blocked.append(tool)
    # Always inject memory tools when memory_enabled, even if not in allowed_tools.
    if agent.memory_enabled:
        for mname in MEMORY_TOOL_NAMES:
            if mname not in seen_names:
                mtool = TOOL_REGISTRY.get(mname)
                if mtool:
                    callable_funcs.append(mtool.func)

    # === System extension: agent-stable content only ===
    # Memory documentation lives in _shared/FRAMEWORK.md and _shared/TOOLS.md
    # only. The previous extension block here duplicated both, adding ~30
    # lines of redundant content to every turn (audit 2026-05-12).
    extension_parts: list[str] = []
    # Tool-usage hygiene — agent-level rule, applies whenever any tools exist.
    # Unconditional so the system prompt stays byte-stable regardless of which
    # speaker is currently messaging.
    if agent.allowed_tools or agent.memory_enabled:
        extension_parts.append(tool_rules.for_extension())
    extension = "\n".join(extension_parts)

    # === User preamble: per-speaker content ===
    preamble_parts: list[str] = []
    if known_but_blocked:
        lines = ["[Speaker-specific access notes for this turn — decline politely if asked to use these tools:]"]
        for t in known_but_blocked:
            lines.append(f"- {t.name}: {t.description}")
        preamble_parts.append("\n".join(lines))
    # Tool-settings summary, filtered to what the current speaker can actually
    # use. Lives on the user message so it doesn't change the system prompt.
    callable_names = {f.__name__ for f in callable_funcs}
    settings_summary = tool_settings.render_summary_for_ai(only=callable_names)
    if settings_summary:
        preamble_parts.append(settings_summary)
    preamble = "\n\n".join(preamble_parts)

    # Chain-terminator turns keep the FULL toolset (the 2026-05-19 fix for
    # the silent-failure bug class — the old narrowed-3-tool / forced-choice
    # design pressured the model into empty replies), but since the 2026-06
    # leak fix they run SILENT: plain text is saved to history only, never
    # auto-posted. The extension below is what steers delivery — when a
    # human is awaiting the chain's outcome, the model must explicitly
    # send_message; inter-agent chatter otherwise never reaches a human
    # channel. `chain_root_operator` distinguishes the two cases.
    if chain_terminator_mode:
        if chain_root_operator:
            extension = (extension + "\n\n" if extension else "") + (
                "[returning from peer] The previous message is an auto-routed "
                "reply from a peer agent you talked to. This chain was started "
                "by a human who is likely waiting in this conversation, and "
                "NOTHING you say here auto-posts: if the reply answers what "
                "they asked, deliver it NOW with send_message (no channel_id "
                "needed — it posts to this conversation). Call talk_to_agent "
                "to continue with the peer first if needed. Plain text is "
                "saved to the log only and the human will never see it."
            )
        else:
            extension = (extension + "\n\n" if extension else "") + (
                "[returning from peer] The previous message is an auto-routed "
                "reply from a peer agent you talked to. This chain is "
                "agent-initiated (cron/heartbeat/kairos/another agent) — it "
                "stays off all human channels. Call talk_to_agent to continue "
                "the exchange, use send_message ONLY if a human genuinely "
                "needs to see something, or end with plain text to close the "
                "chain (saved to the log, posted nowhere)."
            )

    return callable_funcs, extension, preamble
