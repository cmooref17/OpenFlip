"""Slash commands. User-facing + owner-only. All registered per-agent so each agent's bot
exposes the same surface; owner-only commands check user id at runtime."""
from __future__ import annotations
import asyncio
import io
import os
import shutil
import time
from typing import Optional

import nextcord

from .acl import is_admin, is_owner
from .agent import Agent, ToolACL
from .tools import TOOL_REGISTRY
from . import tool_settings
from .utils import print_ts, project_root, save_json, COLOR_YELLOW, COLOR_END


def _project_root_path() -> str:
    return project_root()


def _conv_key_for_interaction(runner, interaction) -> int | str:
    """Conversation-dict key for the channel a slash command fired in.

    Mirrors the runtime's conversation keying: an identity-linked DM resolves
    to the shared "linked:<canonical>" key — via the runner's alias map, or
    the link map directly when the alias isn't registered yet this process
    (in a DM the invoking user IS the conversation peer, so the lookup is
    deterministic). Everything else returns the native channel id, unchanged.
    Routing and auth never use this key — it only selects conversation state.
    """
    ch_id = int(getattr(interaction.channel, "id", 0) or 0)
    key = runner.conv_key(ch_id)
    if key == ch_id and isinstance(interaction.channel, nextcord.DMChannel):
        from .config_global import resolve_linked_conversation_id
        linked = resolve_linked_conversation_id("discord", int(interaction.user.id))
        if linked:
            if ch_id:
                runner._linked_channel_keys[ch_id] = linked
            key = linked
    return key


def register_commands(bot: nextcord.ext.commands.Bot, runner):
    # registry holds the shared agent/runner state; main holds the lifecycle
    # functions. Importing both via `from . import …` inside the function
    # avoids any chance of stale closure state.
    from . import main as flipmain
    from . import registry

    # ---------------- USER COMMANDS ----------------

    @bot.slash_command(name="reset", description="Reset the conversation in this channel.")
    async def reset_cmd(interaction: nextcord.Interaction):
        import os
        ch_id = interaction.channel.id
        conv_key = _conv_key_for_interaction(runner, interaction)
        # Stop + wipe must be race-safe: everything from here to the file
        # delete runs synchronously (no await), so no in-flight or queued turn
        # can resume in the gap. Ordering is deliberate:
        #
        # 1. Bump the epoch FIRST. A turn already mid-flight on this channel
        #    holds its own reference to the conversation object and re-saves it
        #    at end-of-turn (including from its CancelledError + finally cleanup
        #    paths) — without this the delete below would be silently undone
        #    (history resurrected) by that re-save. Bumping the epoch makes the
        #    in-flight turn's _save_conv() a no-op (see bump_conv_epoch /
        #    _run_turn._turn_epoch). It must precede the interrupt so the save
        #    that may fire during the cancelled turn's teardown already no-ops.
        runner.bump_conv_epoch(conv_key)
        # 2. Hard-interrupt the actively-generating turn — the SAME machinery
        #    /stop uses (runner._hard_interrupt), so its reply + save are
        #    abandoned instead of orphaned. Cancellation is async teardown, but
        #    .cancel() itself is synchronous and the epoch guard (step 1) already
        #    suppresses the teardown save.
        runner._hard_interrupt(conv_key)
        # 3. Drop any QUEUED-but-not-yet-dispatched turns for this conversation
        #    (synthetic/cron turns, or a same-channel message that raced the
        #    soft-inject guard) so none replays against the just-wiped history.
        #    Scoped to this conversation key only — other channels untouched.
        runner._drop_queued_turns(conv_key)
        conv = runner.conversations.pop(conv_key, None)
        if conv and hasattr(conv, "clear_history"):
            conv.clear_history()
        else:
            # Conversation isn't loaded in memory yet (e.g. fresh restart
            # before any message in this channel). clear_history() would
            # never fire, leaving the on-disk JSONL untouched — next message
            # would reload all the old turns. Delete the canonical files
            # directly so the operator actually gets a fresh history.
            agent_dir = os.path.dirname(runner.agent.path)
            # Linked DMs store history under the canonical linked id, not
            # discord:<channel> — delete the file the conversation actually uses.
            conv_id = conv_key if isinstance(conv_key, str) else f"discord:{ch_id}"
            # conversation_path / fs_encode handle the Windows filename
            # encoding (":" → "%3A") — never join a raw conv_id into a name.
            from . import _conversation_io as _cio_reset
            jsonl_path = _cio_reset.conversation_path(agent_dir, conv_id)
            # Pre-reset backup so /reset is recoverable. Only the .jsonl
            # is backed up; the .meta.json sidecar is just compaction
            # bookkeeping and gets regenerated cleanly.
            if os.path.exists(jsonl_path):
                try:
                    import shutil, time as _time
                    backup = f"{jsonl_path}.pre_reset_{int(_time.time())}.bak.jsonl"
                    shutil.copy2(jsonl_path, backup)
                    print_ts(
                        f"/reset: backed up conversation to {backup}",
                        agent=runner.agent.id,
                    )
                except Exception as _bk_err:
                    print_ts(
                        f"{COLOR_YELLOW}/reset: pre-reset backup failed for {jsonl_path}: {_bk_err}{COLOR_END}",
                        agent=runner.agent.id,
                    )
                # Retention sweep: keep last 5 pre_reset backups per channel.
                try:
                    import glob as _glob
                    _all_bak = sorted(_glob.glob(f"{jsonl_path}.pre_reset_*.bak.jsonl"))
                    if len(_all_bak) > 5:
                        for _stale in _all_bak[:-5]:
                            try:
                                os.remove(_stale)
                            except OSError:
                                pass
                except Exception as _retain_e:
                    print_ts(
                        f"{COLOR_YELLOW}/reset: backup retention sweep failed: {_retain_e}{COLOR_END}",
                        agent=runner.agent.id,
                    )
            for ext in (".jsonl", ".meta.json"):
                target = os.path.join(agent_dir, "conversations", _cio_reset.fs_encode(conv_id) + ext)
                try:
                    if os.path.exists(target):
                        os.remove(target)
                except Exception as _rm_err:
                    print_ts(
                        f"{COLOR_YELLOW}/reset: failed to delete {target}: {_rm_err}{COLOR_END}",
                        agent=runner.agent.id,
                    )
        # Wipe any pending soft-inject buffer for this channel — the operator
        # is starting over, queued mid-turn messages from the prior session
        # would be noise in the fresh history.
        runner._pending_inject.pop(conv_key, None)
        await interaction.response.send_message("Conversation reset.", ephemeral=True)

    @bot.slash_command(name="compact", description="Force compaction on the next message in this channel.")
    async def compact_cmd(interaction: nextcord.Interaction):
        # Owner-only: /compact forces a server-side Anthropic compaction
        # (irreversible) and fires a synthetic turn — must not be triggerable
        # by non-owners. Mirrors /uncompact, /dream, /stop gating.
        if not await _owner_check(interaction): return
        # Anthropic-only: sets two flags on the conversation. force_compact_next
        # opts the next chat() into server-side compaction; force_compact_trigger_
        # override makes that request send the low _MANUAL_COMPACT_TRIGGER (50k,
        # Anthropic's floor) instead of the real per-model trigger — so Anthropic
        # compacts regardless of current input size, as long as the conversation
        # is at least ~50k tokens (below that floor Anthropic cannot compact at
        # all). Ollama has no equivalent — surface that to the user.
        conv = runner.conversations.get(_conv_key_for_interaction(runner, interaction))
        if conv is None or not hasattr(conv, "force_compact_next"):
            await interaction.response.send_message(
                "`/compact` is Anthropic-only and there's no active conversation in this channel.",
                ephemeral=True,
            )
            return
        conv.force_compact_next = True
        conv.force_compact_trigger_override = True
        await interaction.response.send_message(
            "⚙️ Compacting now…",
            ephemeral=True,
        )
        # Compaction is a server-side Anthropic operation that only runs
        # during a chat request — there's no "compact this conversation now"
        # endpoint. So fire a tiny synthetic turn immediately to drive the
        # round-trip. The agent sees the prompt below; anthropic compacts
        # because force_compact_next is set on the conv.
        ch_id = int(getattr(interaction.channel, "id", 0) or 0)
        try:
            await runner.run_synthetic_turn(
                ch_id,
                "[FRAMEWORK]: Compaction requested via /compact. Acknowledge briefly so the round-trip fires.",
                speaker_id=int(interaction.user.id),
                originator_visibility="operator_channel",
            )
        except Exception as e:
            print_ts(f"/compact: failed to enqueue synthetic turn: {e}", error=True)

    @bot.slash_command(name="effort", description="Set the reasoning-effort override for THIS conversation (owner-only, Anthropic-only).")
    async def effort_cmd(
        interaction: nextcord.Interaction,
        level: str = nextcord.SlashOption(
            choices=["default", "low", "medium", "high", "xhigh", "max"],
            description="Effort level; 'default' clears the override and falls back to the model config."),
    ):
        # Owner-only: /effort changes the Anthropic request body (output_config.effort)
        # for this conversation, which affects reasoning depth + billing. Mirror
        # /compact's gating.
        if not await _owner_check(interaction): return
        # Anthropic-only: only AnthropicConversation carries `effort_override`.
        # Ollama/DiscordConversation have no such attr — surface that, same
        # shape as /compact's hasattr(force_compact_next) check.
        conv = runner.conversations.get(_conv_key_for_interaction(runner, interaction))
        if conv is None or not hasattr(conv, "effort_override"):
            await interaction.response.send_message(
                "`/effort` is Anthropic-only and there's no active conversation in this channel.",
                ephemeral=True,
            )
            return
        if level == "default":
            conv.effort_override = None
            conv._save_meta()
            await interaction.response.send_message(
                "⚙️ Effort override cleared for THIS conversation — falling back to the model default.",
                ephemeral=True,
            )
            return
        # SlashOption choices are enforced by Discord, so `level` is one of the
        # five valid levels here; _effort_level re-validates defensively anyway.
        conv.effort_override = level
        conv._save_meta()
        await interaction.response.send_message(
            f"⚙️ Effort for THIS conversation set to `{level}` (overrides the model default). "
            f"Use `/effort default` to clear.",
            ephemeral=True,
        )

    @bot.slash_command(name="uncompact", description="Undo the last compaction in this channel (restore full history).")
    async def uncompact_cmd(interaction: nextcord.Interaction):
        # Recover from an unwanted compaction:
        #   1. Find the most recent `.compaction_<ts>.bak.jsonl` backup.
        #   2. Copy it over the live `.jsonl` (restoring pre-compaction history).
        #   3. Delete the meta.json's compaction_block (rewriting the file
        #      without that field; preserves last_usage).
        #   4. Pop the in-memory conv so the next message rebuilds from
        #      the now-restored .jsonl.
        #
        # Anthropic-only; ollama has no compaction.
        import os, glob, json, shutil
        if not is_owner(interaction.user.id):
            await interaction.response.send_message("Owner only.", ephemeral=True)
            return
        # Anthropic-only: we don't gate on an in-memory conversation having
        # the `_compaction_block` attribute because the conv may not be
        # loaded yet on this restart. Instead we check provider on the agent
        # and look for backups on disk directly.
        if runner.agent.provider != "anthropic":
            await interaction.response.send_message(
                "`/uncompact` is Anthropic-only.",
                ephemeral=True,
            )
            return
        agent_dir = os.path.join(os.path.dirname(runner.agent.path), "conversations")
        # Linked DMs store history under the canonical linked id.
        _uc_key = _conv_key_for_interaction(runner, interaction)
        conv_id = _uc_key if isinstance(_uc_key, str) else f"discord:{interaction.channel.id}"
        from . import _conversation_io as _cio_uc
        _fs_conv = _cio_uc.fs_encode(conv_id)  # Windows-safe filename stem
        live_path = os.path.join(agent_dir, f"{_fs_conv}.jsonl")
        meta_path = os.path.join(agent_dir, f"{_fs_conv}.meta.json")
        backups = sorted(glob.glob(os.path.join(agent_dir, f"{_fs_conv}.jsonl.compaction_*.bak.jsonl")))
        if not backups:
            await interaction.response.send_message(
                "No compaction backup found for this channel — nothing to undo. "
                "(Compaction backups are written as `.jsonl.compaction_<ts>.bak.jsonl` "
                "at the moment compaction fires.)",
                ephemeral=True,
            )
            return
        if not os.path.isfile(live_path):
            await interaction.response.send_message(
                f"Found backup(s) but no live conversation file at `{live_path}`. "
                f"Nothing to overwrite. Has this channel ever had a conversation?",
                ephemeral=True,
            )
            return
        # Most recent backup is last in sorted order (timestamps are unix-second ints).
        latest_backup = backups[-1]
        # Defense: keep an extra safety copy of the current live .jsonl before
        # overwriting it. If the user changes their mind, the live state from
        # right before /uncompact is preserved.
        safety_copy = live_path + f".pre_uncompact_{int(__import__('time').time())}.bak.jsonl"
        # Race fix: pop the in-memory conv BEFORE the copy too, so an in-flight
        # message handler that's about to call get_conversation() will rebuild
        # from disk AFTER the restore. Without this, a message landing during
        # the copy could load the pre-restore .jsonl into memory, and the
        # post-restore pop below would only catch loads that completed AFTER
        # the restore — leaving the rare mid-copy window unhandled.
        runner.conversations.pop(_uc_key, None)
        try:
            shutil.copy2(live_path, safety_copy)
            shutil.copy2(latest_backup, live_path)
        except Exception as e:
            await interaction.response.send_message(
                f"Failed to restore from backup: {e}", ephemeral=True,
            )
            return
        # Retention sweep on pre_uncompact safety copies — keep last N.
        try:
            pre_keep = 5
            all_pre = sorted(glob.glob(live_path + ".pre_uncompact_*.bak.jsonl"))
            if len(all_pre) > pre_keep:
                for stale in all_pre[:-pre_keep]:
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
        except Exception:
            pass
        # Scrub the compaction_block from meta.json. Preserve last_usage if present.
        try:
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta.pop("compaction_block", None)
                save_json(meta_path, meta)
        except Exception as e:
            # Non-fatal — meta scrub failing just means the next conv load
            # will restore the compaction block. We can warn but not abort.
            await interaction.response.send_message(
                f"⚠️ Restored .jsonl but failed to scrub meta.json: {e}. "
                f"You may need to manually delete {meta_path} and restart.",
                ephemeral=True,
            )
            return
        # Pop the in-memory conv so the next message reloads from the restored .jsonl.
        runner.conversations.pop(_uc_key, None)
        backup_basename = os.path.basename(latest_backup)
        live_size_kb = os.path.getsize(live_path) // 1024
        await interaction.response.send_message(
            f"✅ Uncompacted. Restored from `{backup_basename}` "
            f"(live .jsonl now {live_size_kb} KB). "
            f"Compaction block cleared. Your next message will reload the full history.",
            ephemeral=True,
        )

    @bot.slash_command(name="help", description="What can this bot do?")
    async def help_cmd(interaction: nextcord.Interaction):
        agent = runner.agent
        # Build per-user visible tool list using the same logic as message handling.
        from .pipeline import build_visible_tools
        speaker_id = interaction.user.id
        role_ids = [r.id for r in getattr(interaction.user, "roles", [])] if interaction.guild else []
        callable_funcs, _ext, _preamble = build_visible_tools(
            agent,
            speaker_id=speaker_id,
            speaker_role_ids=role_ids,
            channel_id=interaction.channel.id,
            owner=is_owner(speaker_id),
        )
        if not callable_funcs:
            await interaction.response.send_message(f"**{agent.display_name}** doesn't have any tools available to you here.", ephemeral=True)
            return
        lines = [f"**{agent.display_name}** can do the following for you:"]
        for f in callable_funcs:
            t = TOOL_REGISTRY.get(f.__name__)
            if t:
                lines.append(f"• `{t.name}` — {t.description}")
        # Discord caps content at 2000 chars. Owner often has 25+ tools and
        # the joined list blows past it — send first chunk via response, rest
        # via followups. Each chunk ephemeral so the channel stays clean.
        body = "\n".join(lines)
        chunks = []
        remaining = body
        while remaining:
            chunks.append(remaining[:1900])
            remaining = remaining[1900:]
        await interaction.response.send_message(chunks[0], ephemeral=True)
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk, ephemeral=True)

    @bot.slash_command(name="status", description="Show this agent's runtime state.")
    async def status_cmd(interaction: nextcord.Interaction):
        agent = runner.agent
        from .config_global import get_model_context_window

        _st_key = _conv_key_for_interaction(runner, interaction)
        conv = runner.conversations.get(_st_key)
        window = get_model_context_window(agent.model, agent.provider)

        lines = [f"**{agent.display_name}**"]
        lines.append(f"• Model: `{agent.model}`")

        # Context usage. Both anthropic and ollama populate `last_usage` on
        # the conversation after each turn; we display it uniformly. If
        # the conversation isn't loaded in memory (cold start, no inbound
        # since restart), fall back to reading the meta sidecar directly
        # so /status shows the persisted numbers instead of 0.
        usage = getattr(conv, "last_usage", None) if conv else None
        if not usage:
            try:
                import glob, json as _json
                _agent_dir = os.path.dirname(agent.path)
                _convs_dir = os.path.join(_agent_dir, "conversations")
                _ch_id = int(interaction.channel.id)
                # Channel-id-suffixed meta files come from any transport
                # (discord:<id>.meta.json, imessage:<id>.meta.json, etc.).
                # Match any prefix on the same channel id.
                _candidates = []
                # Linked conversations: meta lives under the canonical id,
                # which the channel-id glob below can't match.
                from . import _conversation_io as _cio_st
                if isinstance(_st_key, str):
                    _lmp = os.path.join(_convs_dir, f"{_cio_st.fs_encode(_st_key)}.meta.json")
                    if os.path.exists(_lmp):
                        _candidates.append((os.path.getmtime(_lmp), _lmp))
                # fs_encode on the glob pattern too — on Windows the on-disk
                # separator is "%3A", so the pattern must match that form.
                for _mp in glob.glob(os.path.join(_convs_dir, _cio_st.fs_encode(f"*:{_ch_id}") + ".meta.json")):
                    try:
                        _candidates.append((os.path.getmtime(_mp), _mp))
                    except OSError:
                        continue
                if _candidates:
                    # Newest meta file wins.
                    _candidates.sort(reverse=True)
                    _meta = _json.load(open(_candidates[0][1]))
                    _disk_usage = _meta.get("last_usage")
                    if isinstance(_disk_usage, dict) and "total_input" in _disk_usage:
                        usage = _disk_usage
            except Exception:
                pass
        if usage:
            total = usage["total_input"]
            lines.append(f"• Context: {total:,} / {window:,}")
            # Cache stats only meaningful for anthropic + openai (ollama
            # always 0). Both populate the same cache_* keys in last_usage.
            if agent.provider in ("anthropic", "openai"):
                lines.append(
                    f"• Cache: read {usage['cache_read_input_tokens']:,} • "
                    f"create {usage['cache_creation_input_tokens']:,}"
                )
        else:
            lines.append(f"• Context: 0 / {window:,}")

        # Message count.
        if conv:
            in_mem = len([m for m in conv.messages if m.get("role") != "system"])
            try:
                from . import _conversation_io as _cio
                # The conversation object knows its real id (handles linked
                # conversations and any transport prefix); fall back to the
                # legacy discord:<id> guess only if the attr is missing.
                _st_conv_id = getattr(conv, "conversation_id", "") or f"discord:{interaction.channel.id}"
                path = _cio.conversation_path(os.path.dirname(agent.path), _st_conv_id)
                on_disk = sum(1 for _ in open(path)) if os.path.exists(path) else in_mem
            except Exception:
                on_disk = in_mem
            lines.append(f"• Messages: {in_mem} in memory / {on_disk} on disk")

        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    # ---------------- OWNER COMMANDS ----------------

    async def _owner_check(interaction: nextcord.Interaction) -> bool:
        if not is_owner(interaction.user.id):
            await interaction.response.send_message("You don't have permission to run this.", ephemeral=True)
            return False
        return True

    async def _admin_check(interaction: nextcord.Interaction) -> bool:
        if not is_admin(interaction.user.id):
            await interaction.response.send_message("You don't have permission to run this.", ephemeral=True)
            return False
        return True

    # ----- Autocomplete callbacks (silent on failure; Discord eats exceptions) -----

    async def _autocomplete_agents(interaction: nextcord.Interaction, current: str):
        try:
            return [a for a in registry.ALL_AGENTS.keys() if current.lower() in a.lower()][:25]
        except Exception: return []

    async def _autocomplete_all_tools(interaction: nextcord.Interaction, current: str):
        try:
            return [n for n in TOOL_REGISTRY.keys() if current.lower() in n.lower()][:25]
        except Exception: return []

    async def _autocomplete_agent_tools(interaction: nextcord.Interaction, current: str):
        # Tools currently in the agent's allowed_tools list (for revoke / disallow / grant).
        try:
            agent = registry.ALL_AGENTS.get(runner.agent.id)
            if not agent: return []
            return [a.name for a in agent.allowed_tools if current.lower() in a.name.lower()][:25]
        except Exception: return []

    async def _autocomplete_unallowed_tools(interaction: nextcord.Interaction, current: str):
        # Tools registered but NOT in this agent's allowed_tools (for allow_tool).
        try:
            agent = registry.ALL_AGENTS.get(runner.agent.id)
            if not agent: return []
            allowed = {a.name for a in agent.allowed_tools}
            return [n for n in TOOL_REGISTRY.keys() if n not in allowed and current.lower() in n.lower()][:25]
        except Exception: return []

    async def _autocomplete_models(interaction: nextcord.Interaction, current: str):
        try:
            from openflip import ollama_api
            models = await ollama_api.ollama_list()
            return [m["model"] for m in models if current.lower() in m["model"].lower()][:25]
        except Exception: return []

    @bot.slash_command(name="agents",
        description="(owner) Open the interactive agents panel — list, enable/disable, reload, manage tools.")
    async def agents_cmd(interaction: nextcord.Interaction):
        if not await _owner_check(interaction): return
        from . import agents_ui
        await agents_ui.open_agents_panel(interaction, runner_agent_id=runner.agent.id)

    @bot.slash_command(name="usage",
        description="(owner) Aggregated API usage over a time window.")
    async def usage_cmd(
        interaction: nextcord.Interaction,
        window: str = nextcord.SlashOption(
            choices=["24h", "7d", "30d", "all"],
            default="24h",
            required=False,
            description="Time window to aggregate (default 24h)."),
        group_by: str = nextcord.SlashOption(
            name="group_by",
            choices=["session", "channel", "agent", "user", "model"],
            default="session",
            required=False,
            description="How to break down the totals (default session)."),
    ):
        if not await _owner_check(interaction): return
        from . import usage_ledger
        # Window → since_ts. 'all' = everything (since 0).
        _deltas = {"24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400}
        now = time.time()
        since_ts = 0.0 if window == "all" else now - _deltas.get(window, 86400)
        # Friendly group name → ledger column.
        _col_map = {"agent": "agent_id", "channel": "channel_id",
                    "session": "session_id", "user": "user_id", "model": "model"}
        col = _col_map.get(group_by, "session_id")

        grand = usage_ledger.totals(since_ts)
        if not grand.get("turns"):
            await interaction.response.send_message(
                f"No usage recorded in the last {window} yet.", ephemeral=True)
            return
        rows = usage_ledger.aggregate(since_ts, group_by=col)

        header = (
            f"Usage — window: {window} • group by: {group_by}\n"
            f"Totals: {grand['turns']:,} turns • "
            f"{_humanize_tokens(grand['total_tokens'])} tokens • "
            f"${grand['est_cost_usd']:.2f}"
        )
        # Build an aligned monospace table. Truncate to top N groups so the
        # message stays under Discord's 2000-char cap.
        TOP_N = 25
        shown = rows[:TOP_N]
        name_w = max([len(str(r["group"] or "—")) for r in shown] + [5])
        name_w = min(name_w, 28)
        lines = [f"{'group'.ljust(name_w)}  {'turns':>6}  {'tokens':>8}  {'cost':>9}"]
        lines.append("-" * (name_w + 2 + 6 + 2 + 8 + 2 + 9))
        for r in shown:
            grp = str(r["group"] or "—")
            if len(grp) > name_w:
                grp = grp[: name_w - 1] + "…"
            lines.append(
                f"{grp.ljust(name_w)}  {r['turns']:>6,}  "
                f"{_humanize_tokens(r['total_tokens']):>8}  "
                f"${r['est_cost_usd']:>8.2f}"
            )
        if len(rows) > TOP_N:
            lines.append(f"… {len(rows) - TOP_N} more group(s) not shown")

        body = header + "\n```\n" + "\n".join(lines) + "\n```"
        chunks = _chunk_for_discord(body)
        await interaction.response.send_message(chunks[0], ephemeral=True)
        for extra in chunks[1:]:
            await interaction.followup.send(extra, ephemeral=True)

    @bot.slash_command(name="reload",
        description="(admin) Reload this agent's config + system files from disk.")
    async def reload_cmd(interaction: nextcord.Interaction):
        if not await _admin_check(interaction): return
        try:
            changed = runner.reload_agent_config()
        except Exception as e:
            await interaction.response.send_message(f"⚠️ Reload failed: {e}", ephemeral=True)
            return
        if changed:
            await interaction.response.send_message(
                f"♻️ **{runner.agent.display_name}** reloaded — system files re-read, conversations re-applied.",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                f"No on-disk changes detected for **{runner.agent.display_name}**.",
                ephemeral=True,
            )

    @bot.slash_command(name="restart",
        description="(owner) Restart the openflip framework (every agent goes briefly offline).")
    async def restart_cmd(
        interaction: nextcord.Interaction,
        reason: str = nextcord.SlashOption(
            description="Why you're restarting — posted to this channel after restart.",
            required=False,
            default="Manual restart from /restart command.",
        ),
    ):
        if not await _owner_check(interaction): return
        # Reuse the existing restart_gateway tool flow — it writes the
        # sentinel and shells out to systemctl. We invoke it directly so
        # we don't need the agent to emit a tool call for this.
        from .tools.restart import restart_gateway
        from .tool_executor import CURRENT_AGENT, CURRENT_CHANNEL_ID, CURRENT_SPEAKER_ID
        CURRENT_AGENT.set(runner.agent)
        CURRENT_CHANNEL_ID.set(int(getattr(interaction.channel, "id", 0) or 0))
        CURRENT_SPEAKER_ID.set(int(interaction.user.id))
        await interaction.response.send_message(
            f"♻️ Restart triggered — {reason}. Hold on…",
            ephemeral=True,
        )
        try:
            await restart_gateway(reason)
        except Exception as e:
            # If we got here the restart didn't kill us; surface the error.
            try:
                await interaction.followup.send(f"⚠️ Restart failed: {e}", ephemeral=True)
            except Exception:
                pass

    @bot.slash_command(name="stop",
        description="(owner) Hard-interrupt the agent's current turn. Add an instruction to redirect.")
    async def stop_cmd(
        interaction: nextcord.Interaction,
        instruction: str = nextcord.SlashOption(
            description="Optional new instruction — fires as a fresh turn after cancellation.",
            required=False,
            default="",
        ),
    ):
        # Discord-side mirror of the transport-agnostic `/stop` text prefix
        # handled in runtime._handle_message. Both paths funnel through
        # runner._hard_interrupt() so the cancel/clear semantics are
        # identical regardless of how the operator fired it.
        if not await _owner_check(interaction): return
        ch_id = int(getattr(interaction.channel, "id", 0) or 0)
        # Interrupt by conversation key: a linked DM's active turn is slotted
        # under "linked:<canonical>", not the native channel id.
        runner._hard_interrupt(_conv_key_for_interaction(runner, interaction))
        # Form the synthetic user_text exactly as the text-prefix path
        # would land it ("/stop" or "/stop <instruction>") so the model
        # sees a consistent message on both Discord paths.
        instr = (instruction or "").strip()
        synthetic_text = f"/stop {instr}".strip() if instr else "/stop"
        # Enqueue as a synthetic turn from the operator. run_synthetic_turn
        # owns the queue plumbing — we just hand it the right speaker +
        # visibility tag so the new turn behaves like an operator-initiated
        # message (chains can route back to this channel, etc).
        try:
            await runner.run_synthetic_turn(
                ch_id,
                synthetic_text,
                speaker_id=int(interaction.user.id),
                originator_visibility="operator_channel",
            )
        except Exception as e:
            print_ts(f"/stop: failed to enqueue follow-up turn: {e}", error=True)
        reply = f"🛑 Stopped — switching to: {instr}" if instr else "🛑 Stopped."
        await interaction.response.send_message(reply, ephemeral=True)

    @bot.slash_command(name="grant",
        description="(admin) Grant a Discord user access to one of this agent's tools.")
    async def grant_cmd(
        interaction: nextcord.Interaction,
        tool: str = nextcord.SlashOption(autocomplete_callback=_autocomplete_all_tools, description="Tool to grant access to"),
        user: nextcord.User = nextcord.SlashOption(description="Who to grant access"),
    ):
        if not await _admin_check(interaction): return
        agent = registry.ALL_AGENTS.get(runner.agent.id)
        if not agent:
            await interaction.response.send_message("This bot's agent isn't loaded.", ephemeral=True); return
        from .agent import TransportAuth
        acl = _ensure_acl(agent, tool)
        block = acl.auth.setdefault("discord", TransportAuth())
        if user.id not in block.users:
            block.users.append(user.id)
        agent.save()
        await interaction.response.send_message(f"✅ {user.mention} can now use `{tool}` via **{agent.display_name}**.", ephemeral=True)

    @bot.slash_command(name="revoke",
        description="(admin) Revoke a Discord user's access to one of this agent's tools.")
    async def revoke_cmd(
        interaction: nextcord.Interaction,
        tool: str = nextcord.SlashOption(autocomplete_callback=_autocomplete_agent_tools, description="Tool to revoke"),
        user: nextcord.User = nextcord.SlashOption(description="Who to revoke from"),
    ):
        if not await _admin_check(interaction): return
        agent = registry.ALL_AGENTS.get(runner.agent.id)
        if not agent:
            await interaction.response.send_message("This bot's agent isn't loaded.", ephemeral=True); return
        for acl in agent.allowed_tools:
            if acl.name != tool:
                continue
            block = acl.auth.get("discord")
            if block and user.id in block.users:
                block.users = [u for u in block.users if u != user.id]
        agent.save()
        await interaction.response.send_message(f"✅ {user.mention} no longer has access to `{tool}` via **{agent.display_name}**.", ephemeral=True)

    @bot.slash_command(name="inject_context",
        description="(owner) Silently inject context into another agent's conversation history.")
    async def inject_context_cmd(
        interaction: nextcord.Interaction,
        agent_id: str = nextcord.SlashOption(
            autocomplete_callback=_autocomplete_agents,
            description="Target agent"),
        channel_id: str = nextcord.SlashOption(
            description="Discord channel ID to inject into"),
        text: str = nextcord.SlashOption(
            description="Context text to inject"),
    ):
        if not await _owner_check(interaction): return
        target_agent = registry.ALL_AGENTS.get(agent_id)
        if not target_agent:
            await interaction.response.send_message(
                f"❌ Unknown agent: `{agent_id}`", ephemeral=True); return
        try:
            ch_id = int(channel_id)
        except ValueError:
            await interaction.response.send_message(
                "❌ channel_id must be a numeric Discord channel ID.", ephemeral=True); return

        conv_id = f"discord:{ch_id}"
        marked_text = f"[INJECTED CONTEXT]: {text}"
        # Identity links: if the target runner knows this channel belongs to a
        # linked conversation, inject into THAT history (key + file), not a
        # parallel discord:<id> file the linked conversation never reads.
        _link_key = None
        target_runner_early = registry.RUNNERS.get(agent_id)
        if target_runner_early is not None and hasattr(target_runner_early, "conv_key"):
            _k = target_runner_early.conv_key(ch_id)
            if isinstance(_k, str):
                _link_key = _k
                conv_id = _k

        # The LIVE in-memory conversation list is the source of truth for the
        # target's next model turn — a bare JSONL append is invisible until the
        # conversation reloads. So whenever the target runner is active, inject
        # into the in-memory object (then persist), mirroring the battle-tested
        # _drain_pending_injects path in runtime.py.
        target_runner = registry.RUNNERS.get(agent_id)
        if target_runner:
            # get_conversation get-or-creates: if the channel is already loaded
            # we get the live object; if not, it constructs + load()s from disk
            # and registers it in runner.conversations. Either way the list we
            # append to IS the one the next turn reads. (was_loaded only labels
            # the log line.)
            _inj_key = _link_key if _link_key is not None else ch_id
            was_loaded = _inj_key in target_runner.conversations
            conv = target_runner.get_conversation(_inj_key, conv_id)
            # Same ChatMessage construction _drain_pending_injects uses, branched
            # on provider. Append to conv.messages (the live list), then save()
            # — after a fresh load() _persisted_count == len(history), so save()
            # appends ONLY this new message, no duplication.
            from .providers import chat_message_class
            _Msg = chat_message_class(target_agent.provider)
            conv.messages.append(_Msg("user", marked_text))
            conv.save()
            _how = "in-memory (already loaded)" if was_loaded else "in-memory (loaded on demand)"
            print_ts(f"/inject_context: injected into {agent_id} channel {ch_id} [{_how}] + disk",
                     agent=runner.agent.id)
        else:
            from . import _conversation_io as _cio
            agent_dir = os.path.dirname(target_agent.path)
            jsonl_path = _cio.conversation_path(agent_dir, conv_id)
            _cio.append_messages(
                jsonl_path,
                [{"role": "user", "content": marked_text}],
                content_extractor=lambda m: m.get("content", ""),
            )
            # Runner not active at all — disk append is correct; the agent will
            # load() this message fresh on its next inbound for the channel.
            print_ts(f"/inject_context: appended to {agent_id} channel {ch_id} JSONL (runner not active)",
                     agent=runner.agent.id)

        await interaction.response.send_message(
            f"✅ Injected context into **{target_agent.display_name}** for channel `{ch_id}`.",
            ephemeral=True)

    @bot.slash_command(name="dream",
        description="(owner) Trigger a memory-consolidation pass on this agent now.")
    async def dream_cmd(interaction: nextcord.Interaction):
        # Manual /dream works REGARDLESS of the agent's dream.enabled flag —
        # that flag gates only AUTO-fire (wired up separately). Here we just
        # fire a synthetic turn telling the agent to consolidate. The turn is
        # attributed to the owner, so owner-bypass in acl._check_acl makes the
        # dream() tool callable even when it isn't in the agent's allowed_tools.
        if not await _owner_check(interaction): return
        ch_id = int(getattr(interaction.channel, "id", 0) or 0)
        if not ch_id:
            await interaction.response.send_message("Can't dream here — no channel context.", ephemeral=True); return
        prompt = (
            "It's time to dream — consolidate your long-term memory now. "
            "Call the dream() tool, then follow its 4-phase instructions: "
            "distill durable facts, convert relative dates to absolute, prune "
            "facts that were later contradicted, and write the cleaned-up "
            "result back with update_core_memory()."
        )
        try:
            await runner.run_synthetic_turn(
                ch_id,
                prompt,
                auto_post_final_text=True,  # post the consolidation summary back to the invoking channel
                speaker_id=int(interaction.user.id),
                originator_visibility="operator_channel",
            )
        except Exception as e:
            print_ts(f"/dream: failed to enqueue consolidation turn: {e}", error=True)
            await interaction.response.send_message(f"❌ Failed to start dream: {e}", ephemeral=True); return
        await interaction.response.send_message("💤 Dreaming — consolidating memory…", ephemeral=True)

    @bot.slash_command(name="models",
        description="(owner) Open the interactive models panel — list installed, pull, unload.")
    async def models_cmd(interaction: nextcord.Interaction):
        if not await _owner_check(interaction): return
        from . import models_ui
        await models_ui.open_models_panel(interaction)

    # TTS training was removed from the public framework — it's part of the
    # maintainer's personal stack (ComfyUI + custom voice models). To add TTS
    # back, ship your own tts_train module + slash commands.


    # ---------------- TOOLSET (owner-controlled tool parameters) ----------------

    async def _autocomplete_settings_tools(interaction: nextcord.Interaction, current: str):
        try:
            return [n for n in tool_settings.list_tools() if current.lower() in n.lower()][:25]
        except Exception: return []

    async def _autocomplete_tool_keys(interaction: nextcord.Interaction, current: str):
        # Discord autocomplete sees previously-entered options on `interaction.data["options"]`.
        # Walk it to find the `tool` arg the user already chose, then list that tool's keys.
        try:
            tool_name = ""
            for opt in (interaction.data or {}).get("options", []) or []:
                # Drill into subcommand groups if present.
                if opt.get("type") == 1 and opt.get("options"):
                    for sub in opt["options"]:
                        if sub.get("name") == "tool":
                            tool_name = sub.get("value", "")
                if opt.get("name") == "tool":
                    tool_name = opt.get("value", "")
            schema = tool_settings.get_schema(tool_name)
            if not schema: return []
            return [k for k in schema.settings.keys() if current.lower() in k.lower()][:25]
        except Exception: return []

    @bot.slash_command(name="toolset", description="Open the interactive tool settings panel (owner-only).")
    async def toolset_cmd(
        interaction: nextcord.Interaction,
        tool: str = nextcord.SlashOption(default="", required=False,
            autocomplete_callback=_autocomplete_settings_tools,
            description="Optional: open directly to this tool"),
    ):
        # Owner-check is inside open_panel so the response goes through one path.
        from . import toolset_ui
        await toolset_ui.open_panel(interaction, initial_tool=tool or None)

    @bot.slash_command(name="toolset_reset", description="Reset a tool's parameters to their defaults.")
    async def toolset_reset_cmd(
        interaction: nextcord.Interaction,
        tool: str = nextcord.SlashOption(autocomplete_callback=_autocomplete_settings_tools,
            description="Tool whose settings to reset"),
    ):
        if not await _owner_check(interaction): return
        ok, msg = tool_settings.reset_tool(tool)
        await interaction.response.send_message(("✅ " if ok else "❌ ") + msg, ephemeral=True)

    @bot.slash_command(name="toolset_reset_all", description="Reset EVERY tool's parameters to defaults.")
    async def toolset_reset_all_cmd(interaction: nextcord.Interaction):
        if not await _owner_check(interaction): return
        ok, msg = tool_settings.reset_all()
        await interaction.response.send_message(("✅ " if ok else "❌ ") + msg, ephemeral=True)

    # ---------------- MODEL / OPTIONS PANELS (interactive) ---------------------

    @bot.slash_command(name="model", description="(owner) Show or change an agent's model — interactive panel.")
    async def model_cmd(interaction: nextcord.Interaction):
        if not await _owner_check(interaction): return
        from . import agent_ui
        await agent_ui.open_model_panel(interaction, runner_agent_id=runner.agent.id)

    @bot.slash_command(name="options", description="(owner) Show or change an agent's Ollama options — interactive panel.")
    async def options_cmd(interaction: nextcord.Interaction):
        if not await _owner_check(interaction): return
        from . import agent_ui
        await agent_ui.open_options_panel(interaction, runner_agent_id=runner.agent.id)

    @bot.slash_command(name="conversation",
        description="(owner) Dump THIS channel's full conversation payload — messages, tools, options, system extension.")
    async def conversation_cmd(
        interaction: nextcord.Interaction,
        full: bool = nextcord.SlashOption(default=True, required=False,
            description="Include the system extension that would be injected on the next turn"),
    ):
        if not await _owner_check(interaction): return
        await _dump_conversation(interaction, runner=runner, include_extension=full)

    # ---------------- EVAL (behavioral test suite) ------------------------------

    @bot.slash_command(name="eval",
        description="Run the behavioral test suite against an agent (owner-only). Results post in this channel.")
    async def eval_cmd(
        interaction: nextcord.Interaction,
        agent: str = nextcord.SlashOption(
            default="", required=False, autocomplete_callback=_autocomplete_agents,
            description="Agent to test (default: this bot's agent)"),
        mode: str = nextcord.SlashOption(
            choices=["fast (pinned strings only, ~3 min)", "full (with AI-generated prompts, ~10 min)"],
            default="fast (pinned strings only, ~3 min)", required=False,
            description="Speed/coverage tradeoff"),
        category: str = nextcord.SlashOption(
            default="", required=False,
            description="Only run cases whose category contains this (optional)"),
    ):
        if not await _owner_check(interaction): return
        # The behavioral-eval suite (tests/eval_behavior.py) is not currently
        # installed in this deployment — fail honestly instead of firing a
        # background task that ModuleNotFound's silently and leaves the
        # operator with a "starting…" ack that never resolves.
        try:
            import importlib.util as _ilu
            if _ilu.find_spec("tests.eval_behavior") is None:
                raise ModuleNotFoundError
        except Exception:
            await interaction.response.send_message(
                "⚠️ `/eval` is unavailable — the behavioral test suite "
                "(`tests/eval_behavior.py`) isn't installed in this deployment.",
                ephemeral=True,
            )
            return
        target_agent = agent or runner.agent.id
        # Agents are DIRECTORIES: agents/<id>/agent.json (not agents/<id>.json).
        agent_path = os.path.join(_project_root_path(), "agents", target_agent, "agent.json")
        if not os.path.isfile(agent_path):
            await interaction.response.send_message(f"❌ Agent JSON not found: `{target_agent}/agent.json`", ephemeral=True)
            return
        is_full = mode.startswith("full")
        await interaction.response.send_message(
            f"🧪 Eval starting for `{target_agent}` ({'full' if is_full else 'fast'}). Progress + results in this channel.",
            ephemeral=True,
        )
        # Run in a background task so we don't tie this to the interaction's 15-min window.
        asyncio.create_task(_run_eval_to_channel(
            channel=interaction.channel, agent_id=target_agent,
            full=is_full, category=category, invoker_name=interaction.user.name,
        ))

    # ---------------- AGENT-SPECIFIC COMMANDS ----------------
    # Each block below is gated on `if "<name>" in runner.agent.agent_specific_commands:`
    # so the command is registered ONLY on bots whose agent.json opts in via
    # `"agent_specific_commands": ["<name>"]`. Adding a new agent-specific
    # command: add another gated block here and add the name to the
    # relevant agent's agent.json.

    # (no agent-specific commands currently registered.)


def _humanize_tokens(n: int) -> str:
    """Render a token count compactly: 2.1M / 800k / 512. Used by /usage."""
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _chunk_for_discord(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]
    out = []
    while text:
        # Prefer to break on a blank line within the limit.
        slice_ = text[:limit]
        cut = slice_.rfind("\n\n")
        if cut < 200:
            cut = slice_.rfind("\n")
        if cut <= 0:
            cut = limit
        out.append(text[:cut])
        text = text[cut:].lstrip()
    return out


async def _dump_conversation(interaction: nextcord.Interaction, *, runner, include_extension: bool):
    """Dump the conversation for this channel as JSON. Inline if short, file if not.

    Shows: agent id, model, options, messages (role/content/tool_calls/thinking),
    callable tool names, and optionally the system extension that build_visible_tools
    would currently inject on the next turn (so you can see exactly what the model
    sees, not just what's in conv.messages)."""
    import json as _json
    from .pipeline import build_visible_tools
    from .acl import is_owner

    channel_id = interaction.channel.id
    conv = runner.conversations.get(_conv_key_for_interaction(runner, interaction))

    # Channel label: prefer a real human-readable name when nextcord exposes it.
    # DMs and partial channels won't have one; surface that explicitly instead
    # of pretending the numeric id is a channel name.
    ch = interaction.channel
    if isinstance(ch, nextcord.DMChannel):
        ch_label = "DM"
    elif isinstance(ch, nextcord.Thread):
        ch_label = f"thread:{getattr(ch, 'name', None) or '(unnamed)'}"
    elif getattr(ch, 'name', None):
        ch_label = f"#{ch.name}"
    else:
        ch_label = f"(channel id {channel_id} — name unavailable)"

    payload: dict = {
        "agent": {
            "id": runner.agent.id,
            "display_name": runner.agent.display_name,
            "model": runner.agent.model,
            "ollama_options": dict(runner.agent.ollama_options or {}),
            "tool_response_mode": runner.agent.tool_response_mode,
        },
        "channel_id": channel_id,
        "channel_label": ch_label,
    }

    if not conv or not getattr(conv, "messages", None):
        payload["conversation"] = "(no conversation in this channel yet)"
    else:
        msgs = []
        for m in conv.messages:
            entry: dict = {"role": m.get("role") if hasattr(m, "get") else getattr(m, "role", None)}
            content = (
                getattr(m, "content_text", None)
                if hasattr(m, "content_text") and getattr(m, "content_text", None)
                else (m.get("content") if hasattr(m, "get") else getattr(m, "content", None))
            )
            entry["content"] = content
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                entry["tool_calls"] = [
                    {"function_name": getattr(tc, "function_name", None),
                     "args": getattr(tc, "args", None)}
                    for tc in tool_calls
                ]
            thinking = getattr(m, "thinking", None) or (m.get("thinking") if hasattr(m, "get") else None)
            if thinking:
                entry["thinking"] = thinking[:500] + ("…" if len(thinking) > 500 else "")
            images = m.get("images") if hasattr(m, "get") else None
            if images:
                entry["images"] = images
            msgs.append(entry)
        payload["messages_count"] = len(msgs)
        payload["messages"] = msgs

    if include_extension:
        speaker_id = interaction.user.id
        role_ids = [r.id for r in getattr(interaction.user, "roles", [])] if interaction.guild else []
        callable_funcs, ext, preamble = build_visible_tools(
            runner.agent,
            speaker_id=speaker_id,
            speaker_role_ids=role_ids,
            channel_id=channel_id,
            owner=is_owner(speaker_id),
        )
        payload["next_turn"] = {
            "callable_tools": sorted(f.__name__ for f in callable_funcs),
            "system_extension_chars": len(ext),
            "system_extension": ext,
            "user_preamble_chars": len(preamble),
            "user_preamble": preamble,
        }

    # Sensitive things (tokens, etc.) are NOT in this payload — agent options
    # only contain Ollama params, no secrets. Safe to expose to the owner.
    body = _json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if len(body) <= 1900:
        await interaction.response.send_message(f"```json\n{body}\n```", ephemeral=True)
        return
    # Too long → attach as file.
    buf = io.BytesIO(body.encode("utf-8"))
    fname = f"conversation_{runner.agent.id}_{int(time.time())}.json"
    await interaction.response.send_message(
        f"Conversation payload for `{runner.agent.id}` in this channel "
        f"(`{payload.get('messages_count', 0)}` messages):",
        file=nextcord.File(buf, filename=fname),
        ephemeral=True,
    )


async def _run_eval_to_channel(*, channel, agent_id: str, full: bool, category: str, invoker_name: str):
    """Background runner for /eval. Posts a single progress message that updates,
    then a final scoreboard embed + failure details file."""
    # Lazy import — eval module is heavy and only needed when /eval fires.
    from tests import eval_behavior as ev
    progress_msg = None
    last_edit = 0.0

    async def progress_cb(i, total, case, result):
        nonlocal progress_msg, last_edit
        # Throttle edits to ~once every 6s + on the last one to avoid Discord rate limits.
        now = time.time()
        if i != total and now - last_edit < 6:
            return
        last_edit = now
        passes = sum(1 for ln in ev_progress_lines if "✅" in ln)
        fails = sum(1 for ln in ev_progress_lines if "❌" in ln)
        text = (
            f"🧪 **Eval running** — `{agent_id}` ({'full' if full else 'fast'})\n"
            f"`{i}/{total}` cases — ✅ {passes} | ❌ {fails}"
        )
        try:
            if progress_msg is None:
                progress_msg = await asyncio.wait_for(channel.send(text), timeout=30.0)
            else:
                await asyncio.wait_for(progress_msg.edit(content=text), timeout=30.0)
        except asyncio.TimeoutError:
            print_ts(f"eval progress update timed out after 30s", error=True)
        except Exception:
            pass

    # The eval's progress_cb passes us (i, total, case, result) — but we want to
    # also track running pass/fail counts for the message. Capture them locally.
    ev_progress_lines: list[str] = []
    async def wrapped_progress_cb(i, total, case, result):
        flag = "✅" if result.passed else "❌"
        actual = result.actual_tool or "(chat)"
        line = f"[{i}/{total}] {case.category}/{case.name} {flag} {actual}"
        if result.notes:
            line += f"  — {result.notes}"
        ev_progress_lines.append(line)
        await progress_cb(i, total, case, result)

    try:
        report = await ev.run_eval(
            agent_id=agent_id,
            pinned_only=not full,
            filter_str=category,
            progress_cb=wrapped_progress_cb,
        )
    except Exception as e:
        try:
            await asyncio.wait_for(channel.send(f"❌ Eval crashed: `{e}`"), timeout=30.0)
        except asyncio.TimeoutError:
            print_ts(f"eval crash report send timed out after 30s", error=True)
        except Exception:
            pass
        return

    # Build a final embed + (optionally) attach failure details as a file.
    color = 0x57F287 if report.failed == 0 else (0xED4245 if report.failed > report.passed // 2 else 0xFEE75C)
    pct = (100 * report.passed / report.total) if report.total else 0
    embed = nextcord.Embed(
        title=f"🧪 Eval — {report.agent_id} ({report.model})",
        description=f"**{report.passed}/{report.total} passed** ({pct:.0f}%)  •  {report.elapsed_s:.0f}s  •  triggered by {invoker_name}",
        color=color,
    )
    for cat in sorted(report.by_category.keys()):
        d = report.by_category[cat]
        line = f"{d['passes']}/{d['total']}"
        if d['gen_n']:
            line += f"  (pinned {d['pinned_p']}/{d['pinned_n']}, gen {d['gen_p']}/{d['gen_n']})"
        embed.add_field(name=cat, value=line, inline=False)
    files: list[nextcord.File] = []
    if report.failures:
        # Build a text dump of failures.
        buf = io.StringIO()
        buf.write(f"# Failures ({len(report.failures)})\n\n")
        for f in report.failures:
            buf.write(f"## {f['category']}/{f['name']}  ({f['source']})\n")
            buf.write(f"- prompt: {f['prompt']}\n")
            buf.write(f"- actual: {f['actual_tool'] or '(chat)'}\n")
            buf.write(f"- notes:  {f['notes']}\n")
            if f['reply']:
                buf.write(f"- reply:  {f['reply'][:600]}\n")
            buf.write("\n")
        buf.seek(0)
        files.append(nextcord.File(buf, filename=f"eval_failures_{int(time.time())}.txt"))

    try:
        if progress_msg is not None:
            try:
                await asyncio.wait_for(progress_msg.delete(), timeout=15.0)
            except asyncio.TimeoutError:
                print_ts(f"eval progress_msg.delete timed out after 15s", error=True)
            except Exception:
                pass
        # 60s timeout — file upload can take longer than text-only sends.
        await asyncio.wait_for(channel.send(embed=embed, files=files or None), timeout=60.0)
    except Exception as e:
        try:
            await asyncio.wait_for(
                channel.send(f"⚠️ Eval finished but failed to post results: {e}"),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            print_ts(f"eval results-post-failure-fallback send timed out after 30s", error=True)
        except Exception:
            pass


def _ensure_acl(agent: Agent, tool_name: str) -> ToolACL:
    for a in agent.allowed_tools:
        if a.name == tool_name:
            return a
    # Fresh entry: empty auth dict. /grant immediately fills in
    # auth["discord"].users with the granted user, so the tool is never
    # silently created in an open state.
    new = ToolACL(name=tool_name)
    agent.allowed_tools.append(new)
    return new
