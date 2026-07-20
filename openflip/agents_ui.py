"""Interactive /agents panel.

Replaces the flat /agent_list, /agent_enable, /agent_disable, /agent_reload,
/agent_allow_tool, /agent_disallow_tool commands with one ephemeral panel.

Layout:
    [Embed: agent list summary + selected agent's full state]
    [Agent ▾]      <- StringSelect to pick which agent the panel operates on
    [Tool ▾]       <- StringSelect to toggle a tool in allowed_tools
    [Enable/Disable] [Reload JSON] [Refresh] [Close]

For per-USER ACL grants/revokes (which require picking a Discord user), use
/agent_grant and /agent_revoke — those use Discord's native user picker.
"""
from __future__ import annotations
from typing import Optional

import nextcord

from .acl import is_owner


_VIEW_TIMEOUT_S = 15 * 60


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _list_agents() -> list[tuple[str, str]]:
    from . import registry
    return [(a.id, a.display_name) for a in sorted(registry.ALL_AGENTS.values(), key=lambda a: a.id)]


def _get_agent(agent_id: Optional[str]):
    if not agent_id:
        return None
    from . import registry
    return registry.ALL_AGENTS.get(agent_id)


def _is_running(agent_id: str) -> bool:
    from . import registry
    return agent_id in registry.RUNNERS


def _is_enabled(agent_id: str) -> bool:
    from .persistence import is_enabled
    return is_enabled(agent_id)


def _build_embed(selected_id: Optional[str]) -> nextcord.Embed:
    from .tools import TOOL_REGISTRY
    e = nextcord.Embed(title="🤖 Agents", color=0x5865F2)
    # Always show the all-agents summary at the top.
    summary = []
    for aid, dname in _list_agents():
        running = "🟢" if _is_running(aid) else "⚪"
        enabled = "" if _is_enabled(aid) else " _(disabled)_"
        marker = "🔹 " if aid == selected_id else ""
        a = _get_agent(aid)
        model = f" — `{a.model}`" if a else ""
        summary.append(f"{marker}{running} **{dname}** (`{aid}`){model}{enabled}")
    e.description = "\n".join(summary) or "_(no agents)_"

    if selected_id:
        agent = _get_agent(selected_id)
        if agent:
            e.add_field(
                name=f"Selected: {agent.display_name}",
                value=(
                    f"**Status**: {'🟢 running' if _is_running(agent.id) else '⚪ stopped'} • "
                    f"{'enabled' if _is_enabled(agent.id) else 'disabled'}\n"
                    f"**Model**: `{agent.model}`\n"
                    f"**Mode**: `{agent.tool_response_mode}` • "
                    f"**Channels**: `{agent.respond_in}`"
                ),
                inline=False,
            )
            tool_lines = []
            allowed_names = {a.name for a in agent.allowed_tools}
            for name in sorted(TOOL_REGISTRY.keys()):
                if name in allowed_names:
                    acl = next(a for a in agent.allowed_tools if a.name == name)
                    bits = []
                    for transport, tauth in sorted(acl.auth.items()):
                        parts = []
                        if tauth.all_users:
                            parts.append("all")
                        if tauth.users:
                            parts.append(f"{len(tauth.users)} user(s)")
                        if tauth.roles:
                            parts.append(f"{len(tauth.roles)} role(s)")
                        if tauth.channels:
                            parts.append(f"{len(tauth.channels)} chan(s)")
                        if not parts:
                            parts.append("none")
                        bits.append(f"{transport}: {', '.join(parts)}")
                    extra = f"  _({' • '.join(bits)})_" if bits else "  _(no transport)_"
                    tool_lines.append(f"✅ `{name}`{extra}")
                else:
                    tool_lines.append(f"➖ `{name}`")
            e.add_field(name="Tools", value="\n".join(tool_lines) or "_(none registered)_", inline=False)
            e.set_footer(text="Pick a tool below to toggle it in the agent's allowed_tools list.")
    else:
        e.set_footer(text="Pick an agent below to manage it.")
    return e


class _AgentPicker(nextcord.ui.StringSelect):
    def __init__(self, current: Optional[str]):
        opts = []
        for aid, dname in _list_agents():
            opts.append(nextcord.SelectOption(
                label=dname or aid, value=aid, description=f"id: {aid}",
                default=(aid == current),
            ))
        if not opts:
            opts = [nextcord.SelectOption(label="(no agents)", value="__none__")]
        super().__init__(placeholder="Pick an agent…", options=opts, min_values=1, max_values=1, row=0)

    async def callback(self, interaction: nextcord.Interaction):
        view: "AgentsView" = self.view
        if self.values[0] == "__none__":
            await interaction.response.defer()
            return
        view.agent_id = self.values[0]
        await view.refresh(interaction)


class _ToolToggleSelect(nextcord.ui.StringSelect):
    """Pick a tool → toggle its presence in the selected agent's allowed_tools."""
    def __init__(self, agent_id: Optional[str]):
        from .tools import TOOL_REGISTRY
        if agent_id:
            agent = _get_agent(agent_id)
            allowed = {a.name for a in (agent.allowed_tools if agent else [])}
            opts = []
            for name in sorted(TOOL_REGISTRY.keys())[:25]:
                in_list = name in allowed
                opts.append(nextcord.SelectOption(
                    label=_truncate(name, 100),
                    value=name,
                    description=("✅ allowed — pick to remove" if in_list else "➖ not allowed — pick to add"),
                ))
            disabled = False
        else:
            opts = [nextcord.SelectOption(label="(pick an agent first)", value="__none__")]
            disabled = True
        super().__init__(
            placeholder="Toggle a tool in/out of the allowed list…",
            options=opts, min_values=1, max_values=1, row=1,
            disabled=disabled,
        )

    async def callback(self, interaction: nextcord.Interaction):
        view: "AgentsView" = self.view
        if self.values[0] == "__none__" or not view.agent_id:
            await interaction.response.defer()
            return
        agent = _get_agent(view.agent_id)
        if not agent:
            await interaction.response.defer()
            return
        from .agent import ToolACL, TransportAuth
        from .config_global import get_owner_id
        name = self.values[0]
        existing = next((a for a in agent.allowed_tools if a.name == name), None)
        if existing:
            agent.allowed_tools = [a for a in agent.allowed_tools if a.name != name]
            verb = "removed"
        else:
            # New tools default to discord owner-only — safer than the old
            # open-to-everyone default. Use /grant to add other users.
            owner_id = get_owner_id("discord")
            agent.allowed_tools.append(ToolACL(
                name=name,
                auth={"discord": TransportAuth(users=[owner_id])} if owner_id else {},
            ))
            verb = "added (owner-only)"
        ok = agent.save()
        if not ok:
            await interaction.response.send_message("❌ Save failed.", ephemeral=True)
            return
        await view.refresh(interaction)
        try:
            await interaction.followup.send(f"✅ {verb} `{name}` for `{agent.id}`.", ephemeral=True)
        except Exception:
            pass


class _ToggleEnabledButton(nextcord.ui.Button):
    def __init__(self, agent_id: Optional[str]):
        if agent_id and _is_enabled(agent_id):
            super().__init__(label="Disable & stop", style=nextcord.ButtonStyle.danger, emoji="⏹", row=2,
                             disabled=(agent_id is None))
        else:
            super().__init__(label="Enable & start", style=nextcord.ButtonStyle.success, emoji="▶",  row=2,
                             disabled=(agent_id is None))

    async def callback(self, interaction: nextcord.Interaction):
        view: "AgentsView" = self.view
        if not view.agent_id:
            await interaction.response.defer()
            return
        from . import main as flipmain
        from .persistence import set_enabled
        agent = _get_agent(view.agent_id)
        if not agent:
            await interaction.response.send_message("Agent not found.", ephemeral=True)
            return
        if _is_enabled(view.agent_id):
            set_enabled(view.agent_id, False)
            await flipmain.stop_runner(view.agent_id)
            note = f"⏹ `{view.agent_id}` disabled and stopped."
        else:
            set_enabled(view.agent_id, True)
            ok = await flipmain.start_runner(agent)
            note = f"▶ `{view.agent_id}` enabled and {'started' if ok else 'queued (no token)'}."
        await view.refresh(interaction)
        try:
            await interaction.followup.send(note, ephemeral=True)
        except Exception:
            pass


class _ReloadButton(nextcord.ui.Button):
    def __init__(self, agent_id: Optional[str]):
        super().__init__(label="Reload JSON", style=nextcord.ButtonStyle.secondary, emoji="🔁", row=2,
                         disabled=(agent_id is None))

    async def callback(self, interaction: nextcord.Interaction):
        view: "AgentsView" = self.view
        if not view.agent_id:
            await interaction.response.defer()
            return
        from . import registry
        agent = _get_agent(view.agent_id)
        if not agent:
            await interaction.response.send_message("Agent not found.", ephemeral=True)
            return
        agent.reload()
        # Push the freshly-loaded system message to live conversations,
        # tracking failures — a conversation whose reapply raised is still
        # on the STALE system prompt and must not be reported as reloaded.
        reapplied = 0
        failed: list[str] = []
        if view.agent_id in registry.RUNNERS:
            for key, conv in registry.RUNNERS[view.agent_id].conversations.items():
                try:
                    conv.reapply_agent()
                    reapplied += 1
                except Exception as e:
                    failed.append(str(key))
                    from .utils import print_ts, COLOR_RED, COLOR_END
                    print_ts(
                        f"{COLOR_RED}/agents reload: reapply_agent failed for "
                        f"conversation {key} — still on the stale system prompt: {e}{COLOR_END}",
                        error=True, agent=view.agent_id,
                    )
        await view.refresh(interaction)
        msg = (
            f"🔁 Reloaded `{view.agent_id}` from JSON — re-applied to "
            f"{reapplied}/{reapplied + len(failed)} live conversation(s)."
        )
        if failed:
            msg += f"\n⚠️ Re-apply FAILED for: {', '.join(failed)} (details in log)."
        try:
            await interaction.followup.send(msg, ephemeral=True)
        except Exception:
            pass


class _RefreshButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Refresh", style=nextcord.ButtonStyle.secondary, emoji="🔄", row=2)
    async def callback(self, interaction: nextcord.Interaction):
        await self.view.refresh(interaction)


class _CloseButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Close", style=nextcord.ButtonStyle.secondary, emoji="✖", row=2)
    async def callback(self, interaction: nextcord.Interaction):
        view = self.view
        view.stop()
        msg = view.close_message() if hasattr(view, "close_message") else "✖ Closed."
        await interaction.response.edit_message(content=msg, embed=None, view=None)


class AgentsView(nextcord.ui.View):
    def __init__(self, owner_id: int, *, agent_id: Optional[str] = None):
        super().__init__(timeout=_VIEW_TIMEOUT_S)
        self.owner_id = owner_id
        self.agent_id = agent_id
        self.message: Optional[nextcord.Message] = None
        self._build()

    def _build(self):
        self.clear_items()
        self.add_item(_AgentPicker(self.agent_id))
        self.add_item(_ToolToggleSelect(self.agent_id))
        self.add_item(_ToggleEnabledButton(self.agent_id))
        self.add_item(_ReloadButton(self.agent_id))
        self.add_item(_RefreshButton())
        self.add_item(_CloseButton())

    async def interaction_check(self, interaction: nextcord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("This panel belongs to someone else.", ephemeral=True)
            return False
        return True

    async def refresh(self, interaction: nextcord.Interaction):
        self._build()
        await interaction.response.edit_message(embed=_build_embed(self.agent_id), view=self)

    async def refresh_message(self):
        if not self.message:
            return
        self._build()
        try:
            await self.message.edit(embed=_build_embed(self.agent_id), view=self)
        except Exception:
            pass

    async def on_timeout(self):
        for c in self.children:
            try: c.disabled = True
            except Exception: pass
        if self.message:
            try: await self.message.edit(view=self)
            except Exception: pass

    def close_message(self) -> str:
        if not self.agent_id:
            n = len(_list_agents())
            return f"ℹ️ {n} agent(s) — closed without changes."
        agent = _get_agent(self.agent_id)
        if not agent:
            return "✖ Closed."
        return f"ℹ️ `{agent.id}` — {'running' if _is_running(agent.id) else 'stopped'}, model `{agent.model}`."


async def open_agents_panel(interaction: nextcord.Interaction, *, runner_agent_id: str):
    if not is_owner(interaction.user.id):
        await interaction.response.send_message("You don't have permission to run this.", ephemeral=True)
        return
    view = AgentsView(owner_id=interaction.user.id, agent_id=runner_agent_id)
    await interaction.response.send_message(embed=_build_embed(view.agent_id), view=view, ephemeral=True)
    view.message = await interaction.original_message()
