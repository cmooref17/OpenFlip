"""Interactive Discord UI for /toolset.

One ephemeral message hosts the entire flow:
    [Embed listing current settings for the selected tool]
    [Tool ▾]      <- StringSelect (Row 0)
    [Setting ▾]   <- StringSelect (Row 1)
    [↺ Reset key] [⚠️ Reset tool] [Refresh] [Close]   <- Buttons (Row 2)

Picking a setting opens a context-aware editor:
    bool   → toggled inline (no popup)
    choice → another StringSelect popup with the choices
    int/float/str → Modal with TextInput pre-filled with current value

Owner-only — verified by ToolsetView.interaction_check.
"""
from __future__ import annotations
from typing import Optional

import nextcord

from . import tool_settings
from .acl import is_owner


_VIEW_TIMEOUT_S = 15 * 60  # 15 minutes; user can re-open with /toolset


def _format_value(v) -> str:
    """Truncate long values for inline display so embeds don't overflow."""
    s = str(v)
    if len(s) > 80:
        return s[:77] + "…"
    return s


def _build_embed(tool_name: Optional[str], selected_key: Optional[str] = None) -> nextcord.Embed:
    if not tool_name:
        e = nextcord.Embed(
            title="🔧 Tool Configuration",
            description="Pick a tool below to view and change its parameters.\nThese settings are owner-controlled — the AI sees them but can't change them.",
            color=0x5865F2,
        )
        return e

    schema = tool_settings.get_schema(tool_name)
    if not schema:
        return nextcord.Embed(title=f"Unknown tool: {tool_name}", color=0xED4245)

    values = tool_settings.get_all(tool_name)
    e = nextcord.Embed(
        title=f"🔧 {tool_name}",
        description=f"_{len(schema.settings)} settings_",
        color=0x5865F2,
    )
    for key, s in schema.settings.items():
        current = values.get(key, s.default)
        is_default = (current == s.default)
        is_selected = (key == selected_key)
        marker = ""
        if is_selected:
            marker = "🔹 "
        elif not is_default:
            marker = "✏️ "
        # Field name: marker + key + type/range/choices hint
        type_hint = s.type
        if s.type == "choice" and s.choices:
            type_hint = f"choice[{len(s.choices)}]"
        elif s.type in ("int", "float") and (s.min is not None or s.max is not None):
            lo = s.min if s.min is not None else "-∞"
            hi = s.max if s.max is not None else "∞"
            type_hint = f"{s.type} {lo}..{hi}"
        name = f"{marker}{key} _({type_hint})_"
        # Field value: current value + default if different + description
        body_lines = [f"`{_format_value(current)}`"]
        if not is_default:
            body_lines.append(f"_default: `{_format_value(s.default)}`_")
        body_lines.append(s.description)
        e.add_field(name=name, value="\n".join(body_lines), inline=False)
    if selected_key:
        e.set_footer(text=f"Selected: {selected_key} — use the buttons below to edit or reset.")
    else:
        e.set_footer(text="Pick a setting from the dropdown to edit it.")
    return e


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


class _ToolPicker(nextcord.ui.StringSelect):
    def __init__(self, current: Optional[str]):
        tools = tool_settings.list_tools()
        opts = [
            nextcord.SelectOption(label=name, value=name, default=(name == current))
            for name in tools
        ]
        if not opts:
            opts = [nextcord.SelectOption(label="(no tools registered)", value="__none__")]
        super().__init__(placeholder="Pick a tool…", options=opts, min_values=1, max_values=1, row=0)

    async def callback(self, interaction: nextcord.Interaction):
        view: ToolsetView = self.view
        if self.values[0] == "__none__":
            await interaction.response.defer()
            return
        view.tool_name = self.values[0]
        view.selected_key = None
        await view.refresh(interaction)


class _SettingPicker(nextcord.ui.StringSelect):
    def __init__(self, tool_name: Optional[str], selected_key: Optional[str]):
        if tool_name:
            schema = tool_settings.get_schema(tool_name)
            opts: list[nextcord.SelectOption] = []
            if schema:
                for key, s in schema.settings.items():
                    desc = _truncate(s.description, 95) or s.type
                    opts.append(nextcord.SelectOption(
                        label=key, value=key, description=desc,
                        default=(key == selected_key),
                    ))
            if not opts:
                opts = [nextcord.SelectOption(label="(no settings)", value="__none__")]
        else:
            opts = [nextcord.SelectOption(label="(pick a tool first)", value="__none__")]
        disabled = (tool_name is None) or (opts[0].value == "__none__")
        super().__init__(
            placeholder="Pick a setting to edit…",
            options=opts, min_values=1, max_values=1, row=1,
            disabled=disabled,
        )

    async def callback(self, interaction: nextcord.Interaction):
        view: ToolsetView = self.view
        if self.values[0] == "__none__" or not view.tool_name:
            await interaction.response.defer()
            return
        key = self.values[0]
        schema = tool_settings.get_schema(view.tool_name)
        if not schema or key not in schema.settings:
            await interaction.response.defer()
            return
        s = schema.settings[key]
        view.selected_key = key
        # Context-aware action.
        if s.type == "bool":
            current = tool_settings.get(view.tool_name, key)
            new_val = "false" if bool(current) else "true"
            ok, msg = tool_settings.set_value(view.tool_name, key, new_val)
            if not ok:
                await interaction.response.send_message(f"❌ {msg}", ephemeral=True)
                return
            await view.refresh(interaction)
            return
        if s.type == "choice":
            await interaction.response.send_message(
                f"Choose a new value for **{key}**:",
                view=_ChoicePopupView(view, key, s.choices or []),
                ephemeral=True,
            )
            return
        # str / int / float → modal
        await interaction.response.send_modal(_EditValueModal(view, key, s))


class _ChoicePopupView(nextcord.ui.View):
    def __init__(self, parent: "ToolsetView", key: str, choices: list):
        super().__init__(timeout=120)
        self.parent_view = parent
        self.key = key
        self.add_item(_ChoiceSelect(parent, key, choices))


class _ChoiceSelect(nextcord.ui.StringSelect):
    def __init__(self, parent: "ToolsetView", key: str, choices: list):
        current = tool_settings.get(parent.tool_name, key)
        opts = [
            nextcord.SelectOption(
                label=_truncate(str(c), 100), value=str(c),
                default=(str(c) == str(current)),
            )
            for c in choices[:25]
        ]
        super().__init__(placeholder=f"Pick a value for {key}", options=opts)
        self.parent_view = parent
        self.key = key

    async def callback(self, interaction: nextcord.Interaction):
        ok, msg = tool_settings.set_value(self.parent_view.tool_name, self.key, self.values[0])
        if not ok:
            await interaction.response.send_message(f"❌ {msg}", ephemeral=True)
            return
        # Re-render the parent embed (it's a different message; edit via stored handle).
        await self.parent_view.refresh_message()
        await interaction.response.send_message(f"✅ {msg}", ephemeral=True)


class _EditValueModal(nextcord.ui.Modal):
    def __init__(self, parent: "ToolsetView", key: str, schema):
        super().__init__(title=f"Edit {key}", timeout=300)
        self.parent_view = parent
        self.key = key
        self.schema = schema
        current = tool_settings.get(parent.tool_name, key)
        # Long string defaults (negatives, paths) get a paragraph-style input.
        is_long = isinstance(current, str) and (len(current) > 80 or "\n" in current)
        style = nextcord.TextInputStyle.paragraph if is_long else nextcord.TextInputStyle.short
        constraint_hint = ""
        if schema.type in ("int", "float") and (schema.min is not None or schema.max is not None):
            lo = schema.min if schema.min is not None else "-∞"
            hi = schema.max if schema.max is not None else "∞"
            constraint_hint = f" ({lo}..{hi})"
        elif schema.type == "bool":
            constraint_hint = " (true/false)"
        self.field = nextcord.ui.TextInput(
            label=_truncate(f"{key}{constraint_hint}", 45),
            default_value=str(current),
            placeholder=_truncate(schema.description, 100),
            required=True,
            style=style,
        )
        self.add_item(self.field)

    async def callback(self, interaction: nextcord.Interaction):
        raw = self.field.value or ""
        ok, msg = tool_settings.set_value(self.parent_view.tool_name, self.key, raw)
        if not ok:
            await interaction.response.send_message(f"❌ {msg}", ephemeral=True)
            return
        await self.parent_view.refresh_message()
        await interaction.response.send_message(f"✅ {msg}", ephemeral=True)


class _ConfirmResetToolView(nextcord.ui.View):
    def __init__(self, parent: "ToolsetView"):
        super().__init__(timeout=60)
        self.parent_view = parent

    @nextcord.ui.button(label="Yes, reset everything for this tool", style=nextcord.ButtonStyle.danger)
    async def confirm(self, button: nextcord.ui.Button, interaction: nextcord.Interaction):
        if not self.parent_view.tool_name:
            await interaction.response.send_message("No tool selected.", ephemeral=True)
            return
        ok, msg = tool_settings.reset_tool(self.parent_view.tool_name)
        await self.parent_view.refresh_message()
        await interaction.response.edit_message(content=("✅ " if ok else "❌ ") + msg, view=None)

    @nextcord.ui.button(label="Cancel", style=nextcord.ButtonStyle.secondary)
    async def cancel(self, button: nextcord.ui.Button, interaction: nextcord.Interaction):
        await interaction.response.edit_message(content="Cancelled.", view=None)


class ToolsetView(nextcord.ui.View):
    def __init__(self, owner_id: int, *, initial_tool: Optional[str] = None):
        super().__init__(timeout=_VIEW_TIMEOUT_S)
        self.owner_id = owner_id
        self.tool_name = initial_tool
        self.selected_key: Optional[str] = None
        self.message: Optional[nextcord.Message] = None
        self._build_components()

    def _build_components(self) -> None:
        self.clear_items()
        self.add_item(_ToolPicker(self.tool_name))
        self.add_item(_SettingPicker(self.tool_name, self.selected_key))
        # Row 2 buttons
        self.add_item(_ResetKeyButton(disabled=(not self.tool_name or not self.selected_key)))
        self.add_item(_ResetToolButton(disabled=(not self.tool_name)))
        self.add_item(_RefreshButton())
        self.add_item(_CloseButton())

    async def interaction_check(self, interaction: nextcord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("This panel belongs to someone else.", ephemeral=True)
            return False
        return True

    async def refresh(self, interaction: nextcord.Interaction) -> None:
        """Re-render in response to a component interaction (uses the interaction's response)."""
        self._build_components()
        await interaction.response.edit_message(
            embed=_build_embed(self.tool_name, self.selected_key),
            view=self,
        )

    async def refresh_message(self) -> None:
        """Re-render after a side-channel mutation (modal/popup) where we already used the response."""
        if not self.message:
            return
        self._build_components()
        try:
            await self.message.edit(embed=_build_embed(self.tool_name, self.selected_key), view=self)
        except Exception:
            pass

    async def on_timeout(self) -> None:
        # Disable everything so stale clicks don't error out.
        for child in self.children:
            try:
                child.disabled = True
            except Exception:
                pass
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass

    def close_message(self) -> str:
        if not self.tool_name:
            return "✖ Closed."
        schema = tool_settings.get_schema(self.tool_name)
        if not schema:
            return f"✖ Closed."
        values = tool_settings.get_all(self.tool_name)
        # Highlight non-default overrides if any, else just confirm tool name.
        overridden = []
        for key, s in schema.settings.items():
            if values.get(key) != s.default:
                overridden.append(f"{key}={values.get(key)}")
        if overridden:
            preview = ", ".join(overridden[:4])
            if len(overridden) > 4:
                preview += f", +{len(overridden) - 4} more"
            return f"ℹ️ Settings unchanged for `{self.tool_name}` ({preview})."
        return f"ℹ️ Settings unchanged for `{self.tool_name}` (all defaults)."


class _ResetKeyButton(nextcord.ui.Button):
    def __init__(self, disabled: bool):
        super().__init__(label="Reset key", style=nextcord.ButtonStyle.secondary, emoji="🔁", disabled=disabled, row=2)

    async def callback(self, interaction: nextcord.Interaction):
        view: ToolsetView = self.view
        if not view.tool_name or not view.selected_key:
            await interaction.response.defer()
            return
        # Single-key reset: drop just that override, persist, refresh.
        all_values = tool_settings._VALUES.get(view.tool_name) or {}
        if view.selected_key in all_values:
            all_values.pop(view.selected_key, None)
            if not all_values:
                tool_settings._VALUES.pop(view.tool_name, None)
            tool_settings._persist()
        await view.refresh(interaction)


class _ResetToolButton(nextcord.ui.Button):
    def __init__(self, disabled: bool):
        super().__init__(label="Reset tool", style=nextcord.ButtonStyle.danger, emoji="⚠️", disabled=disabled, row=2)

    async def callback(self, interaction: nextcord.Interaction):
        view: ToolsetView = self.view
        if not view.tool_name:
            await interaction.response.defer()
            return
        await interaction.response.send_message(
            f"Reset **all** settings for `{view.tool_name}` to defaults?",
            view=_ConfirmResetToolView(view), ephemeral=True,
        )


class _RefreshButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Refresh", style=nextcord.ButtonStyle.secondary, emoji="🔄", row=2)

    async def callback(self, interaction: nextcord.Interaction):
        view: ToolsetView = self.view
        await view.refresh(interaction)


class _CloseButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Close", style=nextcord.ButtonStyle.secondary, emoji="✖", row=2)

    async def callback(self, interaction: nextcord.Interaction):
        view: ToolsetView = self.view
        view.stop()
        # Confirm current state on close — view supplies the message.
        msg = view.close_message() if hasattr(view, "close_message") else "✖ Closed."
        await interaction.response.edit_message(content=msg, embed=None, view=None)


async def open_panel(interaction: nextcord.Interaction, *, initial_tool: Optional[str] = None) -> None:
    """Entry point — shows the panel as an ephemeral message and stores the message handle so
    modal/popup callbacks can edit it."""
    if not is_owner(interaction.user.id):
        await interaction.response.send_message("You don't have permission to run this.", ephemeral=True)
        return
    view = ToolsetView(owner_id=interaction.user.id, initial_tool=initial_tool)
    embed = _build_embed(view.tool_name, view.selected_key)
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
    # Capture the message handle for subsequent edits from modals/popups (they
    # consume the interaction.response themselves, so we need a stored ref).
    view.message = await interaction.original_message()
