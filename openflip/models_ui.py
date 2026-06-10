"""Interactive /models panel.

Replaces the flat /list_models, /pull_model, /unload_models, /set_model commands
with one ephemeral panel that shows installed Ollama models and lets you pull
new ones or unload everything.

To CHANGE which model an agent uses, use /model (the agent-model panel).
This panel is for the Ollama-side inventory.
"""
from __future__ import annotations
import asyncio
from typing import Optional

import nextcord

from .acl import is_owner


_VIEW_TIMEOUT_S = 15 * 60


def _fmt_size(n: int) -> str:
    if not n:
        return "cloud"
    mb = n / (1024 * 1024)
    if mb < 1024:
        return f"{mb:.0f} MB"
    return f"{mb/1024:.1f} GB"


async def _fetch_models() -> list[dict]:
    try:
        from openflip import ollama_api
        return await asyncio.wait_for(ollama_api.ollama_list(), timeout=30) or []
    except Exception:
        return []


def _build_embed(models: list[dict]) -> nextcord.Embed:
    e = nextcord.Embed(title="🤖 Installed Models", color=0x5865F2)
    if not models:
        e.description = "No models installed (or Ollama unreachable)."
        return e
    lines = [f"**{len(models)} installed:**"]
    for m in models[:60]:  # cap to avoid 6000-char embed limit
        name = m.get("model") or "?"
        size = _fmt_size(m.get("size", 0))
        lines.append(f"• `{name}`  ({size})")
    if len(models) > 60:
        lines.append(f"… +{len(models) - 60} more")
    e.description = "\n".join(lines)
    e.set_footer(text="Use /model to change which model an agent uses. This panel manages the Ollama install only.")
    return e


class _PullModal(nextcord.ui.Modal):
    def __init__(self, parent: "ModelsView"):
        super().__init__(title="Pull a model", timeout=300)
        self.parent_view = parent
        self.field = nextcord.ui.TextInput(
            label="Model tag",
            placeholder="e.g. qwen3.5:cloud, llama3.3:70b, ollama/foo",
            required=True,
            style=nextcord.TextInputStyle.short,
        )
        self.add_item(self.field)

    async def callback(self, interaction: nextcord.Interaction):
        name = (self.field.value or "").strip()
        if not name:
            await interaction.response.send_message("❌ Empty model name.", ephemeral=True)
            return
        await interaction.response.send_message(f"⏳ Pulling `{name}`…", ephemeral=True)
        try:
            from openflip import ollama_api
            # A pull DOWNLOADS the model — minutes, not seconds. Generous cap.
            await asyncio.wait_for(ollama_api.ollama_pull(name), timeout=1800)
            await interaction.followup.send(f"✅ Pulled `{name}`.", ephemeral=True)
        except asyncio.TimeoutError:
            await interaction.followup.send(
                f"❌ Pull of `{name}` timed out (30 min).", ephemeral=True
            )
            return
        except Exception as e:
            await interaction.followup.send(f"❌ Pull failed: `{e}`", ephemeral=True)
            return
        await self.parent_view.refresh_message()


class _PullButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Pull model", style=nextcord.ButtonStyle.primary, emoji="➕", row=0)
    async def callback(self, interaction: nextcord.Interaction):
        await interaction.response.send_modal(_PullModal(self.view))


class _UnloadButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Unload all loaded", style=nextcord.ButtonStyle.danger, emoji="🗑", row=0)
    async def callback(self, interaction: nextcord.Interaction):
        try:
            from openflip import ollama_api
            await asyncio.wait_for(ollama_api.ollama_unload(), timeout=30)
            await interaction.response.send_message("✅ Unloaded all loaded models.", ephemeral=True)
        except asyncio.TimeoutError:
            await interaction.response.send_message("❌ Unload timed out.", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(f"❌ Unload failed: `{e}`", ephemeral=True)


class _RefreshButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Refresh", style=nextcord.ButtonStyle.secondary, emoji="🔄", row=0)
    async def callback(self, interaction: nextcord.Interaction):
        await self.view.refresh(interaction)


class _CloseButton(nextcord.ui.Button):
    def __init__(self):
        super().__init__(label="Close", style=nextcord.ButtonStyle.secondary, emoji="✖", row=0)
    async def callback(self, interaction: nextcord.Interaction):
        view = self.view
        view.stop()
        msg = view.close_message() if hasattr(view, "close_message") else "✖ Closed."
        await interaction.response.edit_message(content=msg, embed=None, view=None)


class ModelsView(nextcord.ui.View):
    def __init__(self, owner_id: int, *, models: list[dict]):
        super().__init__(timeout=_VIEW_TIMEOUT_S)
        self.owner_id = owner_id
        self.models = models
        self.message: Optional[nextcord.Message] = None
        self._build()

    def _build(self):
        self.clear_items()
        self.add_item(_PullButton())
        self.add_item(_UnloadButton())
        self.add_item(_RefreshButton())
        self.add_item(_CloseButton())

    async def interaction_check(self, interaction: nextcord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("This panel belongs to someone else.", ephemeral=True)
            return False
        return True

    async def refresh(self, interaction: nextcord.Interaction):
        self.models = await _fetch_models()
        self._build()
        await interaction.response.edit_message(embed=_build_embed(self.models), view=self)

    async def refresh_message(self):
        if not self.message:
            return
        self.models = await _fetch_models()
        self._build()
        try:
            await self.message.edit(embed=_build_embed(self.models), view=self)
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
        return f"ℹ️ {len(self.models)} model(s) installed."


async def open_models_panel(interaction: nextcord.Interaction):
    if not is_owner(interaction.user.id):
        await interaction.response.send_message("You don't have permission to run this.", ephemeral=True)
        return
    models = await _fetch_models()
    view = ModelsView(owner_id=interaction.user.id, models=models)
    await interaction.response.send_message(embed=_build_embed(models), view=view, ephemeral=True)
    view.message = await interaction.original_message()
