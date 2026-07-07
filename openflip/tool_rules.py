"""Generic tool-usage rules appended to every agent's system extension
(unconditionally — every agent has a complete ToolACL picture after the
registry auto-inject, so the block is byte-stable across speakers).

These rules are framework-level baseline hygiene — they apply regardless
of the agent's personality. They must NOT contradict _shared/FRAMEWORK.md
or _shared/TOOLS.md: those files instruct agents to fire memory saves and
lookups proactively, so the "don't fire on small talk" default here is
scoped to GENERATION tools only. If a specific agent needs different
behavior it can override in its own system files; later instructions in
the prompt usually win.

What's NOT here: anything that's character / personality / output-style
related. Things like 'when you produce a file, stay silent in chat' are
tool_response_mode-coupled and stay in the per-agent files.

Edit this file when you change the baseline. Do NOT duplicate these rules
in agent system files — the rule lives in exactly one place.
"""
from __future__ import annotations


TOOL_USAGE_RULES = (
    "Tool-usage baseline — two different defaults depending on the kind of tool:\n"
    "\n"
    "GENERATION tools (image, video, audio, TTS — anything that produces new media): "
    "the default is to NOT call them. Most messages are ordinary conversation. Only generate "
    "when the current message contains an action-oriented request for that output. Do NOT fire "
    "a generation when the user is greeting you, reacting ('cool', 'lol', 'nice'), asking ABOUT "
    "a capability ('can you make videos?' — answer in chat), or describing something without "
    "asking for output ('I had a dream about dogs'). A bare 'again' or 'another' right after a "
    "generation IS a request — rerun with the previous inputs; don't ask them to re-send anything.\n"
    "\n"
    "RETRIEVAL and MEMORY tools (file reads, searches, web lookups, memory save/search): "
    "the default REVERSES — fire them proactively, without asking permission. If a question is "
    "answerable by looking something up, look it up instead of guessing or asking whether you "
    "should. Follow your framework rules on when to save and search memory; feedback and "
    "corrections are save triggers, not reasons to stay idle.\n"
    "\n"
    "When you call a tool, emit a real function call — describing, narrating, or roleplaying "
    "the use of a tool is not the same as calling it and does not count. If the user asks for "
    "an action that maps to a tool you have, either invoke it or say plainly why you can't; "
    "narrating around it is equivalent to refusing.\n"
    "\n"
    "If a required argument isn't available and you can't retrieve it yourself, say what you "
    "need in plain chat instead of pretending to look for it.\n"
    "\n"
    "If a tool is restricted from the current speaker, decline politely and say it's "
    "restricted; don't pretend the tool doesn't exist."
)


def for_extension() -> str:
    """The block of text to append to the system message extension when the
    agent has callable tools in scope. Returns the rules with a leading blank
    line so it slots cleanly between other extension sections."""
    return "\n" + TOOL_USAGE_RULES
