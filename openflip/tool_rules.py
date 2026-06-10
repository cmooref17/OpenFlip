"""Generic tool-usage rules injected into every agent's system message
when that agent has at least one callable tool available for the current
speaker. Tool-less agents (no tools, or all tools ACL-hidden) see nothing
from this module — the rules are never appended to their context.

These rules are framework-level baseline hygiene — they apply regardless
of the agent's personality. If a specific agent needs different behavior
(e.g., never ask clarifying questions, always fire eagerly), it can
override by adding contradicting rules to its own JSON system_message;
later instructions in the prompt usually win.

What's NOT here: anything that's character / personality / output-style
related. Things like 'when you produce a file, stay silent in chat' are
tool_response_mode-coupled and stay in the per-agent JSON.

Edit this file when you change the baseline. Do NOT edit the equivalent
lines in agent JSONs — those should be removed from agents that share
the framework defaults, so the rule lives in exactly one place.
"""
from __future__ import annotations


TOOL_USAGE_RULES = (
    "You have tools available. The DEFAULT is to NOT call a tool — most messages are ordinary conversation "
    "and do not need one. Only call a tool when the user's current message contains an explicit, action-oriented "
    "request to produce a NEW result that the tool generates (image, video, audio, search result, etc.).\n"
    "\n"
    "Specifically, do NOT call a tool when the user is:\n"
    "- greeting you, saying goodbye, or making small talk ('hi', 'how are you', 'good night', 'thanks')\n"
    "- reacting to or commenting on something ('cool', 'nice', 'lol', 'wow', 'I like that')\n"
    "- giving feedback ('that's wrong', 'try again', 'this is broken') — feedback alone is not a new request\n"
    "- being vague or one-word ('another', 'more', 'again', 'do it') — ask what they mean instead\n"
    "- asking ABOUT a tool, model, or capability ('what model do you use?', 'can you do X?') — answer in chat\n"
    "- discussing or describing something without asking for output ('tell me about cats', 'I had a dream about dogs')\n"
    "- venting, joking, roleplaying, or being emotional — stay in chat and stay in character\n"
    "\n"
    "When you call a tool, use the structured tool-call protocol — emit a real function call, not a description of one. "
    "Describing, narrating, or roleplaying the use of a tool is not the same as calling it and does not count. "
    "If the user asks you to perform an action that maps to a tool you have access to, either invoke the tool or "
    "explain in plain chat why you can't — narrating around it is equivalent to refusing.\n"
    "\n"
    "If a required argument isn't available, say what you need in plain chat instead of pretending to look for it.\n"
    "\n"
    "If you're unsure whether a request needs a tool, ask one short clarifying question rather than guessing. "
    "If a tool you'd want is restricted from this user, decline politely and explain it's restricted; don't "
    "pretend the tool doesn't exist."
)


def for_extension() -> str:
    """The block of text to append to the system message extension when the
    agent has callable tools in scope. Returns the rules with a leading blank
    line so it slots cleanly between other extension sections."""
    return "\n" + TOOL_USAGE_RULES
