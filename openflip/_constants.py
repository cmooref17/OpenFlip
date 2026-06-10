"""Framework-wide constants with NO openflip imports — safe to import from
anywhere without risking a cycle. Keep this module dependency-free."""
from __future__ import annotations


# Single source of truth for "tools too dangerous to grant freely". Two
# untrusted-grant paths consult this denylist and must agree:
#
#   * the web config editor (openflip/web/openflip_data.py) refuses to ADD any
#     of these to an agent that doesn't already have them, and
#   * the inbound trigger endpoint (openflip/web/app.py) subtracts these from
#     whatever an unattended trigger requests, regardless of how the set was
#     built.
#
# These are the owner-only / admin-gated / RCE-class powers: shell, framework
# restart, a full coding agent, filesystem mutation/exfiltration, scheduling
# persistence, destructive message ops, and cross-agent context injection.
# More restrictive wins — never drop a name from this set, since dropping one
# would loosen a security boundary at every call site at once.
DANGEROUS_TOOL_NAMES = frozenset({
    "run_command", "claude_code", "restart_gateway", "restart_flask_app",  # code execution / framework control
    "write_file", "edit_file", "delete_file", "send_file",                 # filesystem mutation / exfiltration
    "restore_snapshot",                                                    # filesystem mutation
    "add_cron_job", "cancel_cron_job",                                     # scheduling persistence
    "delete_message", "inject_context",                                    # destructive / cross-agent injection
})
