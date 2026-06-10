"""Single-operator auth. Stores an argon2id-hashed password in
data/auth.json. On first boot, prompts via a setup flow (handled by the
route layer). No registration — exactly one credential exists.

We use argon2id from argon2-cffi (modern, fine defaults). Sessions use
quart-auth's signed cookies — secret key persisted to data/secret_key so
sessions survive restarts."""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

from .config import AUTH_FILE, SECRET_KEY_FILE, TRIGGER_TOKEN_FILE, WEB_DATA_DIR
from ..utils import print_ts, COLOR_YELLOW, COLOR_END


_PH = PasswordHasher()


def _ensure_data_dir() -> None:
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_or_create_secret_key() -> str:
    """Returns the persisted secret key, creating one if absent. Sessions
    survive restarts because the key is stable on disk."""
    _ensure_data_dir()
    if SECRET_KEY_FILE.exists():
        return SECRET_KEY_FILE.read_text().strip()
    key = secrets.token_urlsafe(48)
    SECRET_KEY_FILE.write_text(key)
    os.chmod(SECRET_KEY_FILE, 0o600)
    return key


def ensure_trigger_token() -> str:
    """Create the inbound /trigger bearer token if absent. Call ONCE at
    webapp startup — NOT on the request path.

    This is a machine-to-machine credential (read back verbatim, compared
    with hmac.compare_digest), so it's stored in plaintext — same trust
    model as secret_key. Written atomically, mode 0600. Returns the token so
    the caller can log its location once for the operator.

    Deliberately split from `read_trigger_token`: minting a credential is a
    privileged startup action, never something a request handler should be
    able to trigger. The declined branch called a get-OR-create inside the
    per-request auth check — so deleting the token file mid-flight silently
    rotated the credential out from under live callers. We never do that.
    """
    _ensure_data_dir()
    if TRIGGER_TOKEN_FILE.exists():
        try:
            return TRIGGER_TOKEN_FILE.read_text().strip()
        except OSError:
            return ""
    token = secrets.token_urlsafe(32)
    tmp = TRIGGER_TOKEN_FILE.with_suffix(".tmp")
    tmp.write_text(token + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, TRIGGER_TOKEN_FILE)
    return token


def read_trigger_token() -> str:
    """READ-ONLY accessor for the inbound /trigger bearer token, used on the
    request path. NEVER creates or mutates the file.

    Returns "" (fail-closed) when the file is missing, empty, or unreadable —
    so a deleted/zeroed token file rejects every caller rather than minting a
    fresh secret or, worse, authenticating against an empty string. The
    endpoint additionally refuses to compare against an empty expected token.
    """
    try:
        if not TRIGGER_TOKEN_FILE.exists():
            return ""
        return TRIGGER_TOKEN_FILE.read_text().strip()
    except OSError:
        return ""


def read_trigger_token_secrets(agent_id: str) -> dict[str, str]:
    """READ-ONLY accessor for the per-AGENT, per-hook /trigger token secret store.

    Returns {token_name: secret} for the GIVEN `agent_id` from
    data/trigger_token_secrets.json, or {} on any problem
    (missing/empty/corrupt/unreadable, or no block for this agent) —
    fail-closed, exactly like read_trigger_token. NEVER creates or mutates the
    file; minting per-hook secrets is an operator action (write the file by
    hand, chmod 0600), not a request-path side effect.

    The store is NAMESPACED PER AGENT: {agent_id: {token_name: secret}}. Two
    different agents may both define a token named "email-hook" with DIFFERENT
    secrets; a secret is scoped to exactly one agent and can never authenticate
    to another. This closes the shared-secret-name cross-agent privilege bridge
    that a flat global {token_name: secret} map would open (whoever held agent
    A's "email-hook" secret could otherwise authenticate to agent B and assert
    B's ids, since bindings are per-agent but the secret would be global).

    Secrets are stored OUT of agent.json on purpose: agent.json is
    world-readable framework config that gets hot-reloaded, diffed, and
    re-serialized; a long-lived bearer secret does not belong there. This file
    is the same machine-to-machine plaintext trust model as `trigger_token`
    (read back verbatim, compared with hmac.compare_digest).
    """
    from .config import TRIGGER_TOKENS_FILE  # add in config.py, see 3b
    try:
        if not TRIGGER_TOKENS_FILE.exists():
            return {}
        # Perms warn-on-read: a bearer-secret store should be 0600. If it is
        # group/other-readable, warn (don't fail) — same posture as the other
        # sensitive-file readers. We still read it; the operator owns the chmod.
        try:
            mode = TRIGGER_TOKENS_FILE.stat().st_mode
            if mode & 0o077:
                print_ts(
                    f"{COLOR_YELLOW}auth: {TRIGGER_TOKENS_FILE} is "
                    f"group/other-readable (mode {oct(mode & 0o777)}); bearer "
                    f"secrets should be 0600{COLOR_END}",
                )
        except OSError:
            pass
        data = json.loads(TRIGGER_TOKENS_FILE.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    agent_block = data.get(agent_id)
    if not isinstance(agent_block, dict):
        return {}  # no secrets configured for this agent → fail-closed
    out: dict[str, str] = {}
    for name, secret in agent_block.items():
        if isinstance(name, str) and isinstance(secret, str) and secret.strip():
            out[name] = secret.strip()
    return out


def is_configured() -> bool:
    """True iff a password has been set.

    Fail CLOSED on a corrupt/unreadable auth file: if the file EXISTS but
    can't be parsed, treat the app as configured (return True) so the
    unauthenticated /setup route stays closed. Returning False here on a
    corrupt file would let anyone re-run account creation and seize the
    operator account. Only a genuinely-absent file opens setup.
    """
    if not AUTH_FILE.exists():
        return False
    try:
        data = json.loads(AUTH_FILE.read_text())
        return bool(data.get("hash") and data.get("username"))
    except Exception:
        # File exists but is corrupt/unreadable — fail closed, never reopen setup.
        return True


def set_credentials(username: str, password: str) -> None:
    """Initial setup: writes the argon2id-hashed password + username."""
    _ensure_data_dir()
    payload = {
        "username": username,
        "hash": _PH.hash(password),
    }
    tmp = AUTH_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, AUTH_FILE)


def verify(username: str, password: str) -> bool:
    """Returns True if username+password matches stored credentials."""
    if not is_configured():
        return False
    try:
        data = json.loads(AUTH_FILE.read_text())
    except Exception:
        return False
    if data.get("username") != username:
        return False
    try:
        _PH.verify(data["hash"], password)
    except (VerifyMismatchError, InvalidHashError):
        return False
    # Re-hash if argon2 parameters changed (good hygiene)
    if _PH.check_needs_rehash(data["hash"]):
        try:
            data["hash"] = _PH.hash(password)
            tmp = AUTH_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.chmod(tmp, 0o600)
            os.replace(tmp, AUTH_FILE)
        except Exception:
            pass
    return True


def get_username() -> Optional[str]:
    """Returns the configured username or None."""
    if not AUTH_FILE.exists():
        return None
    try:
        return json.loads(AUTH_FILE.read_text()).get("username")
    except Exception:
        return None
