"""CSRF protection for openflip's web app.

Quart-Auth handles authentication but provides no CSRF. Without explicit
CSRF tokens, any page the operator visits while logged in can POST to
the web app — including malicious sites in another tab — and execute
state-changing actions (toggle agents, edit agent.json to grant tools to
attackers, save tool settings, etc).

This module implements a simple synchronizer-token pattern:
  * On the first GET after login, a random token is stored in the Quart
    session and exposed to templates via the `csrf_token()` jinja global.
  * Every POST handler decorated with `@csrf_protect` reads the
    `csrf_token` form field (or `X-CSRF-Token` header) and rejects the
    request with 403 if it doesn't match the session value.
  * Login itself is intentionally NOT protected — there's no prior
    authenticated session to bind a token to. Setup (first-run bootstrap)
    is the same; once login exists, every subsequent state-changing POST
    must carry a valid token.

Token lifecycle: stored in the session, rotated on logout. A single token
per session is fine for our scale; if we ever needed per-form tokens for
defense against XSS, swap to a dict-of-tokens here.
"""
from __future__ import annotations

import secrets
from functools import wraps
from typing import Callable

from quart import session, request, abort


_TOKEN_KEY = "_csrf_token"


def get_or_create_csrf_token() -> str:
    """Return the session's CSRF token, creating one if absent.

    Templates and form rendering call this. Idempotent.
    """
    token = session.get(_TOKEN_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[_TOKEN_KEY] = token
    return token


def clear_csrf_token() -> None:
    """Wipe the token. Called on logout so a fresh session gets a fresh one."""
    session.pop(_TOKEN_KEY, None)


def csrf_protect(handler: Callable) -> Callable:
    """Decorator: enforce CSRF token presence + match on a POST handler.

    Wraps any async route handler so it 403s when the submitted token is
    missing or doesn't match the session token. Reads the token from
    either the `csrf_token` form field OR the `X-CSRF-Token` header
    (for fetch/XHR callers that don't use form-encoded bodies).

    Constant-time comparison (secrets.compare_digest) so timing attacks
    can't enumerate token chars.
    """
    @wraps(handler)
    async def wrapper(*args, **kwargs):
        expected = session.get(_TOKEN_KEY) or ""
        if not expected:
            # No token in session means the user isn't really logged in
            # the way we expect — bounce.
            abort(403, "CSRF token missing from session.")
        submitted = ""
        # Form-encoded first (the common case for HTML forms).
        try:
            form = await request.form
            submitted = form.get("csrf_token", "") or ""
        except Exception:
            submitted = ""
        # Header fallback for AJAX/fetch callers.
        if not submitted:
            submitted = request.headers.get("X-CSRF-Token", "") or ""
        if not submitted or not secrets.compare_digest(submitted, expected):
            abort(403, "CSRF token missing or invalid.")
        return await handler(*args, **kwargs)
    return wrapper
