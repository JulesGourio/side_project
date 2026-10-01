"""User service — resolve the current user's identity from request headers.

There is no capability matrix here: this app does one thing, and access is
gated at the Databricks App level (workspace group permissions on the app
resource), not per-feature in application code. `require_translate` is kept
as a no-op dependency purely so the translate.py router's
`Depends(require_translate)` call sites need no edits.
"""

import asyncio
import logging
import os
from typing import Optional, TypedDict

from fastapi import Request


class UserIdentity(TypedDict):
    user_id: str                 # Numeric part only — e.g. "78655966095635"
    workspace_id: Optional[str]  # Workspace code (part after '@') — e.g. "2865348338307293"
    email: Optional[str]         # Real email if x-forwarded-user is one, otherwise None


logger = logging.getLogger(__name__)

_dev_user_cache: Optional[str] = None


def _is_dev() -> bool:
    return os.getenv('ENV', 'development') == 'development'


def _is_real_email(value: str) -> bool:
    """True if value looks like a real email (not a numeric Databricks workspace-scoped ID)."""
    if '@' not in value:
        return False
    local, _, domain = value.partition('@')
    return not (local.isdigit() and domain.isdigit())


def get_workspace_url() -> str:
    """Return the Databricks workspace base URL."""
    host = os.getenv('DATABRICKS_HOST', '')
    if not host:
        try:
            from databricks.sdk import WorkspaceClient
            host = WorkspaceClient().config.host or ''
        except Exception:
            pass
    if not host:
        return ''
    host = host.rstrip('/')
    if not host.startswith('http'):
        host = f'https://{host}'
    return host


async def get_current_user(request: Request) -> str:
    """Return the authenticated user's email or numeric user_id.

    Resolution order:
    1. x-forwarded-user — real email → use directly
    2. x-forwarded-user — numeric Databricks ID → return the numeric part
    3. WorkspaceClient.current_user.me() — local development only
    """
    forwarded = request.headers.get('x-forwarded-user', '').strip()
    if forwarded and _is_real_email(forwarded):
        return forwarded
    if forwarded:
        return forwarded.split('@', 1)[0]
    if _is_dev():
        return await _dev_user()
    raise ValueError('No user identity found in request headers.')


async def get_user_identity(request: Request) -> UserIdentity:
    """Return the split user_id, workspace_id, and email from the request headers."""
    forwarded = request.headers.get('x-forwarded-user', '').strip()

    if not forwarded:
        if _is_dev():
            user = await _dev_user()
            return UserIdentity(user_id=user, workspace_id=None, email=user)
        logger.warning('x-forwarded-user header is absent — cannot identify user')
        return UserIdentity(user_id='', workspace_id=None, email=None)

    if _is_real_email(forwarded):
        return UserIdentity(user_id=forwarded, workspace_id=None, email=forwarded)

    parts = forwarded.split('@', 1)
    clean_uid = parts[0]
    workspace_id: Optional[str] = parts[1] if len(parts) > 1 else None
    return UserIdentity(user_id=clean_uid, workspace_id=workspace_id, email=None)


async def require_translate(request: Request) -> None:
    """No-op dependency — access is gated at the Databricks App level, not here."""
    return None


async def _dev_user() -> str:
    global _dev_user_cache
    if _dev_user_cache:
        return _dev_user_cache
    from databricks.sdk import WorkspaceClient
    try:
        me = await asyncio.to_thread(lambda: WorkspaceClient().current_user.me())
        _dev_user_cache = me.user_name or me.display_name or ''
    except Exception as e:
        raise ValueError(f'Could not determine current user: {e}') from e
    return _dev_user_cache
