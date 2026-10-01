"""Config endpoint — exposes app configuration to the frontend.

/me returns identity only — no can_* capability flags, since access is
gated at the Databricks App level, not per-feature in application code.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..services.user import get_current_user, get_workspace_url

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('/config/app')
async def get_app_config():
    """Return application configuration (branding)."""
    return {
        'app_name': 'LatLang',
        'branding': {
            'name': 'LatLang',
            'logo': '/logos/LOGO_LATECOERE.png',
            'company_name': 'Powered by Databricks',
        },
    }


@router.get('/me')
async def get_me(request: Request):
    """Return current user info."""
    try:
        user = await get_current_user(request)
        return {'user': user, 'workspace_url': get_workspace_url()}
    except Exception as e:
        logger.error(f'Error getting user info: {e}')
        return JSONResponse({'error': str(e)}, status_code=500)
