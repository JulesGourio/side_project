"""Health check endpoint."""

import logging
import time

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('/health')
async def health_check():
  """Returns application health status."""
  return {
    'status': 'healthy',
    'timestamp': int(time.time() * 1000),
  }
