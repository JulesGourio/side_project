"""FastAPI app — LatLang (bilingual .docx translation pipeline).

Single-purpose app: only the translate router is mounted, and there is no
MLflow tracing (translate.py has no MLflow dependency).
"""

import logging
import os
import time
import traceback
import uuid
from logging import Formatter
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from .routers import config, health, translate
from .services.lakebase import get_pool, init_lakebase, shutdown_lakebase, store_error

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  handlers=[logging.StreamHandler()],
)
Formatter.converter = time.gmtime  # force all log timestamps to UTC
logger = logging.getLogger(__name__)

env_local_loaded = load_dotenv(dotenv_path='.env.local')
env = os.getenv('ENV', 'development' if env_local_loaded else 'production')
logger.info(f'Starting in {env} mode')

_lakebase_id = os.getenv('LAKEBASE_PROJECT_ID', '')
_app_version = os.getenv('APP_VERSION', '1')
logger.info(f'LAKEBASE_PROJECT_ID = {_lakebase_id!r}')
logger.info(f'APP_VERSION         = {_app_version!r}')


async def _reconcile_orphaned_translation_jobs() -> None:
  """Mark as failed any translation job an earlier process was ACTIVELY
  processing when it died (e.g. an app restart)."""
  pool = get_pool()
  if not pool:
    return
  active = ', '.join(f"'{s}'" for s in translate.ACTIVE_PROCESSING_STATUSES)
  try:
    async with pool.acquire() as conn:
      result = await conn.execute(f'''
        UPDATE translation_jobs
        SET status = 'failed',
            error_type = 'OrphanedJob',
            error_msg = 'Interrupted by app restart',
            updated_at = NOW()
        WHERE status IN ({active})
          AND (worker_heartbeat IS NULL OR worker_heartbeat < NOW() - INTERVAL '2 minutes')
      ''')
    logger.info(f'Translation job reconciliation: {result}')
  except Exception as e:
    logger.warning(f'Translation job reconciliation failed: {e}')


@asynccontextmanager
async def lifespan(app: FastAPI):
  await init_lakebase()
  await _reconcile_orphaned_translation_jobs()
  yield
  await shutdown_lakebase()


app = FastAPI(title='LatLang', version='1.0.0', lifespan=lifespan)

allowed_origins = ['http://localhost:3000'] if env == 'development' else []
app.add_middleware(
  CORSMiddleware,
  allow_origins=allowed_origins,
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)


@app.middleware('http')
async def log_requests(request: Request, call_next):
  req_id = str(uuid.uuid4())[:8]
  start = time.monotonic()
  logger.info(f'[{req_id}] --> {request.method} {request.url.path}')
  try:
    response = await call_next(request)
    elapsed = time.monotonic() - start
    logger.info(f'[{req_id}] <-- {response.status_code} ({elapsed:.2f}s)')
    response.headers['X-Request-ID'] = req_id
    return response
  except Exception as exc:
    elapsed = time.monotonic() - start
    logger.error(f'[{req_id}] !! unhandled exception after {elapsed:.2f}s: {exc}', exc_info=True)
    raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
  logger.error(f'Unhandled exception on {request.method} {request.url.path}: {exc}', exc_info=True)
  import asyncio
  asyncio.create_task(store_error(
      endpoint=f'{request.method} {request.url.path}',
      error_type=type(exc).__name__,
      error_msg=str(exc),
      stack_trace=traceback.format_exc(),
  ))
  return JSONResponse(status_code=500, content={'error': 'Internal server error'})

API_PREFIX = '/api'
app.include_router(health.router, prefix=API_PREFIX, tags=['health'])
app.include_router(config.router, prefix=API_PREFIX, tags=['config'])
app.include_router(translate.router, prefix=API_PREFIX, tags=['translate'])

# Serve Vite static build in production
build_path = Path('.') / 'client/out'
if build_path.exists():
  logger.info(f'Serving static files from {build_path}')
  app.mount('/assets', StaticFiles(directory=str(build_path / 'assets')), name='assets')

  for static_dir in ['images', 'logos', 'videos', 'content']:
    dir_path = build_path / static_dir
    if dir_path.exists():
      app.mount(f'/{static_dir}', StaticFiles(directory=str(dir_path)), name=static_dir)

  @app.get('/{full_path:path}')
  async def serve_spa(request: Request, full_path: str):
    """SPA catch-all: serve index.html for any non-API route."""
    file_path = build_path / full_path
    if full_path and file_path.is_file():
      return FileResponse(str(file_path))
    return FileResponse(
      str(build_path / 'index.html'),
      headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
    )
else:
  logger.warning(
    f'Build directory {build_path} not found. '
    'Run: cd client && bun run build'
  )
