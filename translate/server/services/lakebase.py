"""Lakebase (autoscaling PostgreSQL) service.

Manages an asyncpg connection pool backed by a Databricks Lakebase project.
A background task refreshes the OAuth token every 55 minutes (token lifetime ~1h).

LAKEBASE_PROJECT_ID points at an existing shared Lakebase Postgres project
(reused to avoid provisioning new compute), but LAKEBASE_DATABASE is this
app's own dedicated database ("latlang"/"latlang_test"). This app owns only
the translation_*/glossary_*/dnt_rules tables — no shared `users` capability
table (access is gated at the Databricks App level, not per-feature).

Required env vars:
  LAKEBASE_PROJECT_ID   — Lakebase project ID
  LAKEBASE_DATABASE     — PostgreSQL database name (default: "latlang")

Optional:
  LAKEBASE_BRANCH       — branch (default: "production")
  LAKEBASE_ENDPOINT     — endpoint (default: "primary")
"""

import asyncio
import logging
import os
import time
from typing import Optional

import asyncpg
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None
_refresh_task: Optional[asyncio.Task] = None

_DEFAULT_REFRESH_INTERVAL_S = 55 * 60  # 55 min — token lifetime is ~1h
_REFRESH_SAFETY_BUFFER_S = 5 * 60  # refresh this long before actual expiry
_MIN_REFRESH_INTERVAL_S = 60  # never refresh tighter than this

_CONNECT_TIMEOUT_S = float(os.getenv('LAKEBASE_CONNECT_TIMEOUT_S', '5'))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cfg() -> dict:
    return {
        'project_id': os.getenv('LAKEBASE_PROJECT_ID', ''),
        'branch': os.getenv('LAKEBASE_BRANCH', 'production'),
        'endpoint': os.getenv('LAKEBASE_ENDPOINT', 'primary'),
        'database': os.getenv('LAKEBASE_DATABASE', 'latlang'),
    }


def _get_host_and_token(
    project_id: str, branch: str, endpoint: str
) -> tuple[str, str, str, Optional[float]]:
    """Fetch host and OAuth token from Databricks SDK (sync, run in thread).

    Lakebase OAuth auth uses the literal username "token"; the identity is
    carried inside the OAuth token itself, not in the PostgreSQL user field.
    """
    w = WorkspaceClient()

    branch_path = f'projects/{project_id}/branches/{branch}'
    endpoint_path = f'{branch_path}/endpoints/{endpoint}'

    eps = list(w.postgres.list_endpoints(parent=branch_path))
    if not eps:
        raise RuntimeError(f'No endpoints found for {branch_path}')
    host = eps[0].status.hosts.host
    logger.debug(f'Lakebase host resolved: {host}')

    me = w.current_user.me()
    username = me.user_name or me.display_name or ''
    if not username:
        raise RuntimeError('Could not resolve current user identity for Lakebase auth')
    logger.debug(f'Lakebase connecting as: {username}')

    token: str | None = None
    expires_in_s: Optional[float] = None
    try:
        cred = w.postgres.generate_database_credential(endpoint=endpoint_path)
        token = cred.token or None
        if token:
            if cred.expire_time is not None:
                expires_in_s = cred.expire_time.ToSeconds() - time.time()
            logger.debug(
                f'generate_database_credential succeeded (token len={len(token)}, '
                f'expires_in={expires_in_s}s)'
            )
        else:
            logger.warning('generate_database_credential returned empty token — falling back to workspace token')
    except Exception as e:
        logger.warning(f'generate_database_credential failed ({e}) — falling back to workspace token')

    if not token:
        token = w.config.token
        if token:
            logger.info('Using workspace OAuth token for Lakebase auth (fallback, unknown expiry)')
        else:
            raise RuntimeError('No authentication token available for Lakebase')

    return host, username, token, expires_in_s


def _next_refresh_delay(expires_in_s: Optional[float]) -> float:
    if expires_in_s is None:
        return _DEFAULT_REFRESH_INTERVAL_S
    return max(expires_in_s - _REFRESH_SAFETY_BUFFER_S, _MIN_REFRESH_INTERVAL_S)


async def _build_pool(host: str, username: str, token: str, database: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        host=host,
        port=5432,
        database=database,
        user=username,
        password=token,
        ssl='require',
        min_size=1,
        max_size=10,
        server_settings={'timezone': 'UTC'},
        timeout=_CONNECT_TIMEOUT_S,
    )


async def _ensure_database(host: str, username: str, token: str, database: str) -> None:
    """Create the PostgreSQL database if it does not exist."""
    conn = await asyncpg.connect(
        host=host, port=5432, database='postgres',
        user=username, password=token, ssl='require',
        timeout=_CONNECT_TIMEOUT_S,
    )
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", database
        )
        if not exists:
            safe_db = database.replace('"', '""')
            await conn.execute(f'CREATE DATABASE "{safe_db}"')
            logger.info(f'Database "{database}" created')
        else:
            logger.debug(f'Database "{database}" already exists')
    finally:
        await conn.close()


async def _ensure_schema(pool: asyncpg.Pool) -> None:
    """Create/upgrade this app's own tables — never DROP TABLE or TRUNCATE.

    (translation_jobs / translation_llm_calls / translation_segments /
    translation_questions / translation_feedbacks / glossary_terms /
    glossary_candidates / dnt_rules / errors). No shared `users` capability
    table — access here is gated at the Databricks App level.
    """
    async with pool.acquire() as conn:
        # ── errors (application errors persisted for monitoring) ──────────────
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id              SERIAL PRIMARY KEY,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                user_id         TEXT,
                workspace_id    TEXT,
                endpoint        TEXT,
                error_type      TEXT,
                error_msg       TEXT,
                stack_trace     TEXT
            )
        ''')

        # ── translation_jobs (one row per translation job) ─────────────────────
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS translation_jobs (
                id                       SERIAL PRIMARY KEY,
                created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                user_id                  TEXT,
                workspace_id             TEXT,
                status                   TEXT NOT NULL DEFAULT 'uploaded',
                stage_progress           TEXT,
                source_lang              TEXT,
                target_lang              TEXT,
                original_filename        TEXT,
                input_volume_path        TEXT,
                output_volume_path       TEXT,
                segment_count            INTEGER,
                needs_translation_count  INTEGER,
                error_type               TEXT,
                error_msg               TEXT,
                worker_pid               INTEGER,
                worker_heartbeat         TIMESTAMPTZ,
                claimed_at               TIMESTAMPTZ,
                llm_call_count           INTEGER NOT NULL DEFAULT 0,
                total_input_tokens       INTEGER NOT NULL DEFAULT 0,
                total_output_tokens      INTEGER NOT NULL DEFAULT 0,
                total_cost_eur           DOUBLE PRECISION NOT NULL DEFAULT 0,
                notes                    TEXT,
                glossary_validated_at    TIMESTAMPTZ
            )
        ''')
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS translation_jobs_user_idx
            ON translation_jobs(user_id)
        ''')
        # share_token: set once the owner clicks "Share" — grants read-only
        # access to anyone with the link, independent of the row's own id.
        await conn.execute('''
            ALTER TABLE translation_jobs ADD COLUMN IF NOT EXISTS share_token TEXT
        ''')
        await conn.execute('''
            CREATE UNIQUE INDEX IF NOT EXISTS translation_jobs_share_token_idx
            ON translation_jobs(share_token) WHERE share_token IS NOT NULL
        ''')
        # selected_image_paths: JSON array of word/media/... filenames the
        # user opted into OCR+translating (see docx_images.py) — NULL/empty
        # means the feature wasn't used for this job.
        await conn.execute('''
            ALTER TABLE translation_jobs ADD COLUMN IF NOT EXISTS selected_image_paths TEXT
        ''')
        # page_filter: raw "3-10,15" spec the user typed at upload, kept for
        # display/audit — the resolved page set lives on each segment as
        # out_of_page_range, not re-parsed from this string at rebuild time.
        await conn.execute('''
            ALTER TABLE translation_jobs ADD COLUMN IF NOT EXISTS page_filter TEXT
        ''')

        # ── translation_llm_calls (audit: one row per LLM call in a translate job) ──
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS translation_llm_calls (
                id             SERIAL PRIMARY KEY,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                job_id         INTEGER NOT NULL REFERENCES translation_jobs(id) ON DELETE CASCADE,
                endpoint_name  TEXT,
                batch_size     INTEGER,
                input_tokens   INTEGER,
                output_tokens  INTEGER,
                cost_eur       DOUBLE PRECISION
            )
        ''')
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS translation_llm_calls_job_idx
            ON translation_llm_calls(job_id)
        ''')

        # ── translation_segments (one row per extracted/translated segment) ───
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS translation_segments (
                id                     SERIAL PRIMARY KEY,
                job_id                 INTEGER NOT NULL REFERENCES translation_jobs(id) ON DELETE CASCADE,
                seg_id                 TEXT NOT NULL,
                part                   TEXT,
                location_type          TEXT,
                xml_choice_path        TEXT,
                xml_fallback_path      TEXT,
                source_text            TEXT,
                detected_lang          TEXT,
                lang_confidence        DOUBLE PRECISION,
                pattern_type           TEXT,
                pair_id                TEXT,
                conflict_flag          BOOLEAN NOT NULL DEFAULT FALSE,
                conflict_detail        TEXT,
                dnt_tokens             TEXT,
                inline_split           TEXT,
                translated_text        TEXT,
                keep_as_is             BOOLEAN NOT NULL DEFAULT FALSE,
                answered_question_id   TEXT,
                UNIQUE (job_id, seg_id)
            )
        ''')
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS translation_segments_job_idx
            ON translation_segments(job_id)
        ''')
        # warning_dismissed: reviewer says "this is fine as-is" (e.g. a proper
        # noun/surname the residual-language check keeps flagging) without
        # forcing a translation — the rebuild stage excludes dismissed
        # seg_ids from residual_warnings on every subsequent run.
        await conn.execute('''
            ALTER TABLE translation_segments ADD COLUMN IF NOT EXISTS warning_dismissed BOOLEAN NOT NULL DEFAULT FALSE
        ''')
        # out_of_page_range: segment sits outside the job's requested page
        # filter (pages.py) — never sent to translation, rebuilt verbatim.
        await conn.execute('''
            ALTER TABLE translation_segments ADD COLUMN IF NOT EXISTS out_of_page_range BOOLEAN NOT NULL DEFAULT FALSE
        ''')

        # ── translation_questions (batched Q&A per job) ────────────────────────
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS translation_questions (
                id                SERIAL PRIMARY KEY,
                job_id            INTEGER NOT NULL REFERENCES translation_jobs(id) ON DELETE CASCADE,
                q_id              TEXT NOT NULL,
                seg_ids           TEXT,
                category          TEXT,
                question_text     TEXT,
                context           TEXT,
                suggested_answer  TEXT,
                answer            TEXT,
                answered_at       TIMESTAMPTZ,
                UNIQUE (job_id, q_id)
            )
        ''')
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS translation_questions_job_idx
            ON translation_questions(job_id)
        ''')

        # ── glossary_terms (one row per concept) ────────────────────────────────
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS glossary_terms (
                id                SERIAL PRIMARY KEY,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                term_id           TEXT UNIQUE NOT NULL,
                en                TEXT,
                fr                TEXT,
                cs                TEXT,
                bg                TEXT,
                de                TEXT,
                es                TEXT,
                pt                TEXT,
                ar                TEXT,
                domain            TEXT,
                notes             TEXT,
                definition        TEXT,
                definition_source TEXT
            )
        ''')
        # pt/ar: added after the initial six-language schema shipped — existing
        # deployed databases (latlang-uat, latlang-uat-test) only have the CREATE
        # TABLE's original columns, so a plain CREATE TABLE IF NOT EXISTS above is
        # a no-op on them; ALTER is what actually adds the columns there.
        await conn.execute('''
            ALTER TABLE glossary_terms ADD COLUMN IF NOT EXISTS pt TEXT
        ''')
        await conn.execute('''
            ALTER TABLE glossary_terms ADD COLUMN IF NOT EXISTS ar TEXT
        ''')

        # ── glossary_candidates (pending review queue) ──────────────────────────
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS glossary_candidates (
                id                SERIAL PRIMARY KEY,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                en                TEXT,
                fr                TEXT,
                cs                TEXT,
                bg                TEXT,
                de                TEXT,
                es                TEXT,
                pt                TEXT,
                ar                TEXT,
                n_docs            INTEGER NOT NULL DEFAULT 0,
                sources           TEXT,
                definition        TEXT,
                definition_source TEXT,
                priority          SMALLINT,
                status            TEXT NOT NULL DEFAULT 'pending',
                reviewed_by       TEXT,
                reviewed_at       TIMESTAMPTZ,
                reject_reason     TEXT
            )
        ''')
        await conn.execute('''
            ALTER TABLE glossary_candidates ADD COLUMN IF NOT EXISTS pt TEXT
        ''')
        await conn.execute('''
            ALTER TABLE glossary_candidates ADD COLUMN IF NOT EXISTS ar TEXT
        ''')
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS glossary_candidates_status_idx
            ON glossary_candidates(status, priority, n_docs DESC)
        ''')

        # ── translation_feedbacks (thumbs up/down + optional comment on a job) ──
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS translation_feedbacks (
                id                SERIAL PRIMARY KEY,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                job_id            INTEGER REFERENCES translation_jobs(id) ON DELETE SET NULL,
                user_id           TEXT,
                workspace_id      TEXT,
                workspace_url     TEXT,
                vote              TEXT NOT NULL CHECK (vote IN ('up', 'down')),
                comment           TEXT,
                resolved          BOOLEAN NOT NULL DEFAULT FALSE,
                resolution_reason TEXT
            )
        ''')
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS translation_feedbacks_job_idx
            ON translation_feedbacks(job_id)
        ''')

        # ── dnt_rules ────────────────────────────────────────────────────────────
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS dnt_rules (
                id          SERIAL PRIMARY KEY,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                pattern     TEXT NOT NULL,
                type        TEXT,
                match_mode  TEXT NOT NULL,
                notes       TEXT
            )
        ''')

    logger.info('Lakebase schema ready')


async def _token_refresh_loop(
    project_id: str, branch: str, endpoint: str, database: str, delay: float
) -> None:
    global _pool
    while True:
        await asyncio.sleep(delay)
        try:
            host, username, token, expires_in_s = await asyncio.to_thread(
                _get_host_and_token, project_id, branch, endpoint
            )
            new_pool = await _build_pool(host, username, token, database)
            old_pool = _pool
            _pool = new_pool
            if old_pool:
                await asyncio.sleep(30)
                await old_pool.close()
            delay = _next_refresh_delay(expires_in_s)
            logger.info(f'Lakebase token refreshed (next refresh in {delay:.0f}s)')
        except Exception as e:
            delay = 60
            logger.error(f'Lakebase token refresh failed (retrying in {delay}s): {e}')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def init_lakebase() -> None:
    global _pool, _refresh_task
    cfg = _cfg()
    project_id = cfg['project_id']
    if not project_id:
        logger.warning('LAKEBASE_PROJECT_ID not set — history feature disabled')
        return
    try:
        host, username, token, expires_in_s = await asyncio.to_thread(
            _get_host_and_token, project_id, cfg['branch'], cfg['endpoint']
        )
        await _ensure_database(host, username, token, cfg['database'])
        _pool = await _build_pool(host, username, token, cfg['database'])
        try:
            await _ensure_schema(_pool)
        except Exception as e:
            logger.error(f'Lakebase schema ensure failed — continuing with existing schema: {e}')
        _refresh_task = asyncio.create_task(
            _token_refresh_loop(
                project_id, cfg['branch'], cfg['endpoint'], cfg['database'],
                _next_refresh_delay(expires_in_s),
            )
        )
        logger.info(f'Lakebase ready ({host})')
    except Exception as e:
        logger.error(f'Lakebase init failed — history disabled: {e}')


async def shutdown_lakebase() -> None:
    global _pool, _refresh_task
    if _refresh_task:
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass
    if _pool:
        await _pool.close()
    logger.info('Lakebase shut down')


def get_pool() -> Optional[asyncpg.Pool]:
    return _pool


async def store_error(
    *,
    endpoint: str = '',
    error_type: str = '',
    error_msg: str = '',
    user_id: str = '',
    workspace_id: str = '',
    stack_trace: str = '',
) -> None:
    """Persist an application error to the errors table (best-effort, never raises)."""
    pool = get_pool()
    if not pool:
        return
    try:
        async with pool.acquire(timeout=5.0) as conn:
            await conn.execute(
                '''
                INSERT INTO errors (endpoint, error_type, error_msg, user_id, workspace_id, stack_trace)
                VALUES ($1, $2, $3, $4, $5, $6)
                ''',
                endpoint or None,
                error_type or None,
                (error_msg or '')[:2000],
                user_id or None,
                workspace_id or None,
                (stack_trace or '')[:4000] or None,
            )
    except Exception as e:
        logger.warning(f'store_error failed (best-effort, ignored): {e}')
