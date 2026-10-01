"""Shared helper: run SQL statements against the UAT warehouse via the
Databricks CLI (statements API). Used by the glossary prototype scripts."""

import json
import subprocess
import time

DEFAULT_WAREHOUSE = '5890912c31867b77'  # Serverless Starter Warehouse (UAT)
DEFAULT_PROFILE = 'UAT'
CHUNKS_TABLE = 'uat_landingzone.qualibot.chunks_v2'


def _cli_json(args: list[str]) -> dict:
    # encoding explicite : sous Windows, text=True décode en cp1252 et casse
    # sur les réponses UTF-8 de l'API.
    return json.loads(subprocess.check_output(args, text=True, encoding='utf-8'))


def run_sql(statement: str, warehouse: str = DEFAULT_WAREHOUSE,
            profile: str = DEFAULT_PROFILE) -> list[list]:
    payload = json.dumps({'warehouse_id': warehouse, 'statement': statement, 'wait_timeout': '50s'})
    d = _cli_json(['databricks', 'api', 'post', '/api/2.0/sql/statements', '-p', profile, '--json', payload])
    while d.get('status', {}).get('state') in ('PENDING', 'RUNNING'):
        time.sleep(3)
        d = _cli_json(['databricks', 'api', 'get', f"/api/2.0/sql/statements/{d['statement_id']}", '-p', profile])
    state = d.get('status', {}).get('state')
    if state != 'SUCCEEDED':
        raise RuntimeError(f'SQL failed ({state}): {d.get("status")}')
    rows = d.get('result', {}).get('data_array', []) or []
    # Follow chunked result pagination if present
    next_idx = d.get('result', {}).get('next_chunk_index')
    while next_idx is not None:
        c = _cli_json(['databricks', 'api', 'get',
                       f"/api/2.0/sql/statements/{d['statement_id']}/result/chunks/{next_idx}", '-p', profile])
        rows.extend(c.get('data_array', []) or [])
        next_idx = c.get('next_chunk_index')
    return rows


# REF language-suffix conventions observed in the corpus (GB/UK/US → en).
# CZ is the corpus's actual Czech suffix (1546 chunks) -- CS was assumed but
# doesn't occur; without CZ recognized here, every Czech-suffixed document
# silently drops out of find_families(), which undercounted cs/* families as
# zero (confirmed 2026-07-16: adding CZ raises cs/fr 0->17, cs/en 0->32,
# bg/cs 0->17, cs/es 0->3). Kept both CS and CZ in case either convention
# appears.
LANG_SUFFIX_TO_CODE = {
    'FR': 'fr', 'EN': 'en', 'GB': 'en', 'UK': 'en', 'US': 'en',
    'BG': 'bg', 'ES': 'es', 'CS': 'cs', 'CZ': 'cs', 'DE': 'de',
}
SUFFIX_RE = r'[_-](FR|GB|EN|DE|ES|CS|CZ|BG|US|UK)$'
