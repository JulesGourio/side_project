"""Shared Lakebase glossary helpers used by both the per-job web pipeline
(server/routers/translate.py) and the offline corpus-sync pipeline
(utils/glossary/sync_candidates_to_lakebase.py). Distinct from glossary_io.py,
which is CSV-file I/O for the legacy prototype path.
"""

TERM_ID_LOCK_KEY = "glossary_terms.term_id"


async def generate_term_id(conn) -> str:
    """Atomically allocate the next 'T####' glossary_terms id.

    Must be called inside an open conn.transaction() block — the transaction-
    scoped advisory lock (pg_advisory_xact_lock) it takes is what serializes
    concurrent callers against the same MAX(term_id)+1 read; taken outside a
    transaction it would release before the caller's INSERT runs, and two
    concurrent requests could allocate the same id.
    """
    await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", TERM_ID_LOCK_KEY)
    max_id = await conn.fetchval(
        r"SELECT COALESCE(MAX(SUBSTRING(term_id FROM 2)::int), 0) FROM glossary_terms WHERE term_id ~ '^T\d+$'"
    )
    return f'T{(max_id or 0) + 1:04d}'
