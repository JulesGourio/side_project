"""One-off migration: LLM-verify the existing backlog of `glossary_candidates`
(status='pending', published by an earlier run of sync_candidates_to_lakebase.py
before it was redesigned to skip the pending queue) and promote survivors
straight into `glossary_terms`.

This is a ONE-TIME cleanup of the initial Intraqual-corpus bulk import, not a
replacement for the standard glossary-growth mechanism. Future terms coming
from real, human-validated translation work should keep going through human
review (the panel's manual "add term" form, and/or whatever confirms terms
during an actual translation job) — this script only clears the backlog that
predates the corpus-import redesign, using the same llm_verify() check as
sync_candidates_to_lakebase.py (reject headings/fragments/mismatches,
normalize stray ALL-CAPS casing).

Kept candidates -> inserted into glossary_terms, candidate row marked
status='approved' (reviewed_by='llm-auto-verify'). Rejected candidates ->
status='rejected' with a reject_reason, left in glossary_candidates for audit
(not deleted).

Dry-run by default (just counts per language pair) — pass --apply to verify
and write for real.

    python -m utils.glossary.promote_pending_candidates --env UAT-TEST
    python -m utils.glossary.promote_pending_candidates --env UAT-TEST --apply
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'utils' / 'databricks_ops' / 'lakebase_sync'))

from .build_review_batch import _norm
from .sync_candidates_to_lakebase import LANGS, _databricks_auth, llm_verify


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', choices=['DEV', 'UAT', 'UAT-TEST'], default='UAT-TEST',
                    help='Environnement Lakebase cible (défaut : UAT-TEST, la base jetable)')
    ap.add_argument('--apply', action='store_true', help='vérifier et écrire réellement (défaut : dry-run)')
    ap.add_argument('--endpoint', default=os.getenv('GLOSSARY_EXTRACT_ENDPOINT', 'databricks-gpt-5-4-mini'))
    ap.add_argument('--status', default='pending', help='statut des candidats à traiter (défaut : pending)')
    args = ap.parse_args()

    from migrate_lakebase import sdk_lakebase_connect
    print(f'Connexion Lakebase {args.env}…')
    conn = sdk_lakebase_connect(args.env)
    try:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT id, en, fr, cs, bg, de, es, definition, definition_source '
                'FROM glossary_candidates WHERE status = %s ORDER BY id',
                (args.status,),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        print(f"{len(rows)} candidat(s) au statut '{args.status}' dans glossary_candidates.")
        if not rows:
            return

        by_pair: dict[tuple, list[dict]] = {}
        for r in rows:
            present = tuple(sorted(l for l in LANGS if r.get(l)))
            by_pair.setdefault(present, []).append(r)
        for present, group in by_pair.items():
            label = '/'.join(present) if len(present) == 2 else str(present)
            print(f'  {label}: {len(group)} candidat(s)')

        if not args.apply:
            print('\nDRY-RUN — rien vérifié ni écrit. Relancer avec --apply pour publier.')
            return

        endpoint, host, token = _databricks_auth(args.endpoint)

        with conn.cursor() as cur:
            cur.execute('SELECT en, fr, cs, bg, de, es FROM glossary_terms')
            known = {_norm(v) for row in cur.fetchall() for v in row if v and v.strip()}
            cur.execute(
                r"SELECT COALESCE(MAX(SUBSTRING(term_id FROM 2)::int), 0) FROM glossary_terms WHERE term_id ~ '^T\d+$'"
            )
            next_id = (cur.fetchone()[0] or 0) + 1

        n_kept = n_rejected_verify = n_rejected_dup = n_skipped_pair = 0
        for present, group in by_pair.items():
            if len(present) != 2:
                print(f'  ! {len(group)} ligne(s) avec {len(present)} langue(s) ({present}) — '
                      f'ignorées (paire attendue)')
                n_skipped_pair += len(group)
                continue
            lang_a, lang_b = present
            print(f'Vérification LLM {lang_a}/{lang_b} : {len(group)} candidat(s)…')
            kept = llm_verify(group, lang_a, lang_b, endpoint, host, token)
            kept_ids = {r['id'] for r in kept}
            n_rejected_verify += sum(1 for r in group if r['id'] not in kept_ids)
            print(f'  -> {len(kept)}/{len(group)} gardé(s) après vérification')

            with conn.cursor() as cur:
                for r in group:
                    if r['id'] in kept_ids:
                        continue
                    cur.execute(
                        "UPDATE glossary_candidates SET status = 'rejected', reviewed_by = %s, "
                        "reviewed_at = NOW(), reject_reason = %s WHERE id = %s",
                        ('llm-auto-verify', 'Automated verification: heading/fragment/mismatch', r['id']),
                    )
                for r in kept:
                    keys = {_norm(r[l]) for l in LANGS if r.get(l)}
                    if keys & known:
                        cur.execute(
                            "UPDATE glossary_candidates SET status = 'rejected', reviewed_by = %s, "
                            "reviewed_at = NOW(), reject_reason = %s WHERE id = %s",
                            ('llm-auto-verify', 'Duplicate of an existing glossary term', r['id']),
                        )
                        n_rejected_dup += 1
                        continue
                    known |= keys
                    term_id = f'T{next_id:04d}'
                    next_id += 1
                    cur.execute(
                        '''
                        INSERT INTO glossary_terms (term_id, en, fr, cs, bg, de, es, definition, definition_source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''',
                        (term_id, *[r.get(l, '') for l in LANGS], r.get('definition', ''), r.get('definition_source', '')),
                    )
                    cur.execute(
                        "UPDATE glossary_candidates SET status = 'approved', reviewed_by = %s, reviewed_at = NOW() "
                        "WHERE id = %s",
                        ('llm-auto-verify', r['id']),
                    )
                    n_kept += 1
        conn.commit()
        print(f'\nPublié {n_kept} terme(s) dans glossary_terms.')
        print(f'Rejetés : {n_rejected_verify} (vérification LLM), {n_rejected_dup} (doublon), '
              f'{n_skipped_pair} ignorés (paire de langues inattendue).')
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()
