"""Importeur candidats revus → glossary_terms (Lakebase).

Comble le trou de schéma constaté le 2026-07-16 : les CSV de candidats
(`Translator/glossary/candidates/review_batch_*.csv`, `index_terms_*.csv`,
`candidates_*.csv`) n'avaient AUCUN chemin scripté vers `glossary_terms` —
même après revue humaine. Ce script :

1. lit un ou plusieurs CSV de candidats (délimiteur ';', colonnes de langue
   détectées parmi fr/en/cs/bg/de/es, + definition/definition_source),
2. ne retient QUE les lignes dont status marque une revue positive
   (approved/ok/validé/oui/yes — jamais 'pending' : la politique reste
   « aucune insertion sans revue humaine »),
3. saute les termes déjà présents dans glossary_terms (même normalisation
   que build_review_batch),
4. génère les term_id T#### à la suite du max existant et upsert.

Par défaut en DRY-RUN (montre ce qui serait importé) — passer --apply pour
écrire réellement.

    python -m utils.glossary.import_reviewed_candidates Translator/glossary/candidates/review_batch_fr-en.csv --env UAT --apply
    python -m utils.glossary.import_reviewed_candidates utils/glossary/index_terms_DLV-2012.csv --env UAT-TEST
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'utils' / 'databricks_ops' / 'lakebase_sync'))

from .build_review_batch import _norm  # même normalisation que la dédup amont

LANGS = ('en', 'fr', 'cs', 'bg', 'de', 'es')
APPROVED = {'approved', 'approve', 'ok', 'valid', 'valide', 'validé', 'validated', 'oui', 'yes'}


def load_candidates(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            fields = [c.strip().lower() for c in (reader.fieldnames or [])]
            lang_cols = [c for c in fields if c in LANGS]
            if not lang_cols:
                print(f'  {path.name}: aucune colonne de langue ({fields}) — ignoré')
                continue
            n_total = n_kept = 0
            for raw in reader:
                row = {(k or '').strip().lower(): (v or '').strip() for k, v in raw.items()}
                n_total += 1
                if row.get('status', '').lower() not in APPROVED:
                    continue
                terms = {c: row[c] for c in lang_cols if row.get(c)}
                if not terms:
                    continue
                n_kept += 1
                rows.append({
                    **{c: terms.get(c, '') for c in LANGS},
                    'definition': row.get('definition', ''),
                    'definition_source': row.get('definition_source', ''),
                    'notes': row.get('sources') or row.get('source') or '',
                    '_file': path.name,
                })
            print(f'  {path.name}: {n_kept}/{n_total} ligne(s) approuvée(s)')
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csvs', nargs='+', type=Path, help='CSV de candidats revus')
    ap.add_argument('--env', choices=['DEV', 'UAT', 'UAT-TEST'], default='UAT-TEST',
                    help='Environnement Lakebase cible (défaut : UAT-TEST, la base jetable)')
    ap.add_argument('--apply', action='store_true', help="écrire réellement (défaut : dry-run)")
    args = ap.parse_args()

    candidates = load_candidates([p for p in args.csvs])
    if not candidates:
        print('Aucune ligne approuvée — rien à importer (les lignes restent status=pending tant que la revue humaine n\'est pas faite).')
        return

    from migrate_lakebase import sdk_lakebase_connect
    print(f'Connexion Lakebase {args.env}…')
    conn = sdk_lakebase_connect(args.env)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT en, fr, cs, bg, de, es FROM glossary_terms')
            known = {_norm(v) for row in cur.fetchall() for v in row if v and v.strip()}
            cur.execute(r"SELECT COALESCE(MAX(SUBSTRING(term_id FROM 2)::int), 0) FROM glossary_terms WHERE term_id ~ '^T\d+$'")
            next_id = (cur.fetchone()[0] or 0) + 1

        to_import, skipped = [], 0
        for c in candidates:
            if any(_norm(c[l]) in known for l in LANGS if c[l]):
                skipped += 1
                continue
            for l in LANGS:
                if c[l]:
                    known.add(_norm(c[l]))
            c['term_id'] = f'T{next_id:04d}'
            next_id += 1
            to_import.append(c)

        print(f'{len(candidates)} approuvés, {skipped} déjà au glossaire, {len(to_import)} à importer')
        for c in to_import[:15]:
            langs = ' | '.join(f'{l}={c[l]}' for l in LANGS if c[l])
            print(f"  {c['term_id']}: {langs}"
                  + (f"  [déf: {c['definition'][:50]}… ({c['definition_source']})]" if c['definition'] else ''))
        if len(to_import) > 15:
            print(f'  … et {len(to_import) - 15} de plus')

        if not args.apply:
            print('\nDRY-RUN — rien écrit. Relancer avec --apply pour importer.')
            return

        with conn.cursor() as cur:
            for c in to_import:
                cur.execute(
                    '''
                    INSERT INTO glossary_terms (term_id, en, fr, cs, bg, de, es, domain, notes,
                                                definition, definition_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (term_id) DO UPDATE SET
                        en = EXCLUDED.en, fr = EXCLUDED.fr, cs = EXCLUDED.cs, bg = EXCLUDED.bg,
                        de = EXCLUDED.de, es = EXCLUDED.es, notes = EXCLUDED.notes,
                        definition = EXCLUDED.definition,
                        definition_source = EXCLUDED.definition_source, updated_at = NOW()
                    ''',
                    (c['term_id'], *[c[l] for l in LANGS], '', c['notes'],
                     c['definition'], c['definition_source']),
                )
        conn.commit()
        print(f'Importé {len(to_import)} terme(s) dans glossary_terms ({args.env}).')
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()
