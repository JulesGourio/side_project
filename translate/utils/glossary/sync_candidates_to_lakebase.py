"""Publication des candidats glossaire -> `glossary_terms` (Lakebase), avec
verif LLM au lieu d'une revue humaine.

Ces paires de termes viennent de traductions professionnelles reelles (deux
versions officielles du meme document LATECOERE, alignees section par
section) : on part du principe que ce sont deja de bonnes traductions, pas
des candidats a valider un par un a la main. Le seul risque vient de
l'extraction elle-meme (alignement de chunk approximatif, LLM d'extraction
qui se trompe de paire, ou qui garde la casse ALL-CAPS d'un titre de section
comme si c'etait un terme) - d'ou une passe de verification LLM juste avant
publication, qui rejette les artefacts (titres/fragments de phrase/paires
non correspondantes) et corrige la casse (minuscule pour un nom commun,
casse d'origine gardee seulement pour les noms propres/acronymes).

1. lit un ou plusieurs CSV de candidats (delimiteur ';', memes colonnes que
   `review_shortlist.csv` : priority/pair/fr/en/cs/bg/de/es/n_docs/
   definition/definition_source/sources/status - defaut : review_shortlist.csv),
2. dedoublonne (meme normalisation `_norm` que build_review_batch.py) contre
   ce qui est deja dans `glossary_terms` en Lakebase (le vrai glossaire
   vivant - PAS Translator/glossary/terms.csv, qui est reste un gabarit
   obsolete depuis que les seeds vont directement en Lakebase),
3. verifie chaque paire restante via LLM (par lot, groupe par paire de
   langues) - normalise la casse, rejette ce qui n'est pas un terme
   autonome valide,
4. insere directement dans `glossary_terms` (term_id auto T####) ce qui
   passe la verification. Pas de detour par `glossary_candidates` : cette
   table de revue humaine reste disponible pour l'ajout manuel depuis le
   panel, mais n'est plus alimentee par ce script.

Par defaut en DRY-RUN (pas d'appel LLM, juste le compte apres dedup) -
passer --apply pour verifier et ecrire reellement.

    python -m utils.glossary.sync_candidates_to_lakebase --env UAT-TEST
    python -m utils.glossary.sync_candidates_to_lakebase --env UAT-TEST --apply
    python -m utils.glossary.sync_candidates_to_lakebase Translator/glossary/candidates/review_batch_fr-en.csv --env UAT-TEST --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'utils' / 'databricks_ops' / 'lakebase_sync'))

from .build_review_batch import _norm  # même normalisation que la dédup amont
from server.services.translation.glossary_verify import llm_verify  # noqa: F401 (re-exported for CLI usage below)

LANGS = ('en', 'fr', 'cs', 'bg', 'de', 'es')
DEFAULT_CSV = REPO_ROOT / 'Translator' / 'glossary' / 'candidates' / 'review_shortlist.csv'


def _databricks_auth(endpoint: str) -> tuple[str, str, str]:
    env_host, env_token = os.environ.get('DATABRICKS_HOST'), os.environ.get('DATABRICKS_TOKEN')
    if env_host and env_token:
        return endpoint, env_host.rstrip('/'), env_token
    host = 'https://dbc-3a17bfce-9e88.cloud.databricks.com'
    token = json.loads(subprocess.check_output(
        ['databricks', 'auth', 'token', '-p', 'UAT'], text=True, encoding='utf-8'))['access_token']
    return endpoint, host, token


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            fields = [c.strip().lower() for c in (reader.fieldnames or [])]
            lang_cols = [c for c in fields if c in LANGS]
            if not lang_cols:
                print(f'  {path.name}: aucune colonne de langue ({fields}) — ignoré')
                continue
            n = 0
            for raw in reader:
                row = {(k or '').strip().lower(): (v or '').strip() for k, v in raw.items()}
                terms = {c: row[c] for c in lang_cols if row.get(c)}
                if not terms:
                    continue
                n += 1
                try:
                    priority = int(row['priority']) if row.get('priority') else None
                except ValueError:
                    priority = None
                try:
                    n_docs = int(row.get('n_docs') or 0)
                except ValueError:
                    n_docs = 0
                rows.append({
                    **{c: terms.get(c, '') for c in LANGS},
                    'n_docs': n_docs,
                    'sources': row.get('sources') or row.get('source') or '',
                    'definition': row.get('definition', ''),
                    'definition_source': row.get('definition_source', ''),
                    'priority': priority,
                    'status': (row.get('status') or 'pending').lower() or 'pending',
                })
            print(f'  {path.name}: {n} ligne(s) avec au moins un terme')
    return rows


def main() -> None:
    # Console Windows en cp1252 : les caractères type flèche/tiret cadratin du
    # docstring/résumé planteraient sur --help ou en fin de run (même bug que
    # build_review_batch.py, constaté 2026-07-17) — reconfigurer plutôt que
    # perdre l'affichage.
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csvs', nargs='*', type=Path, default=[DEFAULT_CSV],
                    help=f'CSV de candidats (défaut : {DEFAULT_CSV.relative_to(REPO_ROOT)})')
    ap.add_argument('--env', choices=['DEV', 'UAT', 'UAT-TEST'], default='UAT-TEST',
                    help='Environnement Lakebase cible (défaut : UAT-TEST, la base jetable)')
    ap.add_argument('--apply', action='store_true', help="écrire réellement (défaut : dry-run)")
    ap.add_argument('--endpoint', default=os.getenv('GLOSSARY_EXTRACT_ENDPOINT', 'databricks-gpt-5-4-mini'))
    ap.add_argument('--no-llm-verify', action='store_true',
                    help='publier sans passer par la verification LLM (deconseille - depannage seulement)')
    args = ap.parse_args()

    rows = load_rows(args.csvs)
    if not rows:
        print('Rien à publier.')
        return

    from migrate_lakebase import sdk_lakebase_connect
    print(f'Connexion Lakebase {args.env}…')
    conn = sdk_lakebase_connect(args.env)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT en, fr, cs, bg, de, es FROM glossary_terms')
            known = {_norm(v) for row in cur.fetchall() for v in row if v and v.strip()}
            # Also dedupe against glossary_candidates: rows already sitting there
            # from an older run of this script (back when it published to the
            # pending-review queue) must not be republished as new terms now
            # that this script inserts straight into glossary_terms.
            cur.execute('SELECT en, fr, cs, bg, de, es FROM glossary_candidates')
            known |= {_norm(v) for row in cur.fetchall() for v in row if v and v.strip()}

        to_insert, skipped = [], 0
        for r in rows:
            if any(_norm(r[l]) in known for l in LANGS if r[l]):
                skipped += 1
                continue
            for l in LANGS:
                if r[l]:
                    known.add(_norm(r[l]))
            to_insert.append(r)

        print(f'{len(rows)} lignes lues, {skipped} déjà au glossaire ou en attente de revue, '
              f'{len(to_insert)} à publier')
        for r in to_insert[:15]:
            langs = ' | '.join(f'{l}={r[l]}' for l in LANGS if r[l])
            print(f"  [P{r['priority']} n_docs={r['n_docs']}] {langs}")
        if len(to_insert) > 15:
            print(f'  … et {len(to_insert) - 15} de plus')

        if not args.apply:
            print('\nDRY-RUN — rien vérifié ni écrit. Relancer avec --apply pour publier.')
            return

        if args.no_llm_verify:
            print('--no-llm-verify : aucune vérification, publication telle quelle.')
            verified = to_insert
        else:
            by_pair: dict[tuple, list[dict]] = {}
            for r in to_insert:
                present = tuple(sorted(l for l in LANGS if r[l]))
                by_pair.setdefault(present, []).append(r)
            endpoint, host, token = _databricks_auth(args.endpoint)
            verified = []
            for present, group in by_pair.items():
                if len(present) != 2:
                    print(f'  ! {len(group)} ligne(s) avec {len(present)} langue(s) ({present}) — '
                          f'gardées sans vérification (paire attendue)')
                    verified.extend(group)
                    continue
                lang_a, lang_b = present
                print(f'Vérification LLM {lang_a}/{lang_b} : {len(group)} candidat(s)…')
                ok = llm_verify(group, lang_a, lang_b, endpoint, host, token)
                print(f'  -> {len(ok)}/{len(group)} gardé(s) après vérification')
                verified.extend(ok)

            # Une normalisation de casse peut faire converger deux lignes du même
            # lot vers le même terme — filtrer les doublons introduits ici, en
            # plus de ceux déjà écartés contre le glossaire existant.
            seen_now: set = set()
            deduped = []
            for r in verified:
                keys = {_norm(r[l]) for l in LANGS if r[l]}
                if keys & seen_now:
                    continue
                seen_now |= keys
                deduped.append(r)
            verified = deduped

        with conn.cursor() as cur:
            cur.execute(
                r"SELECT COALESCE(MAX(SUBSTRING(term_id FROM 2)::int), 0) FROM glossary_terms WHERE term_id ~ '^T\d+$'"
            )
            next_id = (cur.fetchone()[0] or 0) + 1
            for r in verified:
                term_id = f'T{next_id:04d}'
                next_id += 1
                cur.execute(
                    '''
                    INSERT INTO glossary_terms (term_id, en, fr, cs, bg, de, es, definition, definition_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''',
                    (term_id, *[r[l] for l in LANGS], r['definition'], r['definition_source']),
                )
        conn.commit()
        print(f'Publié {len(verified)} terme(s) dans glossary_terms ({args.env}).')
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()
