"""Extraction en masse des sections terminologie du corpus.

Enchaîne extract_index_terms sur tous les documents repérés par la recherche
d'en-têtes (find_index_sections) — le « gisement #2 » du glossaire. Reprend
là où il s'est arrêté (un CSV par REF déjà produit = sauté), donc relançable
après une coupure sans coût LLM supplémentaire.

    python -m utils.glossary.extract_index_terms_bulk
    python -m utils.glossary.extract_index_terms_bulk --limit 20 --workers 3

Sorties : Translator/glossary/candidates/index_terms/index_terms_<REF>.csv
(status=pending, politique inchangée : revue humaine avant import) + un
merge index_terms_all.csv pour cross_lang_definitions/import.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ._sql import CHUNKS_TABLE, run_sql
from .extract_index_terms import (_dedupe_identical_langs, _doc_lang,
                                  _HEADER_TERMS, llm_entries,
                                  terminology_chunks)

LANGS = ('fr', 'en', 'cs', 'bg', 'de', 'es')
OUT_DIR = Path(__file__).resolve().parent.parent.parent / 'Translator' / 'glossary' / 'candidates' / 'index_terms'


def refs_with_terminology_sections() -> list[str]:
    conds = ' OR '.join(f"lower(semantic_headers) LIKE '%{t}%'" for t in _HEADER_TERMS)
    rows = run_sql(f'SELECT DISTINCT REF FROM {CHUNKS_TABLE} WHERE {conds} ORDER BY REF')
    return [r[0] for r in rows]


def _safe(ref: str) -> str:
    return ref.replace('/', '_').replace('\\', '_').replace(':', '_')


def process_ref(ref: str, endpoint: str, host: str, token: str,
                max_chunks: int, empty_tolerance: int) -> tuple[str, int, str]:
    """Returns (ref, n_entries, status)."""
    out = OUT_DIR / f'index_terms_{_safe(ref)}.csv'
    if out.exists():
        return ref, -1, 'skipped (déjà extrait)'
    try:
        chunks = terminology_chunks(ref, empty_tolerance=empty_tolerance)
        doc_lang = _doc_lang(ref)
        entries: list[dict] = []
        seen: set = set()
        for c in chunks[:max_chunks]:
            for e in llm_entries(c['text'], endpoint, host, token):
                e = _dedupe_identical_langs(e, doc_lang)
                key = tuple((e.get(k) or '').strip().lower() for k in LANGS)
                if any(key) and key not in seen:
                    seen.add(key)
                    e['source'] = f"{ref}#chunk{c['idx']}"
                    entries.append(e)
        with out.open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow([*LANGS, 'definition', 'source', 'status'])
            for e in entries:
                w.writerow([*[e.get(k, '') for k in LANGS],
                            (e.get('definition') or '')[:300], e['source'], 'pending'])
        return ref, len(entries), 'ok'
    except Exception as exc:  # famille perdue = visible, pas silencieuse
        return ref, 0, f'ERREUR: {exc}'


def merge_all() -> Path:
    merged = OUT_DIR / 'index_terms_all.csv'
    rows = []
    for path in sorted(OUT_DIR.glob('index_terms_*.csv')):
        if path.name == merged.name:
            continue
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            rows.extend(csv.DictReader(f, delimiter=';'))
    with merged.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=[*LANGS, 'definition', 'source', 'status'], delimiter=';')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})
    return merged


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--endpoint', default=os.getenv('GLOSSARY_EXTRACT_ENDPOINT', 'databricks-gpt-5-4-mini'))
    ap.add_argument('--max-chunks', type=int, default=4)
    ap.add_argument('--empty-tolerance', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='ne traiter que N documents (0 = tous)')
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    refs = refs_with_terminology_sections()
    if args.limit:
        refs = refs[:args.limit]
    print(f'{len(refs)} document(s) avec section terminologie')

    env_host, env_token = os.environ.get('DATABRICKS_HOST'), os.environ.get('DATABRICKS_TOKEN')
    if env_host and env_token:
        host, token = env_host.rstrip('/'), env_token
    else:
        host = 'https://dbc-3a17bfce-9e88.cloud.databricks.com'
        token = json.loads(subprocess.check_output(
            ['databricks', 'auth', 'token', '-p', 'UAT'], text=True, encoding='utf-8'))['access_token']

    total = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_ref, r, args.endpoint, host, token,
                          args.max_chunks, args.empty_tolerance) for r in refs]
        for i, fut in enumerate(as_completed(futs), 1):
            ref, n, status = fut.result()
            if status.startswith('ERREUR'):
                errors += 1
                print(f'  ! {ref}: {status}')
            elif n >= 0:
                total += n
            if i % 20 == 0:
                print(f'  {i}/{len(refs)} documents — {total} entrées')

    merged = merge_all()
    print(f'{total} nouvelle(s) entrée(s), {errors} erreur(s) -> {merged}')
    if errors:
        print('Relancer la même commande pour retenter les documents en erreur (reprise automatique).')


if __name__ == '__main__':
    main()
