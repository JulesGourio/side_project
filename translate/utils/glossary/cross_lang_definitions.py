"""Corroboration inter-langues des définitions (definition_source=cross_lang_pair).

Idée de Jules (2026-07-16, voir docs/ROADMAP.md § glossaire) : une définition
AUTHENTIQUE extraite d'une section terminologie dans UNE langue
(extract_index_terms.py) + la paire de traduction confirmée du même terme
dans une autre famille (build_review_batch.py) suffisent quasiment toujours —
la traduction professionnelle en contexte désambiguïse le sens, la définition
n'a besoin d'être authentique que d'un seul côté.

Ce script croise les deux gisements avec la même normalisation `_norm` que la
dédup amont :

- pour chaque ligne d'un review_batch dont la définition est vide ou
  seulement 'generated' (inventée par le LLM, confiance faible),
- cherche le même terme (n'importe quelle langue de la paire) dans les
  entrées index_terms_*.csv qui portent une définition extraite du corpus,
- si trouvé → definition = celle du corpus, definition_source=cross_lang_pair
  (confiance forte, mais la ligne RESTE status=pending — revue humaine
  toujours requise, cohérent avec le reste du glossaire).

    python -m utils.glossary.cross_lang_definitions            # défauts globaux
    python -m utils.glossary.cross_lang_definitions --in-place
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .build_review_batch import _norm

LANGS = ('fr', 'en', 'cs', 'bg', 'de', 'es')
GLOSSARY_DIR = Path(__file__).resolve().parent
CANDIDATES_DIR = GLOSSARY_DIR.parent.parent / 'Translator' / 'glossary' / 'candidates'


def load_index_definitions(paths: list[Path]) -> dict[str, dict]:
    """_norm(term) -> {'definition', 'source'} pour chaque terme d'une entrée
    index qui porte une définition non vide."""
    defs: dict[str, dict] = {}
    for path in paths:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f, delimiter=';'):
                definition = (row.get('definition') or '').strip()
                if not definition:
                    continue
                for lang in LANGS:
                    term = (row.get(lang) or '').strip()
                    if term:
                        # premier arrivé gagne : les fichiers passés en premier
                        # (ou les entrées les plus hautes) priment
                        defs.setdefault(_norm(term), {
                            'definition': definition,
                            'source': (row.get('source') or path.stem),
                        })
    return defs


def enrich_batch(path: Path, index_defs: dict[str, dict], in_place: bool) -> tuple[int, Path]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    lang_cols = [c for c in fieldnames if c.lower() in LANGS]

    enriched = 0
    for row in rows:
        source_kind = (row.get('definition_source') or '').strip().lower()
        if (row.get('definition') or '').strip() and source_kind not in ('', 'generated'):
            continue  # déjà une définition authentique/corroborée
        for col in lang_cols:
            hit = index_defs.get(_norm(row.get(col) or ''))
            if hit:
                row['definition'] = hit['definition']
                row['definition_source'] = 'cross_lang_pair'
                enriched += 1
                break

    out = path if in_place else path.with_name(f'{path.stem}_crosslang{path.suffix}')
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        w.writeheader()
        w.writerows(rows)
    return enriched, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', nargs='*', type=Path,
                    help='CSV index_terms_*.csv (défaut : utils/glossary/ + candidates/)')
    ap.add_argument('--batch', nargs='*', type=Path,
                    help='CSV review_batch_*.csv à enrichir (défaut : candidates/review_batch_*.csv)')
    ap.add_argument('--in-place', action='store_true',
                    help='réécrire le batch en place (défaut : *_crosslang.csv à côté)')
    args = ap.parse_args()

    index_paths = args.index or (
        sorted(GLOSSARY_DIR.glob('index_terms_*.csv'))
        + sorted(CANDIDATES_DIR.glob('index_terms_*.csv'))
    )
    batch_paths = args.batch or sorted(
        p for p in CANDIDATES_DIR.glob('review_batch_*.csv')
        if not p.stem.endswith('_crosslang')
    )
    if not index_paths:
        print('Aucun index_terms_*.csv trouvé — lancer extract_index_terms d\'abord.')
        return
    if not batch_paths:
        print('Aucun review_batch_*.csv trouvé — lancer build_review_batch d\'abord.')
        return

    index_defs = load_index_definitions(index_paths)
    print(f'{len(index_defs)} terme(s) avec définition authentique dans {len(index_paths)} fichier(s) index')

    for path in batch_paths:
        enriched, out = enrich_batch(path, index_defs, args.in_place)
        print(f'  {path.name}: {enriched} définition(s) corroborée(s) -> {out.name}')


if __name__ == '__main__':
    main()
