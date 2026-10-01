"""Shortlist de revue — les candidats glossaire les plus sûrs, tous lots confondus.

~9 000 candidats status=pending répartis sur 10 CSV rendent la revue humaine
décourageante. Ce script fusionne les review_batch_*.csv en UNE shortlist
triée par confiance décroissante, pour une première passe de revue efficace :

  priorité 1 — confirmés par >=2 documents ET définition corroborée
               (definition_source=cross_lang_pair)
  priorité 2 — définition corroborée seule
  priorité 3 — confirmés par >=N documents (défaut 2)

Le reste (vu 1 fois, sans définition authentique) reste dans les lots
d'origine pour une passe ultérieure. La revue se fait dans la colonne
`status` (pending → approved/rejected) puis se reporte dans les CSV
d'origine ou s'importe directement :

    python -m utils.glossary.make_review_shortlist
    python -m utils.glossary.make_review_shortlist --min-docs 3
    python -m utils.glossary.import_reviewed_candidates Translator/glossary/candidates/review_shortlist.csv --env UAT-TEST
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

LANGS = ('fr', 'en', 'cs', 'bg', 'de', 'es')
CANDIDATES_DIR = Path(__file__).resolve().parent.parent.parent / 'Translator' / 'glossary' / 'candidates'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--min-docs', type=int, default=2,
                    help='seuil de confirmation multi-documents (défaut : 2)')
    ap.add_argument('--out', type=Path, default=CANDIDATES_DIR / 'review_shortlist.csv')
    args = ap.parse_args()

    rows: list[dict] = []
    for path in sorted(CANDIDATES_DIR.glob('review_batch_*.csv')):
        if path.stem.endswith('_crosslang') or path.name == args.out.name:
            continue
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter=';')
            lang_cols = [c for c in (reader.fieldnames or []) if c in LANGS]
            for r in reader:
                if (r.get('status') or '').strip().lower() != 'pending':
                    continue
                try:
                    n_docs = int(r.get('n_docs') or 0)
                except ValueError:
                    n_docs = 0
                cross = (r.get('definition_source') or '').strip() == 'cross_lang_pair'
                if cross and n_docs >= args.min_docs:
                    prio = 1
                elif cross:
                    prio = 2
                elif n_docs >= args.min_docs:
                    prio = 3
                else:
                    continue
                rows.append({
                    'priority': prio,
                    'pair': '-'.join(lang_cols),
                    **{c: r.get(c, '') for c in LANGS if c in lang_cols},
                    **{c: '' for c in LANGS if c not in lang_cols},
                    'n_docs': n_docs,
                    'definition': r.get('definition', ''),
                    'definition_source': r.get('definition_source', ''),
                    'sources': r.get('sources', ''),
                    'status': 'pending',
                })

    # Dédoublonner un même couple apparu dans plusieurs lots (garde la
    # meilleure priorité / le plus de docs).
    best: dict[tuple, dict] = {}
    for r in rows:
        key = tuple((r[c] or '').strip().lower() for c in LANGS)
        cur = best.get(key)
        if cur is None or (r['priority'], -r['n_docs']) < (cur['priority'], -cur['n_docs']):
            best[key] = r
    final = sorted(best.values(), key=lambda r: (r['priority'], -r['n_docs'],
                                                 (r['fr'] or r['en']).lower()))

    fieldnames = ['priority', 'pair', *LANGS, 'n_docs', 'definition',
                  'definition_source', 'sources', 'status']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        w.writeheader()
        w.writerows(final)

    from collections import Counter
    counts = Counter(r['priority'] for r in final)
    print(f'{len(final)} candidat(s) dans la shortlist -> {args.out}')
    print(f"  P1 (>= {args.min_docs} docs + définition corroborée) : {counts.get(1, 0)}")
    print(f'  P2 (définition corroborée seule)                 : {counts.get(2, 0)}')
    print(f'  P3 (>= {args.min_docs} docs)                            : {counts.get(3, 0)}')


if __name__ == '__main__':
    main()
