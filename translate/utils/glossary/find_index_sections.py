"""Prototype glossaire #3 — repérer les sections index / lexique / glossaire
dans le corpus.

Certains documents contiennent une section de type index ou lexique : des
entrées terminologiques à fort signal, directement exploitables pour le
glossaire (même workflow de vérification humaine que les paires multilingues).

    python -m utils.glossary.find_index_sections
    python -m utils.glossary.find_index_sections --terms "lexique,abréviations"

Recherche par LIKE sur les en-têtes sémantiques et le texte des chunks — pour
la variante par similarité vectorielle, interroger l'index chunks_index_v2 avec
les mêmes libellés (voir server/services/vector_search.py).
Coût : une requête SQL (aucun LLM).
"""

import argparse

from ._sql import CHUNKS_TABLE, run_sql

DEFAULT_TERMS = [
    'lexique', 'glossaire', 'glossary', 'définitions', 'definitions',
    'abréviations', 'abreviations', 'abbreviations', 'acronymes', 'acronyms',
    'terminologie', 'terminology', 'index des termes',
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--terms', help='libellés à chercher, séparés par des virgules')
    ap.add_argument('--limit', type=int, default=40)
    args = ap.parse_args()

    terms = [t.strip().lower() for t in (args.terms.split(',') if args.terms else DEFAULT_TERMS) if t.strip()]
    conds = ' OR '.join(
        f"lower(semantic_headers) LIKE '%{t}%'" for t in terms
    )
    rows = run_sql(f"""
        SELECT REF, division, semantic_headers, substring(chunk_text, 1, 200) AS excerpt
        FROM {CHUNKS_TABLE}
        WHERE {conds}
        ORDER BY REF
        LIMIT {args.limit}
    """)
    print(f'{len(rows)} chunk(s) avec une section index/lexique (en-têtes) :')
    seen_refs = set()
    for ref, division, headers, excerpt in rows:
        marker = '' if ref in seen_refs else ' *'
        seen_refs.add(ref)
        print(f'  [{division}] {ref}{marker}')
        print(f'      headers: {str(headers)[:100]}')
        print(f'      texte  : {str(excerpt)[:100]!r}')
    print(f'\n{len(seen_refs)} document(s) distincts — candidats à une extraction d entrées de lexique.')


if __name__ == '__main__':
    main()
