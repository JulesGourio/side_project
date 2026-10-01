"""Prototype glossaire #4 — extraire les entrées des sections terminologie.

Deuxième gisement (avec les familles multilingues) : les documents qui portent
directement une section TERMINOLOGIE / Definitions / List of Acronyms —
repérés par find_index_sections.py. Beaucoup sont bilingues (DLV tchèque/
anglais) : leurs entrées donnent des paires de termes prêtes pour le glossaire.

    python -m utils.glossary.extract_index_terms DLV-2012
    python -m utils.glossary.extract_index_terms 19--060 --out candidats.csv

Sortie : CSV status=pending (terme par langue détectée + définition en notes),
à relire avant import — jamais d'insertion directe.
"""

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

import httpx

from ._sql import CHUNKS_TABLE, LANG_SUFFIX_TO_CODE, SUFFIX_RE, run_sql

_HEADER_TERMS = ('terminolog', 'glossa', 'definition', 'définition', 'abbreviation',
                 'abréviation', 'abreviation', 'acronym', 'lexique')

_EXTRACT_PROMPT = """You extract terminology entries from the glossary/terminology \
section of a LATECOERE aerospace document. Entries may be monolingual \
(term + definition) or bilingual (e.g. Czech / English side by side).

Return ONLY JSON of the shape:
{"entries": [{"fr": "", "en": "", "cs": "", "bg": "", "de": "", "es": "", "definition": ""}, ...]}
Fill every language actually present for the entry, leave the others as "". \
NEVER copy the same string into two language fields: an acronym or \
language-neutral term goes in ONE field only — the language of the section \
it appears in ("en" if genuinely neutral). Only fill two languages when the \
document itself shows two different renderings side by side. No part \
numbers, no section numbers, no prose outside the JSON."""


def _all_chunks(ref: str) -> list[dict]:
    ref_sql = ref.replace("'", "''")
    rows = run_sql(f"""
        SELECT chunk_index, semantic_headers, chunk_text
        FROM {CHUNKS_TABLE}
        WHERE REF = '{ref_sql}'
        ORDER BY CAST(chunk_index AS INT)
    """)
    return [{'idx': int(r[0]), 'headers': r[1] or '', 'text': r[2] or ''} for r in rows]


def _is_empty_header(headers: str) -> bool:
    """semantic_headers is '{}' (or blank) on 22.5% of the corpus — no signal
    at all, usually a continuation of the previous section."""
    h = (headers or '').strip()
    if not h or h == '{}':
        return True
    try:
        return not any(str(v).strip() for v in json.loads(h).values())
    except (json.JSONDecodeError, AttributeError):
        return False


def terminology_chunks(ref: str, empty_tolerance: int = 2) -> list[dict]:
    """Chunks of a REF's terminology/glossary section(s).

    The LIKE-on-header seed only catches chunks whose OWN header matches —
    a long lexicon split across several chunks loses the matching header as
    soon as the chunker starts a new chunk (one header field per chunk,
    verified on MI-13601 2026-07-16). Walk forward from each seed: keep
    following chunks while their header is identical to the seed's OR empty
    ('{}', no signal — treated as a continuation), tolerating at most
    `empty_tolerance` consecutive empty-header chunks, and stop at the first
    DIFFERENT non-empty header. Best-effort by design — a section whose
    continuation runs past the tolerance is truncated, and everything stays
    status=pending for human review anyway.
    """
    chunks = _all_chunks(ref)
    seeds = [i for i, c in enumerate(chunks)
             if any(t in c['headers'].lower() for t in _HEADER_TERMS)]
    picked: dict[int, dict] = {}
    for i in seeds:
        picked[chunks[i]['idx']] = chunks[i]
        seed_header = chunks[i]['headers']
        empties = 0
        for c in chunks[i + 1:]:
            if c['headers'] == seed_header:
                empties = 0
            elif _is_empty_header(c['headers']):
                empties += 1
                if empties > empty_tolerance:
                    break
            else:
                break
            picked[c['idx']] = c
    return [picked[k] for k in sorted(picked)]


def llm_entries(text: str, endpoint: str, host: str, token: str) -> list[dict]:
    from .extract_glossary_candidates import _post_with_backoff
    r = _post_with_backoff(
        f'{host}/serving-endpoints/{endpoint}/invocations',
        {'Authorization': f'Bearer {token}'},
        {'messages': [{'role': 'system', 'content': _EXTRACT_PROMPT},
                      {'role': 'user', 'content': text[:6000]}],
         'max_tokens': 3000},
    )
    out = r.json().get('choices', [{}])[0].get('message', {}).get('content', '')
    m = re.search(r'\{[\s\S]*\}', out)
    if not m:
        return []
    try:
        entries = json.loads(m.group(0)).get('entries', [])
    except json.JSONDecodeError:
        return []
    return [e for e in entries if isinstance(e, dict) and any(e.get(k) for k in ('fr', 'en', 'cs', 'bg', 'de', 'es'))]


_LANGS = ('fr', 'en', 'cs', 'bg', 'de', 'es')


def _doc_lang(ref: str) -> str | None:
    """Language code from the REF's language suffix, when it has one."""
    m = re.search(SUFFIX_RE, ref.upper())
    return LANG_SUFFIX_TO_CODE.get(m.group(1)) if m else None


def _dedupe_identical_langs(entry: dict, doc_lang: str | None) -> dict:
    """Deterministic guard behind the prompt instruction: the LLM sometimes
    still copies one string (typically an acronym) into several language
    fields, which would fabricate a translation pair out of thin air. Keep
    the copy in the document's own language (or 'en' as the neutral
    fallback), blank the duplicates."""
    filled = [k for k in _LANGS if (entry.get(k) or '').strip()]
    by_value: dict[str, list[str]] = {}
    for k in filled:
        by_value.setdefault(entry[k].strip().lower(), []).append(k)
    for _value, langs in by_value.items():
        if len(langs) < 2:
            continue
        keep = doc_lang if doc_lang in langs else ('en' if 'en' in langs else langs[0])
        for k in langs:
            if k != keep:
                entry[k] = ''
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ref', help='REF du document (ex: DLV-2012)')
    ap.add_argument('--endpoint', default=os.getenv('GLOSSARY_EXTRACT_ENDPOINT', 'databricks-gpt-5-4-mini'))
    ap.add_argument('--max-chunks', type=int, default=4)
    ap.add_argument('--empty-tolerance', type=int, default=2,
                    help="chunks à header vide ('{}') tolérés d'affilée dans le walk-forward")
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()

    chunks = terminology_chunks(args.ref, empty_tolerance=args.empty_tolerance)
    if not chunks:
        print(f'{args.ref}: aucune section terminologie détectée dans les chunks')
        return
    print(f'{args.ref}: {len(chunks)} chunk(s) terminologie — en-têtes: {chunks[0]["headers"][:90]}')

    env_host, env_token = os.environ.get('DATABRICKS_HOST'), os.environ.get('DATABRICKS_TOKEN')
    if env_host and env_token:
        host, token = env_host.rstrip('/'), env_token
    else:
        host = 'https://dbc-3a17bfce-9e88.cloud.databricks.com'
        token = json.loads(subprocess.check_output(
            ['databricks', 'auth', 'token', '-p', 'UAT'], text=True, encoding='utf-8'))['access_token']

    doc_lang = _doc_lang(args.ref)
    entries: list[dict] = []
    seen: set = set()
    for c in chunks[:args.max_chunks]:
        for e in llm_entries(c['text'], args.endpoint, host, token):
            e = _dedupe_identical_langs(e, doc_lang)
            key = tuple((e.get(k) or '').strip().lower() for k in ('fr', 'en', 'cs', 'bg', 'de', 'es'))
            if any(key) and key not in seen:
                seen.add(key)
                e['source'] = f"{args.ref}#chunk{c['idx']}"
                entries.append(e)

    out = args.out or Path(__file__).parent / f'index_terms_{args.ref.replace("/", "_")}.csv'
    with out.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(['fr', 'en', 'cs', 'bg', 'de', 'es', 'definition', 'source', 'status'])
        for e in entries:
            w.writerow([e.get('fr', ''), e.get('en', ''), e.get('cs', ''), e.get('bg', ''),
                        e.get('de', ''), e.get('es', ''), (e.get('definition') or '')[:300],
                        e['source'], 'pending'])
    print(f'{len(entries)} entrées candidates -> {out} (status=pending)')
    for e in entries[:10]:
        langs = ' | '.join(f'{k}={e[k]}' for k in ('fr', 'en', 'cs', 'bg', 'de', 'es') if e.get(k))
        print(f'  {langs}' + (f'  [{(e.get("definition") or "")[:50]}]' if e.get('definition') else ''))


if __name__ == '__main__':
    main()
