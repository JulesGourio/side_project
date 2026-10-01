"""Prototype glossaire #2 — extraction de termes candidats depuis une famille
de documents multilingues (même REF + suffixes de langue).

Pipeline en deux temps, pensé pour la VÉRIFICATION HUMAINE (jamais d'insertion
directe dans glossary_terms) :

1. Alignement déterministe (gratuit) : les chunks des deux versions sont
   appariés par position relative + ancres communes (références de pièces,
   normes, nombres — invariants entre langues). Sortie : paires de sections
   alignées + score de confiance d'alignement.
2. Extraction LLM (optionnelle, --llm) : chaque paire alignée passe dans un
   petit modèle (gemini-flash-lite par défaut — coût négligeable) qui extrait
   les correspondances de termes techniques. Sortie : CSV de candidats avec
   status=pending, à relire puis importer via l'API glossaire de l'app.

    python -m utils.glossary.extract_glossary_candidates IF20335 --langs fr en
    python -m utils.glossary.extract_glossary_candidates IF20335 --langs fr en --llm --max-sections 3
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

_ANCHOR_RE = re.compile(r'\b(?:[A-Z]{2,}[-_]?\d[\w.-]*|\d+(?:[.,]\d+)+|Ø\s?\d+[.,]?\d*)\b')

_EXTRACT_PROMPT = """You are a bilingual aerospace terminology extractor.
Given the SAME section of a LATECOERE work instruction in {lang_a} and {lang_b},
extract the technical term correspondences (tools, operations, components,
materials, defects). Only terms clearly present in BOTH texts. No part numbers,
no standards references, no sentences — noun phrases of 1-4 words.
Return ONLY JSON: {{"terms": [{{"{code_a}": "<term>", "{code_b}": "<term>"}}, ...]}}"""

_DEFINE_PROMPT = """You write one-sentence French definitions for an aerospace \
manufacturing glossary (LATECOERE). For each term pair below ({lang_a_name} / {lang_b_name}), \
write a short factual definition in French (max 25 words): what the thing/operation IS \
in an aerospace production context. No circular definitions, no examples, no "terme qui...".
Return ONLY JSON: {{"definitions": {{"<id>": "<définition>", ...}}}} — one entry per id."""


def llm_define(candidates: list[dict], lang_a: str, lang_b: str,
               endpoint: str, host: str, token: str, batch: int = 25) -> None:
    """Generate a short French definition for each candidate (marked
    definition_source=generated — reviewers can tell it apart from
    document-extracted definitions). Mutates candidates in place."""
    names = {'fr': 'French', 'en': 'English', 'bg': 'Bulgarian', 'es': 'Spanish', 'cs': 'Czech', 'de': 'German',
              'pt': 'Portuguese', 'ar': 'Arabic'}
    system = _DEFINE_PROMPT.format(lang_a_name=names[lang_a], lang_b_name=names[lang_b])
    for i in range(0, len(candidates), batch):
        chunk = candidates[i:i + batch]
        items = [{'id': str(j), lang_a: c[lang_a], lang_b: c[lang_b]} for j, c in enumerate(chunk)]
        r = _post_with_backoff(
            f'{host}/serving-endpoints/{endpoint}/invocations',
            {'Authorization': f'Bearer {token}'},
            {'messages': [{'role': 'system', 'content': system},
                          {'role': 'user', 'content': json.dumps({'terms': items}, ensure_ascii=False)}],
             'max_tokens': 3000},
        )
        out = r.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        m = re.search(r'\{[\s\S]*\}', out)
        defs = {}
        if m:
            try:
                defs = json.loads(m.group(0)).get('definitions', {})
            except json.JSONDecodeError:
                pass
        for j, c in enumerate(chunk):
            d = defs.get(str(j))
            if isinstance(d, str) and d.strip():
                c['definition'] = d.strip()
                c['definition_source'] = 'generated'


def fetch_chunks(ref: str) -> list[dict]:
    ref_sql = ref.replace("'", "''")
    rows = run_sql(f"""
        SELECT chunk_index, semantic_headers, chunk_text
        FROM {CHUNKS_TABLE}
        WHERE REF = '{ref_sql}'
        ORDER BY CAST(chunk_index AS INT)
    """)
    return [{'idx': int(r[0]), 'headers': r[1] or '', 'text': r[2] or ''} for r in rows]


def anchors(text: str) -> set:
    return {a.replace(' ', '') for a in _ANCHOR_RE.findall(text)}


def align(chunks_a: list[dict], chunks_b: list[dict]) -> list[dict]:
    """Pair chunks across languages: shared anchors first, relative position as
    tie-breaker. Translations keep part numbers / dimensions verbatim, so
    anchor overlap is a strong language-independent signal."""
    pairs = []
    used_b = set()
    for ca in chunks_a:
        aa = anchors(ca['text'])
        best, best_score = None, 0.0
        for cb in chunks_b:
            if cb['idx'] in used_b:
                continue
            ab = anchors(cb['text'])
            inter = len(aa & ab)
            union = len(aa | ab) or 1
            # position proximity (both docs have the same structure)
            pos_a = ca['idx'] / max(len(chunks_a) - 1, 1)
            pos_b = cb['idx'] / max(len(chunks_b) - 1, 1)
            score = (inter / union) + max(0.0, 0.3 - abs(pos_a - pos_b))
            if score > best_score:
                best, best_score = cb, score
        if best is not None and best_score > 0.15:
            used_b.add(best['idx'])
            pairs.append({'a': ca, 'b': best, 'score': round(best_score, 3)})
    return pairs


def _post_with_backoff(url: str, headers: dict, payload: dict, retries: int = 4) -> httpx.Response:
    """Les endpoints Databricks sont plafonnés (~100k ITPM / 20k OTPM) — un
    batch parallèle prend des 429 : attendre et réessayer plutôt que perdre
    la famille. Respecte Retry-After quand le serveur l'annonce."""
    import time as _time
    for attempt in range(retries + 1):
        r = httpx.post(url, headers=headers, json=payload, timeout=120, verify=False)
        if r.status_code != 429 or attempt == retries:
            r.raise_for_status()
            return r
        try:
            wait = float(r.headers.get('retry-after', ''))
        except ValueError:
            wait = 0.0
        _time.sleep(max(wait, 20 * (attempt + 1)))
    return r


def llm_extract(pair: dict, lang_a: str, lang_b: str, endpoint: str, host: str, token: str) -> list[dict]:
    names = {'fr': 'French', 'en': 'English', 'bg': 'Bulgarian', 'es': 'Spanish', 'cs': 'Czech', 'de': 'German',
              'pt': 'Portuguese', 'ar': 'Arabic'}
    system = _EXTRACT_PROMPT.format(lang_a=names[lang_a], lang_b=names[lang_b], code_a=lang_a, code_b=lang_b)
    user = f"--- {lang_a.upper()} ---\n{pair['a']['text'][:4000]}\n\n--- {lang_b.upper()} ---\n{pair['b']['text'][:4000]}"
    r = _post_with_backoff(
        f'{host}/serving-endpoints/{endpoint}/invocations',
        {'Authorization': f'Bearer {token}'},
        {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
         'max_tokens': 1500, 'temperature': 0.0},
    )
    text = r.json().get('choices', [{}])[0].get('message', {}).get('content', '')
    m = re.search(r'\{[\s\S]*\}', text)
    if not m:
        return []
    try:
        terms = json.loads(m.group(0)).get('terms', [])
    except json.JSONDecodeError:
        return []
    return [t for t in terms if isinstance(t, dict) and t.get(lang_a) and t.get(lang_b)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('base', help='base REF de la famille (ex: IF20335)')
    ap.add_argument('--langs', nargs=2, default=['fr', 'en'], help='paire de langues (codes glossaire)')
    ap.add_argument('--llm', action='store_true', help='extraire les termes via LLM (sinon: alignement seul)')
    ap.add_argument('--define', action='store_true',
                    help='générer une définition FR courte par terme (marquée definition_source=generated)')
    # gpt-5-4-mini : bas coût et accessible aux utilisateurs UAT (gemini/haiku
    # renvoient 403 sans droit Can Query ; Sonnet marche mais coûte ~30×).
    ap.add_argument('--endpoint', default=os.getenv('GLOSSARY_EXTRACT_ENDPOINT', 'databricks-gpt-5-4-mini'))
    ap.add_argument('--max-sections', type=int, default=5, help='cap de paires envoyées au LLM')
    ap.add_argument('--out', type=Path, help='CSV des candidats (défaut: candidates_<base>.csv à côté)')
    args = ap.parse_args()
    lang_a, lang_b = args.langs

    # Retrouver les REFs de la famille pour ces deux langues
    base_sql = args.base.replace("'", "''")
    rows = run_sql(f"""
        SELECT DISTINCT REF, upper(regexp_extract(REF, '{SUFFIX_RE}', 1)) AS suffix
        FROM {CHUNKS_TABLE}
        WHERE regexp_replace(REF, '{SUFFIX_RE}', '') = '{base_sql}'
    """)
    by_code: dict = {}
    for ref, suffix in rows:
        code = LANG_SUFFIX_TO_CODE.get(suffix or '')
        if code:
            by_code.setdefault(code, ref)
    if lang_a not in by_code or lang_b not in by_code:
        print(f'famille {args.base}: langues disponibles {sorted(by_code)} — paire {lang_a}/{lang_b} indisponible')
        return

    ref_a, ref_b = by_code[lang_a], by_code[lang_b]
    chunks_a, chunks_b = fetch_chunks(ref_a), fetch_chunks(ref_b)
    print(f'{ref_a}: {len(chunks_a)} chunks | {ref_b}: {len(chunks_b)} chunks')

    pairs = align(chunks_a, chunks_b)
    strong = [p for p in pairs if p['score'] > 0.3]
    print(f'alignement: {len(pairs)} paires ({len(strong)} fortes, score>0.3)')
    for p in pairs[:6]:
        print(f"  score={p['score']:5} | A#{p['a']['idx']} {p['a']['text'][:60]!r}")
        print(f"          | B#{p['b']['idx']} {p['b']['text'][:60]!r}")

    if not args.llm:
        print('\n(--llm non passé : alignement seul, aucun coût LLM)')
        return

    # N'utiliser le couple env HOST+TOKEN que s'il est complet : un PAT machine
    # résiduel (souvent DEV) sans host assorti provoquerait des 403 trompeurs
    # sur le workspace UAT.
    env_host, env_token = os.environ.get('DATABRICKS_HOST'), os.environ.get('DATABRICKS_TOKEN')
    if env_host and env_token:
        host, token = env_host.rstrip('/'), env_token
    else:
        host = 'https://dbc-3a17bfce-9e88.cloud.databricks.com'
        token = json.loads(subprocess.check_output(
            ['databricks', 'auth', 'token', '-p', 'UAT'], text=True, encoding='utf-8'))['access_token']

    candidates: dict = {}
    for p in sorted(strong, key=lambda x: -x['score'])[:args.max_sections]:
        for t in llm_extract(p, lang_a, lang_b, args.endpoint, host, token):
            key = t[lang_a].strip().lower()
            if key and key not in candidates:
                # args.base (not ref_a/ref_b) — the language-independent family
                # id, so 'source' reflects the document pair, not just whichever
                # language happened to be lang_a.
                candidates[key] = {lang_a: t[lang_a].strip(), lang_b: t[lang_b].strip(),
                                   'source': f"{args.base}#chunk{p['a']['idx']}", 'align_score': p['score']}

    if args.define and candidates:
        llm_define(list(candidates.values()), lang_a, lang_b, args.endpoint, host, token)

    out = args.out or Path(__file__).parent / f'candidates_{args.base}_{lang_a}-{lang_b}.csv'
    with out.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow([lang_a, lang_b, 'definition', 'definition_source', 'source', 'align_score', 'status'])
        for c in candidates.values():
            w.writerow([c[lang_a], c[lang_b], c.get('definition', ''), c.get('definition_source', ''),
                        c['source'], c['align_score'], 'pending'])
    print(f'\n{len(candidates)} termes candidats -> {out} (status=pending : à relire avant import)')
    for c in list(candidates.values())[:12]:
        line = f"  {c[lang_a]:38} -> {c[lang_b]}"
        if c.get('definition'):
            line += f"\n      déf: {c['definition']}"
        print(line)


if __name__ == '__main__':
    main()
