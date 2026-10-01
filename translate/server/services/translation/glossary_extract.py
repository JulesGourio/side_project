"""Job-driven glossary candidate extraction — the standard way the glossary
grows going forward. The offline Intraqual corpus import
(utils/glossary/{build_review_batch,sync_candidates_to_lakebase,promote_pending_candidates}.py)
is a one-time supplementary bootstrap; it stays, but is not this.

Mirrors the original standalone Translator prototype's workflow (its
CLAUDE.md documented step 8: a validated job proposes term candidates, a
human reviews and confirms them into the glossary) — "propose" here means an
LLM extraction over the job's own confirmed source->target segment pairs (a
translation a human has actually reviewed via the job's Q&A + preview, not a
guess), and "confirm" is the existing glossary_candidates pending-review
queue (GlossaryPanel.tsx "To review" tab) rather than a hand-edited CSV.
Candidates land at status='pending' — never auto-published like the corpus
bootstrap, human validation stays the standard gate.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any

from ..llm import call_llm_json

LANGS = ('en', 'fr', 'cs', 'bg', 'de', 'es', 'pt', 'ar')


def normalize_term(term: str) -> str:
    """Same normalization as utils/glossary/build_review_batch.py's _norm —
    kept as an independent copy rather than importing across the
    dev-tooling/runtime boundary (utils/ scripts assume a CLI environment:
    `databricks auth token` subprocess calls, sys.path insertion — not
    something the deployed app should depend on)."""
    t = unicodedata.normalize('NFKD', term.lower().strip())
    return ''.join(c for c in t if not unicodedata.combining(c))

_LANG_NAMES = {'en': 'English', 'fr': 'French', 'cs': 'Czech', 'bg': 'Bulgarian', 'de': 'German', 'es': 'Spanish',
               'pt': 'Portuguese', 'ar': 'Arabic'}

_EXTRACT_PROMPT = """You are a bilingual aerospace terminology extractor.
Given numbered pairs of a {source_name} sentence and its confirmed {target_name} \
translation (from a real, human-reviewed translation job), extract the technical \
term correspondences (tools, operations, components, materials, defects) that \
appear in BOTH sentences of a pair. Only terms clearly present in both texts; \
no part numbers, no standards references, no full sentences — noun phrases of \
1-4 words.
Return ONLY JSON: {{"terms": [{{"{source_code}": "<term>", "{target_code}": "<term>"}}, ...]}}"""


async def extract_job_candidates(
    pairs: list[tuple[str, str]],
    source_lang: str,
    target_lang: str,
    host: str,
    token: str,
    endpoint: str,
    max_pairs: int = 40,
) -> list[dict[str, str]]:
    """pairs: (source_text, translated_text) for segments actually translated
    in one job (whole-segment translations only — see caller). Returns
    candidate {source_lang: term, target_lang: term} dicts.

    Deduplicates and ranks by repeat count first: a job can have hundreds of
    segments (too many to put in one prompt), and a phrase recurring several
    times within the SAME job is the strongest signal it's a real term worth
    proposing, not a one-off sentence fragment.
    """
    counts = Counter(pairs)
    top = [p for p, _n in counts.most_common(max_pairs)]
    if not top:
        return []

    system = _EXTRACT_PROMPT.format(
        source_name=_LANG_NAMES.get(source_lang, source_lang.upper()),
        target_name=_LANG_NAMES.get(target_lang, target_lang.upper()),
        source_code=source_lang, target_code=target_lang,
    )
    user = '\n'.join(f'{i}. {src} -> {tgt}' for i, (src, tgt) in enumerate(top))
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]
    result, _usage = await call_llm_json(host, token, endpoint, messages)
    terms: list[Any] = result.get('terms', []) if isinstance(result, dict) else []
    return [
        {source_lang: t[source_lang].strip(), target_lang: t[target_lang].strip()}
        for t in terms
        if isinstance(t, dict) and t.get(source_lang) and t.get(target_lang)
    ]
