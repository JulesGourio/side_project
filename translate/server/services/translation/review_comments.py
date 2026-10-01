"""Generate Word review-comment specs from a job's own already-computed
data — ported from the POC's tools/gen_review.py, minus its hardcoded
Czech/French terminology dictionary (tied to the vocabulary of two specific
pilot documents, not generalizable to arbitrary language pairs; Phase 1's
job-driven glossary candidates — server/services/translation/glossary_extract.py —
cover "flag notable terminology" going forward instead).

Every category here reuses a signal the pipeline already computes rather
than re-deriving it: conflict_detail (audit.py's assess_pair), the
translated/failed/kept_* category buckets the Segments panel already shows
(server/routers/translate.py::_segment_category), the bilingual-inline
pattern_type, and check_fit()'s flagged list (with its per-location-type
thresholds — more nuanced than the POC's one flat 140% txbx-only check).

See comments.py for how these specs get anchored into the .docx as real
Word comments.
"""
from __future__ import annotations

from typing import Any

_LOCATION_LABELS = {
    'header': 'Header/footer', 'footer': 'Header/footer', 'txbx': 'Text box',
    'table': 'Table cell', 'body_direct': 'Body text', 'sdt': 'Content control',
}

# Same first-match-per-segment priority order the POC used (one comment per
# segment to keep the Review pane readable, not one per issue detected).
_TYPE_PRIORITY = {'header': 0, 'untranslated': 1, 'conflict': 2, 'overflow': 3, 'layout': 4}

_INLINE_PATTERNS = ('bilingual_inline_slash', 'bilingual_inline_concat', 'bilingual_inline_charsplit')

_AUTHOR = 'Translation review'


def generate_review_comments(
    segments: list[dict[str, Any]], flagged_fit: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """segments: one dict per segment with at least seg_id, pattern_type,
    conflict_flag, conflict_detail, category (the same bucket
    _segment_category assigns: translated/kept_dnt/kept_numeric/
    kept_other_language/needs_review). flagged_fit: check_fit()'s return value.

    Returns [{seg_id, author, type, comment}, ...], deduplicated to one
    comment per segment (ties broken by the priority above), ready for
    comments.inject_comments().
    """
    comments: list[dict[str, str]] = []

    for s in segments:
        if s.get('conflict_flag') and s.get('conflict_detail'):
            comments.append({
                'seg_id': s['seg_id'], 'author': _AUTHOR, 'type': 'conflict',
                'comment': f"{s['conflict_detail']} Please verify against the original.",
            })

    for s in segments:
        if s.get('category') == 'needs_review':
            comments.append({
                'seg_id': s['seg_id'], 'author': _AUTHOR, 'type': 'untranslated',
                'comment': ('This segment could not be translated automatically after retries. '
                            'Source text was kept as-is — please provide the translation or '
                            'confirm it should stay untranslated.'),
            })

    for s in segments:
        if s.get('pattern_type') in _INLINE_PATTERNS:
            comments.append({
                'seg_id': s['seg_id'], 'author': _AUTHOR, 'type': 'layout',
                'comment': ('This paragraph mixed both languages together in the source. '
                            'Please verify both the kept and translated portions read correctly.'),
            })

    for f in flagged_fit:
        label = _LOCATION_LABELS.get(f['location'], f['location'].replace('_', ' '))
        ctype = 'header' if f['location'] in ('header', 'footer') else 'overflow'
        comments.append({
            'seg_id': f['seg_id'], 'author': _AUTHOR, 'type': ctype,
            'comment': (f"{label}: translated text is {f['ratio']}% of the original length "
                        f"({f['new_len']} vs {f['orig_len']} chars, threshold {f['threshold']}%). "
                        f"May overflow its container — consider shortening or check the layout."),
        })

    seen: set = set()
    ordered: list[dict[str, str]] = []
    for c in sorted(comments, key=lambda c: _TYPE_PRIORITY.get(c['type'], 9)):
        if c['seg_id'] in seen:
            continue
        seen.add(c['seg_id'])
        ordered.append(c)
    return ordered
