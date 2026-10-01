"""Translation pipeline adapter — Translate tab.

Thin wrapper around the server/services/translation engine modules
(segmentation, bilingual pairing, conflict detection, question generation),
adapted to run in-memory against uploaded bytes instead of CLI file paths.
"""

import io
import logging
import zipfile
from typing import Any, Dict, List, Tuple

from ..translation import audit as _audit
from ..translation import extract as _extract
from ..translation import fit_check as _fit_check
from ..translation import glossary_io as _glossary_io
from ..translation import langdetect as _langdetect
from ..translation import length_adapt as _length_adapt
from ..translation import rebuild as _rebuild
from ..translation import validate as _validate
from ..translation.docx_images import IMAGE_TRANSLATION_MARKER as _IMAGE_TRANSLATION_MARKER

logger = logging.getLogger(__name__)


def extract_docx_segments(docx_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract translatable segments from a bilingual .docx (in-memory).

    Reuses the engine's extract.py XML walk verbatim — it already
    handles the tricky part (text boxes appear twice, in mc:Choice AND
    mc:Fallback; both positional paths are recorded so rebuild can update
    both), tables, and structured-document-tag containers.
    """
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
        text_parts = _extract.discover_text_parts(z.namelist())
        segments: List[Dict[str, Any]] = []
        for part in text_parts:
            xml_bytes = z.read(part)
            segments.extend(_extract.extract_part(xml_bytes, part))
    return segments


def audit_segments(
    segments: List[Dict[str, Any]],
    dnt_rows: List[Dict[str, str]],
    source_lang: str | None = None,
    target_lang: str | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Language-detect, bilingual-pair, and flag conflicts across segments;
    generate the batched clarifying questions for ambiguous/conflicting ones.

    Mirrors the engine audit.py's main(), minus its file I/O. dnt_rows comes
    from the Lakebase dnt_rules table (not the CSV file) — built into a
    DntMatcher per call (never module state, so concurrent jobs can't share
    one reviewer's rules) and applied live during DNT classification, on top
    of the built-in DNT_REGEX (dates/part numbers/standards).

    source_lang/target_lang are the user's declared pair. When given, language
    detection is constrained to that pair (the ensemble detector) and the
    document translation mode (monolingual vs bilingual) is determined — both
    fed forward to the translation-planning stage.
    """
    matcher = _glossary_io.DntMatcher(dnt_rows)
    audited = _audit.audit_all_segments(segments, matcher, source_lang, target_lang)
    mode = (_audit.determine_mode(audited, source_lang, target_lang)
            if source_lang and target_lang else "bilingual")

    # Cross-paragraph/row/textbox pairing (and its conflict detection) only
    # makes sense when two languages sit side by side. On a monolingual doc it
    # fires on incidental false positives and invents bogus pairs/conflicts
    # (measured on a real French doc: 18 spurious conflicts) — skip in mono.
    if mode == 'bilingual':
        pairer = _audit.Pairer()
        _audit.pair_txbx_siblings(audited, pairer)
        _audit.pair_adjacent_body(audited, pairer)
        _audit.pair_table_rows(audited, pairer)

    # Inline splits (TWO languages concatenated in ONE segment/run — e.g. a
    # Bulgarian sentence immediately followed by its English echo) run in BOTH
    # modes: the source span must be translated and the other-language span
    # kept even when the document as a whole is "monolingual" in the sense that
    # the declared target never appears in it. This is the case that was
    # silently left half-untranslated when inline splitting was mono-gated.
    _audit.pair_inline_slash(audited)
    _audit.pair_inline_format_concat(audited)
    _audit.pair_inline_charspan(audited)

    if mode == 'bilingual':
        pairs_by_id: Dict[str, List[dict]] = {}
        for s in audited:
            if s['pair_id']:
                pairs_by_id.setdefault(s['pair_id'], []).append(s)
        for _pid, segs in pairs_by_id.items():
            if len(segs) != 2:
                continue
            primary = next((s for s in segs if s['pair_role'] == 'primary'), None)
            secondary = next((s for s in segs if s['pair_role'] == 'secondary'), None)
            if primary and secondary:
                _audit.assess_pair(primary, secondary)

    primary_lang, secondary_lang, all_langs = _audit.determine_languages(audited)

    questions = _audit.generate_questions(
        audited, primary_lang, secondary_lang, source_file='', mode=mode,
    )

    # make_audited/strip_internal keep only the audit fields — the CLI reads
    # part/location/XML paths from its separate extract.json, but the web
    # pipeline persists ONE row per segment and the rebuild stage needs those
    # fields to know where to write each translation. Merge them back in.
    ext_by_id = {s['seg_id']: s for s in segments}
    merged = []
    for s in audited:
        row = _audit.strip_internal(s)
        ext = ext_by_id.get(row['seg_id'], {})
        for k in ('part', 'location_type', 'xml_choice_path', 'xml_fallback_path'):
            row.setdefault(k, ext.get(k))
        merged.append(row)

    audit_report = {
        'mode': mode,
        'detected_languages': all_langs,
        'primary_language': primary_lang,
        'secondary_language': secondary_lang,
        'segment_count': len(audited),
        'pair_count': len([1 for s in audited if s['pair_role'] == 'primary']),
        'conflict_count': len([1 for s in audited if s['pair_role'] == 'primary' and s['conflict_flag']]),
        'dnt_token_count': sum(len(s['dnt_tokens']) for s in audited),
        'segments': merged,
    }
    return audit_report, questions


# ---------------------------------------------------------------------------
# fit_check / length_adapt — overflow detection + shortening
# ---------------------------------------------------------------------------

def check_fit(
    extract_segments: List[Dict[str, Any]],
    translated_segments: List[Dict[str, Any]],
    threshold_override: int | None = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Flag segments whose translated text is likely to overflow its
    container. Mirrors the engine fit_check.py's main(), minus its
    file I/O. Returns (flagged, critical_count)."""
    ext_by_id = {s['seg_id']: s for s in extract_segments}
    flagged: List[Dict[str, Any]] = []
    critical = 0
    for tseg in translated_segments:
        if tseg.get('keep_as_is') or tseg.get('translated_text') is None:
            continue
        seg_id = tseg['seg_id']
        ext = ext_by_id.get(seg_id)
        if not ext:
            continue
        orig_len = len(tseg['original_text'])
        new_len = len(tseg['translated_text'])
        if orig_len == 0:
            continue
        inline_split = tseg.get('inline_split')
        if inline_split and inline_split.get('span_translated'):
            # Format-kind bilingual segment: translated_text holds only the
            # source-side span (the kept side stays in its own runs), so the
            # rendered length is kept-side + span translation, not the span
            # alone.
            new_len += orig_len - inline_split.get('source_span_len', 0)
        ratio = (new_len / orig_len) * 100
        loc = _fit_check.classify_location(seg_id, ext['location_type'], ext['part'])
        threshold = threshold_override or _fit_check.THRESHOLDS.get(loc, 130)
        severity = _fit_check.SEVERITY.get(loc, 'INFO')
        if ratio > threshold:
            flagged.append({
                'seg_id': seg_id, 'location': loc, 'severity': severity,
                'orig_len': orig_len, 'new_len': new_len, 'ratio': round(ratio, 1),
                'threshold': threshold,
                'original_text': tseg['original_text'][:80],
                'translated_text': tseg['translated_text'][:80],
            })
            if severity == 'CRITICAL':
                critical += 1
    sev_order = {'CRITICAL': 0, 'WARNING': 1, 'INFO': 2}
    flagged.sort(key=lambda f: (sev_order.get(f['severity'], 9), -f['ratio']))
    return flagged, critical


def adapt_lengths(
    extract_segments: List[Dict[str, Any]],
    translated_segments: List[Dict[str, Any]],
    target_lang: str = "fr",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Shorten overflowing translations using per-language abbreviation
    rules (see length_adapt.SHORTENINGS_BY_LANG). Mutates translated_segments'
    translated_text in place (matching length_adapt.py's --apply behaviour)
    and returns (translated_segments, suggestions)."""
    ext_by_id = {s['seg_id']: s for s in extract_segments}
    thresholds = {'header': 100, 'footer': 100, 'txbx': 130, 'table': 150}
    suggestions: List[Dict[str, Any]] = []
    for tseg in translated_segments:
        if tseg.get('keep_as_is') or tseg.get('translated_text') is None:
            continue
        inline_split = tseg.get('inline_split')
        if inline_split and inline_split.get('span_translated'):
            # Span-only translation (format-kind bilingual segment): the
            # orig_len-based shortening targets below would mis-size it.
            continue
        seg_id = tseg['seg_id']
        ext = ext_by_id.get(seg_id)
        if not ext:
            continue
        orig_len = len(tseg['original_text'])
        new_text = tseg['translated_text']
        new_len = len(new_text)
        if orig_len == 0:
            continue
        part = ext['part']
        loc = 'header' if 'header' in part else 'footer' if 'footer' in part else ext['location_type']
        threshold = thresholds.get(loc)
        if threshold is None:
            continue
        ratio = (new_len / orig_len) * 100
        if ratio <= threshold:
            continue

        if loc in ('header', 'footer'):
            alt = _length_adapt.HEADER_ALTERNATIVES_BY_LANG.get(target_lang, {}).get(new_text.strip())
            if alt:
                shortened, rules = alt, ['header alternative']
            else:
                shortened, rules = _length_adapt.shorten(new_text, orig_len, lang=target_lang)
        else:
            shortened, rules = _length_adapt.shorten(new_text, int(orig_len * threshold / 100), lang=target_lang)

        shortened_ratio = (len(shortened) / orig_len) * 100
        suggestions.append({
            'seg_id': seg_id, 'location': loc, 'orig_len': orig_len,
            'before': new_text, 'before_len': new_len,
            'after': shortened, 'after_len': len(shortened),
            'fits': shortened_ratio <= threshold, 'rules': rules,
        })
        tseg['translated_text'] = shortened
    return translated_segments, suggestions


# ---------------------------------------------------------------------------
# rebuild — reconstruct a translated .docx in-memory
# ---------------------------------------------------------------------------

def build_rebuild_inputs(segments: List[Dict[str, Any]]) -> Tuple[Dict[str, List[dict]], Dict[str, Any]]:
    """Group DB-shaped translation_segments rows by XML part and build the
    per-segment metadata dicts rebuild.apply_translations_lxml() expects.

    `segments` items carry (already JSON-decoded): seg_id, part,
    xml_choice_path, xml_fallback_path, pattern_type, inline_split,
    translated_text, keep_as_is, original_text (source_text).
    """
    translations_by_part: Dict[str, List[dict]] = {}
    metadata: Dict[str, Any] = {
        'choice_paths': {}, 'fallback_paths': {}, 'fmt_signatures': {},
        'inline_splits': {}, 'pattern_types': {},
    }
    for s in segments:
        part = s['part']
        translations_by_part.setdefault(part, []).append({
            'seg_id': s['seg_id'],
            'original_text': s['original_text'],
            'translated_text': s.get('translated_text'),
            'keep_as_is': bool(s.get('keep_as_is')),
        })
        metadata['choice_paths'][s['seg_id']] = s.get('xml_choice_path')
        metadata['fallback_paths'][s['seg_id']] = s.get('xml_fallback_path')
        metadata['inline_splits'][s['seg_id']] = s.get('inline_split')
        metadata['pattern_types'][s['seg_id']] = s.get('pattern_type') or 'mono'
    return translations_by_part, metadata


def rebuild_docx_bytes(
    src_bytes: bytes,
    translations_by_part: Dict[str, List[dict]],
    metadata: Dict[str, Any],
) -> bytes:
    """Rebuild a translated .docx in-memory. Mirrors the engine
    rebuild.py's rebuild_docx(), operating on bytes instead of file paths
    (a web job has no local filesystem for intermediate files). Reuses
    apply_translations_lxml verbatim — lxml preserves namespace declarations
    and attribute formatting byte-for-byte, which is the part that matters
    for Word to accept the file.
    """
    from lxml import etree as lxml_etree

    with zipfile.ZipFile(io.BytesIO(src_bytes), 'r') as zin:
        names = zin.namelist()
        new_bytes: Dict[str, bytes] = {}

        for part, segs in translations_by_part.items():
            if part not in names:
                continue
            xml_bytes = zin.read(part)
            parser = lxml_etree.XMLParser(remove_blank_text=False, strip_cdata=False)
            lxml_root = lxml_etree.fromstring(xml_bytes, parser)

            applied = _rebuild.apply_translations_lxml(
                lxml_root, segs,
                metadata.get('choice_paths', {}), metadata.get('fallback_paths', {}),
                metadata.get('fmt_signatures', {}), metadata.get('inline_splits', {}),
                metadata.get('pattern_types', {}),
            )
            logger.info('rebuild %s: applied %d/%d translations', part, applied, len(segs))

            raw = lxml_etree.tostring(lxml_root, xml_declaration=True, encoding='UTF-8', standalone=True)
            raw = raw.replace(
                b"<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
                b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', 1,
            )
            raw = raw.replace(b'?>\n<', b'?>\r\n<', 1)
            new_bytes[part] = raw

        out_buf = io.BytesIO()
        with zipfile.ZipFile(out_buf, 'w', zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                zout.writestr(name, new_bytes.get(name, zin.read(name)))

    return out_buf.getvalue()


# ---------------------------------------------------------------------------
# validate — 8-check structural integrity suite
# ---------------------------------------------------------------------------

def _check_segment_fidelity_bytes(new_bytes: bytes, reference_segments: List[Dict[str, Any]]) -> List[str]:
    """word/comments.xml is excluded from this comparison: when the caller
    asked for include_review_comments, inject_comments() (comments.py)
    appends real Word comments to it as an intentional part of THIS rebuild
    — extraction now picks those up as new segments (see extract.py's
    NOTE_WRAPPER_TAGS), which would otherwise look like accidental content
    drift to a check meant to catch lost/duplicated body content.

    Same reasoning for the docx_images.py image-translation paragraphs —
    intentional growth, just identified by their IMAGE_TRANSLATION_MARKER
    prefix instead of a separate XML part, since they live in
    word/document.xml itself."""
    def _is_excluded(s: Dict[str, Any]) -> bool:
        return s['part'] == 'word/comments.xml' or s['text'].lstrip().startswith(_IMAGE_TRANSLATION_MARKER)

    errors = []
    new_segments = [s for s in extract_docx_segments(new_bytes) if not _is_excluded(s)]
    reference_segments = [s for s in reference_segments if not _is_excluded(s)]
    ref_ids = {s['seg_id'] for s in reference_segments}
    new_ids = {s['seg_id'] for s in new_segments}
    missing = ref_ids - new_ids
    extra = new_ids - ref_ids
    if missing:
        errors.append(f'Missing {len(missing)} segments vs reference')
    if extra:
        errors.append(f'Extra {len(extra)} segments vs reference')
    if len(reference_segments) != len(new_segments):
        errors.append(f'Segment count mismatch: ref={len(reference_segments)} new={len(new_segments)}')
    return errors


def _check_byte_identity_bytes(new_bytes: bytes, original_bytes: bytes) -> List[str]:
    errors = []
    # word/endnotes.xml and word/footnotes.xml are deliberately NOT in this
    # set: extract.py now translates their paragraphs (NOTE_WRAPPER_TAGS), so
    # they legitimately change during rebuild — asserting byte-identity here
    # would fail every job that actually has footnote/endnote content.
    immutable = {
        'word/styles.xml', 'word/settings.xml', 'word/fontTable.xml',
        'word/webSettings.xml', 'word/numbering.xml', 'word/theme/theme1.xml',
    }
    with zipfile.ZipFile(io.BytesIO(new_bytes)) as z1, zipfile.ZipFile(io.BytesIO(original_bytes)) as z2:
        for name in immutable:
            if name not in z1.namelist() or name not in z2.namelist():
                continue
            a, b = z1.read(name), z2.read(name)
            if a != b:
                errors.append(f'{name} differs from original (orig={len(b)} new={len(a)} delta={len(a)-len(b)})')
    return errors


_RESIDUAL_UNIQUE_CHARS = {
    'cs': set('řůěŘŮĚ'),
    'bg': None,  # Cyrillic block check instead
    'de': set('ßÄÖÜäöü'),
    'es': set('ñÑ¿¡'),
    'pt': set('ãõÃÕ'),  # nasal vowels, not used by any other language in scope
    'ar': None,  # Arabic block check instead — own script, like bg's Cyrillic
}


def _check_residual_source_language_bytes(new_bytes: bytes, source_lang: str) -> List[Dict[str, Any]]:
    """Return the segments in the rebuilt document that still read as the
    source language — a QUALITY signal, not a hard failure. Each item is
    {seg_id, text, detected_lang, confidence} so the UI can render an
    actionable list (jump to the segment, edit it) instead of a wall of
    truncated strings."""
    segments = extract_docx_segments(new_bytes)
    unique_chars = _RESIDUAL_UNIQUE_CHARS.get(source_lang)
    # unique_chars alone has real blind spots — measured 2026-08-24: a genuine
    # Czech word like "propojeni" (accents shared with other languages, no
    # cs-exclusive character) never matches it and was silently invisible to
    # this whole check. Falls back to the same constrained ensemble detector
    # the audit stage uses (case-folded fastText — see _fasttext_scores) for
    # anything the character shortcut doesn't catch. Deliberately NOT passing
    # source_lang (which would engage the audit stage's single-ambiguous-word
    # bias toward source_lang) — that bias is right for "should this be
    # translated" but would flood this confirmation view with false
    # positives on ordinary short kept-as-is words.
    candidates = {source_lang, 'en', 'fr', 'de', 'es', 'cs', 'bg', 'pt', 'ar'}
    items: List[Dict[str, Any]] = []
    for s in segments:
        text = s['text']
        if not text.strip() or len(text.strip()) < 3:
            continue
        if source_lang == 'bg':
            if sum(1 for c in text if 'Ѐ' <= c <= 'ӿ') > 3:
                items.append({'seg_id': s['seg_id'], 'text': text,
                              'detected_lang': 'bg', 'confidence': None})
            continue
        if source_lang == 'ar':
            if sum(1 for c in text if '؀' <= c <= 'ۿ') > 3:
                items.append({'seg_id': s['seg_id'], 'text': text,
                              'detected_lang': 'ar', 'confidence': None})
            continue
        if unique_chars and any(c in unique_chars for c in text):
            items.append({'seg_id': s['seg_id'], 'text': text,
                          'detected_lang': source_lang, 'confidence': None})
            continue
        lang, conf = _langdetect.detect_language_constrained(text, candidates)
        if lang == source_lang and conf >= 0.55:
            items.append({'seg_id': s['seg_id'], 'text': text,
                          'detected_lang': lang, 'confidence': round(conf, 3)})
    return items


def validate_docx(
    new_bytes: bytes,
    original_bytes: bytes | None = None,
    reference_segments: List[Dict[str, Any]] | None = None,
    source_lang: str | None = None,
) -> Dict[str, Any]:
    """Run the 8-check validation suite against a rebuilt .docx (in-memory).

    Checks 1-5 reuse the engine validate.py's functions verbatim —
    zipfile.ZipFile accepts a BytesIO exactly like a path, so no adaptation
    was needed for those. Checks 6-8 need actual bytes (not a path) so
    they're reimplemented here, reusing extract_docx_segments() instead of
    validate.py's tempfile-based extract-to-disk round trip.
    """
    checks: List[Tuple[str, List[str]]] = [
        ('zip_integrity', _validate.check_zip_integrity(io.BytesIO(new_bytes))),
        ('xml_validity', _validate.check_xml_validity(io.BytesIO(new_bytes))),
        ('namespace_integrity', _validate.check_namespace_integrity(io.BytesIO(new_bytes))),
        ('auto_prefix', _validate.check_auto_prefixes(io.BytesIO(new_bytes))),
        ('content_types', _validate.check_content_types(io.BytesIO(new_bytes))),
    ]
    if reference_segments is not None:
        checks.append(('segment_fidelity', _check_segment_fidelity_bytes(new_bytes, reference_segments)))
    if original_bytes is not None:
        checks.append(('byte_identity', _check_byte_identity_bytes(new_bytes, original_bytes)))

    results: Dict[str, Any] = {name: {'passed': len(errs) == 0, 'errors': errs} for name, errs in checks}

    # residual_source_language is a QUALITY signal (structured list of the
    # still-source-language segments), handled apart from the string-list
    # structural checks so the UI can render it actionably.
    if source_lang:
        items = _check_residual_source_language_bytes(new_bytes, source_lang)
        errs = ([f'{len(items)} segments still contain {source_lang.upper()} text']
                + [f"  {it['seg_id']}: {it['text'][:100]}" for it in items[:10]]
                + ([f'  ... and {len(items) - 10} more'] if len(items) > 10 else []))
        results['residual_source_language'] = {
            'passed': not items, 'errors': errs if items else [], 'items': items,
        }

    results['all_passed'] = all(r['passed'] for r in results.values())
    return results
