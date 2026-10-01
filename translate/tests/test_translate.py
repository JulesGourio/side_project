"""Tests for the Translate pipeline server logic.

Coverage:
  - _parse_lang_answer (deterministic answer application)
  - _build_glossary_context batch filtering
  - _translate_batch id-keyed request/response mapping (mocked LLM)
  - check_fit / adapt_lengths None guards (keep-as-is segments carry no translation)
  - soffice exact-PDF engine: discovery failure modes + content-hash cache
  - bilingual inline pre-split (charspan boundary finder, per-segment
    translation planning, span-aware rebuild dispatch)
  - extract_job_candidates: job-driven glossary candidate extraction
    (frequency ranking/cap, malformed-payload tolerance)
  - render_page_diff: pixel-level preview diff (out-of-range pages, identical
    vs differing pages)
  - generate_review_comments / inject_comments: Word-native review comments
    (comment/marker placement on a real docx, segment fidelity preserved)
  - detect_language / audit_all_segments: French false-negatives (missing FR
    stopwords, arbitrary shared-diacritic tie-break, hardcoded 'en' default)
    found on a real 92%-French document — see langdetect.py's docstrings
"""

import asyncio
import hashlib
import zipfile
from io import BytesIO
from unittest.mock import AsyncMock, patch

from server.routers.translate import (
    _build_glossary_context,
    _parse_lang_answer,
    _plan_segment_translation,
    _translate_batch,
)
from server.services import soffice
from server.services.processors.translation import adapt_lengths, audit_segments, check_fit, extract_docx_segments
from server.services.translation import audit as _audit
from server.services.translation import rebuild as _rebuild
from server.services.translation.langdetect import detect_language, is_dnt_candidate
from server.services.translation.comments import inject_comments
from server.services.translation.glossary_extract import extract_job_candidates, normalize_term
from server.services.translation.pdf_diff import get_all_page_changes, render_page_diff
from server.services.translation.review_comments import generate_review_comments


# ---------------------------------------------------------------------------
# Language detection — French false-negatives found on a real document
# (old_FI_252A90008200-B.docx, 92% French, 2026-07-21)
# ---------------------------------------------------------------------------

def test_detect_language_french_sentence_not_lost_to_spanish_keyword_gap():
    # "en" appears in Spanish's stopword list; French's own list used to be
    # missing it entirely, so this unambiguous French sentence scored es@0.85
    # purely because nothing in French's list matched it.
    lang, conf = detect_language('Mise en place sur outillage')
    assert lang == 'fr'


def test_detect_language_shared_diacritic_tie_defers_instead_of_picking_cs():
    # "Rédacteur"/"Création"-style words: 'é' is common to CS/FR/ES, so a
    # single-character tie must not silently resolve to whichever language
    # happens to sort first in LANG_PROFILES (was always 'cs').
    lang, conf = detect_language('Création du document')
    assert lang != 'cs'


def test_detect_language_no_signal_returns_unknown_not_a_guess():
    # No accents, no stopword-list overlap with any language — the old
    # behavior silently guessed 'en' with no basis; the new contract is an
    # honest "no signal" so the caller's document-level bias (see
    # audit_all_segments) can resolve it instead of a hardcoded default.
    lang, conf = detect_language('OPERATION 20 PERCAGE')
    assert lang == '??'
    assert conf == 0.0


def test_detect_language_default_lang_resolves_zero_signal_text():
    lang, conf = detect_language('OPERATION 20 PERCAGE', default_lang='fr')
    assert lang == 'fr'
    assert conf > 0


def test_is_dnt_candidate_digit_first_reference_code():
    # 252A90008200: digits-letter-digits, the one internal-reference shape
    # none of the (letter-first) DNT patterns covered — it used to fall
    # through to language detection and get folded into whatever the
    # segment's language guess was, instead of being flagged do-not-translate.
    assert is_dnt_candidate('252A90008200')


def test_audit_all_segments_biases_zero_signal_mono_to_document_dominant_language():
    segments = [
        {'seg_id': f's{i}', 'part': 'document.xml', 'location_type': 'body_direct',
         'body_p_idx': i, 'table_coords': None, 'txbx_path': None,
         'para_idx_in_container': i, 'fmt_signature': '', 'runs': [], 'text': text}
        for i, text in enumerate([
            'Installer la vis et la rondelle selon le plan.',
            'Vérifier le serrage du boulon avant assemblage.',
            'Contrôler la bonne mise en place de la pièce.',
            'OPERATION 20 PERCAGE',  # zero signal on its own — should inherit 'fr'
        ])
    ]
    audited = _audit.audit_all_segments(segments)
    by_id = {a['seg_id']: a for a in audited}
    assert by_id['s3']['detected_lang'] == 'fr'
    assert by_id['s3']['pattern_type'] == 'mono'


def _extract_seg(i, text):
    return {'seg_id': f's{i}', 'part': 'word/document.xml', 'location_type': 'body_direct',
            'body_p_idx': i, 'table_coords': None, 'txbx_path': None,
            'para_idx_in_container': i, 'fmt_signature': '',
            'runs': [{'text': text, 'fmt_hash': 'x'}], 'text': text,
            'xml_choice_path': None, 'xml_fallback_path': None}


def test_inline_charsplit_runs_and_translates_source_side_regardless_of_mode():
    """A segment concatenating the source language (BG) with a third language
    (EN) must be split so the BG span is translated and the EN span kept —
    even when the document mode comes out 'monolingual' because the declared
    target (FR) never appears in the source document. Regression for the
    BG+EN aerospace segments reported left half-untranslated."""
    bg_en = ('Поставете матрицата за полагане на лак VD-A321-00035 , след което защитите '
             'етикета с безцветен лак . '
             'Place the template for applying varnish VD-A321-00035, then protect the '
             'label with transparent varnish according to standard AIPS 08-03-002.')
    segments = [
        _extract_seg(0, 'Затягане на болтовете преди монтаж на панела.'),
        _extract_seg(1, 'Проверете срещу чертежа преди сглобяване.'),
        _extract_seg(2, bg_en),
    ]
    report, _ = audit_segments(segments, [], 'bg', 'fr')
    s2 = next(a for a in report['segments'] if a['seg_id'] == 's2')
    assert s2['pattern_type'] == 'bilingual_inline_charsplit'
    assert s2['inline_split']['left_lang'] == 'bg'
    assert s2['inline_split']['right_lang'] == 'en'

    row = {'pattern_type': s2['pattern_type'], 'source_text': s2['text'],
           'detected_lang': s2['detected_lang'], 'lang_confidence': s2['lang_confidence'],
           'inline_split': s2['inline_split']}
    plan = _plan_segment_translation(row, 'bg', 'fr', report['mode'])
    assert plan is not None
    assert plan['query'].startswith('Поставете')           # BG span sent to LLM
    composed = plan['compose']('[FR]')
    assert 'Place the template' in composed                # EN side kept verbatim
    assert 'Поставете' not in composed                     # BG side replaced


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def test_parse_lang_answer_variants():
    assert _parse_lang_answer('CS') == 'cs'
    assert _parse_lang_answer("c'est du tchèque") == 'cs'
    assert _parse_lang_answer('This is Czech text') == 'cs'
    assert _parse_lang_answer('bulgare') == 'bg'
    assert _parse_lang_answer('français') == 'fr'


def test_parse_lang_answer_ambiguous_or_absent():
    assert _parse_lang_answer('cs or fr, not sure') is None
    assert _parse_lang_answer('je ne sais pas') is None
    assert _parse_lang_answer('') is None


# ---------------------------------------------------------------------------
# Glossary filtering
# ---------------------------------------------------------------------------

_GLOSSARY = [
    {'cs': 'šroub', 'fr': 'vis'},
    {'cs': 'matice', 'fr': 'écrou'},
    {'cs': 'podložka', 'fr': 'rondelle'},
]


def test_glossary_filtered_to_batch_terms():
    ctx = _build_glossary_context(_GLOSSARY, 'cs', 'fr',
                                  batch_strings=['Utáhněte šroub M5', 'Zkontrolujte matice'])
    assert 'šroub -> vis' in ctx
    assert 'matice -> écrou' in ctx
    assert 'podložka' not in ctx


def test_glossary_full_without_batch():
    ctx = _build_glossary_context(_GLOSSARY, 'cs', 'fr')
    assert ctx.count('->') == 3


# ---------------------------------------------------------------------------
# Batch translation — id-keyed mapping
# ---------------------------------------------------------------------------

def test_translate_batch_maps_ids_back_to_source_strings():
    strings = ['Utáhněte šroub', 'Zkontrolujte "matice"\nznovu', 'Hotovo']
    llm_response = {'translations': {'0': 'Serrez la vis', '1': 'Vérifiez les « écrous » à nouveau', '2': ''}}
    usage = {'input_tokens': 10, 'output_tokens': 5}
    with patch('server.routers.translate.call_llm_json', new=AsyncMock(return_value=(llm_response, usage))) as mock:
        out = asyncio.run(_translate_batch('h', 't', 'ep', strings, 'cs', 'fr', _GLOSSARY, 1))
    # id-keyed: quotes/newlines in source strings can no longer break the mapping
    assert out[strings[0]] == 'Serrez la vis'
    assert out[strings[1]] == 'Vérifiez les « écrous » à nouveau'
    assert strings[2] not in out  # empty translation treated as missing → retried upstream
    sent = mock.call_args.args[3][1]['content']
    assert '"id": "0"' in sent and '"id": "2"' in sent


def test_translate_batch_tolerates_malformed_llm_payload():
    usage = {'input_tokens': 1, 'output_tokens': 1}
    with patch('server.routers.translate.call_llm_json', new=AsyncMock(return_value=(['not', 'a', 'dict'], usage))):
        out = asyncio.run(_translate_batch('h', 't', 'ep', ['abc'], 'cs', 'fr', [], 1))
    assert out == {}


# ---------------------------------------------------------------------------
# Job-driven glossary candidate extraction
# ---------------------------------------------------------------------------

def test_normalize_term_case_and_accent_insensitive():
    assert normalize_term('Écrou-frein') == normalize_term('ECROU-FREIN') == 'ecrou-frein'


def test_extract_job_candidates_ranks_by_frequency_and_caps():
    # "šroub"/"vis" repeats 3x (should survive a max_pairs=1 cap); the
    # single-occurrence pair should not.
    pairs = [('Utáhněte šroub', 'Serrez la vis')] * 3 + [('Zkontrolujte matice', 'Vérifiez les écrous')]
    llm_response = {'terms': [{'cs': 'šroub', 'fr': 'vis'}]}
    usage = {'input_tokens': 5, 'output_tokens': 5}
    with patch(
        'server.services.translation.glossary_extract.call_llm_json',
        new=AsyncMock(return_value=(llm_response, usage)),
    ) as mock:
        out = asyncio.run(extract_job_candidates(pairs, 'cs', 'fr', 'h', 't', 'ep', max_pairs=1))
    assert out == [{'cs': 'šroub', 'fr': 'vis'}]
    sent_user_msg = mock.call_args.args[3][1]['content']
    assert 'Zkontrolujte matice' not in sent_user_msg  # capped out by lower frequency


def test_extract_job_candidates_no_pairs_skips_llm_call():
    with patch('server.services.translation.glossary_extract.call_llm_json', new=AsyncMock()) as mock:
        out = asyncio.run(extract_job_candidates([], 'cs', 'fr', 'h', 't', 'ep'))
    assert out == []
    mock.assert_not_called()


def test_extract_job_candidates_tolerates_malformed_llm_payload():
    usage = {'input_tokens': 1, 'output_tokens': 1}
    with patch(
        'server.services.translation.glossary_extract.call_llm_json',
        new=AsyncMock(return_value=(['not', 'a', 'dict'], usage)),
    ):
        out = asyncio.run(extract_job_candidates([('a', 'b')], 'cs', 'fr', 'h', 't', 'ep'))
    assert out == []


# ---------------------------------------------------------------------------
# Pixel-level preview diff
# ---------------------------------------------------------------------------

def _make_pdf(text: str) -> bytes:
    import pymupdf as fitz
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((20, 100), text, fontsize=24)
    try:
        return doc.tobytes()
    finally:
        doc.close()


def test_render_page_diff_out_of_range_page():
    pdf = _make_pdf('Hello world')
    png, total = render_page_diff(pdf, pdf, page_num=5)
    assert png is None
    assert total == 1


def test_render_page_diff_identical_pages_still_renders():
    pdf = _make_pdf('Hello world')
    png, total = render_page_diff(pdf, pdf, page_num=1)
    assert total == 1
    assert png is not None and len(png) > 0


def test_render_page_diff_differing_pages_renders_larger_image():
    before = _make_pdf('Hello world')
    after = _make_pdf('Bonjour le monde, ceci est different')
    png, total = render_page_diff(before, after, page_num=1)
    assert total == 1
    assert png is not None and len(png) > 0


def _make_multi_page_pdf(pages_texts):
    import pymupdf as fitz
    doc = fitz.open()
    for texts in pages_texts:
        page = doc.new_page(width=400, height=500)
        for pos, t in texts:
            page.insert_text(pos, t, fontsize=18)
    try:
        return doc.tobytes()
    finally:
        doc.close()


def test_get_all_page_changes_separates_distinct_regions_and_skips_unchanged_pages():
    before = _make_multi_page_pdf([
        [((30, 60), 'Hello world'), ((30, 400), 'Bottom original')],
        [((30, 60), 'Same page two')],
    ])
    after = _make_multi_page_pdf([
        [((30, 60), 'Bonjour monde'), ((30, 400), 'Bas modifie ici')],
        [((30, 60), 'Same page two')],
    ])
    changes = get_all_page_changes(before, after)
    assert len(changes) == 2
    assert all(c['page'] == 1 for c in changes)
    # reading order: top region before bottom region
    assert changes[0]['bbox'][1] < changes[1]['bbox'][1]
    for c in changes:
        x0, y0, x1, y1 = c['bbox']
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1


def test_get_all_page_changes_identical_documents_finds_nothing():
    pdf = _make_multi_page_pdf([[((30, 60), 'Same everywhere')]])
    assert get_all_page_changes(pdf, pdf) == []


# ---------------------------------------------------------------------------
# Word-native review comments
# ---------------------------------------------------------------------------

def test_generate_review_comments_covers_all_categories_and_dedupes():
    segments = [
        {'seg_id': 's1', 'pattern_type': 'mono', 'conflict_flag': True,
         'conflict_detail': 'Numeric values differ (10 vs 12).', 'category': 'translated'},
        {'seg_id': 's2', 'pattern_type': 'mono', 'conflict_flag': False,
         'conflict_detail': None, 'category': 'needs_review'},
        {'seg_id': 's3', 'pattern_type': 'bilingual_inline_slash', 'conflict_flag': False,
         'conflict_detail': None, 'category': 'translated'},
        # s1 also flagged by fit-check — conflict (priority 2) must win over
        # overflow (priority 3), one comment per segment.
        {'seg_id': 's4', 'pattern_type': 'mono', 'conflict_flag': False,
         'conflict_detail': None, 'category': 'translated'},
    ]
    flagged_fit = [
        {'seg_id': 's1', 'location': 'txbx', 'severity': 'WARNING', 'orig_len': 10,
         'new_len': 20, 'ratio': 200.0, 'threshold': 130},
        {'seg_id': 's4', 'location': 'header', 'severity': 'CRITICAL', 'orig_len': 5,
         'new_len': 9, 'ratio': 180.0, 'threshold': 100},
    ]
    comments = generate_review_comments(segments, flagged_fit)
    by_seg = {c['seg_id']: c for c in comments}
    assert set(by_seg) == {'s1', 's2', 's3', 's4'}
    assert by_seg['s1']['type'] == 'conflict'  # conflict beats overflow for the same segment
    assert 'Numeric values differ' in by_seg['s1']['comment']
    assert by_seg['s2']['type'] == 'untranslated'
    assert by_seg['s3']['type'] == 'layout'
    assert by_seg['s4']['type'] == 'header'  # header/footer location, not generic overflow
    assert len(comments) == 4  # one per segment, no duplicates


def test_generate_review_comments_empty_input():
    assert generate_review_comments([], []) == []


_COMMENTS_TEST_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

_DOCX_CT = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '</Types>'
)
_PKG_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/></Relationships>'
)
_DOC_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
)
_DOCUMENT_XML = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<w:document xmlns:w="{_COMMENTS_TEST_W}">'
    f'<w:body><w:p><w:r><w:t>Utahnete sroub</w:t></w:r></w:p></w:body>'
    f'</w:document>'
)


def _minimal_docx_bytes() -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', _DOCX_CT)
        z.writestr('_rels/.rels', _PKG_RELS)
        z.writestr('word/document.xml', _DOCUMENT_XML)
        z.writestr('word/_rels/document.xml.rels', _DOC_RELS)
    return buf.getvalue()


def test_inject_comments_places_marker_and_preserves_segment_text():
    docx_bytes = _minimal_docx_bytes()
    segments = extract_docx_segments(docx_bytes)
    assert len(segments) == 1
    seg = segments[0]

    comments = [{'seg_id': seg['seg_id'], 'author': 'Translation review',
                 'type': 'conflict', 'comment': 'Please verify.'}]
    xml_choice_paths = {seg['seg_id']: seg['xml_choice_path']}

    out_bytes, placed = inject_comments(docx_bytes, comments, xml_choice_paths)
    assert placed == 1

    with zipfile.ZipFile(BytesIO(out_bytes)) as z:
        names = z.namelist()
        assert 'word/comments.xml' in names
        comments_xml = z.read('word/comments.xml').decode('utf-8')
        assert 'Please verify.' in comments_xml
        assert '[CONFLICT]' in comments_xml
        doc_xml = z.read('word/document.xml').decode('utf-8')
        assert 'commentRangeStart' in doc_xml
        assert 'commentReference' in doc_xml
        rels_xml = z.read('word/_rels/document.xml.rels').decode('utf-8')
        assert 'comments.xml' in rels_xml
        ct_xml = z.read('[Content_Types].xml').decode('utf-8')
        assert '/word/comments.xml' in ct_xml

    # Segment fidelity: re-extracting the commented docx finds the same text.
    re_extracted = extract_docx_segments(out_bytes)
    assert [s['text'] for s in re_extracted] == [s['text'] for s in segments]


def test_inject_comments_empty_list_is_noop():
    docx_bytes = _minimal_docx_bytes()
    out_bytes, placed = inject_comments(docx_bytes, [], {})
    assert placed == 0
    assert out_bytes == docx_bytes


def test_inject_comments_unknown_seg_id_is_skipped_not_raised():
    docx_bytes = _minimal_docx_bytes()
    comments = [{'seg_id': 'does-not-exist', 'author': 'x', 'type': 'conflict', 'comment': 'x'}]
    out_bytes, placed = inject_comments(docx_bytes, comments, {})
    assert placed == 0
    with zipfile.ZipFile(BytesIO(out_bytes)) as z:
        assert 'word/comments.xml' in z.namelist()  # comments.xml still written (empty comment list)


# ---------------------------------------------------------------------------
# fit_check / adapt_lengths — None guards
# ---------------------------------------------------------------------------

def _seg(seg_id, original, translated, keep=False):
    return {'seg_id': seg_id, 'original_text': original, 'translated_text': translated, 'keep_as_is': keep}


_EXTRACT = [
    {'seg_id': 's1', 'location_type': 'txbx', 'part': 'word/document.xml'},
    {'seg_id': 's2', 'location_type': 'txbx', 'part': 'word/document.xml'},
]


def test_check_fit_skips_untranslated_segments():
    segs = [_seg('s1', 'krátký', None), _seg('s2', 'text', 'une traduction beaucoup beaucoup plus longue')]
    flagged, critical = check_fit(_EXTRACT, segs)
    assert all(f['seg_id'] != 's1' for f in flagged)  # no len(None) crash


def test_adapt_lengths_skips_untranslated_segments():
    segs = [_seg('s1', 'krátký', None), _seg('s2', 'ok', 'ok')]
    out, suggestions = adapt_lengths(_EXTRACT, segs)
    assert out[0]['translated_text'] is None  # untouched, no crash


# ---------------------------------------------------------------------------
# Bilingual inline pre-split (charspan) — boundary finder
# ---------------------------------------------------------------------------

_BG_EN = ('Добавяне на снимка и описание за монтаж на пяните\n'
          'Adding a photo and description for the installation of the foam')


def test_char_boundary_script_switch_bg_en():
    # The real segment class from the A321 BG job that detect_language
    # mislabelled en@0.85 (client/src/components/translate/README.md "Known issue").
    found = _audit._find_char_boundary(_BG_EN)
    assert found is not None
    offset, lang_l, lang_r, conf = found
    assert lang_l == 'bg' and lang_r == 'en'
    assert _BG_EN[offset:].startswith('Adding')


def test_char_boundary_sentence_cs_en():
    # Diacritic-free telegraphic Czech + English echo, same run: only the
    # fastText-backed span detector can label the Czech half. Skip rather
    # than fail if the vendored model can't load in this environment.
    from server.services.translation.langdetect import _fasttext_model
    if _fasttext_model() is None:
        import pytest
        pytest.skip('fastText model unavailable')
    text = 'Namontujte matici. Install the nut according to the drawing.'
    found = _audit._find_char_boundary(text)
    assert found is not None
    offset, lang_l, lang_r, _conf = found
    assert (lang_l, lang_r) == ('cs', 'en')
    assert text[offset:].startswith('Install')


def test_char_boundary_none_for_monolingual():
    assert _audit._find_char_boundary(
        'Install the screw. Then install the washer. Check the torque.') is None


def test_char_boundary_unit_suffix_not_mistaken_for_language_switch():
    # A torque value's unit ("Nm") is Latin-script even inside an otherwise
    # all-Cyrillic sentence. Uncounted, it used to be mistaken for the start
    # of the Latin block, splitting one word early and stranding "Nm" on the
    # kept (untranslated) side while the source side lost its unit.
    text = ('Затегнете гайката с въртящ момент 35 Nm.'
            'Tighten the nut to a torque of 35 Nm.')
    found = _audit._find_char_boundary(text)
    assert found is not None
    offset, lang_l, lang_r, _conf = found
    assert lang_l == 'bg' and lang_r == 'en'
    assert offset == text.index('Tighten')
    assert text[:offset].endswith('35 Nm.')


def test_pair_inline_charspan_marks_segment():
    seg = {
        'seg_id': 's1', 'text': _BG_EN, 'detected_lang': 'en',
        'lang_confidence': 0.85, 'pattern_type': 'mono', 'pair_id': None,
    }
    _audit.pair_inline_charspan([seg])
    assert seg['pattern_type'] == 'bilingual_inline_charsplit'
    assert seg['inline_split']['kind'] == 'char'
    assert seg['inline_split']['left_lang'] == 'bg'


# ---------------------------------------------------------------------------
# Per-segment translation planning
# ---------------------------------------------------------------------------

def _row(text, pattern='mono', detected='bg', inline=None, confidence=0.9):
    return {'source_text': text, 'pattern_type': pattern,
            'detected_lang': detected, 'inline_split': inline,
            'lang_confidence': confidence}


def test_plan_charsplit_translates_source_side_only():
    off = _BG_EN.index('Adding')
    row = _row(_BG_EN, 'bilingual_inline_charsplit', 'en',
               {'kind': 'char', 'offset': off, 'left_lang': 'bg', 'right_lang': 'en'})
    plan = _plan_segment_translation(row, 'bg', 'en', 'bilingual')
    assert plan is not None
    assert plan['query'] == _BG_EN[:off].strip()          # BG span only
    full = plan['compose']('Ajout d’une photo et description')
    assert full.endswith(_BG_EN[off:])                    # EN side verbatim
    assert full.startswith('Ajout')


def test_plan_slash_right_side_source():
    text = 'Serrer la vis / Utáhněte šroub'
    off = text.index('/')
    row = _row(text, 'bilingual_inline_slash', 'fr',
               {'kind': 'slash', 'offset': off, 'left_lang': 'fr', 'right_lang': 'cs'})
    plan = _plan_segment_translation(row, 'cs', 'fr', 'bilingual')
    assert plan['query'] == 'Utáhněte šroub'
    assert plan['compose']('Serrez la vis') == 'Serrer la vis / Serrez la vis'


def test_plan_format_split_marks_span_translated():
    row = _row('Utáhněte šroub Tighten the screw', 'bilingual_inline_concat', 'en',
               {'kind': 'format', 'fmt_a': 'b', 'fmt_b': '', 'lang_a': 'cs',
                'lang_b': 'en', 'text_a': 'Utáhněte šroub ', 'text_b': 'Tighten the screw'})
    plan = _plan_segment_translation(row, 'cs', 'en', 'bilingual')
    assert plan['query'] == 'Utáhněte šroub'
    assert plan['kept_text'] == 'Tighten the screw'
    import json as _json
    updated = _json.loads(plan['inline_json'])
    assert updated['span_translated'] is True and updated['source_fmt'] == 'b'


def test_plan_inline_without_source_side_keeps_whole():
    row = _row('Serrer / Tighten', 'bilingual_inline_slash', 'fr',
               {'kind': 'slash', 'offset': 7, 'left_lang': 'fr', 'right_lang': 'en'})
    assert _plan_segment_translation(row, 'bg', 'en', 'bilingual') is None


def test_plan_mono_bilingual_source_vs_kept():
    # bilingual: translate the source-language side, keep the other.
    assert _plan_segment_translation(
        _row('Завършване', detected='bg'), 'bg', 'en', 'bilingual')['query'] == 'Завършване'
    assert _plan_segment_translation(
        _row('Door closing', detected='en'), 'bg', 'en', 'bilingual') is None


def test_plan_mono_monolingual_translates_unless_confident_target():
    # monolingual: everything is translated EXCEPT a confident, long enough
    # target-language passage.
    # A French cognate title fastText may confidently mislabel 'en' at <0.85
    # is still translated (no silent keep).
    assert _plan_segment_translation(
        _row('OPERATION 20 PERCAGE', detected='en', confidence=0.80),
        'fr', 'en', 'monolingual') is not None
    # An undetermined segment is translated.
    assert _plan_segment_translation(
        _row('RONDELLE', detected='??', confidence=0.0),
        'fr', 'en', 'monolingual') is not None
    # A genuine, confident, long English passage is kept verbatim.
    assert _plan_segment_translation(
        _row('This document is the property of LATECOERE and must not be shared.',
             detected='en', confidence=0.99),
        'fr', 'en', 'monolingual') is None
    # A short confident target label is NOT enough to keep (min word count).
    assert _plan_segment_translation(
        _row('Level 1', detected='en', confidence=0.99),
        'fr', 'en', 'monolingual') is not None


# ---------------------------------------------------------------------------
# Rebuild — span-aware dispatch
# ---------------------------------------------------------------------------

_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def _body_with_paragraph():
    from lxml import etree
    xml = (f'<w:body xmlns:w="{_W}">'
           f'<w:p>'
           f'<w:r><w:rPr><w:b/></w:rPr><w:t>Utáhněte šroub </w:t></w:r>'
           f'<w:r><w:t>Tighten the screw</w:t></w:r>'
           f'</w:p>'
           f'</w:body>')
    return etree.fromstring(xml.encode()), [[f'{{{_W}}}p', 0]]


def _b_fmt_hash():
    # fmt_hash of the bold run above, as extract/rebuild compute it.
    from lxml import etree
    rpr = etree.fromstring(f'<w:rPr xmlns:w="{_W}"><w:b/></w:rPr>'.encode())
    return _rebuild._fmt_hash(rpr)


def test_apply_translations_lxml_format_span_keeps_other_side():
    root, p_path = _body_with_paragraph()
    segs = [{'seg_id': 's1', 'original_text': 'Utáhněte šroub Tighten the screw',
             'translated_text': 'Serrez la vis', 'keep_as_is': False}]
    applied = _rebuild.apply_translations_lxml(
        root, segs,
        choice_paths={'s1': p_path}, fallback_paths={'s1': None},
        fmt_signatures={'s1': ''},
        inline_splits={'s1': {'kind': 'format', 'span_translated': True,
                              'source_fmt': _b_fmt_hash()}},
        pattern_types={'s1': 'bilingual_inline_concat'},
    )
    assert applied == 1
    texts = [t.text for t in root.iter(f'{{{_W}}}t')]
    assert texts == ['Serrez la vis', 'Tighten the screw']


def test_apply_translations_lxml_uniform_for_composed_full_text():
    root, p_path = _body_with_paragraph()
    segs = [{'seg_id': 's1', 'original_text': 'Utáhněte šroub Tighten the screw',
             'translated_text': 'Serrez la vis Tighten the screw', 'keep_as_is': False}]
    applied = _rebuild.apply_translations_lxml(
        root, segs,
        choice_paths={'s1': p_path}, fallback_paths={'s1': None},
        fmt_signatures={'s1': ''},
        inline_splits={'s1': {'kind': 'char', 'offset': 15}},
        pattern_types={'s1': 'bilingual_inline_charsplit'},
    )
    assert applied == 1
    joined = ''.join(t.text or '' for t in root.iter(f'{{{_W}}}t'))
    assert joined == 'Serrez la vis Tighten the screw'


# ---------------------------------------------------------------------------
# soffice exact-PDF engine
# ---------------------------------------------------------------------------

def _reset_soffice_discovery():
    soffice._discovered_path = None
    soffice._discovery_error = None
    soffice._engine_version = None


def test_find_soffice_unavailable_when_nothing_configured(monkeypatch, tmp_path):
    _reset_soffice_discovery()
    monkeypatch.setenv('SOFFICE_PATH', '')
    monkeypatch.setenv('SOFFICE_ARCHIVE_VOLUME_PATH', '')
    monkeypatch.setenv('SOFFICE_EXTRACT_DIR', str(tmp_path / 'nothing_here'))
    monkeypatch.setattr(soffice, '_well_known_paths', lambda: [])
    try:
        soffice.find_soffice()
        assert False, 'expected SofficeUnavailable'
    except soffice.SofficeUnavailable as e:
        assert 'SOFFICE_PATH' in str(e)
    status = soffice.soffice_status()
    assert status['available'] is False
    _reset_soffice_discovery()


def test_find_soffice_reports_bad_explicit_path(monkeypatch, tmp_path):
    _reset_soffice_discovery()
    monkeypatch.setenv('SOFFICE_PATH', str(tmp_path / 'missing' / 'soffice'))
    monkeypatch.setenv('SOFFICE_ARCHIVE_VOLUME_PATH', '')
    monkeypatch.setenv('SOFFICE_EXTRACT_DIR', str(tmp_path / 'nothing_here'))
    monkeypatch.setattr(soffice, '_well_known_paths', lambda: [])
    try:
        soffice.find_soffice()
        assert False, 'expected SofficeUnavailable'
    except soffice.SofficeUnavailable as e:
        assert 'does not exist' in str(e)
    _reset_soffice_discovery()


def test_convert_docx_to_pdf_cache_hit_needs_no_engine(monkeypatch, tmp_path):
    """A cached conversion must be served without any soffice binary — the
    before/after documents of a finished job are immutable, so repeat previews
    (and page refreshes) cost nothing even if the engine later breaks."""
    _reset_soffice_discovery()
    docx_bytes = b'PK fake docx payload'
    pdf_bytes = b'%PDF-1.7 fake'
    monkeypatch.setattr(soffice, '_cache_dir', lambda: tmp_path)
    (tmp_path / (hashlib.sha256(docx_bytes).hexdigest() + '.pdf')).write_bytes(pdf_bytes)
    monkeypatch.setattr(
        soffice, 'find_soffice',
        lambda: (_ for _ in ()).throw(AssertionError('engine must not be consulted on cache hit')),
    )
    assert soffice.convert_docx_to_pdf(docx_bytes) == pdf_bytes
