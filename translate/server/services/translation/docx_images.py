"""OCR + translation of text baked into images embedded in a .docx.

Opt-in (the user picks which images to process, since not every embedded
image carries text worth translating and each one costs a vision-LLM call).
Unlike the abandoned native-PDF pipeline, embedded docx images are already
standalone files in the zip (word/media/imageN.*) — no page rendering/
cropping needed, the raw bytes go straight to the vision call.

Restitution never touches the image's pixels: a lesson from the PDF work is
that a real technical drawing's text is usually scattered across the whole
image (title, paragraph, several labelled boxes), not one caption-friendly
area, and resizing the image to fit an overlay would distort it inside its
existing Word frame (which has its own fixed width/height, independent of
the image's native pixels). Instead, the translation is inserted as a plain,
immediately-visible paragraph right after the image in the document flow.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import zipfile
from typing import Any, Dict, List, Tuple

from PIL import Image

from ..llm import call_llm_json

logger = logging.getLogger(__name__)

# Prefix on every inserted translation paragraph — also used by
# processors/translation.py's segment_fidelity check to recognize and
# exclude this deliberate content growth (same idea as that check's
# existing word/comments.xml exclusion for injected Word comments, just
# identified by content instead of by XML part since these paragraphs live
# inside word/document.xml itself).
IMAGE_TRANSLATION_MARKER = '[Traduction]'

_RASTER_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
_THUMBNAIL_MAX_PX = 160

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
RELS_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
W = f'{{{W_NS}}}'
R = f'{{{R_NS}}}'
A = f'{{{A_NS}}}'

_OCR_PROMPT = (
    'This image is a crop from a technical aerospace maintenance/assembly '
    'instruction document. Transcribe VERBATIM any text visible in it '
    '(titles, instructions, labels, callouts, part numbers). Preserve line '
    'breaks between distinct text elements (e.g. title vs. body vs. a boxed '
    'label) using \\n. If there is truly no legible text, return an empty '
    'string.\n'
    'Return ONLY a JSON object of the exact shape {"text": "<transcription '
    'or empty string>"} — no extra prose, no markdown fence.'
)


def _make_thumbnail(data: bytes) -> str | None:
    """Small JPEG thumbnail, base64-encoded — returns None if Pillow can't
    open the bytes (shouldn't happen for anything list_docx_images already
    let through, but ocr_docx_images calls this too and best-effort fits
    its own never-fail-the-job stance)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.thumbnail((_THUMBNAIL_MAX_PX, _THUMBNAIL_MAX_PX))
        buf = io.BytesIO()
        img.convert('RGB').save(buf, format='JPEG', quality=70)
        return base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception as e:
        logger.warning('docx_images: could not thumbnail image: %s', e)
        return None


def _content_hash(data: bytes) -> str:
    """Exact-content hash — deliberately not a perceptual/near-duplicate
    hash: embedded docx images that are "the same picture" (a repeated
    logo/header graphic reused across many pages) are byte-identical media
    files in the zip, so an exact hash is enough to group them, with no risk
    of merging two genuinely different diagrams that merely look similar."""
    return hashlib.sha256(data).hexdigest()


def list_docx_images(docx_bytes: bytes) -> List[Dict[str, Any]]:
    """List embedded raster images with a small thumbnail for a selection UI.

    Each entry carries a content `hash` and `repeat_count` (how many other
    embedded images are byte-identical to it) so the UI can badge likely
    logos/repeated header graphics — OCR only ever runs once per unique
    hash (see ocr_docx_images), so selecting every instance of a repeated
    image costs exactly one LLM call, not one per occurrence.

    Non-raster media (.emf/.wmf vector metafiles, common from old Excel
    paste-ins) are skipped — out of scope for v1, Pillow can't OCR-prep them
    anyway.
    """
    out: List[Dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
        media_names = sorted(n for n in z.namelist() if n.startswith('word/media/'))
        for name in media_names:
            if not name.lower().endswith(_RASTER_EXTS):
                continue
            data = z.read(name)
            thumb_b64 = _make_thumbnail(data)
            if thumb_b64 is None:
                continue
            out.append({
                'filename': name.rsplit('/', 1)[-1],
                'media_path': name,
                'size_bytes': len(data),
                'thumbnail_base64': thumb_b64,
                'hash': _content_hash(data),
            })

    hash_counts: Dict[str, int] = {}
    for img in out:
        hash_counts[img['hash']] = hash_counts.get(img['hash'], 0) + 1
    for img in out:
        img['repeat_count'] = hash_counts[img['hash']]
    return out


async def ocr_docx_images(
    docx_bytes: bytes, selected_filenames: List[str], host: str, token: str, endpoint: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Transcribe the selected embedded images via a vision LLM call.

    Deduplicated by exact content hash first — a logo or header graphic
    reused across many pages is one LLM call, not one per occurrence, and
    every filename sharing that hash gets the same transcription applied.

    Best-effort per unique image: a failed or empty transcription just skips
    that group, never fails the caller. Returns (results, usage_log) —
    usage_log carries one {filename, usage} entry per call that actually
    reached the endpoint (the representative filename OCR'd for its group),
    for the caller to record via the same translation_llm_calls ledger real
    text-translation batches use.
    """
    selected = set(selected_filenames)
    # hash -> {media_path, filename, data} for one representative per group,
    # plus every selected filename sharing that hash (for fan-out below).
    groups: Dict[str, Dict[str, Any]] = {}
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
        for name in z.namelist():
            if not name.startswith('word/media/'):
                continue
            filename = name.rsplit('/', 1)[-1]
            if filename not in selected:
                continue
            data = z.read(name)
            h = _content_hash(data)
            group = groups.setdefault(h, {'filenames': [], 'representative': None})
            group['filenames'].append(filename)
            if group['representative'] is None:
                group['representative'] = {'filename': filename, 'name': name, 'data': data}

    results: List[Dict[str, str]] = []
    usage_log: List[Dict[str, Any]] = []
    for h, group in groups.items():
        rep = group['representative']
        b64 = base64.b64encode(rep['data']).decode('ascii')
        mime = 'image/png' if rep['name'].lower().endswith('.png') else 'image/jpeg'
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': _OCR_PROMPT},
            {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}},
        ]}]
        try:
            result, usage = await call_llm_json(host, token, endpoint, messages, max_tokens=1024)
        except Exception as e:
            logger.warning('docx_images: transcription failed for %s (hash shared by %d file(s)): %s',
                            rep['filename'], len(group['filenames']), e)
            continue
        usage_log.append({'filename': rep['filename'], 'usage': usage})
        text = (result.get('text') or '').strip()
        if not text:
            continue
        thumb_b64 = _make_thumbnail(rep['data'])
        for filename in group['filenames']:
            results.append({
                'filename': filename, 'media_path': f'word/media/{filename}', 'text': text,
                'thumbnail_base64': thumb_b64,
            })
    return results, usage_log


def _local_name(tag: str) -> str:
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


def _load_image_rels(z: zipfile.ZipFile) -> Dict[str, str]:
    """rId -> word/media/... target, for relationships of type .../image."""
    from lxml import etree as lxml_etree

    rels_path = 'word/_rels/document.xml.rels'
    if rels_path not in z.namelist():
        return {}
    root = lxml_etree.fromstring(z.read(rels_path))
    out = {}
    for rel in root:
        if not rel.get('Type', '').endswith('/image'):
            continue
        target = rel.get('Target', '')
        rid = rel.get('Id', '')
        if rid and target:
            out[rid] = f"word/{target.lstrip('/')}" if not target.startswith('word/') else target
    return out


def find_image_anchors(docx_bytes: bytes, filenames: List[str]) -> Dict[str, int]:
    """Body position (same indexing as extract.py's body_p_idx: the 0-based
    index of an element among ALL direct children of <w:body>) of the
    paragraph or table hosting each requested image's <a:blip>, so OCR'd
    image segments can be ordered near their real location instead of always
    landing at the end of the segments list (see order_ocr_segments).

    Walking up from the <a:blip> to whichever ancestor is a direct child of
    <w:body> handles both a plain paragraph and a table cell transparently —
    a picture inside a table cell resolves to the table's own body index,
    which is exactly what extract_table already uses as body_p_idx for every
    segment in that table (see extract.py). Only word/document.xml is
    scanned (headers/footers out of scope, matching this module's other
    functions); an image with no resolvable anchor (nested inside a text
    box, e.g.) is simply absent from the returned dict.
    """
    from lxml import etree as lxml_etree

    wanted = set(filenames)
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
        if 'word/document.xml' not in z.namelist():
            return {}
        rid_to_media = _load_image_rels(z)
        root = lxml_etree.fromstring(z.read('word/document.xml'))

    body = next((c for c in root if _local_name(c.tag) == 'body'), None)
    if body is None:
        return {}
    body_children = list(body)

    anchors: Dict[str, int] = {}
    for blip in root.iter(f'{A}blip'):
        rid = blip.get(f'{R}embed')
        if not rid or rid not in rid_to_media:
            continue
        filename = rid_to_media[rid].rsplit('/', 1)[-1]
        if filename not in wanted or filename in anchors:
            continue
        node = blip
        top = None
        while node is not None:
            parent = node.getparent()
            if parent is body:
                top = node
                break
            node = parent
        if top is None:
            continue
        try:
            anchors[filename] = body_children.index(top)
        except ValueError:
            continue
    return anchors


def order_ocr_segments(
    base_segments: List[Dict[str, Any]],
    ocr_segments: List[Dict[str, Any]],
    anchors: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Splice OCR'd image segments into `base_segments` near the body
    position of the paragraph/table hosting their image (see
    find_image_anchors) instead of always appending them after every real
    segment — insertion order is the only "document order" signal that
    survives into the DB (segments are listed `ORDER BY id`, see
    translate.py), so this is what actually controls where an image shows
    up in the Segments panel. Falls back to the end for anything without a
    resolved anchor (headers/footers, an image nested in a text box — out
    of v1 scope, see module docstring).
    """
    result = list(base_segments)
    to_place: List[Tuple[int, Dict[str, Any]]] = []
    tail: List[Dict[str, Any]] = []
    for seg in ocr_segments:
        filename = seg['xml_choice_path']['media_filename']
        anchor = anchors.get(filename)
        if anchor is None:
            tail.append(seg)
        else:
            to_place.append((anchor, seg))
    to_place.sort(key=lambda pair: pair[0])

    last_anchor = None
    insert_at = 0
    for anchor, seg in to_place:
        if anchor != last_anchor:
            insert_at = next(
                (i for i, s in enumerate(result) if s.get('body_p_idx') is not None and s['body_p_idx'] > anchor),
                len(result),
            )
            last_anchor = anchor
        result.insert(insert_at, seg)
        insert_at += 1

    result.extend(tail)
    return result


def insert_image_translations(docx_bytes: bytes, translations: Dict[str, str]) -> bytes:
    """Insert a visible paragraph right after each translated image's own
    paragraph in word/document.xml. `translations` maps media filename
    (e.g. 'image4.jpg') -> translated text. Everything else in the zip is
    copied byte-for-byte, same convention as processors/translation.py's
    rebuild_docx_bytes.
    """
    from lxml import etree as lxml_etree

    with zipfile.ZipFile(io.BytesIO(docx_bytes), 'r') as zin:
        names = zin.namelist()
        doc_path = 'word/document.xml'
        if doc_path not in names or not translations:
            return docx_bytes

        rid_to_media = _load_image_rels(zin)
        # filename -> translated text, matched against the media path's basename.
        by_filename = translations

        parser = lxml_etree.XMLParser(remove_blank_text=False, strip_cdata=False)
        root = lxml_etree.fromstring(zin.read(doc_path), parser)

        # Find each drawing's enclosing top-level paragraph, resolve its
        # embedded image to a media filename, and collect (paragraph, text)
        # pairs to insert after — collected first since mutating the tree
        # while iterating it is unsafe.
        to_insert: List[Tuple[Any, str]] = []
        seen_media: set = set()
        for blip in root.iter(f'{A}blip'):
            rid = blip.get(f'{R}embed')
            if not rid or rid not in rid_to_media:
                continue
            media_path = rid_to_media[rid]
            filename = media_path.rsplit('/', 1)[-1]
            text = by_filename.get(filename)
            if not text or media_path in seen_media:
                continue
            # Walk up to the nearest ancestor <w:p> (the paragraph hosting
            # this drawing, whatever its own nesting inside mc:AlternateContent).
            p = blip
            while p is not None and _local_name(p.tag) != 'p':
                p = p.getparent()
            if p is None:
                continue
            seen_media.add(media_path)
            to_insert.append((p, text))

        for p, text in to_insert:
            new_p = lxml_etree.Element(f'{W}p')
            p.addnext(new_p)
            r = lxml_etree.SubElement(new_p, f'{W}r')
            rpr_run = lxml_etree.SubElement(r, f'{W}rPr')
            lxml_etree.SubElement(rpr_run, f'{W}i')
            # A literal '\n' inside <w:t> is not a line break in OOXML — each
            # line needs its own <w:t>, separated by explicit <w:br/>.
            lines = [f'{IMAGE_TRANSLATION_MARKER} {line}' if i == 0 else line
                     for i, line in enumerate(text.split('\n'))]
            for i, line in enumerate(lines):
                if i > 0:
                    lxml_etree.SubElement(r, f'{W}br')
                t = lxml_etree.SubElement(r, f'{W}t')
                t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
                t.text = line

        raw = lxml_etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        raw = raw.replace(
            b"<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
            b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', 1,
        )
        raw = raw.replace(b'?>\n<', b'?>\r\n<', 1)

        out_buf = io.BytesIO()
        with zipfile.ZipFile(out_buf, 'w', zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                zout.writestr(name, raw if name == doc_path else zin.read(name))

    return out_buf.getvalue()
