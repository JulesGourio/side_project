"""
rebuild.py - Reconstruct a translated .docx from the original .docx + a
translation.json (or, with --roundtrip, no translation - just read+write
to verify XML round-trip fidelity).

Strategy:
  1. Open the source .docx as a ZipFile.
  2. For each text-bearing XML part:
     a. Pre-register every namespace prefix declared on the part's root
        element (extracted via regex from the raw bytes), so ElementTree
        does NOT mangle prefixes (turning w:p into ns0:p, etc.).
     b. Parse the XML.
     c. Apply translations (or no-op for round-trip).
     d. Serialize back to bytes, preserving the XML declaration.
  3. Write a new .docx by copying every non-modified zip entry byte-for-byte
     and substituting the modified XML parts.

Translation redistribution strategies (per segment):
  - uniform                   - all runs share fmt_hash; put translated text
                                in run[0], blank the rest.
  - bilingual_format_split    - paragraph contains both languages; only
                                replace the runs whose fmt_hash matches the
                                primary-language signature.
  - slash                     - "X / Y" - replace only the X portion.
  - proportional              - mixed formatting; distribute proportionally,
                                using DNT tokens as anchors when possible.

Usage:
    python rebuild.py --roundtrip <input.docx> <output.docx>
    python rebuild.py <input.docx> <translation.json> <output.docx>
"""

from __future__ import annotations
import json
import re
import sys
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
W = f"{{{W_NS}}}"
MC = f"{{{MC_NS}}}"


# ---------------------------------------------------------------------------
# Namespace pre-registration (avoid ns0:, ns1: prefix mangling)
# ---------------------------------------------------------------------------

XMLNS_DECL_RE = re.compile(rb'xmlns:([A-Za-z0-9_]+)="([^"]+)"')
XMLNS_DEFAULT_RE = re.compile(rb'xmlns="([^"]+)"')
ROOT_TAG_RE = re.compile(rb'<(?!\?)\S[^>]*>', re.DOTALL)  # first element after decl


# Pre-register the full conventional Word/OOXML namespace prefix set so that
# ElementTree never has to invent ns10/ns12 placeholders. Even namespaces
# declared deep in the doc (e.g. on inline <pic:pic> elements) get the
# conventional prefix this way.
_OOXML_NS_PRESET = {
    "w":      "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "mc":     "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "r":      "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "m":      "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "v":      "urn:schemas-microsoft-com:vml",
    "o":      "urn:schemas-microsoft-com:office:office",
    "w10":    "urn:schemas-microsoft-com:office:word",
    "w14":    "http://schemas.microsoft.com/office/word/2010/wordml",
    "w15":    "http://schemas.microsoft.com/office/word/2012/wordml",
    "w16":    "http://schemas.microsoft.com/office/word/2018/wordml",
    "w16cid": "http://schemas.microsoft.com/office/word/2016/wordml/cid",
    "w16se":  "http://schemas.microsoft.com/office/word/2015/wordml/symex",
    "w16du":  "http://schemas.microsoft.com/office/word/2023/wordml/word16du",
    "w16sdtdh":"http://schemas.microsoft.com/office/word/2020/wordml/sdtdatahash",
    "wp":     "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "wp14":   "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing",
    "wpc":    "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas",
    "wpg":    "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    "wps":    "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "wpi":    "http://schemas.microsoft.com/office/word/2010/wordprocessingInk",
    "a":      "http://schemas.openxmlformats.org/drawingml/2006/main",
    "a14":    "http://schemas.microsoft.com/office/drawing/2010/main",
    "a15":    "http://schemas.microsoft.com/office/drawing/2012/main",
    "a16":    "http://schemas.microsoft.com/office/drawing/2014/main",
    "pic":    "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "dgm":    "http://schemas.openxmlformats.org/drawingml/2006/diagram",
    "c":      "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "cx":     "http://schemas.microsoft.com/office/drawing/2014/chartex",
    "cx1":    "http://schemas.microsoft.com/office/drawing/2015/9/8/chartex",
    "cx2":    "http://schemas.microsoft.com/office/drawing/2015/10/21/chartex",
    "cx3":    "http://schemas.microsoft.com/office/drawing/2016/5/9/chartex",
    "cx4":    "http://schemas.microsoft.com/office/drawing/2016/5/10/chartex",
    "cx5":    "http://schemas.microsoft.com/office/drawing/2016/5/11/chartex",
    "cx6":    "http://schemas.microsoft.com/office/drawing/2016/5/12/chartex",
    "cx7":    "http://schemas.microsoft.com/office/drawing/2016/5/13/chartex",
    "cx8":    "http://schemas.microsoft.com/office/drawing/2016/5/14/chartex",
    "aink":   "http://schemas.microsoft.com/office/drawing/2016/ink",
    "am3d":   "http://schemas.microsoft.com/office/drawing/2017/model3d",
    "xml":    "http://www.w3.org/XML/1998/namespace",
}


def extract_original_root_tag(xml_bytes: bytes) -> bytes:
    """Extract the complete opening tag of the root element from XML bytes.

    The original root tag includes ALL namespace declarations (even unused
    ones) and attributes like mc:Ignorable. We preserve this exactly so
    Word doesn't reject the file for undeclared namespace prefixes.
    """
    # Skip past XML declaration if present
    search_start = 0
    if xml_bytes.lstrip()[:5] == b"<?xml":
        end_decl = xml_bytes.find(b"?>")
        if end_decl != -1:
            search_start = end_decl + 2
    # Find the root element's opening tag (first < that isn't <?)
    m = ROOT_TAG_RE.search(xml_bytes, search_start)
    if m:
        return m.group(0)
    return b""


def merge_root_tags(new_xml: bytes, orig_root_tag: bytes) -> bytes:
    """Replace the ET-serialized root element opening tag with one that
    merges namespace declarations from BOTH the original and the new.

    - The original root tag has all the declarations Word requires (including
      "unused" prefixes referenced by mc:Ignorable).
    - The new root tag may have ADDITIONAL declarations for namespaces that
      were originally declared locally on inner elements (e.g. pic:, a14:)
      but which ET hoisted to the root.

    We merge: start with original root tag, then inject any xmlns:xxx
    declarations from the new tag that aren't already present.
    """
    if not orig_root_tag:
        return new_xml
    # Find the new root tag
    search_start = 0
    if new_xml.lstrip()[:5] == b"<?xml":
        end_decl = new_xml.find(b"?>")
        if end_decl != -1:
            search_start = end_decl + 2
    m = ROOT_TAG_RE.search(new_xml, search_start)
    if not m:
        return new_xml
    new_root_tag = m.group(0)

    # Collect xmlns declarations from both tags
    orig_ns = {m2.group(1): m2.group(0)
               for m2 in XMLNS_DECL_RE.finditer(orig_root_tag)}
    new_ns = {m2.group(1): m2.group(0)
              for m2 in XMLNS_DECL_RE.finditer(new_root_tag)}

    # Find xmlns declarations in the new tag that the original doesn't have
    extra_decls = []
    for prefix, full_decl in new_ns.items():
        if prefix not in orig_ns:
            extra_decls.append(b" " + full_decl)

    if extra_decls:
        # Insert extra declarations just before the closing ">" of orig tag
        if orig_root_tag.endswith(b"/>"):
            insert_pos = len(orig_root_tag) - 2
            suffix = b"/>"
        else:
            insert_pos = len(orig_root_tag) - 1
            suffix = b">"
        merged = orig_root_tag[:insert_pos] + b"".join(extra_decls) + suffix
    else:
        merged = orig_root_tag

    return new_xml[:m.start()] + merged + new_xml[m.end():]


def register_namespaces_from_xml(xml_bytes: bytes) -> dict[str, str]:
    """Pre-register every conventional OOXML namespace prefix, then scan the
    ENTIRE XML for any additional xmlns declarations (which may be declared
    on inner elements rather than the root). This ensures ET never invents
    ns10/ns12 placeholder prefixes during serialization.

    Returns the merged {prefix: uri} dict."""
    ns_map = dict(_OOXML_NS_PRESET)
    # Pre-register the conventional prefixes
    for prefix, uri in _OOXML_NS_PRESET.items():
        ET.register_namespace(prefix, uri)
    # Scan FULL bytes for any additional xmlns declarations
    for m in XMLNS_DECL_RE.finditer(xml_bytes):
        prefix = m.group(1).decode("ascii")
        uri = m.group(2).decode("utf-8")
        if ns_map.get(prefix) == uri:
            continue  # already registered with same URI
        ns_map[prefix] = uri
        ET.register_namespace(prefix, uri)
    # Default namespace
    m = XMLNS_DEFAULT_RE.search(xml_bytes[:8192])
    if m:
        uri = m.group(1).decode("utf-8")
        ns_map[""] = uri
        ET.register_namespace("", uri)
    return ns_map


# ---------------------------------------------------------------------------
# Path navigation
# ---------------------------------------------------------------------------

def walk_path(root: ET.Element, path: list) -> ET.Element | None:
    """Walk a positional_path [[tag, idx], ...] from root. Returns the
    target element, or None if the path is invalid."""
    cur = root
    for tag, idx in path:
        children = list(cur)
        if idx >= len(children):
            return None
        if children[idx].tag != tag:
            # Path drift - fall back to scanning for matching tag at this index
            # Should not happen if the source doc hasn't changed.
            return None
        cur = children[idx]
    return cur


# ---------------------------------------------------------------------------
# Text replacement strategies
# ---------------------------------------------------------------------------

def get_runs(p: ET.Element) -> list[ET.Element]:
    """Return all <w:r> elements in document order under p, NOT descending
    into drawings/text-boxes (those are separate segments)."""
    out = []
    SKIP = {W + "drawing", MC + "AlternateContent", W + "pict"}

    def walk(elem):
        for child in elem:
            if child.tag in SKIP:
                continue
            if child.tag == W + "r":
                out.append(child)
            walk(child)
    walk(p)
    return out


def get_t_elements(run: ET.Element) -> list[ET.Element]:
    """Return all <w:t> elements directly inside a run (preserves order)."""
    return [t for t in run if t.tag == W + "t"]


def replace_uniform(p: ET.Element, new_text: str) -> bool:
    """Strategy: put all text in the first <w:t>, blank the rest. Used when
    all runs share the same formatting (~80-90% of segments)."""
    runs = get_runs(p)
    if not runs:
        return False
    first_t = None
    extra_ts = []
    for r in runs:
        for t in get_t_elements(r):
            if first_t is None:
                first_t = t
            else:
                extra_ts.append(t)
    if first_t is None:
        return False
    first_t.text = new_text
    # Preserve whitespace by setting xml:space if necessary
    if new_text != new_text.strip():
        first_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for t in extra_ts:
        t.text = ""
    return True


def replace_bilingual_format_split(p: ET.Element, new_text: str,
                                   primary_fmt: str) -> bool:
    """For a paragraph that holds both languages distinguished by formatting:
    replace only the <w:t> elements whose parent run's fmt_hash matches
    primary_fmt. Other runs (the kept-language) are left untouched.

    new_text is the translated PRIMARY-side text only.
    """
    runs = get_runs(p)
    if not runs:
        return False
    matching_ts = []
    for r in runs:
        rPr = r.find(W + "rPr")
        if _fmt_hash(rPr) == primary_fmt:
            matching_ts.extend(get_t_elements(r))
    if not matching_ts:
        return False
    matching_ts[0].text = new_text
    if new_text != new_text.strip():
        matching_ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for t in matching_ts[1:]:
        t.text = ""
    return True


def replace_slash_split(p: ET.Element, new_left: str,
                        original_left_text: str) -> bool:
    """For 'X / Y' segments: replace only the left half (everything before
    the '/') with new_left. The '/' separator and right half are preserved.

    Implementation: collect all <w:t> text in order, find the '/' position,
    replace only the characters before it with new_left.
    """
    runs = get_runs(p)
    all_ts = []
    for r in runs:
        all_ts.extend(get_t_elements(r))
    if not all_ts:
        return False
    full = "".join(t.text or "" for t in all_ts)
    if "/" not in full:
        return False
    slash_idx = full.index("/")
    # New full text: new_left + " / " + everything after the original "/"
    # We preserve the original separator/spacing characters around the slash
    # by keeping characters from the original "/" position onward.
    after_slash = full[slash_idx:]  # starts with "/"
    # Strip trailing whitespace from new_left and prepend a single space
    # before "/" if the original had one.
    sep_prefix = " " if (slash_idx > 0 and full[slash_idx - 1] == " ") else ""
    new_full = new_left.rstrip() + sep_prefix + after_slash
    # Distribute new_full across the existing <w:t> elements proportionally
    return _redistribute_across_ts(all_ts, new_full)


def replace_proportional(p: ET.Element, new_text: str) -> bool:
    """Mixed formatting fallback: distribute new_text across all <w:t>
    elements proportionally to original character lengths.

    This is approximate - format-span boundaries may shift mid-word. Used
    only when no better strategy applies.
    """
    runs = get_runs(p)
    all_ts = []
    for r in runs:
        all_ts.extend(get_t_elements(r))
    if not all_ts:
        return False
    return _redistribute_across_ts(all_ts, new_text)


def _redistribute_across_ts(ts: list[ET.Element], new_text: str) -> bool:
    """Distribute new_text across ts proportionally to original lengths."""
    if not ts:
        return False
    if len(ts) == 1:
        ts[0].text = new_text
        if new_text != new_text.strip():
            ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return True
    orig_lengths = [len(t.text or "") for t in ts]
    total_orig = sum(orig_lengths) or 1
    n = len(new_text)
    cursor = 0
    for i, t in enumerate(ts):
        if i == len(ts) - 1:
            chunk = new_text[cursor:]
        else:
            take = round(n * orig_lengths[i] / total_orig)
            chunk = new_text[cursor:cursor + take]
            cursor += len(chunk)
        t.text = chunk
        if chunk != chunk.strip():
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return True


def _fmt_hash(rPr: ET.Element | None) -> str:
    """Mirror of extract.fmt_hash - keep in sync with extract.py."""
    if rPr is None:
        return ""
    interesting = {"sz", "szCs", "b", "bCs", "i", "iCs", "u", "color",
                   "rFonts", "lang", "vertAlign", "highlight", "strike"}
    parts = []
    for child in rPr:
        tag = child.tag.split("}", 1)[-1]
        if tag not in interesting:
            continue
        attrs = ",".join(f"{k.split('}',1)[-1]}={v}"
                         for k, v in sorted(child.attrib.items()))
        parts.append(f"{tag}({attrs})" if attrs else tag)
    return "|".join(sorted(parts))


# ---------------------------------------------------------------------------
# Apply translations to a parsed XML tree
# ---------------------------------------------------------------------------

def apply_translations(root: ET.Element, segments_for_part: list[dict],
                       choice_paths: dict, fallback_paths: dict,
                       fmt_signatures: dict, inline_splits: dict,
                       pattern_types: dict) -> int:
    """Apply translations to one parsed XML tree. Returns count of
    successfully applied translations."""
    applied = 0
    for seg in segments_for_part:
        seg_id = seg["seg_id"]
        if seg.get("keep_as_is"):
            continue
        new_text = seg["translated_text"]
        ch_path = choice_paths.get(seg_id)
        fb_path = fallback_paths.get(seg_id)
        fmt_sig = fmt_signatures.get(seg_id, "")
        inline_split = inline_splits.get(seg_id)
        pattern = pattern_types.get(seg_id, "mono")

        # Both copies (mc:Choice + mc:Fallback) must be updated identically
        for path in (ch_path, fb_path):
            if not path:
                continue
            target = walk_path(root, path)
            if target is None:
                continue
            if pattern == "bilingual_inline_slash" and inline_split:
                replace_slash_split(target, new_text, seg["original_text"])
            elif pattern == "bilingual_inline_concat" and inline_split:
                # Only replace runs whose fmt matches the primary side
                primary_fmt = inline_split.get("fmt_a", fmt_sig)
                replace_bilingual_format_split(target, new_text, primary_fmt)
            else:
                # Default: uniform replacement
                replace_uniform(target, new_text)
        applied += 1
    return applied


# ---------------------------------------------------------------------------
# lxml-based translation application
# ---------------------------------------------------------------------------
# Mirrors apply_translations() but works on lxml etree Elements.
# lxml elements have the same .tag, .text, .attrib, list(elem), findall()
# API as stdlib ET, so the path-walking and run-replacement logic is reused.

def apply_translations_lxml(root, segments_for_part, choice_paths,
                            fallback_paths, fmt_signatures, inline_splits,
                            pattern_types) -> int:
    """Apply translations to an lxml tree. Returns count applied."""
    applied = 0
    for seg in segments_for_part:
        seg_id = seg["seg_id"]
        if seg.get("keep_as_is"):
            continue
        new_text = seg["translated_text"]
        ch_path = choice_paths.get(seg_id)
        fb_path = fallback_paths.get(seg_id)
        fmt_sig = fmt_signatures.get(seg_id, "")
        inline_split = inline_splits.get(seg_id)
        pattern = pattern_types.get(seg_id, "mono")

        for path in (ch_path, fb_path):
            if not path:
                continue
            target = walk_path(root, path)
            if target is None:
                continue
            if inline_split and inline_split.get("span_translated"):
                # Format-split bilingual segment: translated_text holds ONLY
                # the source-side span (marked span_translated by the
                # translation stage) — splice it into the runs whose
                # formatting matches that side; the kept-language runs
                # survive verbatim. replace_uniform here would wipe the kept
                # side. No uniform fallback on failure for the same reason —
                # an unmatched fmt leaves the paragraph untouched.
                replace_bilingual_format_split(
                    target, new_text, inline_split.get("source_fmt") or fmt_sig)
            else:
                # Whole-segment and char/slash-split translations carry the
                # full replacement text (kept-language portion included, the
                # translation stage composed it), so uniform replacement is
                # correct for them.
                replace_uniform(target, new_text)
        applied += 1
    return applied


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def rebuild_docx(src_docx: str, out_docx: str,
                 translations_by_part: dict[str, list[dict]] | None = None,
                 metadata: dict | None = None) -> None:
    """If translations_by_part is None or empty, this is a pure round-trip
    (parse and re-serialize every part with no changes)."""
    src = Path(src_docx)
    out = Path(out_docx)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}

    with zipfile.ZipFile(src, "r") as zin:
        names = zin.namelist()
        # Collect new bytes for parts we'll modify
        new_bytes: dict[str, bytes] = {}
        modified_parts = (set(translations_by_part.keys())
                          if translations_by_part else set())
        # During round-trip mode we still re-serialize every text-bearing part
        # to test fidelity; pick them up from the zip namelist.
        if not translations_by_part:
            modified_parts = {n for n in names
                              if n.startswith("word/") and n.endswith(".xml")
                              and (n.endswith("/document.xml")
                                   or "header" in n or "footer" in n
                                   or n.endswith("footnotes.xml")
                                   or n.endswith("endnotes.xml"))}

        for part in modified_parts:
            if part not in names:
                continue
            xml_bytes = zin.read(part)

            # Use lxml for parsing + serialization. Unlike stdlib
            # ElementTree, lxml preserves:
            #   - All namespace declarations (even "unused" ones)
            #   - Attribute ordering
            #   - Whitespace in self-closing tags
            #   - xml:space attributes
            # This is critical because Word rejects files where
            # mc:Ignorable references undeclared prefixes, and subtle
            # whitespace/attribute changes can cause layout shifts.
            try:
                from lxml import etree as lxml_etree
                parser = lxml_etree.XMLParser(
                    remove_blank_text=False,
                    strip_cdata=False,
                )
                lxml_root = lxml_etree.fromstring(xml_bytes, parser)
            except ImportError:
                raise RuntimeError(
                    "lxml is required for reliable .docx round-tripping. "
                    "Install with: pip install lxml"
                )

            if translations_by_part and part in translations_by_part:
                segs = translations_by_part[part]
                ch_paths = metadata.get("choice_paths", {})
                fb_paths = metadata.get("fallback_paths", {})
                fmt_sigs = metadata.get("fmt_signatures", {})
                inline_splits = metadata.get("inline_splits", {})
                pattern_types = metadata.get("pattern_types", {})
                applied = apply_translations_lxml(
                    lxml_root, segs, ch_paths, fb_paths, fmt_sigs,
                    inline_splits, pattern_types,
                )
                print(f"  {part}: applied {applied}/{len(segs)} translations")

            # lxml serialization preserves the original structure faithfully.
            raw = lxml_etree.tostring(
                lxml_root,
                xml_declaration=True,
                encoding="UTF-8",
                standalone=True,
            )
            # lxml cosmetic fixes to match Word's expected XML form:
            #   - double quotes in declaration (lxml uses single)
            #   - CRLF after declaration (lxml uses LF)
            raw = raw.replace(
                b"<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
                b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                1,
            )
            raw = raw.replace(b'?>\n<', b'?>\r\n<', 1)
            new_bytes[part] = raw

        # Write output zip: copy unchanged entries byte-for-byte, substitute
        # modified parts.
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                if name in new_bytes:
                    zout.writestr(name, new_bytes[name])
                else:
                    zout.writestr(name, zin.read(name))

    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Translation-file ingestion
# ---------------------------------------------------------------------------

def load_translation_with_metadata(translation_path: str,
                                   extract_path: str,
                                   audit_path: str) -> tuple[dict, dict]:
    """Load translation.json and cross-reference with extract.json + audit.json
    to obtain: per-segment xml paths, fmt signatures, inline_split info, and
    pattern types. Returns (translations_by_part, metadata)."""
    tr = json.loads(Path(translation_path).read_text(encoding="utf-8"))
    ex = json.loads(Path(extract_path).read_text(encoding="utf-8"))
    au = json.loads(Path(audit_path).read_text(encoding="utf-8"))

    by_id_extract = {s["seg_id"]: s for s in ex["segments"]}
    by_id_audit = {s["seg_id"]: s for s in au["segments"]}

    metadata = {
        "choice_paths": {},
        "fallback_paths": {},
        "fmt_signatures": {},
        "inline_splits": {},
        "pattern_types": {},
    }
    translations_by_part: dict[str, list[dict]] = defaultdict(list)
    for tseg in tr["segments"]:
        seg_id = tseg["seg_id"]
        ext = by_id_extract.get(seg_id)
        aud = by_id_audit.get(seg_id)
        if not ext:
            continue
        translations_by_part[ext["part"]].append(tseg)
        metadata["choice_paths"][seg_id] = ext["xml_choice_path"]
        metadata["fallback_paths"][seg_id] = ext.get("xml_fallback_path")
        metadata["fmt_signatures"][seg_id] = ext["fmt_signature"]
        if aud:
            metadata["pattern_types"][seg_id] = aud["pattern_type"]
            metadata["inline_splits"][seg_id] = aud.get("inline_split")
    return translations_by_part, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def usage():
    print(__doc__.strip().splitlines()[-3].strip())  # last "Usage:" line


def main(argv: list[str]) -> None:
    from job_utils import parse_job_flag, JobPaths, update_manifest
    job_dir, remaining = parse_job_flag(argv[1:])
    if job_dir:
        jp = JobPaths(job_dir)
        translations_by_part, metadata = load_translation_with_metadata(
            str(jp.translation_json), str(jp.extract_json), str(jp.audit_json),
        )
        rebuild_docx(str(jp.source_docx), str(jp.output_docx),
                      translations_by_part, metadata)
        update_manifest(job_dir, status="rebuilt")
        return

    if len(argv) >= 2 and argv[1] == "--roundtrip":
        if len(argv) != 4:
            print("Usage: python rebuild.py --roundtrip <input.docx> <output.docx>")
            sys.exit(1)
        rebuild_docx(argv[2], argv[3])
        return
    if len(argv) != 4:
        print("Usage: python rebuild.py <input.docx> <translation.json> <output.docx>")
        print("       python rebuild.py --roundtrip <input.docx> <output.docx>")
        sys.exit(1)
    src_docx, translation_json, out_docx = argv[1], argv[2], argv[3]
    # Derive extract/audit paths from translation.json's filename. Convention:
    #   <stem>.translation.json  pairs with  <stem>.extract.json + <stem>.audit.json
    tr_path = Path(translation_json)
    stem = tr_path.stem.replace(".translation", "")
    reports_dir = tr_path.parent
    extract_path = reports_dir / f"{stem}.extract.json"
    audit_path = reports_dir / f"{stem}.audit.json"
    translations_by_part, metadata = load_translation_with_metadata(
        translation_json, str(extract_path), str(audit_path),
    )
    rebuild_docx(src_docx, out_docx, translations_by_part, metadata)


if __name__ == "__main__":
    main(sys.argv)
