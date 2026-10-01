"""
extract.py - Deep-structured extraction of translatable segments from a .docx.

For each text-bearing XML part in the .docx, walks the structure and emits
one Segment per translatable paragraph. Handles:

  - Body paragraphs (direct text in <w:p>)
  - Tables (<w:tbl>/<w:tr>/<w:tc>), with row/cell coordinates
  - Drawing text boxes (<w:txbxContent> inside <mc:AlternateContent>),
    descending only into mc:Choice but recording the matching mc:Fallback
    path so rebuild can update both copies.
  - Structured document tags (<w:sdt> - including TOC content controls),
    treated as a transparent container.

For each paragraph we also record:
  - Run-level formatting fingerprint (fmt_hash per run, fmt_signature per
    paragraph) used by audit.py to identify primary vs secondary language
    by formatting alone.
  - A positional XML path (list of (tag, child_idx) tuples) from the part
    root to the paragraph element. rebuild.py uses this path to locate the
    exact node to modify - more reliable than re-parsing or XPath.

Usage:  python extract.py <input.docx> <output.json>
"""

from __future__ import annotations
import json
import sys
import zipfile
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
W = f"{{{W_NS}}}"
MC = f"{{{MC_NS}}}"

# Tags we should NOT descend into when collecting direct runs of a paragraph,
# because they contain text that we extract separately as its own segment(s).
SKIP_DESCENT_TAGS = {
    W + "drawing",
    MC + "AlternateContent",
    W + "pict",
}

# footnotes.xml/endnotes.xml/comments.xml wrap each note/comment's paragraphs
# one level deeper than document.xml's body (<w:footnote>/<w:endnote>/
# <w:comment>, each holding its own <w:p> children) — extract_part must
# descend into these, or every paragraph inside them is silently skipped.
NOTE_WRAPPER_TAGS = {W + "footnote", W + "endnote", W + "comment"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_text_parts(zip_names: list[str]) -> list[str]:
    """Return the list of XML parts in a .docx that may contain user text."""
    candidates = []
    for name in zip_names:
        if not name.startswith("word/"):
            continue
        if not name.endswith(".xml"):
            continue
        base = name.rsplit("/", 1)[-1]
        # Match: document, headerN, footerN, footnotes, endnotes, comments
        if (base == "document.xml"
                or re.fullmatch(r"header\d+\.xml", base)
                or re.fullmatch(r"footer\d+\.xml", base)
                or base in ("footnotes.xml", "endnotes.xml", "comments.xml")):
            candidates.append(name)
    # Stable order: document first, then headers, footers, footnotes, endnotes
    def sort_key(n: str) -> tuple:
        b = n.rsplit("/", 1)[-1]
        priority = (
            0 if b == "document.xml"
            else 1 if b.startswith("header")
            else 2 if b.startswith("footer")
            else 3 if b == "footnotes.xml"
            else 4 if b == "endnotes.xml"
            else 5
        )
        # Sort headerN/footerN numerically, not lexicographically — a plain
        # string compare puts "header10.xml" before "header2.xml", breaking
        # reading order (and any frontend heuristic keyed off emission order)
        # for documents with 10+ distinct headers or footers.
        m = re.search(r"\d+", b)
        num = int(m.group()) if m else 0
        return (priority, num, b)
    return sorted(candidates, key=sort_key)


def build_parent_map(root: ET.Element) -> dict:
    """Map id(child) -> parent for every element in the tree."""
    parent_map = {}
    for parent in root.iter():
        for child in parent:
            parent_map[id(child)] = parent
    return parent_map


def positional_path(elem: ET.Element, root: ET.Element, parent_map: dict) -> list:
    """Walk from root down to elem, returning [[tag, child_idx], ...]."""
    if elem is root:
        return []
    path = []
    cur = elem
    while cur is not root:
        parent = parent_map.get(id(cur))
        if parent is None:
            # elem isn't actually in this tree; return what we have
            break
        # Find child index by identity (not equality - elements may compare equal)
        idx = next((i for i, c in enumerate(parent) if c is cur), None)
        if idx is None:
            break
        path.append([cur.tag, idx])
        cur = parent
    path.reverse()
    return path


def fmt_hash(rPr: ET.Element | None) -> str:
    """Deterministic, compact hash of a run's <w:rPr> formatting properties.

    We only consider properties that distinguish languages or convey real
    formatting (size, bold, italic, color, font, language). We ignore noise
    like whitespace/proofing properties.
    """
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


def walk_skip_drawings(elem: ET.Element):
    """Yield elem and descendants, skipping subtrees that contain text-box
    or drawing content (those are extracted separately)."""
    yield elem
    for child in elem:
        if child.tag in SKIP_DESCENT_TAGS:
            continue
        yield from walk_skip_drawings(child)


def collect_direct_runs(p: ET.Element) -> tuple[str, list[dict]]:
    """Return (concatenated_text, runs_list) for runs directly in this <w:p>,
    excluding runs inside drawings/text-boxes."""
    runs = []
    idx = 0
    for elem in walk_skip_drawings(p):
        if elem.tag != W + "r":
            continue
        rPr = elem.find(W + "rPr")
        h = fmt_hash(rPr)
        for t in elem:
            tag = t.tag
            if tag == W + "t":
                txt = t.text or ""
                if txt:
                    runs.append({"idx": idx, "text": txt, "fmt_hash": h})
                    idx += 1
            elif tag == W + "tab":
                runs.append({"idx": idx, "text": "\t", "fmt_hash": h})
                idx += 1
            elif tag == W + "br":
                runs.append({"idx": idx, "text": "\n", "fmt_hash": h})
                idx += 1
    text = "".join(r["text"] for r in runs)
    return text, runs


def dominant_fmt_signature(runs: list[dict]) -> str:
    """Pick the most common non-empty fmt_hash among runs, weighted by chars."""
    if not runs:
        return ""
    weights: Counter = Counter()
    for r in runs:
        if r["text"].strip():
            weights[r["fmt_hash"]] += len(r["text"])
    if not weights:
        return runs[0]["fmt_hash"]
    return weights.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Segment construction
# ---------------------------------------------------------------------------

def make_segment(*, seg_id, part, location_type, body_p_idx,
                 table_coords=None, txbx_path=None,
                 para_idx_in_container, text, runs,
                 xml_choice_path, xml_fallback_path=None) -> dict:
    return {
        "seg_id": seg_id,
        "part": part,
        "location_type": location_type,
        "body_p_idx": body_p_idx,
        "table_coords": table_coords,
        "txbx_path": txbx_path,
        "para_idx_in_container": para_idx_in_container,
        "text": text,
        "runs": runs,
        "fmt_signature": dominant_fmt_signature(runs),
        "xml_choice_path": xml_choice_path,
        "xml_fallback_path": xml_fallback_path,
    }


# ---------------------------------------------------------------------------
# Extractors per location type
# ---------------------------------------------------------------------------

def extract_paragraph_direct(p: ET.Element, *, part: str, body_p_idx: int,
                             location_type: str, root: ET.Element,
                             parent_map: dict,
                             para_idx_in_container: int,
                             table_coords=None) -> list[dict]:
    """Extract any directly-contained text from a <w:p> (not its drawings)."""
    text, runs = collect_direct_runs(p)
    if not text.strip():
        return []
    if location_type == "table":
        tc = table_coords or {}
        seg_id = (f"{part}#table.bt{tc.get('table_body_idx',body_p_idx)}"
                  f".r{tc.get('row_idx','?')}.c{tc.get('cell_idx','?')}"
                  f"#p{para_idx_in_container}")
    elif location_type == "sdt":
        seg_id = f"{part}#sdt.bp{body_p_idx}#p{para_idx_in_container}"
    else:
        seg_id = f"{part}#body_direct.bp{body_p_idx}#p{para_idx_in_container}"
    return [make_segment(
        seg_id=seg_id,
        part=part,
        location_type=location_type,
        body_p_idx=body_p_idx,
        table_coords=table_coords,
        para_idx_in_container=para_idx_in_container,
        text=text, runs=runs,
        xml_choice_path=positional_path(p, root, parent_map),
    )]


def extract_drawings_in(host: ET.Element, *, part: str, body_p_idx: int,
                        root: ET.Element, parent_map: dict) -> list[dict]:
    """Find every <mc:AlternateContent> directly within `host` (a <w:p> or
    table cell) and extract text from its text boxes.

    For each AlternateContent we descend into mc:Choice, locate every
    <w:txbxContent>, and emit one segment per <w:p> inside. The matching
    paragraph in mc:Fallback (same ordinal) is recorded via xml_fallback_path
    so rebuild can update both copies.
    """
    segments = []
    # Use iter() to find AlternateContent at any depth under host (not just
    # direct children), since wrappers like mc:Choice/wp:inline can nest.
    # But we still process each top-level AC once.
    for ac_idx, ac in enumerate(host.iter(MC + "AlternateContent")):
        choice = ac.find(MC + "Choice")
        fallback = ac.find(MC + "Fallback")
        if choice is None:
            continue

        choice_txbxes = list(choice.iter(W + "txbxContent"))
        fallback_txbxes = list(fallback.iter(W + "txbxContent")) if fallback is not None else []

        for tx_idx, txbx in enumerate(choice_txbxes):
            fb_txbx = fallback_txbxes[tx_idx] if tx_idx < len(fallback_txbxes) else None
            ch_paras = txbx.findall(W + "p")
            fb_paras = fb_txbx.findall(W + "p") if fb_txbx is not None else []
            for p_idx, txp in enumerate(ch_paras):
                text, runs = collect_direct_runs(txp)
                if not text.strip():
                    continue
                fb_path = None
                if p_idx < len(fb_paras):
                    fb_path = positional_path(fb_paras[p_idx], root, parent_map)
                segments.append(make_segment(
                    seg_id=f"{part}#txbx.bp{body_p_idx}.ac{ac_idx}.tx{tx_idx}#p{p_idx}",
                    part=part,
                    location_type="txbx",
                    body_p_idx=body_p_idx,
                    txbx_path={
                        "body_p_idx": body_p_idx,
                        "ac_idx": ac_idx,
                        "txbx_idx": tx_idx,
                        "para_idx": p_idx,
                    },
                    para_idx_in_container=p_idx,
                    text=text, runs=runs,
                    xml_choice_path=positional_path(txp, root, parent_map),
                    xml_fallback_path=fb_path,
                ))
    return segments


def extract_table(tbl: ET.Element, *, part: str, body_p_idx,
                  root: ET.Element, parent_map: dict) -> list[dict]:
    """body_p_idx is an int for a top-level table, or a compound string id
    for a table nested inside a cell (see the w:tbl branch below) — either
    way it only ever feeds the human-readable seg_id, never a positional
    lookup, so the type doesn't matter to rebuild.py (which relocates nodes
    via xml_choice_path)."""
    segments = []
    for row_idx, tr in enumerate(tbl.findall(W + "tr")):
        for cell_idx, tc in enumerate(tr.findall(W + "tc")):
            tcoords = {"table_body_idx": body_p_idx,
                       "row_idx": row_idx, "cell_idx": cell_idx}
            # Iterate the cell's direct children in document order (not just
            # findall(w:p)) so a table nested inside this cell — common in
            # spec/assembly tables with sub-tables — isn't silently skipped;
            # findall(w:p) alone used to drop 100% of nested-table content.
            p_idx = 0
            for child in tc:
                if child.tag == W + "p":
                    segments.extend(extract_paragraph_direct(
                        child, part=part, body_p_idx=body_p_idx,
                        location_type="table", root=root, parent_map=parent_map,
                        para_idx_in_container=p_idx,
                        table_coords=tcoords,
                    ))
                    segments.extend(extract_drawings_in(
                        child, part=part, body_p_idx=body_p_idx,
                        root=root, parent_map=parent_map,
                    ))
                    p_idx += 1
                elif child.tag == W + "tbl":
                    # Compound id keeps nested-table seg_ids distinct from
                    # this cell's own paragraphs and from a nested table at
                    # the same row/cell coordinates elsewhere in the document.
                    nested_id = f"{body_p_idx}.nt{row_idx}.{cell_idx}"
                    segments.extend(extract_table(
                        child, part=part, body_p_idx=nested_id,
                        root=root, parent_map=parent_map,
                    ))
    return segments


def extract_sdt(sdt: ET.Element, *, part: str, body_p_idx: int,
                root: ET.Element, parent_map: dict) -> list[dict]:
    """Treat <w:sdt>/<w:sdtContent> as a transparent container - process
    its child paragraphs/tables like body children."""
    segments = []
    sdt_content = sdt.find(W + "sdtContent")
    if sdt_content is None:
        return segments
    p_counter = 0
    for child in sdt_content:
        if child.tag == W + "p":
            segments.extend(extract_paragraph_direct(
                child, part=part, body_p_idx=body_p_idx,
                location_type="sdt", root=root, parent_map=parent_map,
                para_idx_in_container=p_counter,
            ))
            segments.extend(extract_drawings_in(
                child, part=part, body_p_idx=body_p_idx,
                root=root, parent_map=parent_map,
            ))
            p_counter += 1
        elif child.tag == W + "tbl":
            segments.extend(extract_table(
                child, part=part, body_p_idx=body_p_idx,
                root=root, parent_map=parent_map,
            ))
        elif child.tag == W + "sdt":
            segments.extend(extract_sdt(
                child, part=part, body_p_idx=body_p_idx,
                root=root, parent_map=parent_map,
            ))
    return segments


# ---------------------------------------------------------------------------
# Per-part driver
# ---------------------------------------------------------------------------

def extract_part(xml_bytes: bytes, part: str) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    parent_map = build_parent_map(root)

    # The "body" we iterate is <w:body> for document.xml, otherwise the root
    # itself (headers, footers, footnotes, endnotes have no body wrapper).
    body = root.find(W + "body")
    if body is None:
        body = root

    segments = []
    for body_idx, child in enumerate(body):
        tag = child.tag
        if tag == W + "p":
            segments.extend(extract_paragraph_direct(
                child, part=part, body_p_idx=body_idx,
                location_type="body_direct", root=root, parent_map=parent_map,
                para_idx_in_container=0,
            ))
            segments.extend(extract_drawings_in(
                child, part=part, body_p_idx=body_idx,
                root=root, parent_map=parent_map,
            ))
        elif tag == W + "tbl":
            segments.extend(extract_table(
                child, part=part, body_p_idx=body_idx,
                root=root, parent_map=parent_map,
            ))
        elif tag == W + "sdt":
            segments.extend(extract_sdt(
                child, part=part, body_p_idx=body_idx,
                root=root, parent_map=parent_map,
            ))
        elif tag in NOTE_WRAPPER_TAGS:
            note_p_idx = 0
            for sub in child:
                if sub.tag == W + "p":
                    segments.extend(extract_paragraph_direct(
                        sub, part=part, body_p_idx=body_idx,
                        location_type="body_direct", root=root, parent_map=parent_map,
                        para_idx_in_container=note_p_idx,
                    ))
                    segments.extend(extract_drawings_in(
                        sub, part=part, body_p_idx=body_idx,
                        root=root, parent_map=parent_map,
                    ))
                    note_p_idx += 1
                elif sub.tag == W + "tbl":
                    segments.extend(extract_table(
                        sub, part=part, body_p_idx=body_idx,
                        root=root, parent_map=parent_map,
                    ))
        # other tags (sectPr, etc.) carry no translatable text
    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(src: str, out: str, _job_dir: Path | None = None) -> None:
    src_path = Path(src)
    out_path = Path(out)

    with zipfile.ZipFile(src_path) as z:
        text_parts = discover_text_parts(z.namelist())
        all_segments = []
        for part in text_parts:
            xml_bytes = z.read(part)
            all_segments.extend(extract_part(xml_bytes, part))

    payload = {
        "source_file": str(src_path),
        "parts": text_parts,
        "segment_count": len(all_segments),
        "segments": all_segments,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # Print a summary - useful when invoked from Claude Code
    by_loc: Counter = Counter(s["location_type"] for s in all_segments)
    by_part: Counter = Counter(s["part"] for s in all_segments)
    print(f"Extracted {len(all_segments)} segments from {len(text_parts)} parts -> {out_path}")
    print(f"  By location: {dict(by_loc)}")
    print(f"  By part: {dict(by_part)}")

    if _job_dir:
        from job_utils import update_manifest
        update_manifest(_job_dir, status="extracted")


if __name__ == "__main__":
    from job_utils import parse_job_flag, JobPaths
    job_dir, remaining = parse_job_flag(sys.argv[1:])
    if job_dir:
        jp = JobPaths(job_dir)
        main(str(jp.source_docx), str(jp.extract_json), _job_dir=job_dir)
    else:
        if len(sys.argv) != 3:
            print("Usage: python extract.py <input.docx> <output.json>")
            sys.exit(1)
        main(sys.argv[1], sys.argv[2])
