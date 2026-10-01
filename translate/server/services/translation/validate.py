"""
validate.py - Structural integrity validation of a .docx file.

Checks performed:
  1. ZIP integrity — all entries readable
  2. XML validity — every .xml and .rels part parses without error
  3. Namespace integrity — all mc:Ignorable prefixes are declared
  4. No auto-prefixes — no ns0:/ns1:/ns2: artifacts from ElementTree
  5. Content_Types consistency — all referenced parts exist in the zip
  6. Segment fidelity — if an extract.json is provided, verify segment
     count and text matches (for post-rebuild validation)
  7. Byte identity — for non-modified parts, verify they're unchanged
     from the original (if original path is provided)

Usage:
    python validate.py <file.docx>
    python validate.py <file.docx> --original <original.docx>
    python validate.py <file.docx> --extract <extract.json>

Exit code: 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def check_zip_integrity(path: str) -> list[str]:
    errors = []
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad:
                errors.append(f"ZIP corruption in: {bad}")
    except Exception as e:
        errors.append(f"Cannot open as ZIP: {e}")
    return errors


def check_xml_validity(path: str) -> list[str]:
    errors = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                try:
                    ET.fromstring(z.read(name))
                except ET.ParseError as e:
                    errors.append(f"Invalid XML in {name}: {e}")
    return errors


def check_namespace_integrity(path: str) -> list[str]:
    """Check that mc:Ignorable references only declared prefixes."""
    errors = []
    MC_IGN_RE = re.compile(rb'mc:Ignorable="([^"]+)"')
    XMLNS_PREFIX_RE = re.compile(rb'xmlns:([A-Za-z0-9_]+)="')
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".xml"):
                continue
            data = z.read(name)
            mc_m = MC_IGN_RE.search(data[:16000])
            if not mc_m:
                continue
            prefixes = mc_m.group(1).decode().split()
            # Collect ALL xmlns declarations in the entire file (not just root)
            declared = set(m.group(1).decode()
                           for m in XMLNS_PREFIX_RE.finditer(data))
            missing = [p for p in prefixes if p not in declared]
            if missing:
                errors.append(
                    f"{name}: mc:Ignorable references undeclared prefixes: "
                    f"{missing}"
                )
    return errors


def check_auto_prefixes(path: str) -> list[str]:
    """Check for ns0:/ns1:/ns2: auto-generated namespace prefixes."""
    errors = []
    AUTO_NS_RE = re.compile(rb'<ns\d+:')
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".xml"):
                continue
            data = z.read(name)
            hits = AUTO_NS_RE.findall(data[:50000])
            if hits:
                errors.append(
                    f"{name}: contains auto-generated namespace prefixes "
                    f"({len(hits)} occurrences, e.g. {hits[0].decode()})"
                )
    return errors


def check_content_types(path: str) -> list[str]:
    """Verify [Content_Types].xml references parts that exist."""
    errors = []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if "[Content_Types].xml" not in names:
            errors.append("Missing [Content_Types].xml")
            return errors
        ct = z.read("[Content_Types].xml").decode("utf-8", errors="replace")
        # Find all Override PartName references
        for m in re.finditer(r'PartName="/([^"]+)"', ct):
            part = m.group(1)
            if part not in names:
                errors.append(
                    f"[Content_Types].xml references non-existent part: {part}"
                )
    return errors


def check_segment_fidelity(path: str, extract_json: str) -> list[str]:
    """Compare segment count and IDs against a reference extract.

    word/comments.xml is excluded: when review comments were injected into
    this rebuild (comments.py's inject_comments), extraction now finds those
    new comment paragraphs too (extract.py's NOTE_WRAPPER_TAGS) — that's
    intentional growth from THIS rebuild, not the lost/duplicated body
    content this check exists to catch.
    """
    errors = []
    try:
        from .extract import main as extract_main
        import tempfile
        # Extract from the file being validated
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        extract_main(path, tmp)
        new = json.loads(Path(tmp).read_text(encoding="utf-8"))
        Path(tmp).unlink()

        ref = json.loads(Path(extract_json).read_text(encoding="utf-8"))
        ref_segs = [s for s in ref["segments"] if s["part"] != "word/comments.xml"]
        new_segs = [s for s in new["segments"] if s["part"] != "word/comments.xml"]
        ref_ids = {s["seg_id"] for s in ref_segs}
        new_ids = {s["seg_id"] for s in new_segs}
        missing = ref_ids - new_ids
        extra = new_ids - ref_ids
        if missing:
            errors.append(f"Missing {len(missing)} segments vs reference")
        if extra:
            errors.append(f"Extra {len(extra)} segments vs reference")
        if len(ref_segs) != len(new_segs):
            errors.append(
                f"Segment count mismatch: ref={len(ref_segs)} "
                f"new={len(new_segs)}"
            )
    except Exception as e:
        errors.append(f"Segment fidelity check failed: {e}")
    return errors


def check_byte_identity(path: str, original: str) -> list[str]:
    """For parts that should be untouched, verify byte identity."""
    errors = []
    # Parts that should never change (no user text). word/endnotes.xml and
    # word/footnotes.xml are deliberately NOT here: extract.py now translates
    # their paragraphs, so they legitimately change on any rebuild that
    # actually has footnote/endnote content.
    IMMUTABLE = {"word/styles.xml", "word/settings.xml", "word/fontTable.xml",
                 "word/webSettings.xml", "word/numbering.xml", "word/theme/theme1.xml"}
    with zipfile.ZipFile(path) as z1, zipfile.ZipFile(original) as z2:
        for name in IMMUTABLE:
            if name not in z1.namelist() or name not in z2.namelist():
                continue
            a = z1.read(name)
            b = z2.read(name)
            if a != b:
                errors.append(
                    f"{name} differs from original "
                    f"(orig={len(b)} new={len(a)} delta={len(a)-len(b)})"
                )
    return errors


def check_residual_source_language(path: str, source_lang: str) -> list[str]:
    """Check for segments still in the source language that should have been
    translated. Uses Cyrillic detection for BG, diacritic patterns for CS."""
    errors = []
    try:
        from .extract import main as extract_main
        from .langdetect import detect_language
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        extract_main(path, tmp)
        data = json.loads(Path(tmp).read_text(encoding="utf-8"))
        Path(tmp).unlink()

        # Language-specific unique characters for confirming residual text.
        # The generic detector can confuse CS↔FR (shared diacritics éáí).
        # We require source-language-UNIQUE characters as confirmation.
        UNIQUE_CHARS = {
            "cs": set("řůěŘŮĚ"),       # Czech-only among supported set
            "bg": None,                  # Use Cyrillic block instead
            "de": set("ßÄÖÜäöü"),
            "es": set("ñÑ¿¡"),
        }
        unique_chars = UNIQUE_CHARS.get(source_lang)

        residual = []
        for s in data["segments"]:
            text = s["text"]
            if not text.strip() or len(text.strip()) < 3:
                continue

            if source_lang == "bg":
                # Cyrillic presence is definitive
                cyr = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
                if cyr > 3:
                    residual.append(s)
            elif unique_chars:
                # For Latin-script source languages, require unique chars
                # to avoid false positives (FR flagged as CS, etc.)
                if any(c in unique_chars for c in text):
                    residual.append(s)
            else:
                # Fallback: use detector with high threshold
                lang, conf = detect_language(text)
                if lang == source_lang and conf > 0.85:
                    residual.append(s)

        if residual:
            errors.append(
                f"{len(residual)} segments still contain {source_lang.upper()} text:"
            )
            for s in residual[:10]:
                errors.append(f"  {s['seg_id']}: {s['text'][:100]}")
            if len(residual) > 10:
                errors.append(f"  ... and {len(residual) - 10} more")
    except Exception as e:
        errors.append(f"Residual language check failed: {e}")
    return errors


def main(argv: list[str]) -> int:
    from job_utils import parse_job_flag, JobPaths, update_manifest
    job_dir, remaining = parse_job_flag(argv[1:])
    if job_dir:
        jp = JobPaths(job_dir)
        path = str(jp.output_docx)
        original = str(jp.source_docx)
        extract = str(jp.extract_json)
        source_lang = jp.source_lang
    else:
        if len(argv) < 2:
            print("Usage: python validate.py <file.docx> [--original <orig.docx>] "
                  "[--extract <extract.json>] [--source-lang XX]")
            return 1
        path = argv[1]
        original = None
        extract = None
        source_lang = None
        i = 2
        while i < len(argv):
            if argv[i] == "--original" and i + 1 < len(argv):
                original = argv[i + 1]
                i += 2
            elif argv[i] == "--extract" and i + 1 < len(argv):
                extract = argv[i + 1]
                i += 2
            elif argv[i] == "--source-lang" and i + 1 < len(argv):
                source_lang = argv[i + 1].lower()
                i += 2
            else:
                i += 1

    all_errors = []
    checks = [
        ("ZIP integrity", check_zip_integrity(path)),
        ("XML validity", check_xml_validity(path)),
        ("Namespace integrity", check_namespace_integrity(path)),
        ("Auto-prefix detection", check_auto_prefixes(path)),
        ("Content_Types consistency", check_content_types(path)),
    ]
    if extract:
        checks.append(("Segment fidelity", check_segment_fidelity(path, extract)))
    if original:
        checks.append(("Byte identity (immutable parts)", check_byte_identity(path, original)))
    if source_lang:
        checks.append(("Residual source language", check_residual_source_language(path, source_lang)))

    total_errors = 0
    for name, errors in checks:
        status = "PASS" if not errors else "FAIL"
        print(f"  [{status}] {name}")
        for e in errors:
            print(f"         {e}")
        total_errors += len(errors)

    print()
    if total_errors == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"FAILED: {total_errors} error(s) found")

    if job_dir:
        update_manifest(job_dir, status="validated")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
