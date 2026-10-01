"""
length_adapt.py - Suggest shorter translations for overflowing segments.

Reads the fit-check flagged segments and proposes abbreviations using
common French aerospace shorthand conventions:

  - "Monter" → "Mt."
  - "Installer" → "Inst."
  - "avec mastic d'étanchéité des surfaces de contact" → "avec mastic d'interposition"
  - "selon IF2.03.08" → "cf. IF2.03.08"
  - "dans la plage de" → "entre"
  - "avec la rondelle" → "avec rdl."
  - Remove redundant articles ("la", "le", "les", "l'") where safe

For CRITICAL segments (headers), suggests completely alternative translations.

Usage:
    python length_adapt.py <extract.json> <translation.json> [--apply]

Without --apply: prints suggestions only.
With --apply: writes an updated translation.json with shortened text.
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path


# French aerospace abbreviation rules (ordered by priority)
SHORTENINGS_FR = [
    # Long sealant phrase → standard short form
    (r"avec mastic d'étanchéité des surfaces de contact\s*;\s*", "avec mastic d'interposition ; "),
    (r"avec mastic d'étanchéité des surfaces de contact", "avec mastic d'interposition"),
    # "selon" → "cf." for specification references
    (r"\bselon (IF\d)", r"cf. \1"),
    # "dans la plage de X - Y" → "entre X - Y"
    (r"\bdans la plage de\b", "entre"),
    # Remove "la/le/l'" before part designations when safe
    (r"\bMonter la vis\b", "Monter vis"),
    (r"\bMonter la rondelle\b", "Monter rdl."),
    (r"\bMonter la pièce\b", "Monter pièce"),
    (r"\bMonter la goupille\b", "Monter goupille"),
    (r"\bMonter l'écrou\b", "Monter écrou"),
    (r"\bMonter l'arbre\b", "Monter arbre"),
    (r"\bavec la rondelle\b", "avec rdl."),
    (r"\bet la rondelle\b", "et rdl."),
    (r"\bpuis l'écrou\b", "puis écrou"),
    (r"\bet l'écrou\b", "et écrou"),
    # "sous la tête, sur la tige et sous la rondelle" → "sous tête/tige/rdl."
    (r"sous la tête, sur la tige et sous la rondelle", "sous tête/tige/rdl."),
    (r"sous la tête et sur la tige", "sous tête/tige"),
    # "Assembler" → "Ass." for very tight fits
    (r"\bAssembler les pièces\b", "Ass. pièces"),
    (r"\bAssembler la pièce\b", "Ass. pièce"),
]

# Header-specific alternative translations
HEADER_ALTERNATIVES_FR = {
    "INSTRUCTION TECHNOLOGIQUE": "NOTICE TECHNIQUE",
    "INSTRUCTION\nTECHNOLOGIQUE": "NOTICE TECHNIQUE",
}

# Starter rules for non-French targets: generic, conservative (redundant-article
# removal, common connector shortenings) — NOT aerospace-domain-reviewed like the
# French table above. A translator familiar with each language's maintenance-doc
# conventions should extend these; an unlisted language is a clean no-op, not a guess.
SHORTENINGS_EN = [
    (r"\baccording to (IF\d)", r"cf. \1"),
    (r"\bin the range of\b", "between"),
]
SHORTENINGS_ES = [
    (r"\bseg[uú]n (IF\d)", r"cf. \1"),
    (r"\ben el rango de\b", "entre"),
]
SHORTENINGS_DE = [
    (r"\bgem[aä]ß (IF\d)", r"vgl. \1"),
]
SHORTENINGS_CS = [
    (r"\bpodle (IF\d)", r"viz \1"),
]
SHORTENINGS_BG = [
    (r"\bсъгласно (IF\d)", r"вж. \1"),
]

SHORTENINGS_BY_LANG: dict[str, list[tuple[str, str]]] = {
    "fr": SHORTENINGS_FR,
    "en": SHORTENINGS_EN,
    "es": SHORTENINGS_ES,
    "de": SHORTENINGS_DE,
    "cs": SHORTENINGS_CS,
    "bg": SHORTENINGS_BG,
}
HEADER_ALTERNATIVES_BY_LANG: dict[str, dict[str, str]] = {
    "fr": HEADER_ALTERNATIVES_FR,
}

# Back-compat aliases for existing callers/tests (French was the only language before).
SHORTENINGS = SHORTENINGS_FR
HEADER_ALTERNATIVES = HEADER_ALTERNATIVES_FR


def shorten(text: str, target_len: int, lang: str = "fr") -> tuple[str, list[str]]:
    """Apply abbreviation rules to shorten text. Returns (shortened, rules_applied).

    Unlisted `lang` values resolve to an empty rule set (clean no-op) rather
    than falling back to French, which would silently apply the wrong language's
    phrasing.
    """
    applied = []
    result = text
    for pattern, replacement in SHORTENINGS_BY_LANG.get(lang, []):
        new = re.sub(pattern, replacement, result)
        if new != result:
            applied.append(f"{pattern[:40]}... → {replacement[:30]}...")
            result = new
        if len(result) <= target_len:
            break
    return result, applied


def main(argv: list[str]) -> int:
    from job_utils import parse_job_flag, JobPaths

    def _pop_target_lang(args: list[str]) -> tuple[list[str], str]:
        lang = "fr"
        rest = []
        i = 0
        while i < len(args):
            if args[i] == "--target-lang" and i + 1 < len(args):
                lang = args[i + 1]
                i += 2
                continue
            rest.append(args[i])
            i += 1
        return rest, lang

    job_dir, remaining = parse_job_flag(argv[1:])
    remaining, target_lang = _pop_target_lang(remaining)
    if job_dir:
        jp = JobPaths(job_dir)
        extract_path = str(jp.extract_json)
        translation_path = str(jp.translation_json)
        apply_changes = "--apply" in remaining
    else:
        if len(remaining) < 2:
            print("Usage: python length_adapt.py <extract.json> <translation.json> "
                  "[--apply] [--target-lang <lang>]")
            return 1

        extract_path = remaining[0]
        translation_path = remaining[1]
        apply_changes = "--apply" in remaining

    extract = json.loads(Path(extract_path).read_text(encoding="utf-8"))
    translation = json.loads(Path(translation_path).read_text(encoding="utf-8"))

    ext_by_id = {s["seg_id"]: s for s in extract["segments"]}

    THRESHOLDS = {"header": 100, "footer": 100, "txbx": 130, "table": 150}
    changes = 0
    suggestions = []

    for tseg in translation["segments"]:
        if tseg.get("keep_as_is"):
            continue
        seg_id = tseg["seg_id"]
        ext = ext_by_id.get(seg_id)
        if not ext:
            continue

        orig_len = len(tseg["original_text"])
        new_text = tseg["translated_text"]
        new_len = len(new_text)
        if orig_len == 0:
            continue

        part = ext["part"]
        loc = "header" if "header" in part else "footer" if "footer" in part else ext["location_type"]
        threshold = THRESHOLDS.get(loc)
        if threshold is None:
            continue
        ratio = (new_len / orig_len) * 100
        if ratio <= threshold:
            continue

        # Try header alternatives first
        if loc in ("header", "footer"):
            alt = HEADER_ALTERNATIVES_BY_LANG.get(target_lang, {}).get(new_text.strip())
            if alt:
                shortened = alt
                rules = ["header alternative"]
            else:
                shortened, rules = shorten(new_text, orig_len, lang=target_lang)
        else:
            shortened, rules = shorten(new_text, int(orig_len * threshold / 100), lang=target_lang)

        shortened_ratio = (len(shortened) / orig_len) * 100
        fits = shortened_ratio <= threshold

        suggestions.append({
            "seg_id": seg_id,
            "location": loc,
            "orig_len": orig_len,
            "before": new_text,
            "before_len": new_len,
            "after": shortened,
            "after_len": len(shortened),
            "fits": fits,
            "rules": rules,
        })

        if apply_changes:
            tseg["translated_text"] = shortened
            changes += 1

    # Report
    for s in suggestions:
        status = "FIXED" if s["fits"] else "STILL OVER"
        print(f"[{status:10s}] {s['seg_id']} ({s['location']})")
        print(f"  before ({s['before_len']:3d} chars): {s['before'][:100]}")
        print(f"  after  ({s['after_len']:3d} chars): {s['after'][:100]}")
        print(f"  target: ≤{s['orig_len']} chars | rules: {', '.join(s['rules']) or 'none'}")
        print()

    fixed = sum(1 for s in suggestions if s["fits"])
    still_over = sum(1 for s in suggestions if not s["fits"])
    print(f"Total: {len(suggestions)} segments processed, "
          f"{fixed} fixed, {still_over} still over threshold")

    if apply_changes and changes:
        Path(translation_path).write_text(
            json.dumps(translation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nApplied {changes} changes to {translation_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
