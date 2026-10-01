"""
fit_check.py - Detect text overflow BEFORE rebuild.

Analyzes translation.json against the original .docx to flag segments where
the translated text is likely to overflow its container:

  1. Header/footer text — ANY increase in character count is flagged (these
     are in fixed-width table cells and even 1 extra character can wrap).
  2. Text-box text — flagged if translated text is >THRESHOLD longer than
     the original (default 130%). Text boxes have fixed dimensions.
  3. Table cells — flagged if significantly longer (>150%).
  4. Body paragraphs — generally safe (they reflow), only flagged if >200%.

For each flagged segment, reports:
  - seg_id, location type
  - Original text + length
  - Translated text + length
  - Overflow ratio (%)
  - Container type and severity

Usage:
    python fit_check.py <extract.json> <translation.json>
    python fit_check.py <extract.json> <translation.json> --threshold 120

Exit code: 0 if no critical overflows, 1 if header/footer overflow detected.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path


# Threshold defaults (% of original length)
THRESHOLDS = {
    "header": 100,      # ANY increase in header/footer is critical
    "footer": 100,
    "txbx": 130,         # text boxes: 30% tolerance
    "table": 150,        # table cells: 50% tolerance
    "body_direct": 200,  # body paragraphs reflow, generous tolerance
    "sdt": 200,
}

SEVERITY = {
    "header": "CRITICAL",
    "footer": "CRITICAL",
    "txbx": "WARNING",
    "table": "INFO",
    "body_direct": "INFO",
    "sdt": "INFO",
}


def classify_location(seg_id: str, location_type: str, part: str) -> str:
    """Map a segment to its overflow-sensitivity category."""
    if "header" in part:
        return "header"
    if "footer" in part:
        return "footer"
    return location_type


def main(argv: list[str]) -> int:
    from job_utils import parse_job_flag, JobPaths
    job_dir, remaining = parse_job_flag(argv[1:])
    if job_dir:
        jp = JobPaths(job_dir)
        extract_path = str(jp.extract_json)
        translation_path = str(jp.translation_json)
        # Allow additional flags (e.g. --threshold) from remaining args
        custom_threshold = None
        for i, a in enumerate(remaining):
            if a == "--threshold" and i + 1 < len(remaining):
                custom_threshold = int(remaining[i + 1])
    else:
        if len(argv) < 3:
            print("Usage: python fit_check.py <extract.json> <translation.json> "
                  "[--threshold N]")
            return 1

        extract_path = argv[1]
        translation_path = argv[2]

        # Optional global threshold override
        custom_threshold = None
        for i, a in enumerate(argv):
            if a == "--threshold" and i + 1 < len(argv):
                custom_threshold = int(argv[i + 1])

    extract = json.loads(Path(extract_path).read_text(encoding="utf-8"))
    translation = json.loads(Path(translation_path).read_text(encoding="utf-8"))

    ext_by_id = {s["seg_id"]: s for s in extract["segments"]}
    flagged = []
    critical = 0

    for tseg in translation["segments"]:
        if tseg.get("keep_as_is"):
            continue
        seg_id = tseg["seg_id"]
        ext = ext_by_id.get(seg_id)
        if not ext:
            continue

        orig_len = len(tseg["original_text"])
        new_len = len(tseg["translated_text"])
        if orig_len == 0:
            continue

        ratio = (new_len / orig_len) * 100
        loc = classify_location(seg_id, ext["location_type"], ext["part"])
        threshold = custom_threshold or THRESHOLDS.get(loc, 130)
        severity = SEVERITY.get(loc, "INFO")

        if ratio > threshold:
            flagged.append({
                "seg_id": seg_id,
                "location": loc,
                "severity": severity,
                "orig_len": orig_len,
                "new_len": new_len,
                "ratio": round(ratio, 1),
                "threshold": threshold,
                "original_text": tseg["original_text"][:80],
                "translated_text": tseg["translated_text"][:80],
            })
            if severity == "CRITICAL":
                critical += 1

    # Sort by severity then ratio
    sev_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    flagged.sort(key=lambda f: (sev_order.get(f["severity"], 9), -f["ratio"]))

    # Report
    print(f"Fit check: {len(translation['segments'])} segments analyzed")
    print(f"Flagged: {len(flagged)} overflow risks "
          f"({critical} critical, "
          f"{sum(1 for f in flagged if f['severity']=='WARNING')} warnings, "
          f"{sum(1 for f in flagged if f['severity']=='INFO')} info)")
    print()

    if not flagged:
        print("No overflow risks detected.")
        return 0

    for f in flagged:
        print(f"  [{f['severity']:8s}] {f['seg_id']}")
        print(f"           {f['location']} | {f['orig_len']}→{f['new_len']} chars "
              f"({f['ratio']}% of original, threshold {f['threshold']}%)")
        print(f"           orig: {f['original_text']}")
        print(f"           new:  {f['translated_text']}")
        print()

    if critical > 0:
        print(f"CRITICAL: {critical} header/footer overflow(s) detected. "
              f"These WILL cause layout issues (line wrapping, page inflation).")
        print("Fix: use shorter translations for header/footer text.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
