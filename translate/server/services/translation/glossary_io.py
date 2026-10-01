"""
glossary_io.py - Unified multi-language glossary and pattern-based DNT list.

Glossary: glossary/terms.csv — one row per concept, one column per language.
    term_id, en, fr, cs, bg, de, es, domain, notes, added_date

DNT: glossary/dnt.csv — pattern-based do-not-translate rules.
    pattern, type, match_mode, notes
    match_mode: exact | prefix | glob | regex
"""

from __future__ import annotations
import csv
import fnmatch
import re
from datetime import date
from pathlib import Path

# Resolve glossary dir relative to project root (parent of tools/)
GLOSSARY_DIR = Path(__file__).resolve().parent.parent / "glossary"
TERMS_FILE = GLOSSARY_DIR / "terms.csv"
DNT_FILE = GLOSSARY_DIR / "dnt.csv"

LANGUAGES = ["en", "fr", "cs", "bg", "de", "es"]
TERMS_FIELDS = ["term_id"] + LANGUAGES + ["domain", "notes", "added_date"]
DNT_FIELDS = ["pattern", "type", "match_mode", "notes"]


# ---------------------------------------------------------------------------
# Unified glossary
# ---------------------------------------------------------------------------

def load_glossary() -> list[dict]:
    """Load all glossary terms. Returns list of row dicts."""
    if not TERMS_FILE.exists():
        return []
    rows = []
    with TERMS_FILE.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def save_glossary(rows: list[dict]) -> None:
    """Overwrite the glossary with the given rows."""
    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r.get("term_id", ""))
    with TERMS_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TERMS_FIELDS)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({k: row.get(k, "") for k in TERMS_FIELDS})


def _next_term_id(rows: list[dict]) -> str:
    """Generate next T0001-style ID."""
    max_id = 0
    for row in rows:
        tid = row.get("term_id", "")
        if tid.startswith("T") and tid[1:].isdigit():
            max_id = max(max_id, int(tid[1:]))
    return f"T{max_id + 1:04d}"


def lookup(term: str, source_lang: str, target_lang: str,
           rows: list[dict] | None = None) -> str | None:
    """Find a glossary row where source_lang column matches `term` (case-
    insensitive) and return the target_lang column value. Returns None if
    not found or target column is empty."""
    if rows is None:
        rows = load_glossary()
    term_lower = term.strip().lower()
    for row in rows:
        val = (row.get(source_lang) or "").strip()
        if val.lower() == term_lower:
            target = (row.get(target_lang) or "").strip()
            return target if target else None
    return None


def add_term(translations: dict[str, str], domain: str = "",
             notes: str = "", rows: list[dict] | None = None) -> str:
    """Add a new term row. `translations` is a dict like {"en": "screw", "fr": "vis", "cs": "šroub"}.
    Returns the assigned term_id. Saves to disk."""
    if rows is None:
        rows = load_glossary()
    tid = _next_term_id(rows)
    row = {"term_id": tid, "domain": domain, "notes": notes,
           "added_date": date.today().isoformat()}
    for lang in LANGUAGES:
        row[lang] = translations.get(lang, "")
    rows.append(row)
    save_glossary(rows)
    return tid


def merge_terms(term_pairs: list[dict], source_lang: str,
                target_lang: str) -> int:
    """Bulk-add terms from a translation session. Each dict in term_pairs
    should have keys matching language codes (at minimum source_lang and
    target_lang). Skips terms already in the glossary for that source_lang
    value. Returns count of new terms added."""
    rows = load_glossary()
    existing = {(row.get(source_lang) or "").strip().lower() for row in rows}
    added = 0
    for pair in term_pairs:
        src_val = (pair.get(source_lang) or "").strip()
        if not src_val or src_val.lower() in existing:
            continue
        tid = _next_term_id(rows)
        row = {"term_id": tid, "domain": pair.get("domain", ""),
               "notes": pair.get("notes", ""),
               "added_date": date.today().isoformat()}
        for lang in LANGUAGES:
            row[lang] = pair.get(lang, "")
        rows.append(row)
        existing.add(src_val.lower())
        added += 1
    if added:
        save_glossary(rows)
    return added


# ---------------------------------------------------------------------------
# Pattern-based DNT
# ---------------------------------------------------------------------------

class DntMatcher:
    """Compiled matcher for pattern-based DNT rules."""

    def __init__(self, rules: list[dict]):
        self._exact: set[str] = set()
        self._prefixes: list[str] = []
        self._globs: list[str] = []
        self._regexes: list[re.Pattern] = []
        for rule in rules:
            pattern = rule.get("pattern", "").strip()
            mode = rule.get("match_mode", "exact").strip()
            if not pattern:
                continue
            if mode == "exact":
                self._exact.add(pattern)
            elif mode == "prefix":
                self._prefixes.append(pattern.rstrip("*"))
            elif mode == "glob":
                self._globs.append(pattern)
            elif mode == "regex":
                try:
                    self._regexes.append(re.compile(pattern))
                except re.error:
                    pass

    def is_dnt(self, token: str) -> bool:
        """Check if a token matches any DNT rule."""
        if token in self._exact:
            return True
        for prefix in self._prefixes:
            if token.startswith(prefix):
                return True
        for glob_pat in self._globs:
            if fnmatch.fnmatch(token, glob_pat):
                return True
        for regex in self._regexes:
            if regex.fullmatch(token):
                return True
        return False


def load_dnt_rules() -> list[dict]:
    """Load DNT rules from dnt.csv."""
    if not DNT_FILE.exists():
        return []
    rows = []
    with DNT_FILE.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_dnt_matcher() -> DntMatcher:
    """Load and compile DNT rules into a matcher."""
    return DntMatcher(load_dnt_rules())


def save_dnt_rules(rules: list[dict]) -> None:
    """Save DNT rules to dnt.csv."""
    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    with DNT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DNT_FIELDS)
        writer.writeheader()
        for row in rules:
            writer.writerow({k: row.get(k, "") for k in DNT_FIELDS})


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Glossary dir: {GLOSSARY_DIR}")
    print(f"Terms file exists: {TERMS_FILE.exists()}")
    print(f"DNT file exists: {DNT_FILE.exists()}")
    rows = load_glossary()
    print(f"Glossary terms loaded: {len(rows)}")
    rules = load_dnt_rules()
    print(f"DNT rules loaded: {len(rules)}")
    if rules:
        m = load_dnt_matcher()
        test_tokens = ["NAS6404A22", "MS21299C4", "IF2.03.08", "PR1440MB2",
                        "hello", "vis", "LATECOERE", "12/16/2024"]
        print("DNT matching test:")
        for t in test_tokens:
            print(f"  {t:20s} -> {'DNT' if m.is_dnt(t) else 'translate'}")
