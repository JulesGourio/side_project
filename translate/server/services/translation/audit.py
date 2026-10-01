"""
audit.py - Language detection, bilingual pairing, conflict detection,
and question generation.

Input:  reports/<doc>.extract.json
Output: reports/<doc>.audit.json   (one AuditedSegment per input segment)
        reports/<doc>.questions.json (batched questions for the user)

Pairing strategies (applied in priority order, each segment assigned at most
one pair_id):

  1. bilingual_txbx_pair  - within one text box (same txbx_path prefix), two
                            consecutive paragraphs in different languages.
                            This is the dominant pattern in LATECOERE docs.
  2. paired_body          - consecutive body_direct paragraphs in different
                            languages (e.g., "OPERACE 10..." then
                            "OPERATION 10...").
  3. paired_row           - in a table, two adjacent rows where each row is
                            uniformly one language.
  4. bilingual_inline_slash - single segment with "X / Y" pattern, halves
                            in different languages.
  5. bilingual_inline_concat - single segment whose runs split by formatting
                            into two groups detecting different languages.
  6. bilingual_inline_charsplit - single segment concatenating two languages
                            in the SAME run(s), no separator and no formatting
                            break. The boundary is found structurally FIRST
                            (script switch, then sentence boundaries), then
                            each side is language-detected on its own —
                            detect_language on the whole text can only ever
                            return one label
                            (client/src/components/translate/README.md
                            "Known issue: language detection").

Segments not paired remain pattern_type = "mono" (or "dnt" / "numeric_only").

Usage:  python audit.py <extract.json>
"""

from __future__ import annotations
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .langdetect import (SUPPORTED_LANGS, detect_language,
                         detect_language_constrained, detect_span_language,
                         extract_dnt_tokens, is_dnt_candidate, is_numeric_only)
from .glossary_io import DntMatcher, load_dnt_matcher


# ---------------------------------------------------------------------------
# Per-segment classification
# ---------------------------------------------------------------------------

def make_audited(seg: dict, matcher: DntMatcher | None = None,
                 default_lang: str | None = None,
                 candidates: set[str] | None = None,
                 source_lang: str | None = None) -> dict:
    """matcher applies reviewer-curated dnt_rules (exact/prefix/glob/regex,
    edited by hand in the glossary panel) on top of the built-in DNT_REGEX
    (dates/part numbers/standards) — both decide DNT status live, here,
    rather than the old flow where a per-job Q&A had to "confirm" tokens
    already-caught-by-regex into a one-off exact-match row that never helped
    a later job. Defaults to None (CSV/no custom rules) for the CLI.

    default_lang: the document's own dominant language, passed through as the
    tie-break/no-signal fallback — see audit_all_segments()'s two-pass call
    pattern (this function alone can't compute that bias, since it's a
    per-segment call with no view of the rest of the document).

    candidates: the job's declared {source_lang, target_lang} pair. When
    given, the constrained ensemble detector (fastText + heuristic, restricted
    to that pair) is used instead of the free 6-way heuristic — this is the
    web pipeline's path. None keeps the standalone/CLI behaviour."""
    text = seg["text"]
    if is_dnt_candidate(text, matcher):
        lang, conf = "??", 0.0
        pattern = "dnt"
    elif is_numeric_only(text):
        lang, conf = "??", 0.0
        pattern = "numeric_only"
    elif candidates:
        lang, conf = detect_language_constrained(text, candidates, default_lang=default_lang, source_lang=source_lang)
        pattern = "mono"
    else:
        lang, conf = detect_language(text, default_lang=default_lang)
        pattern = "mono"
    return {
        "seg_id": seg["seg_id"],
        "text": text,
        "detected_lang": lang,
        "lang_confidence": round(conf, 3),
        "pattern_type": pattern,
        "pair_id": None,
        "pair_role": None,
        "richer_side": None,
        "conflict_flag": False,
        "conflict_detail": None,
        "dnt_tokens": extract_dnt_tokens(text, matcher),
        "notes": "",
        "inline_split": None,
        # We carry forward a few extract fields for the pairing logic;
        # they're stripped from the final audit output below.
        "_part": seg["part"],
        "_location": seg["location_type"],
        "_body_p_idx": seg["body_p_idx"],
        "_table_coords": seg.get("table_coords"),
        "_txbx_path": seg.get("txbx_path"),
        "_para_idx": seg["para_idx_in_container"],
        "_fmt_signature": seg["fmt_signature"],
        "_runs": seg["runs"],
    }


# A segment's detected language is trusted as a genuine, translate-or-keep
# decision signal only at/above this confidence. Constrained detection only
# reports >= this when BOTH the heuristic and fastText agree (see
# langdetect.detect_language_constrained) — a single confident-but-maybe-wrong
# detector is capped below it on purpose.
CONFIDENT = 0.85


def _significant_languages(audited: list[dict]) -> set[str]:
    """Languages the document *actually* contains, by confident mono count.
    Used to widen the detection candidate set beyond the declared pair — the
    kept language is frequently a THIRD language (English part-number/label
    text in a BG->FR or CS->FR job), not the declared target, and it must be a
    detection candidate so those segments/spans are recognised and kept rather
    than mislabelled or force-fit to the source."""
    counts = Counter(
        a["detected_lang"] for a in audited
        if a["pattern_type"] == "mono" and a["detected_lang"] != "??"
        and a["lang_confidence"] >= CONFIDENT
    )
    total = sum(counts.values())
    if total == 0:
        return set()
    return {lang for lang, c in counts.items() if c >= 3 and c >= 0.05 * total}


def audit_all_segments(segments: list[dict], matcher: DntMatcher | None = None,
                       source_lang: str | None = None,
                       target_lang: str | None = None) -> list[dict]:
    """Detect each mono segment's language, then two-pass bias the zero-signal
    ones to the document dominant.

    Web pipeline (source/target given): a first pass over the FULL supported
    set discovers which languages the document actually holds; detection is
    then constrained to {source, target} ∪ those languages via the ensemble
    detector (fastText + heuristic). Constraining tightly to the real language
    set is what fixed both the ~50% low-confidence rate of the free heuristic
    AND the case where the kept language is a third language (EN) rather than
    the declared target. The CLI path (no pair) keeps the free heuristic."""
    if source_lang and target_lang:
        # Pass A — full-set detection to learn the document's real languages.
        audited = [make_audited(s, matcher, candidates=set(SUPPORTED_LANGS)) for s in segments]
        candidates = {source_lang, target_lang} | _significant_languages(audited)
        # Pass B — re-detect constrained to the pair + the doc's real languages.
        audited = [make_audited(s, matcher, candidates=candidates, source_lang=source_lang) for s in segments]
    else:
        candidates = None
        audited = [make_audited(s, matcher) for s in segments]

    lang_counts = Counter(
        a["detected_lang"] for a in audited
        if a["pattern_type"] == "mono" and a["detected_lang"] != "??"
    )
    if not lang_counts:
        return audited
    dominant = lang_counts.most_common(1)[0][0]
    for i, (s, a) in enumerate(zip(segments, audited)):
        if a["pattern_type"] == "mono" and a["detected_lang"] == "??":
            audited[i] = make_audited(s, matcher, default_lang=dominant, candidates=candidates)
    return audited


def determine_mode(audited: list[dict], source_lang: str, target_lang: str) -> str:
    """'monolingual' = the document is essentially all source language
    (translate everything); 'bilingual' = it carries meaningful content in
    ANOTHER language that must be kept (replace only the source side).

    The other language is whatever is NOT the source — often a third language
    (EN) rather than the declared target, so this compares source against
    all-non-source confident content, NOT against the target (a BG->FR job on
    a BG+EN document has zero FR, but is very much bilingual). Counts only
    CONFIDENT (agreement-backed) detections so a few mislabeled cognate titles
    can't flip the verdict."""
    src = sum(1 for a in audited if a["pattern_type"] == "mono"
              and a["detected_lang"] == source_lang and a["lang_confidence"] >= CONFIDENT)
    other = sum(1 for a in audited if a["pattern_type"] == "mono"
                and a["detected_lang"] not in ("??", source_lang)
                and a["lang_confidence"] >= CONFIDENT)
    total = src + other
    if total == 0:
        return "monolingual"
    if other >= 5 and other >= 0.15 * total:
        return "bilingual"
    return "monolingual"


# ---------------------------------------------------------------------------
# Pairing strategies
# ---------------------------------------------------------------------------

class Pairer:
    def __init__(self):
        self.next_id = 1

    def assign(self, primary: dict, secondary: dict, pattern_type: str) -> None:
        if primary["pair_id"] or secondary["pair_id"]:
            return
        pair_id = f"pair_{self.next_id:04d}"
        self.next_id += 1
        for seg, role in ((primary, "primary"), (secondary, "secondary")):
            seg["pair_id"] = pair_id
            seg["pair_role"] = role
            seg["pattern_type"] = pattern_type


def _format_groups(runs: list[dict]) -> list[tuple[str, str]]:
    """Group consecutive runs by fmt_hash. Returns [(fmt_hash, joined_text), ...]
    for non-empty fmt-groups (only counts groups whose text has any letters)."""
    if not runs:
        return []
    groups = []
    cur_fmt = runs[0]["fmt_hash"]
    cur_text = ""
    for r in runs:
        if r["fmt_hash"] == cur_fmt:
            cur_text += r["text"]
        else:
            if cur_text.strip():
                groups.append((cur_fmt, cur_text))
            cur_fmt = r["fmt_hash"]
            cur_text = r["text"]
    if cur_text.strip():
        groups.append((cur_fmt, cur_text))
    return groups


def pair_txbx_siblings(audited: list[dict], pairer: Pairer) -> None:
    """Pair consecutive paragraphs within the same text-box container."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for seg in audited:
        if seg["_location"] != "txbx" or seg["pair_id"]:
            continue
        if seg["pattern_type"] in ("dnt", "numeric_only"):
            continue
        tx = seg["_txbx_path"]
        key = (seg["_part"], tx["body_p_idx"], tx["ac_idx"], tx["txbx_idx"])
        groups[key].append(seg)
    for key, segs in groups.items():
        segs.sort(key=lambda s: s["_para_idx"])
        i = 0
        while i < len(segs) - 1:
            a, b = segs[i], segs[i + 1]
            if (a["detected_lang"] != b["detected_lang"]
                    and a["detected_lang"] != "??"
                    and b["detected_lang"] != "??"
                    and not a["pair_id"] and not b["pair_id"]):
                pairer.assign(a, b, "bilingual_txbx_pair")
                i += 2
            else:
                i += 1


def pair_adjacent_body(audited: list[dict], pairer: Pairer) -> None:
    """Pair consecutive body_direct paragraphs in different languages, when
    they're in the same XML part and adjacent by body_p_idx."""
    by_part: dict[str, list[dict]] = defaultdict(list)
    for seg in audited:
        if seg["_location"] != "body_direct" or seg["pair_id"]:
            continue
        if seg["pattern_type"] in ("dnt", "numeric_only"):
            continue
        by_part[seg["_part"]].append(seg)
    for part, segs in by_part.items():
        segs.sort(key=lambda s: s["_body_p_idx"])
        i = 0
        while i < len(segs) - 1:
            a, b = segs[i], segs[i + 1]
            # Adjacent or near-adjacent (allow gap of empty paragraphs)
            gap = b["_body_p_idx"] - a["_body_p_idx"]
            if (1 <= gap <= 3
                    and a["detected_lang"] != b["detected_lang"]
                    and a["detected_lang"] != "??"
                    and b["detected_lang"] != "??"
                    and not a["pair_id"] and not b["pair_id"]):
                pairer.assign(a, b, "paired_body")
                i += 2
            else:
                i += 1


def pair_table_rows(audited: list[dict], pairer: Pairer) -> None:
    """Pair adjacent table rows where each row is uniformly one language."""
    by_table: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for seg in audited:
        if seg["_location"] != "table" or seg["pair_id"]:
            continue
        if seg["pattern_type"] in ("dnt", "numeric_only"):
            continue
        tc = seg["_table_coords"]
        by_table[(seg["_part"], tc["table_body_idx"])][tc["row_idx"]].append(seg)

    for (part, tbl_idx), rows in by_table.items():
        # Determine each row's dominant language
        row_langs: dict[int, str] = {}
        for row_idx, segs in rows.items():
            langs = Counter(s["detected_lang"] for s in segs
                            if s["detected_lang"] != "??")
            if langs:
                row_langs[row_idx] = langs.most_common(1)[0][0]
        sorted_row_idxs = sorted(row_langs.keys())
        used: set = set()
        for i in range(len(sorted_row_idxs) - 1):
            r1 = sorted_row_idxs[i]
            r2 = sorted_row_idxs[i + 1]
            if r1 in used or r2 in used:
                continue
            if r2 - r1 != 1:
                continue
            if row_langs[r1] != row_langs[r2]:
                # Pair cells column-by-column
                for s1 in rows[r1]:
                    for s2 in rows[r2]:
                        if (s1["_table_coords"]["cell_idx"]
                                == s2["_table_coords"]["cell_idx"]):
                            if not s1["pair_id"] and not s2["pair_id"]:
                                pairer.assign(s1, s2, "paired_row")
                used.add(r1)
                used.add(r2)


SLASH_SPLIT_RE = re.compile(r"^(.+?)\s*/\s*(.+)$")


def pair_inline_slash(audited: list[dict]) -> None:
    """Mark mono segments that look like 'X / Y' with halves in different langs."""
    for seg in audited:
        if seg["pair_id"] or seg["pattern_type"] != "mono":
            continue
        m = SLASH_SPLIT_RE.match(seg["text"])
        if not m:
            continue
        left, right = m.group(1).strip(), m.group(2).strip()
        if len(left) < 3 or len(right) < 3:
            continue
        lang_l, conf_l = detect_span_language(left)
        lang_r, conf_r = detect_span_language(right)
        if (lang_l != lang_r and lang_l != "??" and lang_r != "??"
                and min(conf_l, conf_r) > 0.5):
            seg["pattern_type"] = "bilingual_inline_slash"
            # Compute the offset of the "/" in the original text
            slash_idx = seg["text"].index("/")
            seg["inline_split"] = {
                "kind": "slash",
                "offset": slash_idx,
                "left_lang": lang_l,
                "right_lang": lang_r,
            }


def pair_inline_format_concat(audited: list[dict]) -> None:
    """Mark mono segments whose runs split into 2 fmt-groups detecting as
    different languages."""
    for seg in audited:
        if seg["pair_id"] or seg["pattern_type"] != "mono":
            continue
        groups = _format_groups(seg["_runs"])
        if len(groups) < 2:
            continue
        # Take the two largest groups
        groups_sorted = sorted(groups, key=lambda g: -len(g[1]))[:2]
        (fmt_a, txt_a), (fmt_b, txt_b) = groups_sorted
        lang_a, conf_a = detect_span_language(txt_a)
        lang_b, conf_b = detect_span_language(txt_b)
        if (lang_a != lang_b and lang_a != "??" and lang_b != "??"
                and min(conf_a, conf_b) > 0.6):
            seg["pattern_type"] = "bilingual_inline_concat"
            # Find the run index where format changes from group A to B
            split_idx = None
            for i, r in enumerate(seg["_runs"]):
                if r["fmt_hash"] == fmt_b:
                    split_idx = i
                    break
            seg["inline_split"] = {
                "kind": "format",
                "run_split_idx": split_idx,
                "fmt_a": fmt_a,
                "fmt_b": fmt_b,
                "lang_a": lang_a,
                "lang_b": lang_b,
                # Side texts so the translation stage can send ONLY the
                # source-language side to the LLM (the kept side must
                # survive verbatim).
                "text_a": txt_a,
                "text_b": txt_b,
            }


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;:…])\s+|\n+")


_UNIT_SUFFIX_RE = re.compile(r"\d[\d.,]*\s?([A-Za-z°µ]{1,3})(?![A-Za-z])")


def _script_block_boundary(text: str) -> int | None:
    """If the text's letters form one contiguous Cyrillic block followed by
    one contiguous Latin block (or vice versa), return the offset where the
    second block starts. None when scripts interleave or a side is trivial.

    A short Latin abbreviation glued to a digit (units like "35 Nm", "40 mm",
    an ordinal "3rd") carries no language signal. Left uncounted, it gets
    mistaken for the start of the Latin block whenever it sits at the tail of
    an otherwise all-Cyrillic sentence that is itself followed by that
    sentence's Latin-script echo - splitting the source text one word early
    and stranding the unit on the kept (untranslated) side."""
    unit_idx: set[int] = set()
    for m in _UNIT_SUFFIX_RE.finditer(text):
        unit_idx.update(range(m.start(1), m.end(1)))

    cyr_pos = []
    lat_pos = []
    for i, c in enumerate(text):
        if not c.isalpha() or i in unit_idx:
            continue
        if "Ѐ" <= c <= "ӿ":
            cyr_pos.append(i)
        else:
            lat_pos.append(i)
    if len(cyr_pos) < 2 or len(lat_pos) < 2:
        return None
    if cyr_pos[-1] < lat_pos[0]:
        return lat_pos[0]
    if lat_pos[-1] < cyr_pos[0]:
        return cyr_pos[0]
    return None


def _find_char_boundary(text: str) -> tuple[int, str, str, float] | None:
    """Find the language boundary of a two-language concatenation, or None.

    Returns (offset, left_lang, right_lang, min_confidence) with the offset
    pointing at the first character of the right-hand span. Structural
    signals only — a script switch (Bulgarian vs anything Latin, works even
    with no punctuation) first, sentence boundaries otherwise. Interleaved
    (A B A) or single-language texts return None and stay mono.
    """
    p = _script_block_boundary(text)
    if p is not None:
        lang_l, conf_l = detect_span_language(text[:p])
        lang_r, conf_r = detect_span_language(text[p:])
        if (lang_l != lang_r and lang_l != "??" and lang_r != "??"
                and min(conf_l, conf_r) > 0.5):
            return p, lang_l, lang_r, min(conf_l, conf_r)
        return None

    best: tuple[int, str, str, float] | None = None
    for m in _SENTENCE_BOUNDARY_RE.finditer(text):
        p = m.end()
        if not (0 < p < len(text)):
            continue
        left, right = text[:p], text[p:]
        if (sum(1 for c in left if c.isalpha()) < 4
                or sum(1 for c in right if c.isalpha()) < 4):
            continue
        lang_l, conf_l = detect_span_language(left)
        lang_r, conf_r = detect_span_language(right)
        if lang_l == lang_r or lang_l == "??" or lang_r == "??":
            continue
        score = min(conf_l, conf_r)
        if score >= 0.55 and (best is None or score > best[3]):
            best = (p, lang_l, lang_r, score)
    return best


def pair_inline_charspan(audited: list[dict]) -> None:
    """Mark mono segments that concatenate two languages in the SAME run(s),
    with no '/' separator and no formatting break — the case that fell
    through pair_inline_slash/pair_inline_format_concat and got ONE
    detect_language label for two languages' content, silently leaving the
    outvoted language untranslated (client/src/components/translate/README.md "Known issue")."""
    for seg in audited:
        if seg["pair_id"] or seg["pattern_type"] != "mono":
            continue
        found = _find_char_boundary(seg["text"])
        if not found:
            continue
        offset, lang_l, lang_r, conf = found
        seg["pattern_type"] = "bilingual_inline_charsplit"
        seg["inline_split"] = {
            "kind": "char",
            "offset": offset,
            "left_lang": lang_l,
            "right_lang": lang_r,
            "confidence": round(conf, 3),
        }


# ---------------------------------------------------------------------------
# Richness + conflict detection
# ---------------------------------------------------------------------------

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_numbers(text: str) -> set[str]:
    """Extract numbers normalized to use '.' as decimal separator."""
    out = set()
    for n in NUMBER_RE.findall(text):
        out.add(n.replace(",", "."))
    return out


def assess_pair(primary: dict, secondary: dict) -> None:
    """Set richer_side, conflict_flag, conflict_detail on both members."""
    # Richer side: more words wins (ties = "equal")
    wp = len(primary["text"].split())
    ws = len(secondary["text"].split())
    if wp > ws * 1.15:
        richer = "primary"
    elif ws > wp * 1.15:
        richer = "secondary"
    else:
        richer = "equal"

    # Conflict heuristics — reasons are kept in plain, non-technical language
    # since these surface directly to end users who don't know (and don't
    # need to know) what a "DNT token" or a "word-count ratio" is; the
    # heuristics themselves are unchanged, only the wording shown for them.
    reasons = []
    dnt_p, dnt_s = set(primary["dnt_tokens"]), set(secondary["dnt_tokens"])
    if dnt_p - dnt_s or dnt_s - dnt_p:
        reasons.append("a reference code or part number appears on one side but not the other")
    nums_p = normalize_numbers(primary["text"])
    nums_s = normalize_numbers(secondary["text"])
    if nums_p - nums_s or nums_s - nums_p:
        reasons.append("the numbers mentioned don't match on both sides")
    if max(wp, ws) > 5 and (wp / max(ws, 1) > 2.5 or ws / max(wp, 1) > 2.5):
        reasons.append("one side is much shorter than the other, so it may be missing information")

    detail = (
        "These two texts were matched together, but " + "; and ".join(reasons) + "."
        if reasons else None
    )
    flag = bool(reasons)
    for seg in (primary, secondary):
        seg["richer_side"] = richer
        seg["conflict_flag"] = flag
        seg["conflict_detail"] = detail


# ---------------------------------------------------------------------------
# Determine primary/secondary languages doc-wide
# ---------------------------------------------------------------------------

def determine_languages(audited: list[dict]) -> tuple[str, str, list[str]]:
    """Returns (primary_lang, secondary_lang, all_detected_languages)."""
    langs = Counter(s["detected_lang"] for s in audited
                    if s["detected_lang"] != "??"
                    and s["pattern_type"] not in ("dnt", "numeric_only"))
    most = langs.most_common()
    if len(most) == 0:
        return "??", "??", []
    if len(most) == 1:
        return most[0][0], "??", [most[0][0]]
    return most[0][0], most[1][0], [l for l, _ in most]


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

def generate_questions(audited: list[dict], primary_lang: str,
                       secondary_lang: str, source_file: str,
                       mode: str = "bilingual") -> list[dict]:
    """Rule-generated clarifying questions (no LLM). The old
    'which_side_to_replace' question is gone — the user already picks
    source/target in the UI before upload, so asking again was pure noise.
    'unknown_lang' questions are only raised in bilingual mode, where a
    segment's label decides translate-vs-keep; in monolingual mode every
    ambiguous segment is translated anyway, so there is nothing to ask.

    Paired-segment conflicts (numeric/reference-code/length mismatches, see
    assess_pair) are NOT asked here — there is nothing to actually answer for
    them (no real "conflict" question ever needs a free-text reply), so
    presenting them as blocking questions just confused reviewers into
    thinking something was required before continuing. They stay visible as
    flags in the Segments panel for review instead."""
    questions = []
    qid = 1

    def add(category, question_text, seg_ids=None, context="", suggested=None):
        nonlocal qid
        questions.append({
            "q_id": f"q_{qid:03d}",
            "seg_ids": seg_ids or [],
            "category": category,
            "question_text": question_text,
            "context": context,
            "suggested_answer": suggested,
        })
        qid += 1

    # Monolingual documents translate every segment source->target: there is
    # no side-to-keep and cross-paragraph "pairs" are spurious, so there is
    # nothing to clarify before translation. The reviewer checks the result in
    # the Segments panel afterwards instead.
    if mode != "bilingual":
        return questions

    # Genuinely ambiguous language detections (sample) — decides
    # translate-vs-keep in bilingual mode, so worth asking.
    low_conf = [s for s in audited
                if s["pattern_type"] == "mono"
                and s["lang_confidence"] < 0.55
                and len(s["text"].split()) > 1]
    for s in low_conf[:5]:
        add("unknown_lang",
            f"We're not sure what language this text is written in — "
            f"is it {primary_lang.upper()}, {secondary_lang.upper()}, or something else?",
            seg_ids=[s["seg_id"]],
            context=s["text"])

    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def strip_internal(seg: dict) -> dict:
    """Remove the internal _-prefixed fields from an audited segment."""
    return {k: v for k, v in seg.items() if not k.startswith("_")}


def main(extract_path: str, _job_dir: Path | None = None) -> None:
    src = Path(extract_path)
    extract = json.loads(src.read_text(encoding="utf-8"))

    matcher = load_dnt_matcher()
    audited = audit_all_segments(extract["segments"], matcher)

    # Run pairing strategies in order
    pairer = Pairer()
    pair_txbx_siblings(audited, pairer)
    pair_adjacent_body(audited, pairer)
    pair_table_rows(audited, pairer)
    pair_inline_slash(audited)
    pair_inline_format_concat(audited)
    pair_inline_charspan(audited)

    # Compute per-pair richness + conflicts
    pairs_by_id: dict[str, list[dict]] = defaultdict(list)
    for s in audited:
        if s["pair_id"]:
            pairs_by_id[s["pair_id"]].append(s)
    for pid, segs in pairs_by_id.items():
        if len(segs) != 2:
            continue
        primary = next((s for s in segs if s["pair_role"] == "primary"), None)
        secondary = next((s for s in segs if s["pair_role"] == "secondary"), None)
        if primary and secondary:
            assess_pair(primary, secondary)

    primary_lang, secondary_lang, all_langs = determine_languages(audited)

    # Build outputs
    audit_report = {
        "source_file": extract["source_file"],
        "detected_languages": all_langs,
        "primary_language": primary_lang,
        "secondary_language": secondary_lang,
        "segment_count": len(audited),
        "pair_count": len([1 for s in audited if s["pair_role"] == "primary"]),
        "conflict_count": len([1 for s in audited
                               if s["pair_role"] == "primary" and s["conflict_flag"]]),
        "dnt_token_count": sum(len(s["dnt_tokens"]) for s in audited),
        "segments": [strip_internal(s) for s in audited],
    }

    questions = generate_questions(audited, primary_lang, secondary_lang,
                                   extract["source_file"])
    questions_file = {
        "source_file": extract["source_file"],
        "question_count": len(questions),
        "questions": questions,
    }

    # Write outputs alongside the input
    base = src.stem.replace(".extract", "")
    out_dir = src.parent
    audit_out = out_dir / f"{base}.audit.json"
    questions_out = out_dir / f"{base}.questions.json"
    audit_out.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    questions_out.write_text(json.dumps(questions_file, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    # Summary
    pattern_counts = Counter(s["pattern_type"] for s in audit_report["segments"])
    print(f"Audited {len(audited)} segments -> {audit_out.name}")
    print(f"  Languages detected: {all_langs}")
    print(f"  Primary={primary_lang.upper()}  Secondary={secondary_lang.upper()}")
    print(f"  Pair count: {audit_report['pair_count']}")
    print(f"  Conflicts: {audit_report['conflict_count']}")
    print(f"  Patterns: {dict(pattern_counts)}")
    print(f"  DNT tokens (occurrences): {audit_report['dnt_token_count']}")
    print(f"Generated {len(questions)} questions -> {questions_out.name}")

    if _job_dir:
        from job_utils import update_manifest
        update_manifest(_job_dir, status="audited")


if __name__ == "__main__":
    from job_utils import parse_job_flag, JobPaths
    job_dir, remaining = parse_job_flag(sys.argv[1:])
    if job_dir:
        jp = JobPaths(job_dir)
        main(str(jp.extract_json), _job_dir=job_dir)
    else:
        if len(sys.argv) != 2:
            print("Usage: python audit.py <extract.json>")
            sys.exit(1)
        main(sys.argv[1])
