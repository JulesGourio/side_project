"""
langdetect.py - Stdlib-only language detector for {EN, FR, ES, DE, CS, BG, PT, AR}.

Three detection tiers:
  1. Script-based   - Cyrillic block -> Bulgarian, Arabic block -> Arabic
     (the only non-Latin scripts in scope).
  2. Diacritic-based - unique characters per language (e.g. 'r' for CS, 'l' for PL,
     'ss' for DE, 'oe' for FR, 'n' for ES, 'ã/õ' for PT) plus shared-diacritic scoring.
  3. Keyword-based  - for ASCII-heavy text (FR/EN/ES/DE/PT) we score against
     short stopword lists.

Returns (iso_code, confidence) where confidence is in [0.0, 1.0].

Also exports DNT (do-not-translate) helpers used by audit.py:
    is_dnt_candidate(text)
    extract_dnt_tokens(text)
"""

from __future__ import annotations
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache

from .glossary_io import DntMatcher


# ---------------------------------------------------------------------------
# Language profiles
# ---------------------------------------------------------------------------

LANG_PROFILES = {
    "cs": {
        "unique": set("\u0159\u016f"),                  # r-hacek, u-ring (Czech-only among scope)
        "common": set("\u011b\u017e\u0161\u010d\u010f\u0165\u0148\u00fd\u00e1\u00ed\u00fa"),
        "keywords": ["je", "na", "se", "pro", "dle", "ne", "po", "do", "od", "to", "byl", "jsou", "toto"],
    },
    "bg": {
        # Cyrillic-script language; diacritics not used here, script test instead
        "unique": set(),
        "common": set(),
        "keywords": [],
    },
    "de": {
        "unique": set("\u00df"),                       # eszett (German-only)
        "common": set("\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc"),
        "keywords": ["der", "die", "das", "und", "mit", "von", "ein", "ist", "den", "im",
                     "nicht", "auch", "auf", "sind", "werden", "wird"],
    },
    "fr": {
        "unique": set("\u0153\u0152"),                 # ligature oe
        "common": set("\u00e9\u00e8\u00ea\u00eb\u00e0\u00e2\u00f9\u00fb\u00e7\u00ef\u00ee\u00f4"),
        # "en"/"la"/"de" are also common Spanish words (kept \u2014 contributes to
        # both, a wash, not a false Spanish signal) but were previously
        # MISSING from French's own list entirely, silently ceding every
        # French sentence containing them to whichever other language DID
        # list them (measured 2026-07-21: "Mise en place sur outillage",
        # unambiguous French, scored es@0.85 purely because "en" was only
        # in the ES keyword list, never in French's).
        "keywords": ["le", "la", "les", "de", "du", "des", "et", "un", "une", "pour",
                     "en", "sur", "avec", "dans", "que", "qui", "ce", "cette", "ces",
                     "au", "aux", "vous", "nous", "pas", "sont", "est", "\u00eatre", "avoir"],
    },
    "es": {
        "unique": set("\u00f1\u00d1\u00bf\u00a1"),     # n-tilde, inverted punctuation
        "common": set("\u00e1\u00e9\u00ed\u00f3\u00fa\u00c1\u00c9\u00cd\u00d3\u00da"),
        "keywords": ["el", "la", "los", "las", "de", "del", "y", "en", "es", "para",
                     "con", "por", "un", "una", "que", "no", "se"],
    },
    "en": {
        "unique": set(),
        "common": set(),
        "keywords": ["the", "of", "and", "is", "to", "for", "with", "in", "on", "per",
                     "are", "this", "that", "not", "shall", "must"],
    },
    "pt": {
        "unique": set("ãõÃÕ"),     # a-tilde, o-tilde (nasal vowels, PT-only among scope)
        "common": set("áéíóúâêôàçÁÉÍÓÚÂÊÔÀÇ"),
        "keywords": ["o", "a", "os", "as", "de", "do", "da", "e", "em", "para",
                     "com", "por", "um", "uma", "que", "não", "se", "são", "está"],
    },
    "ar": {
        # Arabic script; diacritics not used here, script test instead (see
        # detect_language's Tier 1, same treatment as bg's Cyrillic block).
        "unique": set(),
        "common": set(),
        "keywords": [],
    },
}


SUPPORTED_LANGS = list(LANG_PROFILES.keys())

# Company/proper-noun tokens whose accented letters must NOT count as
# language-detection signal: "Latécoère" reads as a confident French (or
# Spanish -- é/è aren't unique to French) signal to the diacritic tiers
# below even in an otherwise all-English sentence that merely mentions the
# company (measured on a real document: a repeated English copyright
# footer, "This document is the property of Latécoère...", scored fr@0.88
# and got needlessly translated on some pages, inconsistently on others
# depending on incidental block-splitting). Matches any accent combination
# on the two e's (Latécoère/Latecoere/Latécoere/Latecoère) case-insensitively.
_PROPER_NOUN_PATTERN = re.compile(r"lat[eé]co[eè]re", re.IGNORECASE)


def _strip_diacritic_bias_tokens(text: str) -> str:
    """Remove known proper nouns before diacritic-tier language scoring —
    NOT used for Tier 3 keyword scoring, which already tokenizes on
    [A-Za-z]+ and so never matched these accented tokens as one piece."""
    return _PROPER_NOUN_PATTERN.sub(" ", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_language(text: str, default_lang: str | None = None) -> tuple[str, float]:
    """Return (iso_code, confidence). Returns ("??", 0.0) for empty/unanalyzable
    text, UNLESS default_lang is given — then a segment with no distinguishing
    signal at all returns (default_lang, 0.5) instead of guessing blindly.

    default_lang is meant to be the document's own dominant detected language
    (a two-pass bias — see audit.py::make_audited): text with zero signal
    (too short, no accented chars, no stopword matches — e.g. "OPERATION 20",
    an FR/EN technical cognate with no diacritics) is overwhelmingly more
    likely to be whatever language the rest of the document already is than
    a random guess. Previously this fell through to a hardcoded 'en' default
    with no basis at all (measured 2026-07-21: a 92%-French real document had
    58/163 segments silently mislabeled 'en' this way alone)."""
    if not text or not text.strip():
        return "??", 0.0

    # Strip punctuation/digits for analysis but keep accented letters.
    # Lowercase so diacritic matching works regardless of capitalization
    # (avoids FR-vs-ES confusion on text like "PREMIÈRE ÉDITION"). Known
    # proper nouns are stripped first -- see _strip_diacritic_bias_tokens.
    letters = [c.lower() for c in _strip_diacritic_bias_tokens(text) if c.isalpha()]
    if not letters:
        return (default_lang, 0.4) if default_lang else ("??", 0.0)
    letter_count = len(letters)

    # ---- Tier 1: script-based ----
    # Cyrillic is the ONLY script among the supported languages, so any
    # non-trivial presence of it is an unambiguous "this segment contains
    # Bulgarian" signal -- unlike the other tiers there is no other language
    # in scope it could be confused with. A >50%-of-the-segment majority
    # requirement missed segments that concatenate a BG sentence with its
    # (often longer) EN/FR/... echo in the same paragraph or table cell,
    # letting the majority script silently outvote real BG content that
    # still needs translating. Corpus check (see server/services/translation/
    # langdetect.py history) found a clean gap between genuine embedded BG
    # text (cyrillic_count >= 2, ratio >= ~10%) and stray single-character
    # artifacts (one Cyrillic look-alike letter in an otherwise-Latin code),
    # so gate on both count and ratio rather than ratio alone.
    cyrillic = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    cyrillic_ratio = cyrillic / letter_count
    if cyrillic >= 2 and cyrillic_ratio > 0.08:
        return "bg", min(0.99, 0.75 + cyrillic_ratio * 0.24)

    # Same reasoning for Arabic \u2014 the only Arabic-script language in scope.
    arabic = sum(1 for c in letters if "\u0600" <= c <= "\u06ff")
    arabic_ratio = arabic / letter_count
    if arabic >= 2 and arabic_ratio > 0.08:
        return "ar", min(0.99, 0.75 + arabic_ratio * 0.24)

    # ---- Tier 2: unique-diacritic detection ----
    # Strong signals first - one unique char => high confidence
    unique_hits = {lang: sum(1 for c in letters if c in profile["unique"])
                   for lang, profile in LANG_PROFILES.items()
                   if profile["unique"]}
    strong = [(lang, hits) for lang, hits in unique_hits.items() if hits > 0]
    if strong:
        # Pick the language with the most unique-char hits
        strong.sort(key=lambda kv: -kv[1])
        winner, hits = strong[0]
        confidence = min(0.98, 0.80 + 0.04 * hits)
        return winner, confidence

    # ---- Tier 2b: shared-diacritic scoring ----
    # Score each non-EN, non-BG language by its common-diacritic hits. Same
    # class of bug as the tier-1 Cyrillic ratio: a segment that concatenates
    # a short CS/DE/FR/ES sentence with its (often longer) EN echo dilutes
    # the diacritic SHARE below the old ratio-only bar even when the
    # absolute hit count is a perfectly good signal on its own -- gate on
    # count (like tier 2's unique-char check) OR share, not share alone.
    diacritic_hits = {}
    for lang, profile in LANG_PROFILES.items():
        if lang in ("en", "bg"):
            continue
        common = profile["common"]
        if not common:
            continue
        diacritic_hits[lang] = sum(1 for c in letters if c in common)

    max_hits = max(diacritic_hits.values(), default=0)
    # A TIE among 2+ languages at the same hit count means the character(s)
    # matched aren't actually distinguishing here (e.g. 'é' is common to
    # CS/FR/ES) — picking one is an arbitrary, silent dict-iteration-order
    # bias, not a real signal (measured 2026-07-21: "Rédacteur"/"Création"
    # tied CS/FR on shared 'é' and CS won purely by appearing first in
    # LANG_PROFILES). Defer to keyword scoring instead of guessing.
    tied = [lang for lang, hits in diacritic_hits.items() if hits == max_hits and hits > 0]
    if len(tied) == 1 and max_hits > 0:
        best_lang = tied[0]
        ratio = max_hits / letter_count
        if max_hits >= 2 or ratio > 0.02:
            signal = max(ratio, min(max_hits, 10) / 40)
            confidence = min(0.88, 0.55 + signal * 4)
            return best_lang, confidence

    # ---- Tier 3: keyword scoring (ASCII text) ----
    return _keyword_score(text, default_lang=default_lang)


def _keyword_score(text: str, default_lang: str | None = None) -> tuple[str, float]:
    """Score against per-language stopword lists for ASCII text.

    default_lang: see detect_language's docstring — used whenever there is
    zero distinguishing signal (no words at all, no stopword hits, or a
    genuine tie between two languages' hit counts) instead of the old
    hardcoded 'en' guess."""
    # Tokenize lowercase words
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return (default_lang, 0.4) if default_lang else ("??", 0.0)
    word_set = Counter(words)
    total = sum(word_set.values())

    scores = {}
    for lang, profile in LANG_PROFILES.items():
        if not profile["keywords"]:
            continue
        hits = sum(word_set[k] for k in profile["keywords"])
        scores[lang] = hits / total if total else 0

    if not scores:
        return (default_lang, 0.4) if default_lang else ("??", 0.0)
    max_score = max(scores.values())
    if max_score == 0:
        # No stopwords matched at all — no basis to guess a language.
        return (default_lang, 0.45) if default_lang else ("??", 0.0)
    tied = [lang for lang, score in scores.items() if score == max_score]
    if len(tied) > 1:
        # Genuine tie (e.g. the only matched word is common to both
        # languages' lists) — same "don't silently guess" rule as tier 2b.
        return (default_lang, 0.45) if default_lang else ("??", 0.0)
    best_lang = tied[0]
    confidence = min(0.85, 0.45 + max_score * 3)
    return best_lang, confidence


def detect_language_pair(text_a: str, text_b: str) -> tuple[tuple[str, float], tuple[str, float]]:
    """Detect both texts; useful for cross-validating bilingual pairs."""
    return detect_language(text_a), detect_language(text_b)


# ---------------------------------------------------------------------------
# Span-level detection (short halves of a candidate bilingual split)
# ---------------------------------------------------------------------------

# Same vendored model the chat translation bridge uses
# (server/services/translation_bridge.py) — one copy, two consumers.
_FASTTEXT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "lid.176.ftz",
)


@lru_cache(maxsize=1)
def _fasttext_model():
    """Lazy-loaded fastText lid.176 model, or None when the package/model
    isn't available (standalone use outside the app venv)."""
    try:
        import fasttext
        fasttext.FastText.eprint = lambda *_a, **_kw: None
        return fasttext.load_model(_FASTTEXT_MODEL_PATH)
    except Exception:
        return None


def _fasttext_scores(text: str, langs) -> dict[str, float] | None:
    """fastText lid.176 probabilities renormalized over `langs` (a subset of
    the 176 model labels). Returns {code: prob} summing to 1.0, or None when
    the model is unavailable or none of `langs` got any mass.

    Renormalizing over a subset is what makes lid.176 usable on the short
    technical spans in these docs (measured 2026-07-17: raw 'Install the nut.'
    -> fi@0.33; restricted to the six in-scope languages it recovers en; and
    restricted to just a job's {source, target} pair it is sharper still). The
    noise mass goes to out-of-scope languages instead of a wrong in-scope one.

    Lower-cased before inference: lid.176 is confidently WRONG on ALL-CAPS
    accented technical titles (measured 2026-08-24: 'OPERACE 20 - MONTAZ
    TEHEL 30', genuinely Czech, scored en@0.997 as typed but cs@0.96 lower-
    cased) — these operation-title headers are extremely common in this
    corpus, and case carries no language signal to lose.
    """
    model = _fasttext_model()
    if model is None:
        return None
    single_line = " ".join(text.split()).lower() + "\n"
    try:
        # Direct binding call, not .predict(): the Python wrapper breaks
        # under NumPy>=2.0 (see translation_bridge._fast_lang_guess).
        preds = model.f.predict(single_line, 176, 0.0, "strict")
    except Exception:
        return None
    langs = set(langs)
    scores: dict[str, float] = {}
    for prob, label in preds:
        code = label.replace("__label__", "")
        if code in langs:
            scores[code] = scores.get(code, 0.0) + prob
    total = sum(scores.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in scores.items()}


def _fasttext_restricted(text: str) -> tuple[str, float] | None:
    """fastText prediction restricted to the six supported languages, with
    the probability mass renormalized over that subset. Used by the span-level
    second opinion in detect_span_language."""
    scores = _fasttext_scores(text, LANG_PROFILES.keys())
    if not scores:
        return None
    best = max(scores, key=lambda k: scores[k])
    return best, scores[best]


def detect_language_constrained(
    text: str, candidates, default_lang: str | None = None, source_lang: str | None = None,
) -> tuple[str, float]:
    """Ensemble detector restricted to a small candidate set — the primary
    per-segment detector for the Translate pipeline's mono segments.

    `candidates` is the job's declared language pair {source_lang, target_lang}
    (plus optionally the document-dominant language). Restricting to it fixes
    the two failure modes the free 6-way heuristic showed on the real corpus:
    catastrophic mislabels (a 92%-French doc scored en) and a ~50% low-
    confidence rate. It combines the stdlib heuristic (detect_language) with
    fastText scores renormalized over `candidates`.

    Confidence policy is deliberate: only an AGREEMENT between the two
    independent detectors reports high (>=0.85) confidence. A single confident
    detector is capped at 0.80 — this matters because fastText is sometimes
    confidently WRONG on short cognate titles ('OPERATION 20 PERCAGE' ->
    en@0.92); capping keeps such a call below the "keep as target-language"
    bar so it can never be silently left untranslated (see
    server/routers/translate.py::_plan_segment_translation).

    source_lang: the job's declared source — see the single-token override
    below, a distinct bias from `default_lang` (the document's dominant
    language, which for a bilingual doc is often the OTHER language and
    would bias the wrong way here)."""
    candidates = set(candidates)
    if not text or not text.strip():
        return "??", 0.0

    # Cyrillic is the only Cyrillic-script language in scope, so any non-trivial
    # Cyrillic presence is an UNAMBIGUOUS Bulgarian signal — it must win over a
    # confident-but-wrong fastText call on the Latin remainder of a mixed cell
    # (measured: "Гайка/Nut : NAS1726-4E" scored fr, leaving the BG half
    # untranslated). Decide it here before the ensemble can override it.
    if "bg" in candidates:
        letters = [c for c in text if c.isalpha()]
        cyr = sum(1 for c in letters if "Ѐ" <= c <= "ӿ")
        if cyr >= 2 and letters and cyr / len(letters) > 0.08:
            return "bg", min(0.99, 0.85 + (cyr / len(letters)) * 0.14)

    # Same reasoning for Arabic script.
    if "ar" in candidates:
        letters = [c for c in text if c.isalpha()]
        arb = sum(1 for c in letters if "؀" <= c <= "ۿ")
        if arb >= 2 and letters and arb / len(letters) > 0.08:
            return "ar", min(0.99, 0.85 + (arb / len(letters)) * 0.14)

    h_lang, h_conf = detect_language(text, default_lang=default_lang)
    if h_lang not in candidates:
        h_lang, h_conf = "??", 0.0

    ft = _fasttext_scores(text, candidates)
    if ft:
        ft_lang = max(ft, key=lambda k: ft[k])
        ft_conf = ft[ft_lang]
    else:
        ft_lang, ft_conf = "??", 0.0

    # Both detectors agree on a candidate -> two independent methods concur,
    # the strongest signal available; report high confidence.
    if h_lang != "??" and h_lang == ft_lang:
        return h_lang, min(0.99, max(0.88, ft_conf, h_conf))

    # A single word with no heuristic agreement is the hardest case for any
    # language-ID model — measured 2026-08-24: fastText alone confidently
    # (and wrongly) calls bare Czech words like "Sroub"/"Proces"/"Norma"
    # English at 0.6-0.8, no case-folding fixes it (unlike the all-caps
    # title case above). Below, don't let a lone detector's guess on so
    # little text silently keep it untranslated — bias to the source
    # language instead: a word that turns out to have been fine to keep just
    # gets a redundant translation, caught on review; one that needed
    # translating and got silently skipped is the worse failure.
    if len(text.split()) <= 1 and source_lang and source_lang in candidates:
        return source_lang, 0.5

    # Disagreement / single detector: trust the confident one but cap the
    # reported confidence below the keep bar (see docstring).
    if ft_lang != "??" and ft_conf >= 0.70:
        return ft_lang, min(0.80, ft_conf)
    if h_lang != "??" and h_conf >= 0.55:
        return h_lang, min(0.80, h_conf)
    if ft_lang != "??" and ft_conf >= 0.55:
        return ft_lang, min(0.80, ft_conf)

    # No usable signal within the candidate pair.
    return (default_lang, 0.5) if default_lang else ("??", 0.0)


def detect_span_language(text: str) -> tuple[str, float]:
    """detect_language with a fastText second opinion for weak verdicts.

    Meant for the SHORT spans produced by the inline pre-split heuristics
    (audit.py's pair_inline_* functions), where detect_language's tiers often
    have nothing to grip: a telegraphic diacritic-free Czech imperative falls
    through to the en@0.45 keyword default. Only low-confidence results pay
    the fastText call; agreement raises confidence, a confident disagreement
    wins, and anything else keeps the heuristic verdict unchanged.
    """
    lang, conf = detect_language(text)
    if conf >= 0.60:
        return lang, conf
    ft = _fasttext_restricted(text)
    if ft is None:
        return lang, conf
    ft_lang, ft_conf = ft
    if ft_lang == lang:
        return lang, max(conf, ft_conf)
    if ft_conf >= 0.60:
        return ft_lang, ft_conf
    return lang, conf


# ---------------------------------------------------------------------------
# DNT (do-not-translate) detection
# ---------------------------------------------------------------------------

# Patterns observed in LATECOERE sample docs.
DNT_PATTERNS = [
    # Aerospace fastener part numbers: NAS6404A22, MS21299C4, NAS1726-4D
    r"\b[A-Z]{2,5}\d{3,6}[A-Z]?\d*[-\d]*\b",
    # Internal references: IF2.03.08, F46C521A0, D5211535000901
    r"\b[A-Z]{1,3}\d+(?:[.-]\d+)+[A-Z]?\d*\b",
    # Long alpha-numeric part numbers: F46C521124037
    r"\b[A-Z]\d{2}[A-Z]\d{3}\d{6}\b",
    # PR/EN/MS sealants and standards: PR1440MB2, EN9100, AS9100
    r"\b(?:PR|EN|MS|AS|ISO|ASNA|NSA)\d{3,5}[A-Z]{0,3}\d*\b",
    # Date formats: 12/16/2024, 22/04/2024, 08/12/2023
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    # ISO dates: 2024-04-22
    r"\b\d{4}-\d{2}-\d{2}\b",
    # Digit-FIRST internal references (the patterns above all start with a
    # letter) — 252A90008200 (measured 2026-07-21: fell through language
    # detection entirely, its lone letter carrying no signal, and got
    # folded into whatever the segment's language bias happened to be
    # rather than flagged do-not-translate).
    r"\b\d{2,}[A-Z]{1,3}\d{3,}\b",
    # Document revision letters (very short, must be standalone)
    # Caught as numeric_only, not DNT
]

DNT_REGEX = re.compile("|".join(DNT_PATTERNS))


def extract_dnt_tokens(text: str, matcher: DntMatcher | None = None) -> list[str]:
    """Return list of DNT tokens found in the text: built-in regex matches
    (part numbers, standards, dates) plus, if a matcher is given, any
    reviewer-curated dnt_rules pattern (exact/prefix/glob/regex) matching a
    whitespace-delimited token in the text."""
    tokens = DNT_REGEX.findall(text)
    if matcher is not None:
        for word in re.findall(r"\S+", text):
            stripped = word.strip(".,;:()[]")
            if stripped and stripped not in tokens and matcher.is_dnt(stripped):
                tokens.append(stripped)
    return tokens


def is_dnt_candidate(text: str, matcher: DntMatcher | None = None) -> bool:
    """True if the entire text (after stripping whitespace) is a DNT token,
    a single uppercase revision letter, a pure number/date, or matches a
    reviewer-curated dnt_rules pattern.

    Used to skip translation entirely for these segments.
    """
    s = text.strip()
    if not s:
        return False
    # Single uppercase letter - revision indicator
    if len(s) == 1 and s.isupper() and s.isalpha():
        return True
    # Pure number with optional decimal/comma
    if re.fullmatch(r"[\d.,\-+]+", s):
        return True
    # Whole string matches DNT pattern
    if DNT_REGEX.fullmatch(s):
        return True
    # All-uppercase proper noun with hyphen (e.g., "BERNARD-SENTAURENS")
    if re.fullmatch(r"[A-Z][A-Z\-']+", s) and len(s) >= 4:
        # Could be a name OR an acronym/heading. Treat as DNT only if it
        # contains a hyphen or apostrophe (typical of surnames).
        if "-" in s or "'" in s:
            return True
    if matcher is not None and matcher.is_dnt(s):
        return True
    return False


def is_numeric_only(text: str) -> bool:
    """True if text contains no letters (numbers, punctuation, dates only)."""
    return not any(c.isalpha() for c in text)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    samples = [
        ("en", "Install the screw and washer per the assembly drawing."),
        ("fr", "Installer la vis et la rondelle selon le plan d'assemblage."),
        ("es", "Instalar el tornillo y la arandela segun el plano de montaje."),
        ("de", "Schraube und Unterlegscheibe gem\u00e4\u00df Montagezeichnung einbauen."),
        ("cs", "Mont\u00e1\u017e mechanismu dve\u0159\u00ed \u2013 1. \u010d\u00e1st"),
        ("bg", "\u0417\u0430\u0432\u044a\u0440\u0448\u0432\u0430\u043d\u0435 \u043d\u0430 \u0432\u0440\u0430\u0442\u0430"),
        ("bg", "\u041e\u0411\u041e\u0420\u0423\u0414\u0412\u0410\u041d\u0415 ARD"),
        ("fr", "PREMI\u00c8RE \u00c9DITION"),
        # Genuinely French, no accents/stopword overlap with any other list \u2014
        # the exact class of segment that used to default to a blind 'en'
        # guess (measured 2026-07-21 on a real 92%-French document: 58/163
        # segments this way alone). Honest "no signal" now, not a lucky guess.
        ("??", "OPERATION 20 PERCAGE"),
    ]

    print("=== detect_language self-test (no document bias) ===")
    correct = 0
    for expected, text in samples:
        got, conf = detect_language(text)
        ok = "OK" if got == expected else "FAIL"
        if got == expected:
            correct += 1
        print(f"  {ok}  expected={expected}  got={got}({conf:.2f})  text={text[:50]!r}")
    print(f"\n{correct}/{len(samples)} correct\n")

    print("=== detect_language self-test (with document-dominant-language bias) ===")
    biased_samples = [
        # Same zero-signal segments, now resolved by audit.py's two-pass
        # dominant-language bias instead of a hardcoded default.
        ("cs", "Utahovac\u00ed moment 10,2 N.m", "cs"),
        ("en", "DOOR FINISHING STEP 2", "en"),
        ("fr", "OPERATION 20 PERCAGE", "fr"),
    ]
    correct = 0
    for expected, text, bias in biased_samples:
        got, conf = detect_language(text, default_lang=bias)
        ok = "OK" if got == expected else "FAIL"
        if got == expected:
            correct += 1
        print(f"  {ok}  expected={expected}  got={got}({conf:.2f})  bias={bias}  text={text[:50]!r}")
    print(f"\n{correct}/{len(biased_samples)} correct\n")

    print("=== detect_language_constrained self-test (candidates={fr,en}) ===")
    # The short French technical segments that the free heuristic abandoned
    # (??@0.00) or that fastText alone calls confidently-en. Constrained to the
    # declared {fr,en} pair + a French document bias, they should resolve to fr
    # OR stay capped <0.85 so mono-mode still translates them (never a silent
    # en keep). Requires the fastText model to be present.
    constrained_samples = [
        ("fr", "Percer les trous selon plan"),
        ("fr", "Rédacteur"),
        ("fr", "Vérification dimensionnelle"),
        ("fr", "Contrôle visuel de la pièce"),
        # Cognate title fastText calls en@0.92 — must NOT come back en>=0.85.
        ("keep<0.85", "OPERATION 20 PERCAGE"),
        # Genuine English boilerplate — SHOULD be en with high confidence.
        ("en", "This document is the property of LATECOERE, it may not be communicated."),
    ]
    for expected, text in constrained_samples:
        got, conf = detect_language_constrained(text, {"fr", "en"}, default_lang="fr")
        if expected == "keep<0.85":
            ok = "OK" if not (got == "en" and conf >= 0.85) else "FAIL"
        else:
            ok = "OK" if got == expected else "FAIL"
        print(f"  {ok}  expected={expected}  got={got}({conf:.2f})  text={text[:50]!r}")
    print()

    print("=== DNT extraction self-test ===")
    dnt_samples = [
        "Install NAS6404A22 screw and MS21299C4 washer per IF2.03.08.",
        "Apply PR1440MB2 sealant.",
        "Part number F46C521A0 dated 22/04/2024.",
        "ISO 9001 quality standard.",
    ]
    for s in dnt_samples:
        tokens = extract_dnt_tokens(s)
        print(f"  {s}\n    -> {tokens}")

    print("\n=== is_dnt_candidate self-test ===")
    cands = ["A", "BERNARD-SENTAURENS", "NAS6404A22", "12/16/2024", "10,2", "Door"]
    for c in cands:
        print(f"  {c!r:30s} -> {is_dnt_candidate(c)}")
