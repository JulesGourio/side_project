# Translate

Translates bilingual aerospace `.docx` documents: pick the language side to
replace (source) and the language to replace it with (target), and the tab
runs a full extract → clarify → translate → rebuild → validate pipeline,
ending in a side-by-side before/after preview.

## Background

`Translator/` started as a CLI prototype: Python scripts (originally
`Translator/tools/`) handled the `.docx` structural plumbing, while a human
ran Claude Code interactively to do the linguistic work (translation, term
decisions, glossary building) and manually invoked each pipeline stage. It
was validated end-to-end on two real jobs (MECHANISM1_CS-FR, A321_BG-FR)
before being turned into this web feature.

The web version now *owns* this engine: `server/services/translation/
{extract,audit,fit_check,length_adapt,rebuild,validate,langdetect}.py` are
the modules the server actually runs (vendored from the CLI prototype
2026-07-16, package-relative imports instead of `sys.path` insertion — see
`server/services/processors/translation.py` for the thin in-memory
adapter). The original `Translator/tools/` copies were removed 2026-07-20
once confirmed unused (the web tab fully replaced the interactive CLI
workflow) — they had already drifted from the app's copy by then (see
"Known issue: language detection" below for a fix that was applied only to
the app's copy while it still existed). `Translator/glossary/` (the curated
`terms.csv`/`dnt.csv` and the candidate-extraction CSVs feeding the glossary
review workflow) is unrelated, still-active data and was kept.
The one exception was the prototype's own `compare.py`, which shelled out to
Word COM (`win32com`) + PyMuPDF for PDF side-by-side rendering — Databricks
Apps has no Word install to run that (and, per 2026-07-15 research, no
supported mechanism to install one — see `docs/ROADMAP.md`). The web
version's preview (since 2026-07-16) is an **exact-layout PDF rendering,
full stop**: the server converts both `.docx` through headless LibreOffice
(`server/services/soffice.py`) and the browser shows the PDFs side by side,
full-viewport, in its native viewer (zoom/search/page-nav built in) — the
same class of fidelity the CLI prototype got from Word COM. There is no
lower-fidelity fallback mode anymore (a docx-preview client-side mode
existed for one day and was dropped as not faithful enough — its component
and npm dependency were removed).

How the engine exists at all: Databricks Apps can't *install* LibreOffice
(no apt/Dockerfile/init scripts), but it can *run* a portable tree.
`utils/soffice_packaging/package_libreoffice.py` repackages the official
TDF Linux x64 debs **plus the Ubuntu system libraries the slim container
lacks** (X11/NSS/glib/cups chains — the full ELF DT_NEEDED closure computed
by `utils/soffice_packaging/scan_needed_libs.py`) into a plain tar.gz
(~317 MB) stored in a UC Volume; the app downloads and extracts it once per
process at first use (`SOFFICE_ARCHIVE_VOLUME_PATH`), into a directory
versioned by archive filename (ship a fix by uploading under a new name).

Why the preview is near-instant: at the end of a successful rebuild the
job's two PDFs are pregenerated and persisted next to `output.docx` in the
volume (`_generate_preview_pdfs`), so opening the preview is two plain
file downloads (~1-1.5 s each, measured on uat-test), streamed directly
into the two iframes via `GET /translate/jobs/{id}/preview.pdf?side=…`
(parallel, no base64, `Cache-Control` so re-opens are free). Jobs that
finished before pregeneration existed pay one on-demand conversion at
first open (~20-60 s), which then persists the PDFs for every later view.
Conversions are additionally cached on local disk by content hash.
`GET /translate/render-engine` reports whether the engine is usable on the
current deployment. The download buttons fetch the raw `.docx` bytes
lazily (`GET …/preview`, base64) only when clicked.

The Compare tab's file preview uses the same engine for uploaded docx files
(`POST /api/preview/pdf` + `client/src/components/shared/ExactDocxPreview.tsx`).

## Pipeline stages

```
uploaded → extracting → auditing → awaiting_answers → answered
  → translating → translated → fit_checking → rebuilding → validating
  → done | failed
```

1. **Extract** (`extract_docx_segments`) — walks the `.docx` XML (body
   paragraphs, tables, structured-document-tag containers, and text boxes —
   which appear *twice* in the XML, once in `mc:Choice` and once in
   `mc:Fallback`; both positional paths are recorded so rebuild can update
   both copies identically) and emits one `Segment` per translatable
   paragraph.
2. **Audit** (`audit_segments`) — per-segment language detection **constrained
   to the job's declared {source, target} pair** (the ensemble detector:
   fastText `lid.176` renormalized over the pair + the stdlib heuristic, see
   `langdetect.detect_language_constrained` and "Known issue: language
   detection was noisy/ignored the declared pair" below), plus **document mode
   detection** (`audit.determine_mode`: `monolingual` = translate everything
   source→target; `bilingual` = replace one side, keep the other). Then
   bilingual pairing (adjacent text-box paragraphs, adjacent body paragraphs,
   adjacent table rows, inline "X / Y" or format-split patterns), conflict
   detection (DNT-token mismatch, numeric mismatch, suspicious word-count ratio
   between paired sides), and do-not-translate token extraction (part numbers,
   standards references, dates).
3. **Q&A** (`awaiting_answers` → `answered`) — audit.py deterministically
   generates a batch of clarifying questions; a human answers them through the
   UI (`POST /translate/jobs/{id}/answer`). No LLM call is involved — the
   questions are rule-generated. **Monolingual documents generate no questions**
   (every segment is translated, so there is nothing to clarify pre-translation
   — the reviewer checks the result in the Segments panel afterwards). Bilingual
   documents get flagged-conflict and genuinely-ambiguous-language questions.
   There is no "which side to replace" question (the user already picks
   source/target before upload) and no DNT-token confirmation question (see
   "Glossary" below).
4. **Translate** (`translating` → `translated`) — the document mode drives the
   translate-vs-keep decision (`_plan_segment_translation`): in **monolingual**
   mode every segment is translated except a confidently-detected, long-enough
   target-language passage (kept verbatim); in **bilingual** mode only the
   source-language side is translated. Segments needing
   translation (excluding DNT/numeric-only and kept sides) are
   deduped to unique strings and translated in batches of 40 via
   `call_llm_json` (a non-streaming LLM helper with a JSON-repair retry),
   bounded by `TRANSLATE_MAX_CONCURRENT` concurrent calls. A batch that
   fails or comes back incomplete retries whole, then falls back to
   per-string calls so one bad string can't sink the whole batch;
   unresolved strings are flagged `conflict_flag` for manual review rather
   than failing the job.
5. **Fit-check / length-adapt** — flags translations likely to overflow
   their container (any growth in headers/footers is CRITICAL; 30%/50%/200%
   tolerance for text boxes/tables/body text respectively) and
   automatically shortens overflowing ones using French aerospace
   abbreviation conventions where a rule matches.
6. **Rebuild** (`rebuild_docx_bytes`) — reconstructs the `.docx` with
   translations applied, via `lxml` (preserves namespace declarations and
   attribute formatting byte-for-byte — this is what makes Word accept the
   file; stdlib `ElementTree` mangles prefixes and loses "unused" namespace
   declarations that `mc:Ignorable` still references). Optional, opt-in
   (`include_review_comments` on `POST .../rebuild`, off by default since it
   changes the deliverable): real Word comments (Review pane, threaded-reply
   capable) anchored on flagged/conflicting/failed segments —
   `server/services/translation/review_comments.py` turns already-computed
   signals (conflict_detail, the failed/kept-language category buckets,
   bilingual-inline pattern_type, check_fit's flagged list) into comment
   specs, `comments.py` injects them (ported from the POC's `tools/comments.py`:
   `word/comments.xml` + `commentRangeStart/End`/`commentReference` markers +
   the relationship/content-type entries a real Word comment needs — merges
   with any comments already in the source document rather than dropping
   them). Not ported: the POC's hardcoded Czech/French ambiguous-terminology
   dictionary (vocabulary specific to two pilot documents, not
   generalizable — Phase 1's job-driven glossary candidates cover "flag
   notable terminology" going forward instead).
7. **Validate** (`validate_docx`) — the original prototype's 8-check suite:
   ZIP integrity, XML validity, namespace integrity (`mc:Ignorable`
   references all declared), no `ns0:`/`ns1:` auto-prefix artifacts,
   `[Content_Types].xml` consistency, segment fidelity vs. the original
   extraction, byte-identity of parts that should never change
   (`styles.xml`, `settings.xml`, etc.), and no residual untranslated
   source-language text.
8. **Preview** — the finished-job view is two tabs (`PreviewPanel` in
   `TranslateView.tsx`), all UI in English:
   - **Review (default, `ReviewPanel.tsx`)** — the primary review surface,
     reworked 2026-07-22 (the pixel diff was not a pleasant way to review a
     translation, and a text-only two-column view was "no better than the
     Segments table"). It renders the **two real PDF documents side by side,
     full-width, page by page** with **pdf.js** (`pdfjs-dist`, worker bundled
     via Vite `?url`) — so it IS the faithful document (logo, cartouche, CAD
     images, bordered tables), not a reflowed HTML approximation. Pages render
     lazily (one `IntersectionObserver` per scroll column). NB: each column
     must be a real fixed-height scroller — do **not** put `flex-1` on it, or
     `min-height:auto` makes it grow to the full 66-page content height and
     `scrollTo`/`scrollIntoView` silently no-op.
     A **"difference" is a translated segment** (`GET …/segments`, filtered to
     `category === 'translated'` with `translated_text !== source_text`), so
     its source / translation / language / status are exact and **editable**.
     The enlarged bar shows the current difference (Source + language flag,
     Translation + status badge) and lets you **edit the translation inline**
     (`PATCH …/segments/{seg_id}`, optimistic; a status badge flips to
     "Edited"); **Rebuild** bakes edits into the `.docx`. Navigation is
     **prev/next (buttons + `n`/`p`) and a jump-to-Nth number input**.
     To place the difference on the real page, `ReviewPanel` **text-searches
     the segment's text in the pdf.js text layer** (`getTextContent`,
     punctuation-insensitive `matchNorm`, prefix-anchored, nearest-to-expected
     page). It boxes **only the exact contiguous item run** that spells the
     text (a char→item map, `PageText.concat`/`owner`) — not every word that
     happens to appear on the page, which used to inflate/offset the box and
     make it asymmetric between the two documents. The box is positioned
     against the **actual rendered page element's** `offsetLeft`/`offsetTop`
     (pixel-exact, no page-centering drift). If a match can't be found it falls
     back to scrolling near the proportional page (no box) — **editing stays
     correct regardless of box placement** because the segment id is known, not
     inferred. The bar also shows **"Segment N/total"** (the difference's
     absolute row in the Segments table, from the endpoint's new `index` field)
     so a difference cross-references the Segments panel; the two counts differ
     on purpose (differences = only the translated segments; the table = all
     segments). A **Scroll linked/independent** toggle
     (ref-gated so it flips instantly) proportionally syncs the two columns for
     free scrolling; nav always scrolls both to their own located rect.
   - **Layout (PDF, `LayoutPreview.tsx`)** — the same two PDFs in the browser's
     **native** PDF viewer (iframe), side by side, for native search/zoom/print,
     and where the "download `.docx`" / "download `.pdf`" buttons for both the
     original and the translated document live (fetched as a blob and saved,
     same pattern for both formats — `GET /translate/jobs/{id}/preview` for the
     base64 docx pair, `GET …/preview.pdf?side=` for either PDF).
   The standalone HTML comparison report (`GET …/preview/report.html`,
   `render_comparison_report_html`) was removed — redundant with the Review tab
   and the exact-layout PDF pane above. The pixel-diff endpoints
   (`GET …/preview/diff.png`, `GET …/preview/changes`,
   `pdf_diff.py::get_all_page_changes`) are unused by the client now (the live
   Review tab locates changes itself via the pdf.js text layer) but are left in
   place rather than deleted speculatively.
9. **Manual correction** — a reviewer overrides a segment's translation at
   any stage including after the job is `done`
   (`PATCH /translate/jobs/{id}/segments/{seg_id}`), inline in **either** the
   Review tab's difference bar (Edit) **or** the Segments panel table (job
   header). Marks the segment human-confirmed (clears
   `conflict_flag`/`keep_as_is`) but does not itself touch the output
   `.docx` — the reviewer re-runs rebuild (`start_rebuild` accepts `'done'` as
   a re-runnable status; both the Review toolbar's "Rebuild" button and the job
   page's "Rebuild again" button trigger it) to bake the edits in.

## PDF-origin jobs — tried and abandoned (2026-08-24)

Some FI documents exist only as `.pdf`, no `.docx` source. A full parallel
engine (block-level PyMuPDF extract, redact+reinsert rebuild with a vendored
Unicode font, vision-LLM OCR of text baked into raster images) was built,
deployed to `latlang-uat`, and tested against a real 112-page FI. Two real
bugs were found and fixed (lines within a block glued into one run-on
string with no separator; PyMuPDF's `insert_textbox` silently drawing
*nothing* — not even clipped text — when a translation didn't fit even at
the font-size floor, on ~20% of one document's segments). Even after both
fixes, the result was judged not good enough to ship: no bold/color/table
structure preserved, and pages whose actual content lives inside one raster
image (common in this document's "montage" pages) stayed visually
untranslated except for a small OCR annotation icon.

The deciding comparison: the user tried opening the same PDF directly in
Microsoft Word (`Fichier > Enregistrer sous > .docx`) and got a markedly
more faithful result — tables and layout recognized properly — than this
from-scratch PyMuPDF reconstruction ever produced. **Decision: PDF upload
is no longer accepted; `.docx` only.** Users with only a PDF are directed
to convert it via Word first. All PDF-specific code
(`pdf_extract.py`/`pdf_rebuild.py`/`pdf_validate.py`/`pdf_ocr.py`,
`processors/translation_pdf.py`, the vendored font) was removed — none of
it ever modified the docx code path's own behavior, so this is a clean
revert with no side effects on `.docx` jobs. See `CLAUDE.md`'s status log
for the full account.

## Opt-in: OCR + translate text baked into images embedded in a .docx (2026-08-25)

Reuses the vision-LLM-OCR pattern from the abandoned PDF work (crop →
`call_llm_json` with an `image_url` content block → JSON transcription),
but the "crop" step disappears entirely here: a docx's embedded images are
already standalone files in the zip (`word/media/imageN.*`), so the raw
bytes go straight to the vision call — see
`server/services/translation/docx_images.py`.

- **Selection UI** (`ImageTranslationPicker` in `TranslateView.tsx`): an
  opt-in checkbox on the upload screen ("Also translate text found in
  images"). When checked, the client posts the already-picked file to
  `POST /translate/analyze-images` (stateless, no job created) and shows a
  thumbnail gallery (`list_docx_images`) the user checks images in — not
  every embedded image carries text worth an LLM call (logos, decorative
  photos). The selected filenames are sent as a JSON array in the
  `selected_images` form field alongside the normal upload.
- **OCR**: `_run_job` calls `ocr_docx_images` for the selected filenames
  only, right after normal text extraction — best-effort, same as the PDF
  work's Phase 2 (a credentials/endpoint problem just skips it, never fails
  the job). Each transcribed image becomes a segment like any other
  (`location_type='docx_image_ocr'`, the media filename stashed in the
  already-existing `xml_choice_path` column as `{"media_filename": ...}` —
  no new segment-table column needed) with sentinel `body_p_idx=-1`/
  `para_idx_in_container=-1` so it can never accidentally match the
  bilingual pairing heuristics' adjacency checks (which key off real
  paragraph/table/textbox positions). Flows through the exact same
  audit/translate pipeline as native text.
- **Restitution deliberately never touches the image's pixels** — a real
  lesson from the PDF work: a technical drawing's text is usually scattered
  across the whole image (title, paragraph, several labelled boxes), not
  one caption-friendly area, and resizing the image to fit an overlay would
  distort it inside its existing Word frame (fixed width/height, independent
  of the image's native pixel size). Instead, `insert_image_translations`
  inserts a plain, immediately-visible italic paragraph (prefixed
  `[Traduction]`, line breaks preserved via `<w:br/>` — a literal `\n`
  inside `<w:t>` is not a line break in OOXML) directly after the
  paragraph hosting the image's `<w:drawing>`/`<a:blip r:embed>`, resolved
  via `word/_rels/document.xml.rels`. `_run_rebuild_stage` runs this as a
  post-processing pass over the already-rebuilt docx bytes — the XML-
  position-based rebuild (`build_rebuild_inputs`/`rebuild_docx_bytes`) never
  sees these segments (`xml_choice_path` holds a media filename, not a real
  positional path), they're split out before that call.
- Scope: raster images only (`.png`/`.jpg`/`.jpeg`/`.bmp`/`.gif`/`.tiff`) —
  legacy vector metafiles (`.emf`/`.wmf`, common from old Excel paste-ins)
  are skipped, Pillow can't open them for a thumbnail or OCR crop anyway.
  Headers/footers aren't scanned for images, only `word/document.xml`'s body.
- Validated end-to-end (extract → audit → simulated translation → rebuild →
  image-paragraph insertion) on a real sample
  (`Translator/Sample Docs/FI F7XC535-3PTS-FI02 indice I.docx`, 184
  embedded images): the synthetic image segment survives the audit merge
  with its discriminator fields intact, and the final docx re-opens cleanly
  in `python-docx` (an independent OOXML consumer) with the translated
  paragraph present and its line breaks preserved. **Not yet tested with a
  live vision-LLM call or deployed.**

## Opt-in: translate only a page range (2026-08-27)

`.docx` has no native page concept in its XML — page breaks are computed at
layout time, not stored, unlike a PDF's fixed pages. `server/services/
translation/pages.py::assign_pages` gets an approximate page number per
segment anyway by reusing the LibreOffice PDF conversion already used for
previews (`soffice.py::convert_docx_to_pdf`) and walking that PDF's
per-page text in document order, matching each segment's text as it goes
(forward-only cursor — document order tracks non-decreasing page number).
This mirrors what a reviewer would see opening the doc in Word/a PDF
viewer, but can drift on complex table/text-box layouts; a segment the
mapping can't confidently place is left translatable rather than silently
excluded, same reasoning as never dropping OCR'd image segments above.

- **UI**: an optional "Pages to translate" text field on the upload screen
  accepting `"3-10"`, `"1,4,9"`, or a mix like `"1-3,7,12-15"` — sent as the
  `pages` form field alongside the normal upload, parsed server-side by
  `pages.py::parse_page_spec`. Blank means the whole document, as before.
- `_run_job` resolves the page map once against the ORIGINAL upload right
  after extraction, and marks each audited segment's `out_of_page_range`
  bool (a real `translation_segments` column, not an overload of
  `pattern_type` — keeps `audit.py`'s own language-mode/bilingual-pairing
  logic, which reads `pattern_type`, completely unaffected by the filter).
- Out-of-range segments are never sent to the LLM
  (`_plan_segment_translation` returns `None` for them, same as `dnt`/
  `numeric_only`) and rebuild writes their own source text back unchanged
  — the existing "everything not planned for translation keeps its source
  text" `keep_as_is` logic in `_run_translation_stage` already covers this,
  no separate rebuild-path change needed. They surface in the Segments
  panel under their own `kept_page_filtered` category rather than being
  mislabelled `kept_other_language`.
- Doesn't apply to `docx_image_ocr` segments — an OCR'd image's
  transcription rarely matches the underlying PDF page's own text layer, so
  the mapping just can't place them; they always translate regardless of
  the page filter.
- A restart (`POST .../restart`) reuses the job's original `page_filter`
  the same way it reuses `selected_image_paths`.

## Known issue: detection was noisy AND ignored the declared pair — FIXED 2026-07-21 (rework)

The stopword/diacritic heuristic below was fixed incrementally, but on the real
intraqual corpus it was still catastrophic: a 92%-French document scored 457
segments as `en`, and **41–56 % of mono segments landed below 0.55 confidence**
(the "confidences are all null" the reviewer saw). Worse, the translate stage
decided translate-vs-keep purely from that fragile per-segment label
(`if detected_lang == source_lang`), so **the user's declared source language
was never used as a prior** — any mislabeled segment was silently kept
untranslated. Two-part fix:

1. **Constrained ensemble detection** (`langdetect.detect_language_constrained`):
   fastText `lid.176` (already vendored) renormalized over the declared
   {source, target} pair, combined with the heuristic. Only an *agreement*
   between the two independent detectors reports high (≥0.85) confidence; a
   single confident detector is capped at 0.80 — because fastText is sometimes
   confidently WRONG on short cognate titles (`OPERATION 20 PERCAGE` → en@0.92),
   and the cap keeps such a call below the "keep as target language" bar so it
   can never be silently left untranslated. Measured: low-confidence rate
   50 % → **0–2 %** on the same real docs.
2. **Document mode + declared-pair prior** (`audit.determine_mode`,
   `_plan_segment_translation`): monolingual docs translate every segment
   except a confident, long-enough target-language passage (e.g. an English
   confidentiality boilerplate in a French doc); bilingual docs keep the
   replace-one-side behaviour. This is what eliminated the silent drops.

The reviewer-facing presentation was reworked to match: a visual pipeline
stepper, stat cards + mode/language-pair badges instead of a run-on status
line, an **actionable** residual panel (structured `{seg_id, text,
detected_lang, confidence}` items that deep-link into the Segments panel filtered
to `needs_review`, replacing the old amber `<details>` string dump), and
semantic language/confidence/category badges in the Segments panel (no more bare
`@0.00`). Segment categories are now `translated` / `kept_other_language` /
`kept_dnt` / `kept_numeric` / `needs_review`.

The section below documents the earlier heuristic-only fixes, kept for context.

## Known issue: French false-negatives in language detection — FIXED 2026-07-21

Found on a real document (92% French, `old_FI_252A90008200-B.docx`): 58 of
163 segments were mislabeled `en`, plus several confidently mislabeled `es`
or `cs`, corrupting bilingual pairing (two mislabeled mono segments looked
like a genuine bilingual pair to `pair_adjacent_body`, producing bogus
conflict questions). Three distinct bugs in `langdetect.py`'s heuristic
tiers, all fixed without needing an LLM:

1. **French's own stopword list was missing extremely common French words**
   ("en", "sur", "avec", "dans", "que"...) — some of these (e.g. "en") were
   only listed under *Spanish*, so any French sentence containing them
   silently lost to Spanish on tier 3 (keyword scoring). Fixed by expanding
   `LANG_PROFILES["fr"]["keywords"]` (and rebalancing the other lists for
   the same class of gap).
2. **Arbitrary tie-breaking on shared diacritics** — 'é' is common to
   CS/FR/ES; a one-character tie (e.g. "Rédacteur") silently resolved to
   whichever language happened to sort first in `LANG_PROFILES` (always
   `cs`), not a real signal. Fixed: a genuine tie now defers to keyword
   scoring instead of picking a winner.
3. **Hardcoded `'en'` default when no tier found any signal at all** (short
   technical headings/cognates like "OPERATION 20 PERCAGE", identical in
   French and English) — replaced with an honest `"??"`/0.0, resolved by a
   new two-pass bias: `audit.py::audit_all_segments` runs the whole document
   once, computes the dominant language among the *confident* detections,
   then re-runs only the zero-signal segments with that as `detect_language`'s
   `default_lang` — a signal-free segment in an otherwise-92%-French document
   is overwhelmingly more likely to be French than a blind guess.

Also added the one DNT pattern gap this surfaced: digit-first internal
references (`252A90008200` — every existing pattern started with a letter).

Verified via `python -m server.services.translation.langdetect` (self-test)
and re-running the full extract→audit pipeline against all four POC sample
docs (A321, FI_D5211535000901, MECHANISM1, MECHANISM2) — dominant-language
counts unchanged/sane, no new false positives introduced.

## Known issue: language detection is structurally unsound for mixed-language segments — FIXED 2026-07-17

**Status (2026-07-17): the structural fix is in — mixed-language segments are
now pre-split BEFORE language detection, and only the source-language span is
sent for translation (see "The fix" below). The 2026-07-16 analysis is kept
for context.**

### The bug

`detect_language(text)` returns exactly ONE `(lang, confidence)` label for
whatever string it's given. Every one of its tiers (Cyrillic-script ratio,
shared-diacritic density, keyword-stopword share) scores a language by its
**share of the whole input text**. That's fine for a genuinely monolingual
segment. It silently breaks for a segment that legitimately contains TWO
languages concatenated in the same paragraph/run/table-cell — which is a
*normal, common* pattern in these bilingual aerospace docs (a sentence in
the source language immediately followed by its own English echo, with no
XML formatting boundary between them for the pairing heuristics to split on).
When that happens, whichever language has more raw characters in that
specific segment wins the single label, and the other language's content is
silently treated as "the side to leave untouched" — i.e. it never gets
translated, with no error, no warning, and no consistent pattern a reviewer
can predict (whether a given segment gets flagged/translated/skipped depends
on incidental character counts, not on anything about the content).

Confirmed 2026-07-16 on real segments from a live job (A321 BG→FR):
```
Добавяне на снимка и описание за монтаж на пяните на вентилационния клапан
Adding a photo and description for the installation of the foam on the ventilation valve
```
→ detected as `en @0.85` (Cyrillic ratio 46%, just under the old 50% cutoff)
→ the Bulgarian half was never sent for translation.

This is the same root cause as the `résiduel 88 → 8` bilingual-concatenation
note already logged in `docs/ROADMAP.md` (2026-07-11) — that entry described
the *symptom* (residual untranslated cells) without identifying that the
detector itself, not the extraction/pairing step, is the cause.

### What was fixed today (2026-07-16)

Two of `detect_language`'s tiers were changed from "share of the whole text
exceeds a ratio" to "share OR an absolute count of hits" — mirroring how the
unique-character tier (CS `ř`/`ů`, DE `ß`, FR `œ`, ES `ñ`/`¿`/`¡`) already
worked, since that one was never ratio-gated and never had this bug:

- **Cyrillic tier**: was `cyrillic_ratio > 0.5`; now `cyrillic_count >= 2 and
  cyrillic_ratio > 0.08`. Bulgarian is the only Cyrillic-script language in
  scope, so any non-trivial Cyrillic presence is an unambiguous, unique
  signal — there is no other supported language it could be confused with.
  Validated against the full corpus of 4 sample docs to confirm no
  regression (one single-stray-Cyrillic-character artifact in an otherwise-
  Latin document code correctly stays excluded by the `>= 2` floor).
- **Shared-diacritic tier** (CS/DE/FR/ES common accented letters): was
  `density > 0.02`; now `hits >= 2 or density > 0.02`.

Measured impact on real jobs (`audit_segments`, before/after, same document):

| Document | bg detected | wrong "cs" labels | mis-paired table rows | conflicts flagged |
|---|---|---|---|---|
| A321 (BG→FR) | 84 → 91 | — | 12 → 8 | 9 → 5 |
| FI_D5211535000901 (BG) | 190 → **359** | 34 → 2 | — | 77 → 46 |

### The fix (2026-07-17): structural pre-split + per-span translation

Both options sketched on 2026-07-16 were implemented together, plus the
downstream plumbing that the analysis had missed was overhauled — the
detector was only half the bug; the translation stage itself sent the FULL
segment text to the LLM and rebuilt with a uniform replacement, so even a
correctly-detected bilingual segment either got its kept side translated
away or was skipped whole.

1. **Pre-split** (`audit.py::pair_inline_charspan`, runs after the slash and
   format-concat pairings): finds the language boundary structurally FIRST —
   a script switch (one contiguous Cyrillic block against one Latin block,
   no punctuation needed) or, failing that, sentence boundaries scored per
   side — then labels each side on its own. Matching segments become
   `pattern_type=bilingual_inline_charsplit` with the char offset in
   `inline_split`. Interleaved (A-B-A) or undecidable texts stay `mono`,
   exactly as before — no regression risk.
2. **Per-span detection** (`langdetect.py::detect_span_language`): the
   heuristic detector backed by fastText `lid.176` (the model already
   vendored for the chat bridge) **restricted to the six supported languages
   and renormalized**. Empirical check done before committing, as the
   2026-07-16 note asked: raw fastText is unusable on short spans
   (`'Install the nut.'` → fi@0.33) but the restricted+renormalized form
   recovers the documented failing case (`'Namontujte matici.'` → cs@0.66).
   Only low-confidence heuristic verdicts pay the fastText call.
3. **Per-span translation** (`translate.py::_plan_segment_translation`): for
   ALL bilingual-inline kinds (slash / format / charsplit), only the
   source-language side is sent to the LLM; the stored `translated_text` is
   the composed full replacement (char/slash — rebuild replaces uniformly)
   or the bare span (format — rebuild splices it into the matching-format
   runs only, `span_translated`/`source_fmt` markers on `inline_split`).
   This also fixed the pre-existing latent bug where a bilingual segment
   whose whole-text label matched the source language had BOTH sides
   translated, wiping the kept-language copy.

Measured on the four real sample docs (extract + audit, before → after):
FI_D5211535000901 (BG) mono 313→232 (**+81 segments** now recognized as
bilingual: slash 65→92, format-concat 22→67, charsplit +9); A321 (BG) +7;
MECHANISM1 (CS) +6; MECHANISM2 (CS) +4. Language counts, conflicts, pairs
and question counts unchanged everywhere — no false positives introduced.

**Remaining limitation (accepted, best-effort)**: a mixed segment where the
short side has zero language signal for both detectors (e.g.
`"Osadit oba panty"` — restricted fastText confidently calls it `en`) stays
`mono` with one label, same as before. These still surface through the
residual-source-language validation check and the conflict review, not
silently. The rule from 2026-07-16 stands: if another half-translated
segment class shows up, extend the *splitting* logic (`audit.py`), do not
nudge `langdetect.py` thresholds.

## Verification (this session)

Extraction reproduced the exact segment counts from the two real,
previously-validated CLI jobs (1,248 for MECHANISM1_CS-FR, 303 for
A321_BG-FR — matching their `job.json` manifests exactly). The full
rebuild round-trip (extract → audit → apply the real historical
`translation.json` → rebuild → validate) produced the already-shipped,
human-validated `Translator/Translated/MECHANISM1_FR.docx` with **zero text
differences** across all 1,248 segments, and all 8 structural validation
checks passed. `call_llm_json` and the batching/retry/fallback logic were
verified against a mocked HTTP transport (no live LLM call needed for that).

## Async job model

No task queue was added — jobs run as `asyncio.create_task` fire-and-forget
coroutines (the same pattern the Compare tab uses for its background volume
upload), matching this app's single-process deployment. Each stage
transition refreshes a `worker_pid`/`worker_heartbeat` pair; on startup, any
job left in a non-terminal state with a stale heartbeat (> 2 minutes) is
marked `failed` with `error_type='OrphanedJob'` — the whole crash-recovery
story, since only one process ever runs a job at a time. The client polls
`GET /translate/jobs/{id}` every ~2.5s for status/progress; this was chosen
over SSE because a multi-minute job surviving a page refresh needs to be
resumable by re-polling the same job id, and the existing SSE plumbing
(`stream_chat`/`stream_analysis`) has no reconnect story.

**Accepted limitation**: a restart mid-translation-batch loses that
in-flight batch (the stage restarts from its beginning on next trigger, not
mid-batch) — not solved by design, since it would need per-batch
checkpointing for marginal benefit at this job scale.

## Glossary

`glossary_terms` (one row per concept, one column per language:
en/fr/cs/bg/de/es/pt/ar), `glossary_candidates` (legacy pending-review queue, no
longer written to — see below), and `dnt_rules` (pattern-based
do-not-translate matching: exact/prefix/glob/regex) live in Lakebase, not the
prototype's `Translator/glossary/*.csv` files — CSV files can't handle
concurrent writes from multiple users safely. Managed via `GET/POST
/translate/glossary` + `POST/DELETE /translate/glossary/{terms,dnt}` and the
in-app glossary panel.

**Candidate extraction — LLM-verified, no per-term human review (redesigned
2026-07-21).** Candidates are extracted offline per language pair from the
corpus (`utils/glossary/build_review_batch.py`: chunk-aligns two documents of
the same family in different languages by shared anchors [part numbers,
dimensions] + position, then an LLM extracts term-pair correspondences from
each aligned section; the same term seen across multiple document families
raises confidence). Since these pairs come from real, already-published
professional translations, they're treated as correct by default — there is
no per-term human approve/reject step. `utils/glossary/sync_candidates_to_lakebase.py`
(dry-run by default; `--apply` to write) dedupes against both `glossary_terms`
and any leftover `glossary_candidates` rows, groups the rest by language pair,
and runs each group through `llm_verify` — an LLM pass whose only job is to
catch **extraction artifacts** (a document heading/title mistaken for a term,
a sentence fragment, a chunk-misalignment pairing two unrelated words) and
**normalize stray casing** (a document heading's ALL-CAPS or a
sentence-initial capital, kept only for genuine proper nouns/acronyms) —
verified pairs are inserted directly into `glossary_terms` (auto `T####` id),
never into `glossary_candidates`. A batch/row the LLM call fails on is kept
as-is (fail-open), so an endpoint outage can't silently drop good terms.
`glossary_candidates` and the panel's "To review" tab (approve/reject) still
exist for manual additions, and still hold whatever was published there
before this redesign (701 rows in uat-test as of 2026-07-21, not migrated —
they'd need a one-off run through the new script's verify step to promote
into `glossary_terms`, or manual review through the existing UI).

**Known gap in the extraction script (fixed 2026-07-21, not backfilled).**
`extract_glossary_candidates.py`/`build_review_batch.py` used to record a
candidate's `source` as only the source-language document ref (`ref_a`), even
though the pair came from aligning it against a same-family document in the
*other* language (`ref_b`) — so the "Sources" column in the panel always
listed documents in a single language, which looked like a bug (and is one).
Fixed to record the language-independent family id (`fam['base']`) instead,
so a term's sources reflect the actual bilingual document pair. Existing rows
already in Lakebase (both `glossary_terms` and `glossary_candidates`) keep
their old single-language source strings — only future extractions get the
fix.

**DNT rules — no more per-job auto-confirm (fixed 2026-07-21).** Dates, part
numbers, and standards references are recognized generically by a hardcoded
regex (`DNT_REGEX` in `langdetect.py`) that already blocks translation of a
matching segment/token regardless of `dnt_rules` content. The audit step used
to also generate a `dnt_confirm` question listing every regex-matched token
seen in the document, and an affirmative answer inserted each one as a new
exact-match row in `dnt_rules` (`notes = 'Confirmed from job N'`) — but since
a date or PN is almost never repeated verbatim in a later document, this
never actually deduplicated anything; it only grew `dnt_rules` with one-off
junk (168 of 185 rows in uat-test as of 2026-07-21) with zero effect on any
translation outcome, since `dnt_rules` wasn't even wired into the live
audit/translate decision. Both problems are now fixed together: the
`dnt_confirm` question/auto-insert is removed, and `dnt_rules` — now purely a
human-curated list edited by hand in the glossary panel — is compiled into a
`DntMatcher` (`glossary_io.py`) and applied live in `audit.py::make_audited`
alongside `DNT_REGEX`, so a manually-added rule (e.g. a company-specific
product code, in any of the four match modes) actually changes what gets
translated in the next job.

## Capability gating

Same pattern as Compare/Chat: `require_translate` dependency on every
endpoint → `get_capabilities()` reads `users.can_translate` from Lakebase →
fails open if Lakebase is unreachable or the user isn't synced yet.
`TranslatePage.tsx` shows `AccessDenied` if `can_translate` is false, or a
"coming soon" placeholder if `TRANSLATE_ENABLED` is false.

**Not yet done**: the Databricks capability group for `can_translate`
(`Role-Project-LEAP-End-users-LatLang`) hasn't been provisioned — an
IT/directory action, not code: as of 2026-07-17 no such end-user group
exists in the workspace at all. The sync
side is ready since 2026-07-17 (`CAPS_TRANSLATE_GROUPS` in
`utils/databricks_ops/config.py`, wired through
`sync_user_capabilities.py` and the UAT notebook, column written only where
it exists) — the day the group is created, membership flows through with no
code change. Until then `can_translate` stays fail-open (everyone gets
access), same as any newly-added capability column with no synced group
yet.

## Configuration (`app.yaml`)

| Var | Default | Purpose |
|---|---|---|
| `TRANSLATE_ENABLED` | `true` | Feature flag |
| `TRANSLATE_ENDPOINT` | falls back to `COMPARE_ANALYSIS_ENDPOINT` if unset; `databricks-gpt-5-4-mini` on uat/uat-test | LLM endpoint for Q&A/translation calls |
| `TRANSLATE_VOLUME_PATH` | empty; set on uat/uat-test (subfolder of the compare volume) | UC Volume path for job input/output `.docx` files; absence disables rebuild/validate/preview (job stops cleanly at `translated`) |
| `TRANSLATE_MAX_CONCURRENT` | `3` | Concurrent batched-translation LLM calls per job |
| `SOFFICE_ARCHIVE_VOLUME_PATH` | empty; set on uat/uat-test | UC Volume path of the portable LibreOffice tar.gz (exact-PDF preview engine); absence disables the Exact (PDF) mode |
| `SOFFICE_PATH` | empty | Local-dev escape hatch: path to an existing soffice binary (takes precedence over the archive) |
| `SOFFICE_MAX_CONCURRENT` | `2` | Parallel LibreOffice conversion processes |
| `SOFFICE_CONVERT_TIMEOUT_S` | `180` | Per-document conversion timeout |

Set for `uat`/`uat-test` since 2026-07-15 by `utils/deploy/deploy_latlang.ps1`
(`TranslateFeatureEndpoint`/`TranslateVolumePath` per target) — see
[`docs/ROADMAP.md`](../../../../docs/ROADMAP.md) for what's still open.

## Token/cost tracking

Since 2026-07-15, every batched-translation LLM call (`call_llm_json`)
records its token usage: one row per call in `translation_llm_calls`, plus
running totals (`llm_call_count`, `total_input_tokens`, `total_output_tokens`,
`total_cost_eur`) on `translation_jobs`, returned by `GET /translate/jobs/{id}`
and shown in the job's `StageDetails` panel. Cost uses the same pricing table
as Compare (`server/services/streaming.py::_PRICING_USD`) — unlisted
endpoints fall back to Sonnet's rate, so any new `TRANSLATE_ENDPOINT` should
get its own entry there or costs will be overstated.

## Technical reference

Deep-dive material for anyone modifying this feature — exact routes, schemas, and
module internals, file:line accurate as of 2026-07-22. The sections above already
tell the narrative (why the bugs happened, why the design is what it is); this is
the structural reference that complements it rather than repeating it.

### Route inventory (`server/routers/translate.py`, all behind `Depends(require_translate)`)

| Route | Function | Notes |
|---|---|---|
| `POST /translate/jobs` | `create_translation_job` | Validates + uploads, inserts `status='uploaded'`, fires `asyncio.create_task(_run_job(...))`, returns `{id}` immediately — the pipeline runs fully detached from the request/response cycle. |
| `GET /translate/jobs/{id}` | `get_translation_job` | Computes a live `stale` flag (heartbeat older than `_STALE_HEARTBEAT_S`=120s while in an active-processing status) on every call — distinct from the one-time startup reconciliation, see Async job model below. |
| `POST /translate/jobs/{id}/restart` | `restart_translation_job` | 409 while `ACTIVE_PROCESSING_STATUSES`, or if `input_volume_path` was never persisted. Re-downloads the original upload, `DELETE`s `translation_segments`/`translation_questions` for the job, zeroes the LLM usage counters, resets `status='uploaded'`, and re-fires `_run_job` — a full do-over on the same job id, not a new job. |
| `PATCH /translate/jobs/{id}/notes` | `update_job_notes` | Free-text reviewer notes, `{notes: str}`. |
| `POST /translate/jobs/{id}/validate` | `validate_translation_job` | 409 unless `status in ('done', 'done_with_warnings')`. Sets `glossary_validated_at=NOW()` and fires `_propose_glossary_candidates` — the human gate the Segments panel's "Validate translation" button sits on. |
| `GET /translate/jobs/{id}/segments` | `list_translation_segments` | Categorizes every segment (`kept_dnt`/`kept_numeric`/`kept_page_filtered`/`translated`/`needs_review`/`kept_other_language`/`image_translation`/`pending`) and paginates in Python after categorizing, not in SQL. |
| `PATCH /translate/jobs/{id}/segments/{seg_id}` | `update_segment_translation` | `{translated_text}` — updates the segment row only; does **not** touch the output `.docx`. A rebuild must be re-run to bake the edit in. |
| `POST /translate/jobs/{id}/answer` | `answer_translation_questions` | `{answers: [{q_id, answer}]}`; 409 unless `awaiting_answers`; transitions to `answered` once every question has a non-null answer. |
| `POST /translate/jobs/{id}/translate` | `start_translation` | 409 unless `status in ('answered', 'failed')`, and `failed` only retries if the audit stage had already completed. |
| `POST /translate/jobs/{id}/rebuild` | `start_rebuild` | 409 unless `status in ('translated', 'failed', 'done', 'done_with_warnings')` — both terminal-success statuses are explicitly re-runnable (idempotent overwrite of `output.docx`), which is what backs both "Reconstruire" buttons described above. Internally loops fit_check → rebuild → validate up to twice: if the residual-source-language check finds segments still untranslated, it re-translates exactly those via the LLM and rebuilds once more before settling; final status is `done_with_warnings` (not `done`) if residual text or a conflict-flagged segment survives that. |
| `GET /translate/jobs/{id}/preview` / `preview.pdf` | `get_translation_preview[_pdf]` | Tries pregenerated PDFs first, falls back to on-demand `convert_docx_to_pdf` (501 if the LibreOffice engine is unavailable, 502 on a conversion error), then persists the result for next time. |
| `GET /translate/jobs/{id}/preview/diff.png` / `/changes` | — | Pixel-diff backend, unused by the client (the live Review view locates changes itself via the pdf.js text layer) — left in place rather than deleted speculatively. The former `/report.html` (standalone HTML comparison report) was removed as redundant. |
| `GET /translate/render-engine` | `get_render_engine_status` | Diagnostic: reports whether the LibreOffice engine is usable on the current deployment, no job needed. |
| `GET /translate/jobs` | `list_translation_jobs` | Paginated job list. |
| `GET/POST/DELETE /translate/glossary...` | `list_glossary`, `upsert_glossary_term`, `delete_glossary_term`, `add_dnt_rule`, `delete_dnt_rule` | Direct CRUD against `glossary_terms`/`dnt_rules`. |
| `GET /translate/glossary/candidates` + `.../approve` / `.../reject` | — | Manual-addition review queue — separate from the LLM-verified auto-sync pipeline described under Glossary below. |

### `translation_jobs` — schema and state machine

```sql
CREATE TABLE translation_jobs (
    id                       SERIAL PRIMARY KEY,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id                  TEXT,             workspace_id TEXT,
    status                   TEXT NOT NULL DEFAULT 'uploaded',
    stage_progress           TEXT,             -- JSON blob
    source_lang              TEXT,             target_lang  TEXT,
    original_filename        TEXT,
    input_volume_path        TEXT,             output_volume_path TEXT,
    segment_count            INTEGER,          needs_translation_count INTEGER,
    error_type               TEXT,             error_msg    TEXT,
    worker_pid               INTEGER,          worker_heartbeat TIMESTAMPTZ,
    claimed_at               TIMESTAMPTZ,
    llm_call_count           INTEGER NOT NULL DEFAULT 0,   -- added by migration
    total_input_tokens       INTEGER NOT NULL DEFAULT 0,
    total_output_tokens      INTEGER NOT NULL DEFAULT 0,
    total_cost_eur           DOUBLE PRECISION NOT NULL DEFAULT 0,
    notes                    TEXT
);
CREATE INDEX translation_jobs_user_idx ON translation_jobs(user_id);
```

State machine, exactly as diagrammed above but with the function driving each
transition:

| Transition | Driving function |
|---|---|
| → `uploaded` | `create_translation_job` |
| `uploaded` → `extracting`/`auditing`/`awaiting_answers` (or straight past, for monolingual docs with no questions) | `_run_job` |
| `awaiting_answers` → `answered` | `answer_translation_questions`, once every question has an answer |
| `answered`/`failed` → `translating` → `translated` | `_run_translation_stage` |
| `translated`/`failed`/`done` → `fit_checking` → `rebuilding` → `validating` → `done`\|`failed` | `_run_rebuild_stage` |
| any active status → `failed` (`error_type='OrphanedJob'`) | startup reconciliation, see below |

Every write funnels through a single `_update_job` helper that uses `COALESCE` so a
partial update never clobbers unrelated columns, and refreshes
`worker_pid`/`worker_heartbeat` **only** when a `status` value is being written —
i.e. only while this process is actively driving the job forward.

### Supporting tables

```sql
translation_segments (id, job_id FK CASCADE, seg_id, part, location_type,
    xml_choice_path, xml_fallback_path,           -- JSON positional paths (see extract.py below)
    source_text, detected_lang, lang_confidence, pattern_type, pair_id,
    conflict_flag, conflict_detail, dnt_tokens, inline_split,
    translated_text, keep_as_is, answered_question_id,
    UNIQUE (job_id, seg_id))

translation_questions (id, job_id FK CASCADE, q_id, seg_ids, category,
    question_text, context, suggested_answer, answer, answered_at,
    UNIQUE (job_id, q_id))

translation_llm_calls (id, created_at, job_id FK CASCADE, endpoint_name,
    batch_size, input_tokens, output_tokens, cost_eur)

glossary_terms (id, created_at, updated_at, term_id TEXT UNIQUE, en, fr, cs, bg, de, es, pt, ar,
    domain, notes, definition, definition_source)

glossary_candidates (id, created_at, updated_at, en, fr, cs, bg, de, es, pt, ar,
    n_docs INTEGER DEFAULT 0, sources, definition, definition_source,
    priority, status TEXT DEFAULT 'pending', reviewed_by, reviewed_at, reject_reason)

dnt_rules (id, created_at, pattern TEXT NOT NULL, type, match_mode TEXT NOT NULL, notes)
-- match_mode ∈ {exact, prefix, glob, regex}
```

### Pipeline module reference (`server/services/translation/`)

- **`extract.py`** — `positional_path(elem, root, parent_map)` returns
  `[[tag, child_idx], ...]` from the document root down to a paragraph; this exact
  structure is what gets stored as `xml_choice_path`/`xml_fallback_path` per segment,
  and `rebuild.py::walk_path` is its precise inverse, used to relocate the segment
  during rebuild. Text boxes are extracted from **both** `mc:Choice` and
  `mc:Fallback` (they appear twice in the XML for backward-compatibility reasons),
  and both positional paths are recorded so a translation gets applied to both
  copies identically.
- **`audit.py`** — `make_audited` is the per-segment audit builder (DNT check →
  numeric-only check → language detection). `audit_all_segments` runs a genuine
  **two-pass** detection: pass one runs unconstrained across all 6 languages purely
  to discover which languages the document actually contains
  (`_significant_languages`); pass two re-detects constrained to
  `{source, target} ∪ significant`; a third pass biases any zero-signal (`"??"`)
  segment toward the document's own dominant confident language. `determine_mode`
  decides monolingual vs. bilingual via a threshold (bilingual iff confident
  non-source segments are both `≥5` in count and `≥15%` of the total) — and compares
  the source language against *all* non-source content, not specifically the target,
  so a genuinely third-language document is still handled sanely.
  `pair_inline_charspan` is the structural pre-split: it looks for a script switch
  (a contiguous Cyrillic block against a Latin block) or, failing that, scores
  sentence boundaries per side — segments it can't confidently split stay `mono`,
  with no regression risk to the existing pairing heuristics.
  `generate_questions` returns `[]` immediately whenever `mode != 'bilingual'` — this
  is the exact code path behind "monolingual documents generate no questions."
- **`langdetect.py`** — `LANG_PROFILES` holds, per language, a `unique`-character set,
  a `common`-diacritic set, and a `keywords` stopword list; `detect_language` runs
  these as ordered tiers (Cyrillic script → unique diacritic → shared diacritic →
  keyword scoring), each tier requiring either a ratio **or** an absolute hit count
  so a short/mixed segment isn't outvoted by raw character counts. There is no
  hardcoded `'en'` fallback left anywhere — a genuine no-signal segment returns
  `("??", 0.0)` and is resolved later by the document's own dominant-language bias,
  not a guess. `detect_language_constrained` is the primary web-pipeline entry
  point: it runs both the heuristic and a restricted-and-renormalized fastText
  `lid.176` pass, and only reports high confidence (≥0.85) when the two independently
  **agree** — a single confident detector alone is capped at 0.80, specifically so a
  confidently-wrong fastText call on a short cognate title can never cross the
  "keep as target language" bar and get silently left untranslated.
  `DNT_REGEX` recognizes fastener/PN patterns, internal references, standards codes,
  and two date formats via a single compiled alternation — extended in 2026-07-21
  with a "digit-first" internal-reference pattern.
- **`fit_check.py`** — per-location-type overflow tolerances (headers/footers: any
  growth is `CRITICAL`; text boxes ~30%, tables ~50%, body text ~200% before
  flagging), used both to decide what `length_adapt.py` should try to shorten and
  what `review_comments.py` turns into an inline Word comment.
- **`rebuild.py`** — the lxml-based `apply_translations_lxml` is the variant actually
  used in production (a stdlib-`ElementTree` variant also exists but isn't wired in)
  — lxml is what preserves namespace declarations and attribute formatting
  byte-for-byte, which is what makes Word accept the rebuilt file at all.
  `merge_root_tags` operates at the raw-bytes level specifically to avoid
  ElementTree's automatic `ns0:`/`ns1:` prefix mangling and its habit of dropping
  "unused" namespace declarations that `mc:Ignorable` still references.
- **`validate.py`** — the 8-check suite splits cleanly into **structural** checks
  (ZIP integrity, XML validity, namespace integrity, no auto-prefix artifacts,
  content-types consistency, segment fidelity, byte-identity of untouched parts —
  any failure here fails the job) versus **`check_residual_source_language`**, which
  is a quality signal only and never fails the job on its own.
- **`comments.py` / `review_comments.py`** — `generate_review_comments` turns
  already-computed signals (`conflict_detail`, category buckets, bilingual-inline
  `pattern_type`, `check_fit`'s flagged list) into comment specs;
  `inject_comments` merges them with any comments already present in the source
  document (rather than dropping pre-existing ones) and patches all three places a
  real Word comment needs an entry: `word/comments.xml`, the relationships part,
  and `[Content_Types].xml`.
- **`glossary_io.py`** — `DntMatcher` compiles each `dnt_rules` row once at load time
  according to its `match_mode` (exact/prefix/glob/regex); wired live into
  `audit.py::make_audited` alongside the built-in `DNT_REGEX`, so a manually-added
  rule now genuinely changes what the *next* job translates (previously true only on
  paper — fixed 2026-07-21).

### LibreOffice engine (`server/services/soffice.py`)

`find_soffice()` resolves the binary once per process, in priority order:
`SOFFICE_PATH` env (local-dev escape hatch) → an already-extracted portable tree →
`SOFFICE_ARCHIVE_VOLUME_PATH` (triggers a one-time download+extract from the UC
Volume, into a directory versioned by archive filename) → a handful of well-known
system paths. `convert_docx_to_pdf` caches conversions on local disk keyed by content
hash — confirmed by a dedicated test that a cache hit needs no soffice binary present
at all, i.e. a deployment can serve previously-converted PDFs even if the engine
itself is currently unavailable. Conversion concurrency is bounded by
`SOFFICE_MAX_CONCURRENT`; `SofficeUnavailable`/`SofficeConversionError` are the two
distinct exception types that map to the 501/502 responses on the preview routes.

### Glossary system — two parallel pipelines

1. **Per-job, human-gated, best-effort**: `_propose_glossary_candidates` mines
   term-pair candidates from a `done` job's `(source_text, translated_text)`
   segment pairs via `glossary_extract.py`. It is **not** fired automatically
   on rebuild — a reviewer must click "Validate translation" in the Segments
   panel (`POST /translate/jobs/{id}/validate`, sets `glossary_validated_at`)
   before any term pair reaches the `glossary_candidates` queue; re-clicking
   after further segment edits + a rebuild re-runs the extraction over the
   segments' current state.
2. **Corpus-wide, offline, LLM-verified** (redesigned 2026-07-21):
   `utils/glossary/build_review_batch.py` chunk-aligns two same-family documents in
   different languages by shared anchors (part numbers, dimensions) + position, then
   extracts term-pair correspondences per aligned section.
   `sync_candidates_to_lakebase.py` (dry-run by default, `--apply` to write) dedupes
   against both `glossary_terms` and any leftover `glossary_candidates` rows, then
   runs each language-pair group through `llm_verify` — a strict-reviewer LLM pass
   whose only job is catching extraction artifacts (a document heading mistaken for
   a term, a chunk-misalignment pairing two unrelated words) and normalizing stray
   casing. Verified pairs go **directly** into `glossary_terms` — never into
   `glossary_candidates` — since the source pairs are drawn from real, already-published
   professional translations and are treated as correct by default. Any batch/row the
   LLM call fails on is kept as-is (fail-open), so an endpoint outage can't silently
   drop good terms.

Both pipelines share the same `T####` term-id generation logic (`MAX(term_id) + 1`,
duplicated in both the router's raw SQL and `glossary_io.py`'s CSV-era helper — a
side effect of the CLI-to-web migration, not an intentional abstraction).

### Frontend

- **`TranslateView.tsx`** — polls `GET /translate/jobs/{id}` every
  `POLL_INTERVAL_MS = 2500`ms while the job is in a non-terminal status
  (`TERMINAL_STATUSES = ['done', 'done_with_warnings', 'failed']`); tolerates transient network blips by
  only giving up after 4 **consecutive** failed poll attempts, not the first one.
- **`ReviewPanel.tsx`** — the default post-job view: a pure two-column,
  segment-aligned HTML render built entirely from the already-`ORDER BY id` segments
  endpoint (no extra computation needed for reading order). Change-navigation
  (`n`/`p` keys) steps only through `translated`+`needs_review` segments — a
  deliberate choice, since the meaningful unit here is a whole language-swapped
  segment, not an intra-segment word diff. Inline edits go straight to the same
  `PATCH .../segments/{seg_id}` endpoint the Segments panel uses.
- **`SegmentsPanel.tsx`** — `saveEdit` PATCHes the segment, optimistically updates
  local category/flag state, and explicitly toasts that a rebuild is still required
  to apply the edit to the actual document — the PATCH never triggers a rebuild
  itself, by design. Also hosts the "Validate translation" button (`done` jobs
  only) that gates glossary candidate proposal — see Glossary system above.
- **`LayoutPreview.tsx`** — the exact-layout PDF side-by-side (the LibreOffice engine
  described above) plus the `.docx` download buttons (base + translated) — the only
  place documents are downloaded from; there is no separate job-artifact export.
- **`GlossaryPanel.tsx`** — CSV export helpers, paginated term/candidate lists
  (50/page), and the manual approve/reject flow for the legacy `glossary_candidates`
  review queue (distinct from the LLM-verified auto-sync pipeline above).

### Async job model — orphan detection in detail

No task queue: every stage transition is a fire-and-forget
`asyncio.create_task(...)`, matching this app's single-process deployment — the same
pattern Compare uses for its background volume upload. On startup (`server/app.py`
lifespan, right after `init_lakebase()`), a one-time reconciliation query marks any
job still sitting in an **active-processing** status (`extracting`, `auditing`,
`translating`, `fit_checking`, `rebuilding`, `validating` — deliberately **excluding**
the human-gated resting states `awaiting_answers`/`answered`/`translated`, which are
supposed to survive a restart while waiting on a reviewer) as `failed` with
`error_type='OrphanedJob'`, if its heartbeat is missing or older than 2 minutes. The
same 120-second threshold is also checked live on every `GET /translate/jobs/{id}`
call (as the `stale` flag), independent of the one-time startup sweep — so a UI
polling an in-progress job can surface "this looks stuck" well before any restart
actually happens.

**Accepted limitation**: a restart mid-translation-batch loses that in-flight batch
entirely (the stage restarts from its beginning on the next trigger, not
mid-batch) — not solved by design, since per-batch checkpointing would add real
complexity for marginal benefit at this job scale.

### Token/cost tracking — the retry/repair detail

`call_llm_json` is a single non-streaming chat-completion call with a JSON-repair
retry: if the response isn't valid JSON, it retries **once** with an appended
correction turn ("Your previous response was not valid JSON..."), summing token
usage across both calls; a second failure propagates uncaught to the caller (which is
what triggers the batch-level fallback to per-string calls described above). Pricing
(`streaming.py::_PRICING_USD`) is a flat per-endpoint table; any `TRANSLATE_ENDPOINT`
not listed there silently falls back to Sonnet's rate — worth an explicit entry for
any new endpoint, or cost telemetry will overstate the real spend.

### Tests (`tests/test_translate.py`)

Grouped by module: `langdetect` (the French-not-lost-to-Spanish keyword regression,
tie-breaking, no-hardcoded-fallback, digit-first DNT pattern, dominant-language
bias), `glossary`/`glossary_extract` (batch-to-source-string mapping, malformed-LLM-payload
tolerance, frequency ranking/capping), `review_comments`/`comments` (all comment
categories, unknown-seg_id-skipped-not-raised), `fit_check`/`length_adapt`
(untranslated segments correctly skipped by both), `audit.py`'s inline splitting
(real BG/EN and CS/EN segments from actual sample docs), `_plan_segment_translation`
(every pattern_type × mode combination, explicitly including the case where a
fastText-mislabelable French cognate title must still be translated in monolingual
mode — no silent keep), `rebuild.py` (lxml format-preservation), and `soffice.py`
(unavailable-when-unconfigured, cache-hit-needs-no-engine).

See the root [README.md](../../../../README.md) for deploy/environment/local-testing instructions shared across all three features.
