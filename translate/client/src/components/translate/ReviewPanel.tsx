import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChevronUp, ChevronDown, Loader2, RefreshCw, Link2, Link2Off, Check, X, Pencil, Layers, Languages } from 'lucide-react';
import { toast } from 'sonner';
import * as pdfjsLib from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { Flag } from '../shared/Flag';
import { fetchJson } from '../../lib/fetchJson';
import {
  norm, matchNorm, matchIn, rectFor, mapPos, pageOfScroll, scrollForPageFrac, sizesFor,
  type PageText, type RectFrac, type PageAnchor,
} from './reviewPanel/textMatch';

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

// ---------------------------------------------------------------------------
// Document review — the two REAL PDF documents rendered page-by-page (pdf.js)
// side by side, full width. By default steps through "differences" (segments
// that were actually translated), but a toggle switches to every segment —
// needed to catch ones wrongly left untranslated (e.g. a mis-detected
// language) that never show up as a before/after difference. Each step's
// source/translation/language/status are exact and editable inline. The
// toolbar shows the current step and lets you edit its translation on the
// spot; the PDFs scroll to that step and box it (located best-effort via
// pdf.js text search — editing stays correct even when a box can't be placed).
// ---------------------------------------------------------------------------

const ISO_TO_FLAG: Record<string, string> = { fr: 'FR', en: 'EN', es: 'ES', cs: 'CZ', bg: 'BG', pt: 'BR' };
const LANG_OPTIONS = ['fr', 'en', 'es', 'de', 'cs', 'bg', 'pt', 'ar'];

const CATEGORY_STATUS: Record<string, { label: string; fg: string; bg: string }> = {
  translated: { label: 'Translated', fg: '#15803d', bg: 'rgba(22,163,74,0.12)' },
  kept_other_language: { label: 'Kept (other lang)', fg: 'var(--color-accent-primary)', bg: 'rgba(0,85,164,0.10)' },
  kept_dnt: { label: 'Do-not-translate', fg: 'var(--color-text-muted)', bg: 'var(--color-bg-secondary)' },
  kept_numeric: { label: 'Numeric', fg: 'var(--color-text-muted)', bg: 'var(--color-bg-secondary)' },
  needs_review: { label: 'Needs review', fg: '#b45309', bg: 'rgba(245,158,11,0.14)' },
  pending: { label: 'Pending', fg: 'var(--color-text-muted)', bg: 'var(--color-bg-secondary)' },
};

interface Segment {
  seg_id: string;
  source_text: string;
  translated_text: string | null;
  detected_lang: string | null;
  category: string;
  flagged: boolean;
  conflict_detail: string | null;
}

interface LocateHit { rect: RectFrac; page: number; pos: number }

async function loadPdf(url: string): Promise<pdfjsLib.PDFDocumentProxy> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.error || res.statusText);
  }
  const buf = await res.arrayBuffer();
  return pdfjsLib.getDocument({ data: new Uint8Array(buf) }).promise;
}

const RENDER_QUALITY = 2;

export function ReviewPanel({
  jobId, sourceLang, targetLang, onRebuild, rebuilding, includeReviewComments, onIncludeReviewCommentsChange, focusRequest,
}: {
  jobId: number;
  sourceLang: string;
  targetLang: string;
  onRebuild: () => void;
  rebuilding: boolean;
  includeReviewComments: boolean;
  onIncludeReviewCommentsChange: (v: boolean) => void;
  // Set from outside — the Segments panel's "View in document" button — to
  // jump straight to that segment's diff. `token` just needs to change on
  // every click (even re-clicking the same segment) to retrigger the jump.
  // A segment invisible to the diffs-only view (kept-as-is, pending) needs
  // reviewAll flipped on first, so this can take two passes.
  focusRequest?: { segId: string; token: number } | null;
}) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [allSegs, setAllSegs] = useState<(Segment & { docIndex: number })[]>([]);
  const [reviewAll, setReviewAll] = useState(false);
  const [docTotal, setDocTotal] = useState(0);
  const [beforeSizes, setBeforeSizes] = useState<{ w: number; h: number }[]>([]);
  const [afterSizes, setAfterSizes] = useState<{ w: number; h: number }[]>([]);
  const [current, setCurrent] = useState(-1);
  const [jumpValue, setJumpValue] = useState('');
  const [linked, setLinked] = useState(true);

  // inline edit
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  const [editedIds, setEditedIds] = useState<Set<string>>(new Set());
  const [overrides, setOverrides] = useState<Record<string, string>>({});

  const beforeCol = useRef<HTMLDivElement>(null);
  const afterCol = useRef<HTMLDivElement>(null);
  const pdfBefore = useRef<pdfjsLib.PDFDocumentProxy | null>(null);
  const pdfAfter = useRef<pdfjsLib.PDFDocumentProxy | null>(null);
  const linkedRef = useRef(true);
  const suppressSync = useRef(false);
  // cached page text (with per-item positions) for locating a segment on a page
  const textCache = useRef<{ before: Map<number, PageText>; after: Map<number, PageText> }>({ before: new Map(), after: new Map() });
  // last matched (page, concat-position) per pane, used to keep forward
  // navigation searching AFTER the previous match instead of re-scanning from
  // the top of the page — otherwise a word/value that repeats many times in
  // the same table (common in BG source docs) always resolves to its first
  // occurrence, so every later difference on that page boxes the same spot.
  const cursorRef = useRef<{ before: { page: number; pos: number } | null; after: { page: number; pos: number } | null }>({ before: null, after: null });
  // active-box element per pane
  const beforeBox = useRef<HTMLDivElement>(null);
  const afterBox = useRef<HTMLDivElement>(null);
  // per-difference (before page, after page) correspondence, resolved lazily
  // (on navigation and by a background prefetch) — lets linked-scroll map
  // position through real anchors instead of one global length ratio, which
  // drifts wherever the translation locally shrinks/grows the page count.
  const pageAnchors = useRef<(PageAnchor | null)[]>([]);
  // bumped on every goTo() call; a stale call whose locate() resolves after a
  // newer one started must not overwrite the newer call's box/cursor/anchor.
  const navGenRef = useRef(0);

  // ---- load PDFs + segments ------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setAllSegs([]);
    setDocTotal(0);
    setBeforeSizes([]);
    setAfterSizes([]);
    setCurrent(-1);
    cursorRef.current = { before: null, after: null };
    pageAnchors.current = [];
    setEditedIds(new Set());
    setOverrides({});
    textCache.current = { before: new Map(), after: new Map() };

    (async () => {
      try {
        const [segResp, before, after] = await Promise.all([
          fetchJson<{ segments: Segment[] }>(`/api/translate/jobs/${jobId}/segments?category=&limit=100000`),
          loadPdf(`/api/translate/jobs/${jobId}/preview.pdf?side=before`),
          loadPdf(`/api/translate/jobs/${jobId}/preview.pdf?side=after`),
        ]);
        if (cancelled) return;
        pdfBefore.current = before;
        pdfAfter.current = after;

        // Keep each segment's absolute position in the full segment list
        // (1-based) so it correlates with the Segments table's row number.
        const withIndex = segResp.segments.map((s, i) => ({ ...s, docIndex: i + 1 }));
        setDocTotal(segResp.segments.length);

        // Measure each column's own width — the two panes can differ (e.g. one
        // has a visible scrollbar the other doesn't yet), so reusing a single
        // shared width would render one side's pages at the wrong pixel size.
        const beforeColW = (beforeCol.current?.clientWidth ?? 700) - 16;
        const afterColW = (afterCol.current?.clientWidth ?? 700) - 16;
        const [bSizes, aSizes] = await Promise.all([sizesFor(before, beforeColW), sizesFor(after, afterColW)]);
        if (cancelled) return;

        setAllSegs(withIndex);
        setBeforeSizes(bSizes);
        setAfterSizes(aSizes);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load the PDF documents');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      void (pdfBefore.current as unknown as { destroy?: () => void })?.destroy?.();
      void (pdfAfter.current as unknown as { destroy?: () => void })?.destroy?.();
      pdfBefore.current = null;
      pdfAfter.current = null;
    };
  }, [jobId]);

  // ---- lazy page rendering --------------------------------------------------
  useEffect(() => {
    if (loading || error) return;
    const renderDiv = async (div: HTMLElement) => {
      const canvas = div.querySelector('canvas') as HTMLCanvasElement | null;
      if (!canvas || canvas.dataset.rendered === '1' || canvas.dataset.rendering === '1') return;
      const side = div.dataset.side as 'before' | 'after';
      const pageNum = Number(div.dataset.page);
      const pdf = side === 'before' ? pdfBefore.current : pdfAfter.current;
      if (!pdf) return;
      canvas.dataset.rendering = '1';
      try {
        const page = await pdf.getPage(pageNum);
        const base = page.getViewport({ scale: 1 });
        const scale = (div.clientWidth / base.width) * RENDER_QUALITY;
        const vp = page.getViewport({ scale });
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        canvas.width = vp.width;
        canvas.height = vp.height;
        await page.render({ canvas, canvasContext: ctx, viewport: vp }).promise;
        canvas.dataset.rendered = '1';
      } catch { /* blank on failure */ }
      finally { canvas.dataset.rendering = '0'; }
    };
    const observers: IntersectionObserver[] = [];
    for (const col of [beforeCol.current, afterCol.current]) {
      if (!col) continue;
      const io = new IntersectionObserver(
        entries => entries.forEach(e => { if (e.isIntersecting) renderDiv(e.target as HTMLElement); }),
        { root: col, rootMargin: '1000px 0px' },
      );
      col.querySelectorAll('.pdf-page').forEach(el => io.observe(el));
      observers.push(io);
    }
    return () => observers.forEach(o => o.disconnect());
  }, [loading, error, beforeSizes, afterSizes]);

  // ---- linked scrolling (single persistent listener, gated by a ref) -------
  // Maps position via the resolved (before-page, after-page) anchors in
  // pageAnchors instead of a single scrollHeight ratio: a global ratio assumes
  // both documents' content is stretched uniformly, which breaks as soon as
  // the translation locally reflows a page in/out somewhere in the middle —
  // the panes then visibly drift apart around that point. Piecewise-linear
  // interpolation between known anchors (falling back to a page-count ratio
  // where no anchor is known yet, e.g. before the background prefetch below
  // catches up) keeps the two panes aligned near every resolved difference.
  useEffect(() => {
    if (loading) return;
    const b = beforeCol.current, a = afterCol.current;
    if (!b || !a) return;
    let lock = false;

    const sync = (from: HTMLElement, to: HTMLElement, fromSide: 'before' | 'after') => {
      if (lock || suppressSync.current || !linkedRef.current) return;
      lock = true;
      const { page, frac } = pageOfScroll(from, from.scrollTop);
      const fromTotal = (fromSide === 'before' ? pdfBefore.current?.numPages : pdfAfter.current?.numPages) ?? 1;
      const toTotal = (fromSide === 'before' ? pdfAfter.current?.numPages : pdfBefore.current?.numPages) ?? 1;
      const mapped = mapPos(fromSide, page + frac, fromTotal, toTotal, pageAnchors.current);
      const targetPage = Math.max(1, Math.floor(mapped));
      const targetFrac = Math.min(1, Math.max(0, mapped - targetPage));
      to.scrollTop = scrollForPageFrac(to, targetPage, targetFrac);
      requestAnimationFrame(() => { lock = false; });
    };
    const onB = () => sync(b, a, 'before');
    const onA = () => sync(a, b, 'after');
    b.addEventListener('scroll', onB, { passive: true });
    a.addEventListener('scroll', onA, { passive: true });
    return () => { b.removeEventListener('scroll', onB); a.removeEventListener('scroll', onA); };
  }, [loading]);

  const toggleLinked = useCallback(() => {
    setLinked(v => { linkedRef.current = !v; return !v; });
  }, []);

  // "Differences" = segments that were actually translated (the default,
  // matching the before/after PDFs). "All segments" additionally steps
  // through kept/pending/needs-review ones, which show no visible diff but
  // may still be wrongly classified (e.g. a mis-detected language kept as-is).
  const changedSegs = useMemo(
    () => allSegs.filter(s => s.category === 'translated' && s.translated_text != null && norm(s.translated_text) !== norm(s.source_text)),
    [allSegs],
  );
  const diffs = reviewAll ? allSegs : changedSegs;

  // ---- pdf.js text location -------------------------------------------------
  const getPageText = useCallback(async (side: 'before' | 'after', pageNum: number) => {
    const cache = textCache.current[side];
    const hit = cache.get(pageNum);
    if (hit) return hit;
    const pdf = side === 'before' ? pdfBefore.current : pdfAfter.current;
    if (!pdf) return null;
    const page = await pdf.getPage(pageNum);
    const vp = page.getViewport({ scale: 1 });
    const tc = await page.getTextContent();
    const items = tc.items
      .filter((it: unknown) => {
        const o = it as { str?: unknown; transform?: unknown };
        return typeof o.str === 'string' && Array.isArray(o.transform);
      })
      .map((it: unknown) => {
        const item = it as { str: string; transform: number[]; width: number; height: number };
        const x = item.transform[4], y = item.transform[5];
        const w = item.width, h = item.height || 8;
        return {
          s: item.str,
          x0: x / vp.width,
          x1: (x + w) / vp.width,
          y0: 1 - (y + h) / vp.height,
          y1: 1 - y / vp.height,
        };
      })
      .filter((i: { s: string }) => i.s.trim().length > 0);
    // Build a normalized concatenation + a char→item map so a segment's exact
    // contiguous text run can be found (and only those items boxed).
    let concat = '';
    const owner: number[] = [];
    for (let idx = 0; idx < items.length; idx++) {
      const w = matchNorm(items[idx].s);
      if (!w) continue;
      if (concat) { concat += ' '; owner.push(-1); }
      for (let c = 0; c < w.length; c++) { concat += w[c]; owner.push(idx); }
    }
    const pt: PageText = { items, concat, owner };
    cache.set(pageNum, pt);
    return pt;
  }, []);

  // Find the union rect of the words of `text` on some page of `side`. When
  // `cursorHint` is given (forward navigation), search AFTER that position
  // first — same page, then subsequent pages — before falling back to the
  // expected-page scan. That keeps repeated words/values (a table column of
  // identical torque figures, a recurring part name) resolving to the NEXT
  // occurrence in reading order instead of always the first one on the page.
  const locate = useCallback(async (
    side: 'before' | 'after', text: string, expectedPage: number,
    cursorHint: { page: number; pos: number } | null,
  ): Promise<LocateHit | null> => {
    const key = matchNorm(text);
    if (key.length < 2) return null;
    // Cap the search key; short strings match whole. Boxing only the exact
    // matched run (not every word that happens to appear on the page) keeps the
    // box tight and consistent between the two documents.
    const searchKey = key.length > 60 ? key.slice(0, 60) : key;
    const pdf = side === 'before' ? pdfBefore.current : pdfAfter.current;
    if (!pdf) return null;

    const tryPage = async (p: number, fromPos: number): Promise<LocateHit | null> => {
      const pt = await getPageText(side, p);
      if (!pt || !pt.concat) return null;
      const m = matchIn(pt.concat, fromPos, searchKey);
      if (m) {
        const rect = rectFor(pt, m.pos, m.len);
        if (rect) return { page: p, pos: m.pos + m.len, rect: { page: p, ...rect } };
      }
      // A paragraph occasionally straddles a hard page break (common on the
      // original side, whose breaks are fixed, while the rebuilt/translated
      // side may reflow that same paragraph onto a single page) — a
      // single-page search then never finds it and the box silently stays
      // missing on that side only. Bridge this page's tail with the next
      // page's head and retry; box whichever side holds the larger share.
      if (p < pdf.numPages) {
        const ptNext = await getPageText(side, p + 1);
        if (ptNext && ptNext.concat) {
          const nextStart = pt.concat.length + 1;
          const bm = matchIn(`${pt.concat} ${ptNext.concat}`, fromPos, searchKey);
          if (bm && bm.pos < pt.concat.length && bm.pos + bm.len > nextStart) {
            const onThis = pt.concat.length - bm.pos;
            const onNext = bm.pos + bm.len - nextStart;
            const rectThis = onThis > 0 ? rectFor(pt, bm.pos, onThis) : null;
            const rectNext = onNext > 0 ? rectFor(ptNext, 0, onNext) : null;
            if (onThis >= onNext && rectThis) return { page: p, pos: pt.concat.length, rect: { page: p, ...rectThis } };
            if (rectNext) return { page: p + 1, pos: onNext, rect: { page: p + 1, ...rectNext } };
            if (rectThis) return { page: p, pos: pt.concat.length, rect: { page: p, ...rectThis } };
          }
        }
      }
      return null;
    };

    if (cursorHint) {
      const hit = await tryPage(cursorHint.page, cursorHint.pos);
      if (hit) return hit;
      for (let p = cursorHint.page + 1; p <= pdf.numPages; p++) {
        const hit2 = await tryPage(p, 0);
        if (hit2) return hit2;
      }
      // Fell off the end of the document without a match (layouts diverged
      // enough that forward-only search no longer applies) — fall through to
      // the expected-page scan below rather than giving up.
    }

    const order: number[] = [];
    for (let d = 0; d < pdf.numPages; d++) {
      const up = expectedPage + d, down = expectedPage - d;
      if (up >= 1 && up <= pdf.numPages) order.push(up);
      if (d > 0 && down >= 1 && down <= pdf.numPages) order.push(down);
    }
    for (const p of order) {
      const hit = await tryPage(p, 0);
      if (hit) return hit;
    }
    return null;
  }, [getPageText]);

  // Position the active box against the ACTUAL rendered page element (its
  // offsetLeft/offsetTop share the box's offset parent, the scroll column), so
  // there's no page-centering drift or gap-summation error — pixel-exact.
  const placeBox = useCallback((side: 'before' | 'after', boxRef: React.RefObject<HTMLDivElement>, colRef: React.RefObject<HTMLDivElement>, rect: RectFrac | null, doScroll: boolean) => {
    const box = boxRef.current;
    if (!box) return;
    if (!rect || !colRef.current) { box.style.display = 'none'; return; }
    const pageEl = colRef.current.querySelector(`.pdf-page[data-side="${side}"][data-page="${rect.page}"]`) as HTMLElement | null;
    if (!pageEl) { box.style.display = 'none'; return; }
    const pw = pageEl.offsetWidth, ph = pageEl.offsetHeight;
    const left = pageEl.offsetLeft + rect.x0 * pw;
    const top = pageEl.offsetTop + rect.y0 * ph;
    box.style.display = 'block';
    box.style.left = `${left - 2}px`;
    box.style.top = `${top - 2}px`;
    box.style.width = `${(rect.x1 - rect.x0) * pw + 4}px`;
    box.style.height = `${(rect.y1 - rect.y0) * ph + 4}px`;
    if (doScroll) {
      colRef.current.scrollTo({ top: Math.max(0, top - colRef.current.clientHeight / 2), behavior: 'smooth' });
    }
  }, []);

  const goTo = useCallback(async (k: number) => {
    if (k < 0 || k >= diffs.length) return;
    // Only trust the forward cursor when actually stepping forward — a
    // backward jump or an arbitrary "go to N" must re-scan broadly, since the
    // next occurrence in reading order is no longer downstream of it.
    const forward = current >= 0 && k > current;
    const gen = ++navGenRef.current;
    setCurrent(k);
    setEditing(false);
    const seg = diffs[k];
    const expB = Math.max(1, Math.round(((k + 0.5) / diffs.length) * (pdfBefore.current?.numPages ?? 1)));
    const expA = Math.max(1, Math.round(((k + 0.5) / diffs.length) * (pdfAfter.current?.numPages ?? 1)));
    suppressSync.current = true;
    const [rb, ra] = await Promise.all([
      locate('before', seg.source_text, expB, forward ? cursorRef.current.before : null),
      locate('after', overrides[seg.seg_id] ?? seg.translated_text ?? seg.source_text, expA, forward ? cursorRef.current.after : null),
    ]);
    if (navGenRef.current !== gen) return; // superseded by a newer navigation while locate() was in flight
    cursorRef.current = {
      before: rb ? { page: rb.page, pos: rb.pos } : null,
      after: ra ? { page: ra.page, pos: ra.pos } : null,
    };
    pageAnchors.current[k] = { before: rb ? rb.page : expB, after: ra ? ra.page : expA };
    // Navigating always scrolls both panes to their own located rect (each pane
    // is located independently), regardless of the link toggle. If a rect can't
    // be located, fall back to scrolling near the expected page (no box).
    const scrollToPage = (side: 'before' | 'after', colRef: React.RefObject<HTMLDivElement>, p: number) => {
      const el = colRef.current?.querySelector(`.pdf-page[data-side="${side}"][data-page="${p}"]`) as HTMLElement | null;
      if (el && colRef.current) colRef.current.scrollTo({ top: Math.max(0, el.offsetTop - 40), behavior: 'smooth' });
    };
    if (ra) placeBox('after', afterBox, afterCol, ra.rect, true);
    else { placeBox('after', afterBox, afterCol, null, false); scrollToPage('after', afterCol, expA); }
    if (rb) placeBox('before', beforeBox, beforeCol, rb.rect, true);
    else { placeBox('before', beforeBox, beforeCol, null, false); scrollToPage('before', beforeCol, expB); }
    window.setTimeout(() => { suppressSync.current = false; }, 600);
  }, [diffs, locate, overrides, placeBox, current]);

  // An open edit whose draft hasn't been saved would otherwise be silently
  // discarded by goTo()'s unconditional setEditing(false) — confirm first.
  const hasUnsavedEdit = useCallback(() => {
    if (!editing) return false;
    const s = diffs[current];
    const shown = s ? (overrides[s.seg_id] ?? s.translated_text ?? s.source_text) : '';
    return draft !== shown;
  }, [editing, draft, diffs, current, overrides]);

  const guardedGoTo = useCallback((k: number) => {
    if (hasUnsavedEdit() && !window.confirm('Discard the unsaved edit to this translation?')) return;
    goTo(k);
  }, [hasUnsavedEdit, goTo]);

  // Keeps the reviewer's place across the toggle instead of always snapping
  // back to segment 1 — losing your spot mid-review just to check one
  // mis-detected segment elsewhere was the whole complaint. Resolved by the
  // effect below once the target list (diffs) has actually switched.
  const pendingTargetSegId = useRef<string | null>(null);

  const toggleReviewAll = useCallback(() => {
    if (hasUnsavedEdit() && !window.confirm('Discard the unsaved edit to this translation?')) return;
    pendingTargetSegId.current = current >= 0 ? diffs[current].seg_id : null;
    cursorRef.current = { before: null, after: null };
    pageAnchors.current = [];
    setCurrent(-1);
    setReviewAll(v => !v);
  }, [hasUnsavedEdit, current, diffs]);

  // Auto-select once loaded, and re-resolve position once the toggle switches
  // the active list: land back on the same segment if it's in the new list,
  // otherwise the closest one by document position, otherwise the first.
  useEffect(() => {
    if (loading || error || diffs.length === 0 || current !== -1) return;
    const targetId = pendingTargetSegId.current;
    pendingTargetSegId.current = null;
    if (!targetId) { goTo(0); return; }
    let idx = diffs.findIndex(s => s.seg_id === targetId);
    if (idx === -1) {
      const targetDocIndex = allSegs.find(s => s.seg_id === targetId)?.docIndex;
      idx = targetDocIndex == null ? 0 : diffs.reduce(
        (best, s, i) => (Math.abs(s.docIndex - targetDocIndex) < Math.abs(diffs[best].docIndex - targetDocIndex) ? i : best),
        0,
      );
    }
    goTo(idx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, error, diffs, reviewAll, allSegs]);

  // External jump from the Segments panel's "View in document" button.
  // `token` changes on every click (even re-clicking the same segment) so
  // this always fires. A kept/pending segment is invisible to the
  // diffs-only view, so reviewAll is forced on first — the effect re-runs
  // once that flip is reflected in `diffs`, then lands exactly on it
  // (unlike the toggle-preserve logic above, this never falls back to
  // "nearest" — the reviewer asked for this exact segment).
  const lastFocusToken = useRef<number | null>(null);
  useEffect(() => {
    if (!focusRequest || focusRequest.token === lastFocusToken.current || loading || error) return;
    if (hasUnsavedEdit() && !window.confirm('Discard the unsaved edit to this translation?')) {
      lastFocusToken.current = focusRequest.token;
      return;
    }
    if (!reviewAll) { setReviewAll(true); return; }
    const idx = diffs.findIndex(s => s.seg_id === focusRequest.segId);
    lastFocusToken.current = focusRequest.token;
    if (idx >= 0) goTo(idx);
  }, [focusRequest, loading, error, reviewAll, diffs, hasUnsavedEdit, goTo]);

  // Background page-anchor prefetch: resolves every difference's page on both
  // sides (not just the ones the user has navigated to) so linked-scroll has
  // real anchors across the whole document from early on, instead of only
  // around wherever the user has clicked so far.
  useEffect(() => {
    if (loading || error || diffs.length === 0) return;
    let cancelled = false;
    (async () => {
      for (let k = 0; k < diffs.length; k++) {
        if (cancelled) return;
        if (pageAnchors.current[k]) continue;
        const seg = diffs[k];
        const expB = Math.max(1, Math.round(((k + 0.5) / diffs.length) * (pdfBefore.current?.numPages ?? 1)));
        const expA = Math.max(1, Math.round(((k + 0.5) / diffs.length) * (pdfAfter.current?.numPages ?? 1)));
        const [rb, ra] = await Promise.all([
          locate('before', seg.source_text, expB, null),
          locate('after', overrides[seg.seg_id] ?? seg.translated_text ?? seg.source_text, expA, null),
        ]);
        if (cancelled) return;
        pageAnchors.current[k] = { before: rb ? rb.page : expB, after: ra ? ra.page : expA };
      }
    })();
    return () => { cancelled = true; };
  }, [loading, error, diffs, locate, overrides]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key === 'n' || e.key === 'N') { e.preventDefault(); guardedGoTo(current + 1); }
      if (e.key === 'p' || e.key === 'P') { e.preventDefault(); guardedGoTo(current - 1); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [current, guardedGoTo]);

  // ---- inline edit ----------------------------------------------------------
  const seg = current >= 0 ? diffs[current] : null;
  const shownTranslation = seg ? (overrides[seg.seg_id] ?? seg.translated_text ?? seg.source_text) : '';

  const startEdit = useCallback(() => {
    if (!seg) return;
    setDraft(shownTranslation);
    setEditing(true);
  }, [seg, shownTranslation]);

  const saveEdit = useCallback(async () => {
    if (!seg) return;
    setSaving(true);
    try {
      await fetchJson(`/api/translate/jobs/${jobId}/segments/${encodeURIComponent(seg.seg_id)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ translated_text: draft }),
      });
      setOverrides(prev => ({ ...prev, [seg.seg_id]: draft }));
      setEditedIds(prev => new Set(prev).add(seg.seg_id));
      setEditing(false);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to save');
    } finally {
      setSaving(false);
    }
  }, [seg, jobId, draft]);

  // For a segment wrongly kept as-is (e.g. a false "already the target
  // language" detection) — say "this needs translating" instead of having
  // to know/type the correct wording yourself.
  const [retranslating, setRetranslating] = useState(false);
  const retranslateCurrent = useCallback(async () => {
    if (!seg) return;
    setRetranslating(true);
    try {
      const r = await fetchJson<{ translated_text: string }>(
        `/api/translate/jobs/${jobId}/segments/${encodeURIComponent(seg.seg_id)}/retranslate`,
        { method: 'POST' },
      );
      setOverrides(prev => ({ ...prev, [seg.seg_id]: r.translated_text }));
      setEditedIds(prev => new Set(prev).add(seg.seg_id));
      toast.success('Retranslated — click "Rebuild" to apply it to the document.');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Retranslation failed');
    } finally {
      setRetranslating(false);
    }
  }, [seg, jobId]);

  // Correcting the language to the source language means "this needs
  // translating" — go straight to the LLM; anything else is just a label fix.
  const changeLanguage = useCallback(async (lang: string) => {
    if (!seg) return;
    if (lang === sourceLang) {
      await retranslateCurrent();
      return;
    }
    try {
      await fetchJson(`/api/translate/jobs/${jobId}/segments/${encodeURIComponent(seg.seg_id)}/language`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detected_lang: lang }),
      });
      setAllSegs(prev => prev.map(s => (s.seg_id === seg.seg_id ? { ...s, detected_lang: lang } : s)));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to update language');
    }
  }, [seg, jobId, sourceLang, retranslateCurrent]);

  // editedIds only tracks "was ever saved once" and never shrinks (e.g. if the
  // user edits, saves, then edits back to the original wording) — count/label
  // off of whether the current override still actually differs from what was
  // originally loaded, so both stay accurate after a revert.
  const isActuallyEdited = useCallback((s: Segment) => {
    const ov = overrides[s.seg_id];
    return ov !== undefined && ov !== (s.translated_text ?? s.source_text);
  }, [overrides]);
  // Counted over every segment (not just the currently active list) so
  // toggling "All segments" doesn't hide edits made in the other mode.
  const dirty = useMemo(() => allSegs.reduce((n, s) => n + (editedIds.has(s.seg_id) && isActuallyEdited(s) ? 1 : 0), 0), [allSegs, editedIds, isActuallyEdited]);
  const btn = 'inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs disabled:opacity-40';

  const langCode = seg?.detected_lang && seg.detected_lang !== '??' ? seg.detected_lang : null;
  const status = seg
    ? (seg.flagged ? { label: 'Flagged', fg: '#b45309', bg: 'rgba(245,158,11,0.14)' }
      : isActuallyEdited(seg) ? { label: 'Edited', fg: 'var(--color-accent-primary)', bg: 'rgba(0,85,164,0.10)' }
      : CATEGORY_STATUS[seg.category] ?? CATEGORY_STATUS.pending)
    : null;

  const renderColumn = (side: 'before' | 'after', sizes: { w: number; h: number }[], colRef: React.RefObject<HTMLDivElement>, boxRef: React.RefObject<HTMLDivElement>) => (
    <div className="rounded-lg overflow-hidden flex flex-col" style={{ border: '1px solid var(--color-border)' }}>
      <div className="flex items-center px-3 py-1.5" style={{ background: 'var(--color-bg-secondary)' }}>
        <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
          {side === 'before' ? 'Original' : 'Translated'} · {(side === 'before' ? sourceLang : targetLang).toUpperCase()}
        </span>
      </div>
      <div ref={colRef} className="relative overflow-auto flex flex-col items-center gap-3 py-3"
           style={{ height: 'calc(100vh - 220px)', minHeight: '480px', background: '#525659' }}>
        {sizes.map((sz, i) => (
          <div key={i + 1} className="pdf-page relative flex-shrink-0" data-side={side} data-page={i + 1}
               style={{ width: sz.w, height: sz.h, background: '#fff', boxShadow: '0 1px 6px rgba(0,0,0,0.4)' }}>
            <canvas className="block w-full h-full" />
          </div>
        ))}
        {/* single active box, absolutely positioned within the scroll content */}
        <div ref={boxRef} className="pdf-box" style={{ position: 'absolute', display: 'none', left: 0, top: 0 }} />
      </div>
    </div>
  );

  return (
    <div>
      <style>{`
        .pdf-box { border: 3px solid #f59e0b; background: rgba(245,158,11,0.14); border-radius: 3px; box-shadow: 0 0 0 3px rgba(245,158,11,0.2); pointer-events: none; z-index: 5; }
      `}</style>

      {/* Enlarged difference bar */}
      <div className="sticky top-0 z-10 mb-2 rounded-lg" style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}>
        <div className="flex flex-wrap items-center gap-2 px-3 py-2" style={{ borderBottom: seg ? '1px solid var(--color-border)' : 'none' }}>
          <div className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <button className={btn} style={{ border: '1px solid var(--color-border)' }}
                    onClick={() => guardedGoTo(current - 1)} disabled={current <= 0} title={`Previous ${reviewAll ? 'segment' : 'difference'} (P)`}>
              <ChevronUp className="h-3.5 w-3.5" />
            </button>
            <span className="whitespace-nowrap font-medium" style={{ color: 'var(--color-text-primary)' }}>{reviewAll ? 'Segment' : 'Difference'}</span>
            <input
              type="number"
              min={1}
              max={diffs.length}
              value={jumpValue !== '' ? jumpValue : (current >= 0 ? current + 1 : '')}
              onChange={e => setJumpValue(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  const v = parseInt(jumpValue, 10);
                  if (!Number.isNaN(v) && v >= 1 && v <= diffs.length) guardedGoTo(v - 1);
                  else if (jumpValue !== '') toast.error(`Enter a number between 1 and ${diffs.length}`);
                  setJumpValue('');
                }
              }}
              onBlur={() => {
                const v = parseInt(jumpValue, 10);
                if (!Number.isNaN(v) && v >= 1 && v <= diffs.length) guardedGoTo(v - 1);
                else if (jumpValue !== '') toast.error(`Enter a number between 1 and ${diffs.length}`);
                setJumpValue('');
              }}
              className="w-14 text-xs text-center rounded-md px-1 py-1"
              style={{ background: 'var(--color-bg-primary)', border: '1px solid var(--color-border)', color: 'var(--color-text-primary)' }}
            />
            <span className="whitespace-nowrap">/ {diffs.length || (loading ? '…' : 0)}</span>
            <button className={btn} style={{ border: '1px solid var(--color-border)' }}
                    onClick={() => guardedGoTo(current < 0 ? 0 : current + 1)}
                    disabled={diffs.length === 0 || current >= diffs.length - 1} title={`Next ${reviewAll ? 'segment' : 'difference'} (N)`}>
              <ChevronDown className="h-3.5 w-3.5" />
            </button>
            {seg && !reviewAll && (
              <span className="whitespace-nowrap ml-1" style={{ color: 'var(--color-text-muted)' }} title="Row number in the Segments table (All)">
                · Segment {seg.docIndex}/{docTotal}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2 ml-auto">
            <button onClick={toggleReviewAll} className={btn}
                    style={{ border: '1px solid var(--color-border)', color: reviewAll ? 'var(--color-accent-primary)' : 'var(--color-text-muted)' }}
                    title={reviewAll
                      ? 'Stepping through every segment — click to show only translated differences'
                      : 'Stepping through translated differences only — click to include every segment (catches wrongly kept/mis-detected ones)'}>
              <Layers className="h-3.5 w-3.5" /> {reviewAll ? 'All segments' : 'Differences only'}
            </button>
            <button onClick={toggleLinked} className={btn}
                    style={{ border: '1px solid var(--color-border)', color: linked ? 'var(--color-accent-primary)' : 'var(--color-text-muted)' }}
                    title={linked ? 'Scroll is linked — click to unlink the two panes' : 'Scroll is independent — click to re-link'}>
              {linked ? <Link2 className="h-3.5 w-3.5" /> : <Link2Off className="h-3.5 w-3.5" />} Scroll {linked ? 'linked' : 'independent'}
            </button>
            <label className="flex items-center gap-1.5 text-xs cursor-pointer whitespace-nowrap" style={{ color: 'var(--color-text-muted)' }}>
              <input
                type="checkbox"
                checked={includeReviewComments}
                onChange={e => onIncludeReviewCommentsChange(e.target.checked)}
              />
              Include review comments
            </label>
            <button onClick={onRebuild} disabled={rebuilding}
                    className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md disabled:opacity-60"
                    style={{ background: dirty > 0 ? 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' : 'var(--color-bg-primary)',
                      color: dirty > 0 ? '#fff' : 'var(--color-text-muted)', border: dirty > 0 ? 'none' : '1px solid var(--color-border)' }}
                    title={dirty > 0 ? `${dirty} edit(s) not yet applied to the .docx` : 'Rebuild the document'}>
              {rebuilding ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              {rebuilding ? 'Rebuilding…' : dirty > 0 ? `Rebuild (${dirty})` : 'Rebuild'}
            </button>
          </div>
        </div>

        {/* Current difference detail + inline edit */}
        {seg && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 px-3 py-2.5">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-wide mb-1 flex items-center gap-1.5" style={{ color: 'var(--color-text-muted)' }}>
                Source
                {langCode && (
                  <span className="inline-flex items-center gap-1">
                    {ISO_TO_FLAG[langCode] && <Flag code={ISO_TO_FLAG[langCode]} />}
                    <span className="uppercase">{langCode}</span>
                  </span>
                )}
                <select
                  value=""
                  onChange={e => { if (e.target.value) changeLanguage(e.target.value); }}
                  className="text-[10px] normal-case rounded px-0.5 py-0.5"
                  style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)', color: 'var(--color-text-muted)' }}
                  title="Correct the detected language"
                >
                  <option value="">✎</option>
                  {LANG_OPTIONS.map(l => <option key={l} value={l}>{l.toUpperCase()}</option>)}
                </select>
              </div>
              <p className="text-sm whitespace-pre-wrap" style={{ color: 'var(--color-text-primary)' }}>{seg.source_text}</p>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <div className="text-[11px] font-semibold uppercase tracking-wide flex items-center gap-2" style={{ color: 'var(--color-text-muted)' }}>
                  Translation · {targetLang.toUpperCase()}
                  {status && <span className="text-[10px] px-1.5 py-0.5 rounded normal-case font-semibold" style={{ background: status.bg, color: status.fg }} title={seg.conflict_detail ?? undefined}>{status.label}</span>}
                </div>
                {!editing && (
                  <div className="flex items-center gap-1.5">
                    {['kept_other_language', 'needs_review', 'pending'].includes(seg.category) && (
                      <button
                        className={btn}
                        style={{ border: '1px solid var(--color-border)', color: 'var(--color-accent-primary)' }}
                        onClick={retranslateCurrent}
                        disabled={retranslating}
                        title="Wrong language detected — translate this segment"
                      >
                        {retranslating ? <Loader2 className="h-3 w-3 animate-spin" /> : <Languages className="h-3 w-3" />} Translate
                      </button>
                    )}
                    <button className={btn} style={{ border: '1px solid var(--color-border)', color: 'var(--color-text-muted)' }} onClick={startEdit} title="Edit this translation">
                      <Pencil className="h-3 w-3" /> Edit
                    </button>
                  </div>
                )}
              </div>
              {editing ? (
                <div>
                  <textarea
                    autoFocus
                    className="w-full text-sm rounded-md p-2"
                    style={{ background: 'var(--color-bg-primary)', border: '1px solid var(--color-accent-primary)', color: 'var(--color-text-primary)' }}
                    rows={Math.min(8, Math.max(2, Math.ceil(draft.length / 70)))}
                    value={draft}
                    onChange={e => setDraft(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); saveEdit(); }
                      if (e.key === 'Escape') { e.preventDefault(); setEditing(false); }
                    }}
                  />
                  <div className="flex items-center gap-2 mt-1.5">
                    <button onClick={saveEdit} disabled={saving} className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-md text-white disabled:opacity-60"
                            style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}>
                      {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />} Save
                    </button>
                    <button onClick={() => setEditing(false)} disabled={saving} className={btn} style={{ border: '1px solid var(--color-border)', color: 'var(--color-text-muted)' }}>
                      <X className="h-3 w-3" /> Cancel
                    </button>
                    <span className="text-[10px]" style={{ color: 'var(--color-text-muted)' }}>Save, then Rebuild to apply to the .docx</span>
                  </div>
                </div>
              ) : (
                <p className="text-sm whitespace-pre-wrap cursor-text rounded px-1 -mx-1 hover:bg-[var(--color-bg-primary)]" style={{ color: 'var(--color-text-primary)' }} onClick={startEdit}>
                  {shownTranslation}
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {error ? (
        <p className="text-sm py-8 text-center" style={{ color: 'var(--color-text-muted)' }}>{error}</p>
      ) : (
        <div className="relative">
          {loading && (
            <div className="absolute inset-0 z-20 flex items-center justify-center" style={{ background: 'rgba(255,255,255,0.55)' }}>
              <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--color-text-muted)' }}>
                <Loader2 className="h-5 w-5 animate-spin" /> Rendering documents…
              </div>
            </div>
          )}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {renderColumn('before', beforeSizes, beforeCol, beforeBox)}
            {renderColumn('after', afterSizes, afterCol, afterBox)}
          </div>
        </div>
      )}
    </div>
  );
}
