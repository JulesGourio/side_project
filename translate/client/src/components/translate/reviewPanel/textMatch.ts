import type * as pdfjsLib from 'pdfjs-dist';

// Pure, framework-free helpers extracted from ReviewPanel.tsx: PDF text
// matching (locating a segment's text inside pdf.js-extracted page text) and
// linked-scroll position mapping between the two document panes. No React
// state/refs here — callers pass whatever they'd otherwise have read from a
// ref, which is what makes this module unit-testable without a DOM.

export interface RectFrac { page: number; x0: number; y0: number; x1: number; y1: number }

export interface PageTextItem { s: string; x0: number; y0: number; x1: number; y1: number }

export interface PageText {
  items: PageTextItem[];
  // normalized concatenation of all item texts + a parallel array mapping each
  // concatenated character back to its source item index (-1 for inserted gaps)
  concat: string;
  owner: number[];
}

export interface PageAnchor { before: number; after: number }

export const norm = (t: string) => t.replace(/\s+/g, ' ').trim().toLowerCase();

// Punctuation-insensitive normalization for locating text in the PDF: keeps
// only unicode letters/digits, so apostrophes, commas, dashes, "N.m" vs "N m"
// etc. don't defeat the match between segment text and pdf.js-extracted text.
export const matchNorm = (t: string) =>
  t.toLowerCase().normalize('NFC').replace(/[^\p{L}\p{N}]+/gu, ' ').replace(/\s+/g, ' ').trim();

// Find `searchKey` (or its 24-char prefix fallback) at/after `fromPos` in
// `concat`; returns null if not found.
export function matchIn(concat: string, fromPos: number, searchKey: string): { pos: number; len: number } | null {
  let pos = concat.indexOf(searchKey, fromPos);
  let len = searchKey.length;
  if (pos < 0 && searchKey.length > 24) {
    const pref = searchKey.slice(0, 24);
    pos = concat.indexOf(pref, fromPos);
    len = pref.length;
  }
  return pos < 0 ? null : { pos, len };
}

export function rectFor(pt: PageText, pos: number, len: number): { x0: number; y0: number; x1: number; y1: number } | null {
  const covered = new Set<number>();
  for (let i = pos; i < pos + len && i < pt.owner.length; i++) { const o = pt.owner[i]; if (o >= 0) covered.add(o); }
  if (covered.size === 0) return null;
  const its = [...covered].map(i => pt.items[i]);
  return {
    x0: Math.min(...its.map(i => i.x0)),
    y0: Math.min(...its.map(i => i.y0)),
    x1: Math.max(...its.map(i => i.x1)),
    y1: Math.max(...its.map(i => i.y1)),
  };
}

// Piecewise-linear map of a continuous (page + frac) position from one side
// to the other, through real anchors plus synthetic identity anchors at both
// document ends (degrades to a plain page-count ratio wherever no real
// anchor has been resolved yet).
export function mapPos(
  fromSide: 'before' | 'after',
  fromPos: number,
  fromTotal: number,
  toTotal: number,
  anchors: (PageAnchor | null)[],
): number {
  const real = anchors
    .filter((x): x is PageAnchor => x !== null)
    .map(x => (fromSide === 'before' ? { from: x.before, to: x.after } : { from: x.after, to: x.before }));
  const pts = [{ from: 1, to: 1 }, ...real, { from: fromTotal, to: toTotal }].sort((x, y) => x.from - y.from);
  const dedup: { from: number; to: number }[] = [];
  for (const p of pts) {
    if (dedup.length && dedup[dedup.length - 1].from === p.from) dedup[dedup.length - 1] = p;
    else dedup.push(p);
  }
  let lo = dedup[0], hi = dedup[dedup.length - 1];
  for (let i = 0; i < dedup.length - 1; i++) {
    if (fromPos >= dedup[i].from && fromPos <= dedup[i + 1].from) { lo = dedup[i]; hi = dedup[i + 1]; break; }
  }
  const span = hi.from - lo.from;
  const t = span > 0 ? (fromPos - lo.from) / span : 0;
  return lo.to + t * (hi.to - lo.to);
}

export function pageOfScroll(col: HTMLElement, top: number): { page: number; frac: number } {
  const pages = Array.from(col.querySelectorAll('.pdf-page')) as HTMLElement[];
  for (let i = 0; i < pages.length; i++) {
    const el = pages[i];
    const bottom = el.offsetTop + el.offsetHeight;
    if (top < bottom || i === pages.length - 1) {
      const frac = Math.min(1, Math.max(0, (top - el.offsetTop) / Math.max(1, el.offsetHeight)));
      return { page: i + 1, frac };
    }
  }
  return { page: 1, frac: 0 };
}

export function scrollForPageFrac(col: HTMLElement, page: number, frac: number): number {
  const el = col.querySelector(`.pdf-page[data-page="${Math.max(1, Math.round(page))}"]`) as HTMLElement | null;
  return el ? el.offsetTop + frac * el.offsetHeight : 0;
}

export async function sizesFor(pdf: pdfjsLib.PDFDocumentProxy, colW: number): Promise<{ w: number; h: number }[]> {
  const out: { w: number; h: number }[] = [];
  for (let n = 1; n <= pdf.numPages; n++) {
    const page = await pdf.getPage(n);
    const vp = page.getViewport({ scale: 1 });
    out.push({ w: colW, h: (colW * vp.height) / vp.width });
  }
  return out;
}
