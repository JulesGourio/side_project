import { describe, expect, it } from 'vitest';
import {
  norm, matchNorm, matchIn, rectFor, mapPos, pageOfScroll, scrollForPageFrac,
  type PageText, type PageAnchor,
} from './textMatch';

describe('norm', () => {
  it('collapses whitespace, trims, and lowercases', () => {
    expect(norm('  Monter   la  Vis  \n')).toBe('monter la vis');
  });
});

describe('matchNorm', () => {
  it('strips punctuation but keeps letters and digits', () => {
    expect(matchNorm("N.m d'assemblage — 12,5")).toBe('n m d assemblage 12 5');
  });

  it('keeps accented letters (punctuation-insensitive, not accent-insensitive)', () => {
    expect(matchNorm('Étanchéité')).toBe('étanchéité');
  });

  it('collapses repeated separators into a single space', () => {
    expect(matchNorm('A///B  --  C')).toBe('a b c');
  });
});

describe('matchIn', () => {
  it('finds the search key at or after fromPos', () => {
    const concat = 'monter la vis puis monter la rondelle';
    expect(matchIn(concat, 0, 'monter la vis')).toEqual({ pos: 0, len: 13 });
    expect(matchIn(concat, 5, 'monter la')).toEqual({ pos: 19, len: 9 });
  });

  it('returns null when the key is not found', () => {
    expect(matchIn('abc def', 0, 'xyz')).toBeNull();
  });

  it('falls back to a 24-char prefix when the full key (>24 chars) is not found verbatim', () => {
    const searchKey = 'a'.repeat(30);
    const concat = `prefix ${'a'.repeat(24)} suffix-diverges-here`;
    expect(matchIn(concat, 0, searchKey)).toEqual({ pos: 7, len: 24 });
  });

  it('does not use the prefix fallback for keys of 24 chars or fewer', () => {
    expect(matchIn('short text here', 0, 'not present at all')).toBeNull();
  });
});

describe('rectFor', () => {
  const pt: PageText = {
    items: [
      { s: 'Monter', x0: 0.1, y0: 0.2, x1: 0.2, y1: 0.25 },
      { s: 'la', x0: 0.21, y0: 0.2, x1: 0.24, y1: 0.25 },
      { s: 'vis', x0: 0.25, y0: 0.19, x1: 0.3, y1: 0.26 },
    ],
    // "monter la vis" concatenated with gaps (-1) between items
    concat: 'monter la vis',
    owner: [0, 0, 0, 0, 0, 0, -1, 1, 1, -1, 2, 2, 2],
  };

  it('returns the union rect of every item covered by [pos, pos+len)', () => {
    const rect = rectFor(pt, 0, pt.concat.length);
    expect(rect).toEqual({ x0: 0.1, y0: 0.19, x1: 0.3, y1: 0.26 });
  });

  it('covers only the items touched by a partial range', () => {
    const rect = rectFor(pt, 0, 6); // "monter" only
    expect(rect).toEqual({ x0: 0.1, y0: 0.2, x1: 0.2, y1: 0.25 });
  });

  it('returns null when the range only touches gap characters', () => {
    expect(rectFor(pt, 6, 1)).toBeNull(); // the single space between "monter" and "la"
  });
});

describe('mapPos', () => {
  it('interpolates between the synthetic (1,1) and (total,total) endpoints when there are no real anchors', () => {
    expect(mapPos('before', 1, 10, 20, [])).toBeCloseTo(1, 5);
    expect(mapPos('before', 10, 10, 20, [])).toBeCloseTo(20, 5);
    expect(mapPos('before', 5.5, 10, 20, [])).toBeCloseTo(10.5, 5); // midpoint (t=0.5)
  });

  it('is the identity map when both documents have the same page count', () => {
    expect(mapPos('before', 7, 10, 10, [])).toBeCloseTo(7, 5);
  });

  it('interpolates piecewise-linearly through a real anchor', () => {
    const anchors: (PageAnchor | null)[] = [{ before: 4, after: 6 }];
    // Below the anchor: interpolate between (1,1) and (4,6).
    expect(mapPos('before', 4, 10, 15, anchors)).toBeCloseTo(6, 5);
    // Above the anchor: interpolate between (4,6) and (10,15).
    expect(mapPos('before', 10, 10, 15, anchors)).toBeCloseTo(15, 5);
    expect(mapPos('before', 7, 10, 15, anchors)).toBeCloseTo(10.5, 5);
  });

  it('maps in the reverse direction using the anchor\'s other side', () => {
    const anchors: (PageAnchor | null)[] = [{ before: 4, after: 6 }];
    expect(mapPos('after', 6, 15, 10, anchors)).toBeCloseTo(4, 5);
  });

  it('ignores null anchor slots', () => {
    const anchors: (PageAnchor | null)[] = [null, { before: 5, after: 5 }, null];
    expect(mapPos('before', 5, 10, 10, anchors)).toBeCloseTo(5, 5);
  });
});

// Minimal fakes for the DOM surface pageOfScroll/scrollForPageFrac touch —
// no jsdom needed (this suite runs under environment: 'node').
function fakePage(offsetTop: number, offsetHeight: number) {
  return { offsetTop, offsetHeight } as unknown as HTMLElement;
}

function fakeCol(pages: HTMLElement[]) {
  return {
    querySelectorAll: () => pages,
    querySelector: (sel: string) => {
      const m = /data-page="(\d+)"/.exec(sel);
      const idx = m ? Number(m[1]) - 1 : -1;
      return pages[idx] ?? null;
    },
  } as unknown as HTMLElement;
}

describe('pageOfScroll', () => {
  it('finds the page containing the scroll offset and the fraction within it', () => {
    const pages = [fakePage(0, 100), fakePage(100, 100), fakePage(200, 100)];
    const col = fakeCol(pages);
    expect(pageOfScroll(col, 0)).toEqual({ page: 1, frac: 0 });
    expect(pageOfScroll(col, 150)).toEqual({ page: 2, frac: 0.5 });
  });

  it('clamps to the last page when the offset is past the end', () => {
    const pages = [fakePage(0, 100), fakePage(100, 50)];
    const col = fakeCol(pages);
    expect(pageOfScroll(col, 500)).toEqual({ page: 2, frac: 1 });
  });
});

describe('scrollForPageFrac', () => {
  it('computes the scrollTop for a given page + fraction', () => {
    const pages = [fakePage(0, 100), fakePage(100, 100)];
    const col = fakeCol(pages);
    expect(scrollForPageFrac(col, 2, 0.5)).toBe(150);
  });

  it('returns 0 when the page element cannot be found', () => {
    const col = fakeCol([fakePage(0, 100)]);
    expect(scrollForPageFrac(col, 99, 0.5)).toBe(0);
  });
});
