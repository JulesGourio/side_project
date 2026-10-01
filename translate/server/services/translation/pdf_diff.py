"""Pixel-level page diff for the Translate tab's preview — the closest thing
to the original Translator POC's compare.py "Difference" toggle (grayscale
original with red highlights over pixel-level changes).
"""
from __future__ import annotations

import io

from PIL import Image, ImageChops, ImageOps


def _import_fitz():
    """PyMuPDF version-compat import (module was renamed pymupdf -> fitz alias
    in >=1.24)."""
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
        return fitz
    except ImportError:
        pass
    try:
        import fitz  # PyMuPDF < 1.24
        return fitz
    except ImportError as e:
        raise ImportError(
            f'PyMuPDF could not be imported ({e}). '
            'Ensure "pymupdf>=1.25.0" is in pyproject.toml and the app has been redeployed.'
        ) from e


_DIFF_THRESHOLD = 25  # grayscale delta above which a pixel counts as "changed"
_HIGHLIGHT_RGB = (220, 38, 38)


def _render_page_from_doc(fitz_module, doc, page_num: int, zoom: float) -> Image.Image | None:
    """1-indexed; None if page_num is out of range for this already-open doc."""
    if page_num < 1 or page_num > doc.page_count:
        return None
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=fitz_module.Matrix(zoom, zoom))
    return Image.frombytes('RGB', (pix.width, pix.height), pix.samples)


def _render_page(fitz_module, pdf_bytes: bytes, page_num: int, zoom: float):
    """Returns (PIL.Image | None, total_page_count). None if page_num is out
    of range (1-indexed)."""
    doc = fitz_module.open(stream=pdf_bytes, filetype='pdf')
    try:
        return _render_page_from_doc(fitz_module, doc, page_num, zoom), doc.page_count
    finally:
        doc.close()


def _pad_to_match(a: Image.Image, b: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Pad both images (white background) to their shared max dimensions,
    rather than resize/distort — page sizes should match (rebuild preserves
    layout) but aren't guaranteed to in every edge case."""
    w, h = max(a.width, b.width), max(a.height, b.height)

    def _pad(img: Image.Image) -> Image.Image:
        if img.size == (w, h):
            return img
        canvas = Image.new('RGB', (w, h), 'white')
        canvas.paste(img, (0, 0))
        return canvas

    return _pad(a), _pad(b)


def _diff_mask(before_img: Image.Image, after_img: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Returns (padded_before, padded_after, mask) — mask is 'L' mode,
    255=changed/0=unchanged, thresholded grayscale delta."""
    before_img, after_img = _pad_to_match(before_img, after_img)
    gray_before = ImageOps.grayscale(before_img)
    gray_after = ImageOps.grayscale(after_img)
    diff = ImageChops.difference(gray_before, gray_after)
    mask = diff.point(lambda p: 255 if p > _DIFF_THRESHOLD else 0)
    return before_img, after_img, mask


def _composite_diff(before_img: Image.Image, after_img: Image.Image) -> Image.Image:
    """Grayscale BEFORE with red pixel-level highlights wherever AFTER differs."""
    before_img, _after_img, mask = _diff_mask(before_img, after_img)
    out = ImageOps.grayscale(before_img).convert('RGB')
    highlight = Image.new('RGB', out.size, _HIGHLIGHT_RGB)
    return Image.composite(highlight, out, mask)


_CHANGE_CELL_PX = 24  # grid cell size for connected-component grouping
_CHANGE_CELL_MIN_FRAC = 0.06  # fraction of sampled cell pixels that must differ to count the cell as "changed"


def _detect_change_regions(mask: Image.Image) -> list[tuple[float, float, float, float]]:
    """Groups changed pixels into bounding boxes, resolution-independent
    (fractions of image width/height, 0..1) so the frontend can scale to
    whatever size it renders the image at.

    No scipy/opencv dependency: downsamples to a coarse grid (a full page is
    only ~dozens of cells across) and does a plain BFS connected-component
    pass over that grid — cheap and accurate enough for "jump to this
    change," which doesn't need pixel-perfect region edges.
    """
    w, h = mask.size
    cols = max(1, w // _CHANGE_CELL_PX)
    rows = max(1, h // _CHANGE_CELL_PX)
    px = mask.load()
    changed = [[False] * cols for _ in range(rows)]
    for ry in range(rows):
        y0, y1 = ry * _CHANGE_CELL_PX, min((ry + 1) * _CHANGE_CELL_PX, h)
        for rx in range(cols):
            x0, x1 = rx * _CHANGE_CELL_PX, min((rx + 1) * _CHANGE_CELL_PX, w)
            sampled = 0
            hit = 0
            for yy in range(y0, y1, 2):
                for xx in range(x0, x1, 2):
                    sampled += 1
                    if px[xx, yy] > 0:
                        hit += 1
            if sampled and hit / sampled > _CHANGE_CELL_MIN_FRAC:
                changed[ry][rx] = True

    visited = [[False] * cols for _ in range(rows)]
    regions: list[tuple[float, float, float, float]] = []
    for ry in range(rows):
        for rx in range(cols):
            if not changed[ry][rx] or visited[ry][rx]:
                continue
            visited[ry][rx] = True
            stack = [(ry, rx)]
            min_r = max_r = ry
            min_c = max_c = rx
            while stack:
                cy, cx = stack.pop()
                min_r, max_r = min(min_r, cy), max(max_r, cy)
                min_c, max_c = min(min_c, cx), max(max_c, cx)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < rows and 0 <= nx < cols and changed[ny][nx] and not visited[ny][nx]:
                        visited[ny][nx] = True
                        stack.append((ny, nx))
            regions.append((min_c / cols, min_r / rows, (max_c + 1) / cols, (max_r + 1) / rows))

    regions.sort(key=lambda b: (b[1], b[0]))  # reading order: top to bottom, left to right
    return regions


def get_all_page_changes(
    pdf_before: bytes, pdf_after: bytes, zoom: float = 1.5,
) -> list[dict]:
    """Change regions across every page, for cross-page "next/prev change"
    navigation. Returns [{'page': N, 'bbox': [x0, y0, x1, y1]}, ...] in
    reading order (page order, then top-to-bottom within a page). Opens
    both PDFs once rather than reopening per page."""
    fitz = _import_fitz()
    doc_before = fitz.open(stream=pdf_before, filetype='pdf')
    doc_after = fitz.open(stream=pdf_after, filetype='pdf')
    try:
        total = min(doc_before.page_count, doc_after.page_count)
        out: list[dict] = []
        for n in range(1, total + 1):
            before_img = _render_page_from_doc(fitz, doc_before, n, zoom)
            after_img = _render_page_from_doc(fitz, doc_after, n, zoom)
            _b, _a, mask = _diff_mask(before_img, after_img)
            for bbox in _detect_change_regions(mask):
                out.append({'page': n, 'bbox': list(bbox)})
        return out
    finally:
        doc_before.close()
        doc_after.close()


def render_page_diff(
    pdf_before: bytes, pdf_after: bytes, page_num: int, zoom: float = 1.5,
) -> tuple[bytes | None, int]:
    """Grayscale rendering of the BEFORE page with red pixel-level highlights
    wherever the AFTER page differs. Returns (png_bytes, total_pages) —
    total_pages is min(before, after) page counts; png_bytes is None if
    page_num is out of range for either document (caller should 404).
    """
    fitz = _import_fitz()
    before_img, before_pages = _render_page(fitz, pdf_before, page_num, zoom)
    after_img, after_pages = _render_page(fitz, pdf_after, page_num, zoom)
    total = min(before_pages, after_pages)
    if before_img is None or after_img is None:
        return None, total

    out = _composite_diff(before_img, after_img)
    buf = io.BytesIO()
    out.save(buf, format='PNG')
    return buf.getvalue(), total


