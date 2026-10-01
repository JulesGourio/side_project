import { useCallback, useState } from 'react';
import { Download, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

// ---------------------------------------------------------------------------
// Exact-layout preview: the server converts both .docx through headless
// LibreOffice and the browser renders the two PDFs side by side in its native
// viewer (zoom / search / page-nav built in). The PDFs are pregenerated at the
// end of the rebuild stage and persisted in the UC Volume, so opening this is
// normally near-instant. This is the fidelity check (tables / cartouche /
// images) — the change-by-change reviewing happens in the Review tab.
// ---------------------------------------------------------------------------

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

const DOCX_MIME = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';

export function LayoutPreview({ jobId, filename }: { jobId: number; filename: string }) {
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  // key = `${side}:${format}`, e.g. "after:pdf" — only one download in flight
  // at a time keeps the button state simple to reason about.
  const [downloading, setDownloading] = useState<string | null>(null);

  // .docx bytes are only fetched when a download button is clicked — opening
  // the preview never pays for them.
  const downloadDocx = useCallback(async (side: 'before' | 'after') => {
    const key = `${side}:docx`;
    setDownloading(key);
    try {
      const r = await (async () => {
        const res = await fetch(`/api/translate/jobs/${jobId}/preview`);
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body?.error || res.statusText);
        return body as { before_docx_base64: string; after_docx_base64: string };
      })();
      const bytes = base64ToBytes(side === 'before' ? r.before_docx_base64 : r.after_docx_base64);
      const base = filename.replace(/\.docx$/i, '');
      saveBlob(
        new Blob([bytes as BlobPart], { type: DOCX_MIME }),
        side === 'before' ? filename : `${base}_translated.docx`,
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Download failed');
    } finally {
      setDownloading(null);
    }
  }, [jobId, filename]);

  // The rendered PDF — same fetch-as-blob pattern as the .docx download, so
  // it always saves to disk instead of the browser possibly opening it inline
  // (which is what a plain `<a href>` to this endpoint would otherwise risk).
  const downloadPdf = useCallback(async (side: 'before' | 'after') => {
    const key = `${side}:pdf`;
    setDownloading(key);
    try {
      const res = await fetch(`/api/translate/jobs/${jobId}/preview.pdf?side=${side}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || res.statusText);
      }
      const blob = await res.blob();
      const base = filename.replace(/\.docx$/i, '');
      saveBlob(blob, side === 'before' ? `${base}.pdf` : `${base}_translated.pdf`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Download failed');
    } finally {
      setDownloading(null);
    }
  }, [jobId, filename]);

  // A non-PDF iframe response is an error payload: same-origin lets us read it
  // and surface the reason instead of leaving raw JSON in the pane.
  const checkIframeError = useCallback((e: React.SyntheticEvent<HTMLIFrameElement>) => {
    try {
      const doc = e.currentTarget.contentDocument;
      if (doc && doc.contentType && doc.contentType !== 'application/pdf') {
        const text = doc.body?.innerText?.slice(0, 400) || 'Preview unavailable';
        try { setError(JSON.parse(text).error || text); } catch { setError(text); }
      }
    } catch { /* PDF viewer documents are opaque — that's the success case */ }
  }, []);

  if (error) {
    return (
      <div className="flex items-center gap-3 py-4">
        <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>{error}</p>
        <button
          onClick={() => { setError(null); setAttempt(a => a + 1); }}
          className="text-xs px-2.5 py-1 rounded-md"
          style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      {(['before', 'after'] as const).map(side => (
        <div key={side} className="rounded-lg overflow-hidden flex flex-col" style={{ border: '1px solid var(--color-border)' }}>
          <div className="flex items-center justify-between pl-3 pr-1.5 py-1" style={{ background: 'var(--color-bg-secondary)' }}>
            <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
              {side === 'before' ? 'Original' : 'Translated'}
            </span>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => downloadDocx(side)}
                disabled={downloading !== null}
                className="flex items-center gap-1.5 text-xs px-2 py-1 rounded-md disabled:opacity-50"
                style={side === 'after'
                  ? { background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)', color: '#fff', fontWeight: 500 }
                  : { color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
              >
                {downloading === `${side}:docx` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />} .docx
              </button>
              <button
                onClick={() => downloadPdf(side)}
                disabled={downloading !== null}
                className="flex items-center gap-1.5 text-xs px-2 py-1 rounded-md disabled:opacity-50"
                style={side === 'after'
                  ? { background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)', color: '#fff', fontWeight: 500 }
                  : { color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
              >
                {downloading === `${side}:pdf` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />} .pdf
              </button>
            </div>
          </div>
          <iframe
            key={attempt}
            title={side === 'before' ? 'Original document' : 'Translated document'}
            src={`/api/translate/jobs/${jobId}/preview.pdf?side=${side}`}
            onLoad={checkIframeError}
            className="w-full flex-1"
            style={{ border: 'none', background: '#525659', height: '72vh', minHeight: '600px' }}
          />
        </div>
      ))}
    </div>
  );
}
