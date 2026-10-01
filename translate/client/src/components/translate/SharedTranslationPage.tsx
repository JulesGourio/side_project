import { useEffect, useState } from 'react';
import { AlertCircle, Download, FileText, Loader2, Lock } from 'lucide-react';
import { fetchJson } from '../../lib/fetchJson';

interface SharedSegment {
  seg_id: string;
  source_text: string | null;
  translated_text: string | null;
  flagged: boolean;
}

interface SharedJob {
  id: number;
  created_at: string | null;
  status: string;
  original_filename: string;
  source_lang: string;
  target_lang: string;
  segment_count: number | null;
  needs_translation_count: number | null;
  segments: SharedSegment[];
  feedback: { vote: 'up' | 'down'; comment: string | null } | null;
  output_docx_base64: string | null;
}

function downloadBase64Docx(base64: string, filename: string) {
  const bytes = atob(base64);
  const buffer = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) buffer[i] = bytes.charCodeAt(i);
  const blob = new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function SharedTranslationPage({ token }: { token: string }) {
  const [job, setJob] = useState<SharedJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchJson<SharedJob>(`/api/translate/shared/${token}`)
      .then(setJob)
      .catch(e => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [token]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full gap-2" style={{ color: 'var(--color-text-muted)' }}>
        <Loader2 className="h-5 w-5 animate-spin" />
        <span className="text-sm">Loading shared translation…</span>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 px-8 text-center">
        <AlertCircle className="h-8 w-8 opacity-40" style={{ color: 'var(--color-text-muted)' }} />
        <p className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
          This shared link is invalid or has been removed.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      <div className="flex items-center justify-between mb-2">
        <div className="min-w-0">
          <h1 className="text-lg font-bold truncate flex items-center gap-2" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-heading)' }}>
            <FileText className="h-4 w-4 flex-shrink-0" />
            {job.original_filename}
          </h1>
          <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
            {job.source_lang.toUpperCase()} → {job.target_lang.toUpperCase()}
            {job.segment_count != null && ` · ${job.segment_count} segments`}
          </p>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md" style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}>
            <Lock className="h-3 w-3" /> Read-only
          </span>
          {job.output_docx_base64 && (
            <button
              onClick={() => downloadBase64Docx(job.output_docx_base64 as string, `translated_${job.original_filename}`)}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md font-medium text-white"
              style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}
            >
              <Download className="h-3.5 w-3.5" /> Download translated document
            </button>
          )}
        </div>
      </div>

      {job.feedback && (
        <p className="text-xs mb-4" style={{ color: 'var(--color-text-muted)' }}>
          Owner marked this translation as {job.feedback.vote === 'up' ? 'helpful' : 'not helpful'}
          {job.feedback.comment ? `: "${job.feedback.comment}"` : ''}
        </p>
      )}

      <div className="rounded-lg border divide-y" style={{ borderColor: 'var(--color-border)' }}>
        {job.segments.length === 0 ? (
          <p className="text-sm px-4 py-6 text-center" style={{ color: 'var(--color-text-muted)' }}>
            No segments to show yet.
          </p>
        ) : (
          job.segments.map(s => (
            <div
              key={s.seg_id}
              className="grid grid-cols-2 gap-4 px-4 py-3 text-sm"
              style={s.flagged ? { background: 'rgba(245,158,11,0.06)' } : undefined}
            >
              <p style={{ color: 'var(--color-text-muted)' }}>{s.source_text}</p>
              <p style={{ color: 'var(--color-text-primary)' }}>{s.translated_text}</p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
