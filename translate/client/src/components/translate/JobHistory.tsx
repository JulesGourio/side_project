import { useState, useEffect, useCallback } from 'react';
import { History, X, FileText, ChevronRight, Loader2, RefreshCw } from 'lucide-react';
import { fetchJson } from '../../lib/fetchJson';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface HistoryEntry {
  id: number;
  created_at: string;
  status: string;
  original_filename: string;
  source_lang: string;
  target_lang: string;
  segment_count: number | null;
  needs_translation_count: number | null;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onLoad: (id: number) => void | Promise<void>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(iso: string): string {
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

function shortName(name: string, maxLen = 22): string {
  if (name.length <= maxLen) return name;
  const ext = name.lastIndexOf('.');
  const base = ext > 0 ? name.slice(0, ext) : name;
  const suffix = ext > 0 ? name.slice(ext) : '';
  return `${base.slice(0, maxLen - 3 - suffix.length)}…${suffix}`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function JobHistory({ open, onClose, onLoad }: Props) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [available, setAvailable] = useState(true);
  const [loading, setLoading] = useState(false);
  const [loadingId, setLoadingId] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [offset, setOffset] = useState(0);
  const PAGE = 20;

  const fetchHistory = useCallback(async (off = 0) => {
    setLoading(true);
    setError('');
    try {
      const data = await fetchJson<{ jobs: HistoryEntry[]; total: number; available: boolean }>(
        `/api/translate/jobs?limit=${PAGE}&offset=${off}`,
      );
      if (data.available === false) {
        setAvailable(false);
        return;
      }
      setAvailable(true);
      setEntries(off === 0 ? data.jobs : (prev) => [...prev, ...data.jobs]);
      setTotal(data.total ?? 0);
      setOffset(off);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) fetchHistory(0);
  }, [open, fetchHistory]);

  const handleLoad = async (id: number) => {
    setLoadingId(id);
    try {
      await onLoad(id);
      onClose();
    } finally {
      setLoadingId(null);
    }
  };

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/30 backdrop-blur-[2px] z-40"
        onClick={onClose}
      />

      {/* Drawer */}
      <div className="fixed right-0 top-0 h-full w-[400px] max-w-full z-50 flex flex-col shadow-2xl"
        style={{ background: 'var(--color-background)', borderLeft: '1px solid var(--color-border)' }}>

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b"
          style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: 'var(--color-accent-primary)18' }}>
              <History className="h-3.5 w-3.5" style={{ color: 'var(--color-accent-primary)' }} />
            </div>
            <span className="text-sm font-semibold" style={{ color: 'var(--color-text-heading)' }}>
              Job History
            </span>
            {available && total > 0 && (
              <span className="text-xs px-1.5 py-0.5 rounded-full font-medium"
                style={{ background: 'var(--color-accent-primary)15', color: 'var(--color-accent-primary)' }}>
                {total}
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => fetchHistory(0)}
              disabled={loading}
              className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors disabled:opacity-40 cursor-pointer"
              style={{ color: 'var(--color-text-muted)' }}
              title="Refresh"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors cursor-pointer"
              style={{ color: 'var(--color-text-muted)' }}
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto">
          {!available ? (
            <div className="flex flex-col items-center justify-center h-full gap-3 px-8 text-center">
              <History className="h-8 w-8 opacity-20" style={{ color: 'var(--color-text-muted)' }} />
              <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                History requires Lakebase.<br />Set <code className="text-xs">LAKEBASE_PROJECT_ID</code> in app.yaml.
              </p>
            </div>
          ) : error ? (
            <div className="m-4 px-4 py-3 rounded-xl text-sm"
              style={{ background: 'var(--color-error)08', color: 'var(--color-error)', border: '1px solid var(--color-error)25' }}>
              {error}
            </div>
          ) : loading && entries.length === 0 ? (
            <div className="flex items-center justify-center h-full gap-2"
              style={{ color: 'var(--color-text-muted)' }}>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">Loading…</span>
            </div>
          ) : entries.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-3 px-8 text-center">
              <History className="h-8 w-8 opacity-20" style={{ color: 'var(--color-text-muted)' }} />
              <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                No translation jobs yet.<br />Upload a document to see it here.
              </p>
            </div>
          ) : (
            <ul className="divide-y" style={{ borderColor: 'var(--color-border)30' }}>
              {entries.map((e) => (
                <li key={e.id}>
                  <button
                    onClick={() => handleLoad(e.id)}
                    disabled={loadingId !== null}
                    className="w-full text-left px-5 py-4 transition-colors disabled:opacity-50 cursor-pointer group"
                    style={{ background: 'transparent' }}
                    onMouseEnter={(el) => (el.currentTarget.style.background = 'var(--color-accent-primary)06')}
                    onMouseLeave={(el) => (el.currentTarget.style.background = 'transparent')}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center mt-0.5"
                        style={{ background: 'var(--color-accent-primary)12' }}>
                        {loadingId === e.id
                          ? <Loader2 className="h-3.5 w-3.5 animate-spin" style={{ color: 'var(--color-accent-primary)' }} />
                          : <FileText className="h-3.5 w-3.5" style={{ color: 'var(--color-accent-primary)' }} />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-medium truncate" style={{ color: 'var(--color-text-heading)' }}>
                          {shortName(e.original_filename)}
                        </p>
                        <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-muted)' }}>
                          {e.source_lang.toUpperCase()} → {e.target_lang.toUpperCase()} · {e.status} · {formatDate(e.created_at)}
                        </p>
                      </div>
                      <ChevronRight className="h-3.5 w-3.5 flex-shrink-0 mt-1 opacity-0 group-hover:opacity-60 transition-opacity"
                        style={{ color: 'var(--color-accent-primary)' }} />
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Load more */}
        {available && entries.length < total && (
          <div className="px-5 py-3 border-t" style={{ borderColor: 'var(--color-border)' }}>
            <button
              onClick={() => fetchHistory(offset + PAGE)}
              disabled={loading}
              className="w-full py-2 rounded-lg text-xs font-medium transition-colors disabled:opacity-40 cursor-pointer"
              style={{ color: 'var(--color-accent-primary)', background: 'var(--color-accent-primary)08' }}
            >
              {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin mx-auto" /> : `Load more (${total - entries.length} remaining)`}
            </button>
          </div>
        )}
      </div>
    </>
  );
}
