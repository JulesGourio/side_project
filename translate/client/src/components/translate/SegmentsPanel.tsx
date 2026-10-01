import { useCallback, useEffect, useState } from 'react';
import { Check, CheckCircle2, Eye, Languages, Loader2, Pencil, X } from 'lucide-react';
import { toast } from 'sonner';
import { Flag } from '../shared/Flag';
import { fetchJson } from '../../lib/fetchJson';

const ISO_TO_FLAG: Record<string, string> = { fr: 'FR', en: 'EN', es: 'ES', cs: 'CZ', bg: 'BG', pt: 'BR' };
const LANG_OPTIONS = ['fr', 'en', 'es', 'de', 'cs', 'bg', 'pt', 'ar'];

// Semantic language cell: flag + code. Confidence itself stays in Lakebase
// for diagnostics but isn't surfaced here — reviewers found the raw
// percentage noisy and not actionable next to the category badge.
function LangCell({ code }: { code: string | null; confidence: number | null }) {
  if (!code || code === '??') {
    return (
      <span className="inline-flex items-center gap-1">
        <span className="text-[11px]" style={{ color: 'var(--color-text-muted)' }}>Not detected</span>
      </span>
    );
  }
  const flag = ISO_TO_FLAG[code];
  return (
    <span className="inline-flex items-center gap-1.5">
      {flag && <Flag code={flag} />}
      <span className="text-[11px] font-medium uppercase" style={{ color: 'var(--color-text-primary)' }}>{code}</span>
    </span>
  );
}

const CATEGORY_STYLE: Record<string, { bg: string; fg: string; label: string }> = {
  translated: { bg: 'rgba(22,163,74,0.12)', fg: '#15803d', label: 'Translated' },
  kept_other_language: { bg: 'rgba(0,85,164,0.10)', fg: 'var(--color-accent-primary)', label: 'Kept (other lang)' },
  kept_dnt: { bg: 'var(--color-bg-secondary)', fg: 'var(--color-text-muted)', label: 'Do-not-translate' },
  kept_numeric: { bg: 'var(--color-bg-secondary)', fg: 'var(--color-text-muted)', label: 'Numeric' },
  kept_page_filtered: { bg: 'var(--color-bg-secondary)', fg: 'var(--color-text-muted)', label: 'Outside page range' },
  needs_review: { bg: 'rgba(245,158,11,0.14)', fg: '#b45309', label: 'Needs review' },
  pending: { bg: 'var(--color-bg-secondary)', fg: 'var(--color-text-muted)', label: 'Pending' },
  image_translation: { bg: 'rgba(147,51,234,0.12)', fg: '#7e22ce', label: 'Image' },
};

function CategoryBadge({ category }: { category: string }) {
  const s = CATEGORY_STYLE[category] ?? CATEGORY_STYLE.pending;
  return (
    <span className="text-[10px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded whitespace-nowrap" style={{ background: s.bg, color: s.fg }}>
      {s.label}
    </span>
  );
}

interface Segment {
  seg_id: string;
  index: number;
  source_text: string;
  translated_text: string | null;
  detected_lang: string | null;
  lang_confidence: number | null;
  category: string;
  flagged: boolean;
  conflict_detail: string | null;
  thumbnail_base64: string | null;
}

interface SegmentsResponse {
  segment_count: number;
  counts: Record<string, number>;
  flagged_count: number;
  total: number;
  segments: Segment[];
}

// Ordered so the buckets a reviewer cares about most (translated, then the
// "kept — expected" reasons, then the one that actually needs a fix) read
// left to right.
const CATEGORY_TABS: { key: string; label: string }[] = [
  { key: '', label: 'All' },
  { key: 'translated', label: 'Translated' },
  { key: 'kept_other_language', label: 'Kept (other lang)' },
  { key: 'kept_dnt', label: 'Do-not-translate' },
  { key: 'kept_numeric', label: 'Numeric' },
  { key: 'kept_page_filtered', label: 'Outside page range' },
  { key: 'needs_review', label: 'Needs review' },
  { key: 'image_translation', label: 'Images' },
  { key: 'flagged', label: 'Flagged' },
];

export function SegmentsPanel(
  {
    jobId, sourceLang, initialCategory = '', focusSegIds = null, onClose, jobDone, onViewInDocument,
    glossaryValidatedAt, onValidate, validating,
  }:
  {
    jobId: number; sourceLang: string; initialCategory?: string; focusSegIds?: string[] | null; onClose: () => void;
    jobDone: boolean;
    // Only passed once the job is done (the Review tab's side-by-side diff
    // only exists then) — undefined hides the "View in document" button.
    onViewInDocument?: (segId: string) => void;
    glossaryValidatedAt: string | null; onValidate: () => void; validating: boolean;
  },
) {
  const [data, setData] = useState<SegmentsResponse | null>(null);
  const [category, setCategory] = useState(initialCategory);
  // When opened from the residual panel's "Review & fix", scope to exactly the
  // flagged seg_ids (across all categories) until the reviewer clears it.
  const [segIdFilter, setSegIdFilter] = useState<string[] | null>(focusSegIds);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  const [retranslatingId, setRetranslatingId] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<string | null>(null);

  // A seg_id scope needs every row loaded (they can be anywhere in the doc),
  // so drop the category filter and raise the page size; otherwise the normal
  // paginated category view.
  const fetchSegments = useCallback(() => {
    const q = segIdFilter
      ? `category=&limit=100000`
      : `category=${encodeURIComponent(category)}&limit=100000`;
    return fetchJson<SegmentsResponse>(`/api/translate/jobs/${jobId}/segments?${q}`);
  }, [jobId, category, segIdFilter]);

  useEffect(() => {
    setLoading(true);
    fetchSegments()
      .then(setData)
      .catch(e => toast.error(e instanceof Error ? e.message : 'Failed to load segments'))
      .finally(() => setLoading(false));
  }, [fetchSegments]);

  const countFor = (key: string) => {
    if (!data) return null;
    if (key === '') return data.segment_count;
    if (key === 'flagged') return data.flagged_count;
    return data.counts[key] ?? 0;
  };

  const startEdit = (s: Segment) => {
    setEditingId(s.seg_id);
    setDraft(s.translated_text ?? s.source_text);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setDraft('');
  };

  const saveEdit = async (segId: string) => {
    setSaving(true);
    try {
      await fetchJson(`/api/translate/jobs/${jobId}/segments/${encodeURIComponent(segId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ translated_text: draft }),
      });
      // Re-fetch rather than patch the row locally: the server's category
      // logic depends on pattern_type (a DNT/numeric segment stays
      // kept_dnt/kept_numeric no matter what translated_text becomes), and
      // tab counts need to reflect the segment's actual new bucket, which a
      // local patch can't compute correctly.
      try {
        setData(await fetchSegments());
      } catch {
        // best-effort refresh; the edit itself already saved successfully
      }
      toast.success('Saved — click "Rebuild again" on the job page to apply it to the document.');
      setEditingId(null);
      setDraft('');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to save translation');
    } finally {
      setSaving(false);
    }
  };

  // Correcting the language to the document's source language means "this
  // needs translating" — go straight to the LLM instead of a separate step.
  // Correcting it to anything else is just a label fix (no translation).
  const changeLanguage = async (segId: string, lang: string) => {
    if (lang === sourceLang) {
      await retranslate(segId);
      return;
    }
    try {
      await fetchJson(`/api/translate/jobs/${jobId}/segments/${encodeURIComponent(segId)}/language`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detected_lang: lang }),
      });
      try {
        setData(await fetchSegments());
      } catch {
        // best-effort refresh
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to update language');
    }
  };

  // For a segment wrongly kept as-is (e.g. a false "already the target
  // language" detection) — the reviewer says "this needs translating"
  // instead of having to know/type the correct wording themselves.
  const retranslate = async (segId: string) => {
    setRetranslatingId(segId);
    try {
      await fetchJson(`/api/translate/jobs/${jobId}/segments/${encodeURIComponent(segId)}/retranslate`, {
        method: 'POST',
      });
      try {
        setData(await fetchSegments());
      } catch {
        // best-effort refresh; the retranslation itself already saved successfully
      }
      toast.success('Retranslated — click "Rebuild again" on the job page to apply it to the document.');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Retranslation failed');
    } finally {
      setRetranslatingId(null);
    }
  };

  const visibleSegments = data
    ? (segIdFilter ? data.segments.filter(s => segIdFilter.includes(s.seg_id)) : data.segments)
    : [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.5)' }}>
      <div
        className="w-full max-w-5xl max-h-[85vh] rounded-xl flex flex-col"
        style={{ background: 'var(--color-bg-primary)', border: '1px solid var(--color-border)' }}
      >
        <div className="flex items-center justify-between p-5 pb-3 flex-shrink-0 gap-3">
          <h2 className="text-lg font-bold" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-heading)' }}>
            Segments
          </h2>
          <div className="flex items-center gap-3">
            {jobDone && (
              glossaryValidatedAt ? (
                <div className="flex items-center gap-1.5 text-xs" style={{ color: '#15803d' }}>
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Glossary validated {new Date(glossaryValidatedAt).toLocaleString()}
                  <button
                    onClick={onValidate}
                    disabled={validating}
                    className="ml-1 underline disabled:opacity-50"
                    style={{ color: 'var(--color-text-muted)' }}
                    title="Re-run glossary extraction over the segments' current translations"
                  >
                    Re-validate
                  </button>
                </div>
              ) : (
                <button
                  onClick={onValidate}
                  disabled={validating}
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md font-medium text-white disabled:opacity-60"
                  style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}
                  title="Confirm this document's translation is correct — only then are its term pairs proposed to the glossary for review"
                >
                  <Check className="h-3.5 w-3.5" /> {validating ? 'Validating…' : 'Validate translation'}
                </button>
              )
            )}
            <button onClick={onClose}><X className="h-5 w-5" style={{ color: 'var(--color-text-muted)' }} /></button>
          </div>
        </div>

        <div className="px-5 overflow-y-auto flex-1">
          {segIdFilter ? (
            <div className="flex items-center justify-between gap-2 mb-4 px-3 py-2 rounded-lg"
                 style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)' }}>
              <span className="text-xs font-medium" style={{ color: '#92400e' }}>
                Showing {segIdFilter.length} segment{segIdFilter.length === 1 ? '' : 's'} that still read as the source language — edit each, then “Rebuild again”.
              </span>
              <button
                onClick={() => setSegIdFilter(null)}
                className="text-xs px-2 py-1 rounded-md flex-shrink-0"
                style={{ background: 'var(--color-bg-primary)', border: '1px solid rgba(245,158,11,0.4)', color: '#92400e' }}
              >
                Show all
              </button>
            </div>
          ) : (
            <div className="flex flex-wrap gap-1.5 mb-4">
              {CATEGORY_TABS.map(tab => {
                const count = countFor(tab.key);
                const active = category === tab.key;
                return (
                  <button
                    key={tab.key}
                    onClick={() => setCategory(tab.key)}
                    className="px-2.5 py-1 rounded-md text-xs font-medium transition-all"
                    style={{
                      background: active
                        ? 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)'
                        : 'var(--color-bg-secondary)',
                      color: active ? '#fff' : 'var(--color-text-muted)',
                      border: active ? 'none' : '1px solid var(--color-border)',
                    }}
                  >
                    {tab.label}{count != null ? ` (${count})` : ''}
                  </button>
                );
              })}
            </div>
          )}

          {loading ? (
            <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>Loading…</p>
          ) : !data || visibleSegments.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>No segments in this category.</p>
          ) : (
            <div className="overflow-x-auto pb-4">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ color: 'var(--color-text-muted)' }}>
                    <th className="text-left p-1.5">#</th>
                    <th className="text-left p-1.5">Source</th>
                    <th className="text-left p-1.5">Translated</th>
                    <th className="text-left p-1.5">Lang</th>
                    <th className="text-left p-1.5">Status</th>
                    <th className="p-1.5" />
                  </tr>
                </thead>
                <tbody>
                  {visibleSegments.map(s => {
                    const editing = editingId === s.seg_id;
                    return (
                      <tr key={s.seg_id} style={{ borderTop: '1px solid var(--color-border)' }}>
                        <td className="p-1.5 whitespace-nowrap align-top" style={{ color: 'var(--color-text-muted)' }}>{s.index + 1}</td>
                        <td className="p-1.5 max-w-[240px] align-top" style={{ color: 'var(--color-text-primary)' }}>
                          {s.thumbnail_base64 && (
                            <button
                              type="button"
                              onClick={() => setLightbox(s.thumbnail_base64)}
                              className="block mb-1 rounded overflow-hidden"
                              style={{ border: '1px solid var(--color-border)' }}
                              title="Click to enlarge"
                            >
                              <img src={`data:image/jpeg;base64,${s.thumbnail_base64}`} alt="" className="h-16 w-auto" />
                            </button>
                          )}
                          {s.source_text}
                        </td>
                        <td className="p-1.5 max-w-[300px] align-top">
                          {editing ? (
                            <textarea
                              autoFocus
                              className="w-full text-xs rounded p-1.5"
                              style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-accent-primary)', color: 'var(--color-text-primary)' }}
                              rows={3}
                              value={draft}
                              onChange={e => setDraft(e.target.value)}
                            />
                          ) : (
                            <div style={{ color: s.translated_text ? 'var(--color-text-primary)' : 'var(--color-text-muted)' }}>
                              {s.translated_text ?? (s.category === 'failed' ? '— not translated —' : '—')}
                              {s.flagged && (
                                <span
                                  className="ml-1.5 text-[10px] font-semibold uppercase tracking-wide px-1 py-0.5 rounded"
                                  style={{ background: 'rgba(245,158,11,0.12)', color: '#b45309' }}
                                  title={s.conflict_detail ?? undefined}
                                >
                                  flagged
                                </span>
                              )}
                              {s.flagged && s.conflict_detail && (
                                <div className="text-[11px] mt-0.5" style={{ color: '#b45309' }}>
                                  {s.conflict_detail}
                                </div>
                              )}
                            </div>
                          )}
                        </td>
                        <td className="p-1.5 whitespace-nowrap align-top">
                          <div className="flex items-center gap-1">
                            <LangCell code={s.detected_lang} confidence={s.lang_confidence} />
                            <select
                              value=""
                              onChange={e => { if (e.target.value) changeLanguage(s.seg_id, e.target.value); }}
                              className="text-[10px] rounded px-0.5 py-0.5"
                              style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)', color: 'var(--color-text-muted)' }}
                              title="Correct the detected language"
                            >
                              <option value="">✎</option>
                              {LANG_OPTIONS.map(l => <option key={l} value={l}>{l.toUpperCase()}</option>)}
                            </select>
                          </div>
                        </td>
                        <td className="p-1.5 whitespace-nowrap align-top">
                          <CategoryBadge category={s.category} />
                        </td>
                        <td className="p-1.5 whitespace-nowrap align-top">
                          {editing ? (
                            <div className="flex items-center gap-1">
                              <button onClick={() => saveEdit(s.seg_id)} disabled={saving} title="Save">
                                <Check className="h-3.5 w-3.5" style={{ color: 'var(--color-accent-primary)' }} />
                              </button>
                              <button onClick={cancelEdit} disabled={saving} title="Cancel">
                                <X className="h-3.5 w-3.5" style={{ color: 'var(--color-text-muted)' }} />
                              </button>
                            </div>
                          ) : (
                            <div className="flex items-center gap-1.5">
                              <button onClick={() => startEdit(s)} title="Edit translation">
                                <Pencil className="h-3.5 w-3.5" style={{ color: 'var(--color-text-muted)' }} />
                              </button>
                              {onViewInDocument && s.category !== 'image_translation' && (
                                <button onClick={() => onViewInDocument(s.seg_id)} title="View this segment in the document diff">
                                  <Eye className="h-3.5 w-3.5" style={{ color: 'var(--color-text-muted)' }} />
                                </button>
                              )}
                              {['kept_other_language', 'needs_review', 'pending'].includes(s.category) && (
                                <button
                                  onClick={() => retranslate(s.seg_id)}
                                  disabled={retranslatingId === s.seg_id}
                                  title="Wrong language detected — translate this segment"
                                >
                                  {retranslatingId === s.seg_id
                                    ? <Loader2 className="h-3.5 w-3.5 animate-spin" style={{ color: 'var(--color-accent-primary)' }} />
                                    : <Languages className="h-3.5 w-3.5" style={{ color: 'var(--color-accent-primary)' }} />}
                                </button>
                              )}
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {!segIdFilter && data.total > data.segments.length && (
                <p className="text-xs mt-2 px-1.5" style={{ color: 'var(--color-text-muted)' }}>
                  Showing first {data.segments.length} of {data.total}.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
      {lightbox && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center p-8"
          style={{ background: 'rgba(0,0,0,0.8)' }}
          onClick={() => setLightbox(null)}
        >
          <img src={`data:image/jpeg;base64,${lightbox}`} alt="" className="max-w-full max-h-full rounded-lg" />
          <button
            onClick={() => setLightbox(null)}
            className="absolute top-4 right-4 flex items-center justify-center h-8 w-8 rounded-full"
            style={{ background: 'rgba(255,255,255,0.15)' }}
          >
            <X className="h-5 w-5 text-white" />
          </button>
        </div>
      )}
    </div>
  );
}
