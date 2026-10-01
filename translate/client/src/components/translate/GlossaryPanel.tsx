import { useEffect, useMemo, useState } from 'react';
import { Check, Download, Plus, Trash2, X } from 'lucide-react';
import { toast } from 'sonner';
import { fetchJson } from '../../lib/fetchJson';

interface GlossaryTerm {
  term_id: string;
  en: string; fr: string; cs: string; bg: string; de: string; es: string; pt: string; ar: string;
  domain: string; notes: string;
  definition: string;
  // 'REF#chunk' (corpus terminology section), 'generated' (LLM),
  // 'cross_lang_pair' (corroborated by a confirmed translation pair)
  definition_source: string;
}

interface DntRule {
  id: number;
  pattern: string;
  type: string;
  match_mode: string;
  notes: string;
}

interface GlossaryCandidate {
  id: number;
  en: string; fr: string; cs: string; bg: string; de: string; es: string; pt: string; ar: string;
  n_docs: number;
  sources: string;
  definition: string;
  definition_source: string;
  priority: number | null;
  status: string;
  reviewed_by: string | null;
  reviewed_at: string | null;
  reject_reason: string | null;
}

const LANG_COLS: (keyof GlossaryTerm)[] = ['en', 'fr', 'cs', 'bg', 'de', 'es', 'pt', 'ar'];
const CAND_LANG_COLS: (keyof GlossaryCandidate)[] = ['en', 'fr', 'cs', 'bg', 'de', 'es', 'pt', 'ar'];
const CAND_STATUSES = ['pending', 'approved', 'rejected'] as const;
const CAND_PAGE_SIZE = 50;
const TERM_PAGE_SIZE = 50;

function csvCell(v: string | null | undefined): string {
  const s = v ?? '';
  return /[",\n;]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function downloadCsv(filename: string, rows: string[][]) {
  const csv = rows.map(r => r.map(csvCell).join(';')).join('\n');
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function GlossaryPanel({ onClose }: { onClose: () => void }) {
  const [terms, setTerms] = useState<GlossaryTerm[]>([]);
  const [dntRules, setDntRules] = useState<DntRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [newTerm, setNewTerm] = useState<Partial<GlossaryTerm>>({});
  const [newDnt, setNewDnt] = useState<Partial<DntRule>>({ match_mode: 'exact' });

  const [termSearch, setTermSearch] = useState('');
  const [termPage, setTermPage] = useState(0);
  const [expandedDefs, setExpandedDefs] = useState<Set<string>>(new Set());

  const [candidates, setCandidates] = useState<GlossaryCandidate[]>([]);
  const [candTotal, setCandTotal] = useState(0);
  const [candStatus, setCandStatus] = useState<(typeof CAND_STATUSES)[number]>('pending');
  const [candOffset, setCandOffset] = useState(0);
  const [candLoading, setCandLoading] = useState(true);
  const [rejectReasons, setRejectReasons] = useState<Record<number, string>>({});
  const [validatingAll, setValidatingAll] = useState(false);
  const [pendingCandIds, setPendingCandIds] = useState<Set<number>>(new Set());
  const [addingTerm, setAddingTerm] = useState(false);

  const reload = () => {
    fetchJson<{ terms: GlossaryTerm[]; dnt_rules: DntRule[] }>('/api/translate/glossary')
      .then(r => { setTerms(r.terms); setDntRules(r.dnt_rules); })
      .catch(e => toast.error(e instanceof Error ? e.message : 'Failed to load glossary'))
      .finally(() => setLoading(false));
  };

  useEffect(reload, []);

  const loadCandidates = (status: (typeof CAND_STATUSES)[number], offset: number, append: boolean) => {
    setCandLoading(true);
    fetchJson<{ candidates: GlossaryCandidate[]; total: number }>(
      `/api/translate/glossary/candidates?status=${status}&limit=${CAND_PAGE_SIZE}&offset=${offset}`
    )
      .then(r => {
        setCandidates(prev => (append ? [...prev, ...r.candidates] : r.candidates));
        setCandTotal(r.total);
      })
      .catch(e => toast.error(e instanceof Error ? e.message : 'Failed to load candidates'))
      .finally(() => setCandLoading(false));
  };

  useEffect(() => {
    setCandOffset(0);
    loadCandidates(candStatus, 0, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candStatus]);

  const filteredTerms = useMemo(() => {
    const q = termSearch.trim().toLowerCase();
    if (!q) return terms;
    return terms.filter(t =>
      [...LANG_COLS.map(l => t[l]), t.domain, t.definition, t.term_id]
        .some(v => (v || '').toLowerCase().includes(q)),
    );
  }, [terms, termSearch]);

  const termPageCount = Math.max(1, Math.ceil(filteredTerms.length / TERM_PAGE_SIZE));
  const pagedTerms = filteredTerms.slice(termPage * TERM_PAGE_SIZE, (termPage + 1) * TERM_PAGE_SIZE);

  // Keep the current page in range when a delete/search shrinks the list —
  // otherwise it can land past the last page, showing zero rows with no
  // indication why ("Page 4 / 2", Next disabled, Prev the only way out).
  useEffect(() => {
    setTermPage(p => Math.min(p, termPageCount - 1));
  }, [termPageCount]);

  const exportTermsCsv = () => {
    const header = ['term_id', ...LANG_COLS, 'domain', 'definition', 'definition_source'];
    const rows = filteredTerms.map(t => [t.term_id, ...LANG_COLS.map(l => t[l]), t.domain, t.definition, t.definition_source]);
    downloadCsv('glossary_terms.csv', [header, ...rows]);
  };

  const toggleDef = (termId: string) => {
    setExpandedDefs(prev => {
      const next = new Set(prev);
      if (next.has(termId)) next.delete(termId); else next.add(termId);
      return next;
    });
  };

  const approveCandidate = async (id: number) => {
    if (pendingCandIds.has(id)) return;
    setPendingCandIds(prev => new Set(prev).add(id));
    try {
      await fetchJson(`/api/translate/glossary/candidates/${id}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      setCandidates(prev => prev.filter(c => c.id !== id));
      setCandTotal(t => Math.max(0, t - 1));
      reload();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to approve candidate');
    } finally {
      setPendingCandIds(prev => { const next = new Set(prev); next.delete(id); return next; });
    }
  };

  const rejectCandidate = async (id: number) => {
    if (pendingCandIds.has(id)) return;
    setPendingCandIds(prev => new Set(prev).add(id));
    try {
      await fetchJson(`/api/translate/glossary/candidates/${id}/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: rejectReasons[id] || '' }),
      });
      setCandidates(prev => prev.filter(c => c.id !== id));
      setCandTotal(t => Math.max(0, t - 1));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to reject candidate');
    } finally {
      setPendingCandIds(prev => { const next = new Set(prev); next.delete(id); return next; });
    }
  };

  const validateAll = async () => {
    if (!window.confirm(`Approve all ${candTotal} pending term${candTotal === 1 ? '' : 's'} as-is? This moves them straight into the glossary.`)) return;
    setValidatingAll(true);
    try {
      await fetchJson('/api/translate/glossary/candidates/approve-all', { method: 'POST' });
      setCandidates([]);
      setCandTotal(0);
      reload();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to validate all candidates');
    } finally {
      setValidatingAll(false);
    }
  };

  const addTerm = async () => {
    if (addingTerm) return;
    if (!newTerm.en?.trim() || !newTerm.fr?.trim()) {
      toast.error('Enter at least an English and a French term');
      return;
    }
    setAddingTerm(true);
    try {
      await fetchJson('/api/translate/glossary/terms', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTerm),
      });
      setNewTerm({});
      reload();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to add term');
    } finally {
      setAddingTerm(false);
    }
  };

  const deleteTerm = async (termId: string) => {
    try {
      await fetchJson(`/api/translate/glossary/terms/${encodeURIComponent(termId)}`, { method: 'DELETE' });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to delete term');
    }
    reload();
  };

  const addDnt = async () => {
    if (!newDnt.pattern) return;
    try {
      await fetchJson('/api/translate/glossary/dnt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newDnt),
      });
      setNewDnt({ match_mode: 'exact' });
      reload();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to add DNT rule');
    }
  };

  const deleteDnt = async (id: number) => {
    try {
      await fetchJson(`/api/translate/glossary/dnt/${id}`, { method: 'DELETE' });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to delete DNT rule');
    }
    reload();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.5)' }}>
      <div
        className="w-full max-w-4xl max-h-[85vh] rounded-xl flex flex-col"
        style={{ background: 'var(--color-bg-primary)', border: '1px solid var(--color-border)' }}
      >
        <div className="flex items-center justify-between p-5 pb-3 flex-shrink-0">
          <h2 className="text-lg font-bold" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-heading)' }}>
            Glossary & do-not-translate rules
          </h2>
          <button onClick={onClose}><X className="h-5 w-5" style={{ color: 'var(--color-text-muted)' }} /></button>
        </div>

        <div className="px-5 pb-5 overflow-y-auto flex-1">
        {loading ? (
          <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>Loading…</p>
        ) : (
          <>
            <div className="flex items-center justify-between mb-2 gap-2 flex-wrap">
              <h3 className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                Terms ({filteredTerms.length}{filteredTerms.length !== terms.length ? ` / ${terms.length}` : ''})
              </h3>
              <div className="flex items-center gap-2">
                <input
                  className="text-xs rounded-md px-2 py-1"
                  style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)', color: 'var(--color-text-primary)' }}
                  placeholder="Search terms, domain, definition…"
                  value={termSearch}
                  onChange={e => { setTermSearch(e.target.value); setTermPage(0); }}
                />
                <button
                  onClick={exportTermsCsv}
                  className="flex items-center gap-1 text-xs px-2 py-1 rounded-md"
                  style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
                  title="Export terms (respects current search) as CSV"
                >
                  <Download className="h-3.5 w-3.5" /> Export
                </button>
              </div>
            </div>
            <div className="overflow-x-auto mb-2">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ color: 'var(--color-text-muted)' }}>
                    {LANG_COLS.map(l => <th key={l} className="text-left p-1 uppercase">{l}</th>)}
                    <th className="text-left p-1">Domain</th>
                    <th className="text-left p-1">Definition</th>
                    <th className="p-1" />
                  </tr>
                </thead>
                <tbody>
                  {pagedTerms.map(t => {
                    const expanded = expandedDefs.has(t.term_id);
                    return (
                      <tr key={t.term_id} style={{ borderTop: '1px solid var(--color-border)' }}>
                        {LANG_COLS.map(l => (
                          <td key={l} className="p-1" style={{ color: 'var(--color-text-primary)' }}>{t[l]}</td>
                        ))}
                        <td className="p-1" style={{ color: t.domain ? 'var(--color-text-muted)' : 'var(--color-border)' }}>
                          {t.domain || '—'}
                        </td>
                        <td
                          className={`p-1 cursor-pointer ${expanded ? 'max-w-[16rem] whitespace-normal break-words' : 'max-w-[16rem] truncate'}`}
                          style={{ color: 'var(--color-text-muted)' }}
                          onClick={() => t.definition && toggleDef(t.term_id)}
                          title={!expanded && t.definition ? 'Click to expand' : undefined}
                        >
                          {t.definition || '—'}
                          {t.definition && expanded && t.definition_source && (
                            <span className="block text-[10px] mt-0.5 opacity-70">[{t.definition_source}]</span>
                          )}
                        </td>
                        <td className="p-1">
                          <button onClick={() => deleteTerm(t.term_id)}>
                            <Trash2 className="h-3.5 w-3.5" style={{ color: '#ef4444' }} />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                  <tr style={{ borderTop: '1px solid var(--color-border)' }}>
                    {LANG_COLS.map(l => (
                      <td key={l} className="p-1">
                        <input
                          className="w-full text-xs rounded p-1"
                          style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
                          value={newTerm[l] || ''}
                          onChange={e => setNewTerm(n => ({ ...n, [l]: e.target.value }))}
                        />
                      </td>
                    ))}
                    <td className="p-1">
                      <input
                        className="w-full text-xs rounded p-1"
                        style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
                        value={newTerm.domain || ''}
                        onChange={e => setNewTerm(n => ({ ...n, domain: e.target.value }))}
                      />
                    </td>
                    <td className="p-1">
                      <input
                        className="w-full text-xs rounded p-1"
                        style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
                        value={newTerm.definition || ''}
                        onChange={e => setNewTerm(n => ({ ...n, definition: e.target.value }))}
                        placeholder="Definition (optional)"
                      />
                    </td>
                    <td className="p-1">
                      <button onClick={addTerm} disabled={addingTerm}>
                        <Plus className="h-3.5 w-3.5" style={{ color: 'var(--color-accent-primary)', opacity: addingTerm ? 0.4 : 1 }} />
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            {termPageCount > 1 && (
              <div className="flex items-center justify-center gap-2 mb-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                <button
                  onClick={() => setTermPage(p => Math.max(0, p - 1))}
                  disabled={termPage === 0}
                  className="px-2 py-0.5 rounded-md disabled:opacity-40"
                  style={{ border: '1px solid var(--color-border)' }}
                >
                  ‹ Prev
                </button>
                Page {termPage + 1} / {termPageCount}
                <button
                  onClick={() => setTermPage(p => Math.min(termPageCount - 1, p + 1))}
                  disabled={termPage >= termPageCount - 1}
                  className="px-2 py-0.5 rounded-md disabled:opacity-40"
                  style={{ border: '1px solid var(--color-border)' }}
                >
                  Next ›
                </button>
              </div>
            )}

            <h3 className="text-sm font-semibold mb-2 mt-4" style={{ color: 'var(--color-text-primary)' }}>
              Do-not-translate rules ({dntRules.length})
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ color: 'var(--color-text-muted)' }}>
                    <th className="text-left p-1">Pattern</th>
                    <th className="text-left p-1">Type</th>
                    <th className="text-left p-1">Match mode</th>
                    <th className="p-1" />
                  </tr>
                </thead>
                <tbody>
                  {dntRules.map(r => (
                    <tr key={r.id} style={{ borderTop: '1px solid var(--color-border)' }}>
                      <td className="p-1" style={{ color: 'var(--color-text-primary)' }}>{r.pattern}</td>
                      <td className="p-1" style={{ color: 'var(--color-text-muted)' }}>{r.type}</td>
                      <td className="p-1" style={{ color: 'var(--color-text-muted)' }}>{r.match_mode}</td>
                      <td className="p-1">
                        <button onClick={() => deleteDnt(r.id)}>
                          <Trash2 className="h-3.5 w-3.5" style={{ color: '#ef4444' }} />
                        </button>
                      </td>
                    </tr>
                  ))}
                  <tr style={{ borderTop: '1px solid var(--color-border)' }}>
                    <td className="p-1">
                      <input
                        className="w-full text-xs rounded p-1"
                        style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
                        value={newDnt.pattern || ''}
                        onChange={e => setNewDnt(n => ({ ...n, pattern: e.target.value }))}
                        placeholder="NAS*"
                      />
                    </td>
                    <td className="p-1">
                      <input
                        className="w-full text-xs rounded p-1"
                        style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
                        value={newDnt.type || ''}
                        onChange={e => setNewDnt(n => ({ ...n, type: e.target.value }))}
                      />
                    </td>
                    <td className="p-1">
                      <select
                        className="w-full text-xs rounded p-1"
                        style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
                        value={newDnt.match_mode || 'exact'}
                        onChange={e => setNewDnt(n => ({ ...n, match_mode: e.target.value }))}
                      >
                        <option value="exact">exact</option>
                        <option value="prefix">prefix</option>
                        <option value="glob">glob</option>
                        <option value="regex">regex</option>
                      </select>
                    </td>
                    <td className="p-1">
                      <button onClick={addDnt}><Plus className="h-3.5 w-3.5" style={{ color: 'var(--color-accent-primary)' }} /></button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-between mb-2 mt-4 gap-2 flex-wrap">
              <h3 className="text-sm font-semibold" style={{ color: 'var(--color-text-primary)' }}>
                To review ({candTotal})
              </h3>
              <div className="flex items-center gap-2">
                <div className="flex gap-1">
                  {CAND_STATUSES.map(s => (
                    <button
                      key={s}
                      className="text-[10px] px-2 py-0.5 rounded capitalize"
                      style={{
                        background: s === candStatus ? 'var(--color-accent-primary)' : 'var(--color-bg-secondary)',
                        color: s === candStatus ? 'white' : 'var(--color-text-muted)',
                        border: '1px solid var(--color-border)',
                      }}
                      onClick={() => setCandStatus(s)}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                {candStatus === 'pending' && candTotal > 0 && (
                  <button
                    onClick={validateAll}
                    disabled={validatingAll}
                    className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-md font-medium text-white disabled:opacity-60"
                    style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}
                    title="Approve every pending candidate as-is, all at once"
                  >
                    <Check className="h-3.5 w-3.5" /> {validatingAll ? 'Validating…' : `Validate all (${candTotal})`}
                  </button>
                )}
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ color: 'var(--color-text-muted)' }}>
                    <th className="text-left p-1">P</th>
                    {CAND_LANG_COLS.map(l => <th key={l} className="text-left p-1 uppercase">{l}</th>)}
                    <th className="text-left p-1">Docs</th>
                    <th className="text-left p-1">Sources</th>
                    <th className="text-left p-1">Definition</th>
                    {candStatus === 'pending' && <th className="p-1" />}
                    {candStatus === 'rejected' && <th className="text-left p-1">Reason</th>}
                  </tr>
                </thead>
                <tbody>
                  {candidates.map(c => (
                    <tr key={c.id} style={{ borderTop: '1px solid var(--color-border)' }}>
                      <td className="p-1" style={{ color: 'var(--color-text-muted)' }}>{c.priority ?? '-'}</td>
                      {CAND_LANG_COLS.map(l => (
                        <td key={l} className="p-1" style={{ color: 'var(--color-text-primary)' }}>{c[l]}</td>
                      ))}
                      <td className="p-1" style={{ color: 'var(--color-text-muted)' }}>{c.n_docs}</td>
                      <td
                        className="p-1 max-w-[10rem] truncate"
                        style={{ color: 'var(--color-text-muted)' }}
                        title={c.sources}
                      >
                        {c.sources}
                      </td>
                      <td
                        className="p-1 max-w-[14rem] truncate"
                        style={{ color: 'var(--color-text-muted)' }}
                        title={c.definition}
                      >
                        {c.definition}
                      </td>
                      {candStatus === 'pending' && (
                        <td className="p-1">
                          <div className="flex items-center gap-1">
                            <button onClick={() => approveCandidate(c.id)} disabled={pendingCandIds.has(c.id)} title="Approve">
                              <Plus className="h-3.5 w-3.5" style={{ color: 'var(--color-accent-primary)', opacity: pendingCandIds.has(c.id) ? 0.4 : 1 }} />
                            </button>
                            <input
                              className="w-20 text-[10px] rounded p-0.5"
                              style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
                              placeholder="reason"
                              value={rejectReasons[c.id] || ''}
                              onChange={e => setRejectReasons(r => ({ ...r, [c.id]: e.target.value }))}
                            />
                            <button onClick={() => rejectCandidate(c.id)} disabled={pendingCandIds.has(c.id)} title="Reject">
                              <Trash2 className="h-3.5 w-3.5" style={{ color: '#ef4444', opacity: pendingCandIds.has(c.id) ? 0.4 : 1 }} />
                            </button>
                          </div>
                        </td>
                      )}
                      {candStatus === 'rejected' && (
                        <td className="p-1" style={{ color: 'var(--color-text-muted)' }}>{c.reject_reason}</td>
                      )}
                    </tr>
                  ))}
                  {!candLoading && candidates.length === 0 && (
                    <tr>
                      <td colSpan={9} className="p-2 text-center" style={{ color: 'var(--color-text-muted)' }}>
                        No {candStatus} candidates
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
              {candOffset + candidates.length < candTotal && (
                <button
                  className="text-[11px] mt-2 px-2 py-1 rounded"
                  style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)', color: 'var(--color-text-primary)' }}
                  disabled={candLoading}
                  onClick={() => {
                    const next = candOffset + CAND_PAGE_SIZE;
                    setCandOffset(next);
                    loadCandidates(candStatus, next, true);
                  }}
                >
                  Show more
                </button>
              )}
            </div>
          </>
        )}
        </div>
      </div>
    </div>
  );
}
