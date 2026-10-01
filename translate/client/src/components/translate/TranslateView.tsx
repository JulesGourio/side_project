import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Upload, FileText, X, Loader2, AlertCircle, AlertTriangle, CheckCircle2, History, ArrowLeft, BookOpen,
  ListChecks, RefreshCw, ArrowRight, Download, MessageSquare, Eye, Check,
} from 'lucide-react';
import { toast } from 'sonner';
import { GlossaryPanel } from './GlossaryPanel';
import { SegmentsPanel } from './SegmentsPanel';
import { ReviewPanel } from './ReviewPanel';
import { LayoutPreview } from './LayoutPreview';
import { JobHistory } from './JobHistory';
import { FeedbackPanel } from './FeedbackPanel';
import { Share2 } from 'lucide-react';
import { Flag } from '../shared/Flag';
import { fetchJson } from '../../lib/fetchJson';

// iso-639 -> Flag.tsx glyph code (de/ar have no glyph — falls back to the code text)
const ISO_TO_FLAG: Record<string, string> = { fr: 'FR', en: 'EN', es: 'ES', cs: 'CZ', bg: 'BG', pt: 'BR' };

function LangBadge({ code }: { code: string | null | undefined }) {
  if (!code || code === '??') {
    return <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>—</span>;
  }
  const flag = ISO_TO_FLAG[code];
  return (
    <span className="inline-flex items-center gap-1">
      {flag && <Flag code={flag} />}
      <span className="text-xs font-medium uppercase" style={{ color: 'var(--color-text-primary)' }}>{code}</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

const LANGUAGES: { value: string; label: string }[] = [
  { value: 'en', label: 'English' },
  { value: 'fr', label: 'French' },
  { value: 'es', label: 'Spanish' },
  { value: 'de', label: 'German' },
  { value: 'cs', label: 'Czech' },
  { value: 'bg', label: 'Bulgarian' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'ar', label: 'Arabic' },
];

type JobStatus =
  | 'uploaded' | 'extracting' | 'auditing' | 'awaiting_answers' | 'answered'
  | 'translating' | 'translated' | 'fit_checking' | 'rebuilding' | 'validating'
  | 'done' | 'done_with_warnings' | 'failed';

interface Question {
  q_id: string;
  seg_ids: string[];
  category: string;
  question_text: string;
  context: string;
  suggested_answer: string | null;
  answer: string | null;
}

interface JobState {
  id: number;
  status: JobStatus;
  stage_progress: Record<string, unknown> | null;
  source_lang: string;
  target_lang: string;
  original_filename: string;
  segment_count: number | null;
  needs_translation_count: number | null;
  error_type: string | null;
  error_msg: string | null;
  stale: boolean;
  questions: Question[];
  notes: string | null;
  glossary_validated_at: string | null;
}

const TERMINAL_STATUSES: JobStatus[] = ['done', 'done_with_warnings', 'failed'];
// Mirrors server ACTIVE_PROCESSING_STATUSES (translate.py) — a worker is (or
// should be) actively driving the job in these statuses, so /restart refuses
// them server-side; hide the button rather than let the user hit that 409.
const ACTIVE_PROCESSING_STATUSES: JobStatus[] = [
  'uploaded', 'extracting', 'auditing', 'translating',
  'fit_checking', 'rebuilding', 'validating',
];
const POLL_INTERVAL_MS = 2500;

const STATUS_LABELS: Record<JobStatus, string> = {
  uploaded: 'Uploaded',
  extracting: 'Extracting segments…',
  auditing: 'Detecting languages & pairing segments…',
  awaiting_answers: 'Awaiting your answers',
  answered: 'Ready to translate',
  translating: 'Translating…',
  translated: 'Translated — ready to rebuild',
  fit_checking: 'Checking for text overflow…',
  rebuilding: 'Rebuilding document…',
  validating: 'Validating…',
  // Pipeline-complete isn't the same as reviewer-approved — these read as
  // "In review" until the reviewer explicitly clicks Validate (see the
  // glossary_validated_at-driven override in the status bar below), which
  // is when they actually mean "Done".
  done: 'In review',
  done_with_warnings: 'In review — some segments need attention',
  failed: 'Failed',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Upload dropzone
// ---------------------------------------------------------------------------

function DocxDropZone({ onFile, disabled }: { onFile: (f: File) => void; disabled: boolean }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.name.toLowerCase().endsWith('.docx')) {
      toast.error('Only .docx files are supported (bilingual aerospace documents)');
      return;
    }
    onFile(f);
  }, [onFile]);

  return (
    <div
      className="rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition-colors"
      style={{
        borderColor: isDragOver ? 'var(--color-accent-primary)' : 'var(--color-border)',
        background: isDragOver ? 'rgba(77,163,232,0.06)' : 'var(--color-bg-secondary)',
        opacity: disabled ? 0.6 : 1,
        pointerEvents: disabled ? 'none' : 'auto',
      }}
      onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={e => {
        e.preventDefault();
        setIsDragOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) handleFile(f);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".docx"
        className="hidden"
        onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
      />
      <Upload className="mx-auto mb-3 h-8 w-8" style={{ color: 'var(--color-text-muted)' }} />
      <p className="text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
        Drop a bilingual .docx here, or click to browse
      </p>
      <p className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
        Aerospace assembly/maintenance instructions with two language sides
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Opt-in: OCR + translate text baked into images embedded in the .docx
// (server/services/translation/docx_images.py). Off by default — each
// selected image costs a vision-LLM call, and not every embedded image
// carries text worth translating (logos, decorative photos).
// ---------------------------------------------------------------------------

interface DocxImageInfo {
  filename: string;
  media_path: string;
  size_bytes: number;
  thumbnail_base64: string;
  hash: string;
  repeat_count: number;
}

// Shared grid: used both at upload time (ImageTranslationPicker) and when
// reviewing a previous selection before a restart (RestartDialog) — bigger
// thumbnails than the very first cut (feedback: "not visible enough"), a
// repeat-count badge (an image byte-identical to N others in the document —
// almost certainly a logo/header graphic reused across pages; OCR only ever
// runs once per unique image regardless of how many are selected, so
// checking every instance of one is still a single LLM call), and
// select-all/none for documents with dozens of images.
function ImageGrid({
  images, selected, onSelectedChange,
}: {
  images: DocxImageInfo[]; selected: Set<string>; onSelectedChange: (s: Set<string>) => void;
}) {
  const toggle = useCallback((filename: string) => {
    const next = new Set(selected);
    if (next.has(filename)) next.delete(filename); else next.add(filename);
    onSelectedChange(next);
  }, [selected, onSelectedChange]);

  return (
    <div className="mt-2">
      <div className="flex items-center justify-between mb-1.5">
        <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          {images.length} image{images.length === 1 ? '' : 's'} found
          {selected.size > 0 && ` — ${selected.size} selected`}:
        </p>
        <div className="flex items-center gap-2">
          <button type="button" onClick={() => onSelectedChange(new Set(images.map(i => i.filename)))}
                  className="text-xs underline" style={{ color: 'var(--color-accent-primary)' }}>
            Select all
          </button>
          <button type="button" onClick={() => onSelectedChange(new Set())}
                  className="text-xs underline" style={{ color: 'var(--color-text-muted)' }}>
            Deselect all
          </button>
        </div>
      </div>
      <div
        className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 max-h-96 overflow-y-auto p-2 rounded-lg"
        style={{ border: '1px solid var(--color-border)' }}
      >
        {images.map(img => {
          const isSel = selected.has(img.filename);
          return (
            <button
              key={img.filename}
              type="button"
              onClick={() => toggle(img.filename)}
              className="relative rounded-md overflow-hidden aspect-square"
              style={{ border: isSel ? '2px solid var(--color-accent-primary)' : '1px solid var(--color-border)' }}
              title={img.filename}
            >
              <img
                src={`data:image/jpeg;base64,${img.thumbnail_base64}`}
                alt={img.filename}
                className="w-full h-full object-cover"
              />
              {img.repeat_count > 1 && (
                <span
                  className="absolute top-1 left-1 text-[10px] font-semibold px-1.5 py-0.5 rounded-full"
                  style={{ background: 'rgba(0,0,0,0.65)', color: '#fff' }}
                  title={`Identical to ${img.repeat_count - 1} other image(s) in this document — one OCR call covers all of them`}
                >
                  ×{img.repeat_count}
                </span>
              )}
              {isSel && (
                <div className="absolute inset-0 flex items-center justify-center" style={{ background: 'rgba(77,163,232,0.35)' }}>
                  <div className="h-6 w-6 rounded-full flex items-center justify-center" style={{ background: 'var(--color-accent-primary)' }}>
                    <Check className="h-4 w-4 text-white" />
                  </div>
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ImageTranslationPicker({
  file, enabled, onEnabledChange, selected, onSelectedChange,
}: {
  file: File; enabled: boolean; onEnabledChange: (v: boolean) => void;
  selected: Set<string>; onSelectedChange: (s: Set<string>) => void;
}) {
  const [images, setImages] = useState<DocxImageInfo[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) return;
    setLoading(true);
    setError(null);
    const fd = new FormData();
    fd.append('file', file);
    fetchJson<{ images: DocxImageInfo[] }>('/api/translate/analyze-images', { method: 'POST', body: fd })
      .then(r => { setImages(r.images); onSelectedChange(new Set()); })
      .catch(e => setError(e instanceof Error ? e.message : 'Could not analyze images'))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, file]);

  return (
    <div className="mb-4">
      <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--color-text-muted)' }}>
        <input
          type="checkbox"
          checked={enabled}
          onChange={e => { onEnabledChange(e.target.checked); if (!e.target.checked) { setImages(null); onSelectedChange(new Set()); } }}
        />
        Also translate text found in images (beta)
      </label>
      {enabled && loading && (
        <p className="text-xs mt-2 flex items-center gap-1.5" style={{ color: 'var(--color-text-muted)' }}>
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Scanning document for images…
        </p>
      )}
      {enabled && error && <p className="text-xs mt-2" style={{ color: '#e5484d' }}>{error}</p>}
      {enabled && images && images.length === 0 && (
        <p className="text-xs mt-2" style={{ color: 'var(--color-text-muted)' }}>No images found in this document.</p>
      )}
      {enabled && images && images.length > 0 && (
        <ImageGrid images={images} selected={selected} onSelectedChange={onSelectedChange} />
      )}
    </div>
  );
}

// Shown when restarting a job from scratch: the previous "translate text in
// images" selection would otherwise be silently dropped (a plain restart
// re-runs _run_job with whatever's passed, defaulting to nothing) or blindly
// reused with no way to check it — this lets the reviewer see the same
// gallery again, pre-checked with their prior choice, and keep or change it.
function RestartDialog({
  jobId, onClose, onConfirm,
}: {
  jobId: number; onClose: () => void; onConfirm: (selectedImages: string[]) => void;
}) {
  const [images, setImages] = useState<DocxImageInfo[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetchJson<{ images: DocxImageInfo[]; previously_selected: string[] }>(`/api/translate/jobs/${jobId}/images`)
      .then(r => {
        setImages(r.images);
        setSelected(new Set(r.previously_selected));
        setEnabled(r.previously_selected.length > 0);
      })
      .catch(e => setError(e instanceof Error ? e.message : 'Could not load images'))
      .finally(() => setLoading(false));
  }, [jobId]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.5)' }}>
      <div
        className="w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-xl p-5"
        style={{ background: 'var(--color-bg-primary)', border: '1px solid var(--color-border)' }}
      >
        <h2 className="text-base font-bold mb-1" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-heading)' }}>
          Restart this job from scratch?
        </h2>
        <p className="text-sm mb-3" style={{ color: 'var(--color-text-muted)' }}>
          This re-runs extraction, language detection and translation on the original upload, and permanently
          deletes all segments, questions and review edits made so far.
        </p>
        {loading && (
          <p className="text-xs flex items-center gap-1.5" style={{ color: 'var(--color-text-muted)' }}>
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading image list…
          </p>
        )}
        {error && <p className="text-xs" style={{ color: '#e5484d' }}>{error}</p>}
        {!loading && !error && images && (
          <div className="mb-2">
            <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--color-text-muted)' }}>
              <input type="checkbox" checked={enabled} onChange={e => setEnabled(e.target.checked)} />
              Also translate text found in images (beta)
            </label>
            {enabled && images.length === 0 && (
              <p className="text-xs mt-2" style={{ color: 'var(--color-text-muted)' }}>No images found in this document.</p>
            )}
            {enabled && images.length > 0 && (
              <ImageGrid images={images} selected={selected} onSelectedChange={setSelected} />
            )}
          </div>
        )}
        <div className="flex items-center justify-end gap-2 mt-4">
          <button
            onClick={onClose}
            className="text-sm px-3 py-1.5 rounded-md"
            style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(enabled ? [...selected] : [])}
            className="text-sm px-3 py-1.5 rounded-md text-white font-medium"
            style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}
          >
            Restart job
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Language pair picker — a plain <select>; add a language by adding one
// entry to the LANGUAGES array above (backend detection support in
// server/services/translation/langdetect.py's LANG_PROFILES is a separate,
// required step — see that module before adding a language here).
// ---------------------------------------------------------------------------

function LanguagePicker({
  label, value, onChange, disabled,
}: { label: string; value: string; onChange: (v: string) => void; disabled: boolean }) {
  return (
    <div>
      <span
        className="block mb-1 text-[11px] font-medium uppercase tracking-wide"
        style={{ color: 'var(--color-text-muted)' }}
      >
        {label}
      </span>
      <select
        disabled={disabled}
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full text-sm rounded-lg px-2.5 py-1.5"
        style={{
          background: 'var(--color-bg-primary)',
          border: '1px solid var(--color-border)',
          color: 'var(--color-text-primary)',
          opacity: disabled ? 0.6 : 1,
        }}
      >
        {LANGUAGES.map(l => (
          <option key={l.value} value={l.value}>{l.label}</option>
        ))}
      </select>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Question / answer panel
// ---------------------------------------------------------------------------

function QuestionsPanel({ job, onSubmitted }: { job: JobState; onSubmitted: () => void }) {
  const [answers, setAnswers] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    for (const q of job.questions) initial[q.q_id] = q.answer ?? q.suggested_answer ?? '';
    return initial;
  });
  const [submitting, setSubmitting] = useState(false);

  const submit = useCallback(async () => {
    setSubmitting(true);
    try {
      await fetchJson(`/api/translate/jobs/${job.id}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          answers: job.questions.map(q => ({ q_id: q.q_id, answer: answers[q.q_id] || '' })),
        }),
      });
      onSubmitted();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to submit answers');
    } finally {
      setSubmitting(false);
    }
  }, [job.id, job.questions, answers, onSubmitted]);

  // Nothing to actually ask (the common case since removing non-actionable
  // "conflict" pseudo-questions) — skip the empty screen and move straight
  // to translation instead of showing "0 clarifying questions" with a
  // pointless button to click.
  useEffect(() => {
    if (job.questions.length === 0) submit();
  }, [job.id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (job.questions.length === 0) {
    return (
      <p className="text-sm flex items-center gap-2" style={{ color: 'var(--color-text-muted)' }}>
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Nothing to clarify — moving on…
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
        {job.questions.length} clarifying question{job.questions.length === 1 ? '' : 's'} before translation can proceed.
      </p>
      {job.questions.map((q, i) => (
        <div
          key={q.q_id}
          className="rounded-lg p-4"
          style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
        >
          <div className="flex items-center gap-2 mb-2">
            <span
              className="text-[10px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded"
              style={{ background: 'rgba(77,163,232,0.12)', color: 'var(--color-accent-primary)' }}
            >
              {q.category.replace(/_/g, ' ')}
            </span>
            <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Question {i + 1}</span>
          </div>
          <p className="text-sm mb-2 whitespace-pre-wrap" style={{ color: 'var(--color-text-primary)' }}>
            {q.question_text}
          </p>
          {q.context && (
            <pre
              className="text-xs mb-2 p-2 rounded whitespace-pre-wrap"
              style={{ background: 'var(--color-bg-primary)', color: 'var(--color-text-muted)' }}
            >
              {q.context}
            </pre>
          )}
          <textarea
            className="w-full text-sm rounded-md p-2"
            style={{ background: 'var(--color-bg-primary)', border: '1px solid var(--color-border)', color: 'var(--color-text-primary)' }}
            rows={2}
            value={answers[q.q_id] || ''}
            onChange={e => setAnswers(a => ({ ...a, [q.q_id]: e.target.value }))}
            placeholder={q.suggested_answer || 'Your answer…'}
          />
        </div>
      ))}
      <button
        onClick={submit}
        disabled={submitting}
        className="px-4 py-2 rounded-lg text-sm font-medium text-white disabled:opacity-60"
        style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}
      >
        {submitting ? 'Submitting…' : 'Submit answers'}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage details — stat cards + mode/language + an actionable residual panel
// ---------------------------------------------------------------------------

interface ResidualItem { seg_id: string; text: string; detected_lang: string | null; confidence: number | null; }
interface FitCheckItem { seg_id: string; severity: string; text: string; }

function StatCard({ label, value, tone }: { label: string; value: string; tone?: 'warn' | 'ok' }) {
  const color = tone === 'warn' ? '#b45309' : tone === 'ok' ? 'var(--color-success)' : 'var(--color-text-primary)';
  return (
    <div className="rounded-lg px-3 py-2" style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}>
      <div className="text-lg font-bold leading-tight" style={{ color, fontFamily: 'var(--font-heading)' }}>{value}</div>
      <div className="text-[11px] mt-0.5" style={{ color: 'var(--color-text-muted)' }}>{label}</div>
    </div>
  );
}

function StageDetails({ jobId, sp, onOpenResidual, compact = false }: {
  jobId: number; sp: Record<string, unknown> | null; onOpenResidual: (segIds: string[]) => void; compact?: boolean;
}) {
  const n = (k: string) => (sp && typeof sp[k] === 'number' ? (sp[k] as number) : null);
  const s = (k: string) => (sp && typeof sp[k] === 'string' ? (sp[k] as string) : null);

  // Reviewer says "this is fine as-is" (e.g. a surname the check keeps
  // flagging) — hidden immediately here; the server excludes it from
  // residual_warnings for good starting with the next rebuild.
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const dismissWarning = useCallback(async (segId: string) => {
    setDismissed(prev => new Set(prev).add(segId));
    try {
      await fetchJson(`/api/translate/jobs/${jobId}/segments/${encodeURIComponent(segId)}/dismiss-warning`, { method: 'POST' });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to dismiss');
    }
  }, [jobId]);

  const cards: { label: string; value: string; tone?: 'warn' | 'ok' }[] = [];
  if (sp) {
    if (n('done') != null && n('total') != null) cards.push({ label: 'strings', value: `${sp.done}/${sp.total}` });
    if (n('translated_count') != null && n('unique_strings') != null)
      cards.push({ label: 'unique strings translated', value: `${sp.translated_count}/${sp.unique_strings}`, tone: 'ok' });
    if ((n('failed_count') ?? 0) > 0) cards.push({ label: 'failed — kept as source', value: `${sp.failed_count}`, tone: 'warn' });
    if ((n('question_count') ?? 0) > 0) cards.push({ label: 'questions', value: `${sp.question_count}` });
    if ((n('conflict_count') ?? 0) > 0) cards.push({ label: 'conflicts', value: `${sp.conflict_count}`, tone: 'warn' });
    if ((n('length_adapt_suggestions') ?? 0) > 0) cards.push({ label: 'auto-shortened', value: `${sp.length_adapt_suggestions}` });
  }

  const mode = s('mode');
  const sourceLang = s('source_lang');
  const targetLang = s('target_lang');
  const residualChecked = !!(sp && Array.isArray(sp.residual_warnings));
  const residual: ResidualItem[] = (residualChecked
    ? (sp!.residual_warnings as unknown[]).filter((w): w is ResidualItem => typeof w === 'object' && w !== null && 'seg_id' in w)
    : []
  ).filter(r => !dismissed.has(r.seg_id));
  const fitCheckItems: FitCheckItem[] = (sp && Array.isArray(sp.fit_check_items)
    ? (sp.fit_check_items as unknown[]).filter((w): w is FitCheckItem => typeof w === 'object' && w !== null && 'seg_id' in w)
    : []
  ).filter(f => !dismissed.has(f.seg_id));
  // The accurate totals (fit_check_items is capped at 30 for storage size).
  const flaggedCount = Math.max(0, (n('fit_check_flagged') ?? 0) - dismissed.size);
  const criticalCount = n('fit_check_critical') ?? 0;

  const hasMeta = mode || (sourceLang && targetLang);
  // Compact mode (used once the job is done and the reviewer is looking at
  // the side-by-side document) drops the pipeline progress metrics/lang
  // chips (done/total counts, unique strings translated) — those stop being
  // useful once the job is finished. Anything that actually flags a quality
  // problem (overflow, conflicts) stays, alongside the residual-language
  // card below — which is shown even when clean, as a standing confirmation
  // rather than disappearing (a narrow post-rebuild check passing isn't the
  // same guarantee as "nothing left to review").
  const warnCards = cards.filter(c => c.tone === 'warn');
  const visibleCards = compact ? warnCards : cards;
  if (!compact && cards.length === 0 && residual.length === 0 && flaggedCount === 0 && !hasMeta) return null;
  if (compact && visibleCards.length === 0 && flaggedCount === 0 && !residualChecked) return null;

  return (
    <div className="mb-4 space-y-3">
      {!compact && hasMeta && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {mode && (
            <span className="px-2 py-1 rounded-md font-medium capitalize" style={{ background: 'rgba(0,85,164,0.10)', color: 'var(--color-accent-primary)' }}>
              {mode} document
            </span>
          )}
          {sourceLang && targetLang && (
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md" style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}>
              <LangBadge code={sourceLang} />
              <ArrowRight className="h-3 w-3" style={{ color: 'var(--color-text-muted)' }} />
              <LangBadge code={targetLang} />
            </span>
          )}
        </div>
      )}

      {visibleCards.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
          {visibleCards.map((c, i) => <StatCard key={i} {...c} />)}
        </div>
      )}

      {residual.length > 0 ? (
        <div className="rounded-lg p-3" style={{ background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.25)' }}>
          <div className="flex items-center justify-between gap-2 mb-2">
            <div className="flex items-center gap-1.5">
              <AlertCircle className="h-4 w-4 flex-shrink-0" style={{ color: '#b45309' }} />
              <span className="text-sm font-medium" style={{ color: '#92400e' }}>
                {residual.length} segment{residual.length === 1 ? '' : 's'} still read as the source language
                even after an automatic re-translation attempt — needs a manual look
              </span>
            </div>
            <button
              onClick={() => onOpenResidual(residual.map(r => r.seg_id))}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded-md flex-shrink-0"
              style={{ background: 'var(--color-bg-primary)', border: '1px solid rgba(245,158,11,0.4)', color: '#92400e' }}
            >
              Review & fix <ArrowRight className="h-3 w-3" />
            </button>
          </div>
          <div className="space-y-1">
            {residual.slice(0, 6).map((r, i) => (
              <div key={i} className="flex items-start gap-2 text-xs">
                <LangBadge code={r.detected_lang} />
                <span className="truncate flex-1" style={{ color: 'var(--color-text-primary)' }} title={r.text}>{r.text}</span>
                <button
                  onClick={() => dismissWarning(r.seg_id)}
                  className="flex-shrink-0 underline"
                  style={{ color: '#92400e' }}
                  title="Not a translation issue (e.g. a proper noun) — stop flagging this segment"
                >
                  Not an issue
                </button>
              </div>
            ))}
            {residual.length > 6 && (
              <p className="text-[11px] pt-0.5" style={{ color: 'var(--color-text-muted)' }}>…and {residual.length - 6} more</p>
            )}
          </div>
        </div>
      ) : residualChecked && compact ? (
        // Standing confirmation rather than silently vanishing — the check
        // is a narrower, post-rebuild XML scan (see check_residual_source_
        // language_bytes) that can miss real cases (e.g. a Czech word using
        // only diacritics shared with other languages); "clean" here means
        // "nothing this specific check caught", not "definitely nothing left".
        <div className="rounded-lg p-2.5 flex items-center gap-1.5" style={{ background: 'rgba(22,163,74,0.06)', border: '1px solid rgba(22,163,74,0.2)' }}>
          <CheckCircle2 className="h-4 w-4 flex-shrink-0" style={{ color: '#15803d' }} />
          <span className="text-xs" style={{ color: '#15803d' }}>
            No segments flagged as still-source-language after rebuild — still worth a look at "Kept (other lang)" in Segments, this check can miss real cases.
          </span>
        </div>
      ) : null}

      {flaggedCount > 0 && (
        <div className="rounded-lg p-3" style={{ background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.25)' }}>
          <div className="flex items-center justify-between gap-2 mb-2">
            <div className="flex items-center gap-1.5">
              <AlertTriangle className="h-4 w-4 flex-shrink-0" style={{ color: '#b45309' }} />
              <span className="text-sm font-medium" style={{ color: '#92400e' }}>
                {flaggedCount} segment{flaggedCount === 1 ? '' : 's'} flagged for text overflow
                {criticalCount > 0 ? ` (${criticalCount} critical)` : ''} — translation may not fit its box
              </span>
            </div>
            <button
              onClick={() => onOpenResidual(fitCheckItems.map(f => f.seg_id))}
              className="flex items-center gap-1 text-xs px-2 py-1 rounded-md flex-shrink-0"
              style={{ background: 'var(--color-bg-primary)', border: '1px solid rgba(245,158,11,0.4)', color: '#92400e' }}
            >
              Review & fix <ArrowRight className="h-3 w-3" />
            </button>
          </div>
          <div className="space-y-1">
            {fitCheckItems.slice(0, 6).map((f, i) => (
              <div key={i} className="flex items-start gap-2 text-xs">
                <span
                  className="text-[10px] font-semibold uppercase tracking-wide px-1 py-0.5 rounded flex-shrink-0"
                  style={{ background: f.severity === 'CRITICAL' ? 'rgba(239,68,68,0.14)' : 'rgba(245,158,11,0.14)', color: f.severity === 'CRITICAL' ? '#b91c1c' : '#b45309' }}
                >
                  {f.severity}
                </span>
                <span className="truncate flex-1" style={{ color: 'var(--color-text-primary)' }} title={f.text}>{f.text}</span>
                <button
                  onClick={() => dismissWarning(f.seg_id)}
                  className="flex-shrink-0 underline"
                  style={{ color: '#92400e' }}
                  title="Fine as-is — stop flagging this segment"
                >
                  Not an issue
                </button>
              </div>
            ))}
            {flaggedCount > fitCheckItems.length && (
              <p className="text-[11px] pt-0.5" style={{ color: 'var(--color-text-muted)' }}>…and {flaggedCount - fitCheckItems.length} more</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Reviewer notes — free-text, saved on blur. Keyed by job.id at the call
// site (like QuestionsPanel) so switching jobs resets it, but the 2.5s poll
// tick refreshing `job` doesn't wipe in-progress typing.
// ---------------------------------------------------------------------------

function JobNotes({ jobId, initialNotes }: { jobId: number; initialNotes: string | null }) {
  const [value, setValue] = useState(initialNotes ?? '');
  const [saving, setSaving] = useState(false);
  const savedRef = useRef(initialNotes ?? '');

  const save = useCallback(async () => {
    if (value === savedRef.current) return;
    setSaving(true);
    try {
      await fetchJson(`/api/translate/jobs/${jobId}/notes`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: value }),
      });
      savedRef.current = value;
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to save notes');
    } finally {
      setSaving(false);
    }
  }, [jobId, value]);

  return (
    <div className="mb-4">
      <span className="block mb-1 text-[11px] font-medium uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
        Reviewer notes{saving ? ' (saving…)' : ''}
      </span>
      <textarea
        className="w-full text-sm rounded-md p-2"
        style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)', color: 'var(--color-text-primary)' }}
        rows={2}
        value={value}
        onChange={e => setValue(e.target.value)}
        onBlur={save}
        placeholder="Notes for this job (visible to anyone reviewing it later)…"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Preview for a finished job — two tabs: the HTML segment review (default,
// change-by-change reading + inline editing, ReviewPanel.tsx) and the
// exact-layout PDF side-by-side (fidelity check, LayoutPreview.tsx), which is
// also where the translated .docx/.pdf downloads live.
// ---------------------------------------------------------------------------

type PreviewTab = 'review' | 'layout';

function PreviewPanel({
  jobId, filename, sourceLang, targetLang, onRebuild, rebuilding, includeReviewComments, onIncludeReviewCommentsChange,
  focusRequest,
}: {
  jobId: number; filename: string; sourceLang: string; targetLang: string;
  onRebuild: () => void; rebuilding: boolean;
  includeReviewComments: boolean; onIncludeReviewCommentsChange: (v: boolean) => void;
  focusRequest?: { segId: string; token: number } | null;
}) {
  const [tab, setTab] = useState<PreviewTab>('review');
  const tabs: { key: PreviewTab; label: string }[] = [
    { key: 'review', label: 'Review' },
    { key: 'layout', label: 'Layout (PDF)' },
  ];

  // "View in document" from the Segments panel always means the Review
  // (diff) tab, even if the reviewer had switched to the Layout PDF tab.
  useEffect(() => {
    if (focusRequest) setTab('review');
  }, [focusRequest]);

  return (
    <div>
      <div className="flex items-center justify-between gap-2 mb-3 flex-wrap">
        <div className="flex gap-1 p-0.5 rounded-lg" style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}>
          {tabs.map(t => {
            const active = tab === t.key;
            return (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className="px-3 py-1.5 rounded-md text-xs font-medium transition-all"
                style={{
                  background: active ? 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' : 'transparent',
                  color: active ? '#fff' : 'var(--color-text-muted)',
                }}
              >
                {t.label}
              </button>
            );
          })}
        </div>
      </div>

      {tab === 'review' ? (
        <ReviewPanel
          jobId={jobId}
          sourceLang={sourceLang}
          targetLang={targetLang}
          onRebuild={onRebuild}
          rebuilding={rebuilding}
          includeReviewComments={includeReviewComments}
          onIncludeReviewCommentsChange={onIncludeReviewCommentsChange}
          focusRequest={focusRequest}
        />
      ) : (
        <LayoutPreview jobId={jobId} filename={filename} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main view
// ---------------------------------------------------------------------------

export function TranslateView() {
  const [sourceLang, setSourceLang] = useState('cs');
  const [targetLang, setTargetLang] = useState('fr');
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<JobState | null>(null);
  const [actionPending, setActionPending] = useState(false);
  const [includeReviewComments, setIncludeReviewComments] = useState(false);
  const [translateImages, setTranslateImages] = useState(false);
  const [pages, setPages] = useState('');
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());
  const [restartDialogOpen, setRestartDialogOpen] = useState(false);
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [segmentsOpen, setSegmentsOpen] = useState(false);
  const [segmentsCategory, setSegmentsCategory] = useState('');
  const [segmentsFocusIds, setSegmentsFocusIds] = useState<string[] | null>(null);
  const [reviewFocus, setReviewFocus] = useState<{ segId: string; token: number } | null>(null);
  const reviewFocusToken = useRef(0);
  // "View in document" from the Segments panel: close the modal and hand
  // the segment off to the Review tab's diff navigation (see ReviewPanel's
  // focusRequest prop) — token bumps on every call so re-clicking the same
  // segment still re-triggers the jump.
  const viewInDocument = useCallback((segId: string) => {
    setSegmentsOpen(false);
    reviewFocusToken.current += 1;
    setReviewFocus({ segId, token: reviewFocusToken.current });
  }, []);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [sharing, setSharing] = useState(false);
  const [downloading, setDownloading] = useState<'docx' | 'pdf' | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Which job the poll loop is allowed to write into state — guards against a
  // late in-flight response from a previous job overwriting the current one.
  const activeJobIdRef = useRef<number | null>(null);
  const [pollBroken, setPollBroken] = useState(false);

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  // Returns a promise for the FIRST tick specifically so callers that just
  // triggered a status-changing action (rebuild/translate) can await it and
  // know job.status already reflects the action before re-enabling their UI —
  // otherwise actionPending clears (and the trigger button re-enables) well
  // before this first refetch lands, leaving a window where job.status still
  // reads the old (pre-action) value and a double-click can re-submit.
  const pollJob = useCallback((id: number): Promise<void> => {
    stopPolling();
    activeJobIdRef.current = id;
    setPollBroken(false);
    // Keeps the job addressable by URL (?job=123) so an F5 or a shared link
    // to this tab returns to the same document instead of the blank upload
    // form — see the mount effect below that reads this back.
    const url = new URL(window.location.href);
    if (url.searchParams.get('job') !== String(id)) {
      url.searchParams.set('job', String(id));
      window.history.pushState({}, '', url);
    }
    let consecutiveFailures = 0;
    const tick = async () => {
      try {
        const j = await fetchJson<JobState>(`/api/translate/jobs/${id}`);
        if (activeJobIdRef.current !== id) return; // user switched jobs meanwhile
        consecutiveFailures = 0;
        setJob(j);
        if (TERMINAL_STATUSES.includes(j.status)) stopPolling();
      } catch (e) {
        if (activeJobIdRef.current !== id) return;
        // Transient network blips must not silently kill the status view —
        // only give up after several consecutive failures, and say so.
        consecutiveFailures += 1;
        if (consecutiveFailures >= 4) {
          stopPolling();
          setPollBroken(true);
          toast.error(e instanceof Error ? e.message : 'Lost contact with the translation job');
        }
      }
    };
    const first = tick();
    pollRef.current = setInterval(tick, POLL_INTERVAL_MS);
    return first;
  }, [stopPolling]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  // Restore the job from the URL on first mount (page refresh, bookmarked
  // link) instead of always landing on the blank upload form.
  useEffect(() => {
    const id = Number(new URLSearchParams(window.location.search).get('job'));
    if (id > 0) pollJob(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const upload = useCallback(async () => {
    if (!file) return;
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('source_lang', sourceLang);
      fd.append('target_lang', targetLang);
      if (translateImages && selectedImages.size > 0) {
        fd.append('selected_images', JSON.stringify([...selectedImages]));
      }
      if (pages.trim()) {
        fd.append('pages', pages.trim());
      }
      const { id } = await fetchJson<{ id: number }>('/api/translate/jobs', { method: 'POST', body: fd });
      pollJob(id);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [file, sourceLang, targetLang, translateImages, selectedImages, pages, pollJob]);

  const runAction = useCallback(async (path: string) => {
    if (!job) return;
    setActionPending(true);
    try {
      await fetchJson(`/api/translate/jobs/${job.id}/${path}`, { method: 'POST' });
      await pollJob(job.id);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Action failed');
    } finally {
      setActionPending(false);
    }
  }, [job, pollJob]);

  const rebuild = useCallback(
    () => runAction(`rebuild?include_review_comments=${includeReviewComments}`),
    [runAction, includeReviewComments],
  );

  const reset = useCallback(() => {
    stopPolling();
    activeJobIdRef.current = null;
    setPollBroken(false);
    setJob(null);
    setFile(null);
    setTranslateImages(false);
    setSelectedImages(new Set());
    setPages('');
    const url = new URL(window.location.href);
    url.searchParams.delete('job');
    window.history.pushState({}, '', url);
  }, [stopPolling]);

  // Redo the whole pipeline on the same job, reusing the original upload —
  // wipes segments/questions/review edits server-side. Goes through
  // RestartDialog rather than a plain confirm() so the previous "translate
  // text in images" selection is never silently dropped nor blindly reused —
  // the reviewer sees the same gallery again and confirms it.
  const confirmRestart = useCallback(async (selectedImages: string[]) => {
    if (!job) return;
    setRestartDialogOpen(false);
    setActionPending(true);
    try {
      await fetchJson(`/api/translate/jobs/${job.id}/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selected_images: selectedImages }),
      });
      await pollJob(job.id);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Action failed');
    } finally {
      setActionPending(false);
    }
  }, [job, pollJob]);

  const share = useCallback(async () => {
    if (!job || sharing) return;
    setSharing(true);
    try {
      const { share_token } = await fetchJson<{ share_token: string }>(
        `/api/translate/jobs/${job.id}/share`, { method: 'POST' },
      );
      const url = `${window.location.origin}/shared/${share_token}`;
      await navigator.clipboard.writeText(url);
      toast.success('Share link copied to clipboard');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to create share link');
    } finally {
      setSharing(false);
    }
  }, [job, sharing]);

  // Top-level "download the deliverable" buttons — the .docx/.pdf downloads
  // otherwise only existed buried inside the Layout (PDF) preview sub-tab,
  // easy to miss since the Review tab (the default) never mentions them.
  const downloadOutput = useCallback(async (format: 'docx' | 'pdf') => {
    if (!job || downloading) return;
    setDownloading(format);
    try {
      const base = job.original_filename.replace(/\.docx$/i, '');
      if (format === 'docx') {
        const { after_docx_base64 } = await fetchJson<{ after_docx_base64: string }>(
          `/api/translate/jobs/${job.id}/preview`,
        );
        const binary = atob(after_docx_base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const blob = new Blob([bytes as BlobPart], {
          type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${base}_translated.docx`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 10_000);
      } else {
        const res = await fetch(`/api/translate/jobs/${job.id}/preview.pdf?side=after`);
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body?.error || res.statusText);
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${base}_translated.pdf`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 10_000);
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Download failed');
    } finally {
      setDownloading(null);
    }
  }, [job, downloading]);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  if (!job) {
    return (
      <div className="max-w-2xl mx-auto py-8 px-4">
        <div className="flex items-start justify-between mb-1">
          <h1 className="text-xl font-bold" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-heading)' }}>
            Translate a bilingual document
          </h1>
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={() => setHistoryOpen(true)}
              className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md"
              style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
            >
              <History className="h-3.5 w-3.5" /> History
            </button>
            <button
              onClick={() => setGlossaryOpen(true)}
              className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md"
              style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
            >
              <BookOpen className="h-3.5 w-3.5" /> Glossary
            </button>
          </div>
        </div>
        <p className="text-sm mb-6" style={{ color: 'var(--color-text-muted)' }}>
          Pick the language you want replaced (source) and the one that should replace it (target);
          the other language side is kept untouched.
        </p>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <LanguagePicker label="Replace this language" value={sourceLang} onChange={setSourceLang} disabled={uploading} />
          <LanguagePicker label="With this language" value={targetLang} onChange={setTargetLang} disabled={uploading} />
        </div>

        {file ? (
          <div
            className="flex items-center justify-between rounded-lg p-3 mb-4"
            style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
          >
            <div className="flex items-center gap-2 min-w-0">
              <FileText className="h-4 w-4 flex-shrink-0" style={{ color: 'var(--color-text-muted)' }} />
              <span className="text-sm truncate" style={{ color: 'var(--color-text-primary)' }}>{file.name}</span>
            </div>
            <button onClick={() => { setFile(null); setTranslateImages(false); setSelectedImages(new Set()); setPages(''); }} disabled={uploading}>
              <X className="h-4 w-4" style={{ color: 'var(--color-text-muted)' }} />
            </button>
          </div>
        ) : (
          <DocxDropZone onFile={setFile} disabled={uploading} />
        )}

        {file && (
          <ImageTranslationPicker
            file={file}
            enabled={translateImages}
            onEnabledChange={setTranslateImages}
            selected={selectedImages}
            onSelectedChange={setSelectedImages}
          />
        )}

        {file && (
          <div className="mt-4">
            <label className="block text-xs font-medium mb-1" style={{ color: 'var(--color-text-muted)' }}>
              Pages to translate (optional)
            </label>
            <input
              type="text"
              value={pages}
              onChange={e => setPages(e.target.value)}
              disabled={uploading}
              placeholder='e.g. "3-10", "1,4,9", or "1-3,7,12-15" — leave blank for the whole document'
              className="w-full px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)', color: 'var(--color-text-primary)' }}
            />
            <p className="text-[11px] mt-1" style={{ color: 'var(--color-text-muted)' }}>
              Pages are approximated from the document's own rendered layout — content outside the
              range is left untouched, never removed.
            </p>
          </div>
        )}

        <button
          onClick={upload}
          disabled={!file || uploading || sourceLang === targetLang}
          className="mt-4 w-full px-4 py-2.5 rounded-lg text-sm font-medium text-white disabled:opacity-50 flex items-center justify-center gap-2"
          style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}
        >
          {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          {uploading ? 'Uploading…' : 'Start translation job'}
        </button>
        {sourceLang === targetLang && (
          <p className="text-xs mt-2" style={{ color: 'var(--color-text-muted)' }}>Source and target must differ.</p>
        )}

        {glossaryOpen && <GlossaryPanel onClose={() => setGlossaryOpen(false)} />}
        <JobHistory open={historyOpen} onClose={() => setHistoryOpen(false)} onLoad={pollJob} />
      </div>
    );
  }

  // The finished job shows a side-by-side document preview — give it the
  // whole viewport; every other stage is a narrow form/status column.
  // Covers both terminal-success statuses: a clean 'done' and
  // 'done_with_warnings' (structurally valid but still needs a human look).
  const isDone = job.status === 'done' || job.status === 'done_with_warnings';
  return (
    <div className={isDone ? 'w-full py-4 px-4' : 'max-w-3xl mx-auto py-8 px-4'}>
      <div className={`flex items-center justify-between ${isDone ? 'mb-2' : 'mb-4'}`}>
        <div className="flex items-center gap-3 min-w-0">
          <button
            onClick={reset}
            title="Back to Translate"
            className="flex items-center justify-center h-8 w-8 rounded-md flex-shrink-0"
            style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
          >
            <ArrowLeft className="h-4 w-4" />
          </button>
          <div className="min-w-0">
            <h1 className="text-lg font-bold truncate" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-heading)' }}>
              {job.original_filename}
            </h1>
            <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
              {job.source_lang.toUpperCase()} → {job.target_lang.toUpperCase()}
              {job.segment_count != null && ` · ${job.segment_count} segments`}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={() => setHistoryOpen(true)}
            className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md"
            style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
          >
            <History className="h-3.5 w-3.5" /> History
          </button>
          {isDone && (
            <>
              <button
                onClick={() => downloadOutput('docx')}
                disabled={downloading !== null}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md disabled:opacity-60"
                style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
                title="Download the translated .docx"
              >
                {downloading === 'docx' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />} .docx
              </button>
              <button
                onClick={() => downloadOutput('pdf')}
                disabled={downloading !== null}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md disabled:opacity-60"
                style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
                title="Download the translated document as PDF"
              >
                {downloading === 'pdf' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />} .pdf
              </button>
              <button
                onClick={() => setFeedbackOpen(true)}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md"
                style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
              >
                <MessageSquare className="h-3.5 w-3.5" /> Feedback
              </button>
              <button
                onClick={share}
                disabled={sharing}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md disabled:opacity-60"
                style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
              >
                <Share2 className="h-3.5 w-3.5" /> {sharing ? 'Sharing…' : 'Share'}
              </button>
            </>
          )}
          {job.segment_count != null && (
            <button
              onClick={() => { setSegmentsCategory(''); setSegmentsFocusIds(null); setSegmentsOpen(true); }}
              className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md"
              style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
            >
              <ListChecks className="h-3.5 w-3.5" /> Segments
            </button>
          )}
          <button
            onClick={() => setGlossaryOpen(true)}
            className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md"
            style={{ color: 'var(--color-text-muted)', border: '1px solid var(--color-border)' }}
          >
            <BookOpen className="h-3.5 w-3.5" /> Glossary
          </button>
          {!ACTIVE_PROCESSING_STATUSES.includes(job.status) && (
            <button
              onClick={() => setRestartDialogOpen(true)}
              disabled={actionPending}
              title="Wipe segments/questions/edits and re-run extraction → translation → rebuild on the original upload"
              className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md disabled:opacity-60"
              style={{ color: '#b45309', border: '1px solid rgba(245,158,11,0.4)' }}
            >
              <RefreshCw className="h-3.5 w-3.5" /> Restart from scratch
            </button>
          )}
        </div>
      </div>

      <div
        className={`flex items-center gap-2 rounded-lg ${isDone ? 'px-3 py-1.5 mb-4' : 'p-3 mb-6'}`}
        style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
      >
        {job.status === 'failed' ? (
          <AlertCircle className="h-4 w-4 flex-shrink-0" style={{ color: '#ef4444' }} />
        ) : isDone && job.glossary_validated_at ? (
          <CheckCircle2 className="h-4 w-4 flex-shrink-0" style={{ color: '#22c55e' }} />
        ) : job.status === 'done_with_warnings' ? (
          <AlertTriangle className="h-4 w-4 flex-shrink-0" style={{ color: '#b45309' }} />
        ) : isDone ? (
          <Eye className="h-4 w-4 flex-shrink-0" style={{ color: 'var(--color-accent-primary)' }} />
        ) : (
          <Loader2 className="h-4 w-4 flex-shrink-0 animate-spin" style={{ color: 'var(--color-accent-primary)' }} />
        )}
        <span className="text-sm" style={{ color: 'var(--color-text-primary)' }}>
          {isDone && job.glossary_validated_at ? 'Done' : (STATUS_LABELS[job.status] || job.status)}
        </span>
        {job.stale && (
          <span className="text-xs ml-auto" style={{ color: '#f59e0b' }}>
            No progress in a while — this job may have stalled.
          </span>
        )}
      </div>

      <StageDetails
        jobId={job.id}
        sp={job.stage_progress}
        onOpenResidual={(ids) => { setSegmentsFocusIds(ids); setSegmentsCategory(''); setSegmentsOpen(true); }}
        compact={isDone}
      />
      <JobNotes key={job.id} jobId={job.id} initialNotes={job.notes} />

      {pollBroken && (
        <div className="flex items-center gap-3 mb-4 text-xs p-2.5 rounded-lg"
             style={{ background: 'rgba(245,158,11,0.08)', color: '#b45309', border: '1px solid rgba(245,158,11,0.25)' }}>
          <span>Lost contact with the server — the job may still be running.</span>
          <button
            onClick={() => pollJob(job.id)}
            className="px-2.5 py-1 rounded-md font-medium"
            style={{ border: '1px solid rgba(245,158,11,0.4)' }}
          >
            Reconnect
          </button>
        </div>
      )}

      {job.status === 'failed' && job.error_msg && (
        <pre
          className="text-xs p-3 rounded-lg mb-4 whitespace-pre-wrap"
          style={{ background: 'rgba(239,68,68,0.08)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.25)' }}
        >
          {job.error_msg}
        </pre>
      )}

      {job.status === 'failed' && (() => {
        const lastStage = typeof job.stage_progress?.stage === 'string' ? (job.stage_progress.stage as string) : '';
        const canRetryTranslate = job.needs_translation_count != null
          && ['translating', 'awaiting_answers', 'answered'].includes(lastStage);
        const canRetryRebuild = lastStage === 'translated'
          || (job.stage_progress != null && 'validation' in job.stage_progress);
        if (!canRetryTranslate && !canRetryRebuild) return null;
        return (
          <div className="flex gap-2 mb-4">
            {canRetryTranslate && (
              <button
                onClick={() => runAction('translate')}
                disabled={actionPending}
                className="px-4 py-2 rounded-lg text-sm font-medium text-white disabled:opacity-60"
                style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}
              >
                {actionPending ? 'Starting…' : 'Retry translation'}
              </button>
            )}
            {canRetryRebuild && (
              <button
                onClick={rebuild}
                disabled={actionPending}
                className="px-4 py-2 rounded-lg text-sm font-medium text-white disabled:opacity-60"
                style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}
              >
                {actionPending ? 'Starting…' : 'Retry rebuild & validate'}
              </button>
            )}
          </div>
        );
      })()}

      {job.status === 'awaiting_answers' && (
        <QuestionsPanel key={job.id} job={job} onSubmitted={() => pollJob(job.id)} />
      )}

      {job.status === 'answered' && (
        <button
          onClick={() => runAction('translate')}
          disabled={actionPending}
          className="px-4 py-2 rounded-lg text-sm font-medium text-white disabled:opacity-60"
          style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}
        >
          {actionPending ? 'Starting…' : `Translate ${job.needs_translation_count ?? ''} segments`}
        </button>
      )}

      {job.status === 'translated' && (
        <div>
          <button
            onClick={rebuild}
            disabled={actionPending}
            className="px-4 py-2 rounded-lg text-sm font-medium text-white disabled:opacity-60"
            style={{ background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)' }}
          >
            {actionPending ? 'Starting…' : 'Rebuild & validate document'}
          </button>
          <label className="flex items-center gap-1.5 mt-2 text-xs cursor-pointer" style={{ color: 'var(--color-text-muted)' }}>
            <input
              type="checkbox"
              checked={includeReviewComments}
              onChange={e => setIncludeReviewComments(e.target.checked)}
            />
            Include review comments in output (Word comments on flagged/conflicting segments)
          </label>
        </div>
      )}

      {isDone && (
        <PreviewPanel
          jobId={job.id}
          filename={job.original_filename}
          sourceLang={job.source_lang}
          targetLang={job.target_lang}
          onRebuild={rebuild}
          rebuilding={actionPending}
          includeReviewComments={includeReviewComments}
          onIncludeReviewCommentsChange={setIncludeReviewComments}
          focusRequest={reviewFocus}
        />
      )}
      {glossaryOpen && <GlossaryPanel onClose={() => setGlossaryOpen(false)} />}
      <JobHistory open={historyOpen} onClose={() => setHistoryOpen(false)} onLoad={pollJob} />
      {feedbackOpen && <FeedbackPanel jobId={job.id} onClose={() => setFeedbackOpen(false)} />}
      {segmentsOpen && (
        <SegmentsPanel
          jobId={job.id}
          sourceLang={job.source_lang}
          initialCategory={segmentsCategory}
          focusSegIds={segmentsFocusIds}
          onClose={() => setSegmentsOpen(false)}
          jobDone={isDone}
          onViewInDocument={isDone ? viewInDocument : undefined}
          glossaryValidatedAt={job.glossary_validated_at}
          onValidate={() => runAction('validate')}
          validating={actionPending}
        />
      )}
      {restartDialogOpen && (
        <RestartDialog jobId={job.id} onClose={() => setRestartDialogOpen(false)} onConfirm={confirmRestart} />
      )}
    </div>
  );
}
