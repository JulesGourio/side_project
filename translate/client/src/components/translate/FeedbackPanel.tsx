import { useEffect, useState } from 'react';
import { ThumbsUp, ThumbsDown, CheckCircle2, Send, X } from 'lucide-react';

interface FeedbackPanelProps {
  jobId: number;
  onClose: () => void;
}

type Vote = 'up' | 'down';

export function FeedbackPanel({ jobId, onClose }: FeedbackPanelProps) {
  const [vote, setVote] = useState<Vote | null>(null);
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async () => {
    if (!vote || submitting) return;
    setSubmitting(true);
    try {
      await fetch('/api/translate/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vote,
          comment: comment.trim() || null,
          job_id: jobId,
        }),
      });
    } catch {
      // best-effort
    } finally {
      setSubmitting(false);
      setSubmitted(true);
    }
  };

  // The confirmation used to have no way out at all — no close button, no
  // auto-dismiss — leaving the dark backdrop stuck on screen forever
  // (reported as "opens a black screen"). Auto-close plus a click-outside
  // fallback below.
  useEffect(() => {
    if (!submitted) return;
    const t = setTimeout(onClose, 1500);
    return () => clearTimeout(t);
  }, [submitted, onClose]);

  if (submitted) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.5)' }} onClick={onClose}>
        <div className="flex items-center gap-2.5 px-5 py-3.5 rounded-2xl border border-[var(--color-success)]/25 bg-[var(--color-success)]/5 text-sm text-[var(--color-success)]">
          <CheckCircle2 className="h-4 w-4 flex-shrink-0" />
          <span className="font-medium">Thank you for your feedback!</span>
        </div>
      </div>
    );
  }

  const accentUp = '#16a34a';
  const accentDown = '#dc2626';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.5)' }} onClick={onClose}>
    <div className="w-full max-w-md rounded-2xl border border-[var(--color-border)]/40 bg-[var(--color-bg-primary)] px-5 py-4 space-y-4" onClick={e => e.stopPropagation()}>
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-medium text-[var(--color-text-heading)]">
          Was this translation helpful?
        </p>
        <button onClick={onClose}><X className="h-4 w-4" style={{ color: 'var(--color-text-muted)' }} /></button>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={() => setVote('up')}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium border transition-all cursor-pointer"
          style={{
            borderColor: vote === 'up' ? accentUp : 'var(--color-border)',
            color: vote === 'up' ? accentUp : 'var(--color-text-muted)',
            background: vote === 'up' ? `${accentUp}10` : 'transparent',
          }}
        >
          <ThumbsUp className="h-4 w-4" />
          Helpful
        </button>

        <button
          onClick={() => setVote('down')}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium border transition-all cursor-pointer"
          style={{
            borderColor: vote === 'down' ? accentDown : 'var(--color-border)',
            color: vote === 'down' ? accentDown : 'var(--color-text-muted)',
            background: vote === 'down' ? `${accentDown}10` : 'transparent',
          }}
        >
          <ThumbsDown className="h-4 w-4" />
          Not helpful
        </button>
      </div>

      {vote && (
        <div className="space-y-3">
          <textarea
            value={comment}
            onChange={e => setComment(e.target.value)}
            placeholder="Add a comment (optional)…"
            rows={3}
            maxLength={1000}
            className="w-full px-3.5 py-2.5 rounded-xl border text-sm resize-none outline-none transition-colors"
            style={{
              borderColor: 'var(--color-border)',
              background: 'var(--color-background)',
              color: 'var(--color-text-body)',
            }}
            onFocus={e => (e.currentTarget.style.borderColor = 'var(--color-accent-primary)')}
            onBlur={e => (e.currentTarget.style.borderColor = 'var(--color-border)')}
          />
          <div className="flex justify-end">
            <button
              onClick={handleSubmit}
              disabled={submitting}
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold text-white transition-all disabled:opacity-50 cursor-pointer"
              style={{
                background: 'linear-gradient(135deg, var(--color-accent-primary) 0%, var(--color-accent-secondary) 100%)',
              }}
            >
              <Send className="h-3.5 w-3.5" />
              {submitting ? 'Sending…' : 'Submit'}
            </button>
          </div>
        </div>
      )}
    </div>
    </div>
  );
}
