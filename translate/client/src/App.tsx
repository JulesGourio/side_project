import { CustomThemeProvider } from '@/contexts/ThemeContext';
import { TopBar } from '@/components/layout/TopBar';
import { TranslateView } from '@/components/translate/TranslateView';
import { SharedTranslationPage } from '@/components/translate/SharedTranslationPage';

// Single-feature app — no react-router library, no can_translate gating:
// access is enforced at the Databricks App level (workspace group
// permissions), not per-feature in application code. The one exception is
// the read-only /shared/:token page, resolved from the raw pathname below
// (server/app.py's SPA catch-all serves index.html for it either way).
const sharedMatch = window.location.pathname.match(/^\/shared\/([^/]+)/);

export default function App() {
  return (
    <CustomThemeProvider>
      <div
        className="h-screen flex flex-col overflow-hidden"
        style={{ background: 'var(--color-bg-primary)', color: 'var(--color-text-primary)', fontFamily: 'var(--font-body)' }}
      >
        <TopBar />
        <div className="flex-shrink-0 h-[var(--header-height)]" />
        <main className="flex-1 overflow-auto">
          {sharedMatch ? <SharedTranslationPage token={sharedMatch[1]} /> : <TranslateView />}
        </main>
      </div>
    </CustomThemeProvider>
  );
}
