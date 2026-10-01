import { useEffect, useState } from 'react';
import { getAppConfig, type AppBranding } from '@/lib/config';

// Single-feature app — no nav tabs (unlike latec-compare's TopBar, which
// switches between Compare/Chat/Translate). Same visual identity otherwise.
export function TopBar() {
  const [branding, setBranding] = useState<AppBranding>({
    name: 'LatLang',
    logo: '/logos/LOGO_LATECOERE.png',
  });

  useEffect(() => {
    getAppConfig().then((cfg) => setBranding(cfg.branding));
  }, []);

  return (
    <header
      className="fixed top-0 left-0 right-0 z-30 h-[var(--header-height)]"
      style={{
        background: 'rgba(12, 28, 62, 0.92)',
        backdropFilter: 'blur(12px) saturate(1.4)',
        WebkitBackdropFilter: 'blur(12px) saturate(1.4)',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        boxShadow: '0 1px 24px rgba(0,0,0,0.25)',
      }}
    >
      <div className="flex items-center h-full px-5 lg:px-8 max-w-7xl mx-auto">
        <div className="flex items-center gap-3 flex-shrink-0">
          <img
            src={branding.logo}
            alt=""
            className="h-7 w-auto object-contain brightness-0 invert opacity-90"
          />
          <span
            className="text-lg font-bold tracking-tight select-none"
            style={{ color: 'rgba(255,255,255,0.95)', fontFamily: 'var(--font-heading)', letterSpacing: '-0.01em' }}
          >
            {branding.name}
          </span>
        </div>
      </div>
    </header>
  );
}
