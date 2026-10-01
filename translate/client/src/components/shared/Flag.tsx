// Small, simplified flag glyphs for the intraqual document-language badges.
// Emoji flags render inconsistently across platforms (Windows in particular
// often falls back to bare two-letter codes instead of an actual flag glyph),
// so these are plain inline SVGs — same look everywhere, no font dependency.
// Not pixel-accurate (no coat of arms / stars) — recognisable at ~18px is the
// only goal.

import type { ReactNode } from 'react';

const VIEWBOX = '0 0 20 15';

function Svg({ children, title }: { children: ReactNode; title: string }) {
  return (
    <svg
      viewBox={VIEWBOX}
      width="18"
      height="13.5"
      role="img"
      aria-label={title}
      className="rounded-[2px] flex-shrink-0"
      style={{ boxShadow: '0 0 0 1px rgba(0,0,0,0.12)' }}
    >
      {children}
    </svg>
  );
}

const FLAGS: Record<string, ReactNode> = {
  FR: (
    <Svg title="Français">
      <rect width="20" height="15" fill="#ED2939" />
      <rect width="13.33" height="15" fill="#fff" />
      <rect width="6.67" height="15" fill="#002395" />
    </Svg>
  ),
  // EN and GB never occur on the same document — both mean "English".
  GB: (
    <Svg title="English">
      <rect width="20" height="15" fill="#012169" />
      <path d="M0,0 20,15 M20,0 0,15" stroke="#fff" strokeWidth="3" />
      <path d="M0,0 20,15 M20,0 0,15" stroke="#C8102E" strokeWidth="1" />
      <path d="M10,0 10,15 M0,7.5 20,7.5" stroke="#fff" strokeWidth="5" />
      <path d="M10,0 10,15 M0,7.5 20,7.5" stroke="#C8102E" strokeWidth="3" />
    </Svg>
  ),
  ES: (
    <Svg title="Español (España)">
      <rect width="20" height="15" fill="#AA151B" />
      <rect y="3.75" width="20" height="7.5" fill="#F1BF00" />
    </Svg>
  ),
  MX: (
    <Svg title="Español (México)">
      <rect width="20" height="15" fill="#fff" />
      <rect width="6.67" height="15" fill="#006847" />
      <rect x="13.33" width="6.67" height="15" fill="#CE1126" />
    </Svg>
  ),
  BG: (
    <Svg title="Български">
      <rect width="20" height="15" fill="#D62612" />
      <rect width="20" height="10" fill="#00966E" />
      <rect width="20" height="5" fill="#fff" />
    </Svg>
  ),
  CZ: (
    <Svg title="Čeština">
      <rect width="20" height="15" fill="#D7141A" />
      <rect width="20" height="7.5" fill="#fff" />
      <path d="M0,0 10,7.5 0,15 Z" fill="#11457E" />
    </Svg>
  ),
  BR: (
    <Svg title="Português (Brasil)">
      <rect width="20" height="15" fill="#009C3B" />
      <path d="M10,2 18,7.5 10,13 2,7.5 Z" fill="#FFDF00" />
      <circle cx="10" cy="7.5" r="3.2" fill="#002776" />
    </Svg>
  ),
};
FLAGS.EN = FLAGS.GB; // "EN" and "GB" both just mean English on intraqual REFs

export function Flag({ code, className }: { code: string; className?: string }) {
  const flag = FLAGS[code];
  if (!flag) return null;
  return <span className={className}>{flag}</span>;
}
