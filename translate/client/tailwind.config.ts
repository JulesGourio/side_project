import type { Config } from 'tailwindcss';
import typography from '@tailwindcss/typography';

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        border: 'var(--color-border)',
        background: 'var(--color-background)',
        foreground: 'var(--color-foreground)',
      },
      typography: {
        DEFAULT: {
          css: {
            '--tw-prose-body': 'var(--color-text-primary)',
            '--tw-prose-headings': 'var(--color-text-heading)',
            '--tw-prose-bold': 'var(--color-text-heading)',
            '--tw-prose-links': 'var(--color-accent-primary)',
            '--tw-prose-code': 'var(--color-accent-primary)',
            '--tw-prose-bullets': 'var(--color-accent-primary)',
            '--tw-prose-counters': 'var(--color-accent-primary)',
            maxWidth: 'none',
          },
        },
      },
    },
  },
  plugins: [typography],
};

export default config;
