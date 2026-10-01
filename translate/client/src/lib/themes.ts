/**
 * Centralized Theme Definitions
 *
 * All predefined/system themes are defined here.
 * User-created custom themes are stored in localStorage separately.
 */

// Type definitions
export interface ThemeColors {
  // TEXT COLORS - Readability first (WCAG compliant)
  textHeading: string; // Headers, titles (highest contrast)
  textPrimary: string; // Main body text
  textMuted: string; // Secondary text, captions, timestamps

  // BRAND/ACCENT COLOR - Visual identity
  accentPrimary: string; // Main brand color → auto-derives: secondary (lighter/darker), icons, charts, scrollbar

  // ANIMATED BACKGROUND COLOR - Independent from brand
  animatedBgColor: string; // Color for animated network particles (can be different from brand)

  // BACKGROUND COLORS - Surface hierarchy
  bgPrimary: string; // Main background → auto-derives: elevated surfaces
  bgSecondary: string; // Sidebar, panels → auto-derives: inputs, cards, chat assistant bg

  // UI ELEMENTS
  border: string; // Borders, dividers, separators

  // INTERACTION/STATUS COLORS - Semantic meaning
  success: string;
  successHover: string;
  error: string;
  errorHover: string;
  info: string;
  infoHover: string;
  warning: string;
  warningHover: string;
}

export interface Typography {
  primaryFont: string; // Main font for headers, titles
  secondaryFont: string; // Font for body text, paragraphs
}

export interface AnimatedBackgroundSettings {
  particleCount: number; // 20-100 - number of particles
  connectionDistance: number; // 30-100 - max distance for drawing connection lines
  particleOpacity: number; // 0-1 (0.1 increments)
  lineOpacity: number; // 0-1 (0.1 increments)
  particleSize: number; // 0.5-8 (0.5 increments) - wider range for more control
  lineWidth: number; // 0.1-5 (0.1 increments) - finer control
  animationSpeed: number; // 0.1-3 (0.1 increments) - from very slow to fast
}

export interface PredefinedTheme {
  id: string;
  name: string;
  description: string;
  isDefault: boolean; // Marks which theme is the system default
  colors: ThemeColors;
  typography: Typography;
  animatedBackground: AnimatedBackgroundSettings;
}

// All predefined/system themes
export const PREDEFINED_THEMES: PredefinedTheme[] = [
  {
    id: "default",
    name: "Latecoere Light",
    description: "Professional aerospace theme inspired by Latecoere branding",
    isDefault: true,
    colors: {
      // TEXT COLORS - Deep navy tones for corporate feel
      textHeading: "#0C1C3E", // Latecoere deep navy for headings
      textPrimary: "#1A2B4A", // Navy-tinted body text
      textMuted: "#6B7B94", // Muted steel blue for secondary content

      // BRAND/ACCENT - Latecoere corporate blue
      accentPrimary: "#0055A4", // Latecoere blue (French aerospace blue)

      // ANIMATED BACKGROUND
      animatedBgColor: "#0C1C3E", // Deep navy particles

      // BACKGROUNDS - Clean white with cool gray secondary
      bgPrimary: "#FFFFFF", // Pure white for clarity
      bgSecondary: "#F4F6F9", // Cool gray-blue tint

      // UI ELEMENTS - Subtle steel borders
      border: "#D5DBE5", // Cool steel border

      // STATUS COLORS
      success: "#16A34A", // Green-600
      successHover: "#DCFCE7", // Green-100
      error: "#DC2626", // Red-600
      errorHover: "#FEE2E2", // Red-100
      info: "#0055A4", // Latecoere blue
      infoHover: "#DBEAFE", // Blue-100
      warning: "#D97706", // Amber-600
      warningHover: "#FEF3C7", // Amber-100
    },
    typography: {
      primaryFont:
        '"Montserrat", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      secondaryFont:
        '"Open Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    },
    animatedBackground: {
      particleCount: 30,
      connectionDistance: 90,
      particleOpacity: 0.08,
      lineOpacity: 0.3,
      particleSize: 1.2,
      lineWidth: 0.8,
      animationSpeed: 0.6,
    },
  },
  {
    id: "default-dark",
    name: "Latecoere Dark",
    description: "Deep navy aerospace theme for Latecoere",
    isDefault: false,
    colors: {
      // TEXT COLORS - High contrast on deep navy
      textHeading: "#F0F4F8", // Crisp white-blue for headings
      textPrimary: "#CBD5E1", // Slate-300 for readable body text
      textMuted: "#8494A7", // Steel blue for secondary content

      // BRAND/ACCENT - Bright Latecoere blue for dark backgrounds
      accentPrimary: "#3B8DD6", // Lighter corporate blue

      // ANIMATED BACKGROUND - Subtle blue glow
      animatedBgColor: "#1E3A5F", // Deep navy-blue

      // BACKGROUNDS - Deep navy
      bgPrimary: "#0B1424", // Very deep navy
      bgSecondary: "#0F1B2E", // Slightly lighter navy

      // UI ELEMENTS - Subtle navy borders
      border: "#1E2D42", // Dark navy border

      // STATUS COLORS
      success: "#22C55E",
      successHover: "rgba(34, 197, 94, 0.15)",
      error: "#EF4444",
      errorHover: "rgba(239, 68, 68, 0.15)",
      info: "#3B8DD6",
      infoHover: "rgba(59, 141, 214, 0.15)",
      warning: "#F59E0B",
      warningHover: "rgba(245, 158, 11, 0.15)",
    },
    typography: {
      primaryFont:
        '"Montserrat", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      secondaryFont:
        '"Open Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    },
    animatedBackground: {
      particleCount: 30,
      connectionDistance: 90,
      particleOpacity: 0.08,
      lineOpacity: 0.3,
      particleSize: 1.2,
      lineWidth: 0.8,
      animationSpeed: 0.6,
    },
  },
  {
    id: "latecoere-navy",
    name: "Latecoere Navy",
    description: "Full navy immersive aerospace experience",
    isDefault: false,
    colors: {
      // TEXT COLORS
      textHeading: "#FFFFFF",
      textPrimary: "#D1DAE6",
      textMuted: "#8FA3BB",

      // BRAND/ACCENT
      accentPrimary: "#4DA3E8", // Sky blue accent

      // ANIMATED BACKGROUND
      animatedBgColor: "#4DA3E8",

      // BACKGROUNDS
      bgPrimary: "#0A1628", // Deepest navy
      bgSecondary: "#111F36", // Dark navy panel

      // UI ELEMENTS
      border: "#1C2E47",

      // STATUS COLORS
      success: "#34D399",
      successHover: "rgba(52, 211, 153, 0.15)",
      error: "#F87171",
      errorHover: "rgba(248, 113, 113, 0.15)",
      info: "#4DA3E8",
      infoHover: "rgba(77, 163, 232, 0.15)",
      warning: "#FBBF24",
      warningHover: "rgba(251, 191, 36, 0.15)",
    },
    typography: {
      primaryFont:
        '"Montserrat", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      secondaryFont:
        '"Open Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    },
    animatedBackground: {
      particleCount: 35,
      connectionDistance: 85,
      particleOpacity: 0.1,
      lineOpacity: 0.4,
      particleSize: 1.0,
      lineWidth: 0.6,
      animationSpeed: 0.5,
    },
  },
];

// Helper functions
export function getDefaultTheme(): PredefinedTheme {
  const defaultTheme = PREDEFINED_THEMES.find((theme) => theme.isDefault);
  if (!defaultTheme) {
    throw new Error("No default theme found in PREDEFINED_THEMES");
  }
  return defaultTheme;
}

export function getThemeById(themeId: string): PredefinedTheme | undefined {
  return PREDEFINED_THEMES.find((theme) => theme.id === themeId);
}

export function getAllPredefinedThemes(): PredefinedTheme[] {
  return PREDEFINED_THEMES;
}
