/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './hooks/**/*.{js,ts,jsx,tsx,mdx}',
    './lib/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      // Night-alpine palette. The legacy token NAMES (cream/sand/terracotta/
      // charcoal) are kept so the Library and Telemetry views re-skin without
      // touching their markup; the VALUES are now moonlit slate + navy, with
      // "terracotta" repurposed as the moonlit-blue accent.
      colors: {
        cream: {
          50: '#F6F8FC',
          100: '#EEF2F9',
          200: '#E3E9F4',
          300: '#D3DCEC',
          400: '#AEBBD3',
          500: '#8C9AB8',
          600: '#67749A',
          700: '#4A5678',
          800: '#2E3854',
          900: '#161E36',
          950: '#070C1E',
        },
        sand: {
          light: '#EEF2F9',
          border: '#D5DDEB',
          dark: '#0E1630',
          darkBorder: '#223056',
        },
        terracotta: {
          50: '#EEF3FF',
          100: '#DCE6FF',
          300: '#C9D9FF',
          400: '#9DB9FF',
          500: '#5577D9',
          600: '#3F5FC4',
          700: '#2F4BA8',
        },
        charcoal: {
          light: '#4B5675',
          DEFAULT: '#111A33',
          muted: '#5A6684',
          dark: '#0A1128',
        },
        // shadcn/ui semantic tokens for components/ui/*. Values are HSL
        // triples defined in styles/globals.css (light + html.dark), tuned to
        // the night-alpine palette so shadcn primitives match the Space UI.
        border: 'hsl(var(--border) / <alpha-value>)',
        input: 'hsl(var(--input) / <alpha-value>)',
        ring: 'hsl(var(--ring) / <alpha-value>)',
        background: 'hsl(var(--background) / <alpha-value>)',
        foreground: 'hsl(var(--foreground) / <alpha-value>)',
        primary: {
          DEFAULT: 'hsl(var(--primary) / <alpha-value>)',
          foreground: 'hsl(var(--primary-foreground) / <alpha-value>)',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary) / <alpha-value>)',
          foreground: 'hsl(var(--secondary-foreground) / <alpha-value>)',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive) / <alpha-value>)',
          foreground: 'hsl(var(--destructive-foreground) / <alpha-value>)',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted) / <alpha-value>)',
          foreground: 'hsl(var(--muted-foreground) / <alpha-value>)',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent) / <alpha-value>)',
          foreground: 'hsl(var(--accent-foreground) / <alpha-value>)',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover) / <alpha-value>)',
          foreground: 'hsl(var(--popover-foreground) / <alpha-value>)',
        },
        card: {
          DEFAULT: 'hsl(var(--card) / <alpha-value>)',
          foreground: 'hsl(var(--card-foreground) / <alpha-value>)',
        },
      },
      transitionDuration: {
        400: '400ms',
      },
      fontFamily: {
        serif: ['Georgia', 'Cambria', 'Times New Roman', 'serif'],
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'sans-serif',
        ],
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
      backdropBlur: {
        xs: '2px',
        glass: '12px',
      },
      boxShadow: {
        glass: '0 8px 32px 0 rgba(17, 26, 51, 0.06)',
        glassDark: '0 18px 44px -18px rgba(0, 3, 12, 0.8)',
        soft: '0 10px 30px -14px rgba(17, 26, 51, 0.18)',
      }
    },
  },
  plugins: [require('tailwindcss-animate')],
}
