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
      colors: {
        cream: {
          50: '#FAF9F5',
          100: '#F3F0E6',
          200: '#EFECE2',
          300: '#E8E4D8',
          400: '#D9D3C5',
          500: '#C5BDAB',
          600: '#A39985',
          700: '#7E7563',
          800: '#524B3E',
          900: '#2D2821',
          950: '#141413',
        },
        sand: {
          light: '#F3F0E6',
          border: '#E5E0D8',
          dark: '#1F1E1B',
          darkBorder: '#2A2925',
        },
        terracotta: {
          50: '#FFF7ED',
          100: '#FFEDD5',
          500: '#D97706',
          600: '#C85A32',
          700: '#B45309',
        },
        charcoal: {
          light: '#66635B',
          DEFAULT: '#1A1A1A',
          muted: '#8C887B',
          dark: '#121211',
        }
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
        glass: '0 8px 32px 0 rgba(0, 0, 0, 0.04)',
        glassDark: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
        soft: '0 4px 20px -2px rgba(26, 26, 26, 0.05)',
      }
    },
  },
  plugins: [],
}
