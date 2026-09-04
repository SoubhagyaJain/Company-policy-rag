import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import '@/styles/globals.css';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-jetbrains-mono',
});

export const metadata: Metadata = {
  title: 'Company Policy RAG Portal',
  description:
    'Enterprise RAG Chatbot for company policy, HR guidelines, expense reimbursement, and IT compliance — powered by FastAPI, Hybrid Retrieval, and SSE Streaming.',
  icons: {
    icon: '/favicon.ico',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      // Space UI defaults to dark; the client effect in app/page.tsx reconciles
      // to the user's stored `rag_dark_mode` choice. suppressHydrationWarning
      // covers that post-hydration class change.
      className={`dark ${inter.variable} ${jetbrainsMono.variable}`}
      suppressHydrationWarning
    >
      <body className={`${inter.className} antialiased min-h-screen`}>
        {children}
      </body>
    </html>
  );
}
