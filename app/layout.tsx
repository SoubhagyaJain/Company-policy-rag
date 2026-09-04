import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Earth Assistant",
  description: "Interactive 3D Earth globe with chat interface",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
