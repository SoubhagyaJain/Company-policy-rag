'use client';

import React, { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { cn } from '../lib/utils';

interface CodeBlockProps {
  language?: string;
  children: string;
}

// Language color & label mapping
function getLanguageMeta(lang?: string): { label: string; color: string } {
  const l = (lang || '').toLowerCase().trim();
  switch (l) {
    case 'python':
    case 'py':
      return { label: 'Python', color: 'text-amber-400 border-amber-500/30 bg-amber-500/10' };
    case 'typescript':
    case 'ts':
      return { label: 'TypeScript', color: 'text-sky-400 border-sky-500/30 bg-sky-500/10' };
    case 'javascript':
    case 'js':
      return { label: 'JavaScript', color: 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10' };
    case 'bash':
    case 'sh':
    case 'shell':
    case 'zsh':
      return { label: 'Terminal / Shell', color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' };
    case 'json':
      return { label: 'JSON', color: 'text-purple-400 border-purple-500/30 bg-purple-500/10' };
    case 'yaml':
    case 'yml':
      return { label: 'YAML', color: 'text-pink-400 border-pink-500/30 bg-pink-500/10' };
    case 'sql':
      return { label: 'SQL', color: 'text-indigo-400 border-indigo-500/30 bg-indigo-500/10' };
    case 'html':
    case 'xml':
      return { label: 'HTML', color: 'text-orange-400 border-orange-500/30 bg-orange-500/10' };
    case 'css':
      return { label: 'CSS', color: 'text-blue-400 border-blue-500/30 bg-blue-500/10' };
    case 'markdown':
    case 'md':
      return { label: 'Markdown', color: 'text-slate-400 border-slate-500/30 bg-slate-500/10' };
    default:
      return { label: l ? l.toUpperCase() : 'Code', color: 'text-terracotta-400 border-terracotta-500/30 bg-terracotta-500/10' };
  }
}

// Lightweight, instant regex-based syntax highlighter for fast streaming render
function highlightLine(line: string, language?: string): React.ReactNode {
  // Comments
  if (/^\s*(#|\/\/|--|\/\*).*/.test(line)) {
    return <span className="text-[#8E877C] italic">{line}</span>;
  }

  // String literals
  const stringRegex = /(".*?"|'.*?'|`.*?`)/g;
  const parts = line.split(stringRegex);

  return (
    <span>
      {parts.map((part, idx) => {
        if (
          (part.startsWith('"') && part.endsWith('"')) ||
          (part.startsWith("'") && part.endsWith("'")) ||
          (part.startsWith('`') && part.endsWith('`'))
        ) {
          return (
            <span key={idx} className="text-[#A8D49B] dark:text-[#A8D49B]">
              {part}
            </span>
          );
        }

        // Keywords & Built-ins
        const tokenized = part.split(/\b/);
        return (
          <React.Fragment key={idx}>
            {tokenized.map((tok, tIdx) => {
              if (
                /^(def|class|return|import|from|as|if|elif|else|for|while|try|except|finally|with|async|await|const|let|var|function|export|default|new|typeof|instanceof|throw|yield|raise|in|is|not|and|or|lambda|pass|break|continue)$/.test(
                  tok
                )
              ) {
                return (
                  <span key={tIdx} className="text-[#F28B82] font-semibold">
                    {tok}
                  </span>
                );
              }
              if (/^(True|False|None|true|false|null|undefined|NaN|self|this)$/.test(tok)) {
                return (
                  <span key={tIdx} className="text-[#D7AEFB]">
                    {tok}
                  </span>
                );
              }
              if (/^\d+(\.\d+)?$/.test(tok)) {
                return (
                  <span key={tIdx} className="text-[#FDD663]">
                    {tok}
                  </span>
                );
              }
              if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(tok) && (line.includes(`${tok}(`) || line.includes(`${tok} `))) {
                if (/^[A-Z][a-zA-Z0-9_]*$/.test(tok)) {
                  return (
                    <span key={tIdx} className="text-[#8AB4F8]">
                      {tok}
                    </span>
                  );
                }
              }
              return tok;
            })}
          </React.Fragment>
        );
      })}
    </span>
  );
}

export function CodeBlock({ language, children }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const rawCode = String(children || '').replace(/\n$/, '');
  const lines = rawCode.split('\n');
  const meta = getLanguageMeta(language);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(rawCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="relative my-3.5 rounded-xl overflow-hidden border border-[#38342F] bg-[#1A1916] text-[#FAF8F5] shadow-md transition-all font-mono text-[13px] group/code">
      {/* Header bar */}
      <div className="flex items-center justify-between px-3.5 py-2 bg-[#24221E] border-b border-[#33302A] select-none">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 opacity-75 mr-1">
            <span className="w-2.5 h-2.5 rounded-full bg-[#E05A47]/80 inline-block" />
            <span className="w-2.5 h-2.5 rounded-full bg-[#E5A83B]/80 inline-block" />
            <span className="w-2.5 h-2.5 rounded-full bg-[#57AB5A]/80 inline-block" />
          </div>
          <span
            className={cn(
              'px-2 py-0.5 rounded text-[11px] font-mono font-medium tracking-wide border',
              meta.color
            )}
          >
            {meta.label}
          </span>
          <span className="text-[11px] text-[#8E877C] font-mono hidden sm:inline-block">
            {lines.length} {lines.length === 1 ? 'line' : 'lines'}
          </span>
        </div>

        {/* Copy Button */}
        <button
          onClick={handleCopy}
          aria-label="Copy code to clipboard"
          className={cn(
            'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono transition-all',
            copied
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
              : 'text-[#B5AFA4] hover:text-[#FAF8F5] bg-[#2E2B26] hover:bg-[#38352F] border border-[#403C35]'
          )}
          title="Copy code snippet"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              <span className="text-[11px] text-emerald-300 font-medium">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5 shrink-0" />
              <span className="text-[11px]">Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code Container with line numbers and horizontal scroll */}
      <div className="overflow-x-auto p-3.5 leading-relaxed scrollbar-thin scrollbar-thumb-[#38352F] scrollbar-track-transparent">
        <pre className="p-0 m-0 bg-transparent border-0 font-mono text-[13px] leading-6">
          <code>
            {lines.map((line, idx) => (
              <div key={idx} className="table-row group/line hover:bg-white/[0.03]">
                <span className="table-cell pr-3.5 select-none text-right text-[11px] text-[#6E675C] font-mono w-7 shrink-0 opacity-60 group-hover/line:text-[#9E978C] group-hover/line:opacity-100 transition-colors">
                  {idx + 1}
                </span>
                <span className="table-cell whitespace-pre font-mono text-[#EDE8DF]">
                  {highlightLine(line, language)}
                </span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}
