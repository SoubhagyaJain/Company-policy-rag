'use client';

/** Chat message. User = moonlit question bubble (right); assistant = a
 *  near-opaque reading surface (left) with reasoning trace, answer, trace pills
 *  and citation chips. Markdown rendering reuses the existing CodeBlock.
 *
 *  Streaming performance: the whole component is memoized (only the message
 *  being streamed re-renders on each token batch), and the answer's markdown is
 *  rendered as independent top-level blocks, so finished paragraphs and code
 *  blocks are parsed once instead of on every token batch. */

import { memo, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import type { ChatMessageData, Citation } from '../../lib/types';
import { CodeBlock } from '../CodeBlock';
import { SpaceThinkingPanel } from './SpaceThinkingPanel';
import { SpaceTracePills } from './SpaceTracePills';
import { SpaceCitationCard } from './SpaceCitationCard';

interface SpaceMessageProps {
  message: ChatMessageData;
  onOpenCitation: (citation: Citation) => void;
}

const markdownComponents = {
  code({ className, children, ...props }: any) {
    const match = /language-(\w+)/.exec(className || '');
    const codeContent = String(children || '').replace(/\n$/, '');
    const isBlock = Boolean(match) || codeContent.includes('\n');
    if (isBlock) {
      return <CodeBlock language={match ? match[1] : undefined}>{codeContent}</CodeBlock>;
    }
    return (
      <code
        className="sp-mono mx-0.5 rounded-md border border-[var(--sp-hairline)] bg-[var(--sp-field-bg)] px-1.5 py-0.5 text-[12.5px] font-medium text-[var(--sp-accent-text)]"
        {...props}
      >
        {children}
      </code>
    );
  },
  pre({ children }: any) {
    return <>{children}</>;
  },
  h1({ children }: any) {
    return <h1 className="sp-heading text-[24px] font-semibold tracking-[-0.025em]">{children}</h1>;
  },
  h2({ children }: any) {
    return <h2 className="sp-heading text-[20px] font-semibold tracking-[-0.02em]">{children}</h2>;
  },
  h3({ children }: any) {
    return <h3 className="sp-heading text-[17px] font-semibold tracking-[-0.01em]">{children}</h3>;
  },
  a({ children, href }: any) {
    return <a href={href} className="text-[var(--sp-accent-text)] underline underline-offset-2">{children}</a>;
  },
  table({ children }: any) {
    return (
      <div className="my-3 overflow-x-auto rounded-xl border border-[var(--sp-hairline)]">
        <table className="min-w-full text-[13px]">{children}</table>
      </div>
    );
  },
  th({ children }: any) {
    return <th className="sp-text border-b border-[var(--sp-hairline)] bg-[var(--sp-card-bg)] px-3 py-2 text-left font-semibold">{children}</th>;
  },
  td({ children }: any) {
    return <td className="sp-muted border-t border-[var(--sp-hairline)] px-3 py-2">{children}</td>;
  },
};

/** Split markdown into top-level blocks at blank lines — never inside a code
 *  fence, and keeping indented continuation lines (nested list content) with the
 *  block they belong to — so each block parses exactly as it would in the whole
 *  document. */
function splitMarkdownBlocks(md: string): string[] {
  const blocks: string[] = [];
  let cur: string[] = [];
  let fence: string | null = null;
  let gap = false;
  for (const line of md.split('\n')) {
    if (fence) {
      cur.push(line);
      const close = /^\s{0,3}(`{3,}|~{3,})\s*$/.exec(line);
      if (close && close[1][0] === fence[0] && close[1].length >= fence.length) fence = null;
      continue;
    }
    if (line.trim() === '') {
      if (cur.length) gap = true;
      continue;
    }
    if (gap) {
      if (/^\s/.test(line)) cur.push('');
      else {
        blocks.push(cur.join('\n'));
        cur = [];
      }
      gap = false;
    }
    const open = /^\s{0,3}(`{3,}|~{3,})/.exec(line);
    if (open) fence = open[1];
    cur.push(line);
  }
  if (cur.length) blocks.push(cur.join('\n'));
  return blocks;
}

const MarkdownBlock = memo(function MarkdownBlock({ source }: { source: string }) {
  return <ReactMarkdown components={markdownComponents}>{source}</ReactMarkdown>;
});

function AnswerMarkdown({ content }: { content: string }) {
  const blocks = useMemo(() => splitMarkdownBlocks(content), [content]);
  return (
    <>
      {blocks.map((b, i) => (
        <MarkdownBlock key={i} source={b} />
      ))}
    </>
  );
}

export const SpaceMessage = memo(function SpaceMessage({ message, onOpenCitation }: SpaceMessageProps) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="sp-rise flex w-full justify-end">
        <div className="sp-question sp-cv max-w-[min(78%,68ch)] rounded-[22px] rounded-br-md px-5 py-4 sm:px-6">
          <p className="sp-message-label mb-1.5">Your question</p>
          <p className="sp-text text-[15.5px] font-medium leading-[1.65]">{message.content}</p>
        </div>
      </div>
    );
  }

  const citations = message.citations || [];

  return (
    <article className="sp-rise flex w-full flex-col gap-3" aria-label="Grounded answer">
      {/* Reasoning trace */}
      {message.thinking_events && message.thinking_events.length > 0 && (
        <SpaceThinkingPanel
          events={message.thinking_events}
          isStreaming={message.isStreaming}
          totalDurationMs={message.trace?.total_latency_ms}
        />
      )}

      {/* Answer — on a glass surface so the generated text stays legible over the hero */}
      {(message.content || message.isStreaming) && (
        <section className={`sp-answer rounded-[26px] rounded-bl-md px-5 py-5 sm:px-8 sm:py-7 ${message.isStreaming ? '' : 'sp-cv'}`}>
          <div className="sp-answer-label mb-4 flex items-center gap-2 border-b border-[var(--sp-answer-rule)] pb-3">
            <span className="sp-moon h-2 w-2" aria-hidden="true" />
            <span>Grounded answer</span>
          </div>
          <div className="sp-answer-body markdown-content max-w-[72ch]">
            <AnswerMarkdown content={message.content} />
            {message.isStreaming && (
              <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse rounded-sm bg-[var(--sp-accent)] align-middle shadow-[0_0_10px_var(--sp-accent-glow)]" />
            )}
          </div>
        </section>
      )}

      {message.error && (
        <p className="sp-warn sp-mono rounded-xl px-3.5 py-2.5 text-[11.5px]" role="alert">
          {message.error}
        </p>
      )}

      {/* Trace pills */}
      {message.trace && <SpaceTracePills trace={message.trace} />}

      {/* Grounding sources */}
      {citations.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <p className="sp-mono sp-faint px-1 text-[9.5px] uppercase tracking-[0.28em]">
            {citations.length} grounding {citations.length === 1 ? 'source' : 'sources'}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {citations.map((c, i) => (
              <SpaceCitationCard key={c.id || i} citation={c} index={i} onOpen={onOpenCitation} />
            ))}
          </div>
        </div>
      )}
    </article>
  );
});

export default SpaceMessage;
