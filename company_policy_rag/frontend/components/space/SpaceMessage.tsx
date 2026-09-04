'use client';

/** Space-styled chat message. User = glass bubble (right); assistant = open
 *  markdown column (left) with reasoning trace, answer, trace pills and citation
 *  chips. Markdown rendering reuses the existing CodeBlock. */

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
    return <h1 className="sp-heading mt-1 text-[19px] font-semibold tracking-tight">{children}</h1>;
  },
  h2({ children }: any) {
    return <h2 className="sp-heading mt-1 text-[16.5px] font-semibold tracking-tight">{children}</h2>;
  },
  h3({ children }: any) {
    return <h3 className="sp-heading text-[14.5px] font-semibold">{children}</h3>;
  },
  a({ children, href }: any) {
    return <a href={href} className="text-[var(--sp-accent-text)] underline underline-offset-2">{children}</a>;
  },
  table({ children }: any) {
    return (
      <div className="my-3 overflow-x-auto rounded-lg border border-[var(--sp-hairline)]">
        <table className="min-w-full text-xs">{children}</table>
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

export function SpaceMessage({ message, onOpenCitation }: SpaceMessageProps) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div className="flex w-full justify-end">
        <div className="sp-comp sp-text max-w-[80%] rounded-3xl rounded-br-lg px-4 py-3 text-[14.5px] leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  const citations = message.citations || [];

  return (
    <div className="flex w-full flex-col gap-3">
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
        <div className="sp-answer sp-text markdown-content max-w-none space-y-3 rounded-3xl rounded-bl-lg px-4 py-3.5 text-[14.5px] leading-relaxed">
          <ReactMarkdown components={markdownComponents}>{message.content}</ReactMarkdown>
          {message.isStreaming && (
            <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse rounded-sm bg-[var(--sp-accent)] align-middle" />
          )}
        </div>
      )}

      {message.error && (
        <p className="sp-mono rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-[11.5px] text-amber-300">
          {message.error}
        </p>
      )}

      {/* Trace pills */}
      {message.trace && <SpaceTracePills trace={message.trace} />}

      {/* Grounding sources */}
      {citations.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <p className="sp-mono sp-faint text-[9px] uppercase tracking-[0.3em]">
            {citations.length} grounding {citations.length === 1 ? 'source' : 'sources'}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {citations.map((c, i) => (
              <SpaceCitationCard key={c.id || i} citation={c} index={i} onOpen={onOpenCitation} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default SpaceMessage;
