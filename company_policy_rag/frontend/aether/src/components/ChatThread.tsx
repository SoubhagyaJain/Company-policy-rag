import type { ChatMessage } from "../api/client";
import type { Citation } from "./CitationPanel";
import { FeedbackBar } from "./FeedbackBar";
import { ObservabilityPanel } from "./ObservabilityPanel";
import { ThinkingPanel } from "./ThinkingPanel";

interface Props {
  messages: ChatMessage[];
  loading: boolean;
  model: string;
  corpusScope: string;
  onCitationClick: (citations: Citation[], highlightIndex?: number) => void;
  onToast: (msg: string) => void;
}

export function ChatThread({
  messages,
  loading,
  model,
  corpusScope,
  onCitationClick,
  onToast,
}: Props) {
  if (!messages.length && !loading) return null;

  return (
    <div className="max-w-3xl mx-auto w-full space-y-4 pt-6 pb-4">
      {messages.map((m, i) => {
        if (m.role === "user") {
          return (
            <div key={i} className="flex justify-end">
              <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-br-md bg-violet-200/80 text-sidebar text-sm">
                {m.content}
              </div>
            </div>
          );
        }

        const prevUser = [...messages.slice(0, i)].reverse().find((x) => x.role === "user");
        const citations = (m.citations ?? []) as Citation[];
        const showFeedback = !m.isStreaming && m.content && !m.content.startsWith("Error:");

        return (
          <div key={i} className="flex justify-start">
            <div className="max-w-[90%] px-5 py-4 rounded-2xl rounded-bl-md bg-sidebar text-white/90 text-sm shadow-[0_8px_32px_rgba(28,24,48,0.12)] border border-white/5">
              {((m.thinking_events && m.thinking_events.length > 0) || m.thinking) && (
                <ThinkingPanel
                  events={
                    m.thinking_events ||
                    (m.thinking
                      ? [
                          {
                            id: 'thk_legacy',
                            query_id: '',
                            stage: 'answer_planning',
                            status: 'completed',
                            title: 'Reasoning',
                            summary: m.thinking,
                            duration_ms: 0,
                          },
                        ]
                      : [])
                  }
                  isStreaming={m.isStreaming}
                  reasoningSummary={m.reasoning_summary}
                />
              )}
              <div className="whitespace-pre-wrap leading-relaxed">
                {m.content}
                {m.isStreaming && (
                  <span className="inline-block w-1.5 h-4 ml-0.5 bg-primary/80 animate-pulse align-middle" />
                )}
              </div>
              {m.low_confidence && !m.isStreaming && (
                <p className="mt-2 text-xs text-amber-400/90">Low confidence — verify against sources.</p>
              )}
              {citations.length > 0 && (
                <div className="mt-3 pt-2 border-t border-white/10 flex flex-wrap gap-1 items-center">
                  {citations.slice(0, 5).map((c, j) => (
                    <button
                      key={j}
                      type="button"
                      onClick={() => onCitationClick(citations, j + 1)}
                      className="inline-block px-2 py-0.5 rounded-lg bg-white/10 text-white/60 text-xs hover:bg-white/20 hover:text-white/90 transition-colors"
                    >
                      {String(c.section_path || c.source_file || `Source ${j + 1}`)}
                    </button>
                  ))}
                  {citations.length > 5 && (
                    <button
                      type="button"
                      onClick={() => onCitationClick(citations)}
                      className="text-xs text-white/40 hover:text-white/70 px-1"
                    >
                      +{citations.length - 5} more
                    </button>
                  )}
                </div>
              )}
              <ObservabilityPanel trace={m.retrieval_trace} timing={m.timing} />
              {showFeedback && prevUser && (
                <FeedbackBar
                  messageId={m.message_id}
                  question={prevUser.content}
                  answer={m.content}
                  model={model}
                  corpusScope={corpusScope}
                  onToast={onToast}
                />
              )}
            </div>
          </div>
        );
      })}
      {loading && messages[messages.length - 1]?.role !== "assistant" && (
        <div className="flex gap-2 items-center text-sidebar/50 text-sm py-2">
          <span className="w-2 h-2 rounded-full bg-primary animate-bounce" />
          <span className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.15s]" />
          <span className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.3s]" />
          Retrieving and synthesizing…
        </div>
      )}
    </div>
  );
}