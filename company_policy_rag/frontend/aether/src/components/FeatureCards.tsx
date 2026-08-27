import { BarChart3, BookOpen, FileText } from "lucide-react";

interface Props {
  onPolicy: () => void;
  onGuidebook: () => void;
  onEval: () => void;
}

export function FeatureCards({ onPolicy, onGuidebook, onEval }: Props) {
  const cards = [
    {
      icon: FileText,
      title: "Nexus AI",
      desc: "Grounded answers from your employee handbook — leave, benefits, conduct, and compliance.",
      action: "Start Chat",
      onClick: onPolicy,
    },
    {
      icon: BookOpen,
      title: "Guidebook Explorer",
      desc: "Navigate AI agent building blocks, design patterns, tools, and code walkthroughs.",
      action: "Browse Sections",
      onClick: onGuidebook,
    },
    {
      icon: BarChart3,
      title: "Evaluation Studio",
      desc: "Run golden-set evals and track faithfulness, relevancy, and retrieval precision.",
      action: "Run Evaluation",
      onClick: onEval,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-5 max-w-5xl mx-auto mt-4 pb-8 px-8">
      {cards.map((c) => (
        <button
          key={c.title}
          onClick={c.onClick}
          className="text-left bg-sidebar rounded-3xl p-6 border border-white/10 shadow-[0_8px_32px_rgba(28,24,48,0.12)] hover:shadow-[0_0_40px_rgba(168,85,247,0.35)] hover:border-primary/30 transition-all group"
        >
          <div className="w-10 h-10 rounded-2xl bg-primary/20 flex items-center justify-center mb-4 group-hover:bg-primary/30">
            <c.icon className="w-5 h-5 text-highlight" />
          </div>
          <h3 className="text-white font-semibold mb-2">{c.title}</h3>
          <p className="text-white/50 text-sm mb-4">{c.desc}</p>
          <span className="inline-block px-4 py-2 rounded-xl bg-primary/80 group-hover:bg-primary text-white text-xs font-medium">
            {c.action}
          </span>
        </button>
      ))}
    </div>
  );
}