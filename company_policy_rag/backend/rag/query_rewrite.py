from __future__ import annotations

import re
from typing import Any, Dict

from backend.models.rag import QueryRewriteResult
from backend.utils.logging import logger

_POLICY_TOPIC_EXPANSIONS: list[tuple[tuple[str, ...], str]] = [
    (
        ("health benefit", "health insurance", "benefits", "eligible for health", "enrollment"),
        "health insurance medical dental vision eligibility enrollment waiting period 30 days",
    ),
    (
        ("resign", "resignation", "quit", "notice when", "two weeks", "give notice"),
        "employment at-will termination separation resignation notice period",
    ),
    (
        ("dress code", "attire", "grooming", "appearance"),
        "dress code appearance grooming professional attire",
    ),
    (
        ("confidential", "trade secret", "proprietary"),
        "confidential trade secret proprietary non-disclosure",
    ),
    (
        ("disciplinary", "discipline", "policy violation", "corrective action"),
        "disciplinary action corrective action termination violation investigation report supervisor",
    ),
    (
        ("second job", "outside consulting", "moonlight", "outside employment", "consulting while"),
        "outside employment moonlighting conflict of interest electronic communications ethics approval",
    ),
]

_GUIDEBOOK_TOPIC_EXPANSIONS: list[tuple[tuple[str, ...], str]] = [
    (
        ("building block", "building blocks", "six building"),
        "Role-playing Focus Tasks Tools Cooperation Guardrails Planning Memory six AI agents",
    ),
    (
        ("types of memory", "memory do agents", "memory types", "memory agents use"),
        "short-term long-term entity episodic semantic procedural memory",
    ),
    (
        ("design pattern", "design patterns", "agent pattern", "most popular"),
        "ReAct reflection planning tool use multi-agent orchestration design patterns",
    ),
    (
        ("sub-agent", "sub-agents", "subagent", "orchestration roles", "roles can"),
        "manager specialist research summarization delegation orchestration sub-agent",
    ),
    (
        ("currency", "convert_currency", "exchange rate", "real-world capability", "conversion tool"),
        "convert_currency real-time currency conversion exchange rate tool invocation",
    ),
    (
        ("custom tool", "build custom", "create tool", "tool for an agent"),
        "custom tools MCP function implementation agent tools building block",
    ),
    (
        ("code is available", "full code", "code examples", "where does the guidebook point"),
        "code is available Check this code dailydoseofds link repository",
    ),
    (
        ("check this out", "code walkthrough", "walkthrough"),
        "Check this out code walkthrough example snippet",
    ),
    (
        ("guardrails", "guardrail"),
        "Guardrails building block safety constraints limits validation checkpoints",
    ),
    (
        ("planning building block",),
        "Planning building block six building blocks 5 Levels subdividing tasks",
    ),
    (
        ("manager agent", "multi-agent setup"),
        "manager agent coordinates sub-agents multi-agent pattern",
    ),
    (
        ("rag", "agent workflow"),
        "Agentic RAG retriever agent workflow vector DB context",
    ),
]

_COMPREHENSIVE_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blist\b.+\bexplain\b", re.IGNORECASE),
    re.compile(r"\blist\b", re.IGNORECASE),
    re.compile(r"\bbuilding\s+blocks?\b", re.IGNORECASE),
    re.compile(r"\bpay\s+special\s+attention\b", re.IGNORECASE),
    re.compile(r"\bfor\s+each\b", re.IGNORECASE),
    re.compile(r"\ball\s+\d+\b", re.IGNORECASE),
    re.compile(r"\btypes?\s+of\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+are\s+the\b", re.IGNORECASE),
    re.compile(r"\bhow\s+many\b", re.IGNORECASE),
    re.compile(r"\broles?\s+can\b", re.IGNORECASE),
    re.compile(r"\bdesign\s+patterns?\b", re.IGNORECASE),
    re.compile(r"\bmost\s+popular\b", re.IGNORECASE),
    re.compile(r"\bguardrails?\b", re.IGNORECASE),
    re.compile(r"\bplanning\s+building\s+block\b", re.IGNORECASE),
)


_REFERENTIAL_PRONOUNS_PATTERN: re.Pattern[str] = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|his|her|he|she|same|else|another)\b",
    re.IGNORECASE,
)

_REFERENTIAL_PHRASES_PATTERN: re.Pattern[str] = re.compile(
    r"\b(what\s+about|how\s+about|tell\s+me\s+more|explain\s+more|more\s+details|how\s+long|where\s+is|why\s+is|how\s+so|what\s+else|any\s+exceptions?|does\s+it|can\s+you\s+elaborate)\b",
    re.IGNORECASE,
)


def _format_history_for_rewrite(history: list[Dict[str, Any]], max_turns: int = 4) -> str:
    recent = history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class QueryRewriter:
    """
    Query normalization, deterministic term expansion, and LLM-based query rewriting.
    """

    def __init__(self, enable_llm_rewrite: bool = True, llm: Any | None = None) -> None:
        self.enable_llm_rewrite = enable_llm_rewrite
        self.llm = llm

    def _query_matches_triggers(self, query: str, expansions: list[tuple[tuple[str, ...], str]]) -> bool:
        q_lower = query.lower()
        return any(any(t in q_lower for t in triggers) for triggers, _ in expansions)

    def _is_followup_query(self, query: str) -> bool:
        """
        Determines if a query is a follow-up question referencing previous context
        by checking for pronouns, referential phrases, or short implicit queries.
        """
        q_lower = query.strip().lower()
        if not q_lower:
            return False

        if _REFERENTIAL_PRONOUNS_PATTERN.search(q_lower):
            return True

        if _REFERENTIAL_PHRASES_PATTERN.search(q_lower):
            return True

        words = q_lower.split()
        if len(words) <= 3:
            if not (
                self._query_matches_triggers(query, _POLICY_TOPIC_EXPANSIONS)
                or self._query_matches_triggers(query, _GUIDEBOOK_TOPIC_EXPANSIONS)
            ):
                return True

        return False

    def detect_corpus(self, query: str) -> str | None:
        q_lower = query.lower()
        if any(w in q_lower for w in ("vacation", "pto", "sick leave", "resignation", "at-will")):
            return "policy"
        if any(w in q_lower for w in ("building block", "agent", "sub-agent", "convert_currency", "mcp")):
            return "guidebook"
        if self._query_matches_triggers(query, _GUIDEBOOK_TOPIC_EXPANSIONS):
            return "guidebook"
        if self._query_matches_triggers(query, _POLICY_TOPIC_EXPANSIONS):
            return "policy"
        return None

    def expand_terms(self, query: str) -> tuple[str, list[str]]:
        q_lower = query.lower()
        expanded: list[str] = []
        for triggers, terms in _POLICY_TOPIC_EXPANSIONS + _GUIDEBOOK_TOPIC_EXPANSIONS:
            if any(t in q_lower for t in triggers):
                expanded.append(terms)

        if not expanded:
            return query, []

        augmented_query = f"{query} {' '.join(expanded)}"
        return augmented_query, expanded

    def is_comprehensive_list(self, query: str) -> bool:
        text = query.strip()
        if len(text) < 20:
            return False
        return any(pattern.search(text) for pattern in _COMPREHENSIVE_QUERY_PATTERNS)

    def _fallback_rewrite(
        self,
        original: str,
        augmented: str,
        history: list[Dict[str, Any]] | None,
    ) -> str:
        """
        Robust non-LLM query rewrite fallback for multi-turn dialogues.
        Extracts the root topic (first user query) and recent user queries from history
        ONLY when original query is identified as a follow-up.
        """
        if not history or not self._is_followup_query(original):
            return augmented

        user_queries = [
            str(msg.get("content", "")).strip()
            for msg in history
            if msg.get("role") == "user" and str(msg.get("content", "")).strip()
        ]
        if not user_queries:
            return augmented

        root_query = user_queries[0]
        last_query = user_queries[-1]

        if len(user_queries) == 1 or root_query == last_query:
            context_str = root_query
        else:
            context_str = f"{root_query} {last_query}"

        return f"{augmented} {context_str}"

    def rewrite(
        self,
        query: str,
        history: list[Dict[str, Any]] | None = None,
        llm: Any | None = None,
    ) -> QueryRewriteResult:
        """Process user query and return QueryRewriteResult, considering conversation history if available."""
        original = query.strip()
        augmented, expanded_terms = self.expand_terms(original)
        is_comp = self.is_comprehensive_list(original)
        inferred_corp = self.detect_corpus(original)

        rewritten = augmented
        effective_llm = llm or self.llm

        if self.enable_llm_rewrite and effective_llm is not None:
            try:
                if history and len(history) > 0:
                    history_str = _format_history_for_rewrite(history)
                    prompt = (
                        "Given the following conversation history and follow-up question, rewrite the follow-up question into a standalone, clear keyword search query for document retrieval. Resolve all pronouns (such as 'it', 'that', 'they', 'what about...') into specific topic terms.\n\n"
                        f"Conversation History:\n{history_str}\n\n"
                        f"Follow-up Question: {original}\n"
                        "Standalone Search Query:"
                    )
                else:
                    prompt = (
                        "You rewrite questions into concise keyword search queries for a document retrieval system.\n"
                        f"Question: {original}\n"
                        "Search query:"
                    )
                response = str(effective_llm.complete(prompt)).strip()
                first_line = response.splitlines()[0].strip().strip('"').strip("'")
                if len(first_line) >= 3:
                    rewritten = first_line
            except Exception as exc:
                logger.warning("LLM query rewrite failed (%s). Using fallback query rewrite.", exc)
                rewritten = self._fallback_rewrite(original, augmented, history)
        elif history and len(history) > 0:
            rewritten = self._fallback_rewrite(original, augmented, history)

        return QueryRewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            sub_queries=[rewritten],
            expanded_terms=expanded_terms,
            is_comprehensive_list=is_comp,
            inferred_corpus=inferred_corp,
        )


