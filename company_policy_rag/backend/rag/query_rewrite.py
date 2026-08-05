from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple
from backend.models.rag import QueryRewriteResult
from backend.utils.logging import logger

_POLICY_TOPIC_EXPANSIONS: List[Tuple[Tuple[str, ...], str]] = [
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

_GUIDEBOOK_TOPIC_EXPANSIONS: List[Tuple[Tuple[str, ...], str]] = [
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

_COMPREHENSIVE_QUERY_PATTERNS: Tuple[re.Pattern[str], ...] = (
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


class QueryRewriter:
    """
    Query normalization, deterministic term expansion, and LLM-based query rewriting.
    """

    def __init__(self, enable_llm_rewrite: bool = True, llm: Optional[Any] = None) -> None:
        self.enable_llm_rewrite = enable_llm_rewrite
        self.llm = llm

    def _query_matches_triggers(self, query: str, expansions: List[Tuple[Tuple[str, ...], str]]) -> bool:
        q_lower = query.lower()
        return any(any(t in q_lower for t in triggers) for triggers, _ in expansions)

    def detect_corpus(self, query: str) -> Optional[str]:
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

    def expand_terms(self, query: str) -> Tuple[str, List[str]]:
        q_lower = query.lower()
        expanded: List[str] = []
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

    def rewrite(self, query: str) -> QueryRewriteResult:
        """Process user query and return QueryRewriteResult."""
        original = query.strip()
        augmented, expanded_terms = self.expand_terms(original)
        is_comp = self.is_comprehensive_list(original)
        inferred_corp = self.detect_corpus(original)

        rewritten = augmented

        if self.enable_llm_rewrite and self.llm is not None:
            try:
                prompt = (
                    "You rewrite questions into concise keyword search queries for a document retrieval system.\n"
                    f"Question: {original}\n"
                    "Search query:"
                )
                response = str(self.llm.complete(prompt)).strip()
                first_line = response.splitlines()[0].strip().strip('"').strip("'")
                if len(first_line) >= 5:
                    rewritten = first_line
            except Exception as exc:
                logger.warning("LLM query rewrite failed (%s). Using augmented query.", exc)

        return QueryRewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            sub_queries=[rewritten],
            expanded_terms=expanded_terms,
            is_comprehensive_list=is_comp,
            inferred_corpus=inferred_corp,
        )
