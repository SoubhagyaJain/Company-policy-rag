from __future__ import annotations

import re

_BUILDING_BLOCK_SUBQUERIES: tuple[str, ...] = (
    "5 Levels of Agentic AI Systems building blocks overview",
    "six building blocks Role-playing Tools Memory Guardrails Planning",
    "Role-playing building block AI agents",
    "Tools MCP building block AI agents",
    "Memory building block AI agents",
    "Guardrails building block AI agents",
    "Planning building block AI agents",
    "Cooperation Focus Tasks building block AI agents",
)

_MEMORY_TYPE_SUBQUERIES: tuple[str, ...] = (
    "short-term memory agents",
    "long-term memory agents",
    "entity memory agents",
    "episodic semantic procedural memory agents",
)

_DESIGN_PATTERN_SUBQUERIES: tuple[str, ...] = (
    "ReAct agent design pattern",
    "reflection agent pattern",
    "planning pattern agents",
    "tool use agent pattern",
    "multi-agent orchestration design patterns",
)

_SUBAGENT_ROLE_SUBQUERIES: tuple[str, ...] = (
    "research agent orchestration",
    "manager agent sub-agent specialization",
    "sub-agent roles delegation",
)

_CURRENCY_TOOL_SUBQUERIES: tuple[str, ...] = (
    "convert_currency real-time currency conversion tool",
    "currency conversion tool example invocation exchange rate",
    "real-world capability currency tool demonstrate",
)

_CODE_LINKS_SUBQUERIES: tuple[str, ...] = (
    "code is available full code examples guidebook link",
    "Check this code dailydoseofds link repository",
)

_CHECK_THIS_OUT_SUBQUERIES: tuple[str, ...] = (
    "Check this out code walkthrough example snippet",
    "Check this out currency conversion tool",
    "Check this out custom tool MCP",
)


# Cues that mark the start of a genuinely separate question. Used to decide
# whether an "and" joins two questions ("what is X and how do I claim Y") or
# merely two nouns inside one ("terms and conditions", "health and safety").
_QUESTION_START_CUES = (
    "what", "how", "when", "where", "why", "who", "which", "whose",
    "can", "could", "do", "does", "did", "is", "are", "was", "were",
    "should", "would", "will", "may", "must", "am",
    "tell me", "explain", "list", "describe", "summarize", "compare",
    "give me", "show me", "walk me",
)

# Numbered or bulleted parts: "1) ...", "2. ...", "- ...", "(a) ...".
# Users write these inline as often as on their own lines, so a marker is
# accepted mid-sentence too; fragments that do not read as questions are
# discarded afterwards, which keeps prose like "Section 3." from splitting.
_ENUMERATED_PART_PATTERN = re.compile(r"(?:^|\n|\s)\s*(?:\(?[0-9a-d]\)|[1-9][.)]|[-*•])\s+")

_CLAUSE_SPLIT_PATTERN = re.compile(
    r"\s+(?:and\s+also|and\s+then|as\s+well\s+as|and|also|plus)\s+(?=(?:%s)\b)"
    % "|".join(re.escape(cue) for cue in _QUESTION_START_CUES),
    re.IGNORECASE,
)


def _looks_like_question_part(fragment: str) -> bool:
    """A usable standalone part: long enough and headed by an interrogative cue."""
    cleaned = fragment.strip().strip("?.,;: ")
    if len(cleaned.split()) < 3:
        return False
    lowered = cleaned.lower()
    return lowered.startswith(_QUESTION_START_CUES)


def decompose_multi_part(query: str, max_parts: int = 4) -> list[str]:
    """
    Split a message that asks several things into independent sub-questions.

    Returns an empty list for a single question so callers can tell a genuine
    multi-part message ("What is the leave policy and how do I expense travel?")
    from an ordinary one. Splitting is deliberately conservative: a conjunction
    only separates parts when the text after it actually starts a new question.
    """
    text = " ".join((query or "").strip().split())
    if not text:
        return []

    # 1. Enumerated parts win outright when the user numbered them.
    if _ENUMERATED_PART_PATTERN.search(query or ""):
        raw_parts = [p for p in _ENUMERATED_PART_PATTERN.split(query or "") if p and p.strip()]
    else:
        # 2. Otherwise split on hard sentence boundaries, keeping the "?".
        raw_parts = [p for p in re.split(r"(?<=[?;])\s+|\n+", text) if p and p.strip()]

    # 3. Within each sentence, split conjunctions that join two questions.
    parts: list[str] = []
    for raw in raw_parts:
        parts.extend(p for p in _CLAUSE_SPLIT_PATTERN.split(raw) if p and p.strip())

    # 4. Keep only fragments that stand on their own as questions.
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        candidate = part.strip().strip(",;: ").rstrip(".")
        if not _looks_like_question_part(candidate):
            continue
        key = candidate.lower().rstrip("?")
        if key not in seen:
            seen.add(key)
            cleaned.append(candidate)

    if len(cleaned) < 2:
        return []
    return cleaned[:max_parts]


def _split_topic_clause(clause: str) -> list[str]:
    parts = re.split(r",(?![^()]*\))", clause)
    topics: list[str] = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip(" .")
        if len(cleaned) >= 3:
            topics.append(cleaned)
    return topics


class MultiQueryGenerator:
    """
    Decomposes multi-part, comprehensive, or enumeration queries into sub-queries
    targeting specific document sections to maximize Context Recall.
    """

    def generate_subqueries(self, query: str, max_queries: int = 8) -> list[str]:
        """Generate deduplicated list of sub-queries for retrieval aggregation."""
        core = query.strip()
        if not core:
            return []

        queries: list[str] = []
        seen: set[str] = set()

        def add_q(q_text: str) -> None:
            key = q_text.lower()
            if key not in seen:
                seen.add(key)
                queries.append(q_text)

        add_q(core)
        q_lower = core.lower()

        # A message asking several things must retrieve for each part; otherwise
        # the parts after the first are never represented in the candidate pool.
        for part in decompose_multi_part(core):
            add_q(part)

        if re.search(r"guardrails?", q_lower):
            add_q("Guardrails building block AI agents safety constraints")
            add_q("Examples of useful guardrails agents")

        if re.search(r"building\s+blocks?", q_lower):
            for sq in _BUILDING_BLOCK_SUBQUERIES:
                add_q(sq)

        if re.search(r"planning\s+building\s+block", q_lower):
            add_q("Planning building block AI agents 5 levels")

        if re.search(r"how\s+many", q_lower) and "building" in q_lower:
            add_q("six building blocks overview AI agents")

        if re.search(r"types?\s+of\s+memory|memory\s+do\s+agents|memory\s+types?", q_lower):
            for sq in _MEMORY_TYPE_SUBQUERIES:
                add_q(sq)

        if re.search(r"design\s+patterns?|agent\s+patterns?", q_lower) or "popular" in q_lower:
            for sq in _DESIGN_PATTERN_SUBQUERIES:
                add_q(sq)

        if re.search(r"sub-?agents?|orchestration", q_lower) and re.search(r"roles?", q_lower):
            for sq in _SUBAGENT_ROLE_SUBQUERIES:
                add_q(sq)

        if re.search(r"currency|convert_currency|exchange\s+rate|conversion\s+tool", q_lower):
            for sq in _CURRENCY_TOOL_SUBQUERIES:
                add_q(sq)

        if re.search(r"code\s+(is\s+)?available|full\s+code|code\s+example", q_lower):
            for sq in _CODE_LINKS_SUBQUERIES:
                add_q(sq)

        if re.search(r"check\s+this\s+out|code\s+walkthrough", q_lower):
            for sq in _CHECK_THIS_OUT_SUBQUERIES:
                add_q(sq)

        if re.search(r"custom\s+tool|build\s+custom", q_lower):
            add_q("custom tools MCP function implementation")

        if re.search(r"manager\s+agent", q_lower):
            add_q("manager agent coordinates sub-agents multi-agent pattern")

        attention_match = re.search(
            r"(?:pay\s+special\s+attention\s+to|including|covering|focus\s+on)\s+(.+)",
            core,
            re.IGNORECASE,
        )
        if attention_match:
            for topic in _split_topic_clause(attention_match.group(1)):
                add_q(topic)

        return queries[:max_queries]
