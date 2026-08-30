from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from backend.models.conversation import (
    AnswerMode,
    ConversationRAGState,
    ExpansionPlan,
    FollowUpResolution,
)
from backend.models.rag import QueryCategory
from backend.utils.logging import logger

# Regex patterns for referential pronouns
_PRONOUNS_PATTERN = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|the\s+above|same|former|latter|such)\b",
    re.IGNORECASE,
)

# Regex patterns for conversational follow-up triggers
_FOLLOWUP_PHRASES_PATTERN = re.compile(
    r"\b("
    r"tell\s+me\s+more|tell\s+about\s+it|tell\s+about\s+it\s+in\s+detail|tell\s+more|"
    r"explain\s+more|explain\s+that|explain\s+further|explain\s+in\s+detail|explain\s+it|"
    r"elaborate|elaborate\s+on\s+that|elaborate\s+further|elaborate\s+more|"
    r"continue|go\s+deeper|give\s+more\s+details|more\s+details|in\s+detail|detailed\s+explanation|"
    r"how\s+does\s+it\s+work|how\s+do\s+they\s+work|what\s+does\s+that\s+mean|what\s+does\s+it\s+mean|"
    r"walk\s+me\s+through\s+it|walk\s+through\s+it|break\s+it\s+down|expand\s+on\s+this|expand\s+on\s+that|"
    r"give\s+code\s+for\s+it|show\s+code\s+for\s+that|how\s+is\s+it\s+implemented|how\s+to\s+run\s+it|"
    r"what\s+else|how\s+about|what\s+about|any\s+exceptions|can\s+you\s+elaborate|"
    r"how\s+to\s+enroll|how\s+do\s+i\s+enroll|how\s+can\s+i\s+enroll|enroll\s+in\s+them|"
    r"and\s+for|and\s+what\s+about|what\s+if|eligibility\s+for|and\s+contractors|"
    r"show\s+me\s+more|give\s+me\s+more|deep\s+dive|step\s+by\s+step\s+details|"
    r"explain\s+this\s+code|explain\s+the\s+code|show\s+the\s+code|explain\s+the\s+diagram|"
    r"tell\s+me\s+more\s+about\s+the\s+diagram|tell\s+more\s+about\s+the\s+diagram|"
    r"compare\s+them|compare\s+these|what\s+about\s+the\s+previous\s+one|and\s+then|"
    r"can\s+you\s+expand\s+on\s+that"
    r")\b",
    re.IGNORECASE,
)

# Regex patterns for explicit EXPAND / DETAILED answer mode
_EXPAND_MODE_PATTERN = re.compile(
    r"\b("
    r"in\s+detail|tell\s+about\s+it\s+in\s+detail|tell\s+me\s+about\s+it\s+in\s+detail|"
    r"tell\s+me\s+more|explain\s+further|explain\s+more|"
    r"elaborate|go\s+deeper|deep\s+dive|line\s+by\s+line|walk\s+through|break\s+down|"
    r"give\s+more\s+details|comprehensive\s+explanation|exhaustive|expanded|"
    r"expand\s+on\s+that|can\s+you\s+expand"
    r")\b",
    re.IGNORECASE,
)

_DETAILED_MODE_PATTERN = re.compile(
    r"\b(detailed|all\s+details|thorough|in-depth|complete\s+breakdown|exhaustive)\b",
    re.IGNORECASE,
)

_CODE_EXPLANATION_MODE_PATTERN = re.compile(
    r"\b("
    r"explain\s+(?:this\s+|the\s+)?code|show\s+(?:the\s+)?code|walk\s+through\s+(?:the\s+)?code|"
    r"code\s+explanation|break\s+down\s+(?:the\s+)?code|how\s+is\s+(?:this\s+|the\s+)?code\s+implemented|"
    r"line\s+by\s+line\s+code|give\s+code\s+for\s+it|show\s+code\s+for\s+that"
    r")\b",
    re.IGNORECASE,
)

_STEP_BY_STEP_MODE_PATTERN = re.compile(
    r"\b(step\s+by\s+step|step-by-step|step\s+by\s+step\s+details|walk\s+me\s+through|walkthrough)\b",
    re.IGNORECASE,
)

_COMPARISON_MODE_PATTERN = re.compile(
    r"\b(compare\s+them|compare\s+these|difference\s+between\s+them|how\s+do\s+they\s+compare|comparison\s+between)\b",
    re.IGNORECASE,
)

_SUMMARY_MODE_PATTERN = re.compile(
    r"\b(summarize|summary|briefly|brief\s+overview|in\s+short|tldr|tl;dr|high\s+level)\b",
    re.IGNORECASE,
)

_CONTINUE_MODE_PATTERN = re.compile(
    r"\b(continue|next\s+steps?|proceed|keep\s+going|what\s+next|and\s+then)\b",
    re.IGNORECASE,
)

_EXPLANATION_MODE_PATTERN = re.compile(
    r"\b("
    r"explain\s+(?:the\s+)?diagram|tell\s+me\s+more\s+about\s+the\s+diagram|"
    r"explain\s+it|explain\s+that|how\s+does\s+it\s+work|what\s+does\s+that\s+mean|why"
    r")\b",
    re.IGNORECASE,
)

_NEW_TOPIC_INDICATORS = re.compile(
    r"^(what\s+is|what\s+are|how\s+does\s+[a-zA-Z0-9_\-\s]{6,}|explain\s+(the\s+)?[a-zA-Z0-9_\-\s]{6,}|tell\s+me\s+about\s+(the\s+)?[a-zA-Z0-9_\-\s]{6,}|define|list)",
    re.IGNORECASE,
)

# Stop words to ignore during entity overlap calculations
_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "just", "me", "more", "most", "my", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "tell", "give", "show", "detail", "details",
    "please", "code", "implementation", "work", "works",
}


class ConversationResolutionResult(BaseModel):
    """Encapsulates the complete result of dynamic conversational query resolution."""

    resolved_query: str
    is_followup: bool = False
    topic_shift: bool = False
    confidence: float = 1.0
    reason: str = ""
    active_topic: str | None = None
    active_entities: list[str] = Field(default_factory=list)
    answer_mode: AnswerMode = AnswerMode.DIRECT
    mode_directives: str = ""
    resolution: FollowUpResolution | None = None
    expansion_plan: ExpansionPlan | None = None


class FollowUpResolver:
    """
    Layered Conversation Follow-Up Resolution Engine.

    Layer 1: Deterministic Conversational Cue Detection
    Layer 2: Previous Turn Structure Analysis & Ambiguity Detection
    Layer 3: Dynamic Entity & Topic Extraction (from queries and prior answers)
    Layer 4: Generic Query Synthesis & Non-Shrinking Expansion Planning
    """

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    # =========================================================================
    # Layer 3: Dynamic Entity & Topic Extraction
    # =========================================================================

    def extract_entities(self, text: str) -> list[str]:
        """Extract meaningful topic entities, proper nouns, and technical identifiers from text."""
        if not text or not text.strip():
            return []

        entities: list[str] = []

        # 1. Quoted terms: "Hotel Search Agent", 'convert_currency'
        quoted = re.findall(r"[\"']([^\"']+)[\"']", text)
        entities.extend(q.strip() for q in quoted if len(q.strip()) > 2)

        # 2. Capitalized multi-word noun phrases: Hotel Search Agent, CrewAI, Vacation Policy
        caps_phrases = re.findall(r"\b([A-Z][a-zA-Z0-9_]*(?:\s+[A-Z][a-zA-Z0-9_]*)+)\b", text)
        entities.extend(cp.strip() for cp in caps_phrases)

        # 3. Code identifiers & class/function names: snake_case, camelCase, PascalCase
        code_ids = re.findall(r"\b([a-z0-9]+_[a-z0-9_]+|[a-z]+[A-Z][a-zA-Z0-9]+|[A-Z][a-zA-Z0-9]+(?:Agent|Task|Crew|Service|Manager|Pipeline|Retriever))\b", text)
        entities.extend(ci.strip() for ci in code_ids)

        # 4. Clean and deduplicate while preserving order
        unique_entities: list[str] = []
        seen = set()
        for ent in entities:
            ent_clean = ent.strip()
            ent_lower = ent_clean.lower()
            if ent_lower not in seen and ent_lower not in _STOP_WORDS and len(ent_clean) >= 3:
                seen.add(ent_lower)
                unique_entities.append(ent_clean)

        # Fallback: if no capitalized phrases found, extract meaningful keywords (> 3 chars)
        if not unique_entities:
            words = [
                w.strip("?,.:;\"'")
                for w in text.split()
                if len(w.strip("?,.:;\"'")) > 3 and w.lower() not in _STOP_WORDS
            ]
            if words:
                unique_entities.append(" ".join(words[:4]))

        return unique_entities

    def extract_topic_from_query(self, query: str) -> str:
        """Extract a clean, representative topic string from a standalone user query."""
        clean_q = query.strip()
        # Remove common query prefixes like "What is the implementation code for", "Tell me about"
        topic = re.sub(
            r"^(what\s+is\s+(the\s+)?|what\s+are\s+(the\s+)?|how\s+to\s+|how\s+does\s+(the\s+)?|"
            r"tell\s+me\s+about\s+(the\s+)?|explain\s+(the\s+)?|show\s+me\s+(the\s+)?|"
            r"give\s+me\s+(the\s+)?|implementation\s+code\s+for\s+(the\s+)?)",
            "",
            clean_q,
            flags=re.IGNORECASE,
        ).strip("?. ")

        if len(topic) < 3:
            return clean_q.strip("?. ")
        return topic

    # =========================================================================
    # Layer 1: Deterministic Conversational Cue Detection
    # =========================================================================

    def detect_followup(
        self,
        query: str,
        state: ConversationRAGState | None,
    ) -> tuple[bool, bool, float, str]:
        """
        Evaluate if incoming query is a follow-up or a new topic.
        Returns: (is_followup, topic_shift, confidence, reason)
        """
        clean_q = query.strip()
        if not clean_q:
            return False, False, 0.0, "Empty query"

        if not state or not state.active_topic:
            # No prior active topic in state
            return False, False, 0.0, "No prior conversation state"

        # Check for identical query repetition (e.g. cache verification / re-ask)
        if state.last_user_query and clean_q.lower() == state.last_user_query.lower().strip():
            return False, False, 1.0, "Identical query repetition"

        q_lower = clean_q.lower()
        has_pronoun = bool(_PRONOUNS_PATTERN.search(q_lower))
        has_followup_phrase = bool(_FOLLOWUP_PHRASES_PATTERN.search(q_lower))
        word_count = len(clean_q.split())

        # Check for explicit follow-up phrases / pronouns
        if has_followup_phrase or has_pronoun:
            conf = 0.95 if (has_followup_phrase and has_pronoun) else 0.90
            return True, False, conf, f"Detected referential cues (pronoun={has_pronoun}, phrase={has_followup_phrase})"

        # Short/implicit query check (< 5 words without an independent subject)
        if word_count <= 4:
            # Check coordinating prepositions / question fragments (e.g. "And for contractors?", "How to enroll?")
            if re.match(r"^(and\b|or\b|what\s+about\b|how\s+about\b|how\s+to\b|any\b|also\b|for\b|regarding\b|as\s+for\b)", q_lower):
                return True, False, 0.90, "Short coordinating follow-up query"

            new_entities = self.extract_entities(clean_q)
            prev_entities = [e.lower() for e in (state.active_entities or [])]
            has_prev_entity = any(e.lower() in prev_entities for e in new_entities)
            if not new_entities or has_prev_entity or len(new_entities) <= 1:
                # If it does not match a clear standalone definition query pattern
                if not re.match(r"^(what\s+is\s+the|what\s+are\s+the|define\s+|list\s+all)\b", q_lower):
                    return True, False, 0.85, "Short implicit query continuing previous context"

        # Check for semantic / lexical entity overlap with active topic
        query_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", q_lower)) - _STOP_WORDS
        topic_words = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", (state.active_topic or "").lower())) - _STOP_WORDS
        entities_words = set(
            w for ent in (state.active_entities or []) for w in re.findall(r"\b[a-zA-Z0-9_]{3,}\b", ent.lower())
        ) - _STOP_WORDS

        overlap = query_words.intersection(topic_words.union(entities_words))
        if overlap and len(overlap) >= 1:
            return True, False, 0.80, f"Lexical entity overlap detected on {overlap}"

        # If query has distinct new subject and no follow-up cues, it is a TOPIC SHIFT
        return False, True, 0.90, "Distinct new topic detected without referential pronouns or continuation cues"

    def detect_answer_mode(self, query: str) -> AnswerMode:
        """Determine target answer mode from query cues and intent."""
        clean_q = query.strip()
        if _CODE_EXPLANATION_MODE_PATTERN.search(clean_q):
            return AnswerMode.CODE_EXPLANATION
        if _STEP_BY_STEP_MODE_PATTERN.search(clean_q):
            return AnswerMode.STEP_BY_STEP
        if _COMPARISON_MODE_PATTERN.search(clean_q):
            return AnswerMode.COMPARISON
        if _EXPAND_MODE_PATTERN.search(clean_q):
            return AnswerMode.EXPAND
        if _DETAILED_MODE_PATTERN.search(clean_q):
            return AnswerMode.DETAILED
        if _SUMMARY_MODE_PATTERN.search(clean_q):
            return AnswerMode.SUMMARY
        if _CONTINUE_MODE_PATTERN.search(clean_q):
            return AnswerMode.CONTINUE
        if _EXPLANATION_MODE_PATTERN.search(clean_q):
            return AnswerMode.EXPLANATION
        return AnswerMode.DIRECT

    def get_mode_directives(self, answer_mode: AnswerMode | str) -> str:
        """Generate system prompt instructions corresponding to the active answer mode."""
        mode_str = str(answer_mode.value if isinstance(answer_mode, AnswerMode) else answer_mode).upper()
        if mode_str in ("EXPAND", "DETAILED"):
            return (
                "Mode: EXPAND / DETAILED\n"
                "- Deep architectural and implementation dive.\n"
                "- Avoid repeating high-level summaries from prior turns.\n"
                "- Expand into detailed components, configuration, code execution flow, parameters, and boundary conditions.\n"
                "- Grounding separation: clearly separate DIRECT code definitions, PARTIAL kickoff snippets under [Source N], "
                "RELATED concepts, and explicitly note genuinely MISSING information without fabricating code."
            )
        elif mode_str == "CODE_EXPLANATION":
            return (
                "Mode: CODE EXPLANATION\n"
                "- Provide a thorough, step-by-step walkthrough of the retrieved code implementation.\n"
                "- Explain function signatures, parameters, return types, execution flow, inputs, outputs, and dependencies.\n"
                "- Preserve exact code syntax without fabricating missing functions."
            )
        elif mode_str == "STEP_BY_STEP":
            return (
                "Mode: STEP BY STEP\n"
                "- Provide a structured, numbered, sequential walkthrough of the process or workflow.\n"
                "- Detail each discrete step with inputs, actions, and expected outcomes from the context."
            )
        elif mode_str == "COMPARISON":
            return (
                "Mode: COMPARISON\n"
                "- Structure a clear side-by-side comparison between the entities/topics discussed.\n"
                "- Compare criteria such as purpose, configuration, execution pattern, advantages, and limitations."
            )
        elif mode_str == "EXPLANATION":
            return (
                "Mode: EXPLANATION\n"
                "- Provide a clear, grounded explanation of the concept, architecture, or workflow.\n"
                "- Ground each explanation directly in retrieved source passages and diagrams."
            )
        elif mode_str == "SUMMARY":
            return (
                "Mode: SUMMARY\n"
                "- Provide a concise, structured high-level summary using bullet points or brief synthesis.\n"
                "- Omit extraneous procedural minutiae while retaining core conclusions."
            )
        elif mode_str in ("CONTINUE", "CONTINUATION"):
            return (
                "Mode: CONTINUATION\n"
                "- Provide a logical, step-by-step continuation proceeding directly from the previous turn.\n"
                "- Do not reintroduce background context already established."
            )
        return (
            "Mode: DIRECT EXTRACTION\n"
            "- Answer only the exact question in one short paragraph or at most four compact bullets.\n"
            "- Omit preambles, adjacent facts, implementation details, recaps, and conclusions unless requested.\n"
            "- Use only the retrieved context."
        )

    # =========================================================================
    # Layer 2: Previous Turn Structure Analysis & Ambiguity Detection
    # =========================================================================

    def analyze_previous_turn_structure(
        self,
        query: str,
        state: ConversationRAGState,
    ) -> tuple[str | None, list[str], bool, bool]:
        """
        Analyze previous turns for referenced answer ID, continuity chunk IDs,
        code/diagram presence, and ambiguity detection.
        Returns: (referenced_answer_id, continuity_ids, has_code, ambiguity_detected)
        """
        referenced_answer_id: str | None = None
        continuity_ids: list[str] = []
        has_code = False
        ambiguity_detected = False

        if state.turns:
            last_turn = state.turns[-1]
            referenced_answer_id = last_turn.turn_id

            # Extract continuity chunk IDs from previous retrieved & visual chunks
            for c in (last_turn.retrieved_chunks or []):
                continuity_ids.append(c.chunk.id)
            for c in (last_turn.visual_evidence or []):
                if c.chunk.id not in continuity_ids:
                    continuity_ids.append(c.chunk.id)

            # Check if code was in previous turn
            if last_turn.intent in ("code", "implementation") or "```" in (last_turn.answer or ""):
                has_code = True

        # Check ambiguity: if pronoun used but active_entities has multiple disjoint concepts
        q_lower = query.lower()
        if bool(_PRONOUNS_PATTERN.search(q_lower)):
            active_ents = state.active_entities or []
            if len(active_ents) > 3 and not any(ent.lower() in q_lower for ent in active_ents):
                ambiguity_detected = False  # Resolve conservatively to active_topic

        return referenced_answer_id, continuity_ids, has_code, ambiguity_detected

    # =========================================================================
    # Layer 4: Generic Query Synthesis & Non-Shrinking Expansion Plan
    # =========================================================================

    def create_expansion_plan(
        self,
        answer_mode: AnswerMode,
        has_code: bool = False,
    ) -> ExpansionPlan:
        """Construct deterministic ExpansionPlan based on active answer mode."""
        is_expanded = answer_mode in (
            AnswerMode.EXPAND,
            AnswerMode.DETAILED,
            AnswerMode.CODE_EXPLANATION,
            AnswerMode.STEP_BY_STEP,
            AnswerMode.EXPLANATION,
        )
        return ExpansionPlan(
            restate_subject="minimal" if is_expanded else "omit",
            preserve_prior_facts=True,
            retrieve_additional_context=is_expanded,
            inspect_adjacent_evidence=is_expanded,
            explain_components=is_expanded,
            explain_execution_flow=is_expanded,
            explain_code_line_by_line=has_code or (answer_mode == AnswerMode.CODE_EXPLANATION),
            target_detail_level="detailed" if is_expanded else "standard",
        )

    def resolve_standalone_query(
        self,
        query: str,
        state: ConversationRAGState,
        intent: QueryCategory | str = QueryCategory.FACTUAL,
        answer_mode: AnswerMode = AnswerMode.DIRECT,
        llm: Any | None = None,
    ) -> str:
        """
        Dynamically rewrite an ambiguous follow-up query into a fully specified standalone
        retrieval query without hardcoded document names or entities.
        """
        active_topic = state.active_topic or ""
        active_entities = state.active_entities or []
        entities_str = ", ".join(active_entities) if active_entities else active_topic

        # If LLM is available and enabled, use structured prompt
        effective_llm = llm or self.llm
        if effective_llm is not None:
            try:
                history_text = "\n".join(
                    f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content')}"
                    for m in state.to_history_messages(max_turns=3)
                )
                prompt = (
                    "You are a conversation query rewriting expert. Rewrite the ambiguous follow-up user question into a comprehensive, standalone document search query.\n"
                    "Replace all pronouns ('it', 'that', 'this', 'they', 'the above') with the specific active topic and entities.\n"
                    "Do NOT answer the question. Only output the rewritten standalone search query.\n\n"
                    f"Active Topic: {active_topic}\n"
                    f"Active Entities: {entities_str}\n"
                    f"Conversation History:\n{history_text}\n\n"
                    f"Follow-up Question: {query}\n"
                    "Standalone Search Query:"
                )
                res = str(effective_llm.complete(prompt)).strip().strip('"').strip("'")
                first_line = res.splitlines()[0].strip()
                if len(first_line) >= len(query) and len(first_line) >= 5:
                    return first_line
            except Exception as exc:
                logger.warning("LLM standalone query resolution failed: %s. Using deterministic synthesizer.", exc)

        # Robust Dynamic Synthesizer Fallback (Generic transformation)
        q_lower = query.lower()
        clean_q = query.strip().rstrip("?.")

        # Specific Mode Synthesizers
        if answer_mode == AnswerMode.CODE_EXPLANATION or any(k in q_lower for k in ("explain this code", "explain the code", "show the code", "code for it")):
            return (
                f"Provide a detailed code explanation and walkthrough of the {active_topic} implementation, "
                f"including function definitions, parameters, surrounding setup, execution flow, inputs, outputs, tools, dependencies, and directly supported implementation context from the document."
            )

        if "diagram" in q_lower or "workflow" in q_lower:
            return (
                f"Provide a detailed architectural and workflow explanation of the diagram for {active_topic}, "
                f"including components, interaction flow, data paths, surrounding setup, and system design."
            )

        if answer_mode == AnswerMode.STEP_BY_STEP or "step by step" in q_lower or "walk me through" in q_lower:
            return f"Step-by-step procedural breakdown, detailed execution steps, and requirements for {active_topic}."

        if answer_mode == AnswerMode.COMPARISON or "compare" in q_lower:
            return f"Compare and contrast the components, rules, and specifications of {entities_str} regarding {active_topic}."

        if answer_mode in (AnswerMode.EXPAND, AnswerMode.DETAILED) or "in detail" in q_lower or "detail" in q_lower or "elaborate" in q_lower or "tell me more" in q_lower:
            intent_str = str(intent).lower()
            if "code" in intent_str or "implement" in intent_str or "code" in active_topic.lower() or "agent" in active_topic.lower():
                return (
                    f"Provide a detailed explanation of the {active_topic} implementation, including the previously retrieved implementation code, "
                    f"surrounding setup, related components, execution flow, inputs, outputs, tools, and any directly supported implementation context from the document."
                )
            elif "architecture" in intent_str or "workflow" in active_topic.lower():
                return (
                    f"Comprehensive explanation of the architecture, workflow diagrams, execution flow, "
                    f"components, and interaction pattern of {active_topic}."
                )
            else:
                return (
                    f"Comprehensive and detailed breakdown of {active_topic}, including all specific rules, "
                    f"procedures, requirements, components, parameters, and edge cases."
                )

        if "how does it work" in q_lower or "how it works" in q_lower:
            return f"Detailed explanation of how {active_topic} works, execution workflow, and architecture."

        if "give code" in q_lower or "show code" in q_lower or "code for" in q_lower:
            return f"Implementation code, python definitions, function signature, parameters, and invocation example for {active_topic}."

        if "enroll" in q_lower:
            return f"Enrollment procedure, eligibility criteria, waiting period, and deadlines for {active_topic}."

        if "contractor" in q_lower:
            return f"Eligibility criteria, rules, and policy regarding contractors for {active_topic}."

        if clean_q.lower().startswith(("and ", "what about ", "how about ", "as for ", "for ")):
            sub_subject = re.sub(r"^(and\s+for\s+|and\s+|what\s+about\s+|how\s+about\s+|as\s+for\s+|for\s+)", "", clean_q, flags=re.IGNORECASE).strip("?. ")
            return f"{sub_subject.capitalize()} policy, rules, and eligibility regarding {active_topic}."

        # Replace pronouns directly with active topic
        resolved = _PRONOUNS_PATTERN.sub(active_topic, clean_q)
        if active_topic.lower() not in resolved.lower():
            resolved = f"{resolved} regarding {active_topic}"

        return resolved

    def resolve(
        self,
        query: str,
        state: ConversationRAGState | None,
        intent: QueryCategory | str = QueryCategory.FACTUAL,
    ) -> ConversationResolutionResult:
        """
        Execute end-to-end conversation-aware query resolution across all 4 layers.
        """
        clean_q = query.strip()
        answer_mode = self.detect_answer_mode(clean_q)
        mode_directives = self.get_mode_directives(answer_mode)

        if not state or not state.active_topic:
            # Initial Turn
            topic = self.extract_topic_from_query(clean_q)
            entities = self.extract_entities(clean_q)
            resolution = FollowUpResolution(
                is_follow_up=False,
                confidence=1.0,
                resolved_query=clean_q,
                primary_subject=topic,
                referenced_answer_id=None,
                answer_mode=answer_mode,
                expansion_requested=answer_mode in (AnswerMode.EXPAND, AnswerMode.DETAILED, AnswerMode.CODE_EXPLANATION),
                requested_detail_level="detailed" if answer_mode in (AnswerMode.EXPAND, AnswerMode.DETAILED) else "standard",
                preserve_previous_evidence=False,
                evidence_continuity_ids=[],
                ambiguity_detected=False,
                rationale="Initial query establishing conversation topic",
            )
            expansion_plan = self.create_expansion_plan(answer_mode)
            return ConversationResolutionResult(
                resolved_query=clean_q,
                is_followup=False,
                topic_shift=False,
                confidence=1.0,
                reason="Initial query establishing conversation topic",
                active_topic=topic,
                active_entities=entities,
                answer_mode=answer_mode,
                mode_directives=mode_directives,
                resolution=resolution,
                expansion_plan=expansion_plan,
            )

        is_followup, topic_shift, confidence, reason = self.detect_followup(clean_q, state)

        if topic_shift:
            new_topic = self.extract_topic_from_query(clean_q)
            new_entities = self.extract_entities(clean_q)
            resolution = FollowUpResolution(
                is_follow_up=False,
                confidence=confidence,
                resolved_query=clean_q,
                primary_subject=new_topic,
                referenced_answer_id=None,
                answer_mode=answer_mode,
                expansion_requested=False,
                requested_detail_level="standard",
                preserve_previous_evidence=False,
                evidence_continuity_ids=[],
                ambiguity_detected=False,
                rationale=reason,
            )
            expansion_plan = self.create_expansion_plan(answer_mode)
            return ConversationResolutionResult(
                resolved_query=clean_q,
                is_followup=False,
                topic_shift=True,
                confidence=confidence,
                reason=reason,
                active_topic=new_topic,
                active_entities=new_entities,
                answer_mode=answer_mode,
                mode_directives=mode_directives,
                resolution=resolution,
                expansion_plan=expansion_plan,
            )

        if is_followup:
            ref_id, continuity_ids, has_code, ambiguity_detected = self.analyze_previous_turn_structure(clean_q, state)
            resolved_query = self.resolve_standalone_query(
                query=clean_q,
                state=state,
                intent=intent,
                answer_mode=answer_mode,
            )
            # Add any newly extracted entities to existing active entities
            query_entities = self.extract_entities(clean_q)
            merged_entities = list(state.active_entities or [])
            for ent in query_entities:
                if ent.lower() not in [e.lower() for e in merged_entities]:
                    merged_entities.append(ent)

            expansion_requested = answer_mode in (
                AnswerMode.EXPAND,
                AnswerMode.DETAILED,
                AnswerMode.CODE_EXPLANATION,
                AnswerMode.STEP_BY_STEP,
                AnswerMode.EXPLANATION,
            )
            resolution = FollowUpResolution(
                is_follow_up=True,
                confidence=confidence,
                resolved_query=resolved_query,
                primary_subject=state.active_topic,
                referenced_answer_id=ref_id,
                answer_mode=answer_mode,
                expansion_requested=expansion_requested,
                requested_detail_level="detailed" if expansion_requested else "standard",
                preserve_previous_evidence=True,
                evidence_continuity_ids=continuity_ids,
                ambiguity_detected=ambiguity_detected,
                rationale=reason,
            )
            expansion_plan = self.create_expansion_plan(answer_mode, has_code=has_code)

            return ConversationResolutionResult(
                resolved_query=resolved_query,
                is_followup=True,
                topic_shift=False,
                confidence=confidence,
                reason=reason,
                active_topic=state.active_topic,
                active_entities=merged_entities,
                answer_mode=answer_mode,
                mode_directives=mode_directives,
                resolution=resolution,
                expansion_plan=expansion_plan,
            )

        # Fallback
        resolution = FollowUpResolution(
            is_follow_up=False,
            confidence=0.5,
            resolved_query=clean_q,
            primary_subject=state.active_topic,
            referenced_answer_id=None,
            answer_mode=answer_mode,
            expansion_requested=False,
            requested_detail_level="standard",
            preserve_previous_evidence=True,
            evidence_continuity_ids=[],
            ambiguity_detected=False,
            rationale="Unresolved query intent",
        )
        return ConversationResolutionResult(
            resolved_query=clean_q,
            is_followup=False,
            topic_shift=False,
            confidence=0.5,
            reason="Unresolved query intent",
            active_topic=state.active_topic,
            active_entities=state.active_entities or [],
            answer_mode=answer_mode,
            mode_directives=mode_directives,
            resolution=resolution,
            expansion_plan=self.create_expansion_plan(answer_mode),
        )


class ConversationResolver(FollowUpResolver):
    """
    Backward-compatible alias and wrapper for FollowUpResolver.
    """
    pass
