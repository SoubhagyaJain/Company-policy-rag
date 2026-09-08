from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from backend.models.conversation import AnswerMode, ConversationRAGState, ConversationTurn
from backend.models.rag import EvidenceStatus, QueryCategory
from backend.rag.conversation_resolver import ConversationResolutionResult, ConversationResolver
from backend.rag.multi_query import decompose_multi_part
from backend.rag.query_router import QueryRouter
from backend.utils.logging import logger


class RetrievalDecision(str, Enum):
    """The only retrieval actions the conversation layer may request."""

    NO_RETRIEVAL = "no_retrieval"
    REUSE_PREVIOUS = "reuse_previous"
    RETRIEVE = "retrieve"
    DECOMPOSE = "decompose"
    ASK_CLARIFICATION = "ask_clarification"


class ResolvedReference(BaseModel):
    expression: str
    resolved_to: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ConversationInterpretation(BaseModel):
    """Validated output of the single conversation-understanding step."""

    intent: QueryCategory = QueryCategory.FACTUAL
    answer_mode: AnswerMode = AnswerMode.DIRECT
    is_followup: bool = False
    topic_shift: bool = False
    returned_to_topic: bool = False
    active_topic: str | None = None
    active_entities: list[str] = Field(default_factory=list)
    standalone_query: str
    retrieval_decision: RetrievalDecision = RetrievalDecision.RETRIEVE
    reuse_turn_id: str | None = None
    resolved_references: list[ResolvedReference] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    clarification_question: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""

    @model_validator(mode="after")
    def validate_policy(self) -> "ConversationInterpretation":
        self.standalone_query = self.standalone_query.strip()
        self.active_topic = self.active_topic.strip() if self.active_topic else None
        self.active_entities = [item.strip() for item in self.active_entities if item.strip()][:12]
        self.sub_queries = [item.strip() for item in self.sub_queries if item.strip()][:4]
        if self.retrieval_decision == RetrievalDecision.ASK_CLARIFICATION:
            self.ambiguous = True
            if not self.clarification_question:
                self.clarification_question = "What specifically are you referring to?"
        if self.retrieval_decision != RetrievalDecision.REUSE_PREVIOUS:
            self.reuse_turn_id = None
        if not self.standalone_query:
            self.standalone_query = self.clarification_question or "Clarify the user's request"
        return self

    def to_resolution_result(
        self,
        resolver: ConversationResolver,
    ) -> ConversationResolutionResult:
        """Adapt the new model to the existing pipeline contract."""
        base_mode_directives = resolver.get_mode_directives(self.answer_mode)
        return ConversationResolutionResult(
            resolved_query=self.standalone_query,
            is_followup=self.is_followup,
            topic_shift=self.topic_shift,
            confidence=self.confidence,
            reason=self.rationale,
            active_topic=self.active_topic,
            active_entities=self.active_entities,
            answer_mode=self.answer_mode,
            mode_directives=base_mode_directives,
            retrieval_decision=self.retrieval_decision.value,
            reuse_turn_id=self.reuse_turn_id,
            clarification_question=self.clarification_question,
            returned_to_topic=self.returned_to_topic,
            resolved_references={
                item.expression: item.resolved_to for item in self.resolved_references
            },
        )


_REUSE_PATTERN = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+)?(?:show\s+(?:me\s+)?the\s+sources?|show\s+(?:me\s+)?(?:that|its)\s+source|"
    r"where\s+did\s+(?:that|this)\s+come\s+from|cite\s+(?:that|this)|"
    r"explain\s+(?:that|this|it|point\s+\d+)\s+(?:more\s+)?simply|"
    r"simplify\s+(?:that|this|it|point\s+\d+)|put\s+(?:that|this|it)\s+simply)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_RETURN_PATTERN = re.compile(
    r"\b(?:(?:going|go)\s+back|back)\s+to\s+(.+?)(?:,|\?|$)",
    re.IGNORECASE,
)
_FRAGMENT_PATTERN = re.compile(
    r"^(international|domestic|contractors?|employees?|interns?|probation|part[- ]time|"
    r"full[- ]time|temporary|remote|hybrid|how\s+much|how\s+long|when|where|why|who\s+qualifies)\??$",
    re.IGNORECASE,
)
_CONVERSATIONAL_PATTERN = re.compile(
    r"^(?:hi|hello|hey|"
    r"thanks?(?:\s+for\s+(?:that|this|explaining|the\s+explanation|your\s+help))?|"
    r"thank\s+you(?:\s+for\s+(?:that|this|explaining|the\s+explanation|your\s+help))?|"
    r"ok(?:ay)?|got\s+it|bye)[!.\s]*$",
    re.IGNORECASE,
)
_PRONOUN_PATTERN = re.compile(r"\b(it|this|that|they|them|their|those|these)\b", re.IGNORECASE)
_AUDIENCE_PATTERN = re.compile(
    r"\b(contractors?|employees?|interns?|consultants?|vendors?|managers?|workers?|"
    r"part[- ]time(?:\s+employees?)?|full[- ]time(?:\s+employees?)?|probationary\s+employees?)\b",
    re.IGNORECASE,
)


class ConversationInterpreter:
    """Interpret one message turn without letting generated answers become evidence.

    A compact structured LLM call is used when conversation state exists. The
    deterministic result remains the fallback and also enforces safety invariants
    after model output is parsed.
    """

    def __init__(
        self,
        llm: Any | None = None,
        resolver: ConversationResolver | None = None,
        query_router: QueryRouter | None = None,
        enabled: bool = True,
        max_context_turns: int = 4,
    ) -> None:
        self.llm = llm
        # This resolver deliberately has no LLM. The interpreter owns the only
        # conversational model call for a request.
        self.resolver = resolver or ConversationResolver(llm=None)
        self.query_router = query_router or QueryRouter()
        self.enabled = enabled
        self.max_context_turns = max(1, min(max_context_turns, 8))

    def interpret(
        self,
        message: str,
        state: ConversationRAGState | None,
    ) -> ConversationInterpretation:
        message = (message or "").strip()
        baseline = self._deterministic_interpret(message, state)

        # A first standalone turn has no references to resolve. Avoid spending a
        # model call where the deterministic router already has complete input.
        if not self.enabled or self.llm is None or not state or not state.turns:
            return self._enforce_grounding_invariants(baseline, state, message, baseline)

        try:
            prompt = self._build_prompt(message, state, baseline)
            raw = str(self.llm.complete(prompt)).strip()
            parsed = self._parse_model_output(raw)
            interpreted = ConversationInterpretation.model_validate(parsed)
            return self._enforce_grounding_invariants(interpreted, state, message, baseline)
        except Exception as exc:
            logger.warning(
                "Conversation interpreter model output was unusable (%s); using deterministic resolution.",
                exc,
            )
            return self._enforce_grounding_invariants(baseline, state, message, baseline)

    def _deterministic_interpret(
        self,
        message: str,
        state: ConversationRAGState | None,
    ) -> ConversationInterpretation:
        classification = self.query_router.classify(message)
        if _CONVERSATIONAL_PATTERN.fullmatch(message):
            return ConversationInterpretation(
                intent=QueryCategory.CONVERSATIONAL,
                standalone_query=message,
                retrieval_decision=RetrievalDecision.NO_RETRIEVAL,
                confidence=0.98,
                rationale="Conversational message does not require document evidence.",
            )

        base = self.resolver.resolve(message, state, intent=classification.category)
        topic = base.active_topic
        entities = list(base.active_entities)
        mode = base.answer_mode
        references: list[ResolvedReference] = []

        if state and state.turns:
            returned = self._resolve_topic_return(message, state)
            if returned is not None:
                target_topic, target_turn = returned
                remainder = self._return_remainder(message)
                standalone = self._rewrite_fragment(remainder, target_topic, state)
                return ConversationInterpretation(
                    intent=self.query_router.classify(standalone).category,
                    answer_mode=mode,
                    is_followup=True,
                    topic_shift=True,
                    returned_to_topic=True,
                    active_topic=target_topic,
                    active_entities=list(target_turn.active_entities),
                    standalone_query=standalone,
                    retrieval_decision=RetrievalDecision.RETRIEVE,
                    resolved_references=[
                        ResolvedReference(
                            expression="going back to",
                            resolved_to=target_topic,
                            confidence=0.98,
                        )
                    ],
                    confidence=0.96,
                    rationale="Explicitly returned to a known earlier topic.",
                )

            if _REUSE_PATTERN.search(message):
                trusted = self._latest_trusted_turn(state)
                if trusted is not None:
                    return ConversationInterpretation(
                        intent=QueryCategory.EXPLANATION,
                        answer_mode=AnswerMode.EXPLANATION,
                        is_followup=True,
                        active_topic=trusted.active_topic or state.active_topic,
                        active_entities=list(trusted.active_entities or state.active_entities),
                        standalone_query=self._reuse_query(message, trusted),
                        retrieval_decision=RetrievalDecision.REUSE_PREVIOUS,
                        reuse_turn_id=trusted.turn_id,
                        resolved_references=[
                            ResolvedReference(
                                expression=self._reference_expression(message),
                                resolved_to=trusted.active_topic or trusted.resolved_query,
                                confidence=0.98,
                            )
                        ],
                        confidence=0.98,
                        rationale="The request transforms or cites the latest verified evidence.",
                    )

            ambiguous_reason = self._ambiguity_reason(message, state)
            if ambiguous_reason:
                return ConversationInterpretation(
                    intent=classification.category,
                    answer_mode=mode,
                    is_followup=True,
                    active_topic=state.active_topic,
                    active_entities=list(state.active_entities),
                    standalone_query=message,
                    retrieval_decision=RetrievalDecision.ASK_CLARIFICATION,
                    ambiguous=True,
                    clarification_question=self._clarification_question(message, state),
                    confidence=0.97,
                    rationale=ambiguous_reason,
                )

            if self._is_context_fragment(message):
                standalone = self._rewrite_fragment(message, state.active_topic or "", state)
                audience = self._audience_from_text(message)
                if state.active_topic:
                    references.append(
                        ResolvedReference(
                            expression=message.rstrip("?"),
                            resolved_to=state.active_topic,
                            confidence=0.96,
                        )
                    )
                if audience and audience.lower() not in [e.lower() for e in entities]:
                    entities.append(audience)
                return ConversationInterpretation(
                    intent=self.query_router.classify(standalone).category,
                    answer_mode=mode,
                    is_followup=True,
                    active_topic=state.active_topic,
                    active_entities=entities[-12:],
                    standalone_query=standalone,
                    retrieval_decision=RetrievalDecision.RETRIEVE,
                    resolved_references=references,
                    confidence=0.95,
                    rationale="Resolved a short follow-up against the active topic.",
                )

            if _PRONOUN_PATTERN.search(message) and state.active_topic:
                standalone, resolved = self._resolve_pronouns(message, state)
                references.extend(resolved)
                base.resolved_query = standalone
                base.is_followup = True
                base.topic_shift = False
                base.active_topic = state.active_topic

        standalone = base.resolved_query or message
        parts = decompose_multi_part(standalone)
        decision = RetrievalDecision.DECOMPOSE if parts else RetrievalDecision.RETRIEVE
        return ConversationInterpretation(
            intent=self.query_router.classify(standalone).category,
            answer_mode=mode,
            is_followup=base.is_followup,
            topic_shift=base.topic_shift,
            active_topic=topic,
            active_entities=entities,
            standalone_query=standalone,
            retrieval_decision=decision,
            resolved_references=references,
            sub_queries=parts,
            confidence=base.confidence,
            rationale=base.reason,
        )

    def _build_prompt(
        self,
        message: str,
        state: ConversationRAGState,
        baseline: ConversationInterpretation,
    ) -> str:
        turns: list[dict[str, Any]] = []
        for turn in state.turns[-self.max_context_turns :]:
            turns.append(
                {
                    "turn_id": turn.turn_id,
                    "user_query": turn.user_query,
                    "standalone_query": turn.resolved_query,
                    "topic": turn.active_topic,
                    "entities": turn.active_entities[-6:],
                    "has_verified_evidence": self._turn_has_trusted_evidence(turn),
                    "sources": self._turn_source_labels(turn),
                    # A tiny outline may resolve "point 2". It is explicitly
                    # navigation metadata, never factual evidence.
                    "untrusted_answer_outline": self._answer_outline(turn.answer),
                }
            )
        schema = {
            "intent": [item.value for item in QueryCategory],
            "answer_mode": [item.value for item in AnswerMode],
            "is_followup": "boolean",
            "topic_shift": "boolean",
            "returned_to_topic": "boolean",
            "active_topic": "string|null",
            "active_entities": ["string"],
            "standalone_query": "string",
            "retrieval_decision": [item.value for item in RetrievalDecision],
            "reuse_turn_id": "string|null",
            "resolved_references": [
                {"expression": "string", "resolved_to": "string", "confidence": 0.0}
            ],
            "sub_queries": ["string"],
            "ambiguous": "boolean",
            "clarification_question": "string|null",
            "confidence": 0.0,
            "rationale": "short string",
        }
        return (
            "Interpret one message for a grounded document QA system. Return one JSON object only.\n"
            "Resolve references using recent USER requests and topics. Full assistant answers are absent. Any "
            "untrusted_answer_outline is navigation metadata only and must never supply a fact. Use "
            "reuse_previous only for a listed turn with "
            "has_verified_evidence=true and only for source/simplification requests that need no new facts. "
            "Use retrieve for a resolved single question, decompose for independent parts, no_retrieval for "
            "pleasantries, and ask_clarification when a reference has multiple plausible meanings. "
            "A topic switch starts fresh. 'Going back to' may select a matching older topic. Keep the standalone "
            "query concise and preserve names, conditions, comparisons, and requested parts.\n\n"
            f"Schema: {json.dumps(schema, separators=(',', ':'))}\n"
            f"Recent safe state: {json.dumps(turns, ensure_ascii=True, separators=(',', ':'))}\n"
            f"Active topic: {json.dumps(state.active_topic)}\n"
            f"Current message: {json.dumps(message)}\n"
            f"Deterministic baseline: {baseline.model_dump_json()}\n"
            "JSON:"
        )

    @staticmethod
    def _parse_model_output(raw: str) -> dict[str, Any]:
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
        match = re.search(r"\{.*\}", fenced, re.DOTALL)
        if not match:
            raise ValueError("No JSON object returned")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("Interpreter output is not an object")
        return data

    def _enforce_grounding_invariants(
        self,
        result: ConversationInterpretation,
        state: ConversationRAGState | None,
        message: str,
        baseline: ConversationInterpretation,
    ) -> ConversationInterpretation:
        # Explicit returns and unresolved references are deterministic safety
        # boundaries. A malformed or overconfident model response cannot turn
        # either into a guess about another topic or group.
        if baseline.returned_to_topic:
            return baseline
        if state and state.turns:
            ambiguity_reason = self._ambiguity_reason(message, state)
            if ambiguity_reason:
                result.is_followup = True
                result.topic_shift = False
                result.active_topic = state.active_topic
                result.active_entities = list(state.active_entities)
                result.standalone_query = message
                result.retrieval_decision = RetrievalDecision.ASK_CLARIFICATION
                result.reuse_turn_id = None
                result.ambiguous = True
                result.clarification_question = self._clarification_question(message, state)
                result.rationale = ambiguity_reason

        # A clear standalone topic switch must start a fresh evidence search.
        # This blocks old-topic contamination even if the model over-applies
        # conversation continuity.
        if (
            baseline.topic_shift
            and not _PRONOUN_PATTERN.search(message)
            and not _REUSE_PATTERN.search(message)
        ):
            result.is_followup = False
            result.topic_shift = True
            result.active_topic = baseline.active_topic
            result.active_entities = list(baseline.active_entities)
            result.standalone_query = baseline.standalone_query
            result.retrieval_decision = baseline.retrieval_decision
            result.reuse_turn_id = None
            result.sub_queries = list(baseline.sub_queries)

        if result.retrieval_decision == RetrievalDecision.REUSE_PREVIOUS:
            trusted = self.get_trusted_turn(state, result.reuse_turn_id)
            if trusted is None or not _REUSE_PATTERN.search(message):
                result.retrieval_decision = RetrievalDecision.RETRIEVE
                result.reuse_turn_id = None
                result.rationale = "Requested evidence reuse was unavailable; fresh retrieval is required."
        if result.retrieval_decision == RetrievalDecision.NO_RETRIEVAL:
            if (
                result.intent != QueryCategory.CONVERSATIONAL
                or not _CONVERSATIONAL_PATTERN.fullmatch(message)
            ):
                result.retrieval_decision = RetrievalDecision.RETRIEVE
        if result.retrieval_decision == RetrievalDecision.DECOMPOSE and not result.sub_queries:
            result.sub_queries = decompose_multi_part(result.standalone_query)
            if not result.sub_queries:
                result.retrieval_decision = RetrievalDecision.RETRIEVE
        if result.returned_to_topic and state:
            known = self._find_topic_turn(result.active_topic or "", state)
            if known is None:
                result.returned_to_topic = False
                result.topic_shift = True
                result.retrieval_decision = RetrievalDecision.RETRIEVE
        return result

    @staticmethod
    def _turn_has_trusted_evidence(turn: ConversationTurn) -> bool:
        status = str(getattr(turn.evidence_status, "value", turn.evidence_status) or "").upper()
        status = status.rsplit(".", 1)[-1]
        return bool(turn.retrieved_chunks) and status in {
            EvidenceStatus.DIRECT.value,
            EvidenceStatus.PARTIAL.value,
            EvidenceStatus.RELATED.value,
            "SUFFICIENT",
            "SUFFICIENT_CONTEXT",
        }

    def _latest_trusted_turn(self, state: ConversationRAGState) -> ConversationTurn | None:
        for turn in reversed(state.turns):
            if self._turn_has_trusted_evidence(turn):
                return turn
        return None

    def get_trusted_turn(
        self,
        state: ConversationRAGState | None,
        turn_id: str | None,
    ) -> ConversationTurn | None:
        """Return verified evidence from one turn in this exact session state."""
        if not state or not turn_id:
            return None
        for turn in reversed(state.turns):
            if turn.turn_id == turn_id and self._turn_has_trusted_evidence(turn):
                return turn
        return None

    @staticmethod
    def _turn_source_labels(turn: ConversationTurn) -> list[str]:
        labels: list[str] = []
        for chunk in turn.retrieved_chunks:
            meta = chunk.chunk.metadata
            label = str(meta.section_title or meta.section_path or meta.source_file or "").strip()
            if label and label not in labels:
                labels.append(label)
        return labels[:4]

    @staticmethod
    def _answer_outline(answer: str) -> list[str]:
        """Extract short numbered labels for reference resolution, not evidence."""
        outline: list[str] = []
        for line in (answer or "").splitlines():
            match = re.match(r"^\s*(\d{1,2})[.)]\s+(.+)$", line.strip())
            if not match:
                continue
            label = re.sub(r"\[Source\s+\d+\]", "", match.group(2), flags=re.IGNORECASE)
            label = re.sub(r"[`*_#]", "", label).strip()
            if label:
                outline.append(f"{match.group(1)}. {label[:160]}")
        return outline[:8]

    def _resolve_topic_return(
        self,
        message: str,
        state: ConversationRAGState,
    ) -> tuple[str, ConversationTurn] | None:
        match = _RETURN_PATTERN.search(message)
        if not match:
            return None
        requested = next((group for group in match.groups() if group), "").strip()
        found = self._find_topic_turn(requested, state)
        if found is None:
            return None
        return found.active_topic or found.resolved_query, found

    @staticmethod
    def _find_topic_turn(requested: str, state: ConversationRAGState) -> ConversationTurn | None:
        wanted = {token for token in re.findall(r"[a-z0-9]+", requested.lower()) if len(token) > 2}
        if not wanted:
            return None
        for turn in reversed(state.turns):
            haystack = " ".join(
                [turn.active_topic or "", turn.resolved_query or "", turn.user_query or ""]
            ).lower()
            if wanted.issubset(set(re.findall(r"[a-z0-9]+", haystack))):
                return turn
        return None

    @staticmethod
    def _return_remainder(message: str) -> str:
        match = _RETURN_PATTERN.search(message)
        if not match:
            return message
        remainder = message[match.end() :].strip(" ,:;-")
        return remainder or "Tell me more"

    @staticmethod
    def _is_context_fragment(message: str) -> bool:
        clean = message.strip()
        if _FRAGMENT_PATTERN.fullmatch(clean):
            return True
        return bool(
            re.match(r"^(?:and\s+)?(?:what|how)\s+about\b", clean, re.IGNORECASE)
            or re.match(r"^(?:and\s+)?who\s+qualifies\??$", clean, re.IGNORECASE)
            or re.match(
                r"^and\s+(?:international|domestic|contractors?|employees?|interns?|"
                r"probation|part[- ]time|full[- ]time|temporary)\??$",
                clean,
                re.IGNORECASE,
            )
        )

    def _rewrite_fragment(
        self,
        message: str,
        topic: str,
        state: ConversationRAGState,
    ) -> str:
        clean = message.strip().rstrip("?.")
        lowered = clean.lower()
        topic = topic.strip()
        if lowered == "international":
            return f"What are the international rules and limits for {topic}?"
        if lowered == "domestic":
            return f"What are the domestic rules and limits for {topic}?"
        if lowered == "who qualifies":
            return f"Who qualifies for {topic}?"
        if lowered == "how much":
            return f"How much does {topic} provide or allow?"
        if lowered == "how long":
            return f"How long does {topic} apply or last?"
        if lowered == "why":
            return f"Why does {topic} apply?"
        if lowered == "when":
            return f"When does {topic} apply?"
        audience = self._audience_from_text(clean)
        if audience:
            if "probation" in audience.lower():
                return f"Does {topic} apply during probation?"
            return f"What does {topic} specify for {audience}?"
        subject = re.sub(
            r"^(?:and\s+)?(?:what|how)\s+about\s+|^(?:and\s+)",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()
        if subject:
            return f"What does {topic} specify about {subject}?"
        return f"Explain {topic} in more detail."

    def _resolve_pronouns(
        self,
        message: str,
        state: ConversationRAGState,
    ) -> tuple[str, list[ResolvedReference]]:
        topic = state.active_topic or ""
        result = message
        refs: list[ResolvedReference] = []
        if re.search(r"\b(it|this|that)\b", result, re.IGNORECASE):
            result = re.sub(r"\b(it|this|that)\b", topic, result, count=1, flags=re.IGNORECASE)
            refs.append(ResolvedReference(expression="it/this/that", resolved_to=topic, confidence=0.95))
        if re.search(r"\b(them|they|their|those|these)\b", result, re.IGNORECASE):
            audience = self._latest_audience(state)
            if audience:
                result = re.sub(
                    r"\b(them|they|their|those|these)\b",
                    audience,
                    result,
                    count=1,
                    flags=re.IGNORECASE,
                )
                refs.append(ResolvedReference(expression="them/they", resolved_to=audience, confidence=0.9))
        return result, refs

    def _ambiguity_reason(self, message: str, state: ConversationRAGState) -> str | None:
        if not _PRONOUN_PATTERN.search(message):
            return None
        if not state.active_topic:
            return "A reference has no active topic."
        has_singular = bool(re.search(r"\b(it|this|that)\b", message, re.IGNORECASE))
        topic = state.active_topic.lower()
        if has_singular and re.search(r"\b(?:and|versus|vs\.?)\b", topic):
            return "A singular reference could refer to more than one active subject."
        has_plural = bool(re.search(r"\b(them|they|their|those|these)\b", message, re.IGNORECASE))
        if has_plural and not self._latest_audience(state):
            return "The referenced group is not uniquely identified in recent user requests."
        return None

    @staticmethod
    def _clarification_question(message: str, state: ConversationRAGState) -> str:
        if re.search(r"\b(them|they|their|those|these)\b", message, re.IGNORECASE):
            return "Which people or group do you mean?"
        if state.active_topic:
            return f"What does “it” refer to in your question about {state.active_topic}?"
        return "What specifically are you referring to?"

    @staticmethod
    def _audience_from_text(text: str) -> str | None:
        match = _AUDIENCE_PATTERN.search(text)
        return match.group(0).strip() if match else None

    def _latest_audience(self, state: ConversationRAGState) -> str | None:
        for turn in reversed(state.turns):
            audience = self._audience_from_text(f"{turn.user_query} {' '.join(turn.active_entities)}")
            if audience:
                return audience
        return self._audience_from_text(" ".join(state.active_entities))

    @classmethod
    def _reuse_query(cls, message: str, turn: ConversationTurn) -> str:
        if re.search(r"\bsource|cite|where\s+did\b", message, re.IGNORECASE):
            return f"Show the sources for {turn.resolved_query or turn.user_query}."
        point = re.search(r"\bpoint\s+(\d+)\b", message, re.IGNORECASE)
        point_suffix = f" point {point.group(1)}" if point else ""
        if point:
            prefix = f"{point.group(1)}. "
            matching_label = next(
                (item[len(prefix) :] for item in cls._answer_outline(turn.answer) if item.startswith(prefix)),
                None,
            )
            if matching_label:
                point_suffix = f" the item labeled “{matching_label}”"
        return f"Explain{point_suffix} from {turn.resolved_query or turn.user_query} more simply."

    @staticmethod
    def _reference_expression(message: str) -> str:
        match = re.search(r"\b(point\s+\d+|that|this|it|source)\b", message, re.IGNORECASE)
        return match.group(0) if match else "previous evidence"
