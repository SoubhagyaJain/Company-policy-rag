"""Deterministic policy-QA reliability primitives.

This module deliberately sits between semantic reranking and answer generation.
Cross-encoders answer "is this related?"; the selector below answers the more
important policy question: "does this passage directly govern this scenario?"
The implementation is deterministic so it remains available when local models
are degraded or unavailable.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from backend.models.rag import ScoredChunk


_STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "may", "must", "of", "on", "or",
    "should", "the", "their", "they", "to", "was", "what", "when", "with",
}

_NORMATIVE_RE = re.compile(
    r"\b(?:shall|must|required|may not|must not|prohibited|permitted|allowed|"
    r"entitled|will be|is to|are to|cannot)\b",
    re.IGNORECASE,
)
_EXCEPTION_RE = re.compile(
    r"\b(?:except|exception|unless|provided that|however|immediate family|subject to)\b",
    re.IGNORECASE,
)
_DEFINITION_RE = re.compile(
    r"\b(?:means|defined as|definition|includes?|for purposes of)\b",
    re.IGNORECASE,
)
_CONDITION_RE = re.compile(
    r"\b(?:if|when|where|unless|after|before|between|in the event|provided that|subject to)\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", re.IGNORECASE)
_MONEY_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
_DURATION_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*[- ]?(hours?|days?|weeks?|months?|minutes?)\b",
    re.IGNORECASE,
)

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}


@dataclass(frozen=True)
class PolicyTopicProfile:
    name: str
    query_triggers: tuple[str, ...]
    retrieval_terms: tuple[str, ...]
    governing_headings: tuple[str, ...]
    supporting_terms: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()


_TOPIC_PROFILES: tuple[PolicyTopicProfile, ...] = (
    PolicyTopicProfile(
        name="prescription_medication",
        query_triggers=("prescription", "medication", "drowsiness", "drowsy", "prescribed drug"),
        retrieval_terms=("prescribed drugs", "prescription medication", "medication", "advise supervisor", "job performance"),
        governing_headings=("prescribed drugs", "medication", "drugs and alcohol"),
        supporting_terms=("supervisor", "occupational health", "impairment"),
        negative_terms=("unsafe conditions", "customer safety procedures", "drug testing"),
    ),
    PolicyTopicProfile(
        name="private_work",
        query_triggers=("private work", "private job", "electrical work", "sister", "brother", "family member", "own account"),
        retrieval_terms=("working on own account", "own account", "immediate family", "commercial value", "authorization"),
        governing_headings=("working on own account", "outside employment", "private work"),
        supporting_terms=("immediate family", "authorization", "$500", "commercial value"),
        negative_terms=("unattended premises", "customer premises", "company key"),
    ),
    PolicyTopicProfile(
        name="after_hours_call",
        query_triggers=("callout", "call out", "after-hours", "after hours", "emergency call", "2:30", "7:30"),
        retrieval_terms=("after hours calls", "eight hour break", "8 hour break", "midnight", "overtime", "7:30"),
        governing_headings=("after hours calls", "callouts", "hours of work"),
        supporting_terms=("eight hour", "8 hour", "midnight", "6am", "7:30", "overtime", "travelling", "traveling"),
    ),
    PolicyTopicProfile(
        name="smoke_free_vehicle",
        query_triggers=("smoke", "smoking", "cigarette", "company vehicle"),
        retrieval_terms=("smoke free", "smoking prohibited", "company vehicle", "vehicles"),
        governing_headings=("smoking", "smoke free", "company vehicles"),
        supporting_terms=("lunch", "working hours", "vehicle"),
    ),
    PolicyTopicProfile(
        name="unattended_property",
        query_triggers=("unattended", "customer property", "customer's property", "company key", "scheduled appointment"),
        retrieval_terms=("unattended premises", "express owner permission", "customer premises", "entry"),
        governing_headings=("unattended premises", "customer premises", "entry to premises"),
        supporting_terms=("permission", "owner", "scheduled appointment", "key"),
    ),
    PolicyTopicProfile(
        name="sick_leave",
        query_triggers=("sick leave", "sick-leave", "sick days", "paid sick"),
        retrieval_terms=("sick leave", "paid sick leave", "sick days"),
        governing_headings=("sick leave",),
        supporting_terms=("eligibility", "days", "accrual"),
    ),
)


@dataclass
class QueryFacts:
    intent: str
    topic: str | None
    amounts: list[float] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    negated: bool = False
    important_concepts: list[str] = field(default_factory=list)


@dataclass
class PolicyRule:
    id: str
    text: str
    role: str
    source_chunk_id: str
    condition: str | None = None
    action: str | None = None
    source_index: int | None = None


@dataclass
class DeterministicCalculation:
    kind: str
    expression: str
    result: str
    inputs: dict[str, str | float]
    source_chunk_id: str | None = None
    source_index: int | None = None


@dataclass
class ClauseSelection:
    query_facts: QueryFacts
    primary_rules: list[ScoredChunk]
    supporting_rules: list[ScoredChunk]
    exceptions: list[ScoredChunk]
    definitions: list[ScoredChunk]
    irrelevant_but_related: list[ScoredChunk]
    structured_rules: list[PolicyRule]
    calculations: list[DeterministicCalculation]
    missing_inputs: list[str]
    confidence: float
    score_by_chunk: dict[str, float]

    def to_trace_dict(self) -> dict[str, object]:
        def chunk_summary(sc: ScoredChunk) -> dict[str, object]:
            meta = sc.chunk.metadata
            return {
                "chunk_id": sc.chunk.id,
                "source_file": meta.source_file,
                "page_number": meta.display_page_number or meta.page_label or meta.page_number,
                "section_title": meta.section_title,
                "section_number": meta.section_number,
                "score": round(self.score_by_chunk.get(sc.chunk.id, 0.0), 3),
            }

        return {
            "query_facts": asdict(self.query_facts),
            "primary_rules": [chunk_summary(c) for c in self.primary_rules],
            "supporting_rules": [chunk_summary(c) for c in self.supporting_rules],
            "exceptions": [chunk_summary(c) for c in self.exceptions],
            "definitions": [chunk_summary(c) for c in self.definitions],
            "irrelevant_but_related": [chunk_summary(c) for c in self.irrelevant_but_related],
            "structured_rules": [asdict(rule) for rule in self.structured_rules],
            "calculations": [asdict(calc) for calc in self.calculations],
            "missing_inputs": list(self.missing_inputs),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class PolicyAnswerValidation:
    unsupported_numbers: list[str] = field(default_factory=list)
    incorrect_numbers: list[str] = field(default_factory=list)
    missed_conditions: list[str] = field(default_factory=list)
    missed_exceptions: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (self.unsupported_numbers or self.incorrect_numbers or self.missed_conditions or self.missed_exceptions)


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9$]+", (text or "").casefold()))


def _terms(text: str) -> set[str]:
    return {token for token in _normalise(text).split() if len(token) > 2 and token not in _STOP_WORDS}


def _profile_for_query(query: str) -> PolicyTopicProfile | None:
    q = _normalise(query)
    matches = [
        profile
        for profile in _TOPIC_PROFILES
        if any(_normalise(trigger) in q for trigger in profile.query_triggers)
    ]
    if not matches:
        return None
    return max(matches, key=lambda profile: sum(_normalise(t) in q for t in profile.query_triggers))


def extract_query_facts(query: str) -> QueryFacts:
    q = query.casefold()
    profile = _profile_for_query(query)
    if re.search(r"\b(?:can|may|allowed|permitted|permission)\b", q):
        intent = "permission_check"
    elif re.search(r"\b(?:must|required|need to|have to)\b", q):
        intent = "obligation_check"
    elif re.search(r"\b(?:how many|how much|what time|calculate|happens)\b", q):
        intent = "calculation_or_entitlement"
    else:
        intent = "policy_fact"

    relationships = [
        relation
        for relation in ("sister", "brother", "mother", "father", "spouse", "child", "family member")
        if relation in q
    ]
    amounts = [float(raw.replace(",", "")) for raw in _MONEY_RE.findall(query)]
    times = [_canonical_time(match) for match in _TIME_RE.finditer(query)]
    concepts = list(profile.retrieval_terms) if profile else sorted(_terms(query))[:12]
    return QueryFacts(
        intent=intent,
        topic=profile.name if profile else None,
        amounts=amounts,
        times=times,
        relationships=relationships,
        negated=bool(re.search(r"\b(?:not|never|without|prohibited|isn't|aren't)\b", q)),
        important_concepts=concepts,
    )


def expand_policy_queries(query: str) -> list[str]:
    """Return bounded, purpose-specific retrieval queries for policy language."""
    facts = extract_query_facts(query)
    queries = [query.strip()]
    if facts.important_concepts:
        queries.extend(
            [
                " ".join(facts.important_concepts[:6]),
                " ".join(facts.important_concepts[2:]),
            ]
        )
    unique: list[str] = []
    seen: set[str] = set()
    for item in queries:
        key = _normalise(item)
        if key and key not in seen:
            unique.append(item)
            seen.add(key)
    return unique[:4]


def _chunk_search_text(sc: ScoredChunk) -> tuple[str, str]:
    meta = sc.chunk.metadata
    heading = " ".join(
        str(value or "")
        for value in (meta.section_number, meta.section_title, meta.section_path)
    )
    return _normalise(heading), _normalise(sc.chunk.text)


def _specificity_score(query: str, sc: ScoredChunk, profile: PolicyTopicProfile | None) -> float:
    heading, body = _chunk_search_text(sc)
    combined = f"{heading} {body}"
    query_terms = _terms(query)
    base_score = float(sc.rerank_score if sc.rerank_score is not None else sc.score or 0.0)
    # Cross-encoder logits and RRF scores have incompatible scales. A bounded
    # transform keeps either useful without letting it overwhelm policy signals.
    score = 1.5 * math.tanh(max(-5.0, base_score) / 5.0)

    overlap = query_terms.intersection(set(combined.split()))
    score += 3.0 * (len(overlap) / max(1, len(query_terms)))
    if _NORMATIVE_RE.search(sc.chunk.text):
        score += 2.0
    if _CONDITION_RE.search(sc.chunk.text):
        score += 0.75

    if profile:
        for phrase in profile.governing_headings:
            norm = _normalise(phrase)
            if norm and norm in heading:
                score += 7.0
            elif norm and norm in body:
                score += 3.0
        score += 1.5 * sum(_normalise(term) in combined for term in profile.retrieval_terms)
        score += 0.75 * sum(_normalise(term) in combined for term in profile.supporting_terms)
        score -= 3.5 * sum(_normalise(term) in combined for term in profile.negative_terms)

    # Exact user constraints are high-value evidence in policy selection.
    for amount in _MONEY_RE.findall(query):
        if amount.replace(",", "") in combined.replace(" ", ""):
            score += 1.25
    for relation in extract_query_facts(query).relationships:
        if relation in combined or "immediate family" in combined:
            score += 1.5
    return round(score, 4)


def _dedupe_chunks(chunks: Iterable[ScoredChunk]) -> list[ScoredChunk]:
    result: list[ScoredChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk.id in seen:
            continue
        seen.add(chunk.chunk.id)
        result.append(chunk)
    return result


def _sentence_candidates(text: str) -> list[str]:
    lines = [line.strip(" \t-•") for line in text.splitlines() if line.strip()]
    candidates: list[str] = []
    for line in lines:
        candidates.extend(part.strip() for part in re.split(r"(?<=[.;!?])\s+(?=[A-Z0-9])", line) if part.strip())
    return candidates


def _extract_rules(chunks: Sequence[ScoredChunk], roles: dict[str, str]) -> list[PolicyRule]:
    rules: list[PolicyRule] = []
    for sc in chunks:
        source_id = sc.chunk.id
        role = roles.get(source_id, "supporting_rule")
        for sentence in _sentence_candidates(sc.chunk.text):
            if not (_NORMATIVE_RE.search(sentence) or _CONDITION_RE.search(sentence) or _EXCEPTION_RE.search(sentence)):
                continue
            condition: str | None = None
            action: str | None = sentence
            match = re.match(
                r"^((?:if|when|where|unless|after|before|between|provided that|subject to)\b.+?)[,:;]\s*(.+)$",
                sentence,
                re.IGNORECASE,
            )
            if match:
                condition, action = match.group(1).strip(), match.group(2).strip()
            rules.append(
                PolicyRule(
                    id=f"rule_{len(rules) + 1}",
                    text=sentence,
                    role=role,
                    source_chunk_id=source_id,
                    condition=condition,
                    action=action,
                )
            )
    return rules[:20]


def _canonical_time(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2) or "00")
    suffix = match.group(3).replace(".", "").lower()
    if suffix == "pm" and hour != 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def _display_time(value: str) -> str:
    parsed = datetime.strptime(value, "%H:%M")
    return parsed.strftime("%I:%M %p").lstrip("0")


def _break_hours(text: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\s*[- ]?hours?\s+break\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    word_pattern = "|".join(_WORD_NUMBERS)
    match = re.search(rf"\b({word_pattern})\s*[- ]hour\s+break\b", text, re.IGNORECASE)
    return _WORD_NUMBERS[match.group(1).lower()] if match else None


def _calculate(query: str, rules: Sequence[PolicyRule]) -> tuple[list[DeterministicCalculation], list[str]]:
    calculations: list[DeterministicCalculation] = []
    missing_inputs: list[str] = []
    rule_text = " ".join(rule.text for rule in rules)
    times = list(_TIME_RE.finditer(query))
    break_hours = _break_hours(rule_text)

    finish_match = re.search(
        r"(?:finish(?:ed|es)?|ended?|complete(?:d)?)\D{0,35}(\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?))",
        query,
        re.IGNORECASE,
    )
    if break_hours is not None and finish_match:
        parsed = _TIME_RE.search(finish_match.group(1))
        if parsed:
            finish = _canonical_time(parsed)
            result_dt = datetime.strptime(finish, "%H:%M") + timedelta(hours=break_hours)
            result = result_dt.strftime("%H:%M")
            source_id = next((rule.source_chunk_id for rule in rules if "break" in rule.text.casefold()), None)
            calculations.append(
                DeterministicCalculation(
                    kind="time_addition",
                    expression=f"{_display_time(finish)} + {break_hours} hours",
                    result=_display_time(result),
                    inputs={"finish_time": finish, "required_break_hours": break_hours},
                    source_chunk_id=source_id,
                )
            )

    # A midnight travel/work provision needs an actual elapsed duration. A pair
    # of clock times is not a substitute for travel + job duration.
    if re.search(r"midnight|12\s*a\.?m\.?.{0,80}6\s*a\.?m\.?,?", rule_text, re.IGNORECASE):
        has_elapsed_duration = bool(_DURATION_RE.search(query))
        if not has_elapsed_duration and re.search(r"travell?ing|traveling|time spent|on the job", rule_text, re.IGNORECASE):
            missing_inputs.append("travel and on-job duration")

    query_amounts = [float(raw.replace(",", "")) for raw in _MONEY_RE.findall(query)]
    rule_amounts = [float(raw.replace(",", "")) for raw in _MONEY_RE.findall(rule_text)]
    if query_amounts and rule_amounts:
        amount = query_amounts[0]
        threshold = rule_amounts[0]
        relation = "above" if amount > threshold else "at_or_below"
        source_id = next((rule.source_chunk_id for rule in rules if "$" in rule.text), None)
        calculations.append(
            DeterministicCalculation(
                kind="threshold_comparison",
                expression=f"${amount:g} compared with ${threshold:g}",
                result=relation,
                inputs={"amount": amount, "threshold": threshold},
                source_chunk_id=source_id,
            )
        )
    return calculations, missing_inputs


class GoverningClauseSelector:
    """Select and classify the most specific clauses after semantic reranking."""

    def select(
        self,
        query: str,
        reranked_chunks: Sequence[ScoredChunk],
        candidate_pool: Sequence[ScoredChunk] | None = None,
    ) -> ClauseSelection:
        facts = extract_query_facts(query)
        profile = _profile_for_query(query)
        pool = _dedupe_chunks([*reranked_chunks, *(candidate_pool or [])])
        scores = {sc.chunk.id: _specificity_score(query, sc, profile) for sc in pool}
        ranked = sorted(pool, key=lambda sc: scores[sc.chunk.id], reverse=True)

        if not ranked:
            return ClauseSelection(facts, [], [], [], [], [], [], [], [], 0.0, {})

        top_score = scores[ranked[0].chunk.id]
        primary_chunk = ranked[0]
        if _EXCEPTION_RE.search(
            f"{primary_chunk.chunk.metadata.section_title or ''} {primary_chunk.chunk.text}"
        ):
            direct_rule = next(
                (
                    sc
                    for sc in ranked[1:5]
                    if not _EXCEPTION_RE.search(
                        f"{sc.chunk.metadata.section_title or ''} {sc.chunk.text}"
                    )
                    and scores[sc.chunk.id] >= top_score - 5.0
                ),
                None,
            )
            if direct_rule is not None:
                primary_chunk = direct_rule
        primary = [primary_chunk]
        exceptions: list[ScoredChunk] = []
        definitions: list[ScoredChunk] = []
        supporting: list[ScoredChunk] = []
        irrelevant: list[ScoredChunk] = []

        for sc in ranked:
            if sc.chunk.id == primary_chunk.chunk.id:
                continue
            text = f"{sc.chunk.metadata.section_title or ''} {sc.chunk.text}"
            sc_score = scores[sc.chunk.id]
            if _EXCEPTION_RE.search(text) and sc_score >= top_score - 7.0:
                exceptions.append(sc)
            elif _DEFINITION_RE.search(text) and sc_score >= top_score - 7.0:
                definitions.append(sc)
            elif sc_score >= max(1.0, top_score * 0.35):
                supporting.append(sc)
            else:
                irrelevant.append(sc)

        # Keep bounded context while preserving distinct conditions/exceptions.
        supporting = supporting[:4]
        exceptions = exceptions[:3]
        definitions = definitions[:2]
        irrelevant = irrelevant[:5]
        selected = _dedupe_chunks([*primary, *exceptions, *definitions, *supporting])
        role_by_id = {sc.chunk.id: "primary_rule" for sc in primary}
        role_by_id.update({sc.chunk.id: "exception" for sc in exceptions})
        role_by_id.update({sc.chunk.id: "definition" for sc in definitions})
        role_by_id.update({sc.chunk.id: "supporting_rule" for sc in supporting})
        rules = _extract_rules(selected, role_by_id)
        calculations, missing_inputs = _calculate(query, rules)

        second_score = scores[ranked[1].chunk.id] if len(ranked) > 1 else 0.0
        absolute = 1.0 / (1.0 + math.exp(-(top_score - 4.0) / 2.0))
        gap = max(0.0, min(1.0, (top_score - second_score) / max(abs(top_score), 1.0)))
        confidence = round(0.75 * absolute + 0.25 * gap, 3)
        return ClauseSelection(
            query_facts=facts,
            primary_rules=primary,
            supporting_rules=supporting,
            exceptions=exceptions,
            definitions=definitions,
            irrelevant_but_related=irrelevant,
            structured_rules=rules,
            calculations=calculations,
            missing_inputs=missing_inputs,
            confidence=confidence,
            score_by_chunk=scores,
        )

    def order_for_context(self, selection: ClauseSelection, max_chunks: int = 8) -> list[ScoredChunk]:
        return _dedupe_chunks(
            [
                *selection.primary_rules,
                *selection.exceptions,
                *selection.definitions,
                *selection.supporting_rules,
            ]
        )[:max_chunks]


def bind_source_indices(selection: ClauseSelection, context_chunks: Sequence[ScoredChunk]) -> None:
    index_by_id = {sc.chunk.id: index for index, sc in enumerate(context_chunks, start=1)}
    for rule in selection.structured_rules:
        rule.source_index = index_by_id.get(rule.source_chunk_id)
    for calculation in selection.calculations:
        calculation.source_index = index_by_id.get(calculation.source_chunk_id or "")


def format_policy_decision_context(selection: ClauseSelection) -> str:
    """Create low-cognitive-load instructions for a small local generator model."""
    lines = [
        "POLICY DECISION SUPPORT (deterministic; do not cite this block as a source)",
        f"QUERY FACTS: {asdict(selection.query_facts)}",
        f"GOVERNING-CLAUSE CONFIDENCE: {selection.confidence:.3f}",
    ]
    if selection.structured_rules:
        lines.append("STRUCTURED RULES — keep every rule separate:")
        for rule in selection.structured_rules:
            source = f"Source {rule.source_index}" if rule.source_index else "retrieved source"
            lines.append(f"- {rule.role} ({source}): {rule.text}")
    if selection.calculations:
        lines.append("DETERMINISTIC CALCULATIONS — use these exact results; do not recalculate:")
        for calculation in selection.calculations:
            source = f"Source {calculation.source_index}" if calculation.source_index else "governing rule"
            lines.append(f"- {calculation.expression} = {calculation.result} ({source})")
    if selection.missing_inputs:
        lines.append("MISSING INPUTS — do not invent them: " + ", ".join(selection.missing_inputs))
    lines.extend(
        [
            "ANSWER CONSTRAINTS:",
            "- Lead with the primary governing rule, not a general related policy.",
            "- Preserve conditions, exceptions, thresholds, actors, and time windows separately.",
            "- Do not invent missing facts, durations, amounts, dates, or authorization.",
            "- Distinguish explicit policy, deterministic calculation, inference, and missing information.",
            "- If the evidence has no sufficiently specific governing rule, abstain explicitly.",
        ]
    )
    return "\n".join(lines)


def allowed_derived_facts(selection: ClauseSelection) -> list[str]:
    values: list[str] = []
    for calc in selection.calculations:
        values.extend([calc.expression, calc.result])
    return values


def validate_policy_answer(answer: str, selection: ClauseSelection) -> PolicyAnswerValidation:
    validation = PolicyAnswerValidation()
    answer_compact = _normalise(answer)
    source_text = " ".join(
        sc.chunk.text for sc in [
            *selection.primary_rules,
            *selection.exceptions,
            *selection.definitions,
            *selection.supporting_rules,
        ]
    )
    allowed_text = f"{source_text} {' '.join(selection.query_facts.times)} " + " ".join(allowed_derived_facts(selection))
    allowed_compact = _normalise(allowed_text).replace(" ", "")

    for claim in _DURATION_RE.findall(answer):
        raw = f"{claim[0]} {claim[1]}"
        if _normalise(raw).replace(" ", "") not in allowed_compact:
            validation.unsupported_numbers.append(raw)

    for calc in selection.calculations:
        if calc.kind == "time_addition":
            result_key = _normalise(calc.result)
            if result_key not in answer_compact:
                validation.incorrect_numbers.append(
                    f"Expected deterministic result {calc.result} from {calc.expression}."
                )

    # High-risk condition preservation: a selected exception or numeric/time
    # condition must remain visible in the answer, not be merged away.
    for rule in selection.structured_rules:
        if rule.role not in {"primary_rule", "exception"}:
            continue
        condition_values = re.findall(
            r"\$\s*[\d,]+(?:\.\d+)?|\b\d+(?::\d+)?\s*(?:a\.?m\.?|p\.?m\.?)|\b\d+\s*[- ]?hours?\b",
            rule.text,
            re.IGNORECASE,
        )
        for value in condition_values:
            if _normalise(value) not in answer_compact:
                target = validation.missed_exceptions if rule.role == "exception" else validation.missed_conditions
                target.append(value)
    return validation


def enforce_deterministic_calculations(answer: str, selection: ClauseSelection) -> str:
    """Remove unsupported time/duration claims and append canonical calculations."""
    if not selection.calculations and not selection.missing_inputs:
        return answer

    evidence = " ".join(
        sc.chunk.text for sc in [
            *selection.primary_rules,
            *selection.exceptions,
            *selection.definitions,
            *selection.supporting_rules,
        ]
    )
    allowed_times = {_canonical_time(m) for m in _TIME_RE.finditer(evidence)}
    allowed_times.update(selection.query_facts.times)
    allowed_durations = {_normalise(" ".join(match)) for match in _DURATION_RE.findall(evidence)}
    for calc in selection.calculations:
        if calc.kind == "time_addition":
            match = _TIME_RE.search(calc.result)
            if match:
                allowed_times.add(_canonical_time(match))
            allowed_durations.add(_normalise(str(calc.inputs.get("required_break_hours", "")) + " hours"))

    kept: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()):
        if not part.strip():
            continue
        unsupported_time = any(_canonical_time(match) not in allowed_times for match in _TIME_RE.finditer(part))
        unsupported_duration = any(
            _normalise(" ".join(match)) not in allowed_durations
            for match in _DURATION_RE.findall(part)
        )
        if unsupported_time or unsupported_duration:
            continue
        kept.append(part.strip())

    cleaned = " ".join(kept).strip()
    additions: list[str] = []
    for calc in selection.calculations:
        if calc.kind == "time_addition" and _normalise(calc.result) not in _normalise(cleaned):
            tag = f" [Source {calc.source_index}]" if calc.source_index else ""
            additions.append(
                f"Deterministic calculation: {calc.expression} = {calc.result}{tag}."
            )
        elif calc.kind == "threshold_comparison" and calc.result not in _normalise(cleaned):
            tag = f" [Source {calc.source_index}]" if calc.source_index else ""
            additions.append(
                f"Deterministic threshold check: {calc.expression} is {calc.result.replace('_', ' ')} the policy threshold{tag}."
            )
    if selection.missing_inputs:
        additions.append(
            "The exact result cannot be determined for "
            + ", ".join(selection.missing_inputs)
            + " because that input was not provided."
        )
    return " ".join(part for part in [cleaned, *additions] if part).strip()
