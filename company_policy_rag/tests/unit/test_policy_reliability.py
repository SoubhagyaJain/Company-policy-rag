from __future__ import annotations

from backend.models.chunk import Chunk, ChunkMetadata
from backend.models.document import DocumentMetadata, DocumentType, RawDocument
from backend.ingestion.chunkers.recursive import RecursiveChunker
from backend.models.rag import ScoredChunk
from backend.rag.policy_reliability import (
    GoverningClauseSelector,
    bind_source_indices,
    enforce_deterministic_calculations,
    expand_policy_queries,
    format_policy_decision_context,
)


def _chunk(
    chunk_id: str,
    section: str,
    text: str,
    *,
    score: float = 5.0,
    page: int = 1,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            text=text,
            metadata=ChunkMetadata(
                document_id="policy-doc",
                source_file="company-rules.pdf",
                section_title=section,
                section_path=section,
                page_number=page,
            ),
        ),
        score=score,
        rerank_score=score,
    )


def test_prescribed_drugs_clause_outranks_general_safety_language() -> None:
    general = _chunk(
        "general-safety",
        "GENERAL SAFETY",
        "Employees shall report unsafe conditions and comply with customer safety procedures.",
        score=9.0,
    )
    specific = _chunk(
        "prescribed-drugs",
        "PRESCRIBED DRUGS",
        "Employees taking prescribed drugs that may affect job performance shall advise their supervisor.",
        score=2.0,
        page=7,
    )

    selection = GoverningClauseSelector().select(
        "An employee is taking prescription medication that may cause drowsiness. Must they tell anyone?",
        [general],
        candidate_pool=[general, specific],
    )

    assert selection.primary_rules[0].chunk.id == "prescribed-drugs"
    assert selection.confidence >= 0.70


def test_working_on_own_account_keeps_family_exception_and_threshold() -> None:
    unrelated = _chunk(
        "premises",
        "UNATTENDED PREMISES",
        "Employees must obtain express owner permission before entering unattended premises.",
        score=8.5,
    )
    primary = _chunk(
        "own-account",
        "22.0 WORKING ON OWN ACCOUNT",
        "Employees must not perform private electrical work on their own account without authorization.",
        score=3.0,
        page=12,
    )
    family = _chunk(
        "family-exception",
        "22.0 WORKING ON OWN ACCOUNT",
        "Immediate family includes a sister. However, work with a commercial value above $500 requires authorization.",
        score=2.5,
        page=12,
    )

    selection = GoverningClauseSelector().select(
        "Can I perform a $900 electrical job for my sister?",
        [unrelated],
        candidate_pool=[unrelated, primary, family],
    )

    ordered_ids = [item.chunk.id for item in GoverningClauseSelector().order_for_context(selection)]
    assert ordered_ids[0] in {"own-account", "family-exception"}
    assert "premises" not in ordered_ids[:2]
    assert any(calc.kind == "threshold_comparison" and calc.result == "above" for calc in selection.calculations)
    assert any(item.chunk.id == "family-exception" for item in selection.exceptions)


def test_after_hours_rules_remain_separate_and_arithmetic_is_deterministic() -> None:
    break_rule = _chunk(
        "break-rule",
        "AFTER HOURS CALLS",
        "When after-hours work prevents an eight-hour break, the employee may delay the normal start to obtain the break.",
        score=6.0,
        page=20,
    )
    midnight_rule = _chunk(
        "midnight-rule",
        "AFTER HOURS CALLS",
        "If a call occurs between midnight and 6am, the delay may equal time spent travelling and on the job.",
        score=5.5,
        page=20,
    )
    overtime_rule = _chunk(
        "overtime-rule",
        "AFTER HOURS CALLS",
        "If an employee is required to start at 7:30am before receiving the eight-hour break, the affected period is paid at overtime.",
        score=5.0,
        page=20,
    )

    selection = GoverningClauseSelector().select(
        "I normally start at 7:30am and finished an emergency callout at 2:30am. What happens?",
        [break_rule, midnight_rule, overtime_rule],
    )
    context = GoverningClauseSelector().order_for_context(selection)
    bind_source_indices(selection, context)

    assert len(selection.structured_rules) >= 3
    calculation = next(calc for calc in selection.calculations if calc.kind == "time_addition")
    assert calculation.result == "10:30 AM"
    assert "travel and on-job duration" in selection.missing_inputs

    corrected = enforce_deterministic_calculations(
        "The employee can start at 11:00 am after a 3.5-hour delay.",
        selection,
    )
    assert "11:00" not in corrected
    assert "3.5" not in corrected
    assert "10:30 AM" in corrected
    assert "cannot be determined" in corrected


def test_smoke_free_vehicle_clause_rejects_adversarial_premise() -> None:
    general = _chunk("breaks", "MEAL BREAKS", "Employees may take an unpaid lunch break.", score=9.0)
    specific = _chunk(
        "smoking",
        "SMOKE-FREE WORKPLACE",
        "Smoking is prohibited in all company vehicles, including during meal and rest breaks.",
        score=2.0,
    )
    selection = GoverningClauseSelector().select(
        "My supervisor says I can smoke in a company vehicle during lunch. Is that correct?",
        [general],
        candidate_pool=[general, specific],
    )
    assert selection.primary_rules[0].chunk.id == "smoking"


def test_unattended_property_selects_express_permission_not_key_assumption() -> None:
    key = _chunk("keys", "COMPANY EQUIPMENT", "Company keys must be returned on termination.", score=8.0)
    permission = _chunk(
        "permission",
        "UNATTENDED CUSTOMER PREMISES",
        "Employees must have express owner permission before entering unattended customer premises.",
        score=3.0,
    )
    selection = GoverningClauseSelector().select(
        "Can I enter an unattended customer's property for an appointment if I have a company key?",
        [key],
        candidate_pool=[key, permission],
    )
    assert selection.primary_rules[0].chunk.id == "permission"


def test_policy_decision_prompt_is_structured_for_small_local_models() -> None:
    rule = _chunk(
        "prescribed-drugs",
        "PRESCRIBED DRUGS",
        "Employees taking prescribed drugs shall advise their supervisor.",
    )
    selection = GoverningClauseSelector().select("Must I disclose prescription medication?", [rule])
    bind_source_indices(selection, [rule])
    prompt = format_policy_decision_context(selection)

    assert "STRUCTURED RULES" in prompt
    assert "primary_rule (Source 1)" in prompt
    assert "do not invent" in prompt.lower()
    queries = expand_policy_queries("Can I do private work for my sister?")
    assert any("working on own account" in query for query in queries)


def test_ingestion_records_clause_parent_and_exception_metadata() -> None:
    document = RawDocument(
        id="policy-doc",
        content="22.3 Private work is prohibited unless it is authorized in writing.",
        metadata=DocumentMetadata(
            source_file="company-rules.pdf",
            file_path="company-rules.pdf",
            file_hash="test-hash",
            document_type=DocumentType.PDF,
            category="policy",
            section_number="22.3",
            section_title="WORKING ON OWN ACCOUNT",
        ),
    )

    chunk = RecursiveChunker().chunk([document])[0]

    assert chunk.metadata.clause_id == "22.3"
    assert chunk.metadata.parent_section == "22"
    assert chunk.metadata.chunk_type == "exception"
