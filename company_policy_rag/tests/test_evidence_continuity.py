"""
Test suite for Evidence Continuity & Downgrade Protection (Phase 3 & Phase 5).
Re-exports and runs tests from test_m1_conversation_evidence_continuity.
"""

from __future__ import annotations

from tests.test_m1_conversation_evidence_continuity import (
    test_m1_01_models_and_enums,
    test_m1_02_layered_follow_up_resolver_tell_me_about_it_in_detail,
    test_m1_03_explain_this_code_resolution,
    test_m1_04_diagram_explanation_resolution,
    test_m1_05_conversation_consistency_guard_downgrade_protection,
    test_m1_06_grounded_system_prompt_rules_a_to_f,
    test_m1_07_chat_service_evidence_context_persistence,
)

__all__ = [
    "test_m1_01_models_and_enums",
    "test_m1_02_layered_follow_up_resolver_tell_me_about_it_in_detail",
    "test_m1_03_explain_this_code_resolution",
    "test_m1_04_diagram_explanation_resolution",
    "test_m1_05_conversation_consistency_guard_downgrade_protection",
    "test_m1_06_grounded_system_prompt_rules_a_to_f",
    "test_m1_07_chat_service_evidence_context_persistence",
]
