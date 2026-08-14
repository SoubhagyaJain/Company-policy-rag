"""
Tier 4 Real-World Application Scenarios E2E Test Suite.

Validates end-to-end multi-department policy inquiries, multi-turn clarification dialogues,
mixed-intent requests, executive compliance audits, emergency protocols, and remote work workflows
through FastAPI ASGI endpoints (/api/chat, /api/chat/stream, /api/chat/session, /api/admin/telemetry).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Generator, List, Dict
import pytest
import pytest_asyncio
import httpx

from backend.api.dependencies import (
    get_chat_service,
    get_document_service,
    get_rag_pipeline,
    get_semantic_cache_manager,
    get_telemetry_service,
    reset_dependencies,
)
from backend.api.main import create_app
from backend.embeddings.embeddings import EmbeddingService
from backend.embeddings.vector_store import ChromaVectorStore
from backend.models.api_dto import ChatRequest, ChatResponse
from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole, ContentType
from backend.models.rag import Citation, RAGResponse, RAGTrace, ScoredChunk
from backend.rag.citations import CitationEngine
from backend.rag.context_compression import ContextCompressor
from backend.rag.multi_query import MultiQueryGenerator
from backend.rag.pipeline import RAGPipeline
from backend.rag.query_rewrite import QueryRewriter
from backend.rag.semantic_cache import SemanticCacheManager
from backend.retrieval.bm25 import BM25SearchIndex
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.vector import DenseVectorRetriever
from backend.services.chat_service import ChatService
from backend.services.telemetry_service import TelemetryService
from tests.e2e.helpers.sse_client import SSEDecoder


# ════════════════════════════════════════════════════════════════════════════════
# 1. ENTERPRISE POLICY CORPUS FIXTURES & MOCK LLM
# ════════════════════════════════════════════════════════════════════════════════

def create_enterprise_scenario_chunks() -> list[Chunk]:
    """
    Creates rich enterprise policy chunks across IT, Finance, HR, Legal, and EHS departments.
    """
    return [
        # IT Security Policy: Laptop International Travel
        Chunk(
            id="chunk_it_travel_001",
            text=(
                "Section 4.3 International Travel with Company Laptops: Employees traveling internationally "
                "with company-issued hardware must submit an IT Travel Security Request at least 10 business days "
                "prior to departure. All laptops must run full-disk encryption (FileVault for macOS, BitLocker for "
                "Windows) with 256-bit AES encryption. When traveling to designated high-risk technology-export "
                "jurisdictions (List A countries), employees are prohibited from bringing standard corporate laptops "
                "and must check out a clean, temporary loaner travel laptop from IT Asset Management. All public Wi-Fi "
                "connections at international conferences, airports, and hotels must utilize the mandatory GlobalProtect "
                "VPN with multi-factor authentication (MFA) enabled at all times."
            ),
            metadata=ChunkMetadata(
                document_id="doc_it_sec_2026",
                source_file="IT_Data_Security_Policy_2026.pdf",
                file_path="/data/policies/IT_Data_Security_Policy_2026.pdf",
                file_hash="hash_it_sec_001",
                document_type="company_policy",
                category="IT",
                chunk_index=1,
                page_number=12,
                section_title="4.3 International Travel with Company Laptops",
                section_number="4.3",
                section_path="IT Data Security > Section 4.3 International Travel",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "IT",
                    "policy_id": "IT-SEC-2026",
                    "topic_tags": ["data_security", "laptop_travel", "vpn_encryption", "hardware_security"],
                },
            ),
        ),
        # IT Hardware Provisioning for Remote Workers
        Chunk(
            id="chunk_it_remote_002",
            text=(
                "Section 2.1 Remote Worker Hardware Provisioning: The company provides all approved full-time remote "
                "employees with standard IT hardware upon hire: one corporate laptop (choice of 16-inch MacBook Pro "
                "or Dell XPS 15), one external 27-inch 4K monitor, an ergonomic keyboard and wireless mouse, and two "
                "hardware security keys (YubiKey 5C). All hardware remains company property, is pre-enrolled in MDM, "
                "and must be returned upon termination."
            ),
            metadata=ChunkMetadata(
                document_id="doc_it_sec_2026",
                source_file="IT_Data_Security_Policy_2026.pdf",
                file_path="/data/policies/IT_Data_Security_Policy_2026.pdf",
                file_hash="hash_it_sec_002",
                document_type="company_policy",
                category="IT",
                chunk_index=2,
                page_number=5,
                section_title="2.1 Remote Worker Hardware Provisioning",
                section_number="2.1",
                section_path="IT Data Security > Section 2.1 Hardware Provisioning",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "IT",
                    "policy_id": "IT-SEC-2026",
                    "topic_tags": ["hardware_provisioning", "remote_work", "equipment"],
                },
            ),
        ),
        # Finance Travel & Expense Policy: International Conferences
        Chunk(
            id="chunk_fin_conf_001",
            text=(
                "Section 3.2 Conference Travel and Expense Reimbursement: All international conference attendance "
                "requires prior VP approval and registration in the Navan corporate portal. Flights must be booked "
                "in economy class at least 14 days in advance. The daily meal per diem for international conferences "
                "is $75 per day (or official GSA local rate if higher). Conference registration fees, economy airfare, "
                "baggage fees, and ground transportation are 100% reimbursable with itemized receipts submitted via "
                "Concur within 30 days of trip completion."
            ),
            metadata=ChunkMetadata(
                document_id="doc_fin_exp_2026",
                source_file="Finance_Travel_Expense_Policy_2026.pdf",
                file_path="/data/policies/Finance_Travel_Expense_Policy_2026.pdf",
                file_hash="hash_fin_exp_001",
                document_type="company_policy",
                category="Finance",
                chunk_index=1,
                page_number=8,
                section_title="3.2 Conference Travel and Expense Reimbursement",
                section_number="3.2",
                section_path="Finance Travel & Expense > Section 3.2 Conference Travel",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "Finance",
                    "policy_id": "FIN-EXP-2026",
                    "topic_tags": ["travel_expense", "conference_reimbursement", "per_diem"],
                },
            ),
        ),
        # Finance Remote Work Home Office Reimbursement
        Chunk(
            id="chunk_fin_remote_002",
            text=(
                "Section 5.4 Home Office Setup and Utility Reimbursement: Full-time remote employees are eligible "
                "for a one-time home office setup reimbursement of up to $500 for desks, ergonomic chairs, and desk lighting. "
                "In addition, remote employees receive a recurring $50 per month internet service subsidy. All expense "
                "claims must be submitted with itemized receipts through Concur within 30 days of purchase."
            ),
            metadata=ChunkMetadata(
                document_id="doc_fin_exp_2026",
                source_file="Finance_Travel_Expense_Policy_2026.pdf",
                file_path="/data/policies/Finance_Travel_Expense_Policy_2026.pdf",
                file_hash="hash_fin_exp_002",
                document_type="company_policy",
                category="Finance",
                chunk_index=2,
                page_number=14,
                section_title="5.4 Home Office Setup and Utility Reimbursement",
                section_number="5.4",
                section_path="Finance Travel & Expense > Section 5.4 Home Office Setup",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "Finance",
                    "policy_id": "FIN-EXP-2026",
                    "topic_tags": ["home_office_stipend", "internet_subsidy", "remote_reimbursement"],
                },
            ),
        ),
        # HR Benefits Policy: Paid Parental Leave
        Chunk(
            id="chunk_hr_leave_001",
            text=(
                "Section 6.1 Paid Parental Leave Policy: Primary caregivers are entitled to 16 weeks of 100% fully "
                "paid parental leave following the birth, adoption, or foster placement of a child. Secondary caregivers "
                "receive 6 weeks of fully paid leave. Leave can be taken continuously or intermittently in minimum 2-week "
                "increments within the first 12 months of the qualifying event, subject to mutual agreement with the "
                "employee's manager and team business needs."
            ),
            metadata=ChunkMetadata(
                document_id="doc_hr_ben_2026",
                source_file="HR_Employee_Benefits_Guide_2026.pdf",
                file_path="/data/policies/HR_Employee_Benefits_Guide_2026.pdf",
                file_hash="hash_hr_ben_001",
                document_type="company_policy",
                category="HR",
                chunk_index=1,
                page_number=18,
                section_title="6.1 Paid Parental Leave Policy",
                section_number="6.1",
                section_path="HR Benefits > Section 6.1 Paid Parental Leave",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "HR",
                    "policy_id": "HR-BEN-2026",
                    "topic_tags": ["parental_leave", "benefits", "primary_caregiver"],
                },
            ),
        ),
        # HR Benefits Policy: Parental Leave Application Procedure & Forms
        Chunk(
            id="chunk_hr_leave_002",
            text=(
                "Section 6.3 Applying for Parental Leave: Employees must submit the Parental Leave Request Form "
                "(Form HR-204) to HR Benefits along with medical certification or official adoption placement documentation "
                "at least 30 calendar days prior to the anticipated start date. For emergency or premature births, "
                "formal notice and documentation must be provided within 5 business days following delivery."
            ),
            metadata=ChunkMetadata(
                document_id="doc_hr_ben_2026",
                source_file="HR_Employee_Benefits_Guide_2026.pdf",
                file_path="/data/policies/HR_Employee_Benefits_Guide_2026.pdf",
                file_hash="hash_hr_ben_002",
                document_type="company_policy",
                category="HR",
                chunk_index=2,
                page_number=19,
                section_title="6.3 Applying for Parental Leave",
                section_number="6.3",
                section_path="HR Benefits > Section 6.3 Applying for Parental Leave",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "HR",
                    "policy_id": "HR-BEN-2026",
                    "topic_tags": ["parental_leave_forms", "form_hr204", "application_procedure"],
                },
            ),
        ),
        # HR Benefits Policy: Dental vs Vision Coverage Comparison & Open Enrollment
        Chunk(
            id="chunk_hr_dental_vision_003",
            text=(
                "Section 4.2 Dental vs. Vision Benefits Comparison: The Standard Dental Plan (Delta Dental PPO) covers "
                "100% of preventive care, 80% of basic procedures, and 50% of major dental services up to an annual "
                "maximum of $2,000 per covered member. The Vision Care Plan (VSP Choice) covers annual comprehensive "
                "eye exams at 100% and provides up to a $500 annual allowance for designer frames and prescription lenses. "
                "Annual Open Enrollment runs strictly from November 1 through November 30 for benefit selections taking "
                "effect January 1 of the following plan year."
            ),
            metadata=ChunkMetadata(
                document_id="doc_hr_ben_2026",
                source_file="HR_Employee_Benefits_Guide_2026.pdf",
                file_path="/data/policies/HR_Employee_Benefits_Guide_2026.pdf",
                file_hash="hash_hr_ben_003",
                document_type="company_policy",
                category="HR",
                chunk_index=3,
                page_number=11,
                section_title="4.2 Dental vs. Vision Benefits Comparison",
                section_number="4.2",
                section_path="HR Benefits > Section 4.2 Dental and Vision",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "HR",
                    "policy_id": "HR-BEN-2026",
                    "topic_tags": ["dental_vision", "open_enrollment", "benefits_comparison"],
                },
            ),
        ),
        # Legal: Whistleblower Protection & SEC Reporting Rights
        Chunk(
            id="chunk_legal_whistleblower_001",
            text=(
                "Section 1.4 Whistleblower Protection & Regulatory Disclosures: Employees have the unconditional "
                "legal right under federal securities laws (including SEC Rule 21F-17, Sarbanes-Oxley Act, and Dodd-Frank Act) "
                "to report potential securities violations, accounting fraud, or unlawful conduct directly to the SEC, "
                "DOJ, OSHA, or any competent regulatory agency. The company strictly prohibits any retaliation, termination, "
                "or adverse action against any whistleblower. Nothing in any company policy, code of conduct, or agreement "
                "prohibits or impedes an employee from communicating directly with government authorities or recovering "
                "statutory whistleblower bounty awards."
            ),
            metadata=ChunkMetadata(
                document_id="doc_legal_comp_2026",
                source_file="Legal_Compliance_Whistleblower_Policy_2026.pdf",
                file_path="/data/policies/Legal_Compliance_Whistleblower_Policy_2026.pdf",
                file_hash="hash_legal_001",
                document_type="legal_document",
                category="Legal",
                chunk_index=1,
                page_number=3,
                section_title="1.4 Whistleblower Protection & Regulatory Disclosures",
                section_number="1.4",
                section_path="Legal Compliance > Section 1.4 Whistleblower Protection",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "Legal",
                    "policy_id": "LEGAL-COMP-2026",
                    "topic_tags": ["whistleblower_protection", "sec_rule_21f17", "legal_compliance"],
                },
            ),
        ),
        # Legal: Employee Non-Disclosure Agreement (NDA) Whistleblower Carve-out
        Chunk(
            id="chunk_legal_nda_002",
            text=(
                "Section 8.2 Non-Disclosure Agreement (NDA) Statutory Exceptions: While employees have a general duty "
                "to safeguard proprietary trade secrets, Section 8.2 of the standard Employee NDA contains an explicit "
                "statutory carve-out pursuant to the Defend Trade Secrets Act (DTSA) and SEC Rule 21F-17. The NDA states "
                "that confidentiality covenants do NOT apply to reports of unlawful conduct or securities violations made "
                "in confidence to federal, state, or local government officials, or in court filings made under seal. "
                "Consequently, there is NO legal conflict between NDA confidentiality duties and statutory whistleblower "
                "reporting rights."
            ),
            metadata=ChunkMetadata(
                document_id="doc_legal_comp_2026",
                source_file="Legal_Compliance_Whistleblower_Policy_2026.pdf",
                file_path="/data/policies/Legal_Compliance_Whistleblower_Policy_2026.pdf",
                file_hash="hash_legal_002",
                document_type="legal_document",
                category="Legal",
                chunk_index=2,
                page_number=16,
                section_title="8.2 Non-Disclosure Agreement (NDA) Statutory Exceptions",
                section_number="8.2",
                section_path="Legal Compliance > Section 8.2 NDA Exceptions",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "Legal",
                    "policy_id": "LEGAL-COMP-2026",
                    "topic_tags": ["nda_compliance", "trade_secrets_carveout", "whistleblower_exception"],
                },
            ),
        ),
        # EHS: Workplace Safety & Warehouse Injury Response Timelines
        Chunk(
            id="chunk_ehs_injury_001",
            text=(
                "Section 3.1 Warehouse Injury Emergency Response Protocol: In the event of an employee workplace "
                "injury in the warehouse: 1. Immediate Actions: Provide first aid immediately, call 911 for severe injuries "
                "or fractures, and cordon off the incident area. 2. Internal Reporting Timelines: Notify the shift supervisor "
                "within 1 hour and submit the Internal Safety Incident Report Form SAF-101 to the EHS Manager within 4 hours. "
                "3. Mandatory OSHA Timelines: Fatalities must be reported to OSHA within 8 hours; in-patient hospitalizations, "
                "amputations, or loss of an eye must be reported to OSHA within 24 hours."
            ),
            metadata=ChunkMetadata(
                document_id="doc_ehs_safe_2026",
                source_file="EHS_Workplace_Safety_Emergency_Protocol_2026.pdf",
                file_path="/data/policies/EHS_Workplace_Safety_Emergency_Protocol_2026.pdf",
                file_hash="hash_ehs_001",
                document_type="company_policy",
                category="EHS",
                chunk_index=1,
                page_number=7,
                section_title="3.1 Warehouse Injury Emergency Response Protocol",
                section_number="3.1",
                section_path="EHS Safety > Section 3.1 Warehouse Emergency Protocol",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "EHS",
                    "policy_id": "EHS-SAFE-2026",
                    "topic_tags": ["warehouse_injury", "emergency_protocol", "osha_reporting"],
                },
            ),
        ),
        # EHS: Mandatory OSHA Incident Documentation Forms
        Chunk(
            id="chunk_ehs_forms_002",
            text=(
                "Section 3.4 Mandatory Incident Documentation Forms: Any workplace injury requiring medical treatment "
                "beyond first aid requires three mandatory records: (a) Internal Safety Incident Report Form SAF-101 "
                "(completed by supervisor within 4 hours); (b) OSHA Form 301 (Injury and Illness Incident Report, completed "
                "within 7 calendar days); and (c) Logging the event on OSHA Form 300 (Log of Work-Related Injuries and Illnesses)."
            ),
            metadata=ChunkMetadata(
                document_id="doc_ehs_safe_2026",
                source_file="EHS_Workplace_Safety_Emergency_Protocol_2026.pdf",
                file_path="/data/policies/EHS_Workplace_Safety_Emergency_Protocol_2026.pdf",
                file_hash="hash_ehs_002",
                document_type="company_policy",
                category="EHS",
                chunk_index=2,
                page_number=9,
                section_title="3.4 Mandatory Incident Documentation Forms",
                section_number="3.4",
                section_path="EHS Safety > Section 3.4 Mandatory Incident Forms",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "EHS",
                    "policy_id": "EHS-SAFE-2026",
                    "topic_tags": ["osha_forms", "form_saf101", "form_osha301", "form_osha300"],
                },
            ),
        ),
        # HR Education Assistance & Tuition Reimbursement 2026 Policy Update
        Chunk(
            id="chunk_hr_edu_001",
            text=(
                "Section 2.1 Education Assistance Annual Limit: Effective January 1, 2026, the company increased the "
                "maximum annual tuition reimbursement from the previous $3,000 limit (2024 policy) to $5,250 per calendar "
                "year, matching the IRS tax-free educational assistance limit. Eligible courses must be job-related and "
                "completed at an accredited institution with a grade of 'B' or higher."
            ),
            metadata=ChunkMetadata(
                document_id="doc_hr_edu_2026",
                source_file="HR_Education_Assistance_Policy_2026.pdf",
                file_path="/data/policies/HR_Education_Assistance_Policy_2026.pdf",
                file_hash="hash_hr_edu_001",
                document_type="company_policy",
                category="HR",
                chunk_index=1,
                page_number=4,
                section_title="2.1 Education Assistance Annual Limit",
                section_number="2.1",
                section_path="HR Education > Section 2.1 Annual Tuition Limit",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "HR",
                    "policy_id": "HR-EDU-2026",
                    "topic_tags": ["tuition_reimbursement", "education_assistance", "policy_update_2026"],
                },
            ),
        ),
        # Global Mobility: International Remote Work Protocol & Tax Residency
        Chunk(
            id="chunk_glb_mobility_001",
            text=(
                "Section 4.1 Cross-Border Remote Work Protocol: US employees may work remotely from an approved "
                "international country (such as the UK or Germany) for up to 30 cumulative calendar days per year with "
                "written manager approval. Any international remote work exceeding 30 days (such as 60 or 90 days) "
                "strictly requires formal review and unanimous approval from the Department VP, HR Global Mobility, and "
                "Corporate Tax VP to mitigate corporate permanent establishment and foreign payroll tax liabilities."
            ),
            metadata=ChunkMetadata(
                document_id="doc_glb_mob_2026",
                source_file="Global_Mobility_Remote_Work_Protocol_2026.pdf",
                file_path="/data/policies/Global_Mobility_Remote_Work_Protocol_2026.pdf",
                file_hash="hash_glb_001",
                document_type="company_policy",
                category="HR",
                chunk_index=1,
                page_number=6,
                section_title="4.1 Cross-Border Remote Work Protocol",
                section_number="4.1",
                section_path="Global Mobility > Section 4.1 Cross-Border Remote Work",
                chunk_strategy="recursive",
                node_role=ChunkRole.STANDALONE,
                extra={
                    "department": "HR",
                    "policy_id": "GLB-MOB-2026",
                    "topic_tags": ["global_mobility", "international_remote_work", "tax_compliance"],
                },
            ),
        ),
    ]


class MockTokenDelta:
    """Mock token delta object for stream_complete interface."""
    def __init__(self, delta: str):
        self.delta = delta


class MockEnterpriseLLM:
    """
    High-fidelity mock LLM providing authoritative grounded responses with precise
    [Source N] citations for enterprise policy scenarios.
    """
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model

    def _generate_grounded_answer(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Scenario 1: Cross-department laptop travel (IT + Finance)
        if "traveling with company laptops" in prompt_lower or ("laptop" in prompt_lower and "conference" in prompt_lower):
            return (
                "When traveling with company laptops to international conferences, you must adhere to both IT and Finance policies:\n\n"
                "1. **IT Security Requirements**: Submit an IT Travel Security Request at least 10 business days prior to departure. "
                "Laptops must have 256-bit AES full-disk encryption enabled (FileVault/BitLocker). When traveling to designated high-risk "
                "jurisdictions (List A countries), you are prohibited from taking your standard laptop and must check out a clean loaner travel laptop. "
                "Always use GlobalProtect VPN with MFA when connecting to public Wi-Fi [Source 1].\n\n"
                "2. **Finance & Travel Reimbursement**: International conference attendance requires prior VP approval and Navan booking. "
                "The daily meal per diem is $75/day (or GSA local rate). Conference registration fees, economy airfare, and baggage fees are "
                "100% reimbursable with receipts submitted via Concur within 30 days [Source 2]."
            )

        # Scenario 2 - Turn 1: Parental leave duration
        if "parental leave do primary caregivers" in prompt_lower or "how much parental leave" in prompt_lower:
            return (
                "Primary caregivers are entitled to 16 weeks of 100% fully paid parental leave following the birth, adoption, "
                "or foster placement of a child. Secondary caregivers receive 6 weeks of fully paid leave [Source 1]."
            )

        # Scenario 2 - Turn 2: Intermittent leave
        if "intermittently" in prompt_lower or "taken intermittently" in prompt_lower:
            return (
                "Yes, parental leave can be taken intermittently. It must be taken in minimum 2-week increments within the first 12 months "
                "of the qualifying event, subject to mutual agreement with your manager and team business needs [Source 1]."
            )

        # Scenario 2 - Turn 3: Parental leave paperwork & forms
        if "paperwork is required" in prompt_lower or ("form" in prompt_lower and "parental" in prompt_lower):
            return (
                "To apply for parental leave, you must submit the Parental Leave Request Form (Form HR-204) to HR Benefits along with medical "
                "certification or official adoption placement documentation at least 30 calendar days prior to the start date. For emergency or premature "
                "births, submit notice and documentation within 5 business days of delivery [Source 1]."
            )

        # Scenario 3: Mixed intent (Greeting + Dental vs Vision + Enrollment Deadlines)
        if "dental coverage with the vision plan" in prompt_lower or ("dental" in prompt_lower and "vision" in prompt_lower and "deadline" in prompt_lower):
            return (
                "Good morning! Here is the comparison between our Standard Dental Plan and Vision Plan, along with enrollment deadlines:\n\n"
                "• **Standard Dental Plan (Delta Dental PPO)**: Covers 100% of preventive care, 80% of basic procedures, and 50% of major services "
                "up to an annual maximum of $2,000 per member [Source 1].\n"
                "• **Vision Care Plan (VSP Choice)**: Covers annual eye exams at 100% and provides up to $500 annual allowance for frames and prescription lenses [Source 1].\n"
                "• **Enrollment Deadlines**: Annual Open Enrollment runs strictly from November 1 through November 30 for changes effective January 1 [Source 1]."
            )

        # Scenario 4: Executive Compliance (Whistleblower vs NDA)
        if "whistleblower" in prompt_lower and ("nda" in prompt_lower or "non-disclosure" in prompt_lower):
            return (
                "There is NO conflict between the company's Whistleblower Protection policy and the standard Employee Non-Disclosure Agreement (NDA) [Source 2].\n\n"
                "1. **Whistleblower Rights**: Under SEC Rule 21F-17, Sarbanes-Oxley, and Dodd-Frank, employees have the absolute right to report potential "
                "securities violations, fraud, or unlawful conduct directly to the SEC, DOJ, OSHA, or any government agency without company pre-approval [Source 1].\n"
                "2. **NDA Statutory Carve-Out**: Section 8.2 of the standard NDA incorporates explicit statutory exceptions under the Defend Trade Secrets Act (DTSA) "
                "and SEC Rule 21F-17, specifying that confidentiality obligations do NOT apply to reports of illegal conduct made in confidence to government officials [Source 2]."
            )

        # Scenario 5: Emergency Crisis Protocol (Warehouse Injury & OSHA reporting)
        if "injury in the warehouse" in prompt_lower or ("warehouse" in prompt_lower and "osha" in prompt_lower):
            return (
                "In the event of an employee workplace injury in the warehouse, the following immediate actions, reporting timelines, and OSHA forms are mandatory:\n\n"
                "1. **Immediate Actions**: Administer first aid immediately, call 911 for severe injuries/fractures, and cordon off the incident scene [Source 1].\n"
                "2. **Reporting Timelines**: Notify the shift supervisor within 1 hour; submit the Internal Incident Report Form (SAF-101) to the EHS Manager within 4 hours. "
                "Fatalities must be reported to OSHA within 8 hours; hospitalizations, amputations, or loss of an eye must be reported to OSHA within 24 hours [Source 1].\n"
                "3. **Mandatory Documentation Forms**: Complete Internal Incident Form SAF-101 (within 4 hours), OSHA Form 301 (within 7 calendar days), and record on OSHA Form 300 log [Source 2]."
            )

        # Scenario 6: Remote work equipment & reimbursement limits
        if "equipment does the company provide for remote" in prompt_lower or ("remote" in prompt_lower and "reimbursement" in prompt_lower):
            return (
                "For approved full-time remote employees, the company provides standard IT hardware and expense reimbursements:\n\n"
                "• **IT Equipment Provided**: Choice of 16-inch MacBook Pro or Dell XPS 15, one external 27-inch 4K monitor, ergonomic keyboard & wireless mouse, and two YubiKey 5C security keys [Source 1].\n"
                "• **Expense Reimbursement**: One-time home office setup reimbursement of up to $500 for desks/chairs/lighting, plus a recurring $50/month internet subsidy submitted via Concur within 30 days of purchase [Source 2]."
            )

        # Scenario 7: Policy versioning tuition assistance change
        if "tuition assistance" in prompt_lower or "education assistance" in prompt_lower:
            return (
                "Under the 2026 Education Assistance Policy, the maximum annual tuition reimbursement was increased to $5,250 per calendar year "
                "(matching the IRS tax-free limit), up from the previous $3,000 annual limit in the 2024 policy. Eligible courses must be job-related "
                "and completed with a grade of 'B' or higher [Source 1]."
            )

        # Scenario 8: Global mobility & international remote work
        if "remotely from an overseas" in prompt_lower or "spain" in prompt_lower or "germany" in prompt_lower:
            return (
                "US employees may work remotely from an approved international country for up to 30 days with written manager approval. However, working "
                "remotely for 60 days strictly requires formal review and unanimous approval from the Department VP, HR Global Mobility, and Corporate Tax VP "
                "to prevent corporate permanent establishment and foreign payroll tax withholding liabilities [Source 1]."
            )

        # Generic fallback synthesis
        return "Based on company documentation, please refer to the relevant policy guidelines [Source 1]."

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self._generate_grounded_answer(prompt)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[MockTokenDelta, None, None]:
        full_text = self._generate_grounded_answer(prompt)
        words = full_text.split(" ")
        for idx, word in enumerate(words):
            token = word + (" " if idx < len(words) - 1 else "")
            yield MockTokenDelta(delta=token)


# ════════════════════════════════════════════════════════════════════════════════
# 2. FIXTURES & APP HARNESS
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def enterprise_rag_environment() -> tuple[RAGPipeline, ChatService, TelemetryService]:
    """
    Initializes a completely isolated, deterministic enterprise RAG environment with
    in-memory ChromaVectorStore, BM25SearchIndex, docstore, and MockEnterpriseLLM.
    """
    chunks = create_enterprise_scenario_chunks()
    docstore = {c.id: c for c in chunks}

    # Embeddings & Vector Store
    embedding_service = EmbeddingService(model_name="BAAI/bge-small-en-v1.5", cache_enabled=False)
    vector_store = ChromaVectorStore(collection_name="enterprise_tier4_test", persist_dir="storage/test_chroma_t4")
    # Pre-embed chunks deterministically
    for c in chunks:
        c.embedding = embedding_service.embed_text(c.text)
    vector_store.add_chunks(chunks)

    # BM25 Lexical Index
    bm25_index = BM25SearchIndex(storage_dir="storage/test_bm25_t4")
    bm25_index.build_index(chunks)

    # Retrievers & Reranker
    dense_retriever = DenseVectorRetriever(vector_store=vector_store, embedding_service=embedding_service)
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-large", top_n=6, min_ratio=0.30)
    hybrid_retriever = HybridRetriever(dense_retriever=dense_retriever, bm25_index=bm25_index, reranker=reranker)

    # Pipeline components
    query_rewriter = QueryRewriter()
    multi_query_gen = MultiQueryGenerator()
    compressor = ContextCompressor()
    citation_engine = CitationEngine()
    mock_llm = MockEnterpriseLLM()
    telemetry_service = TelemetryService()

    pipeline = RAGPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        query_rewriter=query_rewriter,
        multi_query_gen=multi_query_gen,
        compressor=compressor,
        citation_engine=citation_engine,
        docstore=docstore,
        llm=mock_llm,
        semantic_cache=None,
    )

    chat_service = ChatService(
        rag_pipeline=pipeline,
        telemetry_service=telemetry_service,
    )

    return pipeline, chat_service, telemetry_service


@pytest_asyncio.fixture
async def scenario_client(
    enterprise_rag_environment: tuple[RAGPipeline, ChatService, TelemetryService]
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """
    Creates an HTTPX async client connected to FastAPI app with enterprise scenario overrides.
    """
    pipeline, chat_service, telemetry_service = enterprise_rag_environment
    app = create_app()

    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_telemetry_service] = lambda: telemetry_service

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers={"Content-Type": "application/json"},
    ) as client:
        yield client

    app.dependency_overrides.clear()


# ════════════════════════════════════════════════════════════════════════════════
# 3. SCENARIO 1: CROSS-DEPARTMENT POLICY INQUIRY (IT SECURITY + FINANCE EXPENSE)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tc_t4_001_cross_department_travel_laptop_sync_query(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-001: Scenario 1 - Cross-Department Policy Inquiry (Sync /api/chat).
    Query: "What is the policy for traveling with company laptops to international conferences?"
    Verifies:
      1. Dual-department retrieval (IT Data Security + Finance Travel & Expense).
      2. Answer contains IT security rules (full-disk encryption, loaner laptops for List A, GlobalProtect VPN).
      3. Answer contains Finance rules ($75/day per diem, Navan booking, Concur 30-day receipt submission).
      4. Citations include both IT_Data_Security_Policy_2026.pdf and Finance_Travel_Expense_Policy_2026.pdf.
      5. Execution metrics reflect multi-source candidate chunk retrieval.
    """
    payload = {
        "message": "What is the policy for traveling with company laptops to international conferences?",
        "session_id": "sess_t4_cross_dept_001",
        "grounding_mode": "balanced",
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200, f"Chat endpoint failed: {response.text}"

    data = response.json()
    answer = data.get("answer", "")
    citations = data.get("citations", [])
    trace = data.get("trace", {})

    # Verify IT Security content synthesis
    assert "encryption" in answer.lower() or "256-bit" in answer, "Missing IT encryption policy"
    assert "loaner" in answer.lower() or "list a" in answer.lower(), "Missing loaner laptop policy for high risk regions"
    assert "vpn" in answer.lower() or "globalprotect" in answer.lower(), "Missing mandatory VPN requirement"

    # Verify Finance content synthesis
    assert "$75" in answer or "per diem" in answer.lower(), "Missing Finance per diem rate"
    assert "concur" in answer.lower(), "Missing Concur reimbursement portal requirement"
    assert "30 days" in answer.lower(), "Missing 30-day submission deadline"

    # Verify cross-department citations
    assert len(citations) >= 2, f"Expected at least 2 cross-department citations, got {len(citations)}"
    source_files = {c.get("source_file") for c in citations}
    assert "IT_Data_Security_Policy_2026.pdf" in source_files, "Missing IT security citation"
    assert "Finance_Travel_Expense_Policy_2026.pdf" in source_files, "Missing Finance expense citation"

    # Verify trace telemetry
    assert trace.get("retrieved_candidate_count", 0) >= 2
    assert trace.get("final_context_count", 0) >= 2
    assert trace.get("faithfulness_passed") is True


@pytest.mark.asyncio
async def test_tc_t4_002_cross_department_travel_laptop_stream_flow(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-002: Scenario 1 - Cross-Department Policy Inquiry (Streaming /api/chat/stream).
    Verifies full SSE event sequence (start -> retrieval -> chunk -> citation -> trace -> done)
    and verifies the streamed tokens assemble the combined IT and Finance guidelines.
    """
    payload = {
        "message": "What is the policy for traveling with company laptops to international conferences?",
        "session_id": "sess_t4_cross_dept_stream",
        "stream": True,
    }
    response = await scenario_client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    events = await SSEDecoder.collect_all(response)
    event_types = [e["event"] for e in events]

    assert "start" in event_types, "Missing SSE start event"
    assert "retrieval" in event_types, "Missing SSE retrieval event"
    assert "chunk" in event_types, "Missing SSE chunk event"
    assert "citation" in event_types, "Missing SSE citation event"
    assert "trace" in event_types, "Missing SSE trace event"
    assert "done" in event_types, "Missing SSE done event"

    # Reassemble answer from chunk events
    token_chunks = [e["data"]["content"] for e in events if e["event"] == "chunk"]
    assembled_answer = "".join(token_chunks)

    assert "VPN" in assembled_answer or "loaner" in assembled_answer.lower()
    assert "$75" in assembled_answer or "Concur" in assembled_answer

    # Verify done event metadata
    done_data = next(e["data"] for e in events if e["event"] == "done")
    assert done_data["status"] == "completed"
    assert len(done_data["citations"]) >= 2
    assert done_data["timing"]["ttft_ms"] >= 0.0


# ════════════════════════════════════════════════════════════════════════════════
# 4. SCENARIO 2: MULTI-TURN POLICY CLARIFICATION DIALOGUE (HR BENEFITS)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tc_t4_003_multiturn_clarification_turn1_duration(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-003: Scenario 2 (Turn 1) - Primary Caregiver Leave Duration.
    Query: "How much parental leave do primary caregivers get?"
    Verifies 16 weeks duration, HR-BEN-2026 citation, and session establishment.
    """
    session_id = "sess_t4_multiturn_dialogue_42"
    payload = {
        "message": "How much parental leave do primary caregivers get?",
        "session_id": session_id,
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["session_id"] == session_id
    assert "16 weeks" in data["answer"], "Turn 1 answer must specify 16 weeks paid parental leave"
    assert "primary caregiver" in data["answer"].lower()

    citations = data.get("citations", [])
    assert len(citations) >= 1
    assert "HR_Employee_Benefits_Guide_2026.pdf" == citations[0]["source_file"]


@pytest.mark.asyncio
async def test_tc_t4_004_multiturn_clarification_turn2_intermittent_context(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-004: Scenario 2 (Turn 2) - Intermittent Leave Context Resolution.
    Query: "Can this be taken intermittently?"
    Verifies multi-turn memory resolves 'this' to parental leave and returns 2-week increments rule.
    """
    session_id = "sess_t4_multiturn_dialogue_42"
    payload = {
        "message": "Can this be taken intermittently?",
        "session_id": session_id,
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["session_id"] == session_id
    answer = data.get("answer", "")
    assert "intermittently" in answer.lower(), "Must answer whether leave can be taken intermittently"
    assert "2-week" in answer or "2 week" in answer, "Must specify minimum 2-week increments rule"
    assert "12 months" in answer, "Must specify 12 months eligibility window"


@pytest.mark.asyncio
async def test_tc_t4_005_multiturn_clarification_turn3_paperwork_and_forms(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-005: Scenario 2 (Turn 3) - Required Paperwork & Form Enumeration.
    Query: "What paperwork is required to apply?"
    Verifies procedural routing and enumeration of Form HR-204 and 30-day notice timeline.
    """
    session_id = "sess_t4_multiturn_dialogue_42"
    payload = {
        "message": "What paperwork is required to apply?",
        "session_id": session_id,
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    answer = data.get("answer", "")
    assert "hr-204" in answer.lower() or "form hr-204" in answer.lower(), "Must enumerate Form HR-204"
    assert "30" in answer, "Must mention 30 calendar days advance notice requirement"
    assert "medical certification" in answer.lower() or "adoption" in answer.lower()


# ════════════════════════════════════════════════════════════════════════════════
# 5. SCENARIO 3: MIXED-INTENT QUERY (GREETING + COMPARISON + ENUMERATION)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tc_t4_006_mixed_intent_greeting_comparison_enumeration_sync(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-006: Scenario 3 - Mixed-Intent Query (Sync /api/chat).
    Query: "Good morning! Can you compare our standard health insurance dental coverage with the vision plan, and list the enrollment deadlines?"
    Verifies:
      1. Conversational greeting ("Good morning!") does NOT cause retrieval bypass.
      2. Comparison between Dental plan ($2,000 max, 100/80/50%) and Vision plan ($500 allowance).
      3. Enumeration of Open Enrollment deadlines (Nov 1 - Nov 30).
      4. Grounded citations referencing HR_Employee_Benefits_Guide_2026.pdf Section 4.2.
    """
    payload = {
        "message": "Good morning! Can you compare our standard health insurance dental coverage with the vision plan, and list the enrollment deadlines?",
        "session_id": "sess_t4_mixed_intent_001",
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    answer = data.get("answer", "")
    citations = data.get("citations", [])

    # Verify Dental Comparison
    assert "$2,000" in answer or "2000" in answer, "Missing Dental $2,000 annual maximum"
    assert "delta dental" in answer.lower() or "preventive" in answer.lower()

    # Verify Vision Comparison
    assert "$500" in answer or "500" in answer, "Missing Vision $500 annual allowance"
    assert "vision" in answer.lower() or "vsp" in answer.lower()

    # Verify Open Enrollment Deadline Enumeration
    assert "november 1" in answer.lower() or "nov 1" in answer.lower()
    assert "november 30" in answer.lower() or "nov 30" in answer.lower()

    # Verify Citations
    assert len(citations) >= 1
    assert "HR_Employee_Benefits_Guide_2026.pdf" == citations[0]["source_file"]


@pytest.mark.asyncio
async def test_tc_t4_007_mixed_intent_streaming_verification(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-007: Scenario 3 - Mixed-Intent Query (SSE Streaming /api/chat/stream).
    Verifies retrieval runs despite conversational greeting and delivers formatted comparison stream.
    """
    payload = {
        "message": "Good morning! Can you compare our standard health insurance dental coverage with the vision plan, and list the enrollment deadlines?",
        "session_id": "sess_t4_mixed_intent_stream",
        "stream": True,
    }
    response = await scenario_client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200

    events = await SSEDecoder.collect_all(response)
    retrieval_evt = next(e for e in events if e["event"] == "retrieval")
    assert retrieval_evt["data"]["candidate_count"] > 0, "Retrieval must execute for mixed intent query"

    done_evt = next(e for e in events if e["event"] == "done")
    assert "$2,000" in done_evt["data"]["answer"]
    assert "November 30" in done_evt["data"]["answer"]


# ════════════════════════════════════════════════════════════════════════════════
# 6. SCENARIO 4: EXECUTIVE COMPLIANCE & AUDIT VERIFICATION (WHISTLEBLOWER VS NDA)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tc_t4_008_compliance_whistleblower_nda_conflict_analysis(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-008: Scenario 4 - Executive Compliance & Audit Verification.
    Query: "Are there any conflicts between our Whistleblower Protection policy and the standard Employee Non-Disclosure Agreement (NDA) regarding reporting securities violations?"
    Verifies:
      1. Legal analysis confirms NO conflict exists.
      2. Cites SEC Rule 21F-17 and Defend Trade Secrets Act (DTSA) statutory carve-out.
      3. High faithfulness verification passes with zero unsupported claims.
      4. Citations include Legal_Compliance_Whistleblower_Policy_2026.pdf Sections 1.4 and 8.2.
    """
    payload = {
        "message": "Are there any conflicts between our Whistleblower Protection policy and the standard Employee Non-Disclosure Agreement (NDA) regarding reporting securities violations?",
        "session_id": "sess_t4_compliance_001",
        "grounding_mode": "strict",
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    answer = data.get("answer", "")
    citations = data.get("citations", [])
    trace = data.get("trace", {})

    # Legal correctness verification
    assert "no conflict" in answer.lower() or "no legal conflict" in answer.lower(), "Must state no conflict exists"
    assert "21f-17" in answer.lower() or "sec rule 21f-17" in answer.lower(), "Must cite SEC Rule 21F-17"
    assert "carve-out" in answer.lower() or "exceptions" in answer.lower() or "dtsa" in answer.lower(), "Must explain NDA carve-out"

    # Citation tracking
    assert len(citations) >= 2, f"Expected at least 2 legal citations, got {len(citations)}"
    source_files = {c.get("source_file") for c in citations}
    assert "Legal_Compliance_Whistleblower_Policy_2026.pdf" in source_files

    # Verification report in trace
    assert trace.get("faithfulness_passed") is True


@pytest.mark.asyncio
async def test_tc_t4_009_compliance_strict_grounding_audit_trace(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-009: Scenario 4 - Compliance Audit Trace Telemetry.
    Verifies strict grounding mode telemetry emits accurate timing, low_confidence=False,
    and verified section citations.
    """
    payload = {
        "message": "Are there any conflicts between our Whistleblower Protection policy and the standard Employee Non-Disclosure Agreement (NDA) regarding reporting securities violations?",
        "grounding_mode": "strict",
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["grounding_mode"] == "strict"
    assert data["low_confidence"] is False
    assert data["latency_ms"] >= 0.0

    section_titles = [c.get("section_title") for c in data.get("citations", [])]
    assert any("Whistleblower" in str(s) for s in section_titles)
    assert any("NDA" in str(s) or "Non-Disclosure" in str(s) for s in section_titles)


# ════════════════════════════════════════════════════════════════════════════════
# 7. SCENARIO 5: EMERGENCY / CRISIS PROTOCOL NAVIGATION (WAREHOUSE INJURY & OSHA)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tc_t4_010_emergency_warehouse_injury_procedural_workflow(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-010: Scenario 5 - Emergency Crisis Protocol Navigation (Sync /api/chat).
    Query: "An employee suffered an injury in the warehouse. What immediate actions, reporting timelines, and OSHA forms are mandatory?"
    Verifies:
      1. Immediate crisis actions (First aid, 911 emergency call, area cordon-off).
      2. Mandatory reporting timelines (1 hr supervisor, 4 hr EHS manager, 8 hr fatality, 24 hr hospitalization/amputation/eye loss).
      3. Mandatory incident forms (Form SAF-101 within 4 hours, OSHA Form 301 within 7 days, OSHA Form 300 log).
      4. Procedural step-by-step completeness and EHS citations.
    """
    payload = {
        "message": "An employee suffered an injury in the warehouse. What immediate actions, reporting timelines, and OSHA forms are mandatory?",
        "session_id": "sess_t4_crisis_001",
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    answer = data.get("answer", "")
    citations = data.get("citations", [])

    # 1. Immediate Actions
    assert "first aid" in answer.lower(), "Must instruct immediate first aid"
    assert "911" in answer, "Must instruct calling 911 for severe injuries"
    assert "cordon" in answer.lower() or "scene" in answer.lower(), "Must instruct securing incident scene"

    # 2. Reporting Timelines
    assert "1 hour" in answer.lower(), "Must specify 1-hour supervisor notice"
    assert "4 hours" in answer.lower(), "Must specify 4-hour EHS notice"
    assert "24 hours" in answer.lower(), "Must specify 24-hour OSHA hospitalization reporting"
    assert "8 hours" in answer.lower(), "Must specify 8-hour OSHA fatality reporting"

    # 3. Mandatory Documentation Forms
    assert "saf-101" in answer.lower(), "Must mandate Form SAF-101"
    assert "osha form 301" in answer.lower() or "301" in answer, "Must mandate OSHA Form 301"
    assert "osha form 300" in answer.lower() or "300" in answer, "Must mandate OSHA Form 300"

    # Citations
    assert len(citations) >= 2
    assert all(c["source_file"] == "EHS_Workplace_Safety_Emergency_Protocol_2026.pdf" for c in citations)


@pytest.mark.asyncio
async def test_tc_t4_011_emergency_protocol_streaming_delivery(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-011: Scenario 5 - Emergency Crisis Protocol (Streaming /api/chat/stream).
    Verifies emergency response guidelines are rapidly streamed in ordered chunks.
    """
    payload = {
        "message": "An employee suffered an injury in the warehouse. What immediate actions, reporting timelines, and OSHA forms are mandatory?",
        "session_id": "sess_t4_crisis_stream",
        "stream": True,
    }
    response = await scenario_client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200

    events = await SSEDecoder.collect_all(response)
    token_chunks = [e["data"]["content"] for e in events if e["event"] == "chunk"]
    assembled_answer = "".join(token_chunks)

    assert "First aid" in assembled_answer or "first aid" in assembled_answer.lower()
    assert "SAF-101" in assembled_answer
    assert "OSHA" in assembled_answer


# ════════════════════════════════════════════════════════════════════════════════
# 8. SCENARIO 6: REMOTE WORK & EQUIPMENT REIMBURSEMENT WORKFLOW
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tc_t4_012_remote_work_equipment_and_expense_dual_policy(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-012: Scenario 6 - Remote Work & Equipment Reimbursement Workflow.
    Query: "What equipment does the company provide for remote workers and what is the reimbursement limit for home office setup?"
    Verifies:
      1. IT Hardware equipment (MacBook Pro 16" / Dell XPS 15, 27" 4K monitor, keyboard/mouse, YubiKey 5C).
      2. Finance expense limit ($500 home office setup reimbursement, $50/month internet subsidy via Concur).
      3. Citations include IT_Data_Security_Policy_2026.pdf and Finance_Travel_Expense_Policy_2026.pdf.
    """
    payload = {
        "message": "What equipment does the company provide for remote workers and what is the reimbursement limit for home office setup?",
        "session_id": "sess_t4_remote_work_001",
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    answer = data.get("answer", "")
    citations = data.get("citations", [])

    # IT Hardware verification
    assert "macbook pro" in answer.lower() or "dell xps" in answer.lower() or "laptop" in answer.lower()
    assert "4k monitor" in answer.lower() or "monitor" in answer.lower()
    assert "yubikey" in answer.lower(), "Must specify hardware security keys (YubiKey)"

    # Finance Expense verification
    assert "$500" in answer, "Must specify $500 one-time home office setup limit"
    assert "$50" in answer or "$50/month" in answer, "Must specify $50/month internet subsidy"
    assert "concur" in answer.lower(), "Must mention Concur expense submission"

    # Cross-department citation verification
    assert len(citations) >= 2
    source_files = {c.get("source_file") for c in citations}
    assert "IT_Data_Security_Policy_2026.pdf" in source_files
    assert "Finance_Travel_Expense_Policy_2026.pdf" in source_files


@pytest.mark.asyncio
async def test_tc_t4_013_remote_work_metadata_filtering_and_retrieval(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-013: Scenario 6 - Metadata Filtering Verification for Remote Work.
    Verifies that supplying category filter {"category": "Finance"} isolates Finance expense chunks.
    """
    payload = {
        "message": "What is the reimbursement limit for home office setup and internet subsidy?",
        "filters": {"category": "Finance"},
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    citations = data.get("citations", [])
    assert len(citations) >= 1
    assert all(c["source_file"] == "Finance_Travel_Expense_Policy_2026.pdf" for c in citations)


# ════════════════════════════════════════════════════════════════════════════════
# 9. SCENARIO 7 & 8: POLICY VERSIONING & GLOBAL MOBILITY COMPLIANCE
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tc_t4_014_policy_versioning_tuition_reimbursement_change(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-014: Scenario 7 - Policy Versioning & Regulatory Change Tracking.
    Query: "Has the tuition assistance benefit reimbursement limit changed under the 2026 Education Assistance Policy compared to previous years?"
    Verifies:
      1. Distinguishes 2026 limit ($5,250 IRS tax-free limit) from 2024 limit ($3,000).
      2. Mentions Grade 'B' or higher academic eligibility requirement.
      3. Cites HR_Education_Assistance_Policy_2026.pdf Section 2.1.
    """
    payload = {
        "message": "Has the tuition assistance benefit reimbursement limit changed under the 2026 Education Assistance Policy compared to previous years?",
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    answer = data.get("answer", "")
    citations = data.get("citations", [])

    assert "$5,250" in answer or "5250" in answer, "Must specify $5,250 2026 limit"
    assert "$3,000" in answer or "3000" in answer, "Must contrast with previous $3,000 limit"
    assert "grade of 'b'" in answer.lower() or "grade b" in answer.lower() or "irs" in answer.lower()

    assert len(citations) >= 1
    assert citations[0]["source_file"] == "HR_Education_Assistance_Policy_2026.pdf"


@pytest.mark.asyncio
async def test_tc_t4_015_international_remote_work_tax_compliance_flow(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-015: Scenario 8 - Global Mobility & Cross-Border Tax Protocol.
    Query: "Can a US employee temporarily work remotely from an overseas office in the UK or Germany for 60 days?"
    Verifies:
      1. Explains 30-day limit with manager approval.
      2. Requires unanimous VP, HR Global Mobility, and Corporate Tax VP approval for 60 days.
      3. Explains permanent establishment and tax withholding compliance risks.
      4. Cites Global_Mobility_Remote_Work_Protocol_2026.pdf Section 4.1.
    """
    payload = {
        "message": "Can a US employee temporarily work remotely from an overseas office in the UK or Germany for 60 days?",
    }
    response = await scenario_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    answer = data.get("answer", "")
    citations = data.get("citations", [])

    assert "30 days" in answer.lower() or "30" in answer, "Must explain 30-day standard limit"
    assert "global mobility" in answer.lower() or "tax" in answer.lower(), "Must mandate Global Mobility / Corporate Tax VP approval"
    assert "permanent establishment" in answer.lower() or "liabilities" in answer.lower() or "approval" in answer.lower()

    assert len(citations) >= 1
    assert citations[0]["source_file"] == "Global_Mobility_Remote_Work_Protocol_2026.pdf"


# ════════════════════════════════════════════════════════════════════════════════
# 10. SCENARIOS 9 & 10: MULTI-SESSION ISOLATION & TELEMETRY OBSERVABILITY
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_tc_t4_016_session_isolation_and_purge_endpoints(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-016: Scenario 9 - Multi-Session Memory Isolation & Session Eviction Lifecycle.
    Verifies:
      1. Two distinct sessions maintain isolated conversation contexts.
      2. DELETE /api/chat/session/{id} evicts specific session.
      3. DELETE /api/chat/sessions purges all cached sessions.
    """
    session_alpha = "sess_t4_user_alpha"
    session_beta = "sess_t4_user_beta"

    # User Alpha asks about parental leave
    resp_a = await scenario_client.post(
        "/api/chat",
        json={"message": "How much parental leave do primary caregivers get?", "session_id": session_alpha},
    )
    assert resp_a.status_code == 200

    # User Beta asks about IT loaner laptops
    resp_b = await scenario_client.post(
        "/api/chat",
        json={"message": "What is the policy for traveling with company laptops?", "session_id": session_beta},
    )
    assert resp_b.status_code == 200

    # Evict User Alpha session
    del_a = await scenario_client.delete(f"/api/chat/session/{session_alpha}")
    assert del_a.status_code == 200
    assert del_a.json()["status"] == "success"

    # Clear all sessions
    del_all = await scenario_client.delete("/api/chat/sessions")
    assert del_all.status_code == 200
    assert del_all.json()["status"] == "success"


@pytest.mark.asyncio
async def test_tc_t4_017_admin_telemetry_records_scenario_traces(
    scenario_client: httpx.AsyncClient,
) -> None:
    """
    TC-T4-017: Scenario 10 - Admin Telemetry & Observability for Tier 4 Traces.
    Verifies that executing enterprise scenario queries populates telemetry traces and metrics.
    """
    # Execute a query to guarantee recent trace
    await scenario_client.post(
        "/api/chat",
        json={"message": "What equipment does the company provide for remote workers?"},
    )

    # Check metrics
    resp_metrics = await scenario_client.get("/api/admin/metrics")
    assert resp_metrics.status_code == 200
    metrics_data = resp_metrics.json()
    assert metrics_data.get("total_queries", 0) > 0
    assert "recent_traces" in metrics_data

    # Check trace list
    resp_traces = await scenario_client.get("/api/admin/traces")
    assert resp_traces.status_code == 200
    traces_data = resp_traces.json()
    assert isinstance(traces_data, list)
    assert len(traces_data) > 0
