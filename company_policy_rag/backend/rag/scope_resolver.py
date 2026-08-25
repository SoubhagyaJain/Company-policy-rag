"""
Document Scope Resolver for Document-Aware RAG Retrieval.

Resolves query scope across:
- Mode A (CURRENT_DOCUMENT): "this document", "the doc", "this PDF", "page 59", "the uploaded file"
- Mode B (SELECTED_DOCUMENTS): "compare these two PDFs", selected document IDs
- Mode C (GLOBAL): "in the knowledge base", "across all documents", or unrestricted default
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.utils.logging import logger


class DocumentRetrievalScope(str, Enum):
    CURRENT_DOCUMENT = "current_document"
    SELECTED_DOCUMENTS = "selected_documents"
    GLOBAL = "global"


class DocumentScopeDecision(BaseModel):
    scope: DocumentRetrievalScope = DocumentRetrievalScope.GLOBAL
    active_document_id: str | None = None
    active_document_name: str | None = None
    allowed_document_ids: list[str] = Field(default_factory=list)
    page_number: int | None = None
    section_number: str | None = None
    is_structural_query: bool = False
    structural_subqueries: list[str] = Field(default_factory=list)
    reasoning: str = ""


class DocumentScopeResolver:
    """
    Analyzes natural language queries and runtime context to determine
    hard document retrieval boundaries, preventing cross-document bleed.
    """

    # Regex patterns for reference anaphora
    _CURRENT_DOC_PATTERNS = [
        re.compile(r"\b(?:this|the)\s+(?:document|doc|pdf|file|upload|uploaded\s+file|paper|handbook|guidebook|report|text|manual)\b", re.IGNORECASE),
        re.compile(r"\b(?:current\s+document|current\s+file|current\s+pdf|active\s+document)\b", re.IGNORECASE),
        re.compile(r"\b(?:in|from|about)\s+(?:this|the)\s+(?:doc|document|pdf|file|upload|handbook|guidebook|manual)\b", re.IGNORECASE),
        re.compile(r"\b(?:summarize|outline|explain)\s+(?:this|the)\s+(?:doc|document|pdf|file)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+(?:in|on)\s+(?:this|the)\s+(?:doc|document|pdf|file)\b", re.IGNORECASE),
        re.compile(r"\b(?:top\s+projects|main\s+projects|all\s+projects|key\s+projects|projects)\s+in\s+(?:this|the)\s+(?:doc|document|pdf|file)\b", re.IGNORECASE),
    ]

    _PAGE_PATTERN = re.compile(r"\b(?:page|p\.?|pg\.?)\s*(\d+)\b", re.IGNORECASE)
    _SECTION_PATTERN = re.compile(r"\b(?:section|sec\.?)\s*([\d\.]+)\b", re.IGNORECASE)

    _GLOBAL_PATTERNS = [
        re.compile(r"\b(?:all\s+documents|across\s+all\s+(?:docs|documents|files|pdfs))\b", re.IGNORECASE),
        re.compile(r"\b(?:entire\s+knowledge\s+base|in\s+(?:the\s+)?knowledge\s+base|in\s+(?:the\s+)?kb|global\s+search|everywhere)\b", re.IGNORECASE),
        re.compile(r"\b(?:all\s+policies|all\s+files|across\s+the\s+company)\b", re.IGNORECASE),
    ]

    _COMPARISON_PATTERNS = [
        re.compile(r"\b(?:compare\s+(?:these|the|both)\s+(?:two\s+)?(?:docs|documents|files|pdfs))\b", re.IGNORECASE),
        re.compile(r"\b(?:between\s+(?:these|the|both)\s+(?:two\s+)?(?:docs|documents|files|pdfs))\b", re.IGNORECASE),
        re.compile(r"\b(?:differences?\s+between\s+(?:these|the)\s+(?:two\s+)?(?:docs|documents|files|pdfs))\b", re.IGNORECASE),
    ]

    _STRUCTURAL_PATTERNS = [
        re.compile(r"\b(top\s+projects|main\s+projects|project\s+ideas|all\s+projects|best\s+ideas|key\s+concepts|key\s+projects|major\s+projects|list\s+(?:all\s+)?projects|project\s+list)\b", re.IGNORECASE),
        re.compile(r"\b(chapters|sections|table\s+of\s+contents|outline|overview|structure\s+of\s+(?:this|the)\s+(?:doc|document|pdf))\b", re.IGNORECASE),
    ]

    @classmethod
    def detect_document_reference(cls, query: str) -> bool:
        """Check if query explicitly refers to a singular active/uploaded document."""
        if not query:
            return False
        return any(pat.search(query) for pat in cls._CURRENT_DOC_PATTERNS)

    @classmethod
    def detect_page_number(cls, query: str) -> int | None:
        """Extract explicit page number target (e.g. 'page 59', 'p. 12')."""
        if not query:
            return None
        match = cls._PAGE_PATTERN.search(query)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                return None
        return None

    @classmethod
    def detect_section_number(cls, query: str) -> str | None:
        """Extract explicit section number (e.g. 'section 5.1', 'sec 3')."""
        if not query:
            return None
        match = cls._SECTION_PATTERN.search(query)
        if match:
            try:
                return match.group(1)
            except IndexError:
                return None
        return None

    @classmethod
    def detect_global_intent(cls, query: str) -> bool:
        """Check if query explicitly asks for global search across all documents."""
        if not query:
            return False
        return any(pat.search(query) for pat in cls._GLOBAL_PATTERNS)

    @classmethod
    def detect_comparison_intent(cls, query: str) -> bool:
        """Check if query asks to compare specific documents."""
        if not query:
            return False
        return any(pat.search(query) for pat in cls._COMPARISON_PATTERNS)

    @classmethod
    def detect_structural_query(cls, query: str) -> bool:
        """Check if query is asking for document structure, projects, chapters, or overview."""
        if not query:
            return False
        return any(pat.search(query) for pat in cls._STRUCTURAL_PATTERNS)

    @classmethod
    def generate_structural_subqueries(cls, query: str) -> list[str]:
        """Generate targeted sub-queries for document structure, project listings, and sections."""
        subqueries = [query]
        q_lower = query.lower()

        if any(term in q_lower for term in ["project", "projects"]):
            subqueries.extend([
                "projects",
                "key systems",
                "system architecture",
                "applications and implementations",
                "case studies and projects",
            ])
        elif any(term in q_lower for term in ["chapter", "section", "table of contents", "outline"]):
            subqueries.extend([
                "table of contents",
                "overview",
                "sections and chapters",
                "summary of topics",
            ])
        elif any(term in q_lower for term in ["concept", "ideas", "best ideas"]):
            subqueries.extend([
                "key concepts",
                "core principles",
                "main takeaways",
            ])

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for sq in subqueries:
            if sq not in seen:
                seen.add(sq)
                deduped.append(sq)
        return deduped

    def resolve_scope(
        self,
        query: str,
        active_document_id: str | None = None,
        active_document_name: str | None = None,
        selected_document_ids: list[str] | None = None,
        explicit_scope: str | None = None,
        known_documents: dict[str, str] | None = None,  # mapping: doc_id -> filename
    ) -> DocumentScopeDecision:
        """
        Resolve the final search scope and hard document boundary filters for a query.
        """
        clean_q = (query or "").strip()
        has_doc_ref = self.detect_document_reference(clean_q)
        has_global_ref = self.detect_global_intent(clean_q)
        has_comp_ref = self.detect_comparison_intent(clean_q)
        page_num = self.detect_page_number(clean_q)
        sec_num = self.detect_section_number(clean_q)
        is_structural = self.detect_structural_query(clean_q)
        structural_subqueries = (
            self.generate_structural_subqueries(clean_q) if is_structural else [clean_q]
        )

        # 1. Check if a known document filename is explicitly mentioned in the query
        resolved_doc_id = active_document_id
        resolved_doc_name = active_document_name

        if known_documents:
            for doc_id, filename in known_documents.items():
                if filename and (filename.lower() in clean_q.lower() or filename.split(".")[0].lower() in clean_q.lower()):
                    resolved_doc_id = doc_id
                    resolved_doc_name = filename
                    has_doc_ref = True
                    break

        # 2. Handle Explicit Override from Request
        if explicit_scope:
            scope_str = explicit_scope.lower().strip()
            if scope_str in ("current_document", "active_document", "document"):
                target_id = resolved_doc_id or (selected_document_ids[0] if selected_document_ids else None)
                allowed = [target_id] if target_id else []
                return DocumentScopeDecision(
                    scope=DocumentRetrievalScope.CURRENT_DOCUMENT,
                    active_document_id=target_id,
                    active_document_name=resolved_doc_name,
                    allowed_document_ids=allowed,
                    page_number=page_num,
                    section_number=sec_num,
                    is_structural_query=is_structural,
                    structural_subqueries=structural_subqueries,
                    reasoning="Explicit CURRENT_DOCUMENT scope requested by client.",
                )
            elif scope_str in ("selected_documents", "selected"):
                allowed = list(selected_document_ids or [])
                return DocumentScopeDecision(
                    scope=DocumentRetrievalScope.SELECTED_DOCUMENTS,
                    active_document_id=resolved_doc_id,
                    active_document_name=resolved_doc_name,
                    allowed_document_ids=allowed,
                    page_number=page_num,
                    section_number=sec_num,
                    is_structural_query=is_structural,
                    structural_subqueries=structural_subqueries,
                    reasoning="Explicit SELECTED_DOCUMENTS scope requested by client.",
                )
            elif scope_str in ("global", "all"):
                return DocumentScopeDecision(
                    scope=DocumentRetrievalScope.GLOBAL,
                    active_document_id=resolved_doc_id,
                    active_document_name=resolved_doc_name,
                    allowed_document_ids=[],
                    page_number=page_num,
                    section_number=sec_num,
                    is_structural_query=is_structural,
                    structural_subqueries=structural_subqueries,
                    reasoning="Explicit GLOBAL scope requested by client.",
                )

        # 3. Explicit Global Intent in Query
        if has_global_ref:
            return DocumentScopeDecision(
                scope=DocumentRetrievalScope.GLOBAL,
                active_document_id=resolved_doc_id,
                active_document_name=resolved_doc_name,
                allowed_document_ids=[],
                page_number=page_num,
                section_number=sec_num,
                is_structural_query=is_structural,
                structural_subqueries=structural_subqueries,
                reasoning="Query explicitly requests search across the entire knowledge base.",
            )

        # 4. Explicit Multi-Document Comparison Intent
        if (has_comp_ref or (selected_document_ids and len(selected_document_ids) > 1)) and selected_document_ids:
            return DocumentScopeDecision(
                scope=DocumentRetrievalScope.SELECTED_DOCUMENTS,
                active_document_id=resolved_doc_id,
                active_document_name=resolved_doc_name,
                allowed_document_ids=list(selected_document_ids),
                page_number=page_num,
                section_number=sec_num,
                is_structural_query=is_structural,
                structural_subqueries=structural_subqueries,
                reasoning=f"Query requests multi-document comparison across {len(selected_document_ids)} documents.",
            )

        # 5. Document Reference Anaphora OR Active Document Binding
        # "this document", "the doc", "the uploaded file", "page 59", etc.
        if has_doc_ref or page_num is not None or sec_num is not None or resolved_doc_id is not None:
            target_id = resolved_doc_id or (selected_document_ids[0] if selected_document_ids else None)
            if target_id:
                return DocumentScopeDecision(
                    scope=DocumentRetrievalScope.CURRENT_DOCUMENT,
                    active_document_id=target_id,
                    active_document_name=resolved_doc_name,
                    allowed_document_ids=[target_id],
                    page_number=page_num,
                    section_number=sec_num,
                    is_structural_query=is_structural,
                    structural_subqueries=structural_subqueries,
                    reasoning=f"Document reference resolved to active document ID: {target_id}.",
                )
            elif resolved_doc_name:
                return DocumentScopeDecision(
                    scope=DocumentRetrievalScope.CURRENT_DOCUMENT,
                    active_document_id=None,
                    active_document_name=resolved_doc_name,
                    allowed_document_ids=[resolved_doc_name],
                    page_number=page_num,
                    section_number=sec_num,
                    is_structural_query=is_structural,
                    structural_subqueries=structural_subqueries,
                    reasoning=f"Document reference resolved to active document name: {resolved_doc_name}.",
                )
            elif has_doc_ref:
                # User asked about "the doc" but no active_document_id was bound
                logger.warning("Query refers to 'the doc' / 'this document', but no active document ID is set in session or request.")
                return DocumentScopeDecision(
                    scope=DocumentRetrievalScope.CURRENT_DOCUMENT,
                    active_document_id=None,
                    active_document_name=None,
                    allowed_document_ids=[],
                    page_number=page_num,
                    section_number=sec_num,
                    is_structural_query=is_structural,
                    structural_subqueries=structural_subqueries,
                    reasoning="Query referenced 'the doc' but no active document identity was available.",
                )

        # 6. Default to GLOBAL Knowledge Base Retrieval
        return DocumentScopeDecision(
            scope=DocumentRetrievalScope.GLOBAL,
            active_document_id=None,
            active_document_name=None,
            allowed_document_ids=[],
            page_number=page_num,
            section_number=sec_num,
            is_structural_query=is_structural,
            structural_subqueries=structural_subqueries,
            reasoning="Unrestricted global knowledge base search.",
        )
