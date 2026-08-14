from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    UNKNOWN = "unknown"


class DocumentCategory(str, Enum):
    POLICY = "policy"
    LEGAL = "legal"
    GUIDEBOOK = "guidebook"
    GENERAL = "general"


class DocumentMetadata(BaseModel):
    source_file: str = Field(..., description="Original filename e.g. employee_handbook.pdf")
    file_path: str = Field(..., description="Relative or absolute file system path")
    file_hash: str = Field(..., description="SHA-256 or fast hash for change detection")
    document_type: DocumentType = Field(default=DocumentType.UNKNOWN)
    category: str = Field(default="general")
    department: str | None = Field(default=None, description="Department owner e.g. HR, Legal, IT, Finance")
    effective_date: str | None = Field(default=None, description="Effective or revision date (ISO 8601)")
    policy_id: str | None = Field(default=None, description="Formal policy identifier e.g. POL-HR-001")
    key_entities: list[str] = Field(default_factory=list, description="Extracted key entities")
    topic_tags: list[str] = Field(default_factory=list, description="Extracted domain topic tags")
    page_number: int | None = Field(default=None, description="1-indexed page number if applicable")
    page_label: str | None = Field(default=None)
    total_pages: int | None = Field(default=None)
    section_title: str | None = Field(default=None)
    section_number: str | None = Field(default=None)
    section_path: str | None = Field(default=None, description="e.g. 'I. GENERAL > A. At-Will'")
    section_level: int | None = Field(default=None)
    has_tables: bool = Field(default=False)
    has_code: bool = Field(default=False)
    extra: dict[str, Any] = Field(default_factory=dict)


class ExtractedDocumentMetadata(BaseModel):
    department: str = Field(default="General", description="Normalized canonical department code (HR, IT, Finance, Legal, Operations, Engineering, General)")
    category: str = Field(default="general", description="Document category (policy, legal, guidebook, general)")
    effective_date: str | None = Field(default=None, description="ISO 8601 formatted date string YYYY-MM-DD")
    policy_id: str | None = Field(default=None, description="Alphanumeric policy identifier/code e.g. POL-HR-001")
    key_entities: list[str] = Field(default_factory=list, description="Extracted roles, monetary limits, durations, deadlines")
    topic_tags: list[str] = Field(default_factory=list, description="Categorized topic taxonomy tags")
    confidence: float = Field(default=1.0, description="Overall confidence metric (0.0 - 1.0)")
    confidence_scores: dict[str, float] = Field(default_factory=dict, description="Confidence metric per extracted field (0.0 - 1.0)")
    extraction_method: str = Field(default="heuristic", description="heuristic | llm | hybrid")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional unstructured extracted attributes")


class RawDocument(BaseModel):
    id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}")
    content: str = Field(..., description="Raw text or structured markdown content")
    metadata: DocumentMetadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
