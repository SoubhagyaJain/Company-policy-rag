from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
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
    page_number: Optional[int] = Field(default=None, description="1-indexed page number if applicable")
    page_label: Optional[str] = Field(default=None)
    total_pages: Optional[int] = Field(default=None)
    section_title: Optional[str] = Field(default=None)
    section_number: Optional[str] = Field(default=None)
    section_path: Optional[str] = Field(default=None, description="e.g. 'I. GENERAL > A. At-Will'")
    section_level: Optional[int] = Field(default=None)
    has_tables: bool = Field(default=False)
    has_code: bool = Field(default=False)
    extra: Dict[str, Any] = Field(default_factory=dict)


class RawDocument(BaseModel):
    id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}")
    content: str = Field(..., description="Raw text or structured markdown content")
    metadata: DocumentMetadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
