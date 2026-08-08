from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChunkRole(str, Enum):
    PARENT = "parent"
    CHILD = "child"
    STANDALONE = "standalone"


class ContentType(str, Enum):
    PROSE = "prose"
    CODE = "code"
    TABLE = "table"
    MIXED = "mixed"


class ChunkMetadata(BaseModel):
    document_id: str
    source_file: str
    file_path: str
    file_hash: str
    document_type: str
    category: str = "general"
    chunk_index: int = 0
    page_number: int | None = None
    section_title: str | None = None
    section_number: str | None = None
    section_path: str | None = None
    section_level: int | None = None
    chunk_strategy: str = Field(..., description="Strategy used: recursive, semantic, markdown, etc.")
    node_role: ChunkRole = ChunkRole.STANDALONE
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    content_type: ContentType = ContentType.PROSE
    is_atomic: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}")
    text: str = Field(..., description="Chunk content text")
    metadata: ChunkMetadata
    embedding: list[float] | None = Field(default=None, description="Optional vector embedding")
    token_count: int = Field(default=0)
