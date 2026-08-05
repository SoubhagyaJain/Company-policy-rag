from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
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
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    section_path: Optional[str] = None
    section_level: Optional[int] = None
    chunk_strategy: str = Field(..., description="Strategy used: recursive, semantic, markdown, etc.")
    node_role: ChunkRole = ChunkRole.STANDALONE
    parent_id: Optional[str] = None
    child_ids: List[str] = Field(default_factory=list)
    content_type: ContentType = ContentType.PROSE
    is_atomic: bool = False
    extra: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}")
    text: str = Field(..., description="Chunk content text")
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = Field(default=None, description="Optional vector embedding")
    token_count: int = Field(default=0)
