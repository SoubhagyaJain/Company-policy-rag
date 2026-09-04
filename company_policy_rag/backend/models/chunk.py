from __future__ import annotations

import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from backend.models.page_identity import PageIdentity


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
    file_path: str = ""
    file_hash: str = ""
    document_type: str = "pdf"
    category: str = "general"
    department: str | None = Field(default=None, description="Extracted department / organizational scope")
    effective_date: str | None = Field(default=None, description="Extracted document effective date")
    policy_id: str | None = Field(default=None, description="Extracted policy ID / document number")
    key_entities: list[str] = Field(default_factory=list, description="Extracted key entities")
    topic_tags: list[str] = Field(default_factory=list, description="Extracted topic tags")
    chunk_index: int = 0
    page_number: int | None = None
    internal_page_index: int | None = None
    display_page_number: str | int | None = None
    page_label: str | None = None
    section_title: str | None = None
    section_number: str | None = None
    clause_id: str | None = Field(default=None, description="Stable clause or subclause identifier")
    parent_section: str | None = Field(default=None, description="Immediate governing section identifier")
    chunk_type: str = Field(default="policy_clause", description="Policy clause, exception, definition, table, or prose")
    section_path: str | None = None
    section_level: int | None = None
    chunk_strategy: str = Field(default="recursive", description="Strategy used: recursive, semantic, markdown, etc.")
    node_role: ChunkRole = ChunkRole.STANDALONE
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    content_type: ContentType = ContentType.PROSE
    has_code: bool = False
    has_tables: bool = False
    is_atomic: bool = False
    image_assets: list[dict[str, Any]] = Field(default_factory=list)
    visual_asset_ids: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    def get_page_identity(self) -> PageIdentity:
        from backend.models.page_identity import PageIdentity

        extra = self.extra or {}

        def _optional_int(value: Any) -> int | None:
            if value is None or isinstance(value, bool):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        internal_page_index = self.internal_page_index
        if internal_page_index is None:
            internal_page_index = _optional_int(extra.get("internal_page_index"))

        physical_page_number = self.page_number
        if physical_page_number is None:
            physical_page_number = _optional_int(
                extra.get("physical_page_number", extra.get("page_number"))
            )

        display_page_number = self.display_page_number
        if display_page_number is None:
            display_page_number = extra.get("display_page_number")

        page_label = self.page_label
        if page_label is None and extra.get("page_label") is not None:
            page_label = str(extra["page_label"])

        return PageIdentity.from_indices(
            internal_page_index=internal_page_index,
            physical_page_number=physical_page_number,
            display_page_number=display_page_number,
            page_label=page_label,
        )


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}")
    text: str = Field(..., description="Chunk content text")
    metadata: ChunkMetadata
    embedding: list[float] | None = Field(default=None, description="Optional vector embedding")
    token_count: int = Field(default=0)
