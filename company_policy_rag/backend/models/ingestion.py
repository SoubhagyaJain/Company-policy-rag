from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    file_paths: list[Path] = Field(..., description="List of file paths to process")
    force_rebuild: bool = Field(default=False, description="Wipe and rebuild index if true")
    chunk_strategy: str | None = Field(default=None, description="Override chunking strategy")


class IngestionResult(BaseModel):
    documents_loaded: int = Field(default=0)
    chunks_created: int = Field(default=0)
    chunks_inserted: int = Field(default=0)
    chunks_skipped_unchanged: int = Field(default=0)
    files_processed: list[str] = Field(default_factory=list)
    files_skipped: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
