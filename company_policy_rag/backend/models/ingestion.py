from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    file_paths: List[Path] = Field(..., description="List of file paths to process")
    force_rebuild: bool = Field(default=False, description="Wipe and rebuild index if true")
    chunk_strategy: Optional[str] = Field(default=None, description="Override chunking strategy")


class IngestionResult(BaseModel):
    documents_loaded: int = Field(default=0)
    chunks_created: int = Field(default=0)
    chunks_inserted: int = Field(default=0)
    chunks_skipped_unchanged: int = Field(default=0)
    files_processed: List[str] = Field(default_factory=list)
    files_skipped: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
