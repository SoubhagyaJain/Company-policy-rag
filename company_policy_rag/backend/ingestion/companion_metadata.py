"""Optional, source-verified PDF navigation hints for the existing lexical index.

Companions are local operator configuration, never evidence or citation sources.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator

from backend.utils.logging import logger

COMPANION_DIR = Path(__file__).resolve().parents[2] / "data" / "companions"


class Location(BaseModel):
    pdf_page_start: StrictInt = Field(ge=1)
    pdf_page_end: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def valid_range(self):
        if self.pdf_page_end < self.pdf_page_start:
            raise ValueError("page range ends before it starts")
        return self


class Heading(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    source_location: Location
    chapter_id: str
    section_id: str | None = None
    subsection_id: str | None = None
    parent_section_id: str | None = None


class Identity(BaseModel):
    source_sha256: str
    page_count: StrictInt = Field(ge=1)

    @field_validator("source_sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError("source_sha256 must be a full SHA-256 digest")
        return value.lower()


class Companion(BaseModel):
    metadata_schema_version: str
    document: Identity
    chapters: list[Heading]
    sections: list[Heading]
    subsections: list[Heading]

    @model_validator(mode="after")
    def validate_structure(self):
        if self.metadata_schema_version != "1.0":
            raise ValueError("unsupported companion schema")
        for rows, attr in (
            (self.chapters, "chapter_id"),
            (self.sections, "section_id"),
            (self.subsections, "subsection_id"),
        ):
            ids = [getattr(row, attr) for row in rows]
            if any(not item for item in ids) or len(ids) != len(set(ids)):
                raise ValueError(f"missing or duplicate {attr}")
            if any(row.source_location.pdf_page_end > self.document.page_count for row in rows):
                raise ValueError("heading page outside document")
        chapter_ids = {row.chapter_id for row in self.chapters}
        section_by_id = {row.section_id: row for row in self.sections}
        if any(row.chapter_id not in chapter_ids for row in self.sections + self.subsections):
            raise ValueError("unknown chapter reference")
        if any(
            row.parent_section_id not in section_by_id
            or section_by_id[row.parent_section_id].chapter_id != row.chapter_id
            for row in self.subsections
        ):
            raise ValueError("unknown or inconsistent subsection parent")
        return self


def _normalize(value: str) -> str:
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def enrich_chunks(chunks: list[Any], raw_docs: list[Any], source_hash: str, companion_dir: Path | None = None) -> int:
    """Add verified heading hints to original chunks; return number enriched.

    No metadata page is ever written into a citation. The source hash, page count,
    structural links and each heading's claimed starting page must agree first.
    """
    directory = companion_dir if companion_dir is not None else COMPANION_DIR
    matches: list[Companion] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("document"), dict):
                raise ValueError("missing document identity")
            if str(data["document"].get("source_sha256", "")).lower() != source_hash.lower():
                continue
            matches.append(Companion.model_validate(data))
        except (ValueError, OSError, TypeError) as exc:
            logger.warning("Invalid companion %s: %s", path.name, exc)
    if len(matches) != 1:
        if len(matches) > 1:
            logger.warning("Multiple companions match source %s; ignoring all", source_hash)
        return 0
    companion = matches[0]
    pages: dict[int, str] = {}
    total_pages = set()
    for doc in raw_docs:
        meta = doc.metadata
        if meta.total_pages is not None:
            total_pages.add(meta.total_pages)
        if meta.page_number and not (meta.extra or {}).get("is_visual_extraction"):
            pages[meta.page_number] = _normalize(doc.content)
    if total_pages != {companion.document.page_count}:
        logger.warning("Companion page count does not match source %s", source_hash)
        return 0
    page_hints: dict[int, list[str]] = {}
    rejected = 0
    for heading in companion.chapters + companion.sections + companion.subsections:
        location = heading.source_location
        title = _normalize(heading.title)
        if not title or title not in pages.get(location.pdf_page_start, ""):
            rejected += 1
            continue
        # Only bounded page ranges; this is a navigation hint, never source text.
        for page in range(location.pdf_page_start, location.pdf_page_end + 1):
            page_hints.setdefault(page, []).append(heading.title)
    count = 0
    for chunk in chunks:
        titles = page_hints.get(chunk.metadata.page_number, [])
        if titles:
            chunk.metadata.extra["companion_navigation"] = " | ".join(dict.fromkeys(titles))[:2000]
            count += 1
    logger.info("Companion source=%s enriched_chunks=%d rejected_headings=%d", source_hash, count, rejected)
    return count
