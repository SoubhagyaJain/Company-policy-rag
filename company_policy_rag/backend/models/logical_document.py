from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockType(str, Enum):
    TEXT = "text"
    CODE = "code"
    CODE_SCREENSHOT = "code_screenshot"
    DIAGRAM = "diagram"
    TABLE = "table"
    SCANNED_PAGE = "scanned_page"
    CONTINUATION_REF = "continuation_ref"


CONTINUATION_PATTERNS = [
    re.compile(r"here'?s\s+(?:the\s+)?code(?:\s+below)?(?:\s*:)?", re.IGNORECASE),
    re.compile(r"here\s+is\s+(?:the\s+)?code(?:\s+below)?(?:\s*:)?", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:the\s+)?following\s+code(?:\s+implements|\s+defines|\s+shows|\s*:)?", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*code\s*:\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|\n)\s*implementation\s*:\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"see\s+(?:the\s+)?code\s+below", re.IGNORECASE),
    re.compile(r"check\s+(?:this|the)\s+code", re.IGNORECASE),
    re.compile(r"let'?s\s+implement", re.IGNORECASE),
    re.compile(r"(?:output\s+format|expected\s+output)\s*:", re.IGNORECASE),
    re.compile(r"(?:\(continued\)|continued\s+on\s+next\s+page|\.\.\.\s*continued)", re.IGNORECASE),
    re.compile(r"see\s+(?:figure|diagram|architecture|workflow)\s+below", re.IGNORECASE),
    re.compile(r"(?:depicted|illustrated|shown)\s+below", re.IGNORECASE),
    re.compile(r"(?:figure|diagram|chart|table|visual)\s+below", re.IGNORECASE),
]


def detect_continuation_signals(text: str) -> list[str]:
    """Detect cross-page continuation cues in text."""
    if not text:
        return []
    signals: list[str] = []
    for pattern in CONTINUATION_PATTERNS:
        match = pattern.search(text)
        if match:
            signals.append(match.group(0).strip())
    return signals


@dataclass
class DocumentBlock:
    """An individual atomic block of content within a logical section."""
    id: str = field(default_factory=lambda: f"blk_{uuid.uuid4().hex[:8]}")
    block_type: BlockType = BlockType.TEXT
    text: str = ""
    raw_code: str | None = None
    page_number: int = 1
    section_title: str | None = None
    section_number: str | None = None
    section_path: str | None = None
    image_hash: str | None = None
    image_bytes: bytes | None = None
    is_visual_extraction: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LogicalSection:
    """A multi-page logical section that maintains continuity across page breaks."""
    id: str = field(default_factory=lambda: f"sec_{uuid.uuid4().hex[:8]}")
    title: str = "General"
    section_number: str | None = None
    section_path: str = "General"
    level: int = 1
    start_page: int = 1
    end_page: int = 1
    blocks: list[DocumentBlock] = field(default_factory=list)
    has_code: bool = False
    has_diagrams: bool = False
    has_tables: bool = False
    continuation_signals: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_block(self, block: DocumentBlock) -> None:
        self.blocks.append(block)
        if block.page_number > self.end_page:
            self.end_page = block.page_number
        if block.block_type in (BlockType.CODE, BlockType.CODE_SCREENSHOT):
            self.has_code = True
        elif block.block_type == BlockType.DIAGRAM:
            self.has_diagrams = True
        elif block.block_type == BlockType.TABLE:
            self.has_tables = True

    def get_full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    def get_code_blocks(self) -> list[DocumentBlock]:
        return [b for b in self.blocks if b.block_type in (BlockType.CODE, BlockType.CODE_SCREENSHOT)]


@dataclass
class LogicalDocument:
    """Document representation preserving cross-page logical sections and multi-modal blocks."""
    document_id: str
    filename: str
    file_path: str
    total_pages: int
    sections: list[LogicalSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def find_section_by_page(self, page_number: int) -> LogicalSection | None:
        for sec in self.sections:
            if sec.start_page <= page_number <= sec.end_page:
                return sec
        return None

    def find_sections_by_title(self, title_query: str) -> list[LogicalSection]:
        q = title_query.lower()
        return [s for s in self.sections if q in s.title.lower() or q in s.section_path.lower()]
