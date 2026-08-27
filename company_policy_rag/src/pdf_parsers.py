"""
PDF parsing with optional Marker layout extraction and PDFReader fallback.

Marker (marker-pdf) improves code/table preservation on technical PDFs such as
the AI Agents guidebook. When disabled or unavailable, PyMuPDF + PDFReader
provide the same per-page Documents as before.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader

from src.config import settings
from src.utils import logger

ParserName = Literal["marker", "pypdf"]

_FENCE_RE = re.compile(r"```[\w-]*\n.*?```", re.DOTALL)
_CODE_LINE_RE = re.compile(
    r"^\s*(def |class |import |from |async def |function |const |let |var |#include|public |private )",
    re.MULTILINE,
)


def _marker_available() -> bool:
    try:
        import marker  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_marker_device() -> str:
    requested = (settings.marker_device or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_pdf_parser() -> ParserName:
    """Pick parser based on settings and runtime availability."""
    mode = settings.pdf_parser
    if mode == "pypdf":
        return "pypdf"
    if mode == "marker":
        if settings.enable_marker_pdf and _marker_available():
            return "marker"
        logger.warning("PDF_PARSER=marker but Marker unavailable — using pypdf")
        return "pypdf"
    # auto
    if settings.enable_marker_pdf and _marker_available():
        return "marker"
    return "pypdf"


def _detect_block_types(text: str) -> list[dict[str, str | int]]:
    """Heuristic block typing for fallback parser."""
    blocks: list[dict[str, str | int]] = []
    if not text.strip():
        return blocks

    last = 0
    for match in _FENCE_RE.finditer(text):
        if match.start() > last:
            prose = text[last : match.start()].strip()
            if prose:
                blocks.append({"block_type": "text", "text": prose})
        blocks.append({"block_type": "code", "text": match.group(0).strip()})
        last = match.end()

    tail = text[last:].strip()
    if tail:
        if _CODE_LINE_RE.search(tail) and tail.count("\n") >= 2:
            blocks.append({"block_type": "code", "text": tail})
        else:
            blocks.append({"block_type": "text", "text": tail})

    if not blocks:
        blocks.append({"block_type": "text", "text": text})
    return blocks


def _load_with_pypdf(file_path: Path, base_metadata: dict) -> list[Document]:
    """Per-page Documents with optional PyMuPDF block hints and canonical page identity."""
    reader = PDFReader()
    page_docs = reader.load_data(file=file_path)
    enriched: list[Document] = []

    try:
        import fitz

        fitz_doc = fitz.open(file_path)
        use_fitz = True
    except Exception:
        fitz_doc = None
        use_fitz = False

    # Extract all page texts
    pages_text: list[tuple[int, str]] = []
    for idx, page_doc in enumerate(page_docs):
        text = page_doc.text or ""
        if use_fitz and fitz_doc is not None and idx < len(fitz_doc):
            page_text = fitz_doc[idx].get_text("text")
            if page_text.strip():
                text = page_text
        pages_text.append((idx + 1, text))

    from backend.ingestion.page_detector import PrintedPageDetector
    page_identities = PrintedPageDetector.resolve_document_pages(pages_text)

    for idx, (phys_page, text) in enumerate(pages_text):
        page_id = page_identities[idx]
        block_types: list[str] = []
        for block in _detect_block_types(text):
            block_types.append(str(block["block_type"]))

        meta = {
            **base_metadata,
            "page_number": page_id.physical_page_number,
            "physical_page_number": page_id.physical_page_number,
            "internal_page_index": page_id.internal_page_index,
            "display_page_number": page_id.display_page_number,
            "page_label": page_id.page_label,
            "parser": "pypdf",
            "block_types": ",".join(block_types) if block_types else "text",
        }
        enriched.append(Document(text=text, metadata=meta))

    if fitz_doc is not None:
        fitz_doc.close()

    return enriched


def _load_with_marker(file_path: Path, base_metadata: dict) -> list[Document]:
    """Convert PDF to markdown blocks via marker-pdf."""
    device = _resolve_marker_device()
    logger.info("Parsing %s with Marker (device=%s)", file_path.name, device)

    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as exc:
        raise ImportError("marker-pdf is not installed") from exc

    model_dict = create_model_dict(device=device)
    converter = PdfConverter(artifact_dict=model_dict)
    rendered = converter(str(file_path))

    # marker returns full markdown; split on page markers when present
    full_text = getattr(rendered, "markdown", None) or str(rendered)
    page_sections = re.split(r"\n(?:<!--\s*page\s*(\d+)\s*-->|# Page (\d+))\n", full_text, flags=re.I)

    documents: list[Document] = []
    if len(page_sections) <= 1:
        documents.append(
            Document(
                text=full_text,
                metadata={
                    **base_metadata,
                    "page_number": 1,
                    "parser": "marker",
                    "block_types": "text",
                },
            )
        )
        return documents

    # Alternating: preamble, page_num, content, page_num, content...
    i = 1
    while i < len(page_sections):
        page_num_str = page_sections[i]
        content = page_sections[i + 1] if i + 1 < len(page_sections) else ""
        page_number = int(page_num_str) if page_num_str and page_num_str.isdigit() else None
        block_types = [b["block_type"] for b in _detect_block_types(content)]
        documents.append(
            Document(
                text=content.strip(),
                metadata={
                    **base_metadata,
                    "page_number": page_number,
                    "parser": "marker",
                    "block_types": ",".join(block_types) if block_types else "text",
                },
            )
        )
        i += 2

    if not documents:
        documents.append(
            Document(
                text=full_text,
                metadata={**base_metadata, "page_number": 1, "parser": "marker"},
            )
        )
    return documents


def _parse_page_number(page_label: str | int | None) -> int | None:
    if page_label is None:
        return None
    if isinstance(page_label, int):
        return page_label
    digits = "".join(c for c in str(page_label) if c.isdigit())
    return int(digits) if digits else None


def load_pdf_as_documents(
    file_path: Path,
    *,
    base_metadata: dict | None = None,
    parser: ParserName | None = None,
) -> list[Document]:
    """
    Load a PDF into page-level Documents using Marker or PDFReader fallback.

    base_metadata is merged into every page Document (source_file, file_hash, etc.).
    """
    base = dict(base_metadata or {})
    chosen = parser or resolve_pdf_parser()

    try:
        if chosen == "marker":
            docs = _load_with_marker(file_path, base)
        else:
            docs = _load_with_pypdf(file_path, base)
    except Exception as exc:
        if chosen == "marker":
            logger.warning(
                "Marker failed for %s (%s) — falling back to pypdf", file_path.name, exc
            )
            docs = _load_with_pypdf(file_path, base)
        else:
            raise

    logger.info("Loaded %d pages from %s via %s", len(docs), file_path.name, chosen)
    return docs