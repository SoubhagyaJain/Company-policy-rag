from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.ingestion.loaders.base import BaseLoader
from backend.models.document import DocumentMetadata, DocumentType, RawDocument
from backend.utils.logging import logger
from backend.utils.section_tracker import SectionTracker

_CODE_PATTERN = re.compile(r"```|^\s*(def |class |import |function |const |let |var )", re.MULTILINE)
_TABLE_PATTERN = re.compile(r"\|.*\|.*\||(?:\+[-+]+\+)|(?:\t+[^\t\n]+\t+)")


class PDFLoader(BaseLoader):
    """Loader for PDF documents with PyMuPDF (fitz) / pypdf support."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def load(
        self,
        file_path: Path,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawDocument]:
        base_meta = self._build_base_metadata(file_path, DocumentType.PDF, base_metadata)
        documents: List[RawDocument] = []
        section_tracker = SectionTracker()

        # Try fitz (PyMuPDF) first
        fitz_pages = self._read_with_fitz(file_path)
        if fitz_pages is not None:
            total_pages = len(fitz_pages)
            for idx, (page_num, text) in enumerate(fitz_pages, start=1):
                if not text.strip():
                    continue
                # Update section tracker with lines on this page
                current_ctx = None
                for line in text.splitlines():
                    ctx = section_tracker.process_line(line)
                    if ctx:
                        current_ctx = ctx
                if current_ctx is None:
                    current_ctx = section_tracker.current_context()

                has_code = bool(_CODE_PATTERN.search(text))
                has_tables = bool(_TABLE_PATTERN.search(text))

                page_meta = base_meta.model_copy(
                    update={
                        "page_number": page_num,
                        "page_label": str(page_num),
                        "total_pages": total_pages,
                        "section_title": current_ctx.section_title,
                        "section_number": current_ctx.section_number,
                        "section_path": current_ctx.section_path,
                        "section_level": current_ctx.section_level,
                        "has_code": has_code,
                        "has_tables": has_tables,
                    }
                )
                documents.append(RawDocument(content=text, metadata=page_meta))
            return documents

        # Fallback to pypdf
        pypdf_pages = self._read_with_pypdf(file_path)
        total_pages = len(pypdf_pages)
        for idx, (page_num, text) in enumerate(pypdf_pages, start=1):
            if not text.strip():
                continue
            current_ctx = None
            for line in text.splitlines():
                ctx = section_tracker.process_line(line)
                if ctx:
                    current_ctx = ctx
            if current_ctx is None:
                current_ctx = section_tracker.current_context()

            has_code = bool(_CODE_PATTERN.search(text))
            has_tables = bool(_TABLE_PATTERN.search(text))

            page_meta = base_meta.model_copy(
                update={
                    "page_number": page_num,
                    "page_label": str(page_num),
                    "total_pages": total_pages,
                    "section_title": current_ctx.section_title,
                    "section_number": current_ctx.section_number,
                    "section_path": current_ctx.section_path,
                    "section_level": current_ctx.section_level,
                    "has_code": has_code,
                    "has_tables": has_tables,
                }
            )
            documents.append(RawDocument(content=text, metadata=page_meta))

        return documents

    def _read_with_fitz(self, file_path: Path) -> Optional[List[tuple[int, str]]]:
        try:
            import fitz

            doc = fitz.open(file_path)
            pages = []
            for i in range(len(doc)):
                page = doc[i]
                get_text_fn = getattr(page, "get_text", None)
                text = str(get_text_fn("text")) if callable(get_text_fn) else ""
                pages.append((i + 1, text))
            doc.close()
            return pages
        except Exception as e:
            logger.debug("fitz PDF extraction failed for %s: %s", file_path, e)
            return None

    def _read_with_pypdf(self, file_path: Path) -> List[tuple[int, str]]:
        try:
            import pypdf

            reader = pypdf.PdfReader(str(file_path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append((i + 1, text))
            return pages
        except Exception as e:
            logger.error("pypdf extraction failed for %s: %s", file_path, e)
            return []
