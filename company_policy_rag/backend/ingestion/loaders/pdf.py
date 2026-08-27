from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backend.ingestion.loaders.base import BaseLoader
from backend.ingestion.page_detector import PrintedPageDetector
from backend.models.document import DocumentType, RawDocument
from backend.models.logical_document import detect_continuation_signals
from backend.models.page_identity import PageIdentity
from backend.utils.logging import logger
from backend.utils.section_tracker import SectionTracker
from backend.vision.image_asset_manager import ImageAssetManager
from backend.vision.vision_service import VisionService
from src.config import settings

_CODE_PATTERN = re.compile(r"```|^\s*(def |class |import |function |const |let |var |Agent\(|Task\(|Crew\()", re.MULTILINE)
_TABLE_PATTERN = re.compile(r"\|.*\|.*\||(?:\+[-+]+\+)|(?:\t+[^\t\n]+\t+)")


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents with canonical page numbering, cross-page logical section linking,
    continuation signal detection, standalone original image asset extraction, and
    high-fidelity multimodal visual understanding.
    """

    def __init__(
        self,
        vision_service: VisionService | None = None,
        image_asset_manager: ImageAssetManager | None = None,
    ) -> None:
        self.vision_service = vision_service or VisionService()
        self.image_asset_manager = image_asset_manager or ImageAssetManager()

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def _detect_printed_page_number(self, text: str, physical_page_num: int) -> str:
        """
        Detect human-visible printed page number via PrintedPageDetector.
        """
        page_id = PrintedPageDetector.detect_single_page(text, physical_page_num)
        return page_id.display_label

    def load(
        self,
        file_path: Path,
        base_metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        base_meta = self._build_base_metadata(file_path, DocumentType.PDF, base_metadata)
        documents: list[RawDocument] = []
        section_tracker = SectionTracker()
        enable_vision = getattr(settings, "vision_enabled", True)
        doc_id = base_meta.document_id or f"doc_{file_path.stem}"

        # 1. Read all pages
        fitz_pages = self._read_with_fitz(file_path)
        pages = fitz_pages if fitz_pages is not None else self._read_with_pypdf(file_path)
        total_pages = len(pages)

        # 2. Sequence-Aware Printed Page Identity Resolution
        page_identities = PrintedPageDetector.resolve_document_pages(pages)

        prev_continuation_cues: list[str] = []
        prev_section_ctx = None

        for idx, (page_num, text) in enumerate(pages, start=1):
            page_id = page_identities[idx - 1] if idx - 1 < len(page_identities) else PageIdentity.from_indices(
                internal_page_index=idx - 1,
                physical_page_number=idx,
            )

            internal_page_index = page_id.internal_page_index
            physical_page_num = page_id.physical_page_number
            display_page_number = page_id.display_page_number
            page_label = page_id.page_label

            # Parse headings on this page
            page_found_new_heading = False
            current_ctx = None

            for line in text.splitlines():
                ctx = section_tracker.process_line(line)
                if ctx:
                    current_ctx = ctx
                    page_found_new_heading = True

            # If no new heading was found on this page, but previous page had continuation cues, maintain previous section context
            if not page_found_new_heading and prev_section_ctx is not None and prev_continuation_cues:
                current_ctx = prev_section_ctx
            elif current_ctx is None:
                current_ctx = section_tracker.current_context()

            has_code = bool(_CODE_PATTERN.search(text))
            has_tables = bool(_TABLE_PATTERN.search(text))
            continuation_signals = detect_continuation_signals(text)

            is_continuation = bool(not page_found_new_heading and prev_continuation_cues)
            continuation_from_page = (physical_page_num - 1) if is_continuation else None

            # Standalone Original Image Asset Extraction (Direct from PDF, Zero VLM blocking)
            page_assets = self.image_asset_manager.extract_page_images(
                pdf_path=file_path,
                internal_page_index=internal_page_index,
                page_number=physical_page_num,
                page_label=page_label,
                document_id=doc_id,
                display_page_number=display_page_number,
                section_title=current_ctx.section_title,
                section_path=current_ctx.section_path,
            )
            assets_dict = [asdict(a) for a in page_assets]

            page_meta = base_meta.model_copy(
                update={
                    "page_number": physical_page_num,
                    "internal_page_index": internal_page_index,
                    "display_page_number": display_page_number,
                    "page_label": page_label,
                    "total_pages": total_pages,
                    "section_title": current_ctx.section_title,
                    "section_number": current_ctx.section_number,
                    "section_path": current_ctx.section_path,
                    "section_level": current_ctx.section_level,
                    "has_code": has_code,
                    "has_tables": has_tables,
                    "image_assets": assets_dict,
                    "extra": {
                        **base_meta.extra,
                        "continuation_signals": continuation_signals,
                        "is_continuation": is_continuation,
                        "continuation_from_page": continuation_from_page,
                        "internal_page_index": internal_page_index,
                        "physical_page_number": physical_page_num,
                        "display_page_number": display_page_number,
                        "page_label": page_label,
                    },
                }
            )

            if text.strip():
                documents.append(RawDocument(content=text, metadata=page_meta))

            # Visual Content Extraction (Cache-only check during ingestion, never blocks)
            if enable_vision and self.vision_service:
                active_cue = prev_continuation_cues[0] if (is_continuation and prev_continuation_cues) else (continuation_signals[0] if continuation_signals else None)
                visual_chunks = self.vision_service.process_pdf_page_visuals(
                    pdf_path=file_path,
                    page_number=physical_page_num,
                    page_text=text,
                    document_id=doc_id,
                    section_title=current_ctx.section_title,
                    continuation_cue=active_cue,
                    live_inference=False,  # Instant cache lookup during ingestion!
                    page_label=page_label,
                    display_page_number=display_page_number,
                    internal_page_index=internal_page_index,
                )
                for vc in visual_chunks:
                    vis_meta = page_meta.model_copy(
                        update={
                            "has_code": vc.content_type == "code",
                            "has_tables": vc.content_type == "table",
                            "section_title": current_ctx.section_title,
                            "section_path": current_ctx.section_path,
                            "extra": {
                                **page_meta.extra,
                                "is_visual_extraction": True,
                                "visual_type": vc.visual_type,
                                "image_hash": vc.image_hash,
                                "content_type": vc.content_type,
                                "raw_code": vc.raw_code,
                                "continuation_cue": active_cue,
                                "display_page_number": display_page_number,
                                "page_label": page_label,
                            },
                        }
                    )
                    documents.append(RawDocument(content=vc.text, metadata=vis_meta))

            # Store for next page continuation
            prev_continuation_cues = continuation_signals
            prev_section_ctx = current_ctx

        return documents

    def _read_with_fitz(self, file_path: Path) -> list[tuple[int, str]] | None:
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

    def _read_with_pypdf(self, file_path: Path) -> list[tuple[int, str]]:
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
