from __future__ import annotations

import io
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from backend.models.logical_document import detect_continuation_signals
from backend.utils.logging import logger
from backend.vision.image_asset_manager import ImageAssetManager
from backend.vision.vision_cache import VisionCacheManager
from src.config import settings
from src.ollama_client import execute_vision_completion, probe_vision_model_status

_CODE_CUES = re.compile(
    r"(?:def\s+|class\s+|import\s+|from\s+\w+\s+import|function\s+|const\s+|let\s+|"
    r"here'?s\s+(?:the\s+)?code|see\s+code\s+below|implementation|code:|snippet|python|"
    r"agent\s*=|task\s*=|crew|crewai|langchain|analyst agent|writer agent|"
    r"let'?s\s+implement|the\s+following\s+code|check\s+this\s+code)",
    re.IGNORECASE,
)

_DIAGRAM_CUES = re.compile(
    r"(?:diagram|architecture|workflow|flowchart|pipeline|figure\s+\d+|illustration|"
    r"system overview|component graph|data flow|interaction diagram|schema)",
    re.IGNORECASE,
)

_TABLE_CUES = re.compile(
    r"(?:table\s+\d+|benchmark results|comparison table|matrix|summary of features|parameters\s+table)",
    re.IGNORECASE,
)

CODE_EXTRACTION_PROMPT = """Extract the visible code from this document image as faithfully and accurately as possible.

Strict requirements:
1. Preserve exact indentation, formatting, and line breaks.
2. Preserve exact import statements, module names, and aliases.
3. Preserve exact variable names, constants, and data types.
4. Preserve exact class names, base classes, and inheritance.
5. Preserve exact function/method signatures, parameter names, and default values.
6. Preserve exact string literals, docstrings, and comments.
7. Preserve exact decorators, annotations, and configuration parameters.
8. Do NOT explain, summarize, or critique the code.
9. Do NOT rewrite, refactor, or optimize the code.
10. Do NOT invent missing lines or replace visible code with placeholder pass/ellipsis.
11. If any token is blurry or partially obscured, provide the closest visible literal match without fabricating unseen logic.

Output only the extracted code in standard markdown code block syntax (e.g. ```python ... ```)."""

DIAGRAM_EXTRACTION_PROMPT = """Extract and describe all structural information from this diagram, table, or architecture illustration accurately and faithfully.
Include:
- Component names and explicit relationships/connectors
- Flow of data or execution sequence steps
- Inputs, outputs, and dependencies
- Any visible labels, parameter values, or configuration settings.
Do not make speculative assumptions or invent unlisted components."""

TABLE_EXTRACTION_PROMPT = """Extract all visible table data accurately and faithfully into a clean Markdown table.
Preserve:
- Exact column headers
- Exact row labels and hierarchy
- Exact numeric and textual values
Do not add conversational explanations or omit rows."""


class VisualContentType(str, Enum):
    NONE = "none"
    CODE_SCREENSHOT = "code_screenshot"
    DIAGRAM_ARCHITECTURE = "diagram_architecture"
    TABLE_DATA = "table_data"
    SCANNED_TEXT = "scanned_text"
    DECORATIVE_IMAGE = "decorative_image"


class VisionCircuitBreaker:
    """Circuit breaker for Ollama vision model calls. Prevents cascade failures and ingestion blocking."""

    def __init__(self, failure_threshold: int = 3, recovery_cooldown: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_cooldown = recovery_cooldown
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self.failure_count >= self.failure_threshold:
                if time.time() - self.last_failure_time > self.recovery_cooldown:
                    # Half-open: test single request
                    self.failure_count = self.failure_threshold - 1
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self.failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                logger.warning(
                    "[VISION] Circuit breaker tripped! %d consecutive failures. Entering DEGRADED mode for %.0fs.",
                    self.failure_count,
                    self.recovery_cooldown,
                )


@dataclass
class VisualDetectionResult:
    has_visual: bool
    visual_type: VisualContentType
    confidence: float
    reason: str
    image_bytes: bytes | None = None
    image_hash: str | None = None
    page_number: int = 1
    page_label: str = "1"
    internal_page_index: int = 0
    image_count: int = 0
    dimensions: tuple[int, int] | None = None


@dataclass
class VisualExtractionChunk:
    text: str
    content_type: str  # 'code', 'table', 'diagram', or 'prose'
    visual_type: str
    page_number: int
    page_label: str = "1"
    internal_page_index: int = 0
    image_hash: str = ""
    section_title: str | None = None
    raw_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionService:
    """
    Service orchestrating visual page detection heuristics, code screenshot OCR,
    diagram understanding, Ollama vision model calls, and multi-tier ingestion caching.
    """

    _semaphore = threading.Semaphore(1)
    _circuit_breaker = VisionCircuitBreaker()

    def __init__(
        self,
        cache_manager: VisionCacheManager | None = None,
        image_asset_manager: ImageAssetManager | None = None,
        vision_model: str | None = None,
    ) -> None:
        self.cache = cache_manager or VisionCacheManager()
        self.image_asset_manager = image_asset_manager or ImageAssetManager()
        self.vision_model = (vision_model or settings.vision_model).strip()

    def is_available(self) -> tuple[bool, str]:
        """Check whether local Ollama vision capabilities are ready."""
        if not getattr(settings, "vision_enabled", True):
            return False, "Vision processing is disabled via VISION_ENABLED=false."
        return probe_vision_model_status(self.vision_model)

    def detect_visual_content(
        self,
        page_text: str,
        image_bytes: bytes | None = None,
        image_count: int = 0,
        page_number: int = 1,
        page_label: str = "1",
        internal_page_index: int = 0,
        image_width: int = 0,
        image_height: int = 0,
        continuation_cue: str | None = None,
    ) -> VisualDetectionResult:
        """
        Evaluate whether a PDF page contains visual information that warrants vision processing.
        Uses heuristics to skip plain text pages and classify visual pages.
        """
        # Rule 1: No images and no visual cues -> Pure text, skip
        if image_count == 0 and image_bytes is None:
            return VisualDetectionResult(
                has_visual=False,
                visual_type=VisualContentType.NONE,
                confidence=0.99,
                reason="Plain text page with zero embedded images.",
                page_number=page_number,
                page_label=page_label,
                internal_page_index=internal_page_index,
            )

        # Rule 2: Images present, but too small (icons / decorative bullets)
        if image_width > 0 and image_height > 0:
            if image_width < 100 or image_height < 70:
                return VisualDetectionResult(
                    has_visual=False,
                    visual_type=VisualContentType.DECORATIVE_IMAGE,
                    confidence=0.90,
                    reason=f"Small decorative image ignored ({image_width}x{image_height}).",
                    page_number=page_number,
                    page_label=page_label,
                    internal_page_index=internal_page_index,
                )

        img_hash = VisionCacheManager.compute_image_hash(image_bytes) if image_bytes else None
        text_lower = page_text.lower()

        # Rule 3: Prior continuation cue indicates code screenshot on this page
        if continuation_cue and image_bytes is not None:
            return VisualDetectionResult(
                has_visual=True,
                visual_type=VisualContentType.CODE_SCREENSHOT,
                confidence=0.95,
                reason=f"Prior page continuation cue '{continuation_cue}' with visual content on current page.",
                image_bytes=image_bytes,
                image_hash=img_hash,
                page_number=page_number,
                page_label=page_label,
                internal_page_index=internal_page_index,
                image_count=image_count,
                dimensions=(image_width, image_height) if image_width and image_height else None,
            )

        # Rule 4: Text indicates code screenshot or implementation
        if _CODE_CUES.search(text_lower):
            return VisualDetectionResult(
                has_visual=True,
                visual_type=VisualContentType.CODE_SCREENSHOT,
                confidence=0.90,
                reason="Page contains code/implementation cues and embedded image.",
                image_bytes=image_bytes,
                image_hash=img_hash,
                page_number=page_number,
                page_label=page_label,
                internal_page_index=internal_page_index,
                image_count=image_count,
                dimensions=(image_width, image_height) if image_width and image_height else None,
            )

        # Rule 5: Text indicates diagram / architecture / workflow
        if _DIAGRAM_CUES.search(text_lower):
            return VisualDetectionResult(
                has_visual=True,
                visual_type=VisualContentType.DIAGRAM_ARCHITECTURE,
                confidence=0.88,
                reason="Page contains diagram/architecture cues and embedded image.",
                image_bytes=image_bytes,
                image_hash=img_hash,
                page_number=page_number,
                page_label=page_label,
                internal_page_index=internal_page_index,
                image_count=image_count,
                dimensions=(image_width, image_height) if image_width and image_height else None,
            )

        # Rule 6: Text indicates tabular data
        if _TABLE_CUES.search(text_lower):
            return VisualDetectionResult(
                has_visual=True,
                visual_type=VisualContentType.TABLE_DATA,
                confidence=0.85,
                reason="Page contains table/matrix cues and embedded image.",
                image_bytes=image_bytes,
                image_hash=img_hash,
                page_number=page_number,
                page_label=page_label,
                internal_page_index=internal_page_index,
                image_count=image_count,
                dimensions=(image_width, image_height) if image_width and image_height else None,
            )

        # Rule 7: Low text density with a significant image (e.g. scanned page or standalone screenshot)
        if len(page_text.strip()) < 200 and image_bytes is not None:
            return VisualDetectionResult(
                has_visual=True,
                visual_type=VisualContentType.CODE_SCREENSHOT,
                confidence=0.80,
                reason="Low text density with large visual content (code screenshot or diagram candidate).",
                image_bytes=image_bytes,
                image_hash=img_hash,
                page_number=page_number,
                page_label=page_label,
                internal_page_index=internal_page_index,
                image_count=image_count,
                dimensions=(image_width, image_height) if image_width and image_height else None,
            )

        # Default fallback for pages with images
        return VisualDetectionResult(
            has_visual=True,
            visual_type=VisualContentType.DIAGRAM_ARCHITECTURE,
            confidence=0.70,
            reason="Embedded image detected on page.",
            image_bytes=image_bytes,
            image_hash=img_hash,
            page_number=page_number,
            page_label=page_label,
            internal_page_index=internal_page_index,
            image_count=image_count,
        )

    def extract_from_image(
        self,
        image_bytes: bytes,
        visual_type: VisualContentType,
        document_id: str | None = None,
        page_number: int | None = None,
        page_label: str | None = None,
        internal_page_index: int | None = None,
        section_title: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        timeout: float = 40.0,
        is_query_time: bool = False,
    ) -> VisualExtractionChunk | None:
        """
        Extract code or structural description from image bytes using the vision model and disk cache.
        Includes circuit breaker, bounded concurrency, negative caching, and strict retry limits:
        - Ingestion time: ZERO retries (max_attempts = 1).
        - Query time: Controlled at most 1 retry (max_attempts = 2).
        """
        if not image_bytes:
            return None

        image_hash = VisionCacheManager.compute_image_hash(image_bytes)
        disp_label = page_label or str(page_number or 1)
        int_idx = internal_page_index if internal_page_index is not None else ((page_number or 1) - 1)
        t_start = time.perf_counter()

        # 1. Ingestion Positive Cache Check
        cached_entry = self.cache.get(
            image_hash=image_hash,
            vision_model=self.vision_model,
            document_id=document_id,
            page_number=page_number,
        )
        if cached_entry:
            extracted_text = cached_entry.get("extracted_text", "")
            cached_type = cached_entry.get("visual_type", visual_type.value)
            content_type = "code" if cached_type == VisualContentType.CODE_SCREENSHOT.value else "prose"
            try:
                from backend.api.dependencies import get_telemetry_service
                get_telemetry_service().record_vision_event(
                    document_id=document_id,
                    page_number=page_number,
                    visual_type=visual_type.value,
                    status="CACHE_HIT",
                    duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    model_name=self.vision_model,
                )
                get_telemetry_service().record_cache_event(
                    cache_type="Vision Cache",
                    event_type="HIT",
                    latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
                    key_hash=image_hash,
                    model_name=self.vision_model,
                )
            except Exception:
                pass
            return VisualExtractionChunk(
                text=extracted_text,
                content_type=content_type,
                visual_type=cached_type,
                page_number=page_number or 1,
                page_label=disp_label,
                internal_page_index=int_idx,
                image_hash=image_hash,
                section_title=section_title,
                raw_code=extracted_text if content_type == "code" else None,
                metadata=cached_entry.get("metadata", {}),
            )

        # 2. Check Negative Cache
        is_failed, fail_reason = self.cache.is_failed(image_hash, ttl_seconds=300.0)
        if is_failed:
            logger.info(
                "[VISION] Skipping image %s for doc=%s p=%s (label=%s) (negative cache active: %s)",
                image_hash[:8],
                document_id,
                page_number,
                disp_label,
                fail_reason,
            )
            return None

        # 3. Check Circuit Breaker
        if not self._circuit_breaker.allow_request():
            logger.warning(
                "[VISION] Circuit breaker OPEN. Skipping live vision call for doc=%s p=%s (label=%s)",
                document_id,
                page_number,
                disp_label,
            )
            return None

        # 4. Select Specialized Prompt
        if visual_type == VisualContentType.CODE_SCREENSHOT:
            prompt = CODE_EXTRACTION_PROMPT
            content_type = "code"
        elif visual_type == VisualContentType.TABLE_DATA:
            prompt = TABLE_EXTRACTION_PROMPT
            content_type = "table"
        else:
            prompt = DIAGRAM_EXTRACTION_PROMPT
            content_type = "prose"

        # 5. Model Availability Verification
        is_ready, msg = self.is_available()
        if not is_ready:
            logger.warning(
                "Skipping live vision model call for doc=%s p=%s (label=%s): %s",
                document_id,
                page_number,
                disp_label,
                msg,
            )
            return None

        # 6. Optimize/Downscale Image for Fast VLM Inference (Max Dim 1024px)
        max_dim = getattr(settings, "vision_inference_max_dimension", 1024)
        inference_image_bytes = ImageAssetManager.get_optimized_inference_bytes(image_bytes, max_dim=max_dim)

        # 7. Determine Retry Policy: Ingestion = 0 retries (1 attempt), Query-time = 1 retry (2 attempts)
        if is_query_time:
            max_attempts = 1 + getattr(settings, "vision_max_lazy_retries", 1)
        else:
            max_attempts = 1 + getattr(settings, "vision_max_ingestion_retries", 0)

        extracted_text = ""
        last_error = None

        with self._semaphore:
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.info(
                        "[VISION] Processing visual content: model=%s, doc=%s, page=%s (label=%s, idx=%s), type=%s, hash=%s, attempt=%d/%d",
                        self.vision_model,
                        document_id,
                        page_number,
                        disp_label,
                        int_idx,
                        visual_type.value,
                        image_hash[:8],
                        attempt,
                        max_attempts,
                    )
                    extracted_text = execute_vision_completion(
                        prompt=prompt,
                        image_bytes=inference_image_bytes,
                        model_name=self.vision_model,
                        timeout=timeout,
                    )
                    if extracted_text:
                        self._circuit_breaker.record_success()
                        break
                    else:
                        logger.warning("Vision model returned empty response for image %s", image_hash[:8])
                except Exception as exc:
                    last_error = exc
                    elapsed = time.perf_counter() - t_start
                    logger.warning(
                        "[VISION] Ollama vision completion error (%s): %s | attempt %d/%d for doc=%s page=%s (label=%s, %.2fs)",
                        self.vision_model,
                        exc,
                        attempt,
                        max_attempts,
                        document_id,
                        page_number,
                        disp_label,
                        elapsed,
                    )
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

        if not extracted_text:
            self._circuit_breaker.record_failure()
            self.cache.mark_failed(image_hash, str(last_error or "Empty vision response"))
            try:
                from backend.api.dependencies import get_telemetry_service
                err_status = "TIMEOUT" if (last_error and "timeout" in str(last_error).lower()) else "ERROR"
                get_telemetry_service().record_vision_event(
                    document_id=document_id,
                    page_number=page_number,
                    visual_type=visual_type.value,
                    status=err_status,
                    duration_ms=elapsed_ms,
                    model_name=self.vision_model,
                    message=str(last_error or "Empty vision response"),
                )
            except Exception:
                pass

            if document_id:
                self.image_asset_manager.update_asset_vision_status(
                    document_id=document_id,
                    image_hash=image_hash,
                    status="FAILED",
                    error=str(last_error or "Vision timeout/empty response"),
                )
            return None

        # Format code block cleanly if it is code extraction
        if visual_type == VisualContentType.CODE_SCREENSHOT and not extracted_text.startswith("```"):
            extracted_text = f"```python\n{extracted_text}\n```"

        # 8. Persist to Cache & Update Image Asset
        meta_dict = {
            "source": "vision_extraction",
            "vision_model": self.vision_model,
            "section_title": section_title,
            "page_label": disp_label,
            "internal_page_index": int_idx,
            "image_url": f"/api/documents/{document_id}/images/{image_hash}" if document_id else None,
            **(extra_metadata or {}),
        }
        self.cache.set(
            image_hash=image_hash,
            vision_model=self.vision_model,
            extracted_text=extracted_text,
            visual_type=visual_type.value,
            document_id=document_id,
            page_number=page_number,
            metadata=meta_dict,
        )

        try:
            from backend.api.dependencies import get_telemetry_service
            get_telemetry_service().record_vision_event(
                document_id=document_id,
                page_number=page_number,
                visual_type=visual_type.value,
                status="SUCCESS",
                duration_ms=elapsed_ms,
                model_name=self.vision_model,
            )
            get_telemetry_service().record_cache_event(
                cache_type="Vision Cache",
                event_type="SET",
                latency_ms=elapsed_ms,
                key_hash=image_hash,
                model_name=self.vision_model,
            )
        except Exception:
            pass

        if document_id:
            self.image_asset_manager.update_asset_vision_status(
                document_id=document_id,
                image_hash=image_hash,
                status="SUCCESS",
                description=extracted_text,
            )

        return VisualExtractionChunk(
            text=extracted_text,
            content_type=content_type,
            visual_type=visual_type.value,
            page_number=page_number or 1,
            page_label=disp_label,
            internal_page_index=int_idx,
            image_hash=image_hash,
            section_title=section_title,
            raw_code=extracted_text if content_type == "code" else None,
            metadata=meta_dict,
        )

    def process_pdf_page_visuals(
        self,
        pdf_path: Path | str,
        page_number: int,
        page_text: str,
        document_id: str | None = None,
        section_title: str | None = None,
        continuation_cue: str | None = None,
        live_inference: bool = True,
        timeout: float = 40.0,
        page_label: str | None = None,
        internal_page_index: int | None = None,
        is_query_time: bool = False,
    ) -> list[VisualExtractionChunk]:
        """
        Inspect a specific page in a PDF file, extract embedded images or render pixmap,
        and run vision extraction.
        """
        results: list[VisualExtractionChunk] = []
        path = Path(pdf_path)
        if not path.is_file():
            return results

        int_idx = internal_page_index if internal_page_index is not None else (page_number - 1)
        disp_label = page_label or str(page_number)

        try:
            import fitz

            doc = fitz.open(path)
            if int_idx < 0 or int_idx >= len(doc):
                doc.close()
                return results

            page = doc[int_idx]
            image_list = page.get_images(full=True)
            image_count = len(image_list)

            best_image_bytes = None
            max_pixels = 0
            best_w = 0
            best_h = 0

            for img_info in image_list:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if base_image and "image" in base_image:
                    w = base_image.get("width", 0)
                    h = base_image.get("height", 0)
                    pixels = w * h
                    if pixels > max_pixels:
                        max_pixels = pixels
                        best_image_bytes = base_image["image"]
                        best_w = w
                        best_h = h

            # If no embedded images or image extraction failed, or continuation cue is present, render high-res pixmap
            if best_image_bytes is None or continuation_cue is not None:
                should_render = (
                    continuation_cue is not None
                    or _CODE_CUES.search(page_text)
                    or _DIAGRAM_CUES.search(page_text)
                    or len(page_text.strip()) < 150
                )
                if should_render:
                    dpi = getattr(settings, "vision_dpi", 150)
                    pix = page.get_pixmap(dpi=dpi)
                    best_image_bytes = pix.tobytes("png")
                    best_w = pix.width
                    best_h = pix.height
                    image_count = max(image_count, 1)

            doc.close()

            if best_image_bytes:
                detection = self.detect_visual_content(
                    page_text=page_text,
                    image_bytes=best_image_bytes,
                    image_count=image_count,
                    page_number=page_number,
                    page_label=disp_label,
                    internal_page_index=int_idx,
                    image_width=best_w,
                    image_height=best_h,
                    continuation_cue=continuation_cue,
                )
                if detection.has_visual and detection.image_bytes:
                    if not live_inference:
                        # Cache-only lookup during initial fast load
                        img_hash = detection.image_hash or VisionCacheManager.compute_image_hash(detection.image_bytes)
                        cached_entry = self.cache.get(
                            image_hash=img_hash,
                            vision_model=self.vision_model,
                            document_id=document_id,
                            page_number=page_number,
                        )
                        if cached_entry:
                            extracted_text = cached_entry.get("extracted_text", "")
                            cached_type = cached_entry.get("visual_type", detection.visual_type.value)
                            content_type = "code" if cached_type == VisualContentType.CODE_SCREENSHOT.value else "prose"
                            results.append(
                                VisualExtractionChunk(
                                    text=extracted_text,
                                    content_type=content_type,
                                    visual_type=cached_type,
                                    page_number=page_number,
                                    page_label=disp_label,
                                    internal_page_index=int_idx,
                                    image_hash=img_hash,
                                    section_title=section_title,
                                    raw_code=extracted_text if content_type == "code" else None,
                                    metadata=cached_entry.get("metadata", {}),
                                )
                            )
                        return results

                    chunk = self.extract_from_image(
                        image_bytes=detection.image_bytes,
                        visual_type=detection.visual_type,
                        document_id=document_id,
                        page_number=page_number,
                        page_label=disp_label,
                        internal_page_index=int_idx,
                        section_title=section_title,
                        extra_metadata={"continuation_cue": continuation_cue} if continuation_cue else None,
                        timeout=timeout,
                        is_query_time=is_query_time,
                    )
                    if chunk:
                        results.append(chunk)

        except Exception as exc:
            logger.warning("Error processing PDF page visuals for %s p%d (label=%s): %s", path.name, page_number, disp_label, exc)

        return results

    def extract_page_range_visuals(
        self,
        pdf_path: Path | str,
        pages: list[int],
        document_id: str | None = None,
        section_title: str | None = None,
        continuation_cue: str | None = None,
        max_pages: int = 20,
        timeout: float = 40.0,
        is_query_time: bool = True,
    ) -> list[VisualExtractionChunk]:
        """Inspect and extract visual content across multiple adjacent pages (e.g. Page N, N+1, N+2)."""
        all_chunks: list[VisualExtractionChunk] = []
        target_pages = [p for p in pages if p > 0][:max_pages]

        for p in target_pages:
            page_chunks = self.process_pdf_page_visuals(
                pdf_path=pdf_path,
                page_number=p,
                page_text="",
                document_id=document_id,
                section_title=section_title,
                continuation_cue=continuation_cue,
                live_inference=True,
                timeout=timeout,
                is_query_time=is_query_time,
            )
            all_chunks.extend(page_chunks)
        return all_chunks
