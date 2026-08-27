from __future__ import annotations

import hashlib
import io
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

import re

from backend.models.page_identity import PageIdentity
from backend.utils.logging import logger
from src.config import PROJECT_ROOT, settings

_CODE_CUES = re.compile(
    r"(?:def\s+|class\s+|import\s+|from\s+\w+\s+import|function\s+|const\s+|let\s+|"
    r"here'?s\s+(?:the\s+)?code|see\s+code\s+below|implementation|code:|snippet|python|"
    r"agent\s*=|task\s*=|crew|crewai|langchain|analyst agent|writer agent|"
    r"let'?s\s+implement|the\s+following\s+code|check\s+this\s+code|kickoff|result\s*=)",
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


@dataclass
class ImageAsset:
    asset_id: str
    document_id: str
    internal_page_index: int = 0  # 0-indexed integer (fitz doc[i])
    physical_page_number: int = 1  # 1-indexed physical PDF sheet page
    display_page_number: str | int | None = None  # Human-visible printed page number (e.g. 98, "iv", "A-12")
    page_label: str = "1"  # Printed / display page label string (e.g. "98")
    page_number: int = 1  # Kept for backward compatibility (= physical_page_number)
    section: str | None = None  # Section title or path where asset resides
    section_title: str | None = None
    section_path: str | None = None
    visual_type: str = "diagram_architecture"  # diagram_architecture | code_screenshot | table_data | image | figure
    content_type: str = "diagram_architecture"
    image_hash: str = ""  # SHA-256 hash of original image bytes
    file_path: str = ""  # Path on disk
    storage_path: str = ""
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    extraction_status: str = "ASSET_AVAILABLE"  # ASSET_AVAILABLE | VISION_PENDING | VISION_PROCESSING | VISION_READY | VISION_FAILED | VISION_DEGRADED
    vision_status: str = "READY_ON_DEMAND"  # Backward compatibility: PENDING | SUCCESS | FAILED | SKIPPED | READY_ON_DEMAND
    vision_description: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self):
        if not self.storage_path and self.file_path:
            self.storage_path = self.file_path
        if not self.file_path and self.storage_path:
            self.file_path = self.storage_path
        if self.page_number <= 0 and self.physical_page_number > 0:
            self.page_number = self.physical_page_number
        if self.physical_page_number <= 0 and self.page_number > 0:
            self.physical_page_number = self.page_number
        if self.internal_page_index <= 0 and self.physical_page_number > 0:
            self.internal_page_index = max(0, self.physical_page_number - 1)
        if not self.section_title and self.section:
            self.section_title = self.section
        if not self.section and self.section_title:
            self.section = self.section_title

    @property
    def asset_url(self) -> str:
        return f"/api/documents/{self.document_id}/visual-assets/{self.asset_id}"

    @property
    def legacy_image_url(self) -> str:
        return f"/api/documents/{self.document_id}/images/{self.image_hash}"

    @property
    def display_label(self) -> str:
        if self.display_page_number is not None and str(self.display_page_number).strip():
            return str(self.display_page_number).strip()
        if self.page_label and str(self.page_label).strip():
            return str(self.page_label).strip()
        return str(self.physical_page_number or self.page_number)

    def get_page_identity(self) -> PageIdentity:
        return PageIdentity.from_indices(
            internal_page_index=self.internal_page_index,
            physical_page_number=self.physical_page_number or self.page_number,
            display_page_number=self.display_page_number,
            page_label=self.page_label,
        )


class ImageAssetManager:
    """
    Manages standalone extraction, persistence, indexing, and serving of original
    high-resolution document visual assets (diagrams, screenshots, tables, figures).
    Decoupled completely from LLM / VLM understanding.
    """

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        if storage_dir is not None:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = getattr(settings, "images_storage_dir", PROJECT_ROOT / "storage" / "images")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._assets_index: dict[str, dict[str, ImageAsset]] = {}  # document_id -> {image_hash: ImageAsset}
        self._load_all_indices()

    def _doc_dir(self, document_id: str) -> Path:
        p = self.storage_dir / document_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _index_file(self, document_id: str) -> Path:
        return self._doc_dir(document_id) / "assets.json"

    def _load_all_indices(self) -> None:
        with self._lock:
            for doc_folder in self.storage_dir.iterdir():
                if doc_folder.is_dir():
                    idx_file = doc_folder / "assets.json"
                    if idx_file.is_file():
                        try:
                            data = json.loads(idx_file.read_text(encoding="utf-8"))
                            loaded = {}
                            for k, v in data.items():
                                if not isinstance(v, dict):
                                    continue
                                v.setdefault("document_id", doc_folder.name)
                                v.setdefault("asset_id", f"ast_{v.get('image_hash', k)[:12]}")
                                v.setdefault("physical_page_number", v.get("page_number", 1))
                                v.setdefault("internal_page_index", max(0, v.get("physical_page_number", 1) - 1))
                                loaded[k] = ImageAsset(**v)
                            self._assets_index[doc_folder.name] = loaded
                        except Exception as e:
                            logger.warning("Failed to load asset index for %s: %s", doc_folder.name, e)

    def _save_index(self, document_id: str) -> None:
        idx_file = self._index_file(document_id)
        with self._lock:
            doc_assets = self._assets_index.get(document_id, {})
            payload = {k: asdict(v) for k, v in doc_assets.items()}
            idx_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def compute_image_hash(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    @staticmethod
    def get_optimized_inference_bytes(
        image_bytes: bytes,
        max_dim: int = 1024,
        is_code: bool = False,
    ) -> bytes:
        """
        Downscale image for fast VLM inference while preserving visual details.
        For code screenshots, uses higher resolution (max_dim=1280, quality=95)
        to prevent OCR degradation of syntax characters.
        """
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            w, h = pil_img.size
            target_max_dim = 1280 if is_code else max_dim
            target_quality = 95 if is_code else 85

            if max(w, h) > target_max_dim:
                scale = target_max_dim / float(max(w, h))
                new_w = int(w * scale)
                new_h = int(h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            if pil_img.mode in ("RGBA", "P"):
                pil_img = pil_img.convert("RGB")
            pil_img.save(buf, format="JPEG", quality=target_quality, optimize=True)
            return buf.getvalue()
        except Exception as exc:
            logger.debug("Failed to downscale inference image copy: %s. Using original bytes.", exc)
            return image_bytes

    def save_image_asset(
        self,
        document_id: str,
        internal_page_index: int,
        page_number: int,
        page_label: str,
        image_bytes: bytes,
        display_page_number: str | int | None = None,
        section_title: str | None = None,
        section_path: str | None = None,
        visual_type: str = "diagram_architecture",
        content_type: str = "diagram_architecture",
        ext: str = "png",
    ) -> ImageAsset:
        """Persist original high-resolution image asset and record in registry."""
        image_hash = self.compute_image_hash(image_bytes)
        filename = f"page_{page_number}_{image_hash[:12]}.{ext}"
        target_path = self._doc_dir(document_id) / filename

        if not target_path.exists():
            target_path.write_bytes(image_bytes)

        # Measure dimensions
        w, h = 0, 0
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            w, h = pil_img.size
        except Exception:
            pass

        # Normalized visual type
        v_type = visual_type.lower()
        if "code" in v_type or "screenshot" in v_type:
            norm_v_type = "code_screenshot"
        elif "table" in v_type:
            norm_v_type = "table_data"
        elif "figure" in v_type:
            norm_v_type = "figure"
        elif "diagram" in v_type or "architecture" in v_type or "workflow" in v_type:
            norm_v_type = "diagram_architecture"
        else:
            norm_v_type = "image"

        resolved_display = display_page_number
        if resolved_display is None and page_label:
            lbl_str = str(page_label).strip()
            resolved_display = int(lbl_str) if lbl_str.isdigit() else lbl_str

        asset = ImageAsset(
            asset_id=f"ast_{image_hash[:12]}",
            document_id=document_id,
            internal_page_index=internal_page_index,
            physical_page_number=page_number,
            display_page_number=resolved_display,
            page_label=str(page_label or page_number),
            page_number=page_number,
            section=section_title or section_path,
            section_title=section_title,
            section_path=section_path,
            visual_type=norm_v_type,
            content_type=content_type,
            image_hash=image_hash,
            file_path=str(target_path),
            storage_path=str(target_path),
            width=w,
            height=h,
            file_size_bytes=len(image_bytes),
            extraction_status="ASSET_AVAILABLE",
            vision_status="READY_ON_DEMAND",
        )

        with self._lock:
            if document_id not in self._assets_index:
                self._assets_index[document_id] = {}
            self._assets_index[document_id][image_hash] = asset

        self._save_index(document_id)
        logger.info(
            "[IMAGE ASSET] Saved original visual asset doc=%s p=%d (label=%s, %dx%d, %d bytes) type=%s -> %s",
            document_id,
            page_number,
            page_label,
            w,
            h,
            len(image_bytes),
            norm_v_type,
            target_path.name,
        )
        return asset

    def extract_page_images(
        self,
        pdf_path: Path | str,
        internal_page_index: int,
        page_number: int,
        page_label: str,
        document_id: str,
        display_page_number: str | int | None = None,
        section_title: str | None = None,
        section_path: str | None = None,
    ) -> list[ImageAsset]:
        """
        Extract embedded original images and high-fidelity page graphics from PDF directly.
        Does NOT invoke VLM. 100% fast, robust, and deterministic.
        """
        assets: list[ImageAsset] = []
        path = Path(pdf_path)
        if not path.is_file():
            return assets

        try:
            import fitz

            doc = fitz.open(path)
            if internal_page_index < 0 or internal_page_index >= len(doc):
                doc.close()
                return assets

            page = doc[internal_page_index]
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if base_image and "image" in base_image:
                    img_bytes = base_image["image"]
                    w = base_image.get("width", 0)
                    h = base_image.get("height", 0)
                    ext = base_image.get("ext", "png")

                    # Skip tiny decorative icons / bullets (< 100x70)
                    if w < 100 or h < 70:
                        continue

                    # Classify content type intelligently based on page text and section context
                    page_text = page.get_text()
                    combined_ctx = f"{page_text}\n{section_title or ''}\n{section_path or ''}"
                    if _CODE_CUES.search(combined_ctx):
                        content_type = "code_screenshot"
                    elif _TABLE_CUES.search(combined_ctx):
                        content_type = "table_data"
                    elif _DIAGRAM_CUES.search(combined_ctx):
                        content_type = "diagram_architecture"
                    else:
                        content_type = "code_screenshot" if w < h else "diagram_architecture"

                    asset = self.save_image_asset(
                        document_id=document_id,
                        internal_page_index=internal_page_index,
                        page_number=page_number,
                        page_label=page_label,
                        display_page_number=display_page_number,
                        image_bytes=img_bytes,
                        section_title=section_title,
                        section_path=section_path,
                        visual_type=content_type,
                        content_type=content_type,
                        ext=ext,
                    )
                    assets.append(asset)

            # If no embedded images found, check if page contains diagram/code cues to render high-res pixmap
            if not assets:
                text = page.get_text()
                has_visual_cues = (
                    "diagram" in text.lower()
                    or "architecture" in text.lower()
                    or "workflow" in text.lower()
                    or "def " in text
                    or "class " in text
                )
                if has_visual_cues and len(text.strip()) < 300:
                    dpi = getattr(settings, "vision_dpi", 150)
                    pix = page.get_pixmap(dpi=dpi)
                    img_bytes = pix.tobytes("png")
                    asset = self.save_image_asset(
                        document_id=document_id,
                        internal_page_index=internal_page_index,
                        page_number=page_number,
                        page_label=page_label,
                        display_page_number=display_page_number,
                        image_bytes=img_bytes,
                        section_title=section_title,
                        section_path=section_path,
                        visual_type="diagram_architecture",
                        content_type="diagram_architecture",
                        ext="png",
                    )
                    assets.append(asset)

            doc.close()
        except Exception as exc:
            logger.warning("Failed extracting page images for %s p%d: %s", path.name, page_number, exc)

        return assets

    def _ensure_doc_loaded(self, document_id: str) -> None:
        if document_id not in self._assets_index:
            idx_file = self._index_file(document_id)
            if idx_file.is_file():
                try:
                    data = json.loads(idx_file.read_text(encoding="utf-8"))
                    self._assets_index[document_id] = {
                        k: ImageAsset(**v) for k, v in data.items()
                    }
                except Exception as e:
                    logger.warning("Failed to load asset index for %s: %s", document_id, e)

    def get_asset(self, document_id: str, image_hash_or_id: str) -> ImageAsset | None:
        """Find image asset by exact hash, hash prefix, or asset_id."""
        with self._lock:
            self._ensure_doc_loaded(document_id)
            doc_assets = self._assets_index.get(document_id, {})
            if image_hash_or_id in doc_assets:
                return doc_assets[image_hash_or_id]
            for ast in doc_assets.values():
                if ast.asset_id == image_hash_or_id or ast.image_hash.startswith(image_hash_or_id):
                    return ast
        return None

    def get_asset_by_id(self, document_id: str, asset_id: str) -> ImageAsset | None:
        return self.get_asset(document_id, asset_id)

    def get_page_assets(self, document_id: str, page_identifier: int | str) -> list[ImageAsset]:
        """Get assets for a user-facing printed page number or label."""
        matches: list[ImageAsset] = []
        with self._lock:
            self._ensure_doc_loaded(document_id)
            doc_assets = self._assets_index.get(document_id, {})

            for ast in doc_assets.values():
                if ast.get_page_identity().matches_display(page_identifier):
                    matches.append(ast)

        return matches

    def get_page_assets_by_physical_page(self, document_id: str, physical_page_number: int) -> list[ImageAsset]:
        """Get assets for an explicit 1-based PDF sheet number."""
        with self._lock:
            self._ensure_doc_loaded(document_id)
            return [
                ast
                for ast in self._assets_index.get(document_id, {}).values()
                if ast.get_page_identity().matches_physical_page(physical_page_number)
            ]

    def get_page_assets_by_internal_index(self, document_id: str, internal_page_index: int) -> list[ImageAsset]:
        """Get assets for an explicit 0-based parser/renderer page index."""
        with self._lock:
            self._ensure_doc_loaded(document_id)
            return [
                ast
                for ast in self._assets_index.get(document_id, {}).values()
                if ast.get_page_identity().matches_internal_index(internal_page_index)
            ]

    def rank_page_assets(
        self,
        document_id: str,
        page_identifier: int | str,
        query: str = "",
        intent: str = "general",
    ) -> list[ImageAsset]:
        """
        Rank candidate visual assets on a page using query relevance, code-likeness,
        and section metadata so that relevant assets (e.g. code screenshots for code queries)
        are prioritized over generic diagrams.
        """
        assets = self.get_page_assets(document_id, page_identifier)
        if not assets or len(assets) <= 1:
            return assets

        query_lower = query.lower()
        is_code_intent = (
            intent.lower() in ("code", "implementation", "procedural", "code_explanation")
            or any(w in query_lower for w in ("code", "implementation", "agent", "task", "def ", "class ", "kickoff", "explain this code", "explain the code", "show the code"))
        )
        is_diagram_intent = (
            intent.lower() in ("architecture", "explanation", "diagram")
            or any(w in query_lower for w in ("diagram", "architecture", "workflow", "flowchart", "tell me more about the diagram", "explain the diagram"))
        )

        def _score_asset(ast: ImageAsset) -> float:
            score = 0.0
            vtype = (ast.visual_type or "").lower()
            if is_code_intent:
                if "code" in vtype:
                    score += 15.0
                elif "table" in vtype:
                    score += 1.0
                else:
                    score += 2.0
            elif is_diagram_intent:
                if "diagram" in vtype or "figure" in vtype or "architecture" in vtype:
                    score += 15.0
                elif "code" in vtype:
                    score += 3.0
            else:
                if "diagram" in vtype or "figure" in vtype:
                    score += 5.0
                elif "code" in vtype:
                    score += 3.0

            # Section title overlap
            if ast.section_title and any(w in ast.section_title.lower() for w in query_lower.split() if len(w) > 3):
                score += 4.0

            return score

        return sorted(assets, key=_score_asset, reverse=True)

    def get_assets_by_type(self, document_id: str, visual_type: str) -> list[ImageAsset]:
        """Retrieve all visual assets of a specific type (e.g. code_screenshot, diagram_architecture) in a document."""
        with self._lock:
            self._ensure_doc_loaded(document_id)
            doc_assets = self._assets_index.get(document_id, {})
            norm_type = visual_type.lower()
            return [
                ast for ast in doc_assets.values()
                if norm_type in (ast.visual_type or "").lower() or norm_type in (ast.content_type or "").lower()
            ]

    def get_code_screenshot_assets(self, document_id: str, page_number: int | None = None) -> list[ImageAsset]:
        """Retrieve code screenshot assets, optionally by 1-based physical page."""
        assets = self.get_assets_by_type(document_id, "code_screenshot")
        if page_number is not None:
            return [
                a for a in assets
                if a.physical_page_number == page_number
                or a.page_number == page_number
            ]
        return assets

    def get_diagram_assets(self, document_id: str, page_number: int | None = None) -> list[ImageAsset]:
        """Retrieve diagrams, optionally by 1-based physical page."""
        assets = self.get_assets_by_type(document_id, "diagram")
        if not assets:
            assets = self.get_assets_by_type(document_id, "figure")
        if page_number is not None:
            return [
                a for a in assets
                if a.physical_page_number == page_number
                or a.page_number == page_number
            ]
        return assets

    def find_related_assets(
        self,
        document_id: str,
        page_number: int,
        visual_type: str | None = None,
        section_title: str | None = None,
    ) -> list[ImageAsset]:
        """Find assets matching or adjacent to a given page and section context."""
        with self._lock:
            self._ensure_doc_loaded(document_id)
            doc_assets = self._assets_index.get(document_id, {})
            candidates: list[ImageAsset] = []
            for ast in doc_assets.values():
                page_match = abs(ast.physical_page_number - page_number) <= 2
                type_match = True if not visual_type else (visual_type.lower() in (ast.visual_type or "").lower())
                sec_match = True if not section_title else (
                    ast.section_title and (section_title.lower() in ast.section_title.lower() or ast.section_title.lower() in section_title.lower())
                )
                if page_match and (type_match or sec_match):
                    candidates.append(ast)
            return candidates

    def get_page_asset(self, document_id: str, page_identifier: int | str) -> ImageAsset | None:
        """Get first visual asset on a page matching page_identifier."""
        assets = self.get_page_assets(document_id, page_identifier)
        return assets[0] if assets else None

    def list_assets(self, document_id: str) -> list[ImageAsset]:
        with self._lock:
            self._ensure_doc_loaded(document_id)
            return list(self._assets_index.get(document_id, {}).values())

    def list_all_assets(self) -> list[ImageAsset]:
        """List all visual assets across all loaded documents."""
        self._load_all_indices()
        all_assets: list[ImageAsset] = []
        with self._lock:
            for doc_assets in self._assets_index.values():
                all_assets.extend(doc_assets.values())
        return all_assets

    def get_global_visual_stats(self) -> dict[str, Any]:
        """Return comprehensive visual asset and page statistics across all ingested documents."""
        assets = self.list_all_assets()
        unique_pages = {f"{a.document_id}_{a.page_number}" for a in assets}
        code_screens = sum(1 for a in assets if a.visual_type == "code_screenshot")
        diagrams = sum(1 for a in assets if a.visual_type in ("diagram_architecture", "figure", "image"))
        tables = sum(1 for a in assets if a.visual_type == "table_data")
        return {
            "total_assets": len(assets),
            "visual_pages_detected": len(unique_pages),
            "code_screenshots": code_screens,
            "diagrams": diagrams,
            "tables": tables,
        }

    def update_asset_vision_status(
        self,
        document_id: str,
        image_hash: str,
        status: str,
        description: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            doc_assets = self._assets_index.get(document_id, {})
            if image_hash in doc_assets:
                ast = doc_assets[image_hash]
                ast.vision_status = status
                # Update extraction_status accordingly
                if status in ("SUCCESS", "READY", "READY_ON_DEMAND"):
                    ast.extraction_status = "VISION_READY"
                elif status in ("FAILED", "TIMEOUT", "ERROR"):
                    ast.extraction_status = "VISION_FAILED"
                elif status == "DEGRADED":
                    ast.extraction_status = "VISION_DEGRADED"
                elif status in ("PENDING", "PROCESSING"):
                    ast.extraction_status = f"VISION_{status}"
                else:
                    ast.extraction_status = status

                if description:
                    ast.vision_description = description
                if error:
                    ast.error = error
        self._save_index(document_id)
