from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from backend.utils.logging import logger
from src.config import settings


class VisionCacheManager:
    """
    Persistent on-disk cache for vision-language model extractions.
    Prevents re-processing unchanged PDF pages, screenshots, or diagrams.

    Cache Key: (image_hash, vision_model, document_id, page_number)
    """

    def __init__(self, cache_dir: Path | str | None = None, storage_dir: Path | str | None = None) -> None:
        target = cache_dir if cache_dir is not None else storage_dir
        if target is not None:
            self.cache_dir = Path(target)
        else:
            self.cache_dir = settings.vision_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._failures: dict[str, tuple[str, float]] = {}  # image_hash -> (error, timestamp)

    @staticmethod
    def compute_image_hash(image_bytes: bytes) -> str:
        """Compute deterministic SHA256 hex digest of image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()

    def is_failed(self, image_hash: str, ttl_seconds: float = 300.0) -> tuple[bool, str | None]:
        """Check if an image recently failed extraction to prevent repeated timeouts."""
        import time

        with self._lock:
            if image_hash in self._failures:
                err_msg, fail_time = self._failures[image_hash]
                if time.time() - fail_time < ttl_seconds:
                    return True, err_msg
                else:
                    self._failures.pop(image_hash, None)
        return False, None

    def mark_failed(self, image_hash: str, error_message: str) -> None:
        """Record an extraction failure for an image hash."""
        import time

        with self._lock:
            self._failures[image_hash] = (error_message, time.time())

    def _cache_file_path(
        self,
        image_hash: str,
        vision_model: str,
        document_id: str | None = None,
        page_number: int | None = None,
    ) -> Path:
        model_slug = vision_model.replace(":", "_").replace("/", "_").replace(".", "_")
        prefix = f"{document_id}_p{page_number}_" if document_id and page_number is not None else ""
        filename = f"{prefix}{image_hash[:16]}_{model_slug}.json"
        return self.cache_dir / filename

    def get(
        self,
        image_hash: str,
        vision_model: str,
        document_id: str | None = None,
        page_number: int | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve cached extraction result if present and valid."""
        path = self._cache_file_path(image_hash, vision_model, document_id, page_number)
        if not path.is_file():
            # Fallback check for content-addressed hash without document_id prefix
            generic_path = self._cache_file_path(image_hash, vision_model)
            if generic_path.is_file():
                path = generic_path
            else:
                return None

        try:
            with self._lock:
                data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("image_hash") == image_hash:
                logger.debug("Vision cache hit for image hash %s (model=%s)", image_hash[:8], vision_model)
                return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read vision cache file %s: %s", path, exc)
        return None

    def set(
        self,
        image_hash: str,
        vision_model: str,
        extracted_text: str,
        visual_type: str,
        document_id: str | None = None,
        page_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist vision extraction result to disk."""
        path = self._cache_file_path(image_hash, vision_model, document_id, page_number)
        payload = {
            "image_hash": image_hash,
            "vision_model": vision_model,
            "visual_type": visual_type,
            "extracted_text": extracted_text,
            "document_id": document_id,
            "page_number": page_number,
            "metadata": metadata or {},
        }
        try:
            with self._lock:
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                self._failures.pop(image_hash, None)
            logger.debug("Vision extraction cached to %s", path)
        except OSError as exc:
            logger.warning("Failed to write vision cache to %s: %s", path, exc)
        return path

    def clear(self) -> None:
        """Purge all entries from the vision cache."""
        with self._lock:
            self._failures.clear()
            for item in self.cache_dir.glob("*.json"):
                try:
                    item.unlink()
                except OSError:
                    pass
