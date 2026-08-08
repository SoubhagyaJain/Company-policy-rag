from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.ingestion.loaders.base import BaseLoader
from backend.models.document import DocumentType, RawDocument


class JSONLoader(BaseLoader):
    """Loader for JSON and JSONL (.json, .jsonl) documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".json", ".jsonl"]

    def load(
        self,
        file_path: Path,
        base_metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        base_meta = self._build_base_metadata(file_path, DocumentType.JSON, base_metadata)

        content_str = file_path.read_text(encoding="utf-8", errors="replace")

        if file_path.suffix.lower() == ".jsonl":
            documents: list[RawDocument] = []
            lines = [line.strip() for line in content_str.splitlines() if line.strip()]
            for idx, line in enumerate(lines, start=1):
                try:
                    obj = json.loads(line)
                    text = self._format_json_object(obj)
                    item_meta = base_meta.model_copy(
                        update={
                            "extra": {"record_index": idx, **base_meta.extra},
                        }
                    )
                    documents.append(RawDocument(content=text, metadata=item_meta))
                except Exception:
                    continue
            return documents if documents else [RawDocument(content=content_str, metadata=base_meta)]

        try:
            parsed = json.loads(content_str)
            if isinstance(parsed, list):
                # If list of dicts, format each item cleanly or format whole list
                formatted_text = json.dumps(parsed, indent=2, ensure_ascii=False)
                final_meta = base_meta.model_copy(
                    update={"extra": {"record_count": len(parsed), **base_meta.extra}}
                )
                return [RawDocument(content=formatted_text, metadata=final_meta)]
            else:
                formatted_text = json.dumps(parsed, indent=2, ensure_ascii=False)
                return [RawDocument(content=formatted_text, metadata=base_meta)]
        except Exception:
            return [RawDocument(content=content_str, metadata=base_meta)]

    def _format_json_object(self, obj: Any) -> str:
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"**{k}**: {json.dumps(v, ensure_ascii=False)}")
                else:
                    lines.append(f"**{k}**: {v}")
            return "\n".join(lines)
        return str(obj)
