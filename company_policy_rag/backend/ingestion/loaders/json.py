from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.ingestion.loaders.base import BaseLoader
from backend.ingestion.loaders.validation import read_text
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

        content_str = read_text(file_path)

        if file_path.suffix.lower() == ".jsonl":
            documents: list[RawDocument] = []
            lines = content_str.splitlines()
            for idx, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    text = self._format_json_object(obj)
                    item_meta = base_meta.model_copy(
                        update={
                            "extra": {"record_index": idx, **base_meta.extra},
                        }
                    )
                    documents.append(RawDocument(content=text, metadata=item_meta))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {idx}: {exc.msg}") from exc
            return documents if documents else [RawDocument(content=content_str, metadata=base_meta)]

        try:
            parsed = json.loads(content_str)
            if isinstance(parsed, list):
                # Format each record into readable markdown section with clear key-value structure
                formatted_blocks: list[str] = []
                for i, item in enumerate(parsed, start=1):
                    if isinstance(item, dict):
                        title = (
                            item.get("title")
                            or item.get("name")
                            or item.get("policy")
                            or item.get("topic")
                            or f"Record {i}"
                        )
                        formatted_blocks.append(f"### {title}\n\n" + self._format_json_object(item))
                    else:
                        formatted_blocks.append(str(item))
                formatted_text = "\n\n---\n\n".join(formatted_blocks)
                final_meta = base_meta.model_copy(update={"extra": {"record_count": len(parsed), **base_meta.extra}})
                return [RawDocument(content=formatted_text, metadata=final_meta)]
            elif isinstance(parsed, dict):
                formatted_text = self._format_json_object(parsed)
                return [RawDocument(content=formatted_text, metadata=base_meta)]
            else:
                formatted_text = json.dumps(parsed, indent=2, ensure_ascii=False)
                return [RawDocument(content=formatted_text, metadata=base_meta)]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc

    def _format_json_object(self, obj: Any) -> str:
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                k_clean = str(k).replace("_", " ").title()
                if isinstance(v, (dict, list)):
                    lines.append(f"- **{k_clean}**: {json.dumps(v, ensure_ascii=False)}")
                else:
                    lines.append(f"- **{k_clean}**: {v}")
            return "\n".join(lines)
        return str(obj)
