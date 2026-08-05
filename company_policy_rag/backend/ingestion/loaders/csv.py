from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.ingestion.loaders.base import BaseLoader
from backend.models.document import DocumentType, RawDocument


class CSVLoader(BaseLoader):
    """Loader for CSV (.csv) tabular data documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".csv"

    def load(
        self,
        file_path: Path,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawDocument]:
        base_meta = self._build_base_metadata(file_path, DocumentType.CSV, base_metadata)

        rows: List[List[str]] = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        rows.append(row)
        except Exception:
            # Fallback reading
            content = file_path.read_text(encoding="latin-1", errors="replace")
            lines = content.splitlines()
            rows = [line.split(",") for line in lines if line.strip()]

        if not rows:
            return [RawDocument(content="", metadata=base_meta)]

        headers = rows[0]
        markdown_lines = []
        markdown_lines.append("| " + " | ".join(headers) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows[1:]:
            # Pad or truncate row to match header length
            padded_row = (row + [""] * len(headers))[: len(headers)]
            markdown_lines.append("| " + " | ".join(cell.replace("\n", " ") for cell in padded_row) + " |")

        content_str = "\n".join(markdown_lines)

        final_meta = base_meta.model_copy(
            update={
                "has_tables": True,
                "extra": {
                    "columns": headers,
                    "row_count": len(rows) - 1,
                    **base_meta.extra,
                },
            }
        )

        return [RawDocument(content=content_str, metadata=final_meta)]
