from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from backend.ingestion.loaders.base import BaseLoader
from backend.ingestion.loaders.validation import read_text
from backend.models.document import DocumentType, RawDocument


class CSVLoader(BaseLoader):
    """Loader for CSV (.csv) tabular data documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in (".csv", ".tsv")

    def load(
        self,
        file_path: Path,
        base_metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        base_meta = self._build_base_metadata(file_path, DocumentType.CSV, base_metadata)

        try:
            reader = csv.reader(
                io.StringIO(read_text(file_path), newline=""),
                delimiter="\t" if file_path.suffix.lower() == ".tsv" else ",",
                strict=True,
            )
            rows = [row for row in reader if row]
        except csv.Error as exc:
            raise ValueError(f"Invalid delimited document: {exc}") from exc

        if not rows:
            return [RawDocument(content="", metadata=base_meta)]

        width = max(map(len, rows))
        headers = rows[0] + [f"Column {i + 1}" for i in range(len(rows[0]), width)]

        def escape(cell: str) -> str:
            return cell.replace("|", "&#124;").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

        markdown_lines = []
        markdown_lines.append("| " + " | ".join(escape(cell) for cell in headers) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows[1:]:
            # Keep extra columns instead of silently truncating records.
            padded_row = (row + [""] * len(headers))[: len(headers)]
            markdown_lines.append("| " + " | ".join(escape(cell) for cell in padded_row) + " |")

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
