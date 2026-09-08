from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.ingestion.loaders.base import BaseLoader
from backend.ingestion.loaders.validation import read_text
from backend.models.document import DocumentType, RawDocument
from backend.utils.section_tracker import SectionTracker

_CODE_PATTERN = re.compile(r"```|^\s*(def |class |import |function |const |let |var )", re.MULTILINE)
_TABLE_PATTERN = re.compile(r"\|.*\|.*\|")


class TxtLoader(BaseLoader):
    """Loader for plain text (.txt) documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {
            ".txt",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".c",
            ".cpp",
            ".h",
            ".rs",
            ".go",
            ".sql",
            ".log",
            ".rst",
        }

    def load(
        self,
        file_path: Path,
        base_metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        base_meta = self._build_base_metadata(file_path, DocumentType.TXT, base_metadata)

        content = read_text(file_path)

        section_tracker = SectionTracker()
        for line in content.splitlines():
            section_tracker.process_line(line)

        ctx = section_tracker.current_context()
        has_code = bool(_CODE_PATTERN.search(content))
        has_tables = bool(_TABLE_PATTERN.search(content))

        final_meta = base_meta.model_copy(
            update={
                "section_title": ctx.section_title,
                "section_number": ctx.section_number,
                "section_path": ctx.section_path,
                "section_level": ctx.section_level,
                "has_code": has_code,
                "has_tables": has_tables,
            }
        )

        return [RawDocument(content=content, metadata=final_meta)]
