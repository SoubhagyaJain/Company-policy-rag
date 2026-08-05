from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.ingestion.loaders.base import BaseLoader
from backend.models.document import DocumentType, RawDocument
from backend.utils.section_tracker import SectionTracker

_Fenced_Code_Regex = re.compile(r"```[\s\S]*?```")
_Table_Regex = re.compile(r"\|[^\n]+\|\n\|[-:\s|]+\|")


class MarkdownLoader(BaseLoader):
    """Loader for Markdown (.md, .markdown) documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".md", ".markdown"]

    def load(
        self,
        file_path: Path,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawDocument]:
        base_meta = self._build_base_metadata(file_path, DocumentType.MARKDOWN, base_metadata)

        content = ""
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1", errors="replace")

        section_tracker = SectionTracker()
        for line in content.splitlines():
            section_tracker.process_line(line)

        ctx = section_tracker.current_context()
        has_code = bool(_Fenced_Code_Regex.search(content)) or "```" in content
        has_tables = bool(_Table_Regex.search(content)) or ("|" in content and "---" in content)

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
