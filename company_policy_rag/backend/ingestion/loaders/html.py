from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.ingestion.loaders.base import BaseLoader
from backend.models.document import DocumentType, RawDocument
from backend.utils.logging import logger
from backend.utils.section_tracker import SectionTracker


class HTMLLoader(BaseLoader):
    """Loader for HTML (.html, .htm) documents."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in [".html", ".htm"]

    def load(
        self,
        file_path: Path,
        base_metadata: dict[str, Any] | None = None,
    ) -> list[RawDocument]:
        base_meta = self._build_base_metadata(file_path, DocumentType.HTML, base_metadata)

        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            logger.error("beautifulsoup4 is not installed: %s", e)
            raise RuntimeError("beautifulsoup4 required for html files") from e

        html_text = ""
        try:
            html_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            html_text = file_path.read_text(encoding="latin-1", errors="replace")

        soup = BeautifulSoup(html_text, "html.parser")

        # Strip script/style tags
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()

        has_tables = bool(soup.find("table"))
        has_code = bool(soup.find("code") or soup.find("pre"))

        # Convert HTML tables to Markdown tables
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True).replace("\n", " ") for td in tr.find_all(["th", "td"])]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                header_count = len(rows[0].split("|")) - 2
                divider = "| " + " | ".join(["---"] * max(1, header_count)) + " |"
                if len(rows) > 1:
                    rows.insert(1, divider)
                markdown_table = soup.new_tag("p")
                markdown_table.string = "\n" + "\n".join(rows) + "\n"
                table.replace_with(markdown_table)

        # Convert headings to Markdown style headings in text
        for level in range(1, 7):
            for h_tag in soup.find_all(f"h{level}"):
                h_text = h_tag.get_text(strip=True)
                h_tag.string = f"\n\n{'#' * level} {h_text}\n"

        text_content = soup.get_text(separator="\n", strip=True)

        section_tracker = SectionTracker()
        for line in text_content.splitlines():
            section_tracker.process_line(line)

        ctx = section_tracker.current_context()

        # Extract title if present
        title_tag = soup.find("title")
        doc_title = title_tag.get_text(strip=True) if title_tag else ctx.section_title

        final_meta = base_meta.model_copy(
            update={
                "section_title": doc_title or ctx.section_title,
                "section_number": ctx.section_number,
                "section_path": ctx.section_path,
                "section_level": ctx.section_level,
                "has_tables": has_tables,
                "has_code": has_code,
            }
        )

        return [RawDocument(content=text_content, metadata=final_meta)]
