from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class PageIdentity(BaseModel):
    """
    Canonical Page Identity Contract.

    Reconciles:
    1. internal_page_index: 0-based index used by PyMuPDF / fitz (e.g. 98)
    2. physical_page_number: 1-based physical sheet/stream page in PDF (e.g. 99)
    3. display_page_number: Human-visible printed document page number (e.g. 98, "iv", "A-12")
    4. page_label: Normalized string representation of the human-visible page label (e.g. "98")
    """

    internal_page_index: int = Field(..., description="0-based zero-indexed page number in document reader")
    physical_page_number: int = Field(..., description="1-based physical sheet page number in PDF")
    display_page_number: str | int | None = Field(default=None, description="Human-visible printed page number/label")
    page_label: str = Field(..., description="Normalized string representation of display page label")

    @property
    def display_label(self) -> str:
        """
        Resolved human-facing display label with fallback hierarchy:
        display_page_number -> page_label -> physical_page_number -> internal_page_index + 1
        """
        if self.display_page_number is not None and str(self.display_page_number).strip():
            return str(self.display_page_number).strip()
        if self.page_label and str(self.page_label).strip():
            return str(self.page_label).strip()
        if self.physical_page_number > 0:
            return str(self.physical_page_number)
        return str(self.internal_page_index + 1)

    @classmethod
    def from_indices(
        cls,
        internal_page_index: int | None = None,
        physical_page_number: int | None = None,
        display_page_number: str | int | None = None,
        page_label: str | None = None,
    ) -> PageIdentity:
        """
        Factory method to construct a canonical PageIdentity from partial or complete parameters.
        Guarantees that internal_page_index and physical_page_number are consistent.
        """
        if internal_page_index is None and physical_page_number is not None:
            int_idx = max(0, physical_page_number - 1)
            phys_num = physical_page_number
        elif internal_page_index is not None and physical_page_number is None:
            int_idx = max(0, internal_page_index)
            phys_num = int_idx + 1
        elif internal_page_index is not None and physical_page_number is not None:
            int_idx = max(0, internal_page_index)
            phys_num = max(1, physical_page_number)
        else:
            int_idx = 0
            phys_num = 1

        # Resolve display_page_number and page_label
        resolved_display: str | int | None = display_page_number
        if resolved_display is None and page_label is not None and str(page_label).strip():
            lbl_str = str(page_label).strip()
            if lbl_str.isdigit():
                resolved_display = int(lbl_str)
            else:
                resolved_display = lbl_str

        # If display_page_number is provided but page_label is not
        if page_label is None:
            if resolved_display is not None:
                lbl = str(resolved_display).strip()
            else:
                lbl = str(phys_num)
        else:
            lbl = str(page_label).strip() or str(phys_num)

        return cls(
            internal_page_index=int_idx,
            physical_page_number=phys_num,
            display_page_number=resolved_display,
            page_label=lbl,
        )

    def matches_display(self, identifier: int | str | None) -> bool:
        """Return whether a user-facing page reference matches the printed label.

        Internal and physical page numbers deliberately do not participate here.
        A reference such as ``Page 98`` is a reference to the document's printed
        page number, not to a zero-based parser index or a PDF sheet number.
        """
        if identifier is None:
            return False

        if isinstance(identifier, int):
            return (
                (isinstance(self.display_page_number, int) and identifier == self.display_page_number)
                or (str(self.display_page_number).isdigit() and identifier == int(str(self.display_page_number)))
            )

        ident_str = str(identifier).strip().lower()
        if not ident_str:
            return False

        # Direct string matches
        if ident_str in (
            str(self.display_page_number).lower() if self.display_page_number is not None else "",
            self.page_label.lower(),
            self.display_label.lower(),
        ):
            return True

        # Clean "Page 98" or "p. 98"
        import re
        m = re.search(r"\b(\d{1,5}|[ivxlcdm]+|[a-z]-\d+)\b", ident_str)
        if m:
            clean_val = m.group(1)
            if self.display_page_number is not None and clean_val == str(self.display_page_number).lower():
                return True
            if clean_val == self.page_label.lower():
                return True

        return False

    def matches_internal_index(self, internal_page_index: int | None) -> bool:
        """Return whether an explicit parser/renderer index matches this page."""
        return internal_page_index is not None and internal_page_index == self.internal_page_index

    def matches_physical_page(self, physical_page_number: int | None) -> bool:
        """Return whether an explicit 1-based PDF sheet number matches this page."""
        return physical_page_number is not None and physical_page_number == self.physical_page_number

    def matches(self, identifier: int | str | None) -> bool:
        """Backward-compatible alias for user-facing display-page matching."""
        return self.matches_display(identifier)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for embedding in chunk/document metadata."""
        return {
            "internal_page_index": self.internal_page_index,
            "physical_page_number": self.physical_page_number,
            "display_page_number": self.display_page_number,
            "page_label": self.page_label,
            "page_number": self.physical_page_number,  # Backwards compatibility
        }
