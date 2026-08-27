from __future__ import annotations

import re
from typing import Sequence
from backend.models.page_identity import PageIdentity
from backend.utils.logging import logger

# Regex patterns for header/footer page label candidates
_LONE_NUMBER_RE = re.compile(r"^\s*(\d{1,5})\s*$")
_PAGE_PREFIX_RE = re.compile(r"^\s*(?:page|p\.|p\b)\s*([0-9]{1,5}|[ivxlcdm]+|[a-z]-\d{1,4})\s*$", re.IGNORECASE)
_FRACTION_RE = re.compile(r"^\s*(?:page\s*)?([0-9]{1,5}|[ivxlcdm]+|[a-z]-\d{1,4})\s*(?:/|of)\s*\d{1,5}\s*$", re.IGNORECASE)
_ROMAN_RE = re.compile(r"^\s*([ivxlcdm]{1,8})\s*$", re.IGNORECASE)
_ALPHANUMERIC_RE = re.compile(r"^\s*([A-Z]{1,3}-\d{1,4})\s*$", re.IGNORECASE)

# Header/Footer combined line patterns like "DailyDoseofDS.com | 98" or "98 | AI Agents" or "- 98 -"
_EMBEDDED_TRAILING_RE = re.compile(r"(?:[|•·—–\-\s]|^)\s*(?:page\s*)?(\d{1,5}|[ivxlcdm]+|[A-Z]-\d{1,4})\s*$", re.IGNORECASE)
_EMBEDDED_LEADING_RE = re.compile(r"^\s*(?:page\s*)?(\d{1,5}|[ivxlcdm]+|[A-Z]-\d{1,4})\s*(?:[|•·—–\-\s]|$)", re.IGNORECASE)

_VALID_ROMANS = frozenset({
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
})


def _is_valid_roman(s: str) -> bool:
    return s.strip().lower() in _VALID_ROMANS


class PrintedPageDetector:
    """
    Generic Sequence-Aware Printed Page Number & Label Detector.

    Extracts human-visible printed page labels from header/footer text lines,
    validates candidate sequences across consecutive document pages,
    and falls back cleanly when no printed numbers exist.
    """

    @classmethod
    def extract_page_label_candidate(cls, text: str, physical_page_num: int) -> str | None:
        """
        Inspect header (top 3 lines) and footer (bottom 3 lines) of a page text
        to extract the probable printed page number or label.
        """
        if not text or not text.strip():
            return None

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        # Check footer first (bottom 3 lines, prioritizing bottom-most line)
        footer_lines = lines[-3:]
        for line in reversed(footer_lines):
            cand = cls._parse_candidate_line(line)
            if cand is not None:
                return cand

        # Check header next (top 3 lines, prioritizing top-most line)
        header_lines = lines[:3]
        for line in header_lines:
            cand = cls._parse_candidate_line(line)
            if cand is not None:
                return cand

        return None

    @classmethod
    def _parse_candidate_line(cls, line: str) -> str | None:
        """Evaluate a single header/footer line for page number candidates."""
        line_clean = line.strip()
        if not line_clean or len(line_clean) > 80:
            return None

        # 1. Lone number: e.g. "98"
        m = _LONE_NUMBER_RE.match(line_clean)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 99999:
                return str(val)

        # 2. Page prefix: e.g. "Page 98", "p. 98"
        m = _PAGE_PREFIX_RE.match(line_clean)
        if m:
            c = m.group(1).strip()
            if c.isdigit():
                return str(int(c))
            if _is_valid_roman(c) or _ALPHANUMERIC_RE.match(c):
                return c

        # 3. Fraction format: e.g. "98 / 120", "98 of 120"
        m = _FRACTION_RE.match(line_clean)
        if m:
            c = m.group(1).strip()
            if c.isdigit():
                return str(int(c))
            if _is_valid_roman(c) or _ALPHANUMERIC_RE.match(c):
                return c

        # 4. Roman numeral alone: e.g. "iv", "ix"
        m = _ROMAN_RE.match(line_clean)
        if m:
            c = m.group(1).strip()
            if _is_valid_roman(c):
                return c.lower()

        # 5. Alphanumeric label alone: e.g. "A-12"
        m = _ALPHANUMERIC_RE.match(line_clean)
        if m:
            return m.group(1).strip().upper()

        # 6. Embedded in header/footer with titles: e.g. "DailyDoseofDS.com | 98"
        if "|" in line_clean or "—" in line_clean or "–" in line_clean or "·" in line_clean:
            # Trailing page candidate
            m = _EMBEDDED_TRAILING_RE.search(line_clean)
            if m:
                c = m.group(1).strip()
                if c.isdigit() and 1 <= int(c) <= 99999:
                    return str(int(c))
                if _is_valid_roman(c):
                    return c.lower()
                if _ALPHANUMERIC_RE.match(c):
                    return c.upper()

            # Leading page candidate
            m = _EMBEDDED_LEADING_RE.match(line_clean)
            if m:
                c = m.group(1).strip()
                if c.isdigit() and 1 <= int(c) <= 99999:
                    return str(int(c))
                if _is_valid_roman(c):
                    return c.lower()
                if _ALPHANUMERIC_RE.match(c):
                    return c.upper()

        return None

    @classmethod
    def resolve_document_pages(
        cls,
        pages_text: Sequence[tuple[int, str]],  # list of (physical_1_based_page, text)
    ) -> list[PageIdentity]:
        """
        Process an entire document's pages with sequence continuity reconciliation.
        Detects consistent sequence offsets (e.g. physical_page - 1 = printed_page)
        to resolve missing, blurry, or diagram-only pages.
        """
        raw_candidates: list[str | None] = []
        for phys_page, text in pages_text:
            cand = cls.extract_page_label_candidate(text, phys_page)
            raw_candidates.append(cand)

        # Detect predominant integer offset: offset = phys_page - int(cand)
        offset_counts: dict[int, int] = {}
        for (phys_page, _), cand in zip(pages_text, raw_candidates):
            if cand is not None and cand.isdigit():
                offset = phys_page - int(cand)
                offset_counts[offset] = offset_counts.get(offset, 0) + 1

        predominant_offset: int | None = None
        if offset_counts:
            # Pick offset with highest frequency (must appear at least 2 times or on >= 15% of pages)
            best_offset, count = max(offset_counts.items(), key=lambda x: x[1])
            total_pages = len(pages_text)
            if count >= 2 or (total_pages <= 3 and count >= 1):
                predominant_offset = best_offset

        # Build PageIdentity for each page
        identities: list[PageIdentity] = []
        for idx, ((phys_page, text), cand) in enumerate(zip(pages_text, raw_candidates)):
            int_idx = phys_page - 1

            if cand is not None:
                # We have a direct candidate
                if cand.isdigit():
                    disp_val: str | int = int(cand)
                    lbl_val = cand
                else:
                    disp_val = cand
                    lbl_val = cand
            elif predominant_offset is not None:
                # Reconcile via consistent sequence offset
                inferred_num = phys_page - predominant_offset
                if inferred_num > 0:
                    disp_val = inferred_num
                    lbl_val = str(inferred_num)
                else:
                    disp_val = phys_page
                    lbl_val = str(phys_page)
            else:
                # Clean fallback: physical page number
                disp_val = phys_page
                lbl_val = str(phys_page)

            identities.append(
                PageIdentity(
                    internal_page_index=int_idx,
                    physical_page_number=phys_page,
                    display_page_number=disp_val,
                    page_label=lbl_val,
                )
            )

        return identities

    @classmethod
    def detect_single_page(cls, text: str, physical_page_num: int) -> PageIdentity:
        """Detect PageIdentity for a single standalone page."""
        int_idx = max(0, physical_page_num - 1)
        cand = cls.extract_page_label_candidate(text, physical_page_num)

        if cand is not None:
            if cand.isdigit():
                disp = int(cand)
                lbl = cand
            else:
                disp = cand
                lbl = cand
        else:
            disp = physical_page_num
            lbl = str(physical_page_num)

        return PageIdentity(
            internal_page_index=int_idx,
            physical_page_number=physical_page_num,
            display_page_number=disp,
            page_label=lbl,
        )
