from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

_VALID_ROMANS: Set[str] = {
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
}


@dataclass(frozen=True)
class SectionPattern:
    name: str
    level: int
    regex: re.Pattern[str]
    number_group: Optional[int]
    title_group: Optional[int]
    priority: int = 100


@dataclass
class SectionHeading:
    level: int
    section_number: Optional[str]
    section_title: str
    full_label: str
    pattern_name: str


@dataclass
class SectionContext:
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    section_path: Optional[str] = None
    section_level: Optional[int] = None
    headings: List[SectionHeading] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "section_title": self.section_title,
            "section_number": self.section_number,
            "section_path": self.section_path,
            "section_level": self.section_level,
        }


def _build_section_patterns() -> List[SectionPattern]:
    return sorted(
        [
            SectionPattern(
                name="article_section_clause",
                level=4,
                regex=re.compile(
                    r"^(?:Article|Section|Clause|Part|Chapter|Appendix)\s+"
                    r"([\dIVXLC]+(?:\.\d+)*)\s*"
                    r"(?::|\.|\-|–|—)\s*(.+)$",
                    re.IGNORECASE,
                ),
                number_group=1,
                title_group=2,
                priority=10,
            ),
            SectionPattern(
                name="roman_numeral",
                level=1,
                regex=re.compile(r"^([IVXLC]+)\.\s+(.+)$", re.IGNORECASE),
                number_group=1,
                title_group=2,
                priority=20,
            ),
            SectionPattern(
                name="letter_subsection",
                level=2,
                regex=re.compile(r"^([A-Z])\.\s+([A-Z][A-Za-z0-9\s\-,&'()/]{2,})$"),
                number_group=1,
                title_group=2,
                priority=30,
            ),
            SectionPattern(
                name="numbered_section",
                level=3,
                regex=re.compile(r"^(\d+(?:\.\d+){0,3})\s+([A-Z][A-Za-z0-9\s\-,&'()/]{3,})$"),
                number_group=1,
                title_group=2,
                priority=40,
            ),
            SectionPattern(
                name="numbered_trailing_dot",
                level=3,
                regex=re.compile(r"^(\d+(?:\.\d+){0,3})\.\s+([A-Z][A-Za-z0-9\s\-,&'()/]{3,})$"),
                number_group=1,
                title_group=2,
                priority=45,
            ),
            SectionPattern(
                name="all_caps_heading",
                level=5,
                regex=re.compile(r"^([A-Z][A-Z0-9\s\-/&]{5,})$"),
                number_group=None,
                title_group=1,
                priority=60,
            ),
            SectionPattern(
                name="markdown_heading",
                level=1,
                regex=re.compile(r"^(#{1,6})\s+(.+)$"),
                number_group=None,
                title_group=2,
                priority=5,
            ),
        ],
        key=lambda p: p.priority,
    )


SECTION_PATTERNS: List[SectionPattern] = _build_section_patterns()


def is_noise_line(line: str) -> bool:
    lower = line.lower().strip()
    noise_prefixes = (
        "page ",
        "confidential and proprietary",
        "strictly confidential",
        "table of contents",
        "revised ",
        "effective date",
        "last updated",
    )
    if any(lower.startswith(p) for p in noise_prefixes):
        return True
    alpha = sum(c.isalpha() for c in line)
    return alpha < max(3, len(line) * 0.3)


def is_valid_roman(number: str) -> bool:
    return number.upper() in _VALID_ROMANS


def clean_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title.strip())
    return cleaned.rstrip(".:;-–—")[:200]


def parse_section_heading(line: str) -> Optional[SectionHeading]:
    stripped = line.strip()
    if not stripped or len(stripped) < 3 or is_noise_line(stripped):
        return None

    for pattern in SECTION_PATTERNS:
        match = pattern.regex.match(stripped)
        if not match:
            continue

        if pattern.name == "markdown_heading":
            hashes = match.group(1)
            title = clean_title(match.group(2))
            if not title or is_noise_line(title):
                return None
            return SectionHeading(
                level=len(hashes),
                section_number=None,
                section_title=title,
                full_label=stripped[:200],
                pattern_name=pattern.name,
            )

        num_grp = pattern.number_group
        title_grp = pattern.title_group if pattern.title_group is not None else 1

        if num_grp is not None:
            number = match.group(num_grp).strip()
            title = clean_title(match.group(title_grp))
        else:
            number = ""
            title = clean_title(match.group(title_grp))

        if not title or is_noise_line(title):
            continue

        if pattern.name == "roman_numeral" and not is_valid_roman(number):
            continue

        if pattern.name == "all_caps_heading":
            upper = sum(c.isupper() for c in title if c.isalpha())
            alpha = sum(c.isalpha() for c in title)
            if alpha == 0 or upper / alpha < 0.7:
                continue
            number = ""

        if pattern.name == "letter_subsection" and len(number) != 1:
            continue

        return SectionHeading(
            level=pattern.level,
            section_number=number or None,
            section_title=title,
            full_label=stripped[:200],
            pattern_name=pattern.name,
        )

    return None


class SectionTracker:
    """Stack-based section hierarchy tracker for document parsers."""

    def __init__(self) -> None:
        self.stack: List[SectionHeading] = []

    def update(self, heading: SectionHeading) -> SectionContext:
        """Pushes new heading, popping any equal or deeper level headings."""
        while self.stack and self.stack[-1].level >= heading.level:
            self.stack.pop()

        self.stack.append(heading)

        path_elements = [
            f"{h.section_number} {h.section_title}".strip() if h.section_number else h.section_title
            for h in self.stack
        ]
        section_path = " > ".join(path_elements)

        return SectionContext(
            section_title=heading.section_title,
            section_number=heading.section_number,
            section_path=section_path,
            section_level=heading.level,
            headings=list(self.stack),
        )

    def process_line(self, line: str) -> Optional[SectionContext]:
        heading = parse_section_heading(line)
        if heading:
            return self.update(heading)
        return None

    def current_context(self) -> SectionContext:
        if not self.stack:
            return SectionContext()
        path_elements = [
            f"{h.section_number} {h.section_title}".strip() if h.section_number else h.section_title
            for h in self.stack
        ]
        return SectionContext(
            section_title=self.stack[-1].section_title,
            section_number=self.stack[-1].section_number,
            section_path=" > ".join(path_elements),
            section_level=self.stack[-1].level,
            headings=list(self.stack),
        )
