import pytest
from backend.ingestion.page_detector import PrintedPageDetector


def test_candidate_line_parsing():
    # 1. Lone digits
    assert PrintedPageDetector._parse_candidate_line("98") == "98"
    assert PrintedPageDetector._parse_candidate_line("  125  ") == "125"

    # 2. Prefixes
    assert PrintedPageDetector._parse_candidate_line("Page 98") == "98"
    assert PrintedPageDetector._parse_candidate_line("PAGE 98") == "98"
    assert PrintedPageDetector._parse_candidate_line("p. 98") == "98"
    assert PrintedPageDetector._parse_candidate_line("Page iv") == "iv"

    # 3. Fractions
    assert PrintedPageDetector._parse_candidate_line("98 / 120") == "98"
    assert PrintedPageDetector._parse_candidate_line("98 of 120") == "98"
    assert PrintedPageDetector._parse_candidate_line("Page 98 of 120") == "98"

    # 4. Roman numerals
    assert PrintedPageDetector._parse_candidate_line("iv") == "iv"
    assert PrintedPageDetector._parse_candidate_line("XIV") == "xiv"
    assert PrintedPageDetector._parse_candidate_line("xix") == "xix"

    # 5. Alphanumeric
    assert PrintedPageDetector._parse_candidate_line("A-12") == "A-12"
    assert PrintedPageDetector._parse_candidate_line("APP-4") == "APP-4"

    # 6. Embedded in header/footer
    assert PrintedPageDetector._parse_candidate_line("DailyDoseofDS.com | 98") == "98"
    assert PrintedPageDetector._parse_candidate_line("98 | AI Agents Guidebook") == "98"
    assert PrintedPageDetector._parse_candidate_line("Chapter 4 — 98") == "98"

    # 7. False positive rejection
    assert PrintedPageDetector._parse_candidate_line("1. Install all dependencies.") is None
    assert PrintedPageDetector._parse_candidate_line("Firecrawl scrapes web content at $100/mo") is None


def test_sequence_aware_document_reconciliation():
    # Document where physical page 1 is cover (no page number), physical 2 is page 1, physical 99 is page 98
    pages_text = [
        (1, "AI Agents Guidebook\nComplete Hands-on Tutorial"), # Cover
        (2, "Introduction to Agentic Workflows\nOverview\n1"),
        (3, "Core Agent Concepts\n2"),
        (4, "Detailed Diagram\n(No footer)"), # Diagram only
        (5, "Agent Tools and Integrations\n4"),
    ]

    identities = PrintedPageDetector.resolve_document_pages(pages_text)
    assert len(identities) == 5

    # Page 1 (Cover): offset is 1, so inferred display page is 1 - 1 fallback or physical 1
    # Page 2 (Physical 2): detected "1" -> display_page_number = 1
    assert identities[1].physical_page_number == 2
    assert identities[1].internal_page_index == 1
    assert identities[1].display_page_number == 1
    assert identities[1].display_label == "1"

    # Page 3 (Physical 3): detected "2" -> display_page_number = 2
    assert identities[2].display_page_number == 2

    # Page 4 (Physical 4, missing footer text): reconciled via offset = 1 -> display_page_number = 3
    assert identities[3].physical_page_number == 4
    assert identities[3].display_page_number == 3
    assert identities[3].display_label == "3"

    # Page 5 (Physical 5): detected "4" -> display_page_number = 4
    assert identities[4].display_page_number == 4
