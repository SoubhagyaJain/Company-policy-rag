"""
Unit Tests for Logical Document Structure and Cross-Page Continuation Engine.
"""

from __future__ import annotations

import pytest

from backend.models.logical_document import (
    BlockType,
    DocumentBlock,
    LogicalDocument,
    LogicalSection,
    detect_continuation_signals,
)


def test_detect_continuation_signals_various_patterns():
    # 1. "Here's the code:"
    cues = detect_continuation_signals("The agent processes data. Here's the code:")
    assert len(cues) > 0
    assert "code" in cues[0].lower()

    # 2. "Here is the code below"
    cues2 = detect_continuation_signals("Here is the code below for the writer agent.")
    assert len(cues2) > 0

    # 3. "The following code implements"
    cues3 = detect_continuation_signals("The following code implements the custom tool:")
    assert len(cues3) > 0

    # 4. "See below"
    cues4 = detect_continuation_signals("For implementation details, see code below.")
    assert len(cues4) > 0

    # 5. "Let's implement"
    cues5 = detect_continuation_signals("Let's implement the writer agent in Python.")
    assert len(cues5) > 0

    # 6. "Output format:"
    cues6 = detect_continuation_signals("Expected output format:\n```json\n{}```")
    assert len(cues6) > 0

    # 7. Non-continuation plain text
    assert len(detect_continuation_signals("Standard employee policy paragraph.")) == 0


def test_logical_section_spans_multiple_pages():
    section = LogicalSection(
        title="X Writer Agent",
        section_number="6",
        section_path="AI Agents > #6 X Writer Agent",
        level=2,
        start_page=63,
        end_page=63,
    )

    # Page 63 text block
    b1 = DocumentBlock(
        block_type=BlockType.TEXT,
        text="The Agent takes the output of the X Analyst agent and generates insights. Here's the code:",
        page_number=63,
        section_title="X Writer Agent",
    )
    section.add_block(b1)

    # Page 64 visual code block
    code_text = (
        "writer_agent = Agent(\n"
        "    role='X Writer Agent',\n"
        "    goal='Draft engaging insights from analysis',\n"
        "    backstory='Expert copywriter specializing in viral summaries'\n"
        ")\n\n"
        "writer_task = Task(\n"
        "    description='Write a summary post from the analyst report',\n"
        "    expected_output='A clean 3-paragraph summary',\n"
        "    agent=writer_agent\n"
        ")"
    )
    b2 = DocumentBlock(
        block_type=BlockType.CODE_SCREENSHOT,
        text=f"```python\n{code_text}\n```",
        raw_code=code_text,
        page_number=64,
        section_title="X Writer Agent",
        is_visual_extraction=True,
    )
    section.add_block(b2)

    assert section.start_page == 63
    assert section.end_page == 64
    assert section.has_code is True
    assert len(section.get_code_blocks()) == 1
    assert "writer_agent = Agent" in section.get_full_text()


def test_logical_document_find_section():
    doc = LogicalDocument(
        document_id="doc_guidebook_001",
        filename="AI_Agents_Guidebook.pdf",
        file_path="/data/AI_Agents_Guidebook.pdf",
        total_pages=100,
    )

    sec_analyst = LogicalSection(
        title="X Analyst Agent",
        section_number="5",
        section_path="AI Agents > #5 X Analyst Agent",
        start_page=61,
        end_page=62,
    )
    sec_writer = LogicalSection(
        title="X Writer Agent",
        section_number="6",
        section_path="AI Agents > #6 X Writer Agent",
        start_page=63,
        end_page=64,
    )
    doc.sections.extend([sec_analyst, sec_writer])

    # Find by page
    assert doc.find_section_by_page(61) == sec_analyst
    assert doc.find_section_by_page(63) == sec_writer
    assert doc.find_section_by_page(64) == sec_writer
    assert doc.find_section_by_page(99) is None

    # Find by title query
    matches = doc.find_sections_by_title("Writer Agent")
    assert len(matches) == 1
    assert matches[0].title == "X Writer Agent"
