from __future__ import annotations

import json
import os
import time
from pathlib import Path
import pytest

from backend.models.document import DocumentCategory, DocumentMetadata, DocumentType, RawDocument
from backend.models.chunk import Chunk, ChunkMetadata, ChunkRole, ContentType

from backend.ingestion.loaders.base import BaseLoader
from backend.ingestion.loaders.pdf import PDFLoader
from backend.ingestion.loaders.docx import DocxLoader
from backend.ingestion.loaders.txt import TxtLoader
from backend.ingestion.loaders.markdown import MarkdownLoader
from backend.ingestion.loaders.html import HTMLLoader
from backend.ingestion.loaders.csv import CSVLoader
from backend.ingestion.loaders.json import JSONLoader
from backend.ingestion.loaders.loader_factory import LoaderFactory, get_loader_for_file, load_document

from backend.ingestion.chunkers.base import BaseChunker
from backend.ingestion.chunkers.recursive import RecursiveChunker
from backend.ingestion.chunkers.semantic import SemanticChunker
from backend.ingestion.chunkers.markdown_aware import MarkdownAwareChunker
from backend.ingestion.chunkers.heading_aware import HeadingAwareChunker
from backend.ingestion.chunkers.table_aware import TableAwareChunker
from backend.ingestion.chunkers.adaptive_chunker import AdaptiveChunker

from backend.utils.section_tracker import SectionTracker, parse_section_heading


# ── Category 1: Empty / Zero-Byte Files ──────────────────────────────────────

def test_empty_txt_file(tmp_path: Path):
    txt_file = tmp_path / "empty.txt"
    txt_file.write_bytes(b"")

    loader = TxtLoader()
    assert loader.supports(txt_file)

    docs = loader.load(txt_file)
    assert len(docs) == 1
    assert docs[0].content == ""
    assert docs[0].metadata.document_type == DocumentType.TXT

    chunker = AdaptiveChunker()
    chunks = chunker.chunk(docs)
    assert chunks == []


def test_empty_markdown_file(tmp_path: Path):
    md_file = tmp_path / "empty.md"
    md_file.write_bytes(b"")

    loader = MarkdownLoader()
    assert loader.supports(md_file)

    docs = loader.load(md_file)
    assert len(docs) == 1
    assert docs[0].content == ""
    assert docs[0].metadata.document_type == DocumentType.MARKDOWN

    chunker = MarkdownAwareChunker()
    chunks = chunker.chunk(docs)
    assert chunks == []


def test_empty_html_file(tmp_path: Path):
    html_file = tmp_path / "empty.html"
    html_file.write_bytes(b"")

    loader = HTMLLoader()
    assert loader.supports(html_file)

    docs = loader.load(html_file)
    assert len(docs) == 1
    assert docs[0].content == ""
    assert docs[0].metadata.document_type == DocumentType.HTML

    chunker = AdaptiveChunker()
    chunks = chunker.chunk(docs)
    assert chunks == []


def test_empty_json_file(tmp_path: Path):
    json_file = tmp_path / "empty.json"
    json_file.write_bytes(b"")

    loader = JSONLoader()
    assert loader.supports(json_file)

    docs = loader.load(json_file)
    assert len(docs) == 1
    assert docs[0].content == ""
    assert docs[0].metadata.document_type == DocumentType.JSON

    chunker = RecursiveChunker()
    chunks = chunker.chunk(docs)
    assert chunks == []


def test_empty_jsonl_file(tmp_path: Path):
    jsonl_file = tmp_path / "empty.jsonl"
    jsonl_file.write_bytes(b"")

    loader = JSONLoader()
    assert loader.supports(jsonl_file)

    docs = loader.load(jsonl_file)
    assert len(docs) == 1
    assert docs[0].content == ""

    chunker = AdaptiveChunker()
    chunks = chunker.chunk(docs)
    assert chunks == []


def test_empty_csv_file(tmp_path: Path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_bytes(b"")

    loader = CSVLoader()
    assert loader.supports(csv_file)

    docs = loader.load(csv_file)
    assert len(docs) == 1
    assert docs[0].content == ""

    chunker = TableAwareChunker()
    chunks = chunker.chunk(docs)
    assert chunks == []


def test_empty_pdf_file(tmp_path: Path):
    pdf_file = tmp_path / "empty.pdf"
    pdf_file.write_bytes(b"")

    loader = PDFLoader()
    assert loader.supports(pdf_file)

    # PDFLoader safely catches PyMuPDF/pypdf exceptions on 0-byte file and returns []
    docs = loader.load(pdf_file)
    assert docs == []


def test_empty_valid_docx_file(tmp_path: Path):
    import docx
    docx_file = tmp_path / "empty_valid.docx"
    doc_obj = docx.Document()
    doc_obj.save(str(docx_file))

    loader = DocxLoader()
    assert loader.supports(docx_file)

    docs = loader.load(docx_file)
    assert len(docs) == 1
    assert docs[0].content == ""


def test_zero_byte_docx_file_handled_gracefully(tmp_path: Path):
    docx_file = tmp_path / "zero_byte.docx"
    docx_file.write_bytes(b"")

    loader = DocxLoader()
    assert loader.supports(docx_file)

    # Verify that DocxLoader safely catches file open exceptions on zero-byte docx
    docs = loader.load(docx_file)
    assert len(docs) == 1
    assert docs[0].content == ""

    # Verify corrupt docx binary file is also handled gracefully
    corrupt_file = tmp_path / "corrupt.docx"
    corrupt_file.write_bytes(b"PK\x03\x04not_a_valid_docx_zip_structure")
    corrupt_docs = loader.load(corrupt_file)
    assert len(corrupt_docs) == 1
    assert corrupt_docs[0].content == ""



def test_loader_factory_with_empty_files(tmp_path: Path):
    txt_file = tmp_path / "empty_factory.txt"
    txt_file.write_bytes(b"")

    docs = load_document(txt_file)
    assert len(docs) == 1
    assert docs[0].content == ""


# ── Category 2: Extremely Large Text/Markdown Files (>5MB) ────────────────────

def test_large_text_file(tmp_path: Path):
    large_txt = tmp_path / "large_policy.txt"
    
    # Generate ~5.2MB of structured plain text with headers and paragraphs
    section_template = (
        "I. SECTION {sec_num} POLICY DETAILS\n"
        "A. Scope and Guidelines for Section {sec_num}\n"
        "This is paragraph 1 of section {sec_num}. All employees must adhere to company policies.\n"
        "B. Compliance Requirements\n"
        "This is paragraph 2 of section {sec_num}. Non-compliance will lead to disciplinary review.\n"
        "1.1 Sub-clause for Section {sec_num}\n"
        "Detailed operational guidelines for remote work, equipment usage, and data privacy.\n\n"
    )
    
    num_sections = 25000
    with open(large_txt, "w", encoding="utf-8") as f:
        for i in range(1, num_sections + 1):
            f.write(section_template.format(sec_num=i))

    file_size_mb = large_txt.stat().st_size / (1024 * 1024)
    assert file_size_mb > 5.0, f"File size is {file_size_mb:.2f}MB, expected >5MB"

    start_load = time.perf_counter()
    loader = TxtLoader()
    docs = loader.load(large_txt)
    load_time = time.perf_counter() - start_load

    assert len(docs) == 1
    assert len(docs[0].content) > 5_000_000

    start_chunk = time.perf_counter()
    chunker = HeadingAwareChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(docs)
    chunk_time = time.perf_counter() - start_chunk

    assert len(chunks) > 0
    assert load_time < 10.0, f"Loading took too long: {load_time:.2f}s"
    assert chunk_time < 30.0, f"Chunking took too long: {chunk_time:.2f}s"


def test_large_markdown_file(tmp_path: Path):
    large_md = tmp_path / "large_guide.md"

    # Generate ~5.5MB markdown with code blocks and markdown tables
    block_template = (
        "# Module {mod_num} Engineering Guide\n\n"
        "## Setup Instructions for Module {mod_num}\n"
        "Run the automated deployment script:\n\n"
        "```python\n"
        "def deploy_module_{mod_num}(config: dict) -> bool:\n"
        "    print('Deploying module {mod_num}')\n"
        "    return True\n"
        "```\n\n"
        "## Status Matrix\n"
        "| Subsystem | Status | Priority |\n"
        "|---|---|---|\n"
        "| Storage | Active | High |\n"
        "| Service | Ready | Medium |\n\n"
    )

    num_blocks = 20000
    with open(large_md, "w", encoding="utf-8") as f:
        for i in range(1, num_blocks + 1):
            f.write(block_template.format(mod_num=i))

    file_size_mb = large_md.stat().st_size / (1024 * 1024)
    assert file_size_mb > 5.0, f"File size is {file_size_mb:.2f}MB, expected >5MB"

    start_load = time.perf_counter()
    loader = MarkdownLoader()
    docs = loader.load(large_md)
    load_time = time.perf_counter() - start_load

    assert len(docs) == 1
    assert docs[0].metadata.has_code is True
    assert docs[0].metadata.has_tables is True

    start_chunk = time.perf_counter()
    chunker = AdaptiveChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk(docs)
    chunk_time = time.perf_counter() - start_chunk

    assert len(chunks) > 0
    assert load_time < 10.0, f"Loading took too long: {load_time:.2f}s"
    assert chunk_time < 30.0, f"Chunking took too long: {chunk_time:.2f}s"


# ── Category 3: Malformed JSON and HTML ──────────────────────────────────────

def test_malformed_json_syntax(tmp_path: Path):
    json_file = tmp_path / "broken.json"
    broken_content = '{ "title": "Company Policy", "items": [1, 2, 3, '  # Unclosed JSON
    json_file.write_text(broken_content, encoding="utf-8")

    loader = JSONLoader()
    docs = loader.load(json_file)
    assert len(docs) == 1
    assert docs[0].content == broken_content
    assert docs[0].metadata.document_type == DocumentType.JSON


def test_malformed_jsonl_partially_corrupt(tmp_path: Path):
    jsonl_file = tmp_path / "corrupt.jsonl"
    lines = [
        '{"id": 1, "valid": true}',
        'THIS IS NOT JSON AT ALL',
        '{"id": 2, "valid": true}',
        '{"id": 3, "unclosed": ',
        '{"id": 4, "valid": true}',
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    loader = JSONLoader()
    docs = loader.load(jsonl_file)
    # Valid lines (3 of them) should be extracted as RawDocuments
    assert len(docs) == 3
    assert "id" in docs[0].content
    assert docs[0].metadata.extra["record_index"] == 1
    assert docs[1].metadata.extra["record_index"] == 3
    assert docs[2].metadata.extra["record_index"] == 5


def test_malformed_html_unclosed_tags(tmp_path: Path):
    html_file = tmp_path / "unclosed.html"
    unclosed_html = (
        "<html><body>"
        "<h1>Policy Overview"  # Unclosed h1
        "<p>This is paragraph 1 without closing tag"
        "<h2>Section 2"
        "<div><span>nested unclosed elements"
    )
    html_file.write_text(unclosed_html, encoding="utf-8")

    loader = HTMLLoader()
    docs = loader.load(html_file)
    assert len(docs) == 1
    assert "Policy Overview" in docs[0].content
    assert "paragraph 1" in docs[0].content


def test_malformed_html_invalid_nested_tables(tmp_path: Path):
    html_file = tmp_path / "bad_tables.html"
    bad_table_html = (
        "<html><body>"
        "<table><tr><th>Col 1<th>Col 2"  # Missing closing th/tr
        "<tr><td>Val A<td>Val B"
        "<table><tr><td>Inner table</tr></table>"
        "</td></tr></table>"
        "</body></html>"
    )
    html_file.write_text(bad_table_html, encoding="utf-8")

    loader = HTMLLoader()
    docs = loader.load(html_file)
    assert len(docs) == 1
    assert docs[0].metadata.has_tables is True


def test_malformed_html_script_and_style_injection(tmp_path: Path):
    html_file = tmp_path / "script_inject.html"
    injection_html = (
        "<html><head><script>alert('xss');</script><style>body {display:none;}</style></head>"
        "<body>"
        "<h1>Security Policy</h1>"
        "<p>Legitimate content here.</p>"
        "<script>document.write('<p>Malicious script write</p>');</script>"
        "<!-- Unclosed comment tag"
        "</body></html>"
    )
    html_file.write_text(injection_html, encoding="utf-8")

    loader = HTMLLoader()
    docs = loader.load(html_file)
    assert len(docs) == 1
    assert "alert('xss')" not in docs[0].content
    assert "display:none" not in docs[0].content
    assert "Security Policy" in docs[0].content
    assert "Legitimate content here." in docs[0].content


# ── Category 4: Deeply Nested Structures ──────────────────────────────────────

def test_deeply_nested_json(tmp_path: Path):
    json_file = tmp_path / "deep.json"
    
    # Construct 150-level nested dict
    nested_obj: dict = {"value": "bottom_leaf"}
    for i in range(150):
        nested_obj = {f"level_{150 - i}": nested_obj}

    json_file.write_text(json.dumps(nested_obj), encoding="utf-8")

    loader = JSONLoader()
    docs = loader.load(json_file)
    assert len(docs) == 1
    assert "bottom_leaf" in docs[0].content

    chunker = RecursiveChunker()
    chunks = chunker.chunk(docs)
    assert len(chunks) > 0


def test_deeply_nested_html_tags(tmp_path: Path):
    html_file = tmp_path / "deep_tags.html"
    
    # 150 nested div tags
    open_tags = "<div>" * 150
    close_tags = "</div>" * 150
    content = f"<html><body>{open_tags}Deep content inside 150 divs{close_tags}</body></html>"
    html_file.write_text(content, encoding="utf-8")

    loader = HTMLLoader()
    docs = loader.load(html_file)
    assert len(docs) == 1
    assert "Deep content inside 150 divs" in docs[0].content


def test_deeply_nested_markdown_headings(tmp_path: Path):
    md_file = tmp_path / "deep_headings.md"
    
    # Generate 50 levels of numbered sections and markdown headers
    lines = []
    prefix = ""
    for i in range(1, 51):
        prefix = f"{prefix}{i}." if prefix else f"{i}."
        lines.append(f"{prefix} Section Level {i}")
        lines.append(f"Content for level {i} under section {prefix}\n")

    md_file.write_text("\n".join(lines), encoding="utf-8")

    loader = MarkdownLoader()
    docs = loader.load(md_file)
    assert len(docs) == 1

    chunker = HeadingAwareChunker()
    chunks = chunker.chunk(docs)
    assert len(chunks) > 0
    # Check section tracker depth
    assert any(c.metadata.section_path is not None for c in chunks)


# ── Category 5: Unicode, Emoji, and Unusual Encodings ────────────────────────

def test_unicode_emojis_and_multibyte_characters(tmp_path: Path):
    txt_file = tmp_path / "unicode_emoji.txt"
    unicode_content = (
        "I. GLOBAL INCLUSION & TECH POLICY 🚀🤖\n"
        "A. Multi-lingual Support 🌍\n"
        "1. Chinese: 欢迎使用 enterprise RAG 系統。\n"
        "2. Japanese: 社内規定ドキュメントの検索および要約機能。\n"
        "3. Korean: 규정 준수 및 보안 정책 안내.\n"
        "4. Arabic (RTL): سياسة الأمن السيبراني وحماية البيانات.\n"
        "5. Hebrew (RTL): מדיניות פרטיות ואבטחת מידע.\n"
        "6. Emojis with ZWJ: 👨‍👩‍👧‍👦 👨‍💻 👩‍🔬 🔥⚡️🎉\n"
        r"7. Math symbols: ∑(x_i^2) + ∫_0^\infty e^{-x} dx = 1 ≠ 0" + "\n"
    )
    txt_file.write_text(unicode_content, encoding="utf-8")

    loader = TxtLoader()
    docs = loader.load(txt_file)
    assert len(docs) == 1
    assert "🚀🤖" in docs[0].content
    assert "سياسة الأمن" in docs[0].content

    chunker = AdaptiveChunker()
    chunks = chunker.chunk(docs)
    assert len(chunks) > 0
    # Verify emojis and unicode survive chunking intact
    combined_text = "".join(c.text for c in chunks)
    assert "🚀🤖" in combined_text
    assert "欢迎ใช้" not in combined_text or "欢迎使用" in combined_text


def test_non_utf8_latin1_encoding(tmp_path: Path):
    txt_file = tmp_path / "latin1_doc.txt"
    latin1_content = "I. POLITIQUES DE CONFIDENTIALITÉ\nClient rôle: privilège & sécurité pour l'employé naïve über déjà."
    txt_file.write_bytes(latin1_content.encode("latin-1"))

    loader = TxtLoader()
    docs = loader.load(txt_file)
    assert len(docs) == 1
    assert "rôle" in docs[0].content or "r\xf4le" in docs[0].content
    assert "privil\xe8ge" in docs[0].content or "privilège" in docs[0].content


def test_utf16_le_encoding_fallback(tmp_path: Path):
    txt_file = tmp_path / "utf16_doc.txt"
    utf16_content = "I. UTF-16 ENCODED POLICY\nThis document is saved with UTF-16 BOM encoding."
    txt_file.write_bytes(utf16_content.encode("utf-16"))

    loader = TxtLoader()
    docs = loader.load(txt_file)
    assert len(docs) == 1
    # Fallback to latin-1/errors=replace should handle non-UTF8 without crash
    assert docs[0].metadata.document_type == DocumentType.TXT


def test_binary_garbage_and_null_bytes(tmp_path: Path):
    txt_file = tmp_path / "binary_garbage.txt"
    garbage_bytes = b"I. HEADER\nSome text before null byte\x00\x01\x02\x03 garbage text after null byte.\nII. FOOTER\n"
    txt_file.write_bytes(garbage_bytes)

    loader = TxtLoader()
    docs = loader.load(txt_file)
    assert len(docs) == 1
    assert "HEADER" in docs[0].content

    chunker = RecursiveChunker()
    chunks = chunker.chunk(docs)
    assert len(chunks) > 0
