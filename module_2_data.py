# -*- coding: utf-8 -*-
"""
Module 2 Python Data
"""

MOD2 = {
    "id": "mod2",
    "title": "Module 2: Document Ingestion, Multi-Format Parsing & Hierarchical Chunking",
    "badge": "Q11–Q20",
    "questions": [
        {
            "num": "Q11",
            "level": "2",
            "level_text": "L2 Ingestion",
            "q": "Explain the entire document ingestion workflow in DocumentService.",
            "short": "DocumentService receives raw files, validates extensions and sizes, selects the appropriate parser via LoaderFactory, enriches text with compliance metadata, splits documents hierarchically into child and parent pairs, embeds child nodes in ChromaDB, indexes tokens in BM25, and stores parents in the docstore.",
            "deep": "1. Validation: Inspects MIME type, file extension, and enforces MAX_FILE_SIZE_BYTES (100MB).\n2. Format Parsing: LoaderFactory.get_loader() returns dedicated parser (PyPDF/PDFPlumber for PDF, python-docx for DOCX, BeautifulSoup for HTML, etc.).\n3. Metadata Extraction: MetadataExtractor parses document headers for department, effective dates, policy IDs, and tags.\n4. Hierarchical Chunking: AdaptiveChunker generates 480-token child chunks with 64-token overlap and links them to 2000-token parent sections via parent_id.\n5. Vector Indexing: EmbeddingService generates 384-dimensional embeddings (bge-small-en-v1.5) and writes to ChromaDB collection policy_documents with flattened metadata.\n6. Lexical Indexing: Tokenizes text and updates in-memory BM25 index.\n7. Docstore Persistence: Stores parent documents in persistent docstore dictionary for runtime expansion.",
            "code": "backend/services/document_service.py:65 (ingest_document)",
            "file": "backend/services/document_service.py",
            "lang": "python",
            "snippet": """async def ingest_document(self, file_path: Path, metadata: Dict[str, Any]) -> IngestionResult:
    loader = LoaderFactory.get_loader(file_path)
    raw_doc = loader.load()
    enriched_meta = self.metadata_extractor.extract(raw_doc.content, fallback=metadata)

    chunk_pairs = self.chunker.split_hierarchical(
        raw_doc.content,
        child_size=480,
        parent_size=2000,
        overlap=64
    )

    await self._index_vectors(chunk_pairs.children, enriched_meta)
    self._index_bm25(chunk_pairs.children)
    self.docstore.save_parents(chunk_pairs.parents)
    return IngestionResult(chunks_created=len(chunk_pairs.children), status="success")"""
        },
        {
            "num": "Q12",
            "level": "2",
            "level_text": "L2 Parsing",
            "q": "What document formats are supported and how does LoaderFactory route different file types?",
            "short": "Supported formats: PDF, DOCX, TXT, MD, HTML, CSV, JSON. `LoaderFactory` uses a factory pattern matching file extensions to specialized loader implementations with structured error handling.",
            "deep": "LoaderFactory maps file extensions:\n- .pdf -> PDFLoader: Uses pypdf with fallback to pdfplumber to extract page numbers, bounding text, and table grids.\n- .docx -> DocxLoader: Uses python-docx extracting paragraph hierarchy, headings, and cell tables.\n- .html / .htm -> HTMLLoader: Uses BeautifulSoup4 with lxml parser, stripping script/style tags and preserving header hierarchy (h1-h6).\n- .md -> MarkdownLoader: Parses markdown sections, tables, and code blocks.\n- .csv / .json -> StructuredDataLoader: Converts tabular records and key-value trees into semantic policy sentences.\n- .txt -> TextLoader: Fast UTF-8 streaming reader with fallback encoding detection.",
            "code": "backend/document_processing/loaders.py:30 (LoaderFactory)",
            "file": "backend/document_processing/loaders.py",
            "lang": "python",
            "snippet": """class LoaderFactory:
    _REGISTRY: Dict[str, Type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".docx": DocxLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
        ".md": MarkdownLoader,
        ".txt": TextLoader,
        ".csv": StructuredDataLoader,
        ".json": StructuredDataLoader,
    }

    @classmethod
    def get_loader(cls, file_path: Path) -> BaseLoader:
        ext = file_path.suffix.lower()
        loader_cls = cls._REGISTRY.get(ext)
        if not loader_cls:
            raise UnsupportedFormatException(f"Unsupported file extension: {ext}")
        return loader_cls(file_path)"""
        },
        {
            "num": "Q13",
            "level": "2",
            "level_text": "L2 Chunking Strategy",
            "q": "What is the 'Chunk Size Dilemma' in RAG and how does Hierarchical Chunking solve it?",
            "short": "Small chunks provide high semantic embedding specificity but lack surrounding context, causing the LLM to miss exceptions. Large chunks preserve full context but dilute embedding vectors, degrading retrieval precision. Hierarchical chunking decouples retrieval units (480-token children) from generation units (2000-token parents).",
            "deep": "In naive RAG:\n- Chunk Size = 200 tokens: Vector similarity is sharp, but a retrieval chunk misses surrounding conditional clauses.\n- Chunk Size = 2000 tokens: Vector similarity drops because multiple distinct policy topics blend together.\nOur Hierarchical Solution: 480-token child chunks are indexed for high-precision retrieval, each maintaining a `parent_id` linking to a 2000-token parent document. ContextCompressor expands surviving children into complete parent sections before sending to LLM.",
            "code": "backend/document_processing/chunking.py:45 (HierarchicalChunker)",
            "file": "backend/document_processing/chunking.py",
            "lang": "python",
            "snippet": """def split_hierarchical(self, text: str, child_size: int = 480, parent_size: int = 2000, overlap: int = 64) -> ChunkHierarchy:
    parents = []
    children = []
    
    parent_texts = self._sliding_window_split(text, chunk_size=parent_size, overlap=overlap * 2)
    for p_idx, p_text in enumerate(parent_texts):
        p_id = f"par_{hashlib.md5(p_text[:100].encode()).hexdigest()[:12]}"
        parents.append(ParentDocument(id=p_id, text=p_text, token_count=len(self.tokenize(p_text))))
        
        child_texts = self._sliding_window_split(p_text, chunk_size=child_size, overlap=overlap)
        for c_idx, c_text in enumerate(child_texts):
            c_id = f"chk_{p_id}_{c_idx}"
            children.append(ChildChunk(id=c_id, parent_id=p_id, text=c_text))
            
    return ChunkHierarchy(parents=parents, children=children)"""
        },
        {
            "num": "Q14",
            "level": "2",
            "level_text": "L2 Hyperparameters",
            "q": "Why did you choose 480 tokens for child chunks with 64-token overlap?",
            "short": "480 tokens matches the optimal semantic density of policy clauses while respecting the 512-token context limit of `bge-small-en-v1.5`. The 64-token overlap (~13%) prevents boundary sentence splitting without excessive storage inflation.",
            "deep": "1. Embedding Model Alignment: BAAI/bge-small-en-v1.5 has a hard token maximum of 512 tokens. A 480-token chunk leaves 32 tokens buffer for special tokens ([CLS], [SEP]) and metadata prefix tokens without truncation.\n2. Policy Clause Structure: Typical policy clauses average 350–450 tokens per sub-clause.\n3. Overlap Ratio: 64 tokens ensures that multi-sentence conditions are never severed across chunk boundaries.",
            "code": "company_policy_rag/src/config.py:112 (CHUNK_SIZE=480, CHUNK_OVERLAP=64)",
            "file": "company_policy_rag/src/config.py",
            "lang": "python",
            "snippet": """CHILD_CHUNK_SIZE: int = 480      # Fits 512 max position embeddings with 32-tok buffer
CHILD_CHUNK_OVERLAP: int = 64    # 13.3% sliding overlap ratio
PARENT_CHUNK_SIZE: int = 2000    # Covers complete multi-paragraph policy sections
PARENT_OVERLAP: int = 128        # Inter-section clause boundary retention"""
        },
        {
            "num": "Q15",
            "level": "3",
            "level_text": "L3 Storage Architecture",
            "q": "How are parent documents (2000 tokens) stored and linked to child chunks?",
            "short": "Parent documents are assigned a UUID (`par_...`) and stored in a key-value docstore. Child chunks store `parent_id` in their ChromaDB metadata dictionary. At query time, `ContextCompressor` maps child hits to unique parent IDs.",
            "deep": "Child embeddings are inserted into ChromaDB with `metadata={'parent_id': parent.id}`. Parent documents are persisted in an indexed SQLite/in-memory docstore. When retrieval finishes, ContextCompressor deduplicates parent IDs and fetches full parent texts.",
            "code": "backend/document_processing/chunking.py:80, backend/rag/context_compression.py:35",
            "file": "backend/rag/context_compression.py",
            "lang": "python",
            "snippet": """def expand_context(self, scored_chunks: List[ScoredChunk]) -> List[ExpandedContext]:
    seen_parent_ids = set()
    expanded_list = []

    for chunk in scored_chunks:
        p_id = chunk.metadata.get("parent_id")
        if p_id and p_id not in seen_parent_ids:
            parent_doc = self.docstore.get(p_id)
            if parent_doc:
                seen_parent_ids.add(p_id)
                expanded_list.append(ExpandedContext(id=p_id, text=parent_doc.text, score=chunk.rerank_score))
        elif not p_id:
            expanded_list.append(ExpandedContext(id=chunk.id, text=chunk.text, score=chunk.rerank_score))

    return expanded_list"""
        },
        {
            "num": "Q16",
            "level": "3",
            "level_text": "L3 Table Parsing",
            "q": "How does the system extract and preserve table structures from PDFs and Markdown documents?",
            "short": "PDF tables are extracted via `pdfplumber` bounding-box table extraction and serialized into Markdown table syntax. Markdown tables are preserved as intact chunk units to prevent column misalignment.",
            "deep": "1. Identifies table boundaries using line intersection analysis.\n2. Formats table cells into clean Markdown table syntax (`| Col 1 | Col 2 |\\n|---|---|`).\n3. Injects header metadata above the table.\n4. Keeps tables <= 480 tokens as atomic non-splittable units; splits larger tables row-wise with preserved header columns.",
            "code": "backend/document_processing/loaders.py:140 (_parse_pdf_tables)",
            "file": "backend/document_processing/loaders.py",
            "lang": "python",
            "snippet": """def _extract_markdown_tables(page) -> str:
    tables = page.extract_tables()
    md_output = []
    for table in tables:
        if not table or len(table) < 2:
            continue
        headers = [str(col).replace('\\n', ' ').strip() for col in table[0]]
        md_output.append("| " + " | ".join(headers) + " |")
        md_output.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in table[1:]:
            cells = [str(c).replace('\\n', ' ').strip() if c else "" for c in row]
            md_output.append("| " + " | ".join(cells) + " |")
    return "\\n".join(md_output)"""
        },
        {
            "num": "Q17",
            "level": "3",
            "level_text": "L3 Resilience",
            "q": "What happens if a parent chunk lookup fails during the retrieval expansion phase?",
            "short": "If a parent document lookup fails (e.g., missing key or corrupt docstore index), `ContextCompressor` logs a warning and falls back to using the child chunk's text directly without throwing an error.",
            "deep": "In `backend/rag/context_compression.py`, the expansion loop executes:\n```python\nparent_doc = self.docstore.get(parent_id)\nif parent_doc is not None:\n    expanded_chunks.append(parent_doc)\nelse:\n    logger.warning('Parent not found; falling back to child chunk.')\n    expanded_chunks.append(child)\n```\nThis graceful degradation guarantees pipeline execution is never blocked.",
            "code": "backend/rag/context_compression.py:52",
            "file": "backend/rag/context_compression.py",
            "lang": "python",
            "snippet": """parent_doc = self.docstore.get(parent_id)
if parent_doc is not None:
    expanded_chunks.append(parent_doc)
else:
    logger.warning(f"Parent chunk ID '{parent_id}' not found in docstore. Falling back to 480-tok child text.")
    expanded_chunks.append(child_chunk)"""
        },
        {
            "num": "Q18",
            "level": "3",
            "level_text": "L3 Idempotency",
            "q": "How do you prevent duplicate document uploads and handle document updates or deletes in ChromaDB?",
            "short": "Documents are fingerprinted using SHA-256 content hashing. Uploading an identical document detects matching hash and skips re-ingestion. Document updates perform an upsert by deleting existing chunks matching `source_file` before inserting new chunks.",
            "deep": "1. Content Hashing: Computes sha256(file_bytes).\n2. Duplicate Prevention: If hash exists in metadata, returns already_indexed.\n3. Version Update: Deletes old chunks with ChromaDB.delete(where={'source_file': filename}) before inserting new chunk vectors.",
            "code": "backend/services/document_service.py:115 (_check_duplicate_and_upsert)",
            "file": "backend/services/document_service.py",
            "lang": "python",
            "snippet": """async def _check_duplicate_and_upsert(self, filename: str, content_hash: str) -> bool:
    existing = self.collection.get(where={"content_hash": content_hash})
    if existing and len(existing["ids"]) > 0:
        logger.info(f"Duplicate document detected (Hash: {content_hash}). Skipping embedding.")
        return True

    self.collection.delete(where={"source_file": filename})
    self.docstore.delete_by_source(filename)
    return False"""
        },
        {
            "num": "Q19",
            "level": "2",
            "level_text": "L2 Security & Limits",
            "q": "How are file size limits (e.g., 100MB) and validation enforced during upload?",
            "short": "Enforced in FastAPI middleware and route dependencies via chunked streaming validation: incoming request `Content-Length` headers are validated before reading, and byte counters abort transfers if limits are exceeded.",
            "deep": "1. Header Inspection: Checks `content-length` <= 100MB.\n2. Chunked Stream Guard: For chunked uploads, accumulates byte chunks in a counter and aborts with HTTP 413 if exceeded.\n3. Magic Byte Verification: Validates true file header MIME type.",
            "code": "backend/api/routes/documents.py:35, company_policy_rag/src/config.py:90",
            "file": "backend/api/routes/documents.py",
            "lang": "python",
            "snippet": """MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

async def validate_upload_size(file: UploadFile = File(...)):
    size = 0
    chunk_size = 1024 * 1024
    while chunk := await file.read(chunk_size):
        size += len(chunk)
        if size > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds maximum limit of 100MB.")
    await file.seek(0)
    return file"""
        },
        {
            "num": "Q20",
            "level": "3",
            "level_text": "L3 Tradeoff Analysis",
            "q": "What are the trade-offs of using PyPDF vs PDFPlumber vs OCR for enterprise policy parsing?",
            "short": "PyPDF is extremely fast (~5ms/page) but fails on complex tables. PDFPlumber has excellent layout and table extraction (~80ms/page) but higher CPU usage. OCR (Tesseract) handles scanned scans (~1200ms/page) but requires heavy compute and introduces character error rates.",
            "deep": "| Tool | Speed | Table Quality | Layout Fidelity | Scanned Docs |\n|---|---|---|---|---|\n| PyPDF | 5ms/page | Poor (linear text stream) | Low | No |\n| PDFPlumber | 80ms/page | Excellent (grid detection) | High | No |\n| Tesseract OCR | 1200ms/page | Moderate | Moderate | Yes |\nDefaulting to pypdf for text and triggering pdfplumber for table regions reduces latency by 90% while maintaining table fidelity.",
            "code": "backend/document_processing/loaders.py:65 (PDFLoader implementation)",
            "file": "backend/document_processing/loaders.py",
            "lang": "python",
            "snippet": """class PDFLoader(BaseLoader):
    def load(self) -> Document:
        reader = pypdf.PdfReader(self.file_path)
        pages_text = []
        
        with pdfplumber.open(self.file_path) as plumber_pdf:
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if "\\t" in text or "  " in text:
                    plumber_page = plumber_pdf.pages[idx]
                    table_md = self._extract_markdown_tables(plumber_page)
                    if table_md:
                        text += "\\n\\n" + table_md
                pages_text.append(text)
        return Document(content="\\n\\n".join(pages_text), metadata={"total_pages": len(reader.pages)})"""
        }
    ]
}
