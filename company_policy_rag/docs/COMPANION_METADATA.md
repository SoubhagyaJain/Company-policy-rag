# PDF companion metadata

## Scope and source identity

`data/companions/metadata.json` is the supplied knowledge map. The PDF is the only evidence source. The companion is read at PDF ingestion, matched by the **full SHA-256**, validated, and never ingested as document text. The supplied PDF hashes to `d128974852071fd0a2982d85a41e7e705ec9fee9846492542002c11798f39b9d`; the filename is not used for matching. Exactly one valid companion may match. Unsupported schemas, bad structure, duplicates, wrong page counts, and unconfirmed headings are ignored with a warning where applicable. A PDF with no matching hash follows the ordinary path.

The source contains 16 chapters, 179 sections, 698 subsections, 568 concepts, 22 acronyms, seven relationships and 16 topic routes. IDs and parent links in the heading hierarchy and all heading page ranges are validated. The source explicitly describes its excerpts and routes as not independently verified. They, concepts, keywords and relationships are deliberately not used as factual evidence or query expansions. The copy is unchanged.

## Architecture and design

The UI uses `backend/api/routes/documents.py::upload_document_file`; `DocumentService.upload_document` computes a full content digest, generates a UUID-based document ID, deduplicates, saves, and schedules `_execute_ingestion_stages`. `PDFLoader.load` reads physical pages, determines printed page labels, and tracks sections. `AdaptiveChunker.chunk` generates UUID-based chunk IDs and preserves page identity; the service embeds chunk text and indexes in Chroma and BM25. Chunk metadata is persisted in both indexes and the document catalog tracks publication/deletion. Ingestion and deletion clear retrieval cache; the delete API additionally clears semantic cache and conversation state.

`HybridRetriever.retrieve` fuses dense and BM25 rankings with RRF. `RAGPipeline` performs scope/conversation interpretation and rewriting, reranking, context packing and generation; `CitationEngine._build_citation_from_chunk` uses the retrieved chunk's original text, ID and page identity. Citation selection does not itself prove every claim is supported. No new citation or prompt source is introduced.

Options considered:

| Approach | Quality potential | Latency / memory | Complexity and compatibility | Citation integrity |
|---|---|---|---|---|
| Enrich existing chunks (selected) | Modest chapter/section navigation | No extra query service; larger BM25 postings | Small change, needs reingestion | Original chunks remain evidence |
| Separate metadata search | May find aliases and cross-concepts | Additional query index and lookup | Requires lifecycle, authorization and mapping logic | Safe only if hints always lead to original chunks |
| Third hybrid retrieval branch | Potentially broader recall | Additional ranking and fusion cost | More tuning, cache and filter interactions | Same source-evidence requirement |

The chosen implementation validates heading IDs, parents, physical-page bounds and page count. It checks that each heading occurs in extracted source text on its declared **start page**; rejected headings are counted in logs. Confirmed title strings are attached as `extra.companion_navigation` to chunks on the declared page range. The end of each declared range and chunk-level section boundaries are not independently verified, so hints can be coarse or misplaced. The existing BM25 field list indexes those titles; text-only/custom BM25 configurations do not unless `companion_navigation` is explicitly included. No hints go into embeddings or citation fields, no full JSON is placed into prompts, and the existing BM25 catalog and metadata filters remain in force. Cross-concept expansion, acronym expansion, and metadata-aware reranking are not implemented or validated.

Flow: Documents UI → upload and full hash → PDF extraction → chunking → validate matching companion and verify heading starts → attach navigation hints → normal embedding/Chroma/BM25 indexing → existing conversation rewriting and hybrid search → existing reranking and context packing → LLM → citations from PDF chunks. Any retrieved page or relationship still needs evidence in the original text.

## Use and rollback

1. The supplied JSON is already at `company_policy_rag/data/companions/metadata.json`. Add other JSON companions in that directory only when their schema and full `document.source_sha256` match their PDF. Do not upload the JSON as a document.
2. Restart the backend to load the code, then upload the original PDF through Documents. The ingestion log reports `Companion source=... enriched_chunks=... rejected_headings=...`. Wait for READY. The PDF may have any filename if its bytes match.
3. Existing indexed PDFs are not retroactively enriched. The existing duplicate check blocks reuploading identical bytes; to apply or update a companion to an already indexed PDF, delete it using Documents and upload it again. This regenerates embeddings. Plan for the indexing time and loss of its old document ID. Editing or deleting the JSON after indexing does not invalidate existing hints automatically: reingest to apply changes, and clear any relevant answer caches. Deleting the PDF removes its associated chunks and hints; the reusable JSON file remains. Changes to the PDF produce a different hash and will not match this companion.
4. To disable new companion indexing remove/move the JSON, or remove `companion_navigation` from `BM25_METADATA_FIELDS`. Already indexed hints remain in persisted chunk metadata; remove that BM25 field or delete and reingest to remove them. Restore the prior field list and code to roll back; no schema migration is needed.

## Verification performed

From `company_policy_rag`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_companion_metadata.py tests\test_retrieval_experiment_flags.py tests\integration\test_ingestion_process_recovery.py -q
.\.venv\Scripts\python.exe scripts\probe_companion_book.py "C:\Users\jains\Downloads\Machine_Learning_Systems_Vol1_No_Blank_Pages.pdf"
```

The test run passed 30 tests (four Chroma/Pydantic deprecation warnings). The probe uses the real Documents API, loader, chunker, Chroma and BM25, but **stub embeddings and disabled image extraction** in a temporary isolated library. It extracted 980 pages, created/indexed 4,315 chunks, enriched 3,832, rejected 12 heading starts, rejected a duplicate with HTTP 409, and removed all vectors and BM25 entries on deletion. The test client is in-process and does not upload to the user's running application. Windows may retain locked temporary Chroma files until the process exits.

An exploratory BM25-only comparison on five manually inspected physical target pages (45, 440, 730, 825, 548) produced:

| Measure | Before | After | Difference |
|---|---:|---:|---:|
| Target-page hit@5 | 0.80 | 0.80 | 0.00 |
| MRR@10, target page | 0.60 | 0.70 | +0.10 |
| Median BM25 query time | 3.70 ms | 4.24 ms | +0.54 ms |

Each query has only one labeled page; other relevant pages are not exhaustively labeled. The results are not Recall@5 across all relevant evidence, not a full-system benchmark, and the five timing samples cannot characterize latency. nDCG, evidence coverage, answer correctness, citation precision, hybrid Recall@5, end-to-end P95, and multi-turn API conversations were not measured. The running local Ollama service exists, but this probe does not use its real embeddings or generation. A production-model A/B evaluation with independently labeled chunks and multi-turn cited answers is still required before claiming benefit. Document-level authorization is not implemented by this feature; it relies on the application's existing scope/catalog filters and does not add an access-control system.

For a manual conversation after uploading, POST to `/api/chat` with a shared `session_id`, `document_scope` set to `current_document`, and `active_document_id` set to the returned ID. Ask an initial topic question, a follow-up, and a chapter-navigation question, then inspect each returned `citations` entry against the original PDF. Repeat with an unrelated question and a topic switch. These steps are instructions for future verification, not tests already performed.
