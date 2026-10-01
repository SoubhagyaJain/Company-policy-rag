"""What each store is, why it exists and what happens when it is cleared.

Static descriptions only: sizes and counts are measured by ``StorageService``. The
Storage tab renders these as badges, the "why is this here" drawer, the storage
hierarchy and the data-lifecycle diagram.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PERSISTENT = "PERSISTENT"
CACHE = "CACHE"
REBUILDABLE = "REBUILDABLE"
RUNTIME_ONLY = "RUNTIME_ONLY"
USER_DATA = "USER_DATA"
SYSTEM_DATA = "SYSTEM_DATA"
TEMPORARY = "TEMPORARY"

# Order and labels of the segments in the storage map.
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("models", "Models"),
    ("documents", "Documents"),
    ("vector_db", "Vector database"),
    ("caches", "Caches"),
    ("images", "Extracted images"),
    ("telemetry", "Telemetry"),
    ("logs_eval", "Logs and evaluation"),
    ("other", "Other"),
)


@dataclass(frozen=True)
class StoreInfo:
    kinds: tuple[str, ...]
    category: str
    what: str
    why: str
    created_by: str
    read_by: str
    survives_restart: bool
    deletable: str
    delete_effect: str
    rebuild: str
    cleanup_policy: str

    def describe(self) -> dict[str, Any]:
        return asdict(self)


STORES: dict[str, StoreInfo] = {
    "vector_index": StoreInfo(
        kinds=(PERSISTENT, REBUILDABLE, SYSTEM_DATA),
        category="vector_db",
        what="The Chroma database: one embedding vector, the chunk text and its metadata for every indexed chunk, "
        "plus the semantic answer cache collection.",
        why="Dense retrieval searches these vectors on every question.",
        created_by="Document ingestion (embedding and vector-index stages) and answered questions (semantic cache).",
        read_by="Dense retriever and semantic cache lookups.",
        survives_restart=True,
        deletable="Not from here. It can only be compacted; delete documents from the Library tab.",
        delete_effect="Without it no document can be retrieved until every document is indexed again.",
        rebuild="Re-index each document from its uploaded original (Retry in the Library tab).",
        cleanup_policy="Compaction releases unused pages and the applied write log. No rows are removed.",
    ),
    "telemetry_db": StoreInfo(
        kinds=(PERSISTENT, SYSTEM_DATA),
        category="telemetry",
        what="A SQLite file with query traces, vision and cache events, memory resolutions, errors and ingestion history.",
        why="Feeds the Telemetry tab and the hit rates, last-queried times and growth events shown on this page.",
        created_by="The chat service and the ingestion pipeline, through a background writer.",
        read_by="Telemetry tab and this page.",
        survives_restart=True,
        deletable="Yes. Old records can be deleted by age, or all of them cleared.",
        delete_effect="History before the cutoff disappears from the Telemetry tab. Answers are not affected.",
        rebuild="Cannot be rebuilt. New records accumulate as the app is used.",
        cleanup_policy="Nothing is deleted automatically. Deleting rows frees pages inside the file; "
        "Compact returns them to the disk.",
    ),
    "semantic_cache": StoreInfo(
        kinds=(CACHE, REBUILDABLE, PERSISTENT),
        category="vector_db",
        what="Answers and their citations, keyed by the question's embedding, the model, the corpus version and the prompt version.",
        why="A near-identical question is answered from here without retrieval or generation.",
        created_by="The RAG pipeline, after a grounded answer.",
        read_by="The RAG pipeline, before retrieval.",
        survives_restart=True,
        deletable="Yes.",
        delete_effect="The next occurrence of each cached question is answered from scratch.",
        rebuild="Refills as questions are answered.",
        cleanup_policy="Entries stop matching when the document library, model or prompt changes. "
        "It lives inside the vector database, so its bytes are part of that file.",
    ),
    "page_images": StoreInfo(
        kinds=(PERSISTENT, REBUILDABLE, SYSTEM_DATA),
        category="images",
        what="Images extracted from uploaded documents, one folder per document, with an asset index.",
        why="Visual answers and citation previews show the original figure, table or code screenshot.",
        created_by="Document ingestion.",
        read_by="The vision service and the citation drawer.",
        survives_restart=True,
        deletable="Only images of documents that are no longer in the library.",
        delete_effect="Images of current documents are kept, so nothing visible changes.",
        rebuild="Extracted again when a document is re-indexed.",
        cleanup_policy="Removed with their document. Folders left behind by an interrupted delete are orphaned.",
    ),
    "vision_cache": StoreInfo(
        kinds=(CACHE, REBUILDABLE, PERSISTENT),
        category="caches",
        what="One JSON file per page image the vision model has read: the extracted text and visual type.",
        why="Reading a page with the vision model takes about 25 seconds; a cached reading is instant.",
        created_by="The vision service, during ingestion of scanned PDFs and on visual questions.",
        read_by="The vision service, before it runs the model.",
        survives_restart=True,
        deletable="Yes: orphaned entries, entries older than a cutoff, or everything.",
        delete_effect="Cleared pages are read again by the vision model the next time they are needed.",
        rebuild="Regenerated on demand, about 25 seconds per page.",
        cleanup_policy="Removed with their document. Nothing expires automatically.",
    ),
    "logs": StoreInfo(
        kinds=(PERSISTENT, SYSTEM_DATA),
        category="logs_eval",
        what="The application log and the reports written by evaluation and benchmark runs.",
        why="Debugging and comparing evaluation runs.",
        created_by="The backend logger and the evaluation scripts.",
        read_by="People. The app does not read these back.",
        survives_restart=True,
        deletable="Files older than a cutoff. The live application log is always kept.",
        delete_effect="Old log files and evaluation reports are gone.",
        rebuild="Logs cannot be rebuilt. Reports reappear when an evaluation is run again.",
        cleanup_policy="Nothing is deleted automatically.",
    ),
    "eval_artifacts": StoreInfo(
        kinds=(PERSISTENT, REBUILDABLE, SYSTEM_DATA),
        category="logs_eval",
        what="Benchmark corpora and the vector and keyword indexes the evaluation scripts build from them.",
        why="Lets evaluation runs reuse an index instead of embedding the benchmark documents again.",
        created_by="The evaluation scripts.",
        read_by="The evaluation scripts. Chat does not use them.",
        survives_restart=True,
        deletable="Yes.",
        delete_effect="The next evaluation run rebuilds its indexes first, which re-embeds the benchmark documents.",
        rebuild="Rebuilt by the next evaluation run.",
        cleanup_policy="Nothing is deleted automatically.",
    ),
    "session_libraries": StoreInfo(
        kinds=(PERSISTENT, TEMPORARY),
        category="other",
        what="Uploads and indexes of earlier runs in session mode, where every start gets its own empty library.",
        why="Session mode keeps earlier files on disk instead of deleting them at startup.",
        created_by="The document service, when DOCUMENT_LIBRARY_MODE=session.",
        read_by="Nothing in the current run.",
        survives_restart=True,
        deletable="Yes. The active library is never included.",
        delete_effect="The files uploaded in those earlier sessions are deleted.",
        rebuild="Cannot be rebuilt: they contain the documents uploaded in those sessions.",
        cleanup_policy="Nothing is deleted automatically.",
    ),
    "uploads": StoreInfo(
        kinds=(USER_DATA, PERSISTENT),
        category="documents",
        what="The original files you uploaded.",
        why="Source of truth for indexing, retries and page-image rendering.",
        created_by="Document upload.",
        read_by="Ingestion, retry and the page-image extractor.",
        survives_restart=True,
        deletable="Not from here. Delete documents from the Library tab.",
        delete_effect="Deleting a document there also removes its chunks, vectors, images and vision cache.",
        rebuild="Cannot be rebuilt. Upload the file again.",
        cleanup_policy="Never cleaned up by this page.",
    ),
    "bm25_index": StoreInfo(
        kinds=(PERSISTENT, REBUILDABLE, SYSTEM_DATA),
        category="other",
        what="The keyword (BM25) corpus: every indexed chunk with its tokens and metadata.",
        why="Keyword retrieval runs next to vector search on every question.",
        created_by="Document ingestion (BM25 stage).",
        read_by="The hybrid retriever. Loaded into memory at startup.",
        survives_restart=True,
        deletable="Not from here.",
        delete_effect="Keyword retrieval finds nothing until documents are indexed again.",
        rebuild="Rewritten whenever a document is indexed or deleted.",
        cleanup_policy="Maintained with the library.",
    ),
    "frontend_build": StoreInfo(
        kinds=(CACHE, REBUILDABLE, TEMPORARY),
        category="other",
        what="Next.js dev and build output.",
        why="Speeds up the frontend dev server and holds the production build.",
        created_by="`npm run dev` and `npm run build`.",
        read_by="The frontend dev server.",
        survives_restart=True,
        deletable="Not from here. Delete the folder by hand with the frontend server stopped.",
        delete_effect="The next frontend start compiles everything again.",
        rebuild="Rebuilt by the next `npm run dev` or `npm run build`.",
        cleanup_policy="Managed by Next.js.",
    ),
    "legacy_index": StoreInfo(
        kinds=(PERSISTENT, SYSTEM_DATA),
        category="other",
        what="A vector index written by the older Streamlit pipeline.",
        why="Kept for the legacy entry point. This app does not use it.",
        created_by="The legacy pipeline.",
        read_by="The legacy pipeline only.",
        survives_restart=True,
        deletable="Not from here.",
        delete_effect="The legacy pipeline would have to index its documents again.",
        rebuild="Rebuilt by the legacy pipeline.",
        cleanup_policy="Not managed by this app.",
    ),
}

RUNTIME: dict[str, StoreInfo] = {
    "ollama_models": StoreInfo(
        kinds=(RUNTIME_ONLY,),
        category="runtime",
        what="Chat models the Ollama server holds in memory: the weights plus the working buffers for the context window.",
        why="A loaded model answers immediately; loading one takes about 10 seconds.",
        created_by="Ollama, on the first request for a model or at backend start-up.",
        read_by="Every generated answer.",
        survives_restart=False,
        deletable="Yes: unload a model to free its VRAM.",
        delete_effect="The next question waits for the model to load again.",
        rebuild="Reloaded from the model file on disk.",
        cleanup_policy="Pinned models stay loaded; others are released by Ollama after their keep-alive time.",
    ),
    "vision_model": StoreInfo(
        kinds=(RUNTIME_ONLY,),
        category="runtime",
        what="Qwen3-VL weights held inside the backend process.",
        why="Reads page images for scanned documents and visual questions.",
        created_by="The vision service, the first time a page image has to be read.",
        read_by="The vision service.",
        survives_restart=False,
        deletable="Yes: unload it.",
        delete_effect="The next page reading waits for the model to load again.",
        rebuild="Reloaded from the model directory on disk.",
        cleanup_policy="Stays loaded until it is unloaded or the backend stops.",
    ),
    "kv_cache": StoreInfo(
        kinds=(CACHE, REBUILDABLE),
        category="runtime",
        what="Redis, or an in-memory fallback, holding cached query responses, embeddings and sessions as key-value pairs.",
        why="Application-level caching. This is not the attention KV cache inside a language model.",
        created_by="Backend helpers that cache by key.",
        read_by="The same helpers.",
        survives_restart=False,
        deletable="Yes.",
        delete_effect="Cached entries are dropped and recomputed on demand.",
        rebuild="Refills as the app is used.",
        cleanup_policy="Entries expire after their TTL. With Redis they survive a backend restart; "
        "with the in-memory fallback they do not.",
    ),
    "retrieval_cache": StoreInfo(
        kinds=(CACHE, REBUILDABLE, RUNTIME_ONLY),
        category="runtime",
        what="Candidate chunks remembered per question, filters and corpus version.",
        why="A repeated question skips vector and keyword search.",
        created_by="The hybrid retriever.",
        read_by="The hybrid retriever.",
        survives_restart=False,
        deletable="Yes.",
        delete_effect="The next occurrence of each question runs retrieval again.",
        rebuild="Refills as questions are asked.",
        cleanup_policy="Entries expire after their TTL, the oldest are evicted at the size limit, "
        "and the cache is cleared whenever a document is added or deleted.",
    ),
    "embedding_cache": StoreInfo(
        kinds=(CACHE, REBUILDABLE, RUNTIME_ONLY),
        category="runtime",
        what="Vectors already computed for recently seen text, keyed by a hash of the text.",
        why="Embedding the same text twice is avoided.",
        created_by="The embedding service.",
        read_by="The embedding service.",
        survives_restart=False,
        deletable="Yes.",
        delete_effect="The first retrieval after clearing embeds its text again and may be slower.",
        rebuild="Refills as text is embedded.",
        cleanup_policy="The least recently used vectors are evicted at the size limit.",
    ),
    "conversations": StoreInfo(
        kinds=(RUNTIME_ONLY,),
        category="runtime",
        what="Server-side state of each chat session: recent turns and the evidence they used.",
        why="Follow-up questions are resolved against it.",
        created_by="The chat service.",
        read_by="The conversation interpreter.",
        survives_restart=False,
        deletable="Yes.",
        delete_effect="Follow-up questions in open chats lose their context.",
        rebuild="Cannot be rebuilt. New state forms as chats continue.",
        cleanup_policy="Sessions expire on their own; the oldest are evicted at the size limit.",
    ),
    "docstore": StoreInfo(
        kinds=(RUNTIME_ONLY, REBUILDABLE),
        category="runtime",
        what="Every indexed chunk held in memory.",
        why="Context assembly reads chunk text and neighbours without a database round trip.",
        created_by="Document ingestion, and at start-up from the stored indexes.",
        read_by="The RAG pipeline.",
        survives_restart=False,
        deletable="Not from here.",
        delete_effect="Managed with the library.",
        rebuild="Restored from the stored indexes at start-up.",
        cleanup_policy="Chunks are removed when their document is deleted.",
    ),
}


def store_info(store_id: str) -> dict[str, Any] | None:
    info = STORES.get(store_id) or RUNTIME.get(store_id)
    return info.describe() if info else None


def _leaf(label: str, **ref: str) -> dict[str, Any]:
    return {"label": label, **ref}


# The architecture tree. ``store`` / ``runtime`` / ``model`` name the measured item
# a leaf reads its size from; ``model`` matches model ids by prefix.
HIERARCHY: list[dict[str, Any]] = [
    {"label": "Documents", "children": [_leaf("Uploaded originals", store="uploads")]},
    {
        "label": "Databases",
        "children": [
            _leaf("Chroma vector database", store="vector_index"),
            _leaf("SQLite telemetry", store="telemetry_db"),
        ],
    },
    {
        "label": "Indexes",
        "children": [
            _leaf("BM25 keyword index", store="bm25_index"),
            _leaf("Evaluation indexes", store="eval_artifacts"),
            _leaf("Legacy vector index", store="legacy_index"),
        ],
    },
    {
        "label": "Caches",
        "children": [
            _leaf("Semantic answer cache", store="semantic_cache"),
            _leaf("Vision cache", store="vision_cache"),
            _leaf("Embedding cache", runtime="embedding_cache"),
            _leaf("Retrieval cache", runtime="retrieval_cache"),
            _leaf("Application KV cache", runtime="kv_cache"),
            _leaf("Frontend build cache", store="frontend_build"),
        ],
    },
    {
        "label": "Generated artifacts",
        "children": [
            _leaf("Extracted page images", store="page_images"),
            _leaf("Logs and evaluation reports", store="logs"),
            _leaf("Old session libraries", store="session_libraries"),
        ],
    },
    {
        "label": "Models on disk",
        "children": [
            _leaf("Ollama models", model="ollama:"),
            _leaf("Embedding model", model="embedding_model"),
            _leaf("Reranker", model="reranker_model"),
            _leaf("Vision model", model="vision_model"),
        ],
    },
    {
        "label": "Runtime-only memory",
        "runtime_only": True,
        "children": [
            _leaf("Chat sessions", runtime="conversations"),
            _leaf("Chunk store", runtime="docstore"),
            _leaf("Loaded chat models", runtime="ollama_models"),
            _leaf("Loaded vision model", runtime="vision_model"),
        ],
    },
]

# What ingestion and querying write, in order, and where each artifact lives.
LIFECYCLE: list[dict[str, Any]] = [
    {"label": "Upload document", "detail": "The file arrives over HTTP and is hashed to detect duplicates."},
    {"label": "Original document", "detail": "Stored unchanged.", "store": "uploads"},
    {
        "label": "Extracted text and images",
        "detail": "Text stays in memory; page images are written to disk. Scanned pages are read by the vision model.",
        "store": "page_images",
        "also": ["vision_cache"],
    },
    {"label": "Chunks", "detail": "Text is split into chunks and kept in memory.", "runtime": "docstore"},
    {
        "label": "Embeddings",
        "detail": "One vector per chunk, computed by the embedding model.",
        "runtime": "embedding_cache",
    },
    {"label": "Chroma", "detail": "Vectors, chunk text and metadata are persisted.", "store": "vector_index"},
    {"label": "BM25", "detail": "The same chunks are added to the keyword index.", "store": "bm25_index"},
    {
        "label": "Caches created during queries",
        "detail": "Retrieval results, answers and page readings are remembered as questions are asked.",
        "store": "semantic_cache",
        "also": ["retrieval_cache", "kv_cache"],
    },
]

# Runtime components a question passes through, in order.
MEMORY_FLOW: list[dict[str, Any]] = [
    {"label": "User query"},
    {"label": "Conversation memory", "runtime": "conversations"},
    {"label": "Retrieval cache", "runtime": "retrieval_cache"},
    {"label": "Embedding cache", "runtime": "embedding_cache"},
    {"label": "Chunk store", "runtime": "docstore"},
    {"label": "Reranker", "model": "reranker"},
    {"label": "Loaded LLM", "runtime": "ollama_models"},
    {"label": "Response"},
]
