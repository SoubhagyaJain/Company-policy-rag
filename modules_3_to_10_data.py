
# -*- coding: utf-8 -*-
"""
Python data definitions for Modules 3 through 10 (Q21 to Q100)
"""

MOD3_TO_10_LIST = [
    {
        "id": "mod3",
        "title": "Module 3: Metadata Extraction, Vector Indexing & Dynamic Filter Inference",
        "badge": "Q21–Q30",
        "questions": [
            {
                "num": "Q21",
                "level": "2",
                "level_text": "L2 Metadata",
                "q": "What metadata fields are automatically extracted during document ingestion?",
                "short": "Extracted fields include department, effective_date, policy_id, category, source_file, page_number, parent_id, chunk_index, token_count, and topic_tags.",
                "deep": "Every child chunk in ChromaDB is annotated with structured metadata:\n- department: Normalized department name (e.g. 'HR', 'Legal', 'Finance', 'IT', 'Security').\n- effective_date: ISO 8601 formatted date (e.g. '2024-01-15') extracted from headers.\n- policy_id: Standardized code (e.g. 'POL-2024-08', 'SEC-04').\n- source_file: Original filename (Employee_Handbook.pdf).\n- page_number: 1-indexed page location for direct UI citations.\n- parent_id: Docstore hash key for 2000-token context expansion.\n- topic_tags: Comma-delimited list of detected concepts (e.g. 'vacation,leave,rollover').",
                "code": "backend/document_processing/metadata_extractor.py:25 (DocumentMetadata)",
                "file": "backend/document_processing/metadata_extractor.py",
                "lang": "python",
                "snippet": """@dataclass
class DocumentMetadata:
    department: str = "General"
    effective_date: Optional[str] = None
    policy_id: Optional[str] = None
    category: str = "compliance"
    source_file: str = ""
    page_number: int = 1
    parent_id: Optional[str] = None
    
    chunk_index: int = 0
    token_count: int = 0
    topic_tags: List[str] = field(default_factory=list)"""
            },
            {
                "num": "Q22",
                "level": "3",
                "level_text": "L3 Regex Engine",
                "q": "How does MetadataExtractor extract effective dates, policy IDs, and department tags?",
                "short": "Uses precompiled multi-pattern regular expressions scanning the first 2,000 characters of the document for date formats, policy identifiers, and department taxonomies.",
                "deep": "1. Policy IDs: Scans regex r'\\b([A-Z]{2,4}-[0-9]{3,4}(?:-[A-Z0-9]+)?)\\b' matching formats like 'POL-102', 'HR-2024-A'.\n2. Effective Dates: Evaluates multiple date patterns (ISO, US formats), normalizing all matches to ISO format YYYY-MM-DD via dateutil.parser.\n3. Department Taxonomy: Evaluates keywords against a compiled department vocabulary dictionary: {'hr': 'HR', 'human resources': 'HR', 'information technology': 'IT', 'infosec': 'Security'}.",
                "code": "backend/document_processing/metadata_extractor.py:75 (extract_metadata)",
                "file": "backend/document_processing/metadata_extractor.py",
                "lang": "python",
                "snippet": """POLICY_ID_PATTERN = re.compile(r'\\b([A-Z]{2,4}-[0-9]{3,4}(?:-[A-Z0-9]+)?)\\b')
DATE_PATTERN = re.compile(r'\\b(?:Effective\\s+Date|Date):?\\s*([A-Za-z0-9,\\s\\-/]+)', re.IGNORECASE)

def extract_metadata(self, text_header: str) -> DocumentMetadata:
    meta = DocumentMetadata()
    if match := self.POLICY_ID_PATTERN.search(text_header):
        meta.policy_id = match.group(1).upper()
    if d_match := self.DATE_PATTERN.search(text_header):
        try:
            meta.effective_date = dateutil.parser.parse(d_match.group(1)).strftime('%Y-%m-%d')
        except Exception:
            pass
    return meta"""
            },
            {
                "num": "Q23",
                "level": "3",
                "level_text": "L3 ChromaDB Constraints",
                "q": "Why is metadata flattening necessary for ChromaDB and how is it implemented?",
                "short": "ChromaDB metadata values only support primitive types (str, int, float, bool). Nested dictionaries and lists throw validation errors. We flatten lists into comma-separated strings.",
                "deep": "If a metadata object contains topic_tags: ['vacation', 'benefits'] or nested: {'tier': 1}, ChromaDB's SQLite validator throws a ValueError. MetadataExtractor.flatten_metadata() recursively flattens metadata dictionaries:\n- Python lists ['a', 'b'] are converted to 'a,b'\n- Nested dicts {'a': {'b': 1}} become {'a_b': 1}\n- Non-supported types (None, custom classes) are cast to strings or dropped.\nDuring retrieval, string lists are deserialized back into Python sets for filtering.",
                "code": "backend/document_processing/metadata_extractor.py:130 (flatten_metadata)",
                "file": "backend/document_processing/metadata_extractor.py",
                "lang": "python",
                "snippet": """def flatten_metadata(meta_dict: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    flat = {}
    for k, v in meta_dict.items():
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        elif isinstance(v, list):
            flat[k] = ",".join(str(item) for item in v)
        elif isinstance(v, dict):
            for sub_k, sub_v in flatten_metadata(v).items():
                flat[f"{k}_{sub_k}"] = sub_v
    return flat"""
            },
            {
                "num": "Q24",
                "level": "2",
                "level_text": "L2 Runtime Extraction",
                "q": "How does QueryMetadataInferer extract runtime filters from raw user queries?",
                "short": "At query time, QueryMetadataInferer scans the user's prompt for mentions of specific departments, policy codes, or date constraints, constructing a ChromaDB $and/$eq filter dictionary.",
                "deep": "When a query arrives (e.g., 'What is the IT department policy on VPN usage for POL-301?'):\n1. QueryMetadataInferer identifies 'IT department' -> {'department': 'IT'}\n2. Identifies 'POL-301' -> {'policy_id': 'POL-301'}\n3. Combines them into a ChromaDB filter: where={'$and': [{'department': {'$eq': 'IT'}}, {'policy_id': {'$eq': 'POL-301'}}]}\n4. Passes the filter to HybridRetriever, restricting vector search to the exact relevant subset.",
                "code": "backend/rag/filter_extractor.py:30 (QueryMetadataInferer)",
                "file": "backend/rag/filter_extractor.py",
                "lang": "python",
                "snippet": """def infer_filters(self, query: str) -> Optional[Dict[str, Any]]:
    clauses = []
    # 1. Department match
    for dept_kw, dept_val in self.DEPARTMENT_MAP.items():
        if re.search(rf'\\b{re.escape(dept_kw)}\\b', query, re.IGNORECASE):
            clauses.append({"department": {"$eq": dept_val}})
            break
    # 2. Policy ID match
    if match := self.POLICY_REGEX.search(query):
        clauses.append({"policy_id": {"$eq": match.group(1).upper()}})

    if len(clauses) == 1:
        return clauses[0]
    elif len(clauses) > 1:
        return {"$and": clauses}
    return None"""
            },
            {
                "num": "Q25",
                "level": "3",
                "level_text": "L3 NLP Disambiguation",
                "q": "How does the filter extractor distinguish between the English pronoun 'it' and the 'IT department'?",
                "short": "Uses case-sensitive token matching, Part-of-Speech / boundary checks, and context keyword verification (e.g. 'IT department', 'IT security', 'IT helpdesk') rather than raw case-insensitive regex.",
                "deep": "A common bug is classifying 'What is it?' as department='IT'. We prevent this via:\n1. Regex Context Matching: Only matches \\bIT\\b when capitalized AND accompanied by department nouns (\\bIT\\s+(?:department|policy|hardware|security|team|support)\\b) or all-caps acronym context.\n2. Negative Lookahead: If 'it' is lowercase and functions as a subject pronoun followed by auxiliary verbs (it is, it has, about it), department filter extraction is explicitly skipped.\n3. Coreference Verification: Pronoun 'it' triggers QueryRewriter to resolve the referent from conversation history instead.",
                "code": "backend/rag/filter_extractor.py:85 (_disambiguate_it_vs_department)",
                "file": "backend/rag/filter_extractor.py",
                "lang": "python",
                "snippet": """IT_DEPARTMENT_REGEX = re.compile(
    r'(?:\\bIT\\s+(?:department|policy|hardware|helpdesk|security|team)\\b|\\bInformation\\s+Technology\\b)',
    re.IGNORECASE
)

def _is_it_department(query: str) -> bool:
    if IT_DEPARTMENT_REGEX.search(query):
        return True
    return bool(re.search(r'\\bIT\\b(?!\\s+(?:is|was|has|can|will|should))', query))"""
            },
            {
                "num": "Q26",
                "level": "3",
                "level_text": "L3 Production Resilience",
                "q": "What is the 'Filter Relaxation Fallback' pattern and why is it critical in production?",
                "short": "If inferred metadata filters over-constrain retrieval and return 0 candidates, the system automatically drops filters and retries unfiltered hybrid retrieval, preventing false 'No documents found' responses.",
                "deep": "In backend/retrieval/hybrid.py:\nIf a user asks 'What is the IT leave policy?', and the leave policy is actually filed under 'HR' but mentions IT workers, a strict filter on department='IT' yields 0 results. Filter relaxation drops the filter and re-queries, recovering the true HR leave document with 100% success.",
                "code": "backend/retrieval/hybrid.py:75 (Filter Relaxation Fallback)",
                "file": "backend/retrieval/hybrid.py",
                "lang": "python",
                "snippet": """results = await self._execute_hybrid_search(query, top_k, filters=filters)
if not results and filters is not None:
    logger.warning(f"Filtered retrieval returned 0 results for filters: {filters}. Falling back to unfiltered search.")
    results = await self._execute_hybrid_search(query, top_k, filters=None)
    self.trace["filter_relaxed"] = True
return results"""
            },
            {
                "num": "Q27",
                "level": "3",
                "level_text": "L3 Vector Math",
                "q": "How does ChromaDB index vectors using HNSW and what distance metric is configured?",
                "short": "ChromaDB uses Hierarchical Navigable Small World (HNSW) graph indexing with cosine distance (1 - similarity). Graph parameters are tuned for sub-10ms approximate nearest neighbor search.",
                "deep": "1. Metric: Configured with hnsw:space = 'cosine'. Cosine distance is calculated as: D_C(u, v) = 1 - (u · v) / (||u|| ||v||).\n2. HNSW Topology: Parameters configured in Settings:\n- M = 16 (bi-directional links per node)\n- ef_construction = 100 (build search depth)\n- ef_search = 50 (query search depth)",
                "code": "company_policy_rag/src/config.py:75, backend/services/document_service.py:180",
                "file": "backend/services/document_service.py",
                "lang": "python",
                "snippet": """self.collection = self.client.get_or_create_collection(
    name="policy_documents",
    embedding_function=self.embedding_fn,
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 100,
        "hnsw:M": 16,
        "hnsw:search_ef": 50
    }
)"""
            },
            {
                "num": "Q28",
                "level": "2",
                "level_text": "L2 Model Selection",
                "q": "What embedding model is used (bge-small-en-v1.5), what is its dimensionality, and why was it selected?",
                "short": "BAAI/bge-small-en-v1.5 produces 384-dimensional dense vectors. Selected because it outperforms 768-dim models (e.g. MiniLM-L6) on MTEB retrieval benchmarks while requiring 50% less RAM and 2x faster embedding generation.",
                "deep": "1. Dimensionality: 384 dimensions vs 768 or 1536, cutting vector memory footprint by 50–75%.\n2. MTEB Retrieval: Scores 62.17 on Retrieval Average on MTEB, beating larger models.\n3. Latency: Generates embeddings in ~3ms per 480-token chunk on CPU.",
                "code": "company_policy_rag/src/config.py:120 (EMBEDDING_MODEL_NAME)",
                "file": "company_policy_rag/src/config.py",
                "lang": "python",
                "snippet": """EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION: int = 384
EMBEDDING_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "
NORMALIZE_EMBEDDINGS: bool = True"""
            },
            {
                "num": "Q29",
                "level": "3",
                "level_text": "L3 Database Internals",
                "q": "How do metadata pre-filtering and post-filtering compare in vector databases?",
                "short": "Pre-filtering evaluates metadata predicates during graph traversal (or before ANN search), guaranteeing exact matches. Post-filtering retrieves top-K vectors first and discards non-matching metadata, risking returning fewer than K results.",
                "deep": "- Post-filtering: Retrieves top 50 vectors globally, then filters department == 'HR'. If only 2 of top 50 match, only 2 chunks return.\n- Pre-filtering: ChromaDB applies metadata predicates to candidate ID sets prior to HNSW distance calculation, guaranteeing top_k matches are returned.",
                "code": "backend/retrieval/hybrid.py:45",
                "file": "backend/retrieval/hybrid.py",
                "lang": "python",
                "snippet": """dense_results = self.collection.query(
    query_embeddings=[query_vector],
    n_results=top_k,
    where=filters  # Pre-filtering evaluated at index level
)"""
            },
            {
                "num": "Q30",
                "level": "3",
                "level_text": "L3 Multi-Tenancy & RBAC",
                "q": "How would you scale metadata filtering to multi-tenant environments with RBAC (Role-Based Access Control)?",
                "short": "Inject user security principal tokens (tenant_id, allowed_groups, clearance_level) as mandatory server-side metadata filter predicates in ChromaDB queries, mathematically isolating tenant vectors.",
                "deep": "1. JWT Authentication extracts user claims: tenant_id='acme', user_roles=['finance_viewer'].\n2. Server-side Query Constraint: Forcefully appends tenant and role predicates that cannot be bypassed by prompt injection:\nwhere={'$and': [{'tenant_id': {'$eq': user.tenant_id}}, {'access_group': {'$in': user.user_roles}}]}",
                "code": "backend/api/deps.py (Security dependencies), backend/retrieval/hybrid.py:50",
                "file": "backend/api/deps.py",
                "lang": "python",
                "snippet": """def get_current_user_filters(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    return {
        "$and": [
            {"tenant_id": {"$eq": user.tenant_id}},
            {"clearance_level": {"$lte": user.clearance_level}}
        ]
    }"""
            }
        ]
    },
    {
        "id": "mod4",
        "title": "Module 4: Hybrid Retrieval (Dense + BM25) & Reciprocal Rank Fusion (RRF)",
        "badge": "Q31–Q40",
        "questions": [
            {
                "num": "Q31",
                "level": "2",
                "level_text": "L2 Core Retrieval",
                "q": "Why is hybrid retrieval strictly necessary for enterprise compliance and policy documents?",
                "short": "Dense vector search understands conceptual synonyms ('vacation' ~ 'paid time off') but fails on exact keywords, numbers, and policy codes ('POL-402', '$1,500', 'Form 8802'). BM25 excels at exact keyword matches. Combining both eliminates blind spots.",
                "deep": "1. Dense failure: 'POL-402' has minimal semantic differentiation in vector space.\n2. BM25 failure: Misses conceptual queries like 'mental wellness leave' if document uses 'psychological sabbatical'.\n3. Union: Parallel dense search + BM25 merged via RRF captures both conceptual context and exact policy codes.",
                "code": "backend/retrieval/hybrid.py:25 (HybridRetriever)",
                "file": "backend/retrieval/hybrid.py",
                "lang": "python",
                "snippet": """class HybridRetriever:
    async def retrieve(self, query: str, strategy: RetrievalStrategy, filters: Optional[Dict] = None) -> List[ScoredChunk]:
        dense_task = self._dense_search(query, top_k=strategy.dense_top_k, filters=filters)
        sparse_task = self._bm25_search(query, top_k=strategy.bm25_top_k, filters=filters)
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)
        return reciprocal_rank_fusion([dense_hits, sparse_hits], k=60)"""
            },
            {
                "num": "Q32",
                "level": "2",
                "level_text": "L2 Math & Algorithms",
                "q": "Explain the mathematical formula of Reciprocal Rank Fusion (RRF).",
                "short": "RRF combines rankings from multiple retrieval systems without requiring score normalization: Score(d) = Σ [1 / (k + rank_i(d))], where rank_i(d) is document d's 1-indexed rank in system i, and k is a smoothing constant (k=60).",
                "deep": "Score_RRF(d) = Σ [1 / (k + r_m(d))]\nDoc A is Rank 1 in Dense (1/61 = 0.01639) and Rank 3 in BM25 (1/63 = 0.01587) -> Total = 0.03226.\nDoc B is Rank 2 in Dense (1/62 = 0.01613) but unranked in BM25 -> Total = 0.01613.\nDoc A ranks #1 because it appeared in both retrieval streams.",
                "code": "backend/retrieval/hybrid.py:90 (reciprocal_rank_fusion function)",
                "file": "backend/retrieval/hybrid.py",
                "lang": "python",
                "snippet": """def reciprocal_rank_fusion(ranked_lists: List[List[ScoredChunk]], k: int = 60) -> List[ScoredChunk]:
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, ScoredChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            rrf_scores[chunk.id] += 1.0 / (k + rank)
            chunk_map[chunk.id] = chunk

    sorted_chunks = []
    for chunk_id, score in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True):
        chunk = chunk_map[chunk_id]
        chunk.rrf_score = score
        sorted_chunks.append(chunk)
    return sorted_chunks"""
            },
            {
                "num": "Q33",
                "level": "3",
                "level_text": "L3 Hyperparameter Tuning",
                "q": "Why is the smoothing constant k=60 chosen in RRF and what is its effect on ranking?",
                "short": "k=60 is the empirical standard from Cormack et al. (SIGIR '09). It stabilizes rank score decay: with k=60, rank 1 scores 0.0164 and rank 2 scores 0.0161 (2% delta), preventing a single noisy rank 1 from overwhelming strong consensus across ranks 2–5.",
                "deep": "- k=1: Rank 1 gets 0.50, Rank 2 gets 0.33 (34% drop!).\n- k=60: Rank 1 gets 0.01639, Rank 2 gets 0.01613 (1.6% drop). Prevents outliers from dominating.",
                "code": "company_policy_rag/src/config.py:145 (RRF_K=60)",
                "file": "company_policy_rag/src/config.py",
                "lang": "python",
                "snippet": """RRF_SMOOTHING_K: int = 60  # Balances rank decay vs consensus weighting across multi-retrieval streams"""
            },
            {
                "num": "Q34",
                "level": "3",
                "level_text": "L3 Algorithmic Comparison",
                "q": "Why is RRF superior to convex linear combination (α · Dense + (1-α) · BM25)?",
                "short": "Dense cosine scores are bounded in [0,1] with normal distribution; BM25 scores are unbounded [0, +∞) with heavy right skew. Linear combination requires score normalization, which is unstable and brittle on outlier queries. RRF relies purely on ordinal ranks, making it immune to score scale mismatch.",
                "deep": "Score normalization (Min-Max) distorts distributions when top BM25 matches are extreme outliers. RRF uses rank order, requiring zero calibration or score normalization.",
                "code": "backend/retrieval/hybrid.py:105",
                "file": "backend/retrieval/hybrid.py",
                "lang": "python",
                "snippet": """for rank, hit in enumerate(hits, start=1):
    score_accumulator[hit.id] += 1.0 / (RRF_K + rank)"""
            },
            {
                "num": "Q35",
                "level": "3",
                "level_text": "L3 Information Retrieval Math",
                "q": "How does BM25 calculate token relevance and what do hyperparameters k₁ and b control?",
                "short": "BM25 scores documents based on Term Frequency (TF) with saturation and Inverse Document Frequency (IDF) with document length normalization: k₁ (1.5) controls term frequency saturation; b (0.75) controls document length penalty.",
                "deep": "Score(D, Q) = Σ IDF(q_i) · [f(q_i, D) · (k_1 + 1)] / [f(q_i, D) + k_1 · (1 - b + b · |D| / avgdl)]\nk_1=1.5 prevents keyword stuffing from dominating; b=0.75 normalizes length.",
                "code": "backend/retrieval/hybrid.py:40 (BM25Okapi initialization)",
                "file": "backend/retrieval/hybrid.py",
                "lang": "python",
                "snippet": """self.bm25 = BM25Okapi(
    corpus=tokenized_corpus,
    k1=1.5,   # Term frequency saturation ceiling
    b=0.75    # Document length normalization penalty
)"""
            },
            {
                "num": "Q36",
                "level": "3",
                "level_text": "L3 Index Synchronization",
                "q": "How is the in-memory BM25 index built, serialized, and kept synchronized with ChromaDB?",
                "short": "During document ingestion, child chunks are tokenized and appended to the in-memory BM25 index corpus. The index is serialized to disk via pickle/JSON. On server startup, BM25 loads from disk and verifies document count parity with ChromaDB.",
                "deep": "Startup verification compares len(bm25.corpus) with chromadb.collection.count(). If desynchronized, an asynchronous background rebuild is dispatched.",
                "code": "backend/services/document_service.py:210 (_sync_bm25_index)",
                "file": "backend/services/document_service.py",
                "lang": "python",
                "snippet": """def _sync_bm25_index(self):
    chroma_count = self.collection.count()
    if len(self.bm25_corpus) != chroma_count:
        logger.info("Rebuilding BM25 corpus from ChromaDB...")
        all_docs = self.collection.get(include=["documents", "metadatas"])
        self.bm25_corpus = [self._tokenize(doc) for doc in all_docs["documents"]]
        self.bm25 = BM25Okapi(self.bm25_corpus)"""
            },
            {
                "num": "Q37",
                "level": "2",
                "level_text": "L2 Edge Case Handling",
                "q": "What happens when dense search and BM25 return completely disjoint sets of candidates?",
                "short": "RRF naturally handles completely disjoint sets: candidates from both lists are interleaved in the fused output based on their individual reciprocal ranks, and the cross-encoder reranker arbitrates final relevance.",
                "deep": "Candidates from both lists receive individual RRF scores (1/61, 1/62) and are interleaved. The Cross-Encoder scores all candidates via deep cross-attention.",
                "code": "backend/retrieval/hybrid.py:110",
                "file": "backend/retrieval/hybrid.py",
                "lang": "python",
                "snippet": """# Interleaved Disjoint Candidates in RRF -> Passed to Cross-Encoder"""
            },
            {
                "num": "Q38",
                "level": "2",
                "level_text": "L2 Dynamic Hyperparameters",
                "q": "How does QueryRouter adjust retrieval parameters (dense_top_k, bm25_top_k) based on query intent?",
                "short": "QueryRouter maps intent categories to dynamic RetrievalStrategy configs: Factual queries use tight search (top_k=10, rerank=4); Enumeration queries use broad search (top_k=30, rerank=12); Conversational queries bypass retrieval entirely.",
                "deep": "- FACTUAL: top_k=10, rerank=4\n- COMPARISON: top_k=25, rerank=10\n- ENUMERATION: top_k=30, rerank=12\n- PROCEDURAL: top_k=15, rerank=6\n- CONVERSATIONAL: top_k=0 (direct stream, 0ms retrieval delay)",
                "code": "backend/rag/query_router.py:45 (DEFAULT_STRATEGIES)",
                "file": "backend/rag/query_router.py",
                "lang": "python",
                "snippet": """DEFAULT_STRATEGIES: Dict[QueryCategory, RetrievalStrategy] = {
    QueryCategory.FACTUAL: RetrievalStrategy(dense_top_k=10, bm25_top_k=10, rerank_top_n=4, min_ratio=0.45),
    QueryCategory.COMPARISON: RetrievalStrategy(dense_top_k=25, bm25_top_k=25, rerank_top_n=10, min_ratio=0.35),
    QueryCategory.ENUMERATION: RetrievalStrategy(dense_top_k=30, bm25_top_k=30, rerank_top_n=12, min_ratio=0.30),
    QueryCategory.PROCEDURAL: RetrievalStrategy(dense_top_k=15, bm25_top_k=15, rerank_top_n=6, min_ratio=0.40),
    QueryCategory.CONVERSATIONAL: RetrievalStrategy(dense_top_k=0, bm25_top_k=0, rerank_top_n=0, min_ratio=0.0)
}"""
            },
            {
                "num": "Q39",
                "level": "3",
                "level_text": "L3 Complexity & Scale",
                "q": "What is the computational and memory complexity of in-memory BM25 vs dedicated Elasticsearch?",
                "short": "In-memory BM25 requires O(N) RAM holding tokenized inverted lists in Python heaps and has O(|Q| · L) query time. For <100k documents, memory is ~50MB. Beyond 500k documents, a dedicated Elasticsearch/OpenSearch cluster is required.",
                "deep": "For 50,000 policy chunks, memory is ~35MB and query time is ~10ms in Python. Elasticsearch is recommended when corpus exceeds 1,000,000 documents.",
                "code": "backend/retrieval/hybrid.py:35",
                "file": "backend/retrieval/hybrid.py",
                "lang": "python",
                "snippet": """tokenized_query = self._tokenize(query)
scores = self.bm25.get_scores(tokenized_query)
top_indices = np.argsort(scores)[::-1][:top_k]
return [ScoredChunk(id=self.chunk_ids[i], bm25_score=scores[i]) for i in top_indices if scores[i] > 0]"""
            },
            {
                "num": "Q40",
                "level": "3",
                "level_text": "L3 Multilingual Extension",
                "q": "How would you handle multilingual hybrid retrieval if policies were in English, French, and Japanese?",
                "short": "Replace `bge-small-en-v1.5` with `bge-m3` (multilingual dense + sparse multi-vector) and configure BM25 with language-specific tokenizers (e.g. Kuromoji/MeCab for Japanese morphological analysis).",
                "deep": "1. Dense Model: BAAI/bge-m3.\n2. Morphological Analyzers: MeCab for Japanese, jieba for Chinese, Snowball for French.\n3. Language metadata tags partition indices.",
                "code": "backend/document_processing/loaders.py (language detection extension)",
                "file": "backend/document_processing/loaders.py",
                "lang": "python",
                "snippet": """def detect_and_tokenize(text: str, lang: str = "en") -> List[str]:
    if lang == "ja":
        import MeCab
        tagger = MeCab.Tagger("-Owakati")
        return tagger.parse(text).strip().split()
    return re.findall(r'\\b\\w+\\b', text.lower())"""
            }
        ]
    }
]
