from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from backend.models.chunk import Chunk
from backend.models.rag import ScoredChunk
from backend.utils.logging import logger

_token_re = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _token_re.findall(text.lower())


def _searchable_text(chunk: Chunk) -> str:
    meta = chunk.metadata
    parts = [
        chunk.text or "",
        str(meta.section_path or ""),
        str(meta.section_title or ""),
        str(meta.section_number or ""),
        str(meta.source_file or ""),
        str(meta.category or ""),
    ]
    return " ".join(p for p in parts if p).strip()


class BM25SearchIndex:
    """
    Okapi BM25 lexical index supporting tokenization, disk persistence,
    metadata filtering, and incremental updates.
    """

    def __init__(self, storage_dir: str = "app/storage/bm25") -> None:
        self.storage_dir = Path(storage_dir)
        self.entries: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        # Filter views (chunk.metadata.model_dump) precomputed once per chunk so
        # search() does not serialize every chunk's metadata on every query.
        self._meta_dicts: list[dict[str, Any]] = []
        self._bm25: Any | None = None

    @staticmethod
    def _filter_view(chunk: Chunk) -> dict[str, Any]:
        """Serialize a chunk's metadata once for fast repeated filter matching."""
        return chunk.metadata.model_dump()

    def _rebuild_scorer(self) -> None:
        """(Re)build the BM25Okapi scorer from the current tokenized corpus."""
        if not self._tokenized_corpus:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except Exception as exc:
            logger.warning("BM25Okapi import or build failed (%s). BM25 disabled.", exc)
            self._bm25 = None

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index over a list of document chunks."""
        self.entries = []
        self._tokenized_corpus = []
        self._meta_dicts = []

        for chunk in chunks:
            search_text = _searchable_text(chunk)
            if not search_text:
                continue
            self.entries.append(chunk)
            self._tokenized_corpus.append(tokenize(search_text))
            self._meta_dicts.append(self._filter_view(chunk))

        self._rebuild_scorer()
        if self._bm25 is not None:
            logger.info("BM25 index built with %d chunks", len(self.entries))

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Incrementally add chunks: extend the corpus and rebuild only the
        scorer, reusing existing tokenization instead of re-tokenizing the whole
        corpus. Returns the number of chunks added."""
        added = 0
        for chunk in chunks:
            search_text = _searchable_text(chunk)
            if not search_text:
                continue
            self.entries.append(chunk)
            self._tokenized_corpus.append(tokenize(search_text))
            self._meta_dicts.append(self._filter_view(chunk))
            added += 1
        if added:
            self._rebuild_scorer()
            logger.info("BM25 index extended by %d chunks (now %d)", added, len(self.entries))
        return added

    def _matches_filters(
        self,
        meta_dict: dict[str, Any],
        metadata: Any,
        filters: dict[str, Any] | None,
    ) -> bool:
        if not filters:
            return True
        extra = meta_dict.get("extra") or (getattr(metadata, "extra", None) or {})
        for key, value in filters.items():
            actual = meta_dict.get(key)
            # Precomputed model_dump covers every metadata field; the getattr
            # fallback (cheap, no serialization) only catches computed/non-field
            # attributes, and extra covers keys nested in the extra dict.
            if actual is None and metadata is not None and hasattr(metadata, key):
                actual = getattr(metadata, key)
            if actual is None:
                actual = extra.get(key)
            if actual is None:
                return False
            if isinstance(value, list):
                if isinstance(actual, list):
                    if not any(v in actual for v in value):
                        return False
                else:
                    if actual not in value:
                        return False
            else:
                if isinstance(actual, list):
                    if value not in actual:
                        return False
                elif actual != value:
                    return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 25,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Search BM25 index and return ScoredChunk list sorted by BM25 relevance score."""
        if not self._bm25 or not self.entries:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        try:
            scores = self._bm25.get_scores(tokens)
        except Exception as exc:
            logger.warning("BM25 scoring failed: %s", exc)
            return []

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: list[ScoredChunk] = []
        for idx in ranked_indices:
            chunk = self.entries[idx]
            if not self._matches_filters(self._meta_dicts[idx], chunk.metadata, filters):
                continue
            bm25_score = float(scores[idx])
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=bm25_score,
                    sparse_score=bm25_score,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def save(self, storage_dir: str | None = None) -> None:
        target_dir = Path(storage_dir) if storage_dir else self.storage_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        corpus_file = target_dir / "corpus.json"
        index_file = target_dir / "index.pkl"

        corpus_data = [chunk.model_dump() for chunk in self.entries]
        corpus_file.write_text(json.dumps(corpus_data, indent=2), encoding="utf-8")

        with open(index_file, "wb") as f:
            pickle.dump(
                {
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )
        logger.info("Saved BM25 index (%d docs) to %s", len(self.entries), target_dir)

    def load(self, storage_dir: str | None = None) -> bool:
        target_dir = Path(storage_dir) if storage_dir else self.storage_dir
        corpus_file = target_dir / "corpus.json"
        index_file = target_dir / "index.pkl"

        if not corpus_file.is_file():
            return False

        try:
            corpus_data = json.loads(corpus_file.read_text(encoding="utf-8"))
            self.entries = [Chunk(**item) for item in corpus_data]
        except Exception as exc:
            logger.warning("Failed to load BM25 corpus from %s: %s", corpus_file, exc)
            return False

        if index_file.is_file():
            try:
                with open(index_file, "rb") as f:
                    data = pickle.load(f)
                self._tokenized_corpus = data.get("tokenized_corpus", [])
            except Exception as exc:
                logger.warning("BM25 pickle load failed (%s) — re-tokenizing corpus", exc)
                self._tokenized_corpus = [tokenize(_searchable_text(c)) for c in self.entries]
        else:
            self._tokenized_corpus = [tokenize(_searchable_text(c)) for c in self.entries]

        self._meta_dicts = [self._filter_view(c) for c in self.entries]
        self._rebuild_scorer()
        return self._bm25 is not None

    def _remove_where(self, predicate) -> None:
        """Drop entries matching ``predicate`` and rebuild only the scorer,
        keeping tokenization and precomputed filter views for the rest."""
        kept_entries: list[Chunk] = []
        kept_tokens: list[list[str]] = []
        kept_meta: list[dict[str, Any]] = []
        for chunk, tokens, meta in zip(self.entries, self._tokenized_corpus, self._meta_dicts):
            if predicate(chunk):
                continue
            kept_entries.append(chunk)
            kept_tokens.append(tokens)
            kept_meta.append(meta)

        self.entries = kept_entries
        self._tokenized_corpus = kept_tokens
        self._meta_dicts = kept_meta
        self._rebuild_scorer()

    def remove_by_source_file(self, source_file: str) -> None:
        self._remove_where(lambda c: c.metadata.source_file == source_file)

    def remove_by_document_id(self, document_id: str) -> None:
        self._remove_where(lambda c: c.metadata.document_id == document_id)

    def clear(self) -> None:
        self.entries.clear()
        self._tokenized_corpus.clear()
        self._meta_dicts.clear()
        self._bm25 = None
