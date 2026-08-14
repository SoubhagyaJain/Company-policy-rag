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
        self._bm25: Any | None = None

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index over a list of document chunks."""
        self.entries = []
        self._tokenized_corpus = []

        for chunk in chunks:
            search_text = _searchable_text(chunk)
            if not search_text:
                continue
            self.entries.append(chunk)
            self._tokenized_corpus.append(tokenize(search_text))

        if self._tokenized_corpus:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore

                self._bm25 = BM25Okapi(self._tokenized_corpus)
                logger.info("BM25 index built with %d chunks", len(self.entries))
            except Exception as exc:
                logger.warning("BM25Okapi import or build failed (%s). BM25 disabled.", exc)
                self._bm25 = None
        else:
            self._bm25 = None

    def _matches_filters(self, chunk: Chunk, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        meta_dict = chunk.metadata.model_dump()
        for key, value in filters.items():
            actual = meta_dict.get(key)
            if actual is None and hasattr(chunk.metadata, key):
                actual = getattr(chunk.metadata, key)
            if actual is None and chunk.metadata.extra:
                actual = chunk.metadata.extra.get(key)
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
            if not self._matches_filters(chunk, filters):
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

        if self._tokenized_corpus:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore

                self._bm25 = BM25Okapi(self._tokenized_corpus)
                return True
            except Exception as exc:
                logger.warning("BM25Okapi build from loaded corpus failed: %s", exc)
                self._bm25 = None
                return False

        return False

    def remove_by_source_file(self, source_file: str) -> None:
        kept_entries: list[Chunk] = []
        kept_tokens: list[list[str]] = []

        for chunk, tokens in zip(self.entries, self._tokenized_corpus):
            if chunk.metadata.source_file == source_file:
                continue
            kept_entries.append(chunk)
            kept_tokens.append(tokens)

        self.entries = kept_entries
        self._tokenized_corpus = kept_tokens

        if self._tokenized_corpus:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore

                self._bm25 = BM25Okapi(self._tokenized_corpus)
            except Exception:
                self._bm25 = None
        else:
            self._bm25 = None

    def remove_by_document_id(self, document_id: str) -> None:
        kept_entries: list[Chunk] = []
        kept_tokens: list[list[str]] = []

        for chunk, tokens in zip(self.entries, self._tokenized_corpus):
            if chunk.metadata.document_id == document_id:
                continue
            kept_entries.append(chunk)
            kept_tokens.append(tokens)

        self.entries = kept_entries
        self._tokenized_corpus = kept_tokens

        if self._tokenized_corpus:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore

                self._bm25 = BM25Okapi(self._tokenized_corpus)
            except Exception:
                self._bm25 = None
        else:
            self._bm25 = None

    def clear(self) -> None:
        self.entries.clear()
        self._tokenized_corpus.clear()
        self._bm25 = None
