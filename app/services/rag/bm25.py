"""In-memory BM25 keyword index for hybrid retrieval (portfolio item 28).

Complements the vector store with exact-term matching: embeddings generalize
across paraphrase but can miss distinctive identifiers, error codes and other
rare terms that a keyword score catches. This module is the scoring engine;
item 30 wires it into ``RAGService`` startup (rebuild from stored chunks) and
item 32 fuses its ranking with the vector ranking.

Design decisions:

* **No new dependency.** The corpus is small (tens to hundreds of chunks), so a
  hand-rolled Okapi BM25 beats pulling in ``rank_bm25`` or an IR library.
* **In-memory only, no files of its own.** The index is fully derivable from the
  chunk texts already persisted by :class:`VectorStore`, so there is nothing new
  to corrupt, version or quarantine - it is rebuilt on startup and updated in
  lockstep with vector adds/removes.
* **Standard Okapi parameters** (k1=1.5, b=0.75) and the non-negative IDF form
  ``log(1 + (N - df + 0.5)/(df + 0.5))`` so terms present in every chunk score
  zero instead of negative.
* **Deterministic.** Ties are broken by ``(doc_id, chunk_index)``, so repeated
  runs over the same corpus produce identical rankings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from app.services.rag.vector_store import RetrievedChunk, StoredChunk

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; whitespace and punctuation are separators.

    Unicode-aware (``\\w`` matches accented letters), digits kept - error codes
    like ``E1042`` are exactly the terms keyword search exists to catch.
    """
    return _TOKEN_RE.findall(text.lower())


class KeywordIndex:
    """Okapi BM25 over stored chunk texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[StoredChunk] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._df: Counter[str] = Counter()  # chunks containing the term
        self._total_tokens = 0

    @property
    def size(self) -> int:
        return len(self._chunks)

    # ---------- mutation ----------

    def add_chunks(self, chunks: Sequence[StoredChunk]) -> None:
        """Index more chunks (appends; does not deduplicate)."""
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self._chunks.append(chunk)
            self._term_freqs.append(Counter(tokens))
            self._doc_lengths.append(len(tokens))
            self._total_tokens += len(tokens)
            for term in set(tokens):
                self._df[term] += 1

    def remove_by_doc_id(self, doc_id: str) -> int:
        """Drop every chunk of one document and recompute the statistics."""
        kept = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        removed = len(self._chunks) - len(kept)
        if not removed:
            return 0
        self._rebuild([self._chunks[i] for i in kept])
        return removed

    def clear(self) -> None:
        """Remove all indexed chunks."""
        self._chunks.clear()
        self._term_freqs.clear()
        self._doc_lengths.clear()
        self._df.clear()
        self._total_tokens = 0

    # ---------- search ----------

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Rank chunks by BM25 score for the query; zero-score chunks omitted."""
        if not self._chunks or top_k <= 0:
            return []
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        n = len(self._chunks)
        avg_len = self._total_tokens / n
        scored: list[tuple[float, StoredChunk]] = []
        for chunk, tf, length in zip(
            self._chunks, self._term_freqs, self._doc_lengths, strict=True
        ):
            score = 0.0
            for term in query_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (
                    1 - self.b + self.b * (length / avg_len if avg_len else 0.0)
                )
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((score, chunk))

        # Deterministic: score desc, then stable identity of the chunk.
        scored.sort(key=lambda item: (-item[0], item[1].doc_id, item[1].chunk_index))
        return [
            RetrievedChunk(
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=score,
            )
            for score, chunk in scored[:top_k]
        ]

    # ---------- internals ----------

    def _rebuild(self, chunks: list[StoredChunk]) -> None:
        self.clear()
        self.add_chunks(chunks)


def build_keyword_index(chunks: Iterable[StoredChunk]) -> KeywordIndex:
    """One-shot construction from stored chunks (startup rebuild path)."""
    index = KeywordIndex()
    index.add_chunks(list(chunks))
    return index
