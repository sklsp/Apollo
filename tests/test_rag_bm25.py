"""BM25 keyword index tests (portfolio item 28).

The index is pure Python with no external state, so these tests exercise the
scoring math directly against hand-computed expectations.
"""

from __future__ import annotations

from app.services.rag.bm25 import KeywordIndex, build_keyword_index, tokenize
from app.services.rag.vector_store import StoredChunk


def _chunk(doc_id: str, text: str, chunk_index: int = 0) -> StoredChunk:
    return StoredChunk(
        doc_id=doc_id,
        filename=f"{doc_id}.md",
        chunk_index=chunk_index,
        text=text,
    )


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Hello, world! The E1042 error.") == [
            "hello",
            "world",
            "the",
            "e1042",
            "error",
        ]

    def test_keeps_digits_and_underscore_words_together(self):
        assert tokenize("v2.3 and snake_case") == ["v2", "3", "and", "snake_case"]

    def test_unicode_words_survive(self):
        assert tokenize("Café naïve") == ["café", "naïve"]

    def test_empty_and_punctuation_only(self):
        assert tokenize("") == []
        assert tokenize("!!! ??? ...") == []


class TestAddAndSearch:
    def test_search_returns_retrieved_chunks_with_positive_scores(self):
        index = KeywordIndex()
        index.add_chunks([_chunk("a", "the quick brown fox jumps"), _chunk("b", "lazy dogs sleep")])

        results = index.search("quick fox", top_k=5)

        assert len(results) == 1
        result = results[0]
        assert result.doc_id == "a"
        assert result.filename == "a.md"
        assert result.chunk_index == 0
        assert result.text == "the quick brown fox jumps"
        assert result.score > 0

    def test_exact_phrase_beats_partial_match(self):
        index = KeywordIndex()
        index.add_chunks([
            _chunk("partial", "quantum entanglement in photons"),
            _chunk("full", "quantum entanglement of photon pairs over fiber"),
        ])

        results = index.search("quantum entanglement photon")

        assert [r.doc_id for r in results] == ["full", "partial"]

    def test_rare_term_outweighs_common_term(self):
        # "system" appears everywhere (low IDF); "kubernetes" only once (high IDF).
        index = KeywordIndex()
        docs = [f"the system runs on {word}" for word in ("linux", "windows", "macos")]
        for i, text in enumerate(docs):
            index.add_chunks([_chunk(f"d{i}", text)])
        index.add_chunks([_chunk("k8s", "deploying kubernetes manifests")])

        results = index.search("system kubernetes", top_k=1)

        assert results[0].doc_id == "k8s"

    def test_top_k_limits_results(self):
        index = KeywordIndex()
        for i in range(5):
            index.add_chunks([_chunk(f"d{i}", f"common term number {i}")])

        assert len(index.search("common", top_k=3)) == 3
        assert len(index.search("common", top_k=100)) == 5

    def test_no_match_returns_empty(self):
        index = KeywordIndex()
        index.add_chunks([_chunk("a", "nothing to see here")])

        assert index.search("zebra quantum") == []

    def test_empty_or_punctuation_query_returns_empty(self):
        index = KeywordIndex()
        index.add_chunks([_chunk("a", "some text")])

        assert index.search("") == []
        assert index.search("!!! ???") == []

    def test_empty_index_returns_empty(self):
        assert KeywordIndex().search("anything") == []

    def test_repeated_query_terms_are_not_double_counted(self):
        once = KeywordIndex()
        twice = KeywordIndex()
        for idx in (once, twice):
            idx.add_chunks([_chunk("a", "the cat sat"), _chunk("b", "cat")])

        # Deduped query terms: both indexes must rank identically.
        assert [r.doc_id for r in once.search("cat")] == [
            r.doc_id for r in twice.search("cat cat cat")
        ]


class TestMutation:
    def test_remove_by_doc_id_drops_only_that_document(self):
        index = KeywordIndex()
        index.add_chunks([
            _chunk("a", "alpha beta"),
            _chunk("a", "gamma delta", 1),
            _chunk("b", "beta omega"),
        ])

        removed = index.remove_by_doc_id("a")

        assert removed == 2
        assert index.size == 1
        assert index.search("alpha gamma") == []
        assert [r.doc_id for r in index.search("beta")] == ["b"]

    def test_remove_unknown_doc_is_a_noop(self):
        index = KeywordIndex()
        index.add_chunks([_chunk("a", "hello world")])

        assert index.remove_by_doc_id("zzz") == 0
        assert index.size == 1

    def test_clear_empties_the_index(self):
        index = KeywordIndex()
        index.add_chunks([_chunk("a", "hello world")])
        index.clear()

        assert index.size == 0
        assert index.search("hello") == []


class TestDeterminismAndBuild:
    def test_ties_break_by_doc_id_then_chunk_index(self):
        index = KeywordIndex()
        # Identical texts from different docs: order must be stable.
        index.add_chunks([_chunk("z", "same text"), _chunk("m", "same text"), _chunk("a", "same text")])

        results = index.search("same text", top_k=10)

        assert [r.doc_id for r in results] == ["a", "m", "z"]
        # Scores are equal - the ranking is purely the tie-break.
        assert len({round(r.score, 9) for r in results}) == 1

    def test_build_keyword_index_matches_incremental_adds(self):
        chunks = [_chunk("a", "one two three"), _chunk("b", "two four six")]

        built = build_keyword_index(chunks)
        incremental = KeywordIndex()
        incremental.add_chunks([chunks[0]])
        incremental.add_chunks(chunks[1:])

        assert built.size == incremental.size == 2
        assert [r.doc_id for r in built.search("two", top_k=5)] == [
            r.doc_id for r in incremental.search("two", top_k=5)
        ]


class TestEdgeCases:
    def test_empty_chunk_text_is_harmless(self):
        index = KeywordIndex()
        index.add_chunks([_chunk("empty", ""), _chunk("real", "hello there")])

        assert [r.doc_id for r in index.search("hello")] == ["real"]

    def test_all_empty_corpus_search_is_safe(self):
        index = KeywordIndex()
        index.add_chunks([_chunk("a", ""), _chunk("b", "   ")])

        assert index.search("anything") == []

    def test_non_positive_top_k_returns_empty(self):
        index = KeywordIndex()
        index.add_chunks([_chunk("a", "hello world")])

        assert index.search("hello", top_k=0) == []

