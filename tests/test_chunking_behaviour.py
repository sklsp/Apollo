"""Unit tests for ``app.services.rag.chunking`` (sentence-aware text chunker).

The module is pure string processing, so every test feeds a short hand-built
document and asserts the exact list of chunks returned. Expected values were
computed by tracing the algorithm by hand: sentence lengths, overlap windows
and character offsets are all pinned to literals, never re-derived from the
code under test.
"""

from __future__ import annotations

from app.services.rag.chunking import (
    _chunk_by_characters,
    _overlap_tail,
    chunk_text,
)


# --------------------------------------------------------------------------- #
# chunk_text: normalization and short-text paths
# --------------------------------------------------------------------------- #
class TestChunkTextShortPaths:
    def test_empty_string_returns_no_chunks(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_no_chunks(self):
        assert chunk_text(" \n\t  ") == []

    def test_short_text_is_single_normalized_chunk(self):
        # Newlines, tabs and repeated spaces collapse to single spaces.
        result = chunk_text("Hello\n   world!\tHow are you?")
        assert result == ["Hello world! How are you?"]

    def test_text_exactly_at_chunk_size_stays_single(self):
        text = "x" * 10
        assert chunk_text(text, chunk_size=10) == [text]

    def test_one_char_over_chunk_size_uses_character_fallback(self):
        # No sentence boundaries at all: the character fallback cuts at the
        # size limit and keeps the remainder.
        result = chunk_text("x" * 11, chunk_size=10, overlap=0)
        assert result == ["x" * 10, "x"]

    def test_newlines_and_tabs_normalize_before_sentence_split(self):
        result = chunk_text("First line.\n\nSecond line!\tThird?")
        assert result == ["First line. Second line! Third?"]


# --------------------------------------------------------------------------- #
# chunk_text: sentence path with overlap carry-over
# --------------------------------------------------------------------------- #
class TestChunkTextSentencePath:
    # Sentences used below, with exact lengths:
    #   s1 "Alpha beta gamma."      17 chars
    #   s2 "Delta epsilon zeta."    19 chars
    #   s3 "Eta theta iota kappa."  21 chars
    #   s4 "Lambda mu nu xi."       16 chars

    def test_overflow_carries_overlap_tail(self):
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa."
        result = chunk_text(text, chunk_size=40, overlap=20)
        # s1+s2 fit (37 <= 40); s3 overflows (59 > 40). The overlap window of
        # 20 keeps only the trailing sentence s2 for the next chunk.
        assert result == [
            "Alpha beta gamma. Delta epsilon zeta.",
            "Delta epsilon zeta. Eta theta iota kappa.",
        ]

    def test_zero_overlap_carries_nothing(self):
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa."
        result = chunk_text(text, chunk_size=40, overlap=0)
        assert result == [
            "Alpha beta gamma. Delta epsilon zeta.",
            "Eta theta iota kappa.",
        ]

    def test_large_overlap_keeps_every_previous_sentence(self):
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa."
        result = chunk_text(text, chunk_size=40, overlap=100)
        # The whole previous chunk fits inside the 100-char window, so the
        # second chunk repeats it in full (and may exceed chunk_size).
        assert result == [
            "Alpha beta gamma. Delta epsilon zeta.",
            "Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa.",
        ]

    def test_multiple_flushes_produce_three_chunks(self):
        text = (
            "Alpha beta gamma. Delta epsilon zeta. "
            "Eta theta iota kappa. Lambda mu nu xi."
        )
        result = chunk_text(text, chunk_size=40, overlap=20)
        # Each flush keeps only the last sentence of the previous chunk:
        # s2 (19 chars) fits the 20-char window alone; s3 (21) does not share
        # it with anything.
        assert result == [
            "Alpha beta gamma. Delta epsilon zeta.",
            "Delta epsilon zeta. Eta theta iota kappa.",
            "Eta theta iota kappa. Lambda mu nu xi.",
        ]


# --------------------------------------------------------------------------- #
# chunk_text: entry into the character fallback (no usable boundaries)
# --------------------------------------------------------------------------- #
class TestChunkTextCharacterFallbackEntry:
    def test_word_text_without_boundaries_splits_on_spaces(self):
        result = chunk_text("aaaa bbbb cccc dddd", chunk_size=10, overlap=0)
        assert result == ["aaaa bbbb", "cccc dddd"]

    def test_period_not_followed_by_space_is_not_a_boundary(self):
        # "abc.def" has no ".<space>" pair, so the whole text is one sentence
        # and falls through to character chunking.
        result = chunk_text("abc.def ghi", chunk_size=5, overlap=0)
        assert result == ["abc.d", "ef", "ghi"]


# --------------------------------------------------------------------------- #
# _overlap_tail: direct behaviour of the carry-over window
# --------------------------------------------------------------------------- #
class TestOverlapTail:
    def test_empty_parts_returns_empty_list(self):
        assert _overlap_tail([], 50) == []

    def test_zero_overlap_returns_empty_list(self):
        assert _overlap_tail(["one", "two"], 0) == []

    def test_negative_overlap_returns_empty_list(self):
        assert _overlap_tail(["one", "two"], -5) == []

    def test_exact_fit_keeps_all_parts(self):
        # "bb" (2) + space (1) + "aa" (2) = 5, exactly the window: kept.
        assert _overlap_tail(["aa", "bb"], 5) == ["aa", "bb"]

    def test_first_trailing_part_always_kept_even_if_too_big(self):
        # The last sentence is always carried over, even when it alone exceeds
        # the overlap window; only earlier sentences are dropped.
        assert _overlap_tail(["aaaaa", "bbbb"], 3) == ["bbbb"]


# --------------------------------------------------------------------------- #
# _chunk_by_characters: direct behaviour of the fallback chunker
# --------------------------------------------------------------------------- #
class TestChunkByCharacters:
    def test_no_overlap_walks_word_by_word(self):
        result = _chunk_by_characters("aaaa bbbb cccc dddd", 10, 0)
        assert result == ["aaaa bbbb", "cccc dddd"]

    def test_window_without_space_is_cut_at_size(self):
        # The first window ("abcdef") contains no space, so it is cut at the
        # raw size limit mid-word; later windows snap back to a space.
        result = _chunk_by_characters("abcdefghij klmn", 6, 0)
        assert result == ["abcdef", "ghij", "klmn"]

    def test_overlap_larger_than_chunk_advances_one_char(self):
        # With overlap (5) bigger than chunk_size (4), the start pointer can
        # only advance by one character per step, re-reading partial words.
        result = _chunk_by_characters("aaaa bbbb", 4, 5)
        assert result == ["aaaa", "aaa", "aa", "a", "bbb", "bbbb"]

    def test_whitespace_only_window_is_dropped(self):
        # The window at start=1 is a single space: it strips to "" and must be
        # skipped rather than emitted.
        result = _chunk_by_characters("a   b", 2, 0)
        assert result == ["a", "b"]

    def test_text_shorter_than_chunk_returns_whole_text(self):
        assert _chunk_by_characters("hello world", 100, 10) == ["hello world"]
