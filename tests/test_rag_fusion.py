"""Reciprocal rank fusion tests (portfolio item 29).

RRF scores are hand-computed in these tests against k=60, so a regression in
the formula or the tie-break is caught directly.
"""

from __future__ import annotations

import pytest

from app.services.rag.fusion import reciprocal_rank_fuse


class TestBasics:
    def test_single_ranking_preserves_order(self):
        fused = reciprocal_rank_fuse([["a", "b", "c"]])

        assert [item for item, _ in fused] == ["a", "b", "c"]
        # 1-based ranks with k=60.
        assert fused[0][1] == pytest.approx(1 / 61)
        assert fused[1][1] == pytest.approx(1 / 62)
        assert fused[2][1] == pytest.approx(1 / 63)

    def test_item_in_both_rankings_outranks_single_ranking_items(self):
        # "x" appears in both lists (two contributions); the others in one.
        fused = reciprocal_rank_fuse([["x", "a"], ["y", "x"]])
        scores = dict(fused)

        assert scores["x"] == pytest.approx(1 / 61 + 1 / 62)
        assert fused[0][0] == "x"

    def test_hand_computed_two_ranking_fusion(self):
        # a: rank 1 in list 1, rank 3 in list 2 -> 1/61 + 1/63 = 0.032266...
        # b: rank 2 in both               -> 1/62 + 1/62 = 0.032258...
        # d: rank 1 in list 2             -> 1/61; c: rank 3 in list 1 -> 1/63
        fused = reciprocal_rank_fuse([["a", "b", "c"], ["d", "b", "a"]])
        scores = dict(fused)

        assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
        assert scores["b"] == pytest.approx(2 / 62)
        # a's rank-1 hit outweighs b's two rank-2 hits by a hair.
        assert [item for item, _ in fused] == ["a", "b", "d", "c"]

    def test_items_absent_from_a_ranking_get_no_contribution(self):
        fused = reciprocal_rank_fuse([["a"], []])

        assert dict(fused) == {"a": pytest.approx(1 / 61)}


class TestEdgeCases:
    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fuse([]) == []
        assert reciprocal_rank_fuse([[], []]) == []

    def test_duplicates_in_one_ranking_count_once(self):
        fused = reciprocal_rank_fuse([["a", "a", "b"]])
        scores = dict(fused)

        # Second "a" at rank 2 must not add a contribution.
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 63)

    def test_top_k_truncates(self):
        fused = reciprocal_rank_fuse([["a", "b", "c", "d"]], top_k=2)

        assert [item for item, _ in fused] == ["a", "b"]

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            reciprocal_rank_fuse([["a"]], top_k=-1)

    def test_non_positive_k_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            reciprocal_rank_fuse([["a"]], k=0)


class TestDeterminism:
    def test_ties_break_by_item_id_ascending(self):
        # z and a both score 1/61 (rank 1 of their list); m and q both 1/62.
        fused = reciprocal_rank_fuse([["z", "m"], ["a", "q"]])

        assert [item for item, _ in fused] == ["a", "z", "m", "q"]

    def test_independent_of_ranking_order(self):
        vector = ["vec1", "vec2", "vec3"]
        keyword = ["kw1", "vec1"]

        assert reciprocal_rank_fuse([vector, keyword]) == reciprocal_rank_fuse(
            [keyword, vector]
        )

    def test_custom_k_changes_scores_but_not_typical_order(self):
        base = reciprocal_rank_fuse([["a", "b"], ["b", "c"]], k=60)
        tight = reciprocal_rank_fuse([["a", "b"], ["b", "c"]], k=1)

        assert [item for item, _ in base] == [item for item, _ in tight]
        # With k=1 the rank-2 contribution (1/3) is much closer to rank 1 (1/2).
        scores_tight = dict(tight)
        assert scores_tight["b"] == pytest.approx(1 / 2 + 1 / 3)
