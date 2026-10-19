"""Reciprocal rank fusion for hybrid retrieval (portfolio item 29).

Fuses the vector and keyword rankings into one. RRF (Cormack et al., 2009) is
used instead of score-level combination because the two scores live on
incompatible scales - cosine similarity in [0, 1] versus unbounded BM25 sums -
and rank fusion needs no calibration or per-corpus tuning.

The function operates on opaque item ids (chunk identity strings), not chunk
objects, so it stays pure and trivially testable; the caller maps ids back to
chunks afterwards.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into a single ranking.

    Each item's score is the sum of ``1 / (k + rank)`` over every ranking that
    contains it, with 1-based ranks. Items absent from a ranking contribute
    nothing for that ranking - so an item found by both retrievers always
    outranks one found by only one at comparable depth.

    Args:
        rankings: One ranked id list per retriever (e.g. vector then keyword).
            Duplicates inside a single list count once, at the first position.
        k: Smoothing constant; 60 is the value from the original paper and the
            standard default in practice.
        top_k: Optional truncation of the fused result.

    Returns:
        ``(item_id, score)`` pairs sorted by score descending. Ties break by
        item id ascending, so the output depends only on *which* rankings were
        given - not their order (RRF is commutative over its inputs).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if top_k is not None:
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        fused = fused[:top_k]
    return fused
