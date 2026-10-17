# Vector Search Basics

Retrieval-augmented systems answer questions by finding the most similar stored text fragments, and that "most similar" is computed with vectors. This document explains the pieces behind a simple embedding-based search index.

## Embeddings

A text embedding model maps a piece of text to a fixed-length list of floating-point numbers — for example 384 or 768 values. Texts with similar meaning land close together in this vector space, so similarity between two texts reduces to a geometric question about their vectors. The same model must be used to embed both the stored documents and the query; mixing embeddings from different models is meaningless because each model defines its own coordinate system.

## Cosine similarity

The standard similarity measure is cosine similarity: the angle between two vectors, computed as the dot product of their L2-normalized versions. Values range from -1 (opposite) through 0 (unrelated) to 1 (identical direction). A practical trick used by most systems is to normalize every vector once at storage time; then plain inner products already equal cosine similarity and no normalization is needed per query.

## Flat indexes

The simplest index stores all vectors in a dense array and computes the inner product with the query against every row — an exact brute-force search, often called a flat index. FAISS provides this as `IndexFlatIP` (inner product). Exact search costs time proportional to the number of stored vectors, but for tens or hundreds of thousands of short vectors it completes in milliseconds on CPU, which is why local single-user tools rarely need approximate methods at all.

## When to move on

Approximate nearest-neighbor structures such as HNSW graphs trade a small loss in recall for large speedups at scale — millions of vectors and beyond. They add configuration (graph degree, construction ef) and memory overhead, so the rule of thumb is: stay flat until exact search measurably hurts latency, then switch.
