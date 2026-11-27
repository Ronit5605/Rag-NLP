"""
Classical NLP extractive summarization and TF-IDF retrieval utilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extractive_summarize(sentences: list[str], top_k: int = 3) -> list[dict[str, Any]]:
    """
    Select top_k representative sentences using TF-IDF centroid similarity
    (lightweight LexRank-style salience), preserving original narrative order.
    """
    valid = [(idx, s.strip()) for idx, s in enumerate(sentences) if len(s.strip()) > 10]
    if not valid:
        return []

    if len(valid) <= top_k:
        return [
            {"sentence_index": idx, "text": text, "salience_score": 1.0}
            for idx, text in valid
        ]

    texts = [text for _, text in valid]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        centroid = np.asarray(matrix.mean(axis=0))
        similarities = cosine_similarity(matrix, centroid).ravel()
    except ValueError:
        return [
            {"sentence_index": idx, "text": text, "salience_score": 1.0}
            for idx, text in valid[:top_k]
        ]

    top_positions = sorted(similarities.argsort()[::-1][:top_k].tolist())
    return [
        {
            "sentence_index": valid[pos][0],
            "text": valid[pos][1],
            "salience_score": round(float(similarities[pos]), 4),
        }
        for pos in top_positions
    ]


def retrieve_top_chunks_tfidf(
    chunks: list[str],
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rank text chunks against a query using TF-IDF cosine similarity."""
    if not chunks or not query.strip():
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        corpus = chunks + [query]
        matrix = vectorizer.fit_transform(corpus)
        chunk_vecs = matrix[:-1]
        query_vec = matrix[-1]
        sims = cosine_similarity(chunk_vecs, query_vec).ravel()
    except ValueError:
        return []

    top_indices = sims.argsort()[::-1][:top_k]
    return [
        {
            "chunk_index": int(idx),
            "score": round(float(sims[idx]), 4),
            "text": chunks[idx],
        }
        for idx in top_indices
    ]
