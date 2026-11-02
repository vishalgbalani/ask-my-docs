"""OpenAI embeddings, cosine similarity, and retrieval."""

import numpy as np
from typing import List
from openai import OpenAI

from .chunker import Chunk

EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings(texts: List[str], client: OpenAI) -> List[List[float]]:
    """Get embeddings for a list of texts in a single batch call."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def retrieve_chunks(
    query_embedding: List[float],
    chunk_embeddings: List[List[float]],
    chunks: List[Chunk],
    top_k: int = 5
) -> List[tuple[Chunk, float]]:
    """Retrieve top-k most similar chunks to the query."""
    query_vec = np.array(query_embedding)

    similarities = []
    for i, emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_vec, np.array(emb))
        similarities.append((i, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Take top-k
    top_results = similarities[:top_k]

    # Sort by chunk position in document (preserve reading order)
    top_results.sort(key=lambda x: chunks[x[0]].chunk_id)

    return [(chunks[idx], score) for idx, score in top_results]


def get_confidence(top_score: float) -> str:
    """Determine confidence level from top similarity score."""
    if top_score >= 0.45:
        return "High"
    elif top_score >= 0.30:
        return "Medium"
    else:
        return "Low"
