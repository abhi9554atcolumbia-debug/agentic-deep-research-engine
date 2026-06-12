"""
retriever.py
Layer 7 (Retriever).

A simple in-memory embedding-based retriever:
  - Embeds all chunks once using a local sentence-transformers model
    (no API key, no rate limits).
  - For a given sub-question, retrieves the top-k most similar chunks
    via cosine similarity.

This is intentionally simple (no FAISS/Chroma) so it's easy to run
and easy to explain in the demo. Swapping in a real vector DB later
is a drop-in upgrade if you have time.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TOP_K_RETRIEVAL

TRUSTED_DOMAINS = {
    "eia.gov": 0.05,
    "imf.org": 0.05,
    "worldbank.org": 0.05,
    "opec.org": 0.05,
    "iea.org": 0.05,
}
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts):
    """Embed a list of strings, returns an (N, D) numpy array."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def cosine_sim(query_vec, doc_vecs):
    """
    query_vec: (D,) numpy array (already normalized)
    doc_vecs: (N, D) numpy array (already normalized)
    Returns (N,) similarity scores in [-1, 1].
    """
    return doc_vecs @ query_vec


class ChunkIndex:
    """
    Simple in-memory index over all collected chunks.
    Each chunk dict gains an "embedding" field after build().
    """

    def __init__(self, chunks: list):
        self.chunks = chunks
        self._embeddings = None

    def build(self):
        if not self.chunks:
            self._embeddings = np.zeros((0, 384))  # MiniLM dim
            return
        texts = [c["text"] for c in self.chunks]
        self._embeddings = embed_texts(texts)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL):
        """
        Returns the top_k chunks most relevant to `query`,
        while limiting how many chunks can come from the same source.
        """
        if len(self.chunks) == 0:
            return []

        query_vec = embed_texts([query])[0]
        scores = cosine_sim(query_vec, self._embeddings)

        adjusted_scores = scores.copy()

        for i, chunk in enumerate(self.chunks):
            url = chunk.get("url", "")

            for domain, bonus in TRUSTED_DOMAINS.items():
                if domain in url:
                    adjusted_scores[i] += bonus
                    break

        ranked_idx = np.argsort(-adjusted_scores)

        MAX_CHUNKS_PER_SOURCE = 2

        source_counts = {}
        results = []

        for idx in ranked_idx:
            source = self.chunks[idx].get("url", "unknown")

            if source_counts.get(source, 0) >= MAX_CHUNKS_PER_SOURCE:
                continue

            chunk = dict(self.chunks[idx])
            chunk["retrieval_score"] = float(scores[idx])

            results.append(chunk)
            source_counts[source] = source_counts.get(source, 0) + 1

            if len(results) >= top_k:
                break

        unique_sources = len(source_counts)
        print(
            f"Retrieved {len(results)} chunks from "
            f"{unique_sources} unique sources"
        )

        return results


if __name__ == "__main__":
    # Quick manual test
    sample_chunks = [
        {"text": "HEPA filters can remove fine particulate matter from classroom air.", "url": "a", "title": "A"},
        {"text": "Opening windows improves ventilation but depends on outdoor air quality.", "url": "b", "title": "B"},
        {"text": "Plants have minimal measurable effect on indoor air quality at room scale.", "url": "c", "title": "C"},
    ]
    idx = ChunkIndex(sample_chunks)
    idx.build()
    results = idx.retrieve("Does opening windows help air quality?", top_k=2)
    for r in results:
        print(r["retrieval_score"], "-", r["text"])
