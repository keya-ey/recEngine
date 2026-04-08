"""
FAISS-based Approximate Nearest Neighbor Retriever
====================================================
Loads pre-computed item embeddings into a FAISS index and performs
fast similarity search for candidate generation.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class Retriever:
    """Vector similarity search over pre-computed item embeddings."""

    def __init__(self, embed_dim: int = 64) -> None:
        self.embed_dim = embed_dim
        self.index: faiss.IndexFlatIP | None = None
        self.product_ids: np.ndarray | None = None

    # ---- index management ---------------------------------------------------

    def build_index(
        self,
        embeddings: np.ndarray,     # (N, embed_dim) float32
        product_ids: np.ndarray,    # (N,) int
    ) -> None:
        """Build a FAISS inner-product index from pre-computed embeddings."""
        assert embeddings.shape[1] == self.embed_dim
        n = embeddings.shape[0]

        # L2-normalise so inner product == cosine similarity
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeddings)
        self.product_ids = product_ids.copy()

        assert self.index.ntotal == n

    def save(self, dir_path: str) -> None:
        """Persist index and ID mapping to disk."""
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "item.index"))
        np.save(str(p / "product_ids.npy"), self.product_ids)

    def load(self, dir_path: str) -> None:
        """Load index and ID mapping from disk."""
        p = Path(dir_path)
        self.index = faiss.read_index(str(p / "item.index"))
        self.product_ids = np.load(str(p / "product_ids.npy"))
        self.embed_dim = self.index.d

    # ---- retrieval ----------------------------------------------------------

    def retrieve(
        self,
        user_embedding: np.ndarray,  # (embed_dim,) or (1, embed_dim)
        top_k: int = 500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top_k most similar items.

        Returns
        -------
        product_ids : (top_k,) int array
        scores      : (top_k,) float array (cosine similarity)
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded. Call build_index() or load() first.")

        q = user_embedding.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
        scores = scores[0]
        indices = indices[0]

        # filter out -1 (padding if index is smaller than top_k)
        valid = indices >= 0
        return self.product_ids[indices[valid]], scores[valid]
