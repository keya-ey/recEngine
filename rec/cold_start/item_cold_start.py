"""
Item Cold-Start: Content-Based Embedding
==========================================
Uses a sentence-transformer to encode new product text (title + tags)
into a semantic vector, then projects it into the item embedding space
via a learned linear adapter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class ItemColdStart:
    """
    Map new products into the item embedding space using text semantics.

    For prototype purposes, uses a simple average of word hashes when
    sentence-transformers is unavailable, or a real sentence-transformer
    model when loaded.
    """

    def __init__(self, item_embed_dim: int = 64, text_embed_dim: int = 384) -> None:
        self.item_embed_dim = item_embed_dim
        self.text_embed_dim = text_embed_dim
        self.sentence_model = None

        # linear adapter: text_embed_dim → item_embed_dim
        self.adapter = nn.Linear(text_embed_dim, item_embed_dim)
        # initialise as near-identity (scaled down)
        nn.init.xavier_uniform_(self.adapter.weight)

    def load_sentence_model(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load a sentence-transformers model for semantic encoding."""
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer(model_name)
            self.text_embed_dim = self.sentence_model.get_sentence_embedding_dimension()
            # re-create adapter for actual dimension
            self.adapter = nn.Linear(self.text_embed_dim, self.item_embed_dim)
            nn.init.xavier_uniform_(self.adapter.weight)
        except ImportError:
            print("⚠ sentence-transformers not available; using hash-based fallback.")

    def _hash_encode(self, text: str) -> np.ndarray:
        """Deterministic hash-based fallback encoder."""
        vec = np.zeros(self.text_embed_dim, dtype=np.float32)
        for i, word in enumerate(text.lower().split()):
            h = hash(word) % self.text_embed_dim
            vec[h] += 1.0 / (i + 1)
        # normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode_product(self, name: str, tags: str = "", description: str = "") -> np.ndarray:
        """
        Encode a new product into the item embedding space.

        Parameters
        ----------
        name : product title
        tags : pipe-separated tag string (e.g., "Budget|Eco-Friendly")
        description : optional product description

        Returns
        -------
        (item_embed_dim,) float32 array
        """
        text = f"{name} {tags.replace('|', ' ')} {description}".strip()

        if self.sentence_model is not None:
            text_emb = self.sentence_model.encode(text, convert_to_numpy=True)
        else:
            text_emb = self._hash_encode(text)

        text_tensor = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            item_emb = self.adapter(text_tensor).squeeze(0).numpy()

        return item_emb

    def train_adapter(
        self,
        texts: list[str],
        target_embeddings: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> float:
        """
        Train the linear adapter to align text embeddings with
        pre-computed item tower embeddings.

        Returns final MSE loss.
        """
        # encode all texts
        if self.sentence_model is not None:
            text_embs = self.sentence_model.encode(texts, convert_to_numpy=True)
        else:
            text_embs = np.array([self._hash_encode(t) for t in texts])

        X = torch.tensor(text_embs, dtype=torch.float32)
        Y = torch.tensor(target_embeddings, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.adapter.train()
        final_loss = 0.0
        for epoch in range(epochs):
            pred = self.adapter(X)
            loss = loss_fn(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.adapter.eval()
        return final_loss
