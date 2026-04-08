"""
Item Tower
==========
MLP that encodes product features into a 64-dim embedding.

Input features (per product):
  - category   (label-encoded → embedding)
  - brand      (label-encoded → embedding)
  - price      (normalised scalar)
  - tags       (multi-hot vector)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ItemTower(nn.Module):
    """Encodes product features → dense embedding."""

    def __init__(
        self,
        n_categories: int,
        n_brands: int,
        n_tags: int,
        embed_dim: int = 64,
        cat_embed_dim: int = 16,
        brand_embed_dim: int = 16,
    ) -> None:
        super().__init__()

        self.cat_emb = nn.Embedding(n_categories, cat_embed_dim)
        self.brand_emb = nn.Embedding(n_brands, brand_embed_dim)

        # price (1) + cat_embed + brand_embed + n_tags
        mlp_in = cat_embed_dim + brand_embed_dim + 1 + n_tags

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
        )

    def forward(
        self,
        category_ids: torch.Tensor,   # (B,) long
        brand_ids: torch.Tensor,       # (B,) long
        price: torch.Tensor,           # (B, 1) float
        tag_vec: torch.Tensor,         # (B, n_tags) float multi-hot
    ) -> torch.Tensor:
        """Return (B, embed_dim) item embeddings."""
        cat_e = self.cat_emb(category_ids)       # (B, cat_embed_dim)
        brand_e = self.brand_emb(brand_ids)      # (B, brand_embed_dim)
        x = torch.cat([cat_e, brand_e, price, tag_vec], dim=-1)
        return self.mlp(x)
