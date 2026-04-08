"""
Two-Tower Recommendation Model
================================
Wraps User Tower + Item Tower into a single trainable module.
Training uses in-batch negative sampling with dot-product similarity
and cross-entropy loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.item_tower import ItemTower
from model.user_tower import UserTower


class TwoTowerModel(nn.Module):
    """Dual-encoder recommendation model."""

    def __init__(
        self,
        user_feature_dim: int,
        n_categories: int,
        n_brands: int,
        n_product_tags: int,
        embed_dim: int = 64,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()

        self.user_tower = UserTower(
            user_feature_dim=user_feature_dim,
            item_embed_dim=embed_dim,
            d_model=embed_dim,
            embed_dim=embed_dim,
        )
        self.item_tower = ItemTower(
            n_categories=n_categories,
            n_brands=n_brands,
            n_tags=n_product_tags,
            embed_dim=embed_dim,
        )
        self.temperature = temperature
        self.embed_dim = embed_dim

    # ---- convenience encode methods ----------------------------------------

    def encode_user(
        self,
        user_features: torch.Tensor,
        interaction_embeds: torch.Tensor,
        interaction_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.user_tower(user_features, interaction_embeds, interaction_mask)

    def encode_item(
        self,
        category_ids: torch.Tensor,
        brand_ids: torch.Tensor,
        price: torch.Tensor,
        tag_vec: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(category_ids, brand_ids, price, tag_vec)

    # ---- training forward pass ---------------------------------------------

    def forward(
        self,
        user_features: torch.Tensor,         # (B, user_feature_dim)
        interaction_embeds: torch.Tensor,     # (B, T, embed_dim)
        interaction_mask: torch.Tensor | None,
        item_category_ids: torch.Tensor,      # (B,) positive items
        item_brand_ids: torch.Tensor,
        item_price: torch.Tensor,
        item_tag_vec: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute in-batch softmax cross-entropy loss.

        Each row's positive item is the diagonal; all other items in the
        batch act as negatives.
        """
        user_emb = self.encode_user(user_features, interaction_embeds, interaction_mask)
        item_emb = self.encode_item(item_category_ids, item_brand_ids, item_price, item_tag_vec)

        # L2-normalise for cosine similarity
        user_emb = F.normalize(user_emb, dim=-1)
        item_emb = F.normalize(item_emb, dim=-1)

        # similarity matrix (B, B)
        logits = torch.mm(user_emb, item_emb.t()) / self.temperature

        # labels: diagonal = positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "user_emb": user_emb, "item_emb": item_emb, "logits": logits}
