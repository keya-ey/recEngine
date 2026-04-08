"""
User Tower with BERT4Rec-style Sequential Encoder
===================================================
Takes the user's feature vector (RFM segment one-hot + RFM scores +
behavioral tag multi-hot) and recent interaction sequence, then
produces a 64-dim user embedding via lightweight self-attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class UserTower(nn.Module):
    """
    Sequential encoder for user features.

    Architecture
    ------------
    1. Project user feature vector → d_model.
    2. Project each interaction-sequence token → d_model.
    3. Concatenate: [user_feat_token, seq_token_1, …, seq_token_T]
    4. Apply 2 Transformer encoder layers with self-attention.
    5. Mean-pool → final 64-dim embedding.
    """

    def __init__(
        self,
        user_feature_dim: int,
        item_embed_dim: int = 64,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_len: int = 50,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # project raw user features to d_model
        self.user_proj = nn.Linear(user_feature_dim, d_model)

        # project item embeddings (from interaction history) to d_model
        self.item_proj = nn.Linear(item_embed_dim, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        user_features: torch.Tensor,         # (B, user_feature_dim)
        interaction_embeds: torch.Tensor,     # (B, T, item_embed_dim)
        interaction_mask: torch.Tensor | None = None,  # (B, T) bool; True = padded
    ) -> torch.Tensor:
        """Return (B, embed_dim) user embeddings."""
        B, T, _ = interaction_embeds.shape

        # project
        user_tok = self.user_proj(user_features).unsqueeze(1)  # (B, 1, d_model)
        seq_tok = self.item_proj(interaction_embeds)            # (B, T, d_model)

        # concat [CLS-like user token, interaction tokens]
        tokens = torch.cat([user_tok, seq_tok], dim=1)         # (B, 1+T, d_model)
        tokens = self.pos_enc(tokens)

        # build attention mask: first token (user) is never masked
        if interaction_mask is not None:
            # prepend False (unmasked) for the user token
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=interaction_mask.device)
            full_mask = torch.cat([cls_mask, interaction_mask], dim=1)  # (B, 1+T)
        else:
            full_mask = None

        encoded = self.transformer(tokens, src_key_padding_mask=full_mask)  # (B, 1+T, d_model)

        # mean pool over non-padded tokens
        if full_mask is not None:
            # invert mask: True → keep, False → ignore
            keep = (~full_mask).unsqueeze(-1).float()  # (B, 1+T, 1)
            pooled = (encoded * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        return self.output_proj(pooled)  # (B, embed_dim)
