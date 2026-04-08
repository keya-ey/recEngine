"""
Lightweight In-Memory Feature Store
=====================================
Combines offline RFM features with online behavioral tags
and serves concatenated feature vectors for model inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features.rfm import ALL_SEGMENTS, encode_rfm_segment
from features.tagging import ALL_TAGS, encode_tags


@dataclass
class FeatureStore:
    """
    Minimal feature store with an offline store (RFM segments)
    and an online store (behavioral tags).

    Attributes
    ----------
    offline_store : dict[int, dict]
        customer_id → {rfm_segment, R, F, M, recency_days, frequency, monetary}
    online_store  : dict[int, list[str]]
        customer_id → list of behavioral tag strings
    """

    offline_store: dict[int, dict] = field(default_factory=dict)
    online_store: dict[int, list[str]] = field(default_factory=dict)

    # ----- dimensions -------------------------------------------------------
    @property
    def rfm_dim(self) -> int:
        """Dimensionality of one-hot RFM segment vector."""
        return len(ALL_SEGMENTS)

    @property
    def tag_dim(self) -> int:
        """Dimensionality of multi-hot tag vector."""
        return len(ALL_TAGS)

    @property
    def rfm_numeric_dim(self) -> int:
        """3 normalised numeric features: R, F, M raw scores."""
        return 3

    @property
    def total_user_feature_dim(self) -> int:
        return self.rfm_dim + self.rfm_numeric_dim + self.tag_dim

    # ----- population -------------------------------------------------------
    def load_rfm(self, rfm_df: pd.DataFrame) -> None:
        """Populate offline store from the output of `compute_rfm()`."""
        for _, row in rfm_df.iterrows():
            cid = int(row["customer_id"])
            self.offline_store[cid] = {
                "rfm_segment": row["rfm_segment"],
                "R": int(row["R"]),
                "F": int(row["F"]),
                "M": int(row["M"]),
                "recency_days": int(row["recency_days"]),
                "frequency": int(row["frequency"]),
                "monetary": float(row["monetary"]),
            }

    def load_tags(self, tags: dict[int, list[str]]) -> None:
        """Populate online store from the output of `compute_tags()`."""
        self.online_store = tags

    def update_tags(self, customer_id: int, tags: list[str]) -> None:
        """Real-time update: overwrite a single customer's tags."""
        self.online_store[customer_id] = tags

    # ----- retrieval --------------------------------------------------------
    def get_user_features(self, customer_id: int) -> np.ndarray:
        """
        Return a concatenated feature vector:
            [rfm_segment_onehot | R_norm, F_norm, M_norm | tag_multihot]

        For unknown customers (cold-start), returns a zero vector.
        """
        # RFM part
        rfm_info = self.offline_store.get(customer_id)
        if rfm_info is not None:
            seg_vec = encode_rfm_segment(rfm_info["rfm_segment"])
            rfm_numeric = [
                rfm_info["R"] / 5.0,
                rfm_info["F"] / 5.0,
                rfm_info["M"] / 5.0,
            ]
        else:
            seg_vec = [0.0] * self.rfm_dim
            rfm_numeric = [0.0, 0.0, 0.0]

        # Tag part
        tag_list = self.online_store.get(customer_id, [])
        tag_vec = encode_tags(tag_list)

        return np.array(seg_vec + rfm_numeric + tag_vec, dtype=np.float32)

    def get_rfm_segment(self, customer_id: int) -> str:
        """Return the named RFM segment, or 'Unknown' for cold-start users."""
        info = self.offline_store.get(customer_id)
        return info["rfm_segment"] if info else "Unknown"

    def get_tags(self, customer_id: int) -> list[str]:
        """Return the list of behavioral tags for a customer."""
        return self.online_store.get(customer_id, [])
