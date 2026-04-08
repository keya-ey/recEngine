"""
Re-Ranker — Gradient Boosting Cross-Feature Scorer
====================================================
Takes the top-K candidates from the retriever and re-ranks them
using a Gradient Boosting model trained on user-item cross features.
Uses sklearn to avoid LightGBM compatibility issues on macOS.
"""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class ReRanker:
    """
    Sklearn GradientBoosting re-ranker.

    Training features (per user-item pair):
        [user_features | item_features | cross_dot_product]

    Label: 1 if user purchased/clicked the item, 0 otherwise.
    """

    def __init__(self) -> None:
        self.model: Optional[GradientBoostingClassifier] = None

    # ---- feature construction -----------------------------------------------

    @staticmethod
    def build_features(
        user_vec: np.ndarray,           # (user_feature_dim,)
        item_vecs: np.ndarray,          # (K, item_feature_dim)
    ) -> np.ndarray:
        """
        Build cross-feature matrix for K candidates.

        Returns (K, user_dim + item_dim + 1) where the +1 is the
        dot-product cross feature (computed over overlapping dims).
        """
        K = item_vecs.shape[0]
        user_repeated = np.tile(user_vec, (K, 1))  # (K, user_dim)

        # cross feature: dot product over the min shared dimensions
        min_dim = min(user_repeated.shape[1], item_vecs.shape[1])
        cross = np.sum(
            user_repeated[:, :min_dim] * item_vecs[:, :min_dim],
            axis=1, keepdims=True,
        )

        return np.concatenate([user_repeated, item_vecs, cross], axis=1)

    # ---- training -----------------------------------------------------------

    def train(
        self,
        features: np.ndarray,   # (N, feature_dim)
        labels: np.ndarray,     # (N,) binary
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        n_estimators: int = 150,
    ) -> None:
        """Train the Gradient Boosting re-ranker."""
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            max_features=0.8,
            subsample=0.8,
            random_state=42,
            verbose=0,
        )
        self.model.fit(features, labels)

        if val_features is not None and val_labels is not None:
            val_score = self.model.score(val_features, val_labels)
            print(f"  Re-ranker validation accuracy: {val_score:.4f}")

    # ---- inference ----------------------------------------------------------

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score candidates. Returns (K,) predicted CTR probabilities."""
        if self.model is None:
            raise RuntimeError("ReRanker has not been trained yet.")
        return self.model.predict_proba(features)[:, 1]

    def rerank(
        self,
        user_vec: np.ndarray,
        item_vecs: np.ndarray,
        candidate_ids: np.ndarray,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """
        Re-rank candidates and return top_k (product_id, score) tuples.
        """
        feats = self.build_features(user_vec, item_vecs)
        scores = self.score(feats)
        ranked_idx = np.argsort(-scores)[:top_k]
        return [(int(candidate_ids[i]), float(scores[i])) for i in ranked_idx]

    # ---- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        if self.model is not None:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
