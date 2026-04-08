"""
End-to-End Recommendation Pipeline
====================================
Orchestrates: Feature Store → User Tower → FAISS retrieval → Re-ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from features.feature_store import FeatureStore
from model.two_tower import TwoTowerModel
from model.reranker import ReRanker
from serving.retriever import Retriever


@dataclass
class RecommendationResult:
    customer_id: int
    rfm_segment: str
    behavioral_tags: list[str]
    recommendations: list[dict]   # [{product_id, score, …}]


class RecommendationPipeline:
    """Full inference pipeline: features → retrieval → re-ranking."""

    def __init__(
        self,
        model: TwoTowerModel,
        retriever: Retriever,
        reranker: ReRanker,
        feature_store: FeatureStore,
        product_features: dict,      # product_id → {category_id, brand_id, price_norm, tag_vec}
        product_metadata: dict,       # product_id → {name, category, brand, price, tags}
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.retriever = retriever
        self.reranker = reranker
        self.feature_store = feature_store
        self.product_features = product_features
        self.product_metadata = product_metadata
        self.device = device
        self.model.eval()

    def _get_dummy_interaction(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Provide a zero-filled interaction sequence for cold-start users."""
        embed_dim = self.model.embed_dim
        dummy_embeds = torch.zeros(1, 1, embed_dim, device=self.device)
        dummy_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        return dummy_embeds, dummy_mask

    def _build_item_feature_matrix(self, product_ids: np.ndarray) -> np.ndarray:
        """Build a (K, item_feature_dim) matrix for re-ranker input."""
        rows = []
        for pid in product_ids:
            pf = self.product_features.get(int(pid))
            if pf is not None:
                row = np.concatenate([
                    [pf["category_id"] / 20.0],     # normalised
                    [pf["brand_id"] / 30.0],
                    [pf["price_norm"]],
                    pf["tag_vec"],
                ])
            else:
                row = np.zeros(3 + len(next(iter(self.product_features.values()))["tag_vec"]))
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _build_rank_weight_map(entries: list[dict], field: str) -> dict[str, float]:
        weights: dict[str, float] = {}
        for idx, entry in enumerate(entries):
            name = str(entry.get(field, "")).strip().lower()
            if not name:
                continue
            weights[name] = float(np.exp(-0.65 * idx))
        return weights

    def _compute_profile_alignment_score(
        self,
        product_meta: dict,
        customer_profile: dict | None,
    ) -> tuple[float, bool, bool]:
        profile = customer_profile or {}
        top_categories = profile.get("top_categories", [])
        top_brands = profile.get("top_brands", [])
        recent_orders = profile.get("recent_orders", [])

        top_category_weights = self._build_rank_weight_map(top_categories, "name")
        top_brand_weights = self._build_rank_weight_map(top_brands, "name")

        recent_category_weights: dict[str, float] = {}
        for idx, order in enumerate(recent_orders):
            cat = str(order.get("category", "")).strip().lower()
            if not cat:
                continue
            weight = float(np.exp(-0.55 * idx))
            recent_category_weights[cat] = max(recent_category_weights.get(cat, 0.0), weight)

        has_profile_signals = bool(top_category_weights or top_brand_weights or recent_category_weights)
        if not has_profile_signals:
            return 0.0, False, False

        category = str(product_meta.get("category", "")).strip().lower()
        brand = str(product_meta.get("brand", "")).strip().lower()

        category_alignment = max(
            top_category_weights.get(category, 0.0),
            recent_category_weights.get(category, 0.0),
        )
        brand_alignment = top_brand_weights.get(brand, 0.0)
        is_aligned = category_alignment > 0.0 or brand_alignment > 0.0

        alignment_score = (0.65 * category_alignment) + (0.35 * brand_alignment)
        if not is_aligned:
            alignment_score = max(0.0, alignment_score - 0.12)

        return float(np.clip(alignment_score, 0.0, 1.0)), True, is_aligned

    # ── Explanation Generation ──────────────────────────────────────

    def _generate_explanation(
        self,
        rfm_segment: str,
        user_tags: list[str],
        product_meta: dict,
        score: float,
        rank: int,
        customer_profile: dict | None = None,
    ) -> str:
        """
        Build a human-readable explanation for why a product was recommended.

        Uses customer purchase history (top categories/brands) for
        contextual, history-aware explanations.
        """
        reasons: list[str] = []
        item_tags = set(t.strip() for t in product_meta.get("tags", "").split("|") if t.strip())
        price = product_meta.get("price", 0)
        category = product_meta.get("category", "")
        brand = product_meta.get("brand", "")

        # extract purchase history
        profile = customer_profile or {}
        top_cats = [c["name"] for c in profile.get("top_categories", [])]
        top_cat_counts = {c["name"]: c["count"] for c in profile.get("top_categories", [])}
        top_brands_list = [b["name"] for b in profile.get("top_brands", [])]
        top_brand_counts = {b["name"]: b["count"] for b in profile.get("top_brands", [])}

        # ── History-based reasons (most specific, shown first) ──
        if category in top_cats:
            cnt = top_cat_counts.get(category, 0)
            reasons.append(f"You've purchased {cnt} {category} items before — this fits your pattern.")
        elif top_cats:
            reasons.append(f"Extends beyond your usual {', '.join(top_cats[:3])} into {category}.")

        if brand in top_brands_list:
            cnt = top_brand_counts.get(brand, 0)
            reasons.append(f"You've bought from {brand} {cnt} times — a brand you trust.")

        # ── Segment-based reasons ──
        if not reasons:  # only add segment reason if no history reason yet
            segment_reasons = {
                "Champions": f"As a top-tier customer, this {category} item aligns with your purchase history.",
                "Loyal": f"Your consistent purchase pattern suggests strong interest in {category} products.",
                "Promising": f"Based on your growing engagement, this {brand} product matches your emerging preferences.",
                "Potential Loyalist": f"Your recent activity shows increasing interest in products like this.",
                "At Risk": f"We thought you might like this {category} pick to re-engage with products you enjoy.",
                "Hibernating": f"Welcome back — this trending {category} item is popular among returning shoppers.",
                "Lost": f"This highly-rated {category} product is a great way to rediscover our catalog.",
            }
            seg_reason = segment_reasons.get(rfm_segment, "")
            if seg_reason:
                reasons.append(seg_reason)

        # ── Tag-based reasons ──
        if "Price Sensitive" in user_tags and ("Budget" in item_tags or "Discounted" in item_tags or price < 25):
            reasons.append("Matches your preference for budget-friendly products.")
        if "Premium Seeker" in user_tags and ("Premium" in item_tags or price > 100):
            reasons.append("Selected for your taste in premium, high-quality items.")
        if "Bargain Hunter" in user_tags and ("Discounted" in item_tags or "Budget" in item_tags):
            reasons.append("Great value pick matching your deal-hunting behavior.")
        if "Brand Loyal" in user_tags and brand not in top_brands_list:
            reasons.append(f"Recommended from {brand}, a brand similar to ones you prefer.")

        # ── Product attribute highlights ──
        if "Best Seller" in item_tags:
            reasons.append("Best seller in its category.")
        if "Eco-Friendly" in item_tags or "Cruelty-Free" in item_tags:
            reasons.append("Sustainably made — an eco-conscious choice.")
        if "Limited Edition" in item_tags:
            reasons.append("Limited edition — grab it before it's gone.")

        # ── Confidence language ──
        if score > 0.8:
            confidence = "Very strong match"
        elif score > 0.6:
            confidence = "Strong match"
        elif score > 0.4:
            confidence = "Good match"
        else:
            confidence = "Suggested pick"

        # ── Assemble ──
        if not reasons:
            reasons.append(f"This {category} product by {brand} scored highly against your profile.")

        explanation = f"{confidence} — " + " ".join(reasons[:2])
        return explanation

    # ── Main recommend method ──────────────────────────────────────

    @torch.no_grad()
    def recommend(
        self,
        customer_id: int,
        top_k: int = 20,
        retrieval_k: int = 500,
        interaction_embeds: Optional[torch.Tensor] = None,
        interaction_mask: Optional[torch.Tensor] = None,
        customer_profile: dict | None = None,
    ) -> RecommendationResult:
        """
        Generate top_k recommendations for a customer.

        Parameters
        ----------
        customer_id : target customer
        top_k : final number of recommendations
        retrieval_k : candidates to retrieve from FAISS
        interaction_embeds : optional (1, T, D) recent interaction embeddings
        interaction_mask : optional (1, T) padding mask
        """
        # 1. Feature hydration
        user_vec = self.feature_store.get_user_features(customer_id)
        user_tensor = torch.tensor(user_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 2. Interaction history (use dummy if not provided)
        if interaction_embeds is None:
            interaction_embeds, interaction_mask = self._get_dummy_interaction()

        # 3. User Tower forward pass
        user_emb = self.model.encode_user(user_tensor, interaction_embeds, interaction_mask)
        user_emb_np = user_emb.cpu().numpy().flatten()

        # 4. FAISS retrieval
        candidate_ids, retrieval_scores = self.retriever.retrieve(user_emb_np, top_k=retrieval_k)

        rfm_segment = self.feature_store.get_rfm_segment(customer_id)
        user_tags = self.feature_store.get_tags(customer_id)

        if len(candidate_ids) == 0:
            return RecommendationResult(
                customer_id=customer_id,
                rfm_segment=rfm_segment,
                behavioral_tags=user_tags,
                recommendations=[],
            )

        # 5. Re-ranking
        item_feat_matrix = self._build_item_feature_matrix(candidate_ids)
        rerank_pool_k = min(len(candidate_ids), max(top_k * 5, top_k))

        if self.reranker.model is not None:
            reranked = self.reranker.rerank(
                user_vec=user_vec,
                item_vecs=item_feat_matrix,
                candidate_ids=candidate_ids,
                top_k=rerank_pool_k,
            )
        else:
            # fallback: use retrieval scores directly
            top_idx = np.argsort(-retrieval_scores)[:rerank_pool_k]
            reranked = [(int(candidate_ids[i]), float(retrieval_scores[i])) for i in top_idx]

        profile_aligned = []
        for pid, base_score in reranked:
            meta = self.product_metadata.get(pid, {})
            alignment_score, has_profile_signals, is_aligned = self._compute_profile_alignment_score(
                product_meta=meta,
                customer_profile=customer_profile,
            )
            adjusted_score = float(base_score)
            if has_profile_signals:
                adjusted_score = (0.6 * float(base_score)) + (0.4 * alignment_score)
                if not is_aligned:
                    adjusted_score -= 0.05
                adjusted_score = float(np.clip(adjusted_score, 0.0, 1.0))
            profile_aligned.append((pid, adjusted_score))

        profile_aligned.sort(key=lambda x: x[1], reverse=True)
        reranked = profile_aligned[:top_k]

        # 6. Build result with product metadata + explanations
        recommendations = []
        for rank, (pid, score) in enumerate(reranked, 1):
            meta = self.product_metadata.get(pid, {})
            explanation = self._generate_explanation(
                rfm_segment=rfm_segment,
                user_tags=user_tags,
                product_meta=meta,
                score=score,
                rank=rank,
                customer_profile=customer_profile,
            )
            recommendations.append({
                "product_id": pid,
                "score": round(score, 4),
                "name": meta.get("name", f"Product {pid}"),
                "category": meta.get("category", "Unknown"),
                "brand": meta.get("brand", "Unknown"),
                "price": meta.get("price", 0.0),
                "tags": meta.get("tags", ""),
                "explanation": explanation,
            })

        return RecommendationResult(
            customer_id=customer_id,
            rfm_segment=rfm_segment,
            behavioral_tags=user_tags,
            recommendations=recommendations,
        )

