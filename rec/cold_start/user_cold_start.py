"""
User Cold-Start: Instant Behavioral Tagging
=============================================
Handles brand-new users who have no RFM profile or history.
Uses contextual signals and rapidly accumulates micro-interactions
to warm up the user profile.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from features.feature_store import FeatureStore
from features.tagging import ALL_TAGS


# Default contextual signals for cold-start users
DEFAULT_CONTEXT = {
    "time_of_day": "afternoon",     # morning / afternoon / evening / night
    "device": "mobile",             # mobile / desktop / tablet
    "referral": "organic",          # organic / social / ad / email
    "location": "Unknown",
}

TIME_BUCKETS = ["morning", "afternoon", "evening", "night"]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]
REFERRAL_SOURCES = ["organic", "social", "ad", "email"]


def encode_context(
    time_of_day: str = "afternoon",
    device: str = "mobile",
    referral: str = "organic",
) -> np.ndarray:
    """
    Encode contextual signals into a fixed-length vector.

    Returns (len(TIME_BUCKETS) + len(DEVICE_TYPES) + len(REFERRAL_SOURCES),)
    """
    vec = []
    # one-hot time
    for t in TIME_BUCKETS:
        vec.append(1.0 if t == time_of_day else 0.0)
    # one-hot device
    for d in DEVICE_TYPES:
        vec.append(1.0 if d == device else 0.0)
    # one-hot referral
    for r in REFERRAL_SOURCES:
        vec.append(1.0 if r == referral else 0.0)
    return np.array(vec, dtype=np.float32)


class UserColdStartHandler:
    """
    Manages cold-start user profiles.

    For a new user:
    1. Immediately builds a baseline feature vector from context.
    2. As micro-interactions arrive, accumulates behavioural tags
       and updates the feature store in real-time.
    """

    def __init__(self, feature_store: FeatureStore) -> None:
        self.feature_store = feature_store
        # track micro-interactions per new user
        self._interactions: dict[int, list[dict]] = {}

    def is_cold_start(self, customer_id: int) -> bool:
        """Check if a customer has no offline (RFM) profile."""
        return customer_id not in self.feature_store.offline_store

    def initialise_user(
        self,
        customer_id: int,
        time_of_day: Optional[str] = None,
        device: Optional[str] = None,
        referral: Optional[str] = None,
    ) -> np.ndarray:
        """
        Build an initial feature vector from context signals.
        Also seeds the online store with empty tags.
        """
        if time_of_day is None:
            hour = datetime.now().hour
            if hour < 6:
                time_of_day = "night"
            elif hour < 12:
                time_of_day = "morning"
            elif hour < 18:
                time_of_day = "afternoon"
            else:
                time_of_day = "evening"

        device = device or DEFAULT_CONTEXT["device"]
        referral = referral or DEFAULT_CONTEXT["referral"]

        # seed online store
        self.feature_store.update_tags(customer_id, [])
        self._interactions[customer_id] = []

        return encode_context(time_of_day, device, referral)

    def record_interaction(
        self,
        customer_id: int,
        product_id: int,
        action: str,
        product_price: float = 0.0,
        product_category: str = "",
        product_brand: str = "",
    ) -> list[str]:
        """
        Record a micro-interaction and update behavioural tags in real-time.

        Returns the updated list of tags for this user.
        """
        if customer_id not in self._interactions:
            self._interactions[customer_id] = []

        self._interactions[customer_id].append({
            "product_id": product_id,
            "action": action,
            "price": product_price,
            "category": product_category,
            "brand": product_brand,
            "timestamp": datetime.now().isoformat(),
        })

        # re-compute tags from accumulated interactions
        tags = self._infer_tags(customer_id)
        self.feature_store.update_tags(customer_id, tags)
        return tags

    def _infer_tags(self, customer_id: int) -> list[str]:
        """Simple heuristic tagging from accumulated micro-interactions."""
        interactions = self._interactions.get(customer_id, [])
        if not interactions:
            return []

        tags = []
        prices = [i["price"] for i in interactions if i["price"] > 0]
        categories = [i["category"] for i in interactions if i["category"]]
        brands = [i["brand"] for i in interactions if i["brand"]]

        # Price Sensitive: avg price in bottom quartile
        if prices and np.mean(prices) < 25:
            tags.append("Price Sensitive")

        # Premium Seeker: avg price in top quartile
        if prices and np.mean(prices) > 150:
            tags.append("Premium Seeker")

        # Category Explorer: ≥4 distinct categories
        if len(set(categories)) >= 4:
            tags.append("Category Explorer")

        # Brand Loyal: ≥50% views in one brand
        if brands:
            from collections import Counter
            brand_counts = Counter(brands)
            top_brand_pct = brand_counts.most_common(1)[0][1] / len(brands)
            if top_brand_pct >= 0.5 and len(brands) >= 5:
                tags.append("Brand Loyal")

        return tags
