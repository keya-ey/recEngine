"""
Behavioral Tagging Pipeline
=============================
Applies aggregations over browsing_history and cart_activity
to generate behavioral tags for each customer.

Tags generated:
  • Price Sensitive    – majority of views on low-price items
  • Premium Seeker     – majority of views on high-price items
  • Category Explorer  – browsed many distinct categories
  • Brand Loyal        – views concentrated in one brand
  • Cart Abandoner     – many cart adds relative to purchases
  • Bargain Hunter     – strong preference for discounted items
  • Frequent Browser   – top browsing activity
  • Impulse Buyer      – high cart-add rate relative to views
  • Niche Shopper      – views concentrated in one category
  • Window Shopper     – many views but very few cart adds
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tag definitions
# ---------------------------------------------------------------------------
ALL_TAGS = [
    "Price Sensitive",
    "Premium Seeker",
    "Category Explorer",
    "Brand Loyal",
    "Cart Abandoner",
    "Bargain Hunter",
    "Frequent Browser",
    "Impulse Buyer",
    "Niche Shopper",
    "Window Shopper",
]


# ---------------------------------------------------------------------------
# Individual tag detectors — all session-agnostic for reliability
# ---------------------------------------------------------------------------

def _detect_price_sensitive(
    browsing: pd.DataFrame,
    products: pd.DataFrame,
    threshold_pct: float = 0.40,
    min_views: int = 5,
) -> set[int]:
    """Customers where ≥threshold_pct of views are on bottom-quartile-price items."""
    price_q25 = products["price"].quantile(0.25)
    low_pids = set(products.loc[products["price"] <= price_q25, "product_id"])

    tagged: set[int] = set()
    for cid, grp in browsing.groupby("customer_id"):
        total = len(grp)
        if total < min_views:
            continue
        low_count = grp["product_id"].isin(low_pids).sum()
        if low_count / total >= threshold_pct:
            tagged.add(int(cid))
    return tagged


def _detect_premium_seeker(
    browsing: pd.DataFrame,
    products: pd.DataFrame,
    threshold_pct: float = 0.35,
    min_views: int = 5,
) -> set[int]:
    """Customers where ≥threshold_pct of views are on top-quartile-price items."""
    price_q75 = products["price"].quantile(0.75)
    high_pids = set(products.loc[products["price"] >= price_q75, "product_id"])

    tagged: set[int] = set()
    for cid, grp in browsing.groupby("customer_id"):
        total = len(grp)
        if total < min_views:
            continue
        high_count = grp["product_id"].isin(high_pids).sum()
        if high_count / total >= threshold_pct:
            tagged.add(int(cid))
    return tagged


def _detect_category_explorer(
    browsing: pd.DataFrame,
    products: pd.DataFrame,
    threshold: int = 5,
) -> set[int]:
    """Customers who browsed ≥threshold distinct categories overall."""
    merged = browsing.merge(products[["product_id", "category"]], on="product_id", how="left")
    cats_per_user = merged.groupby("customer_id")["category"].nunique()
    return set(int(c) for c, n in cats_per_user.items() if n >= threshold)


def _detect_brand_loyal(
    browsing: pd.DataFrame,
    products: pd.DataFrame,
    threshold_pct: float = 0.35,
    min_views: int = 8,
) -> set[int]:
    """Customers where ≥threshold_pct of views are in a single brand."""
    merged = browsing.merge(products[["product_id", "brand"]], on="product_id", how="left")

    tagged: set[int] = set()
    for cid, grp in merged.groupby("customer_id"):
        total = len(grp)
        if total < min_views:
            continue
        brand_counts = grp["brand"].value_counts()
        if brand_counts.iloc[0] / total >= threshold_pct:
            tagged.add(int(cid))
    return tagged


def _detect_cart_abandoner(
    cart: pd.DataFrame,
    orders: pd.DataFrame,
    threshold_ratio: float = 2.0,
    min_adds: int = 3,
) -> set[int]:
    """Customers whose cart-adds far exceed their purchases."""
    adds_per_user = cart[cart["action"] == "add"].groupby("customer_id").size()
    orders_per_user = orders.groupby("customer_id").size()

    tagged: set[int] = set()
    for cid, n_adds in adds_per_user.items():
        if n_adds < min_adds:
            continue
        n_orders = orders_per_user.get(int(cid), 0)
        if n_orders == 0 or n_adds / max(n_orders, 1) >= threshold_ratio:
            tagged.add(int(cid))
    return tagged


def _detect_bargain_hunter(
    browsing: pd.DataFrame,
    products: pd.DataFrame,
    threshold_pct: float = 0.30,
    min_views: int = 5,
) -> set[int]:
    """Customers who frequently view items tagged 'Discounted' or 'Budget'."""
    discount_pids = set(products.loc[
        products["tags"].str.contains("Discounted|Budget", case=False, na=False),
        "product_id",
    ])
    tagged: set[int] = set()
    for cid, grp in browsing.groupby("customer_id"):
        total = len(grp)
        if total < min_views:
            continue
        disc_count = grp["product_id"].isin(discount_pids).sum()
        if disc_count / total >= threshold_pct:
            tagged.add(int(cid))
    return tagged


def _detect_frequent_browser(
    browsing: pd.DataFrame,
    top_pct: float = 0.20,
) -> set[int]:
    """Top top_pct% of customers by total browsing volume."""
    counts = browsing.groupby("customer_id").size()
    threshold = counts.quantile(1.0 - top_pct)
    return set(int(c) for c, n in counts.items() if n >= threshold)


def _detect_impulse_buyer(
    browsing: pd.DataFrame,
    cart: pd.DataFrame,
    threshold_ratio: float = 0.30,
    min_views: int = 5,
) -> set[int]:
    """Customers with a high cart-add-to-view ratio."""
    views_per_user = browsing.groupby("customer_id").size()
    adds_per_user = cart[cart["action"] == "add"].groupby("customer_id").size()

    tagged: set[int] = set()
    for cid, n_views in views_per_user.items():
        if n_views < min_views:
            continue
        n_adds = adds_per_user.get(int(cid), 0)
        if n_adds / n_views >= threshold_ratio:
            tagged.add(int(cid))
    return tagged


def _detect_niche_shopper(
    browsing: pd.DataFrame,
    products: pd.DataFrame,
    threshold_pct: float = 0.50,
    min_views: int = 8,
) -> set[int]:
    """Customers where ≥threshold_pct of views are in a single category."""
    merged = browsing.merge(products[["product_id", "category"]], on="product_id", how="left")
    tagged: set[int] = set()
    for cid, grp in merged.groupby("customer_id"):
        total = len(grp)
        if total < min_views:
            continue
        cat_counts = grp["category"].value_counts()
        if cat_counts.iloc[0] / total >= threshold_pct:
            tagged.add(int(cid))
    return tagged


def _detect_window_shopper(
    browsing: pd.DataFrame,
    cart: pd.DataFrame,
    threshold_ratio: float = 0.05,
    min_views: int = 20,
) -> set[int]:
    """Customers with many views but very few cart-adds."""
    views_per_user = browsing.groupby("customer_id").size()
    adds_per_user = cart[cart["action"] == "add"].groupby("customer_id").size()

    tagged: set[int] = set()
    for cid, n_views in views_per_user.items():
        if n_views < min_views:
            continue
        n_adds = adds_per_user.get(int(cid), 0)
        if n_adds / n_views <= threshold_ratio:
            tagged.add(int(cid))
    return tagged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_tags(
    browsing_history: pd.DataFrame,
    cart_activity: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
) -> dict[int, list[str]]:
    """
    Compute behavioral tags for all customers.

    Returns
    -------
    dict mapping customer_id → list of tag strings
    """
    price_sensitive = _detect_price_sensitive(browsing_history, products)
    premium_seeker = _detect_premium_seeker(browsing_history, products)
    category_explorer = _detect_category_explorer(browsing_history, products)
    brand_loyal = _detect_brand_loyal(browsing_history, products)
    cart_abandoner = _detect_cart_abandoner(cart_activity, orders)
    bargain_hunter = _detect_bargain_hunter(browsing_history, products)
    frequent_browser = _detect_frequent_browser(browsing_history)
    impulse_buyer = _detect_impulse_buyer(browsing_history, cart_activity)
    niche_shopper = _detect_niche_shopper(browsing_history, products)
    window_shopper = _detect_window_shopper(browsing_history, cart_activity)

    tag_sets = {
        "Price Sensitive": price_sensitive,
        "Premium Seeker": premium_seeker,
        "Category Explorer": category_explorer,
        "Brand Loyal": brand_loyal,
        "Cart Abandoner": cart_abandoner,
        "Bargain Hunter": bargain_hunter,
        "Frequent Browser": frequent_browser,
        "Impulse Buyer": impulse_buyer,
        "Niche Shopper": niche_shopper,
        "Window Shopper": window_shopper,
    }

    all_cids: set[int] = set()
    all_cids.update(browsing_history["customer_id"].unique())
    all_cids.update(cart_activity["customer_id"].unique())

    result: dict[int, list[str]] = {}
    for cid in all_cids:
        tags: list[str] = []
        for tag_name, tag_set in tag_sets.items():
            if cid in tag_set:
                tags.append(tag_name)
        result[int(cid)] = tags

    return result


# ---------------------------------------------------------------------------
# One-hot encoding for model input
# ---------------------------------------------------------------------------

def encode_tags(tags: list[str]) -> list[float]:
    """Return a multi-hot vector for the given list of tag names."""
    vec = [0.0] * len(ALL_TAGS)
    for t in tags:
        if t in ALL_TAGS:
            vec[ALL_TAGS.index(t)] = 1.0
    return vec
