"""
RFM (Recency, Frequency, Monetary) Scoring Pipeline
=====================================================
Computes per-customer RFM scores from orders/order_items data
and discretizes them into named segments.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Segment labels (high → low quintile)
# ---------------------------------------------------------------------------
RFM_SEGMENTS = {
    (5, 5, 5): "Champions",
    (5, 5, 4): "Champions",
    (5, 4, 5): "Champions",
    (5, 4, 4): "Loyal",
    (4, 5, 5): "Loyal",
    (4, 5, 4): "Loyal",
    (4, 4, 5): "Loyal",
    (4, 4, 4): "Loyal",
    (5, 3, 3): "Promising",
    (4, 3, 3): "Promising",
    (5, 2, 2): "Promising",
    (4, 2, 2): "Promising",
    (3, 3, 3): "Potential Loyalist",
    (3, 4, 4): "Potential Loyalist",
    (3, 4, 3): "Potential Loyalist",
    (3, 3, 4): "Potential Loyalist",
    (3, 5, 5): "Potential Loyalist",
    (2, 3, 3): "At Risk",
    (2, 2, 3): "At Risk",
    (2, 3, 2): "At Risk",
    (2, 2, 2): "At Risk",
    (1, 3, 3): "Hibernating",
    (1, 2, 2): "Hibernating",
    (1, 1, 2): "Hibernating",
    (1, 2, 1): "Hibernating",
    (1, 1, 1): "Lost",
}


def _nearest_segment(r: int, f: int, m: int) -> str:
    """Find best matching segment; fall back to nearest key by Manhattan distance."""
    key = (r, f, m)
    if key in RFM_SEGMENTS:
        return RFM_SEGMENTS[key]

    best_dist = float("inf")
    best_seg = "At Risk"  # default fallback
    for (kr, kf, km), seg in RFM_SEGMENTS.items():
        dist = abs(r - kr) + abs(f - kf) + abs(m - km)
        if dist < best_dist:
            best_dist = dist
            best_seg = seg
    return best_seg


def compute_rfm(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    reference_date: Optional[datetime] = None,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Compute RFM scores for every customer who has at least one order.

    Parameters
    ----------
    orders : DataFrame with columns [order_id, customer_id, order_date, total]
    order_items : DataFrame with columns [order_id, product_id, quantity, price]
    reference_date : anchor date for Recency; defaults to max(order_date) + 1 day
    n_quantiles : number of bins (default 5 → quintiles)

    Returns
    -------
    DataFrame with columns:
        customer_id, recency_days, frequency, monetary,
        R, F, M, rfm_segment
    """
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    if reference_date is None:
        reference_date = orders["order_date"].max() + pd.Timedelta(days=1)

    # --- Monetary from order_items (quantity × price) -----------------------
    item_totals = order_items.copy()
    item_totals["line_total"] = item_totals["quantity"] * item_totals["price"]
    monetary_per_order = item_totals.groupby("order_id")["line_total"].sum().reset_index()
    monetary_per_order.columns = ["order_id", "monetary_value"]

    orders = orders.merge(monetary_per_order, on="order_id", how="left")
    orders["monetary_value"] = orders["monetary_value"].fillna(0)

    # --- Aggregate per customer ---------------------------------------------
    rfm = orders.groupby("customer_id").agg(
        recency_days=("order_date", lambda x: (reference_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("monetary_value", "sum"),
    ).reset_index()

    rfm["monetary"] = rfm["monetary"].round(2)

    # --- Quintile scoring (1 = worst, 5 = best) -----------------------------
    rfm["R"] = pd.qcut(rfm["recency_days"], q=n_quantiles, labels=False, duplicates="drop")
    rfm["R"] = n_quantiles - rfm["R"]  # invert: lower recency → higher score

    rfm["F"] = pd.qcut(rfm["frequency"], q=n_quantiles, labels=False, duplicates="drop") + 1

    rfm["M"] = pd.qcut(rfm["monetary"], q=n_quantiles, labels=False, duplicates="drop") + 1

    # Clamp to [1, n_quantiles]
    for col in ("R", "F", "M"):
        rfm[col] = rfm[col].clip(1, n_quantiles).astype(int)

    # --- Map to named segments ----------------------------------------------
    rfm["rfm_segment"] = rfm.apply(
        lambda row: _nearest_segment(row["R"], row["F"], row["M"]), axis=1
    )

    return rfm


# ---------------------------------------------------------------------------
# Convenience: one-hot encode RFM segment for model input
# ---------------------------------------------------------------------------
ALL_SEGMENTS = sorted(set(RFM_SEGMENTS.values()))  # deterministic order


def encode_rfm_segment(segment: str) -> list[float]:
    """Return a one-hot vector for the given segment name."""
    vec = [0.0] * len(ALL_SEGMENTS)
    if segment in ALL_SEGMENTS:
        vec[ALL_SEGMENTS.index(segment)] = 1.0
    return vec
