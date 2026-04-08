"""
FastAPI Recommendation API
============================
Endpoints:
  GET  /recommend/{customer_id}?top_k=20  → top-K recommendations
  POST /event                              → ingest a browsing/cart event
  GET  /health                             → health check
  GET  /                                   → demo UI
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.rfm import compute_rfm, ALL_SEGMENTS
from features.feature_store import FeatureStore
from features.tagging import ALL_TAGS
from model.two_tower import TwoTowerModel
from model.reranker import ReRanker
from serving.retriever import Retriever
from serving.pipeline import RecommendationPipeline
from cold_start.user_cold_start import UserColdStartHandler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "generated"

app = FastAPI(
    title="Tag-Aware Recommendation Engine",
    version="1.0.0",
    description="Personalized, tag-aware product recommendations via Two-Tower + re-ranking.",
)

# ---------------------------------------------------------------------------
# Global state (populated on startup)
# ---------------------------------------------------------------------------
pipeline: RecommendationPipeline | None = None
cold_start_handler: UserColdStartHandler | None = None
product_metadata_global: dict = {}
customer_profiles: dict = {}  # customer_id → profile dict
customers_data: dict = {}  # customer_id → {name, location, signup_date}
analytics_cache: dict = {}  # precomputed analytics data
churn_score_cache: dict[int, float] = {}
lifecycle_stage_cache: dict[int, str] = {}

LIFECYCLE_STAGES = [
    "Potential Customer",
    "New",
    "Loyal Customers",
    "Champions",
    "Churn Risk",
]

RISK_SEGMENTS = {"At Risk", "Hibernating", "Lost"}

SEGMENT_RISK_BIAS = {
    "Champions": -1.2,
    "Loyal": -0.8,
    "Potential Loyalist": -0.2,
    "Promising": 0.15,
    "At Risk": 0.8,
    "Hibernating": 1.0,
    "Lost": 1.25,
}

ARPU_THRESHOLD_FACTOR = 1.0
ARPU_TREND_WEEKS = 12
SAVE_VALUE_HIGH_THRESHOLD = 500.0
SAVE_VALUE_MEDIUM_THRESHOLD = 200.0
SAVE_ESCALATION_DAYS = 7
SAVE_MONTHLY_THRESHOLD_RATIO = 0.30


# ---------------------------------------------------------------------------
# Build customer profile from raw order data
# ---------------------------------------------------------------------------
def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_product_tags(tag_string: object, price: float) -> str:
    raw = str(tag_string or "")
    tags = [t.strip() for t in raw.split("|") if t and t.strip()]
    seen: set[str] = set()
    cleaned: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            cleaned.append(t)

    if not cleaned:
        if price < 25:
            cleaned = ["Budget", "Trending"]
        elif price < 100:
            cleaned = ["Mid-Range", "Best Seller"]
        else:
            cleaned = ["Premium", "Best Seller"]

    return "|".join(cleaned)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _value_tier_from_annual_value(annual_value: float) -> str:
    if annual_value >= SAVE_VALUE_HIGH_THRESHOLD:
        return "High Value"
    if annual_value >= SAVE_VALUE_MEDIUM_THRESHOLD:
        return "Medium Value"
    return "Low Value"


def _build_churn_scores(rfm_df: pd.DataFrame) -> dict[int, float]:
    if rfm_df.empty:
        return {}

    recency = rfm_df["recency_days"].astype(float)
    frequency = rfm_df["frequency"].astype(float)
    monetary = rfm_df["monetary"].astype(float)

    recency_z = (recency - recency.mean()) / max(recency.std(ddof=0), 1e-6)
    frequency_z = (frequency - frequency.mean()) / max(frequency.std(ddof=0), 1e-6)
    monetary_z = (monetary - monetary.mean()) / max(monetary.std(ddof=0), 1e-6)

    scores: dict[int, float] = {}
    for i, row in rfm_df.reset_index(drop=True).iterrows():
        segment = str(row.get("rfm_segment", ""))
        linear_score = (
            1.7 * float(recency_z.iloc[i])
            - 1.2 * float(frequency_z.iloc[i])
            - 0.9 * float(monetary_z.iloc[i])
            + SEGMENT_RISK_BIAS.get(segment, 0.0)
        )
        scores[int(row["customer_id"])] = round(_sigmoid(linear_score), 4)

    return scores


def _infer_lifecycle_stage(
    has_orders: bool,
    signup_dt: pd.Timestamp | None,
    new_cutoff: pd.Timestamp,
    rfm_segment: str,
    churn_score: float,
) -> str:
    if churn_score >= 0.62 or rfm_segment in RISK_SEGMENTS:
        return "Churn Risk"
    if rfm_segment == "Champions":
        return "Champions"
    if rfm_segment in {"Loyal", "Potential Loyalist"}:
        return "Loyal Customers"
    if not has_orders:
        return "Potential Customer"
    if signup_dt is not None and signup_dt >= new_cutoff:
        return "New"
    return "New"


def _build_marketing_cache(
    customers_df: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    fs: FeatureStore,
    churn_scores: dict[int, float],
) -> tuple[dict, dict[int, str]]:
    customer_order_counts = orders.groupby("customer_id").size().to_dict()
    total_customers = int(customers_df["customer_id"].nunique())
    paying_customers = int(orders["customer_id"].nunique())
    total_orders = int(orders["order_id"].nunique())
    total_revenue = float((order_items["price"] * order_items["quantity"]).sum())

    parsed_signup = pd.to_datetime(customers_df["signup_date"], errors="coerce")
    latest_signup = parsed_signup.max()
    if pd.isna(latest_signup):
        latest_signup = pd.Timestamp.utcnow().normalize()
    new_cutoff = latest_signup - pd.Timedelta(days=90)

    lifecycle_counter: Counter = Counter()
    lifecycle_map: dict[int, str] = {}

    for _, row in customers_df.iterrows():
        cid = int(row["customer_id"])
        signup_dt = pd.to_datetime(row.get("signup_date"), errors="coerce")
        if pd.isna(signup_dt):
            signup_dt = None

        has_orders = customer_order_counts.get(cid, 0) > 0
        segment = fs.get_rfm_segment(cid)
        churn_score = churn_scores.get(cid, 0.5)

        stage = _infer_lifecycle_stage(
            has_orders=has_orders,
            signup_dt=signup_dt,
            new_cutoff=new_cutoff,
            rfm_segment=segment,
            churn_score=churn_score,
        )
        lifecycle_map[cid] = stage
        lifecycle_counter[stage] += 1

    customers_with_tags = sum(1 for cid in customers_df["customer_id"] if len(fs.get_tags(int(cid))) > 0)

    repeat_customer_count = sum(1 for v in customer_order_counts.values() if int(v) >= 2)
    customer_revenue_map = (
        order_items.merge(
            orders[["order_id", "customer_id"]],
            on="order_id",
            how="left",
        )
        .assign(line_total=lambda d: d["price"] * d["quantity"])
        .groupby("customer_id")["line_total"]
        .sum()
        .to_dict()
    )
    high_risk_customers = [
        {
            "customer_id": int(cid),
            "score": score,
            "segment": fs.get_rfm_segment(int(cid)),
            "name": customers_data.get(int(cid), {}).get("name", f"Customer {cid}"),
            "annual_value": round(float(customer_revenue_map.get(int(cid), 0.0)), 2),
            "value_tier": _value_tier_from_annual_value(float(customer_revenue_map.get(int(cid), 0.0))),
        }
        for cid, score in churn_scores.items()
        if score >= 0.62
    ]
    high_risk_customers.sort(key=lambda x: x["score"], reverse=True)

    discounted_pids = set(
        products.loc[
            products["tags"].astype(str).str.contains("Discounted", case=False, na=False),
            "product_id",
        ].astype(int)
    )
    item_revenue = order_items.copy()
    item_revenue["line_total"] = item_revenue["price"] * item_revenue["quantity"]
    promo_items = item_revenue[item_revenue["product_id"].isin(discounted_pids)]
    promo_order_ids = set(promo_items["order_id"].astype(int).tolist())

    order_totals = item_revenue.groupby("order_id")["line_total"].sum()
    promo_order_totals = order_totals[order_totals.index.isin(promo_order_ids)]
    non_promo_order_totals = order_totals[~order_totals.index.isin(promo_order_ids)]

    promo_aov = float(promo_order_totals.mean()) if len(promo_order_totals) else 0.0
    non_promo_aov = float(non_promo_order_totals.mean()) if len(non_promo_order_totals) else 0.0
    promo_lift_pct = (
        ((promo_aov - non_promo_aov) / non_promo_aov) * 100.0
        if non_promo_aov > 0
        else 0.0
    )

    marketing_cache = {
        "lifecycle": [
            {"stage": stage, "count": int(lifecycle_counter.get(stage, 0))}
            for stage in LIFECYCLE_STAGES
        ],
        "focus_segments": [
            {
                "segment": "Potential Customer",
                "focus": "Improve acquisition and first purchase conversion",
                "playbook": "Sharper landing pages, proof points, and one-click onboarding",
            },
            {
                "segment": "New",
                "focus": "Improve activation and time-to-value",
                "playbook": "Onboarding checklist and first-week nudges",
            },
            {
                "segment": "Loyal Customers",
                "focus": "Increase ARPU and basket size",
                "playbook": "Bundles, cross-sell, and personalized product sets",
            },
            {
                "segment": "Champions",
                "focus": "Drive referrals and advocacy",
                "playbook": "Referral rewards and social proof campaigns",
            },
            {
                "segment": "Churn Risk",
                "focus": "Reduce churn with statistical risk targeting",
                "playbook": "Priority outreach + save offers for highest risk scores",
            },
        ],
        "kpis": {
            "sales_conversion_pct": round((paying_customers / max(total_customers, 1)) * 100.0, 2),
            "arpu": round(total_revenue / max(paying_customers, 1), 2),
            "aov": round(total_revenue / max(total_orders, 1), 2),
            "new_customers_90d": int((parsed_signup >= new_cutoff).sum()),
            "repeat_purchase_rate_pct": round((repeat_customer_count / max(paying_customers, 1)) * 100.0, 2),
            "churn_risk_count": len(high_risk_customers),
            "churn_risk_rate_pct": round((len(high_risk_customers) / max(paying_customers, 1)) * 100.0, 2),
            "experience_coverage_pct": round((customers_with_tags / max(total_customers, 1)) * 100.0, 2),
            "promo_revenue_share_pct": round((promo_items["line_total"].sum() / max(total_revenue, 1.0)) * 100.0, 2),
            "promo_order_share_pct": round((len(promo_order_ids) / max(total_orders, 1)) * 100.0, 2),
            "promo_aov_lift_pct": round(promo_lift_pct, 2),
        },
        "targets": {
            "sales_conversion_pct": 20,
            "arpu_growth_pct": 12,
            "new_customers_growth_pct": 25,
            "churn_reduction_pct": 20,
            "experience_coverage_pct": 85,
            "promo_incremental_lift_pct": 15,
        },
        "churn_model": {
            "average_score": round(float(np.mean(list(churn_scores.values()))) if churn_scores else 0.0, 4),
            "high_risk_customers": high_risk_customers[:15],
        },
    }

    return marketing_cache, lifecycle_map


def _build_weekly_arpu_cache(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    marketing_cache: dict,
) -> dict:
    order_cols = orders[["order_id", "customer_id", "order_date"]].copy()
    order_cols["order_date"] = pd.to_datetime(order_cols["order_date"], errors="coerce")
    order_cols = order_cols.dropna(subset=["order_date"])

    if order_cols.empty:
        return {
            "metric": "weekly_arpu",
            "threshold": 0.0,
            "latest_value": 0.0,
            "below_threshold": False,
            "gap_pct": 0.0,
            "weeks": [],
            "root_causes": [],
            "suggested_actions": [],
        }

    line_items = order_items.merge(order_cols, on="order_id", how="inner")
    line_items["line_total"] = line_items["price"] * line_items["quantity"]
    line_items["week_start"] = (
        line_items["order_date"]
        .dt.to_period("W-MON")
        .apply(lambda p: p.start_time.normalize())
    )

    weekly = (
        line_items.groupby("week_start")
        .agg(
            revenue=("line_total", "sum"),
            active_customers=("customer_id", "nunique"),
            orders=("order_id", "nunique"),
        )
        .reset_index()
        .sort_values("week_start")
    )
    weekly["arpu"] = weekly["revenue"] / weekly["active_customers"].clip(lower=1)
    weekly = weekly.tail(ARPU_TREND_WEEKS).copy()

    weekly_baseline = float(weekly["arpu"].mean()) if not weekly.empty else 0.0
    global_arpu = float(marketing_cache.get("kpis", {}).get("arpu", weekly_baseline))
    if weekly_baseline > 0 and (global_arpu > weekly_baseline * 2.0 or global_arpu < weekly_baseline * 0.5):
        baseline_arpu = weekly_baseline
    else:
        baseline_arpu = global_arpu
    threshold = round(max(1.0, baseline_arpu * ARPU_THRESHOLD_FACTOR), 2)
    latest_arpu = round(float(weekly["arpu"].iloc[-1]), 2) if not weekly.empty else 0.0
    below_threshold = latest_arpu < threshold
    gap_pct = (
        round(((latest_arpu - threshold) / max(threshold, 1e-6)) * 100.0, 2)
        if threshold > 0
        else 0.0
    )

    kpis = marketing_cache.get("kpis", {})
    root_causes: list[str] = []
    if below_threshold:
        root_causes.append("Weekly ARPU dropped below target threshold.")
    if float(kpis.get("repeat_purchase_rate_pct", 0.0)) < 55:
        root_causes.append("Repeat purchase rate is soft; basket expansion opportunities are underused.")
    if float(kpis.get("churn_risk_rate_pct", 0.0)) > 20:
        root_causes.append("High churn-risk segment share is suppressing monetization.")
    if float(kpis.get("promo_aov_lift_pct", 0.0)) < 5:
        root_causes.append("Promo AOV lift is low; existing offers are not improving basket value enough.")
    if not root_causes:
        root_causes.append("Monitor closely; ARPU is near expected range.")

    suggested_actions = [
        {
            "id": "bundle-loyal-customers",
            "phase": "Plan",
            "title": "Launch bundle offers for Loyal Customers",
            "description": "Create 2-3 high-affinity bundles in top categories and push to repeat buyers.",
            "expected_arpu_lift_pct": 4.5,
            "owner": "Marketing",
            "requires_approval": True,
        },
        {
            "id": "upsell-champions",
            "phase": "Plan",
            "title": "Premium upsell journey for Champions",
            "description": "Serve premium alternatives and add-on accessories with targeted messaging.",
            "expected_arpu_lift_pct": 3.2,
            "owner": "Recommendations Agent",
            "requires_approval": True,
        },
        {
            "id": "save-at-risk",
            "phase": "Execute",
            "title": "Retention save flow for Churn Risk users",
            "description": "Trigger time-boxed save offers and monitor weekly conversion delta.",
            "expected_arpu_lift_pct": 2.7,
            "owner": "Lifecycle Marketing",
            "requires_approval": True,
        },
    ]

    week_points = [
        {
            "week_start": str(row.week_start.date()),
            "arpu": round(float(row.arpu), 2),
            "threshold": threshold,
            "revenue": round(float(row.revenue), 2),
            "active_customers": int(row.active_customers),
            "orders": int(row.orders),
        }
        for row in weekly.itertuples(index=False)
    ]

    return {
        "metric": "weekly_arpu",
        "threshold": threshold,
        "latest_value": latest_arpu,
        "below_threshold": below_threshold,
        "gap_pct": gap_pct,
        "weeks": week_points,
        "root_causes": root_causes,
        "suggested_actions": suggested_actions,
    }


def _build_arpu_workflow_cache(arpu_cache: dict) -> dict:
    phases = [
        {
            "id": "diagnose",
            "name": "Diagnose",
            "human_actions": ["Check dashboard", "Identify gap"],
            "agent_actions": ["Segment analysis", "Find root causes"],
        },
        {
            "id": "plan",
            "name": "Plan",
            "human_actions": ["Approve actions", "Set rules"],
            "agent_actions": ["Suggest bundles", "Suggest upsell"],
        },
        {
            "id": "execute",
            "name": "Execute",
            "human_actions": ["Monitor performance", "Validate rollout"],
            "agent_actions": ["Run campaigns", "Track and adjust"],
        },
        {
            "id": "optimize",
            "name": "Optimize",
            "human_actions": ["Review results", "Scale decisions"],
            "agent_actions": ["Track recovery", "Suggest next steps"],
        },
    ]

    below = bool(arpu_cache.get("below_threshold", False))
    threshold = float(arpu_cache.get("threshold", 0.0))
    latest = float(arpu_cache.get("latest_value", 0.0))

    return {
        "triggered": below,
        "status": "triggered" if below else "monitoring",
        "trigger_metric": "weekly_arpu",
        "latest_arpu": latest,
        "threshold": threshold,
        "trigger_reason": (
            f"Weekly ARPU {latest:.2f} is below threshold {threshold:.2f}."
            if below
            else f"Weekly ARPU {latest:.2f} is healthy against threshold {threshold:.2f}."
        ),
        "approval_required": True,
        "phases": phases,
        "recommendations": arpu_cache.get("suggested_actions", []),
    }


def _build_save_customer_cache(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    churn_scores: dict[int, float],
    fs: FeatureStore,
    rfm_df: pd.DataFrame,
) -> dict:
    order_dates = orders[["customer_id", "order_date"]].copy()
    order_dates["order_date"] = pd.to_datetime(order_dates["order_date"], errors="coerce")
    order_dates = order_dates.dropna(subset=["order_date"]).sort_values(["customer_id", "order_date"])

    latest_order_ts = order_dates["order_date"].max()
    if pd.isna(latest_order_ts):
        latest_order_ts = pd.Timestamp.utcnow().normalize()

    customer_revenue = (
        order_items.merge(
            orders[["order_id", "customer_id"]],
            on="order_id",
            how="left",
        )
        .assign(line_total=lambda d: d["price"] * d["quantity"])
        .groupby("customer_id")["line_total"]
        .sum()
        .to_dict()
    )

    recency_map = {
        int(row.customer_id): float(row.recency_days)
        for row in rfm_df.itertuples(index=False)
    }

    gap_map: dict[int, tuple[float, float]] = {}
    for cid, grp in order_dates.groupby("customer_id"):
        unique_dates = pd.Series(grp["order_date"].drop_duplicates().tolist())
        if unique_dates.empty:
            gap_map[int(cid)] = (0.0, 0.0)
            continue
        if len(unique_dates) == 1:
            last_gap = float((latest_order_ts - unique_dates.iloc[0]).days)
            gap_map[int(cid)] = (last_gap, 30.0)
            continue

        gap_days = unique_dates.diff().dt.days.dropna().astype(float)
        last_gap = float(gap_days.iloc[-1]) if not gap_days.empty else 0.0
        prev_gap = float(gap_days.iloc[:-1].mean()) if len(gap_days) > 1 else 30.0
        gap_map[int(cid)] = (last_gap, prev_gap)

    high_risk_customers: list[dict] = []
    for cid, score in churn_scores.items():
        if score < 0.62:
            continue
        customer_id = int(cid)
        annual_value = float(customer_revenue.get(customer_id, 0.0))
        value_tier = _value_tier_from_annual_value(annual_value)
        last_gap, prev_gap = gap_map.get(customer_id, (0.0, 0.0))
        recency_days = float(recency_map.get(customer_id, 0.0))
        tags_lower = " ".join(fs.get_tags(customer_id)).lower()

        high_risk_customers.append(
            {
                "customer_id": customer_id,
                "name": customers_data.get(customer_id, {}).get("name", f"Customer {customer_id}"),
                "score": round(float(score), 4),
                "segment": fs.get_rfm_segment(customer_id),
                "annual_value": round(annual_value, 2),
                "value_tier": value_tier,
                "warning_low_email_engagement": recency_days >= 60,
                "warning_purchase_gap_expanded": last_gap >= 90 and prev_gap <= 45,
                "warning_price_complaint_proxy": any(k in tags_lower for k in ("budget", "discount", "price")),
                "last_gap_days": int(round(last_gap)),
                "previous_gap_days": int(round(prev_gap)),
            }
        )

    high_risk_customers.sort(key=lambda x: (x["score"], x["annual_value"]), reverse=True)
    tier_counter = Counter([c["value_tier"] for c in high_risk_customers])

    high_risk_count = len(high_risk_customers)
    monthly_threshold = max(50, int(round(len(churn_scores) * SAVE_MONTHLY_THRESHOLD_RATIO)))
    threshold_gap = max(0, high_risk_count - monthly_threshold)
    threshold_gap_pct = round((threshold_gap / max(monthly_threshold, 1)) * 100.0, 2)
    triggered = high_risk_count > monthly_threshold

    low_email_count = sum(1 for c in high_risk_customers if c["warning_low_email_engagement"])
    purchase_gap_count = sum(1 for c in high_risk_customers if c["warning_purchase_gap_expanded"])
    price_signal_count = sum(1 for c in high_risk_customers if c["warning_price_complaint_proxy"])

    expected_save_rate = 18.0
    expected_redemption_rate = 12.0
    expected_saved_customers = int(round(high_risk_count * (expected_save_rate / 100.0)))
    expected_discount_cost = round(expected_saved_customers * 83.33, 2)
    expected_retained_value = round(
        float(sum(c["annual_value"] for c in high_risk_customers)) * (expected_save_rate / 100.0),
        2,
    )
    expected_roi = round(
        expected_retained_value / max(expected_discount_cost, 1.0),
        2,
    )

    return {
        "alert": {
            "triggered": triggered,
            "status": "triggered" if triggered else "monitoring",
            "high_risk_count": high_risk_count,
            "monthly_threshold": monthly_threshold,
            "threshold_gap": threshold_gap,
            "threshold_gap_pct": threshold_gap_pct,
        },
        "value_tiers": [
            {
                "tier": "High Value",
                "criteria": "$500+ annual spend",
                "count": int(tier_counter.get("High Value", 0)),
                "priority": "P1",
            },
            {
                "tier": "Medium Value",
                "criteria": "$200-$499 annual spend",
                "count": int(tier_counter.get("Medium Value", 0)),
                "priority": "P2",
            },
            {
                "tier": "Low Value",
                "criteria": "<$200 annual spend",
                "count": int(tier_counter.get("Low Value", 0)),
                "priority": "P3",
            },
        ],
        "warning_signs": {
            "low_email_engagement_customers": low_email_count,
            "purchase_gap_expanded_customers": purchase_gap_count,
            "price_complaint_signal_customers": price_signal_count,
        },
        "intervention_plan": [
            {
                "tier": "High Value",
                "strategy": "Personal outreach calls + 20% loyalty discount",
                "owner": "Service Team + Customer Success",
                "requires_human_approval": True,
            },
            {
                "tier": "Medium Value",
                "strategy": "Win-back email sequence + 15% offer",
                "owner": "Retention Team",
                "requires_human_approval": True,
            },
            {
                "tier": "Low Value",
                "strategy": "Standard re-engagement campaign",
                "owner": "Lifecycle Marketing",
                "requires_human_approval": False,
            },
        ],
        "campaign_controls": {
            "escalation_days": SAVE_ESCALATION_DAYS,
            "assigned_high_value_accounts": min(50, int(tier_counter.get("High Value", 0))),
            "human_follow_up_team": "Customer Success",
        },
        "performance_snapshot": {
            "expected_redemption_rate_pct": expected_redemption_rate,
            "expected_save_rate_pct": expected_save_rate,
            "expected_saved_customers": expected_saved_customers,
        },
        "outcome_benchmark": {
            "offer_cost": expected_discount_cost,
            "retained_annual_value": expected_retained_value,
            "roi_multiple": expected_roi,
        },
        "priority_accounts": high_risk_customers[:20],
    }


def _build_save_customer_workflow_cache(save_cache: dict) -> dict:
    alert = save_cache.get("alert", {})
    triggered = bool(alert.get("triggered", False))
    high_risk_count = int(alert.get("high_risk_count", 0))
    threshold = int(alert.get("monthly_threshold", 0))

    phases = [
        {
            "id": "find-risk",
            "name": "Find Who's At Risk",
            "human_actions": [
                "Retention Team starts churn rescue workflow",
                "Customer Success validates sampled high-risk accounts",
            ],
            "agent_actions": [
                "Churn Risk Scoring Agent scores all customers",
                "Agent segments customers by value tier and warning signs",
            ],
        },
        {
            "id": "design-rescue",
            "name": "Design the Rescue Plan",
            "human_actions": [
                "Retention Manager approves or rejects each intervention",
                "Service Team assigns top high-value accounts",
            ],
            "agent_actions": [
                "Intervention Design Agent creates tiered playbook",
                "Agent adapts strategy based on manager decisions",
            ],
        },
        {
            "id": "rescue-mission",
            "name": "Run the Rescue Mission",
            "human_actions": [
                "CRM Team monitors daily performance",
                "Customer Success handles escalated non-responders",
            ],
            "agent_actions": [
                "Save Campaign Agent launches approved interventions",
                "Agent escalates non-responders after 7 days",
            ],
        },
        {
            "id": "count-saves",
            "name": "Count the Saves",
            "human_actions": [
                "Retention Manager reviews 30-day outcomes",
                "Leadership decides whether to standardize playbook",
            ],
            "agent_actions": [
                "Churn Measurement Agent calculates save rate and ROI",
                "Agent reports channel-level impact",
            ],
        },
    ]

    approvals = [
        {
            "id": "validate-risk-signals",
            "phase_id": "find-risk",
            "title": "Validate at-risk findings",
            "description": "Customer Success must validate a sample of flagged accounts before planning starts.",
            "owner": "Customer Success Lead",
            "required_decision_from": "Human",
        },
        {
            "id": "approve-high-value-playbook",
            "phase_id": "design-rescue",
            "title": "Approve high-value rescue playbook",
            "description": "Retention Manager approves personal outreach + 20% loyalty discount for high-value customers.",
            "owner": "Retention Manager",
            "required_decision_from": "Human",
        },
        {
            "id": "approve-discount-guardrails",
            "phase_id": "design-rescue",
            "title": "Set discount guardrails",
            "description": "Manager confirms no blanket discounts for all tiers.",
            "owner": "Retention Manager",
            "required_decision_from": "Human",
        },
        {
            "id": "confirm-high-value-assignment",
            "phase_id": "rescue-mission",
            "title": "Confirm account assignment",
            "description": "Service Team Lead confirms high-value accounts are assigned for personal outreach.",
            "owner": "Service Team Lead",
            "required_decision_from": "Human",
        },
        {
            "id": "approve-standardization",
            "phase_id": "count-saves",
            "title": "Approve playbook standardization",
            "description": "Leadership approves standard rollout based on 30-day save performance and ROI.",
            "owner": "Leadership",
            "required_decision_from": "Human",
        },
    ]

    return {
        "triggered": triggered,
        "status": "triggered" if triggered else "monitoring",
        "trigger_metric": "high_risk_customers",
        "high_risk_count": high_risk_count,
        "monthly_threshold": threshold,
        "trigger_reason": (
            f"High-risk customers ({high_risk_count}) exceeded monthly threshold ({threshold})."
            if triggered
            else f"High-risk customers ({high_risk_count}) are within threshold ({threshold})."
        ),
        "approval_required": True,
        "phases": phases,
        "approvals": approvals,
        "interventions": save_cache.get("intervention_plan", []),
        "escalation_days": int(save_cache.get("campaign_controls", {}).get("escalation_days", SAVE_ESCALATION_DAYS)),
        "expected_30_day_outcome": save_cache.get("performance_snapshot", {}),
        "roi_projection": save_cache.get("outcome_benchmark", {}),
    }


def build_customer_profiles(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    browsing: pd.DataFrame,
) -> dict:
    """Build a rich profile dict for each customer from their history."""
    profiles: dict = {}

    # rename products price to avoid collision with order_items.price
    prod_cols = products[["product_id", "category", "brand", "price"]].rename(
        columns={"price": "product_price"}
    )

    # merge order_items with product info
    items_with_meta = order_items.merge(prod_cols, on="product_id", how="left")
    # merge with orders to get customer_id and order_date
    items_full = items_with_meta.merge(
        orders[["order_id", "customer_id", "order_date"]],
        on="order_id", how="left",
    )

    for cid, grp in items_full.groupby("customer_id"):
        cid = int(cid)
        total_orders = grp["order_id"].nunique()
        total_items = len(grp)
        total_spend = float((grp["price"] * grp["quantity"]).sum())

        # top categories
        cat_counts = grp.groupby("category")["quantity"].sum().sort_values(ascending=False)
        top_categories = [
            {"name": cat, "count": int(cnt)}
            for cat, cnt in cat_counts.head(5).items()
        ]

        # top brands
        brand_counts = grp.groupby("brand")["quantity"].sum().sort_values(ascending=False)
        top_brands = [
            {"name": brand, "count": int(cnt)}
            for brand, cnt in brand_counts.head(5).items()
        ]

        # price range
        prices = grp["price"].dropna()
        avg_price = float(prices.mean()) if len(prices) > 0 else 0
        min_price = float(prices.min()) if len(prices) > 0 else 0
        max_price = float(prices.max()) if len(prices) > 0 else 0

        # recent orders (last 5)
        recent = grp.sort_values("order_date", ascending=False)
        seen_orders = set()
        recent_orders = []
        for _, row in recent.iterrows():
            oid = int(row["order_id"])
            if oid in seen_orders:
                continue
            seen_orders.add(oid)
            recent_orders.append({
                "order_id": oid,
                "date": str(row["order_date"])[:10],
                "product": str(row.get("product_id", "?")),
                "category": str(row.get("category", "?")),
                "brand": str(row.get("brand", "?")),
            })
            if len(recent_orders) >= 5:
                break

        profiles[cid] = {
            "total_orders": total_orders,
            "total_items": total_items,
            "total_spend": round(total_spend, 2),
            "avg_order_value": round(total_spend / max(total_orders, 1), 2),
            "avg_item_price": round(avg_price, 2),
            "price_range": {"min": round(min_price, 2), "max": round(max_price, 2)},
            "top_categories": top_categories,
            "top_brands": top_brands,
            "recent_orders": recent_orders,
        }

    # also add browsing stats
    for cid, grp in browsing.groupby("customer_id"):
        cid = int(cid)
        if cid not in profiles:
            profiles[cid] = {
                "total_orders": 0, "total_items": 0, "total_spend": 0,
                "avg_order_value": 0, "avg_item_price": 0,
                "price_range": {"min": 0, "max": 0},
                "top_categories": [], "top_brands": [], "recent_orders": [],
            }
        browse_products = grp.merge(
            products[["product_id", "category"]],
            on="product_id", how="left",
        )
        cat_views = browse_products["category"].value_counts().head(5)
        profiles[cid]["browsed_categories"] = [
            {"name": cat, "views": int(cnt)}
            for cat, cnt in cat_views.items()
        ]
        profiles[cid]["total_product_views"] = int(len(grp))

    return profiles


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_models():
    global pipeline, cold_start_handler, product_metadata_global, customer_profiles
    global customers_data, analytics_cache, churn_score_cache, lifecycle_stage_cache

    product_metadata_global.clear()
    customer_profiles.clear()
    customers_data.clear()
    analytics_cache.clear()
    churn_score_cache.clear()
    lifecycle_stage_cache.clear()

    # load encoders
    with open(ARTIFACT_DIR / "encoders.json") as f:
        encoders = json.load(f)

    cat2id = encoders["cat2id"]
    brand2id = encoders["brand2id"]
    all_ptags = encoders["all_ptags"]

    n_categories = len(cat2id)
    n_brands = len(brand2id)
    n_product_tags = len(all_ptags)

    # feature store
    rfm_df = pd.read_csv(ARTIFACT_DIR / "rfm_scores.csv")
    with open(ARTIFACT_DIR / "tags.json") as f:
        tags_raw = json.load(f)
    tags = {int(k): v for k, v in tags_raw.items()}

    fs = FeatureStore()
    fs.load_rfm(rfm_df)
    fs.load_tags(tags)

    # model
    user_feature_dim = fs.total_user_feature_dim
    model = TwoTowerModel(
        user_feature_dim=user_feature_dim,
        n_categories=n_categories,
        n_brands=n_brands,
        n_product_tags=n_product_tags,
        embed_dim=64,
    )
    model.load_state_dict(torch.load(ARTIFACT_DIR / "two_tower.pt", map_location="cpu"))
    model.eval()

    # retriever
    retriever = Retriever(embed_dim=64)
    retriever.load(str(ARTIFACT_DIR / "faiss"))

    # reranker
    reranker = ReRanker()
    reranker.load(str(ARTIFACT_DIR / "reranker.lgb"))

    # product features
    with open(ARTIFACT_DIR / "product_metadata.json") as f:
        product_metadata_global.update({int(k): v for k, v in json.load(f).items()})

    # reconcile product metadata with generated dataset (better names + guaranteed tags)
    products = pd.read_csv(DATA_DIR / "products.csv")
    products_idx = products.set_index("product_id").to_dict(orient="index")
    for pid, meta in product_metadata_global.items():
        row = products_idx.get(pid, {})
        candidate_name = str(row.get("name", "")).strip()
        candidate_category = str(row.get("category", "")).strip()
        candidate_brand = str(row.get("brand", "")).strip()

        if candidate_name:
            meta["name"] = candidate_name
        if candidate_category:
            meta["category"] = candidate_category
        if candidate_brand:
            meta["brand"] = candidate_brand

        meta["price"] = round(_safe_float(row.get("price"), _safe_float(meta.get("price"), 0.0)), 2)
        meta["tags"] = _normalize_product_tags(row.get("tags", meta.get("tags", "")), meta["price"])

    # reconstruct product_features dict
    ptag2idx = {t: i for i, t in enumerate(all_ptags)}
    max_price = max(m["price"] for m in product_metadata_global.values())
    product_features = {}
    for pid, meta in product_metadata_global.items():
        tag_vec = np.zeros(n_product_tags, dtype=np.float32)
        for t in meta["tags"].split("|"):
            t = t.strip()
            if t in ptag2idx:
                tag_vec[ptag2idx[t]] = 1.0
        product_features[pid] = {
            "category_id": cat2id.get(meta["category"], 0),
            "brand_id": brand2id.get(meta["brand"], 0),
            "price_norm": meta["price"] / max_price,
            "tag_vec": tag_vec,
        }

    # pipeline
    pipeline = RecommendationPipeline(
        model=model,
        retriever=retriever,
        reranker=reranker,
        feature_store=fs,
        product_features=product_features,
        product_metadata=product_metadata_global,
    )

    cold_start_handler = UserColdStartHandler(fs)

    # build customer profiles from raw order data
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    order_items = pd.read_csv(DATA_DIR / "order_items.csv")
    browsing = pd.read_csv(DATA_DIR / "browsing_history.csv")
    customer_profiles.update(
        build_customer_profiles(orders, order_items, products, browsing)
    )

    # load customers.csv into customers_data
    customers_df = pd.read_csv(DATA_DIR / "customers.csv")
    customers_data.update({
        int(row.customer_id): {
            "name": str(row["name"]),
            "location": str(row["location"]),
            "signup_date": str(row["signup_date"]),
        }
        for _, row in customers_df.iterrows()
    })

    # precompute analytics cache
    total_revenue = float((order_items["price"] * order_items["quantity"]).sum())

    segment_counter: Counter = Counter()
    tag_counter: Counter = Counter()
    for cid in customer_profiles:
        seg = fs.get_rfm_segment(cid)
        segment_counter[seg] += 1
        for t in fs.get_tags(cid):
            tag_counter[t] += 1

    churn_score_cache.update(_build_churn_scores(rfm_df))

    items_with_products = order_items.merge(
        products[["product_id", "category", "brand"]], on="product_id", how="left"
    )
    items_with_products["revenue"] = items_with_products["price"] * items_with_products["quantity"]

    cat_revenue = (
        items_with_products.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    brand_revenue = (
        items_with_products.groupby("brand")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    marketing_cache, lifecycle_map = _build_marketing_cache(
        customers_df=customers_df,
        orders=orders,
        order_items=order_items,
        products=products,
        fs=fs,
        churn_scores=churn_score_cache,
    )
    arpu_weekly_cache = _build_weekly_arpu_cache(
        orders=orders,
        order_items=order_items,
        marketing_cache=marketing_cache,
    )
    arpu_workflow_cache = _build_arpu_workflow_cache(arpu_weekly_cache)
    save_customer_cache = _build_save_customer_cache(
        orders=orders,
        order_items=order_items,
        churn_scores=churn_score_cache,
        fs=fs,
        rfm_df=rfm_df,
    )
    save_customer_workflow_cache = _build_save_customer_workflow_cache(save_customer_cache)
    lifecycle_stage_cache.update(lifecycle_map)

    analytics_cache.update({
        "summary": {
            "total_customers": int(customers_df["customer_id"].nunique()),
            "total_products": len(product_metadata_global),
            "total_orders": int(orders["order_id"].nunique()),
            "total_revenue": round(total_revenue, 2),
        },
        "segments": [
            {"segment": seg, "count": cnt}
            for seg, cnt in segment_counter.most_common()
        ],
        "tags": [
            {"tag": tag, "count": cnt}
            for tag, cnt in tag_counter.most_common()
        ],
        "top-categories": [
            {"category": cat, "revenue": round(float(rev), 2)}
            for cat, rev in cat_revenue.items()
        ],
        "top-brands": [
            {"brand": brand, "revenue": round(float(rev), 2)}
            for brand, rev in brand_revenue.items()
        ],
        "arpu-weekly": arpu_weekly_cache,
        "arpu-workflow": arpu_workflow_cache,
        "marketing": marketing_cache,
        "churn-risk": marketing_cache["churn_model"]["high_risk_customers"],
        "save-customer": save_customer_cache,
        "save-customer-workflow": save_customer_workflow_cache,
    })

    print(f"[OK] All models loaded, pipeline ready, {len(customer_profiles)} customer profiles built.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{customer_id}")
def recommend(customer_id: int, top_k: int = Query(20, ge=1, le=100)):
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Models not loaded"})

    # get customer profile first (needed for history-aware explanations)
    profile = customer_profiles.get(customer_id, {
        "total_orders": 0, "total_items": 0, "total_spend": 0,
        "avg_order_value": 0, "avg_item_price": 0,
        "price_range": {"min": 0, "max": 0},
        "top_categories": [], "top_brands": [],
        "browsed_categories": [], "total_product_views": 0,
        "recent_orders": [],
    })

    result = pipeline.recommend(
        customer_id=customer_id,
        top_k=top_k,
        customer_profile=profile,
    )

    return {
        "customer_id": result.customer_id,
        "rfm_segment": result.rfm_segment,
        "behavioral_tags": result.behavioral_tags,
        "profile": profile,
        "recommendations": result.recommendations,
    }


class EventPayload(BaseModel):
    customer_id: int
    product_id: int
    action: str
    product_price: float = 0.0
    product_category: str = ""
    product_brand: str = ""


@app.post("/event")
def ingest_event(payload: EventPayload):
    if cold_start_handler is None:
        return JSONResponse(status_code=503, content={"error": "Not ready"})

    tags = cold_start_handler.record_interaction(
        customer_id=payload.customer_id,
        product_id=payload.product_id,
        action=payload.action,
        product_price=payload.product_price,
        product_category=payload.product_category,
        product_brand=payload.product_brand,
    )
    return {"customer_id": payload.customer_id, "updated_tags": tags}


@app.get("/products")
def list_products(
    category: str | None = Query(None, description="Filter by category"),
    brand: str | None = Query(None, description="Filter by brand"),
    search: str | None = Query(None, description="Search by product name"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """Paginated product listing with optional category, brand, and name filters."""
    items = list(product_metadata_global.items())

    if category:
        items = [
            (pid, m) for pid, m in items
            if m["category"].lower() == category.lower()
        ]
    if brand:
        items = [
            (pid, m) for pid, m in items
            if m["brand"].lower() == brand.lower()
        ]
    if search:
        search_lower = search.lower()
        items = [
            (pid, m) for pid, m in items
            if search_lower in m["name"].lower()
        ]

    total = len(items)
    start = (page - 1) * limit
    page_items = items[start : start + limit]

    return {
        "items": [
            {
                "product_id": pid,
                "name": m["name"],
                "category": m["category"],
                "brand": m["brand"],
                "price": m["price"],
                "tags": m["tags"],
            }
            for pid, m in page_items
        ],
        "total": total,
        "page": page,
        "limit": limit,
    }


@app.get("/customers")
def list_customers(
    segment: str | None = Query(None, description="Filter by RFM segment"),
    tag: str | None = Query(None, description="Filter by behavioral tag"),
    search: str | None = Query(None, description="Search by name or customer ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """Paginated customer listing enriched with RFM segment, tags, and order stats."""
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Models not loaded"})

    enriched = []
    for cid, info in customers_data.items():
        rfm_segment = pipeline.feature_store.get_rfm_segment(cid)
        behavioral_tags = pipeline.feature_store.get_tags(cid)
        profile = customer_profiles.get(cid, {})

        enriched.append({
            "customer_id": cid,
            "name": info["name"],
            "location": info["location"],
            "signup_date": info["signup_date"],
            "rfm_segment": rfm_segment,
            "behavioral_tags": behavioral_tags,
            "total_orders": profile.get("total_orders", 0),
            "total_spend": profile.get("total_spend", 0),
        })

    # apply filters
    if segment:
        segment_lower = segment.lower()
        enriched = [c for c in enriched if c["rfm_segment"].lower() == segment_lower]
    if tag:
        tag_lower = tag.lower()
        enriched = [
            c for c in enriched
            if any(tag_lower in t.lower() for t in c["behavioral_tags"])
        ]
    if search:
        search_lower = search.lower()
        enriched = [
            c for c in enriched
            if search_lower in c["name"].lower() or search_lower in str(c["customer_id"])
        ]

    total = len(enriched)
    start = (page - 1) * limit
    page_items = enriched[start : start + limit]

    return {
        "items": page_items,
        "total": total,
        "page": page,
        "limit": limit,
    }


@app.get("/analytics")
def analytics(
    type: str = Query(
        ...,
        description=(
            "Analytics type: summary, segments, tags, top-categories, top-brands, "
            "marketing, churn-risk, arpu-weekly, arpu-workflow, "
            "save-customer, save-customer-workflow"
        ),
    ),
):
    """Return precomputed analytics data by type."""
    valid_types = {
        "summary",
        "segments",
        "tags",
        "top-categories",
        "top-brands",
        "marketing",
        "churn-risk",
        "arpu-weekly",
        "arpu-workflow",
        "save-customer",
        "save-customer-workflow",
    }
    if type not in valid_types:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid type '{type}'. Must be one of: {', '.join(sorted(valid_types))}"},
        )
    return analytics_cache.get(type, {})


# ---------------------------------------------------------------------------
# Demo UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def demo_ui():
    html_path = PROJECT_ROOT / "app" / "static" / "index.html"
    return html_path.read_text()
