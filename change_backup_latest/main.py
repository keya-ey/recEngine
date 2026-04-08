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


# ---------------------------------------------------------------------------
# Build customer profile from raw order data
# ---------------------------------------------------------------------------
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
    global customers_data, analytics_cache

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
    products = pd.read_csv(DATA_DIR / "products.csv")
    browsing = pd.read_csv(DATA_DIR / "browsing_history.csv")
    customer_profiles.update(
        build_customer_profiles(orders, order_items, products, browsing)
    )

    # load customers.csv into customers_data
    customers_df = pd.read_csv(DATA_DIR / "customers.csv")
    customers_data.update({
        int(row.customer_id): {
            "name": row.name,
            "location": row.location,
            "signup_date": row.signup_date,
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

    analytics_cache.update({
        "summary": {
            "total_customers": len(customer_profiles),
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
    type: str = Query(..., description="Analytics type: summary, segments, tags, top-categories, top-brands"),
):
    """Return precomputed analytics data by type."""
    valid_types = {"summary", "segments", "tags", "top-categories", "top-brands"}
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
