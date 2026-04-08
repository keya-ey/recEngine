"""
Synthetic E-Commerce Data Generator
====================================
Generates six realistic CSV datasets for the recommendation system:
  - customers.csv   (~1,000 rows)
  - products.csv    (~5,000 rows)
  - orders.csv      (~20,000 rows)
  - order_items.csv (~50,000 rows)
  - browsing_history.csv (~200,000 rows)
  - cart_activity.csv    (~30,000 rows)

Usage:
    python data/generate_data.py [--output-dir data/generated]
"""

import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NUM_CUSTOMERS = 1_000
NUM_PRODUCTS = 5_000
NUM_ORDERS = 20_000
NUM_ORDER_ITEMS = 50_000
NUM_BROWSING = 200_000
NUM_CART = 30_000

CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Beauty",
    "Sports & Outdoors", "Books", "Toys & Games", "Grocery",
    "Automotive", "Health & Wellness",
]

BRANDS = [
    "TechNova", "UrbanEdge", "HomeHaven", "GlowUp", "ActivePeak",
    "PageTurner", "FunZone", "FreshPick", "AutoPro", "VitaLife",
    "ElectraWave", "StyleCraft", "ComfortPlus", "PureSkin", "TrailBlaze",
    "ReadMore", "PlayStar", "NatureBite", "DriveMax", "WellnessHub",
]

FIRST_NAMES = [
    "Aarav", "Aditi", "Aisha", "Akshay", "Ananya", "Arjun", "Diya", "Isha", "Kabir", "Karan",
    "Kiara", "Meera", "Neha", "Nikhil", "Priya", "Rahul", "Riya", "Rohan", "Sana", "Vikram",
    "Ava", "Benjamin", "Charlotte", "Daniel", "Ethan", "Grace", "Hannah", "Isabella", "Jack", "Liam",
    "Lucas", "Mason", "Mia", "Noah", "Olivia", "Sophia", "William", "Zoe", "Emma", "James",
]

LAST_NAMES = [
    "Agarwal", "Bansal", "Chopra", "Desai", "Gupta", "Kapoor", "Khanna", "Mehta", "Patel", "Sharma",
    "Singh", "Verma", "Brown", "Clark", "Davis", "Garcia", "Harris", "Johnson", "Jones", "Lee",
    "Lewis", "Martin", "Miller", "Moore", "Roberts", "Smith", "Taylor", "Thomas", "Walker", "Wilson",
]

PRODUCT_TAGS = [
    "Budget", "Mid-Range", "Premium", "Eco-Friendly", "Cruelty-Free",
    "Organic", "Summer Wear", "Winter Wear", "Limited Edition",
    "Best Seller", "New Arrival", "Discounted", "Trending", "Handmade",
    "Imported", "Locally Made",
]

LOCATIONS = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Mumbai", "Delhi", "Bangalore", "London", "Toronto",
]

BROWSING_ACTIONS = ["view", "click", "search", "filter", "zoom", "compare"]
CART_ACTIONS = ["add", "remove"]

CATEGORY_TAG_HINTS = {
    "Electronics": "Trending",
    "Clothing": "Summer Wear",
    "Home & Kitchen": "Best Seller",
    "Beauty": "Cruelty-Free",
    "Sports & Outdoors": "Best Seller",
    "Books": "New Arrival",
    "Toys & Games": "Trending",
    "Grocery": "Organic",
    "Automotive": "Imported",
    "Health & Wellness": "Eco-Friendly",
}

PRODUCT_STYLE_WORDS = [
    "Essential", "Signature", "Everyday", "Select", "Prime", "Smart", "Comfort", "Classic", "Modern", "Ultra",
]

PRODUCT_SUFFIX_WORDS = ["Kit", "Set", "Collection", "Edition", "Pack", "Series"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_date(start: datetime, end: datetime, rng: np.random.Generator) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=int(rng.integers(0, int(delta))))


def _assign_tags(rng: np.random.Generator, price: float, category: str) -> str:
    """Assign 2-4 product tags and guarantee clean, non-empty tag values."""
    tags = set()
    if price < 25:
        tags.add("Budget")
    elif price < 100:
        tags.add("Mid-Range")
    else:
        tags.add("Premium")

    tags.add(CATEGORY_TAG_HINTS.get(category, "Trending"))

    extra_count = int(rng.integers(0, 3))
    extra = rng.choice(PRODUCT_TAGS, size=extra_count, replace=False)
    tags.update(extra)

    if "Discounted" not in tags and rng.random() < 0.25:
        tags.add("Discounted")

    return "|".join(sorted(tags))


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    start = datetime(2022, 1, 1)
    end = datetime(2025, 12, 31)
    rows = []
    used_names: set[str] = set()
    for cid in range(1, NUM_CUSTOMERS + 1):
        while True:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name not in used_names:
                used_names.add(name)
                break

        rows.append({
            "customer_id": cid,
            "name": name,
            "location": rng.choice(LOCATIONS),
            "signup_date": _random_date(start, end, rng).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


def generate_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for pid in range(1, NUM_PRODUCTS + 1):
        cat = rng.choice(CATEGORIES)
        # match brand to category loosely
        brand = rng.choice(BRANDS)
        price = round(float(rng.lognormal(mean=3.5, sigma=1.0)), 2)
        price = max(1.99, min(price, 2999.99))
        style = rng.choice(PRODUCT_STYLE_WORDS)
        suffix = rng.choice(PRODUCT_SUFFIX_WORDS)
        cleaned_cat = cat.replace("&", "and")
        rows.append({
            "product_id": pid,
            "name": f"{brand} {style} {cleaned_cat} {suffix}",
            "category": cat,
            "brand": brand,
            "price": price,
            "tags": _assign_tags(rng, price, cat),
        })
    return pd.DataFrame(rows)


def generate_orders(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    """Power-law customer activity: some customers order much more than others."""
    cids = customers["customer_id"].values
    # Zipf-like distribution for customer activity
    weights = 1.0 / np.arange(1, len(cids) + 1) ** 0.8
    weights /= weights.sum()

    start = datetime(2023, 1, 1)
    end = datetime(2026, 3, 1)
    rows = []
    for oid in range(1, NUM_ORDERS + 1):
        rows.append({
            "order_id": oid,
            "customer_id": int(rng.choice(cids, p=weights)),
            "order_date": _random_date(start, end, rng).strftime("%Y-%m-%d %H:%M:%S"),
            "total": 0.0,  # will be filled after order_items
        })
    return pd.DataFrame(rows)


def generate_order_items(
    rng: np.random.Generator,
    orders: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    pids = products["product_id"].values
    prices = products.set_index("product_id")["price"].to_dict()
    order_ids = orders["order_id"].values

    # distribute ~50k items across ~20k orders (avg 2.5 items/order)
    rows = []
    items_per_order = rng.poisson(lam=2.5, size=len(order_ids))
    items_per_order = np.clip(items_per_order, 1, 10)

    idx = 0
    for oid, n_items in zip(order_ids, items_per_order):
        chosen = rng.choice(pids, size=int(n_items), replace=False)
        for pid in chosen:
            qty = int(rng.integers(1, 4))
            rows.append({
                "order_id": int(oid),
                "product_id": int(pid),
                "quantity": qty,
                "price": prices[int(pid)],
            })
            idx += 1
            if idx >= NUM_ORDER_ITEMS:
                break
        if idx >= NUM_ORDER_ITEMS:
            break

    df = pd.DataFrame(rows)

    # back-fill order totals
    totals = df.groupby("order_id").apply(
        lambda g: (g["quantity"] * g["price"]).sum()
    ).reset_index(name="total")
    orders_updated = orders.merge(totals, on="order_id", how="left", suffixes=("_old", ""))
    orders_updated["total"] = orders_updated["total"].fillna(0).round(2)
    if "total_old" in orders_updated.columns:
        orders_updated.drop(columns=["total_old"], inplace=True)

    return df, orders_updated


def generate_browsing_history(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    cids = customers["customer_id"].values
    pids = products["product_id"].values
    weights = 1.0 / np.arange(1, len(cids) + 1) ** 0.6
    weights /= weights.sum()

    start = datetime(2024, 6, 1)
    end = datetime(2026, 3, 15)
    rows = []
    for _ in range(NUM_BROWSING):
        rows.append({
            "customer_id": int(rng.choice(cids, p=weights)),
            "product_id": int(rng.choice(pids)),
            "timestamp": _random_date(start, end, rng).strftime("%Y-%m-%d %H:%M:%S"),
            "action": rng.choice(BROWSING_ACTIONS),
        })
    return pd.DataFrame(rows)


def generate_cart_activity(
    rng: np.random.Generator,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    cids = customers["customer_id"].values
    pids = products["product_id"].values
    weights = 1.0 / np.arange(1, len(cids) + 1) ** 0.6
    weights /= weights.sum()

    start = datetime(2024, 6, 1)
    end = datetime(2026, 3, 15)
    rows = []
    for _ in range(NUM_CART):
        rows.append({
            "customer_id": int(rng.choice(cids, p=weights)),
            "product_id": int(rng.choice(pids)),
            "timestamp": _random_date(start, end, rng).strftime("%Y-%m-%d %H:%M:%S"),
            "action": rng.choice(CART_ACTIONS, p=[0.7, 0.3]),  # more adds than removes
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: str = "data/generated") -> None:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    print("Generating customers …")
    customers = generate_customers(rng)
    customers.to_csv(os.path.join(output_dir, "customers.csv"), index=False)
    print(f"  ✓ {len(customers)} customers")

    print("Generating products …")
    products = generate_products(rng)
    products.to_csv(os.path.join(output_dir, "products.csv"), index=False)
    print(f"  ✓ {len(products)} products")

    print("Generating orders …")
    orders = generate_orders(rng, customers)

    print("Generating order items …")
    order_items, orders = generate_order_items(rng, orders, products)
    orders.to_csv(os.path.join(output_dir, "orders.csv"), index=False)
    order_items.to_csv(os.path.join(output_dir, "order_items.csv"), index=False)
    print(f"  ✓ {len(orders)} orders, {len(order_items)} order items")

    print("Generating browsing history …")
    browsing = generate_browsing_history(rng, customers, products)
    browsing.to_csv(os.path.join(output_dir, "browsing_history.csv"), index=False)
    print(f"  ✓ {len(browsing)} browsing events")

    print("Generating cart activity …")
    cart = generate_cart_activity(rng, customers, products)
    cart.to_csv(os.path.join(output_dir, "cart_activity.csv"), index=False)
    print(f"  ✓ {len(cart)} cart events")

    print(f"\nAll datasets saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic e-commerce data")
    parser.add_argument("--output-dir", default="data/generated", help="Output directory")
    args = parser.parse_args()
    main(args.output_dir)
