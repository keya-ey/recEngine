from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "generated"

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

STYLE_WORDS = ["Essential", "Signature", "Everyday", "Select", "Prime", "Smart", "Comfort", "Classic", "Modern", "Ultra"]
SUFFIX_WORDS = ["Kit", "Set", "Collection", "Edition", "Pack", "Series"]

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


def make_customer_name(idx: int) -> str:
    first = FIRST_NAMES[idx % len(FIRST_NAMES)]
    last = LAST_NAMES[(idx // len(FIRST_NAMES)) % len(LAST_NAMES)]
    return f"{first} {last}"


def make_product_name(brand: str, category: str, idx: int) -> str:
    style = STYLE_WORDS[idx % len(STYLE_WORDS)]
    suffix = SUFFIX_WORDS[idx % len(SUFFIX_WORDS)]
    cleaned_category = str(category).replace("&", "and").replace("  ", " ").strip()
    return f"{brand} {style} {cleaned_category} {suffix}"


def normalize_tags(raw_tags: object, category: str, price: float) -> str:
    tags = [t.strip() for t in str(raw_tags or "").split("|") if t and t.strip()]
    deduped: list[str] = []
    seen = set()
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)

    if not deduped:
        if price < 25:
            deduped.append("Budget")
        elif price < 100:
            deduped.append("Mid-Range")
        else:
            deduped.append("Premium")

    hint = CATEGORY_TAG_HINTS.get(str(category), "Trending")
    if hint not in deduped:
        deduped.append(hint)

    if len(deduped) == 1:
        deduped.append("Best Seller")

    return "|".join(deduped)


def main() -> None:
    customers_path = DATA_DIR / "customers.csv"
    products_path = DATA_DIR / "products.csv"

    customers_df = pd.read_csv(customers_path)
    products_df = pd.read_csv(products_path)

    customers_df = customers_df.sort_values("customer_id").reset_index(drop=True)
    customers_df["name"] = [make_customer_name(i) for i in range(len(customers_df))]

    products_df = products_df.sort_values("product_id").reset_index(drop=True)
    products_df["name"] = [
        make_product_name(str(row.brand), str(row.category), i)
        for i, row in products_df.iterrows()
    ]
    products_df["tags"] = [
        normalize_tags(row.tags, str(row.category), float(row.price))
        for _, row in products_df.iterrows()
    ]

    customers_df.to_csv(customers_path, index=False)
    products_df.to_csv(products_path, index=False)

    print(f"Updated {len(customers_df)} customers and {len(products_df)} products")


if __name__ == "__main__":
    main()
