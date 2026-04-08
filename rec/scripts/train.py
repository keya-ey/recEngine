"""
Training Orchestrator
======================
End-to-end script that:
  1. Loads synthetic data
  2. Computes RFM segments & behavioral tags
  3. Prepares training pairs (positive interactions + in-batch negatives)
  4. Trains the Two-Tower model
  5. Pre-computes item embeddings → FAISS index
  6. Trains the LightGBM re-ranker
  7. Saves all artefacts to disk

Usage:
    python scripts/train.py [--epochs 10] [--smoke-test]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.rfm import compute_rfm
from features.tagging import compute_tags, ALL_TAGS
from features.feature_store import FeatureStore
from model.two_tower import TwoTowerModel
from model.reranker import ReRanker
from serving.retriever import Retriever


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data/generated"
ARTIFACT_DIR = "artifacts"
EMBED_DIM = 64
BATCH_SIZE = 256
LR = 1e-3
MAX_SEQ_LEN = 10  # recent interactions per user


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def build_product_encoders(products: pd.DataFrame):
    """Create label encoders for category and brand, plus tag multi-hot."""
    categories = sorted(products["category"].unique())
    brands = sorted(products["brand"].unique())
    cat2id = {c: i for i, c in enumerate(categories)}
    brand2id = {b: i for i, b in enumerate(brands)}

    # collect all unique product tags
    all_ptags = set()
    for tag_str in products["tags"].fillna(""):
        for t in tag_str.split("|"):
            t = t.strip()
            if t:
                all_ptags.add(t)
    all_ptags = sorted(all_ptags)
    ptag2idx = {t: i for i, t in enumerate(all_ptags)}

    def encode_tags(tag_str: str) -> list[float]:
        vec = [0.0] * len(all_ptags)
        for t in tag_str.split("|"):
            t = t.strip()
            if t in ptag2idx:
                vec[ptag2idx[t]] = 1.0
        return vec

    return cat2id, brand2id, all_ptags, ptag2idx, encode_tags


def build_product_features(products: pd.DataFrame, cat2id, brand2id, encode_tags_fn):
    """Build dict product_id → {category_id, brand_id, price_norm, tag_vec}."""
    max_price = products["price"].max()
    feats = {}
    for _, row in products.iterrows():
        pid = int(row["product_id"])
        feats[pid] = {
            "category_id": cat2id[row["category"]],
            "brand_id": brand2id[row["brand"]],
            "price_norm": float(row["price"]) / max_price,
            "tag_vec": np.array(encode_tags_fn(row["tags"]), dtype=np.float32),
        }
    return feats


def build_product_metadata(products: pd.DataFrame):
    """Build dict product_id → display metadata."""
    meta = {}
    for _, row in products.iterrows():
        pid = int(row["product_id"])
        meta[pid] = {
            "name": row["name"],
            "category": row["category"],
            "brand": row["brand"],
            "price": float(row["price"]),
            "tags": row["tags"],
        }
    return meta


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InteractionDataset(Dataset):
    """Per-sample: (user_features, interaction_embeds, interaction_mask,
                    item_category_id, item_brand_id, item_price, item_tag_vec)"""

    def __init__(
        self,
        user_ids: list[int],
        positive_items: list[int],
        feature_store: FeatureStore,
        product_features: dict,
        user_histories: dict[int, list[int]],
        embed_dim: int,
        max_seq_len: int,
    ):
        self.samples = list(zip(user_ids, positive_items))
        self.feature_store = feature_store
        self.product_features = product_features
        self.user_histories = user_histories
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, pid = self.samples[idx]

        # user features
        user_vec = self.feature_store.get_user_features(uid)

        # interaction history embeddings (use product features as proxy during training)
        history = self.user_histories.get(uid, [])[-self.max_seq_len:]
        seq_embeds = []
        for h_pid in history:
            pf = self.product_features.get(h_pid)
            if pf:
                e = np.concatenate([[pf["category_id"] / 20.0, pf["brand_id"] / 30.0, pf["price_norm"]], pf["tag_vec"]])
                # pad or truncate to embed_dim
                if len(e) < self.embed_dim:
                    e = np.pad(e, (0, self.embed_dim - len(e)))
                else:
                    e = e[:self.embed_dim]
                seq_embeds.append(e)

        if not seq_embeds:
            seq_embeds = [np.zeros(self.embed_dim)]

        # pad to max_seq_len
        mask = [False] * len(seq_embeds) + [True] * (self.max_seq_len - len(seq_embeds))
        while len(seq_embeds) < self.max_seq_len:
            seq_embeds.append(np.zeros(self.embed_dim))

        seq_tensor = np.array(seq_embeds[:self.max_seq_len], dtype=np.float32)
        mask_tensor = np.array(mask[:self.max_seq_len], dtype=bool)

        # positive item features
        pf = self.product_features[pid]

        return (
            user_vec.astype(np.float32),
            seq_tensor,
            mask_tensor,
            np.int64(pf["category_id"]),
            np.int64(pf["brand_id"]),
            np.float32(pf["price_norm"]),
            pf["tag_vec"].astype(np.float32),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_two_tower(
    model: TwoTowerModel,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            user_vec, seq_emb, seq_mask, cat_id, brand_id, price, tag_vec = [
                b.to(device) for b in batch
            ]
            price = price.unsqueeze(-1)

            out = model(
                user_features=user_vec,
                interaction_embeds=seq_emb,
                interaction_mask=seq_mask,
                item_category_ids=cat_id,
                item_brand_ids=brand_id,
                item_price=price,
                item_tag_vec=tag_vec,
            )
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Pre-compute item embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_item_embeddings(
    model: TwoTowerModel,
    product_features: dict,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, product_ids) arrays."""
    model.eval()
    pids = sorted(product_features.keys())
    cat_ids, brand_ids, prices, tag_vecs = [], [], [], []

    for pid in pids:
        pf = product_features[pid]
        cat_ids.append(pf["category_id"])
        brand_ids.append(pf["brand_id"])
        prices.append(pf["price_norm"])
        tag_vecs.append(pf["tag_vec"])

    cat_t = torch.tensor(cat_ids, dtype=torch.long, device=device)
    brand_t = torch.tensor(brand_ids, dtype=torch.long, device=device)
    price_t = torch.tensor(prices, dtype=torch.float32, device=device).unsqueeze(-1)
    tag_t = torch.tensor(np.array(tag_vecs), dtype=torch.float32, device=device)

    embs = model.encode_item(cat_t, brand_t, price_t, tag_t)
    embs = F.normalize(embs, dim=-1)

    return embs.cpu().numpy().astype(np.float32), np.array(pids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Train re-ranker
# ---------------------------------------------------------------------------

def train_reranker(
    reranker: ReRanker,
    feature_store: FeatureStore,
    product_features: dict,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    all_pids: list[int],
    rng: np.random.Generator,
) -> None:
    """Train the sklearn re-ranker on purchase vs non-purchase pairs."""
    print("\n── Training Re-Ranker ────────────────────────────────────────")

    features_list = []
    labels = []

    # build positive pairs from order_items
    merged = order_items.merge(orders[["order_id", "customer_id"]], on="order_id")
    positive_pairs = set(zip(merged["customer_id"].values, merged["product_id"].values))

    for cid, pid in list(positive_pairs)[:5000]:  # cap for speed
        user_vec = feature_store.get_user_features(int(cid))
        pf = product_features.get(int(pid))
        if pf is None:
            continue
        item_vec = np.concatenate([
            [pf["category_id"] / 20.0, pf["brand_id"] / 30.0, pf["price_norm"]],
            pf["tag_vec"],
        ])
        feat = ReRanker.build_features(user_vec, item_vec.reshape(1, -1))
        features_list.append(feat[0])
        labels.append(1)

        # negative sample
        neg_pid = int(rng.choice(all_pids))
        while (cid, neg_pid) in positive_pairs:
            neg_pid = int(rng.choice(all_pids))
        npf = product_features.get(neg_pid)
        if npf is None:
            continue
        neg_item_vec = np.concatenate([
            [npf["category_id"] / 20.0, npf["brand_id"] / 30.0, npf["price_norm"]],
            npf["tag_vec"],
        ])
        neg_feat = ReRanker.build_features(user_vec, neg_item_vec.reshape(1, -1))
        features_list.append(neg_feat[0])
        labels.append(0)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    # 80/20 split
    n = len(X)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    reranker.train(
        X[idx[:split]], y[idx[:split]],
        X[idx[split:]], y[idx[split:]],
        n_estimators=100,
    )
    print("  ✓ Re-ranker trained")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 2 epochs on a tiny subset")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # ── 1. Load data ────────────────────────────────────────────────
    print("── Loading Data ──────────────────────────────────────────────")
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    order_items = pd.read_csv(os.path.join(DATA_DIR, "order_items.csv"))
    browsing = pd.read_csv(os.path.join(DATA_DIR, "browsing_history.csv"))
    cart = pd.read_csv(os.path.join(DATA_DIR, "cart_activity.csv"))
    print(f"  Customers={len(customers)}  Products={len(products)}  "
          f"Orders={len(orders)}  OrderItems={len(order_items)}")

    # ── 2. Feature engineering ──────────────────────────────────────
    print("\n── Feature Engineering ───────────────────────────────────────")
    print("  Computing RFM scores …")
    rfm = compute_rfm(orders, order_items)
    print(f"  ✓ RFM computed for {len(rfm)} customers")

    print("  Computing behavioral tags …")
    if args.smoke_test:
        # use only a small subset for speed
        browsing_sub = browsing[browsing["customer_id"] <= 50]
        cart_sub = cart[cart["customer_id"] <= 50]
    else:
        browsing_sub = browsing
        cart_sub = cart
    tags = compute_tags(browsing_sub, cart_sub, products, orders)
    print(f"  ✓ Tags computed for {len(tags)} customers")

    # populate feature store
    fs = FeatureStore()
    fs.load_rfm(rfm)
    fs.load_tags(tags)

    # ── 3. Product encoders ─────────────────────────────────────────
    cat2id, brand2id, all_ptags, ptag2idx, encode_tags_fn = build_product_encoders(products)
    product_features = build_product_features(products, cat2id, brand2id, encode_tags_fn)
    product_metadata = build_product_metadata(products)

    # save encoders for serving
    encoders = {
        "cat2id": cat2id,
        "brand2id": brand2id,
        "all_ptags": all_ptags,
    }
    with open(os.path.join(ARTIFACT_DIR, "encoders.json"), "w") as f:
        json.dump(encoders, f)

    # ── 4. Build training pairs ─────────────────────────────────────
    print("\n── Preparing Training Data ───────────────────────────────────")
    merged = order_items.merge(orders[["order_id", "customer_id"]], on="order_id")
    valid_pids = set(product_features.keys())
    merged = merged[merged["product_id"].isin(valid_pids)]

    if args.smoke_test:
        merged = merged.head(2000)

    user_ids = merged["customer_id"].astype(int).tolist()
    positive_items = merged["product_id"].astype(int).tolist()

    # user histories
    user_histories: dict[int, list[int]] = {}
    for uid, pid in zip(user_ids, positive_items):
        user_histories.setdefault(uid, []).append(pid)

    dataset = InteractionDataset(
        user_ids, positive_items, fs, product_features,
        user_histories, EMBED_DIM, MAX_SEQ_LEN,
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True,
    )
    print(f"  ✓ {len(dataset)} training pairs, {len(dataloader)} batches")

    # ── 5. Train Two-Tower ──────────────────────────────────────────
    print("\n── Training Two-Tower Model ──────────────────────────────────")
    model = TwoTowerModel(
        user_feature_dim=fs.total_user_feature_dim,
        n_categories=len(cat2id),
        n_brands=len(brand2id),
        n_product_tags=len(all_ptags),
        embed_dim=EMBED_DIM,
    ).to(device)

    t0 = time.time()
    losses = train_two_tower(model, dataloader, args.epochs, LR, device)
    train_time = time.time() - t0
    print(f"  ✓ Training complete in {train_time:.1f}s  "
          f"Final loss={losses[-1]:.4f}")

    # save model
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "two_tower.pt"))

    # ── 6. Pre-compute item embeddings & FAISS index ────────────────
    print("\n── Building FAISS Index ──────────────────────────────────────")
    embeddings, pids_arr = precompute_item_embeddings(model, product_features, device)

    retriever = Retriever(embed_dim=EMBED_DIM)
    retriever.build_index(embeddings, pids_arr)
    retriever.save(os.path.join(ARTIFACT_DIR, "faiss"))
    print(f"  ✓ Indexed {len(pids_arr)} products into FAISS")

    # ── 7. Train re-ranker ──────────────────────────────────────────
    all_pids = list(product_features.keys())
    reranker = ReRanker()
    train_reranker(reranker, fs, product_features, orders, order_items, all_pids, rng)
    reranker.save(os.path.join(ARTIFACT_DIR, "reranker.lgb"))

    # ── 8. Save feature store state ─────────────────────────────────
    rfm.to_csv(os.path.join(ARTIFACT_DIR, "rfm_scores.csv"), index=False)
    with open(os.path.join(ARTIFACT_DIR, "tags.json"), "w") as f:
        json.dump({str(k): v for k, v in tags.items()}, f)

    # save product features & metadata
    np.savez(
        os.path.join(ARTIFACT_DIR, "product_features.npz"),
        **{str(k): np.concatenate([
            [v["category_id"], v["brand_id"], v["price_norm"]], v["tag_vec"]
        ]) for k, v in product_features.items()},
    )
    with open(os.path.join(ARTIFACT_DIR, "product_metadata.json"), "w") as f:
        json.dump({str(k): v for k, v in product_metadata.items()}, f)

    print("\n══════════════════════════════════════════════════════════════")
    print(f"  All artifacts saved to {ARTIFACT_DIR}/")
    print("  • two_tower.pt         – model weights")
    print("  • faiss/               – FAISS index + product IDs")
    print("  • reranker.lgb         – LightGBM re-ranker")
    print("  • encoders.json        – category/brand encoders")
    print("  • rfm_scores.csv       – RFM scores")
    print("  • tags.json            – behavioral tags")
    print("  • product_metadata.json – product display data")
    print("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
