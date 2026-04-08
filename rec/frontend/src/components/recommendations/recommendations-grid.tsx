"use client";

import { ProductCard } from "./product-card";
import type { Recommendation } from "@/lib/types";

export function RecommendationsGrid({ items }: { items: Recommendation[] }) {
  if (items.length === 0) return null;
  const maxScore = Math.max(...items.map((r) => r.score), 0.01);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      {items.map((rec, i) => (
        <ProductCard key={rec.product_id} rec={rec} rank={i + 1} maxScore={maxScore} />
      ))}
    </div>
  );
}
