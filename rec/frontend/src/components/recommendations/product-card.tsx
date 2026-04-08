"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Recommendation } from "@/lib/types";

interface Props {
  rec: Recommendation;
  rank: number;
  maxScore: number;
}

export function ProductCard({ rec, rank, maxScore }: Props) {
  const pct = Math.round((rec.score / maxScore) * 100);
  const tags = rec.tags
    .split("|")
    .map((t) => t.trim())
    .filter(Boolean);

  return (
    <Card className="relative transition-all hover:-translate-y-0.5 hover:ring-primary/35">
      <CardContent className="pt-5 space-y-2">
        <Badge
          variant="outline"
          className="absolute top-3 right-3 text-[0.62rem] text-primary bg-primary/10 ring-primary/25"
        >
          #{rank}
        </Badge>
        <div className="font-semibold text-[0.88rem] pr-8 leading-tight">
          {rec.name}
        </div>
        <div className="text-[0.72rem] text-muted-foreground">
          {rec.category} / {rec.brand}
        </div>
        <div className="text-lg font-bold text-primary">
          ${rec.price.toFixed(2)}
        </div>
        <div className="flex gap-1 flex-wrap">
          {tags.map((t) => (
            <span
              key={t}
              className="text-[0.58rem] px-2 py-0.5 rounded-full bg-muted/50 text-muted-foreground ring-1 ring-white/10"
            >
              {t}
            </span>
          ))}
        </div>
        {rec.explanation && (
          <div className="text-[0.68rem] text-muted-foreground italic bg-primary/8 rounded-xl p-2 ring-1 ring-primary/20 leading-relaxed">
            {rec.explanation}
          </div>
        )}
        <div className="h-0.5 rounded-full bg-muted/50 overflow-hidden mt-2">
          <div
            className="h-full rounded-full bg-primary transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      </CardContent>
    </Card>
  );
}
