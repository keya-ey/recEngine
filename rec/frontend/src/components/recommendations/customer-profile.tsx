"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import type { CustomerProfile as ProfileType } from "@/lib/types";

const segmentColors: Record<string, string> = {
  Champions: "bg-yellow-500/15 text-yellow-400 ring-yellow-500/30",
  Loyal: "bg-blue-500/15 text-blue-400 ring-blue-500/30",
  Promising: "bg-green-500/15 text-green-400 ring-green-500/30",
  "Potential Loyalist": "bg-cyan-500/15 text-cyan-400 ring-cyan-500/30",
  "At Risk": "bg-orange-500/15 text-orange-400 ring-orange-500/30",
  Hibernating: "bg-purple-500/15 text-purple-400 ring-purple-500/30",
  Lost: "bg-red-500/15 text-red-400 ring-red-500/30",
};

interface Props {
  customerId: number;
  segment: string;
  tags: string[];
  profile: ProfileType;
}

export function CustomerProfileCard({
  customerId,
  segment,
  tags,
  profile,
}: Props) {
  const segClass =
    segmentColors[segment] ||
    "bg-muted text-muted-foreground ring-white/12";

  return (
    <Card>
      <CardContent className="pt-5 space-y-4">
        <div className="flex items-center gap-2 flex-wrap pb-2">
          <span className="font-bold">Customer #{customerId}</span>
          <Badge variant="outline" className={segClass}>
            {segment}
          </Badge>
          {tags.map((t) => (
            <Badge key={t} variant="secondary" className="text-[0.66rem]">
              {t}
            </Badge>
          ))}
          {tags.length === 0 && (
            <Badge variant="secondary" className="text-[0.66rem]">
              No tags yet
            </Badge>
          )}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
          {[
            { label: "Orders", value: profile.total_orders },
            { label: "Items Bought", value: profile.total_items },
            {
              label: "Total Spend",
              value: `$${profile.total_spend.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
            },
            {
              label: "Avg Order",
              value: `$${profile.avg_order_value.toFixed(0)}`,
            },
            {
              label: "Avg Price",
              value: `$${profile.avg_item_price.toFixed(0)}`,
            },
            {
              label: "Product Views",
              value: profile.total_product_views ?? 0,
            },
          ].map((s) => (
            <div
              key={s.label}
              className="rounded-xl bg-card/45 p-2.5 text-center ring-1 ring-white/10"
            >
              <div className="text-lg font-bold text-primary">{s.value}</div>
              <div className="text-[0.6rem] font-medium text-muted-foreground uppercase tracking-wider">
                {s.label}
              </div>
            </div>
          ))}
        </div>

        <Separator />

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
          <div>
            <h4 className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              Top Categories
            </h4>
            {profile.top_categories.length > 0 ? (
              profile.top_categories.map((c) => (
                <div
                  key={c.name}
                  className="flex justify-between rounded-lg px-2 py-1.5 text-[0.76rem] hover:bg-muted/35"
                >
                  <span className="text-foreground/80">{c.name}</span>
                  <Badge variant="outline" className="text-[0.66rem] text-primary">
                    {c.count} items
                  </Badge>
                </div>
              ))
            ) : (
              <span className="text-muted-foreground text-xs">
                No purchase history
              </span>
            )}
          </div>
          <div>
            <h4 className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              Top Brands
            </h4>
            {profile.top_brands.length > 0 ? (
              profile.top_brands.map((b) => (
                <div
                  key={b.name}
                  className="flex justify-between rounded-lg px-2 py-1.5 text-[0.76rem] hover:bg-muted/35"
                >
                  <span className="text-foreground/80">{b.name}</span>
                  <Badge variant="outline" className="text-[0.66rem] text-primary">
                    {b.count} items
                  </Badge>
                </div>
              ))
            ) : (
              <span className="text-muted-foreground text-xs">No data</span>
            )}
          </div>
          <div>
            <h4 className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              Recent Orders
            </h4>
            {profile.recent_orders.length > 0 ? (
              profile.recent_orders.map((o) => (
                <div
                  key={o.order_id}
                  className="rounded-lg px-2 py-1.5 text-[0.72rem] text-muted-foreground hover:bg-muted/35"
                >
                  <span className="text-muted-foreground/60 text-[0.66rem]">
                    {o.date}
                  </span>{" "}
                  —{" "}
                  <span className="text-primary">{o.category}</span> /{" "}
                  {o.brand}
                </div>
              ))
            ) : (
              <span className="text-muted-foreground text-xs">
                No recent orders
              </span>
            )}
            {profile.browsed_categories &&
              profile.browsed_categories.length > 0 && (
                <>
                  <h4 className="text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground mb-2 mt-3">
                    Most Browsed
                  </h4>
                  {profile.browsed_categories.map((c) => (
                    <div
                      key={c.name}
                      className="flex justify-between rounded-lg px-2 py-1.5 text-[0.76rem] hover:bg-muted/35"
                    >
                      <span className="text-foreground/80">{c.name}</span>
                      <Badge
                        variant="outline"
                        className="text-[0.66rem] text-primary"
                      >
                        {c.views} views
                      </Badge>
                    </div>
                  ))}
                </>
              )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
