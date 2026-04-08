"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/layout/header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Bar, BarChart, XAxis, YAxis, CartesianGrid } from "recharts";
import { Users, Package, ShoppingCart, DollarSign } from "lucide-react";
import {
  getAnalyticsSummary,
  getSegments,
  getTags,
  getTopCategories,
  getTopBrands,
} from "@/lib/api";
import type {
  AnalyticsSummary,
  SegmentData,
  TagData,
  CategoryRevenue,
  BrandRevenue,
} from "@/lib/types";

const segmentConfig: ChartConfig = {
  count: { label: "Customers", color: "var(--chart-1)" },
};
const tagConfig: ChartConfig = {
  count: { label: "Customers", color: "var(--chart-2)" },
};
const catConfig: ChartConfig = {
  revenue: { label: "Revenue", color: "var(--chart-1)" },
};
const brandConfig: ChartConfig = {
  revenue: { label: "Revenue", color: "var(--chart-3)" },
};

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [segments, setSegments] = useState<SegmentData[]>([]);
  const [tags, setTags] = useState<TagData[]>([]);
  const [categories, setCategories] = useState<CategoryRevenue[]>([]);
  const [brands, setBrands] = useState<BrandRevenue[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [s, seg, t, cat, br] = await Promise.all([
          getAnalyticsSummary(),
          getSegments(),
          getTags(),
          getTopCategories(),
          getTopBrands(),
        ]);
        setSummary(s);
        setSegments(seg);
        setTags(t);
        setCategories(cat);
        setBrands(br);
      } catch {
        // silent
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <>
        <Header title="Analytics" />
        <div className="p-4 space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-24" />
            ))}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-72" />
            ))}
          </div>
        </div>
      </>
    );
  }

  const kpis = [
    {
      label: "Total Customers",
      value: summary?.total_customers.toLocaleString() ?? "0",
      icon: Users,
    },
    {
      label: "Total Products",
      value: summary?.total_products.toLocaleString() ?? "0",
      icon: Package,
    },
    {
      label: "Total Orders",
      value: summary?.total_orders.toLocaleString() ?? "0",
      icon: ShoppingCart,
    },
    {
      label: "Total Revenue",
      value: `$${(summary?.total_revenue ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
      icon: DollarSign,
    },
  ];

  return (
    <>
      <Header title="Analytics" />
      <div className="p-4 space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {kpis.map((k) => (
            <Card key={k.label}>
              <CardContent className="pt-4 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10">
                  <k.icon className="size-5 text-primary" />
                </div>
                <div>
                  <div className="text-xl font-bold">{k.value}</div>
                  <div className="text-[0.65rem] text-muted-foreground uppercase tracking-wider">
                    {k.label}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">RFM Segment Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer config={segmentConfig} className="h-64 w-full">
                <BarChart data={segments}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="segment"
                    tick={{ fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="count" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Behavioral Tag Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer config={tagConfig} className="h-64 w-full">
                <BarChart data={tags} layout="vertical">
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis
                    dataKey="tag"
                    type="category"
                    tick={{ fontSize: 9 }}
                    tickLine={false}
                    axisLine={false}
                    width={110}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="count" fill="var(--chart-2)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Top Categories by Revenue</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer config={catConfig} className="h-64 w-full">
                <BarChart data={categories}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="category"
                    tick={{ fontSize: 9 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="revenue" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Top Brands by Revenue</CardTitle>
            </CardHeader>
            <CardContent>
              <ChartContainer config={brandConfig} className="h-64 w-full">
                <BarChart data={brands}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="brand"
                    tick={{ fontSize: 9 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="revenue" fill="var(--chart-3)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
