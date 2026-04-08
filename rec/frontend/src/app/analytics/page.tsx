"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Header } from "@/components/layout/header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Bar, BarChart, XAxis, YAxis, CartesianGrid, Line, LineChart } from "recharts";
import { Users, Package, ShoppingCart, DollarSign, Bot, LifeBuoy, TriangleAlert } from "lucide-react";
import {
  getAnalyticsSummary,
  getSegments,
  getTags,
  getTopCategories,
  getTopBrands,
  getArpuWeeklyAnalytics,
  getSaveCustomerAnalytics,
} from "@/lib/api";
import type {
  AnalyticsSummary,
  SegmentData,
  TagData,
  CategoryRevenue,
  BrandRevenue,
  ArpuWeeklyAnalytics,
  SaveCustomerAnalytics,
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
const arpuConfig: ChartConfig = {
  arpu: { label: "Weekly ARPU", color: "var(--chart-1)" },
  threshold: { label: "Threshold", color: "var(--destructive)" },
};

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [segments, setSegments] = useState<SegmentData[]>([]);
  const [tags, setTags] = useState<TagData[]>([]);
  const [categories, setCategories] = useState<CategoryRevenue[]>([]);
  const [brands, setBrands] = useState<BrandRevenue[]>([]);
  const [arpuWeekly, setArpuWeekly] = useState<ArpuWeeklyAnalytics | null>(null);
  const [saveCustomer, setSaveCustomer] = useState<SaveCustomerAnalytics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [s, seg, t, cat, br, aw, sc] = await Promise.all([
          getAnalyticsSummary(),
          getSegments(),
          getTags(),
          getTopCategories(),
          getTopBrands(),
          getArpuWeeklyAnalytics(),
          getSaveCustomerAnalytics(),
        ]);
        setSummary(s);
        setSegments(seg);
        setTags(t);
        setCategories(cat);
        setBrands(br);
        setArpuWeekly(aw);
        setSaveCustomer(sc);
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
      <div className="p-4 space-y-5">
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

        <Card>
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle className="text-sm">Weekly ARPU Trend</CardTitle>
              <div className="flex items-center gap-2">
                {arpuWeekly?.below_threshold ? (
                  <Badge variant="destructive">Below Threshold</Badge>
                ) : (
                  <Badge variant="secondary">Healthy</Badge>
                )}
                <Badge variant="outline">
                  Threshold: ${arpuWeekly?.threshold.toFixed(2) ?? "0.00"}
                </Badge>
                <Badge variant="outline">
                  Current: ${arpuWeekly?.latest_value.toFixed(2) ?? "0.00"}
                </Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <ChartContainer config={arpuConfig} className="h-72 w-full">
              <LineChart data={arpuWeekly?.weeks ?? []}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis
                  dataKey="week_start"
                  tick={{ fontSize: 10 }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis tick={{ fontSize: 10 }} tickLine={false} axisLine={false} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line
                  type="monotone"
                  dataKey="arpu"
                  stroke="var(--chart-1)"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                />
                <Line
                  type="monotone"
                  dataKey="threshold"
                  stroke="var(--destructive)"
                  strokeDasharray="6 4"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ChartContainer>
            {arpuWeekly?.below_threshold && (
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-destructive/10 p-3 ring-1 ring-destructive/35">
                <p className="text-sm text-destructive">
                  ARPU is below target by {Math.abs(arpuWeekly.gap_pct).toFixed(2)}%. Trigger the recovery workflow.
                </p>
                <Button nativeButton={false} render={<Link href="/arpu-recovery" />} size="sm">
                  <Bot className="size-4" />
                  Trigger Recovery Agent
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle className="text-sm">Save-the-Customer Alert</CardTitle>
              <div className="flex items-center gap-2">
                {saveCustomer?.alert.triggered ? (
                  <Badge variant="destructive">Threshold Breached</Badge>
                ) : (
                  <Badge variant="secondary">Within Threshold</Badge>
                )}
                <Badge variant="outline">
                  High-risk: {saveCustomer?.alert.high_risk_count ?? 0}
                </Badge>
                <Badge variant="outline">
                  Threshold: {saveCustomer?.alert.monthly_threshold ?? 0}
                </Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {(saveCustomer?.value_tiers ?? []).map((tier) => (
                <div key={tier.tier} className="rounded-2xl bg-muted/30 p-3 ring-1 ring-white/10">
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-sm font-medium">{tier.tier}</div>
                    <Badge variant="outline">{tier.priority}</Badge>
                  </div>
                  <div className="text-xl font-semibold mt-1">{tier.count}</div>
                  <div className="text-xs text-muted-foreground">{tier.criteria}</div>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-muted-foreground">
              <div className="rounded-xl bg-muted/25 p-3 ring-1 ring-white/10">
                Low email engagement:{" "}
                <span className="text-foreground font-semibold">
                  {saveCustomer?.warning_signs.low_email_engagement_customers ?? 0}
                </span>
              </div>
              <div className="rounded-xl bg-muted/25 p-3 ring-1 ring-white/10">
                Purchase gap expanded:{" "}
                <span className="text-foreground font-semibold">
                  {saveCustomer?.warning_signs.purchase_gap_expanded_customers ?? 0}
                </span>
              </div>
              <div className="rounded-xl bg-muted/25 p-3 ring-1 ring-white/10">
                Price complaint signals:{" "}
                <span className="text-foreground font-semibold">
                  {saveCustomer?.warning_signs.price_complaint_signal_customers ?? 0}
                </span>
              </div>
            </div>
            {saveCustomer?.alert.triggered ? (
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-destructive/10 p-3 ring-1 ring-destructive/35">
                <p className="text-sm text-destructive flex items-center gap-2">
                  <TriangleAlert className="size-4" />
                  Save mission is triggered. Human approvals are required at each phase.
                </p>
                <Button nativeButton={false} render={<Link href="/save-the-customer" />} size="sm">
                  <LifeBuoy className="size-4" />
                  Open Save-the-Customer Agent
                </Button>
              </div>
            ) : (
              <div className="flex justify-end">
                <Button nativeButton={false} render={<Link href="/save-the-customer" />} variant="outline" size="sm">
                  <LifeBuoy className="size-4" />
                  Open Save-the-Customer Agent
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

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
