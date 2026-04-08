"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Header } from "@/components/layout/header";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent } from "@/components/ui/card";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import {
  ChevronLeft,
  ChevronRight,
  Search,
  ArrowRight,
} from "lucide-react";
import { getCustomers } from "@/lib/api";
import type { Customer, PaginatedResponse } from "@/lib/types";

const SEGMENTS = [
  "Champions",
  "Loyal",
  "Promising",
  "Potential Loyalist",
  "At Risk",
  "Hibernating",
  "Lost",
];

const segmentColors: Record<string, string> = {
  Champions: "bg-yellow-500/15 text-yellow-400 ring-yellow-500/30",
  Loyal: "bg-blue-500/15 text-blue-400 ring-blue-500/30",
  Promising: "bg-green-500/15 text-green-400 ring-green-500/30",
  "Potential Loyalist": "bg-cyan-500/15 text-cyan-400 ring-cyan-500/30",
  "At Risk": "bg-orange-500/15 text-orange-400 ring-orange-500/30",
  Hibernating: "bg-purple-500/15 text-purple-400 ring-purple-500/30",
  Lost: "bg-red-500/15 text-red-400 ring-red-500/30",
};

export default function CustomersPage() {
  const router = useRouter();
  const [data, setData] = useState<PaginatedResponse<Customer> | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [segment, setSegment] = useState("");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<Customer | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getCustomers({
        search: search || undefined,
        segment: segment || undefined,
        page,
        limit: 20,
      });
      setData(result);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [search, segment, page]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const totalPages = data ? Math.ceil(data.total / data.limit) : 0;

  return (
    <>
      <Header title="Customers" />
      <div className="p-4 space-y-4">
        <Card>
          <CardContent className="pt-4">
            <div className="flex flex-wrap items-end gap-3">
              <div className="flex-1 min-w-[200px] space-y-1">
                <label className="text-[0.68rem] font-medium text-muted-foreground uppercase tracking-wider">
                  Search
                </label>
                <div className="relative">
                  <Search className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
                  <Input
                    placeholder="Search by name or ID..."
                    className="pl-8"
                    value={search}
                    onChange={(e) => {
                      setSearch(e.target.value);
                      setPage(1);
                    }}
                  />
                </div>
              </div>
              <div className="w-44 space-y-1">
                <label className="text-[0.68rem] font-medium text-muted-foreground uppercase tracking-wider">
                  RFM Segment
                </label>
                <Select
                  value={segment}
                  onValueChange={(v) => {
                    setSegment(!v || v === "all" ? "" : v);
                    setPage(1);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="All" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    {SEGMENTS.map((s) => (
                      <SelectItem key={s} value={s}>
                        {s}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-4">
            {loading ? (
              <div className="space-y-2">
                {Array.from({ length: 10 }).map((_, i) => (
                  <Skeleton key={i} className="h-10 w-full" />
                ))}
              </div>
            ) : (
              <>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-16">ID</TableHead>
                      <TableHead>Name</TableHead>
                      <TableHead>Location</TableHead>
                      <TableHead>RFM Segment</TableHead>
                      <TableHead className="text-right">Orders</TableHead>
                      <TableHead className="text-right">Spend</TableHead>
                      <TableHead className="w-10"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data?.items.map((c) => (
                      <TableRow
                        key={c.customer_id}
                        className="cursor-pointer"
                        onClick={() => setSelected(c)}
                      >
                        <TableCell className="text-muted-foreground text-xs">
                          {c.customer_id}
                        </TableCell>
                        <TableCell className="font-medium text-sm">
                          {c.name}
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {c.location}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className={
                              segmentColors[c.rfm_segment || ""] ||
                              "bg-muted text-muted-foreground ring-white/12"
                            }
                          >
                            {c.rfm_segment || "Unknown"}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right text-sm">
                          {c.total_orders ?? 0}
                        </TableCell>
                        <TableCell className="text-right text-sm font-semibold text-primary">
                          $
                          {(c.total_spend ?? 0).toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="size-7"
                            onClick={(e) => {
                              e.stopPropagation();
                              router.push(`/?cid=${c.customer_id}`);
                            }}
                          >
                            <ArrowRight className="size-3.5" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>

                <div className="flex items-center justify-between mt-4">
                  <span className="text-xs text-muted-foreground">
                    {data?.total ?? 0} customers total
                  </span>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={page <= 1}
                      onClick={() => setPage((p) => p - 1)}
                    >
                      <ChevronLeft className="size-4" />
                    </Button>
                    <span className="text-xs text-muted-foreground">
                      Page {page} of {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={page >= totalPages}
                      onClick={() => setPage((p) => p + 1)}
                    >
                      <ChevronRight className="size-4" />
                    </Button>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <Sheet open={!!selected} onOpenChange={() => setSelected(null)}>
        <SheetContent>
          <SheetHeader>
            <SheetTitle>{selected?.name}</SheetTitle>
          </SheetHeader>
          {selected && (
            <div className="space-y-4 mt-4 px-1">
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Customer ID</span>
                  <span>{selected.customer_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Location</span>
                  <span>{selected.location}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Signed Up</span>
                  <span>{selected.signup_date}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Orders</span>
                  <span>{selected.total_orders ?? 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Spend</span>
                  <span className="font-bold text-primary">
                    $
                    {(selected.total_spend ?? 0).toLocaleString(undefined, {
                      maximumFractionDigits: 0,
                    })}
                  </span>
                </div>
              </div>
              <Separator />
              <div>
                <span className="text-xs text-muted-foreground uppercase tracking-wider">
                  RFM Segment
                </span>
                <div className="mt-1">
                  <Badge
                    variant="outline"
                    className={
                      segmentColors[selected.rfm_segment || ""] ||
                      "bg-muted text-muted-foreground ring-white/12"
                    }
                  >
                    {selected.rfm_segment || "Unknown"}
                  </Badge>
                </div>
              </div>
              <Separator />
              <Button
                className="w-full"
                onClick={() => {
                  setSelected(null);
                  router.push(`/?cid=${selected.customer_id}`);
                }}
              >
                <ArrowRight className="size-4 mr-2" />
                View Recommendations
              </Button>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </>
  );
}
