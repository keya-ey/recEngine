"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
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
import { ChevronLeft, ChevronRight, Search } from "lucide-react";
import { getProducts } from "@/lib/api";
import type { Product, PaginatedResponse } from "@/lib/types";

const CATEGORIES = [
  "Electronics",
  "Clothing",
  "Home & Kitchen",
  "Beauty",
  "Sports & Outdoors",
  "Books",
  "Toys & Games",
  "Grocery",
  "Automotive",
  "Health & Wellness",
];

const BRANDS = [
  "TechNova", "UrbanEdge", "HomeHaven", "GlowUp", "ActivePeak",
  "PageTurner", "FunZone", "FreshPick", "AutoPro", "VitaLife",
  "ElectraWave", "StyleCraft", "ComfortPlus", "PureSkin", "TrailBlaze",
  "ReadMore", "PlayStar", "NatureBite", "DriveMax", "WellnessHub",
];

const BRANDS_BY_CATEGORY: Record<string, string[]> = {
  Electronics: ["TechNova", "ElectraWave"],
  Clothing: ["UrbanEdge", "StyleCraft"],
  "Home & Kitchen": ["HomeHaven", "ComfortPlus"],
  Beauty: ["GlowUp", "PureSkin"],
  "Sports & Outdoors": ["ActivePeak", "TrailBlaze"],
  Books: ["PageTurner", "ReadMore"],
  "Toys & Games": ["FunZone", "PlayStar"],
  Grocery: ["FreshPick", "NatureBite"],
  Automotive: ["AutoPro", "DriveMax"],
  "Health & Wellness": ["VitaLife", "WellnessHub"],
};

const ALL_OPTION = "All";

export default function ProductsPage() {
  const [data, setData] = useState<PaginatedResponse<Product> | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("");
  const [brand, setBrand] = useState("");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<Product | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getProducts({
        search: search || undefined,
        category: category || undefined,
        brand: brand || undefined,
        page,
        limit: 20,
      });
      setData(result);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [search, category, brand, page]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const availableBrands = useMemo(() => {
    if (!category) return BRANDS;
    return BRANDS_BY_CATEGORY[category] ?? BRANDS;
  }, [category]);

  const totalPages = data ? Math.ceil(data.total / data.limit) : 0;

  return (
    <>
      <Header title="Products" />
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
                    placeholder="Search products..."
                    className="pl-8"
                    value={search}
                    onChange={(e) => {
                      setSearch(e.target.value);
                      setPage(1);
                    }}
                  />
                </div>
              </div>
              <div className="w-40 space-y-1">
                <label className="text-[0.68rem] font-medium text-muted-foreground uppercase tracking-wider">
                  Category
                </label>
                <Select
                  value={category || ALL_OPTION}
                  onValueChange={(v) => {
                    const nextCategory = !v || v === ALL_OPTION ? "" : v;
                    setCategory(nextCategory);
                    setBrand((prevBrand) => {
                      if (!nextCategory || !prevBrand) return prevBrand;
                      const allowed = BRANDS_BY_CATEGORY[nextCategory] ?? BRANDS;
                      return allowed.includes(prevBrand) ? prevBrand : "";
                    });
                    setPage(1);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="All" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={ALL_OPTION}>All</SelectItem>
                    {CATEGORIES.map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="w-40 space-y-1">
                <label className="text-[0.68rem] font-medium text-muted-foreground uppercase tracking-wider">
                  Brand
                </label>
                <Select
                  key={`brand-${category || "all"}`}
                  value={brand || ALL_OPTION}
                  onValueChange={(v) => {
                    setBrand(!v || v === ALL_OPTION ? "" : v);
                    setPage(1);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="All" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={ALL_OPTION}>All</SelectItem>
                    {availableBrands.map((b) => (
                      <SelectItem key={b} value={b}>
                        {b}
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
                      <TableHead>Category</TableHead>
                      <TableHead>Brand</TableHead>
                      <TableHead className="text-right">Price</TableHead>
                      <TableHead>Tags</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data?.items.map((p) => (
                      <TableRow
                        key={p.product_id}
                        className="cursor-pointer"
                        onClick={() => setSelected(p)}
                      >
                        <TableCell className="text-muted-foreground text-xs">
                          {p.product_id}
                        </TableCell>
                        <TableCell className="font-medium text-sm">
                          {p.name}
                        </TableCell>
                        <TableCell className="text-sm">{p.category}</TableCell>
                        <TableCell className="text-sm">{p.brand}</TableCell>
                        <TableCell className="text-right text-sm font-semibold text-primary">
                          ${p.price.toFixed(2)}
                        </TableCell>
                        <TableCell>
                          <div className="flex gap-1 flex-wrap">
                            {p.tags
                              .split("|")
                              .filter(Boolean)
                              .map((t) => (
                                <Badge
                                  key={t}
                                  variant="secondary"
                                  className="text-[0.6rem]"
                                >
                                  {t.trim()}
                                </Badge>
                              ))}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>

                <div className="flex items-center justify-between mt-4">
                  <span className="text-xs text-muted-foreground">
                    {data?.total ?? 0} products total
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
                  <span className="text-muted-foreground">Product ID</span>
                  <span>{selected.product_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Category</span>
                  <span>{selected.category}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Brand</span>
                  <span>{selected.brand}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Price</span>
                  <span className="font-bold text-primary">
                    ${selected.price.toFixed(2)}
                  </span>
                </div>
              </div>
              <div>
                <span className="text-xs text-muted-foreground uppercase tracking-wider">
                  Tags
                </span>
                <div className="flex gap-1 flex-wrap mt-1">
                  {selected.tags
                    .split("|")
                    .filter(Boolean)
                    .map((t) => (
                      <Badge key={t} variant="secondary">
                        {t.trim()}
                      </Badge>
                    ))}
                </div>
              </div>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </>
  );
}
