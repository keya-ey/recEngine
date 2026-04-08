import type {
  RecommendResponse,
  PaginatedResponse,
  Product,
  Customer,
  AnalyticsSummary,
  SegmentData,
  TagData,
  CategoryRevenue,
  BrandRevenue,
} from "./types";

const BASE = "/api";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function getRecommendations(
  customerId: number,
  topK: number = 20
): Promise<RecommendResponse> {
  return fetchJson(`${BASE}/recommend/${customerId}?top_k=${topK}`);
}

export async function getProducts(params: {
  category?: string;
  brand?: string;
  search?: string;
  page?: number;
  limit?: number;
}): Promise<PaginatedResponse<Product>> {
  const sp = new URLSearchParams();
  if (params.category) sp.set("category", params.category);
  if (params.brand) sp.set("brand", params.brand);
  if (params.search) sp.set("search", params.search);
  if (params.page) sp.set("page", String(params.page));
  if (params.limit) sp.set("limit", String(params.limit));
  return fetchJson(`${BASE}/products?${sp.toString()}`);
}

export async function getCustomers(params: {
  segment?: string;
  tag?: string;
  search?: string;
  page?: number;
  limit?: number;
}): Promise<PaginatedResponse<Customer>> {
  const sp = new URLSearchParams();
  if (params.segment) sp.set("segment", params.segment);
  if (params.tag) sp.set("tag", params.tag);
  if (params.search) sp.set("search", params.search);
  if (params.page) sp.set("page", String(params.page));
  if (params.limit) sp.set("limit", String(params.limit));
  return fetchJson(`${BASE}/customers?${sp.toString()}`);
}

export async function getAnalyticsSummary(): Promise<AnalyticsSummary> {
  return fetchJson(`${BASE}/analytics?type=summary`);
}

export async function getSegments(): Promise<SegmentData[]> {
  return fetchJson(`${BASE}/analytics?type=segments`);
}

export async function getTags(): Promise<TagData[]> {
  return fetchJson(`${BASE}/analytics?type=tags`);
}

export async function getTopCategories(): Promise<CategoryRevenue[]> {
  return fetchJson(`${BASE}/analytics?type=top-categories`);
}

export async function getTopBrands(): Promise<BrandRevenue[]> {
  return fetchJson(`${BASE}/analytics?type=top-brands`);
}
