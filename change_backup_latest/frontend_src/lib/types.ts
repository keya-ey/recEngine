export interface Recommendation {
  product_id: number;
  score: number;
  name: string;
  category: string;
  brand: string;
  price: number;
  tags: string;
  explanation: string;
}

export interface CustomerProfile {
  total_orders: number;
  total_items: number;
  total_spend: number;
  avg_order_value: number;
  avg_item_price: number;
  price_range: { min: number; max: number };
  top_categories: { name: string; count: number }[];
  top_brands: { name: string; count: number }[];
  recent_orders: {
    order_id: number;
    date: string;
    product: string;
    category: string;
    brand: string;
  }[];
  browsed_categories?: { name: string; views: number }[];
  total_product_views?: number;
}

export interface RecommendResponse {
  customer_id: number;
  rfm_segment: string;
  behavioral_tags: string[];
  profile: CustomerProfile;
  recommendations: Recommendation[];
}

export interface Product {
  product_id: number;
  name: string;
  category: string;
  brand: string;
  price: number;
  tags: string;
}

export interface Customer {
  customer_id: number;
  name: string;
  location: string;
  signup_date: string;
  rfm_segment?: string;
  behavioral_tags?: string[];
  total_orders?: number;
  total_spend?: number;
}

export interface AnalyticsSummary {
  total_customers: number;
  total_products: number;
  total_orders: number;
  total_revenue: number;
}

export interface SegmentData {
  segment: string;
  count: number;
}

export interface TagData {
  tag: string;
  count: number;
}

export interface CategoryRevenue {
  category: string;
  revenue: number;
}

export interface BrandRevenue {
  brand: string;
  revenue: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
}
