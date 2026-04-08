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

export interface ArpuWeeklyPoint {
  week_start: string;
  arpu: number;
  threshold: number;
  revenue: number;
  active_customers: number;
  orders: number;
}

export interface ArpuWorkflowRecommendation {
  id: string;
  phase: string;
  title: string;
  description: string;
  expected_arpu_lift_pct: number;
  owner: string;
  requires_approval: boolean;
}

export interface ArpuWeeklyAnalytics {
  metric: string;
  threshold: number;
  latest_value: number;
  below_threshold: boolean;
  gap_pct: number;
  weeks: ArpuWeeklyPoint[];
  root_causes: string[];
  suggested_actions: ArpuWorkflowRecommendation[];
}

export interface ArpuWorkflowPhase {
  id: string;
  name: string;
  human_actions: string[];
  agent_actions: string[];
}

export interface ArpuWorkflowPlan {
  triggered: boolean;
  status: "triggered" | "monitoring";
  trigger_metric: string;
  latest_arpu: number;
  threshold: number;
  trigger_reason: string;
  approval_required: boolean;
  phases: ArpuWorkflowPhase[];
  recommendations: ArpuWorkflowRecommendation[];
}

export interface SaveCustomerAlert {
  triggered: boolean;
  status: "triggered" | "monitoring";
  high_risk_count: number;
  monthly_threshold: number;
  threshold_gap: number;
  threshold_gap_pct: number;
}

export interface SaveCustomerValueTier {
  tier: "High Value" | "Medium Value" | "Low Value";
  criteria: string;
  count: number;
  priority: "P1" | "P2" | "P3";
}

export interface SaveCustomerWarningSigns {
  low_email_engagement_customers: number;
  purchase_gap_expanded_customers: number;
  price_complaint_signal_customers: number;
}

export interface SaveCustomerIntervention {
  tier: "High Value" | "Medium Value" | "Low Value";
  strategy: string;
  owner: string;
  requires_human_approval: boolean;
}

export interface SaveCustomerCampaignControls {
  escalation_days: number;
  assigned_high_value_accounts: number;
  human_follow_up_team: string;
}

export interface SaveCustomerPerformanceSnapshot {
  expected_redemption_rate_pct: number;
  expected_save_rate_pct: number;
  expected_saved_customers: number;
}

export interface SaveCustomerOutcomeBenchmark {
  offer_cost: number;
  retained_annual_value: number;
  roi_multiple: number;
}

export interface SaveCustomerPriorityAccount {
  customer_id: number;
  name: string;
  score: number;
  segment: string;
  annual_value: number;
  value_tier: "High Value" | "Medium Value" | "Low Value";
  warning_low_email_engagement: boolean;
  warning_purchase_gap_expanded: boolean;
  warning_price_complaint_proxy: boolean;
  last_gap_days: number;
  previous_gap_days: number;
}

export interface SaveCustomerAnalytics {
  alert: SaveCustomerAlert;
  value_tiers: SaveCustomerValueTier[];
  warning_signs: SaveCustomerWarningSigns;
  intervention_plan: SaveCustomerIntervention[];
  campaign_controls: SaveCustomerCampaignControls;
  performance_snapshot: SaveCustomerPerformanceSnapshot;
  outcome_benchmark: SaveCustomerOutcomeBenchmark;
  priority_accounts: SaveCustomerPriorityAccount[];
}

export interface SaveCustomerWorkflowPhase {
  id: string;
  name: string;
  human_actions: string[];
  agent_actions: string[];
}

export interface SaveCustomerWorkflowApproval {
  id: string;
  phase_id: string;
  title: string;
  description: string;
  owner: string;
  required_decision_from: string;
}

export interface SaveCustomerWorkflowPlan {
  triggered: boolean;
  status: "triggered" | "monitoring";
  trigger_metric: string;
  high_risk_count: number;
  monthly_threshold: number;
  trigger_reason: string;
  approval_required: boolean;
  phases: SaveCustomerWorkflowPhase[];
  approvals: SaveCustomerWorkflowApproval[];
  interventions: SaveCustomerIntervention[];
  escalation_days: number;
  expected_30_day_outcome: SaveCustomerPerformanceSnapshot;
  roi_projection: SaveCustomerOutcomeBenchmark;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
}
