"use client";

import { Suspense, useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { Header } from "@/components/layout/header";
import { CustomerSearch } from "@/components/recommendations/customer-search";
import { CustomerProfileCard } from "@/components/recommendations/customer-profile";
import { RecommendationsGrid } from "@/components/recommendations/recommendations-grid";
import { Skeleton } from "@/components/ui/skeleton";
import { getRecommendations } from "@/lib/api";
import type { RecommendResponse } from "@/lib/types";

function DashboardContent() {
  const searchParams = useSearchParams();
  const cidParam = searchParams.get("cid");
  const [customerId, setCustomerId] = useState(cidParam || "1");
  const [topK, setTopK] = useState("20");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState<RecommendResponse | null>(null);

  useEffect(() => {
    if (cidParam) {
      setCustomerId(cidParam);
      handleSubmit(cidParam);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cidParam]);

  async function handleSubmit(overrideCid?: string) {
    const cid = parseInt(overrideCid || customerId);
    if (isNaN(cid) || cid < 1) return;
    setLoading(true);
    setError("");
    try {
      const result = await getRecommendations(cid, parseInt(topK));
      setData(result);
    } catch (e) {
      setError(
        `Error: ${e instanceof Error ? e.message : "Unknown error"}. Make sure the backend is running.`
      );
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-4 space-y-4">
      <CustomerSearch
        customerId={customerId}
        topK={topK}
        loading={loading}
        onCustomerIdChange={setCustomerId}
        onTopKChange={setTopK}
        onSubmit={() => handleSubmit()}
      />

      {loading && (
        <div className="space-y-3">
          <Skeleton className="h-48 w-full" />
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-52" />
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="text-center py-8 text-destructive text-sm">
          {error}
        </div>
      )}

      {data && !loading && (
        <>
          <CustomerProfileCard
            customerId={data.customer_id}
            segment={data.rfm_segment}
            tags={data.behavioral_tags}
            profile={data.profile}
          />
          <h2 className="text-sm font-semibold text-muted-foreground">
            Recommended for you
          </h2>
          <RecommendationsGrid items={data.recommendations} />
        </>
      )}
    </div>
  );
}

export default function DashboardPage() {
  return (
    <>
      <Header title="Dashboard" />
      <Suspense
        fallback={
          <div className="p-4 space-y-3">
            <Skeleton className="h-10 w-96" />
            <Skeleton className="h-48 w-full" />
          </div>
        }
      >
        <DashboardContent />
      </Suspense>
    </>
  );
}
