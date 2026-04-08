"use client";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search } from "lucide-react";

interface CustomerSearchProps {
  customerId: string;
  topK: string;
  loading: boolean;
  onCustomerIdChange: (v: string) => void;
  onTopKChange: (v: string) => void;
  onSubmit: () => void;
}

export function CustomerSearch({
  customerId,
  topK,
  loading,
  onCustomerIdChange,
  onTopKChange,
  onSubmit,
}: CustomerSearchProps) {
  return (
    <div className="flex items-end gap-3 max-w-xl">
      <div className="flex-1 space-y-1">
        <label className="text-[0.68rem] font-medium text-muted-foreground uppercase tracking-wider">
          Customer ID
        </label>
        <Input
          type="number"
          placeholder="e.g. 1"
          value={customerId}
          onChange={(e) => onCustomerIdChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSubmit()}
          min={1}
        />
      </div>
      <div className="w-28 space-y-1">
        <label className="text-[0.68rem] font-medium text-muted-foreground uppercase tracking-wider">
          Results
        </label>
        <Select value={topK} onValueChange={(v) => onTopKChange(v ?? "20")}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="10">10</SelectItem>
            <SelectItem value="20">20</SelectItem>
            <SelectItem value="50">50</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <Button onClick={onSubmit} disabled={loading}>
        <Search className="size-4 mr-1.5" />
        {loading ? "Loading..." : "Recommend"}
      </Button>
    </div>
  );
}
