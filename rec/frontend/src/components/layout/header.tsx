"use client";

import { SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";

export function Header({ title }: { title: string }) {
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center gap-3 bg-background/45 px-4 backdrop-blur-xl shadow-[0_8px_28px_-20px_rgba(0,0,0,0.95)]">
      <SidebarTrigger />
      <Separator orientation="vertical" className="h-5" />
      <h1 className="text-sm font-semibold">{title}</h1>
    </header>
  );
}
