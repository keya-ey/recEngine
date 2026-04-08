"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import {
  LayoutDashboard,
  Package,
  Users,
  BarChart3,
  Workflow,
  LifeBuoy,
} from "lucide-react";

const navItems = [
  { title: "Dashboard", href: "/", icon: LayoutDashboard },
  { title: "Products", href: "/products", icon: Package },
  { title: "Customers", href: "/customers", icon: Users },
  { title: "Analytics", href: "/analytics", icon: BarChart3 },
  { title: "ARPU Recovery", href: "/arpu-recovery", icon: Workflow },
  { title: "Save the Customer", href: "/save-the-customer", icon: LifeBuoy },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <Sidebar>
      <SidebarHeader className="px-4 py-5">
        <div className="text-xs font-semibold tracking-[3px] uppercase text-primary">
          RecEngine
        </div>
        <div className="text-[0.7rem] text-muted-foreground">
          Tag-Aware Recommendations
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    render={<Link href={item.href} />}
                    isActive={
                      item.href === "/"
                        ? pathname === "/"
                        : pathname.startsWith(item.href)
                    }
                  >
                    <item.icon className="size-4" />
                    <span>{item.title}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
