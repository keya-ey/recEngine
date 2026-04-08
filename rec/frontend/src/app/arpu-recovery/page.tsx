"use client";

import { useEffect, useMemo, useState } from "react";
import { Header } from "@/components/layout/header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getArpuWorkflowPlan } from "@/lib/api";
import type { ArpuWorkflowPlan } from "@/lib/types";
import { Bot, CheckCircle2, Clock3, Loader2, XCircle } from "lucide-react";

type PhaseState = "pending" | "in_progress" | "completed";
type Decision = "approved" | "rejected";

const phaseLabel: Record<PhaseState, string> = {
  pending: "Pending",
  in_progress: "Processing",
  completed: "Completed",
};

export default function ArpuRecoveryPage() {
  const [plan, setPlan] = useState<ArpuWorkflowPlan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activePhaseIndex, setActivePhaseIndex] = useState(0);
  const [awaitingApproval, setAwaitingApproval] = useState(false);
  const [phaseStates, setPhaseStates] = useState<Record<string, PhaseState>>({});
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    async function load() {
      try {
        const response = await getArpuWorkflowPlan();
        const initialStates = Object.fromEntries(
          response.phases.map((p) => [p.id, "pending" as PhaseState])
        );
        setPlan(response);
        setPhaseStates(initialStates);
        setLogs([`Agent initialized for ${response.trigger_metric}.`]);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load workflow.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const approvalItems = useMemo(
    () => (plan?.recommendations ?? []).filter((r) => r.requires_approval),
    [plan]
  );
  const allDecided =
    approvalItems.length > 0 &&
    approvalItems.every((item) => decisions[item.id] !== undefined);

  useEffect(() => {
    if (!plan || !plan.triggered || awaitingApproval) return;
    if (activePhaseIndex >= plan.phases.length) return;

    const phase = plan.phases[activePhaseIndex];
    const currentState = phaseStates[phase.id];
    if (!currentState) return;

    if (currentState === "pending") {
      setPhaseStates((prev) => ({ ...prev, [phase.id]: "in_progress" }));
      setLogs((prev) => [...prev, `Running ${phase.name} phase...`]);
    }

    const timer = setTimeout(() => {
      if (phase.id === "plan") {
        const pendingApprovals = approvalItems.filter((item) => !decisions[item.id]).length;
        if (pendingApprovals > 0) {
          setAwaitingApproval(true);
          setLogs((prev) => [
            ...prev,
            "Marketing manager approval required: approve/reject recommendations.",
          ]);
          return;
        }
      }

      setPhaseStates((prev) => ({ ...prev, [phase.id]: "completed" }));
      setLogs((prev) => [...prev, `${phase.name} phase completed.`]);
      setActivePhaseIndex((prev) => prev + 1);
    }, 1800);

    return () => clearTimeout(timer);
  }, [activePhaseIndex, approvalItems, awaitingApproval, decisions, phaseStates, plan]);

  function setDecision(recommendationId: string, decision: Decision) {
    setDecisions((prev) => ({ ...prev, [recommendationId]: decision }));
    setLogs((prev) => [...prev, `Recommendation ${recommendationId}: ${decision}.`]);
  }

  function continueExecution() {
    if (!plan || !allDecided) return;
    const planPhase = plan.phases.find((p) => p.id === "plan");
    if (planPhase) {
      setPhaseStates((prev) => ({ ...prev, [planPhase.id]: "completed" }));
    }
    setAwaitingApproval(false);
    setActivePhaseIndex((prev) => Math.max(prev + 1, 2));
    setLogs((prev) => [...prev, "Manager decisions captured. Resuming automated execution."]);
  }

  if (loading) {
    return (
      <>
        <Header title="ARPU Recovery Agent" />
        <div className="p-4 space-y-4">
          <Skeleton className="h-28" />
          <Skeleton className="h-80" />
          <Skeleton className="h-60" />
        </div>
      </>
    );
  }

  if (error || !plan) {
    return (
      <>
        <Header title="ARPU Recovery Agent" />
        <div className="p-4 text-sm text-destructive">Failed to load workflow: {error || "Unknown error"}</div>
      </>
    );
  }

  const approvedCount = approvalItems.filter((item) => decisions[item.id] === "approved").length;
  const rejectedCount = approvalItems.filter((item) => decisions[item.id] === "rejected").length;
  const done = activePhaseIndex >= plan.phases.length;

  return (
    <>
      <Header title="ARPU Recovery Agent" />
      <div className="p-4 space-y-4">
        <Card>
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle className="text-sm">Workflow Status</CardTitle>
              <Badge variant={plan.triggered ? "destructive" : "secondary"}>
                {plan.status.toUpperCase()}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>{plan.trigger_reason}</p>
            <p>
              Latest ARPU: <span className="text-foreground font-semibold">${plan.latest_arpu.toFixed(2)}</span> ·
              Threshold: <span className="text-foreground font-semibold"> ${plan.threshold.toFixed(2)}</span>
            </p>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <div className="xl:col-span-2 space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Automated Phases</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {plan.phases.map((phase) => {
                  const status = phaseStates[phase.id] ?? "pending";
                  return (
                    <div key={phase.id} className="rounded-2xl bg-muted/25 p-3 space-y-2 ring-1 ring-white/10">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="font-medium">{phase.name}</div>
                        <Badge variant={status === "completed" ? "secondary" : status === "in_progress" ? "default" : "outline"}>
                          {phaseLabel[status]}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-muted-foreground">
                        <div>
                          <div className="mb-1 uppercase tracking-wider">Human</div>
                          <ul className="space-y-1">
                            {phase.human_actions.map((item) => (
                              <li key={item}>• {item}</li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <div className="mb-1 uppercase tracking-wider">Agent</div>
                          <ul className="space-y-1">
                            {phase.agent_actions.map((item) => (
                              <li key={item}>• {item}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <CardTitle className="text-sm">Manager Approval Queue</CardTitle>
                  <div className="flex items-center gap-2 text-xs">
                    <Badge variant="secondary">Approved: {approvedCount}</Badge>
                    <Badge variant="outline">Rejected: {rejectedCount}</Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Marketing manager: please approve or reject each recommendation before execution continues.
                </p>
                {approvalItems.map((rec) => {
                  const decision = decisions[rec.id];
                  return (
                    <div key={rec.id} className="rounded-2xl bg-muted/25 p-3 space-y-2 ring-1 ring-white/10">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="font-medium text-sm">{rec.title}</div>
                        <Badge variant={decision === "approved" ? "secondary" : decision === "rejected" ? "destructive" : "outline"}>
                          {decision ? decision.toUpperCase() : "PENDING"}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">{rec.description}</p>
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <p className="text-xs text-muted-foreground">
                          Owner: {rec.owner} · Expected ARPU lift: {rec.expected_arpu_lift_pct.toFixed(1)}%
                        </p>
                        <div className="flex gap-2">
                          <Button size="xs" variant="secondary" onClick={() => setDecision(rec.id, "approved")}>
                            Approve
                          </Button>
                          <Button size="xs" variant="destructive" onClick={() => setDecision(rec.id, "rejected")}>
                            Reject
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div className="flex justify-end">
                  <Button onClick={continueExecution} disabled={!awaitingApproval || !allDecided}>
                    Continue Automated Execution
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Agent Console</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center gap-2 text-sm">
                {done ? (
                  <CheckCircle2 className="size-4 text-emerald-500" />
                ) : awaitingApproval ? (
                  <Clock3 className="size-4 text-amber-500" />
                ) : (
                  <Loader2 className="size-4 animate-spin text-primary" />
                )}
                <span className="font-medium">
                  {done ? "Workflow completed" : awaitingApproval ? "Waiting for approval" : "Automation running"}
                </span>
              </div>
              <div className="h-[420px] overflow-auto space-y-2 rounded-2xl bg-muted/30 p-3 ring-1 ring-white/10">
                {logs.map((entry, idx) => (
                  <div key={`${entry}-${idx}`} className="text-xs text-muted-foreground flex items-start gap-2">
                    <Bot className="size-3 mt-0.5 text-primary shrink-0" />
                    <span>{entry}</span>
                  </div>
                ))}
              </div>
              {done && (
                <div className="rounded-2xl bg-emerald-500/10 p-3 text-xs text-emerald-300 ring-1 ring-emerald-500/35">
                  Recovery cycle finished. Monitor weekly ARPU trend for sustained uplift.
                </div>
              )}
              {awaitingApproval && (
                <div className="flex items-center gap-2 rounded-2xl bg-amber-500/10 p-3 text-xs text-amber-300 ring-1 ring-amber-500/35">
                  <XCircle className="size-4" />
                  Awaiting manager decisions on all recommendations.
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
