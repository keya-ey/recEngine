"use client";

import { useEffect, useMemo, useState } from "react";
import { Header } from "@/components/layout/header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getSaveCustomerWorkflowPlan } from "@/lib/api";
import type { SaveCustomerWorkflowPlan } from "@/lib/types";
import { Bot, CheckCircle2, Clock3, Loader2, ShieldAlert } from "lucide-react";

type PhaseState = "pending" | "in_progress" | "completed";
type Decision = "approved" | "rejected";

const phaseLabel: Record<PhaseState, string> = {
  pending: "Pending",
  in_progress: "Processing",
  completed: "Completed",
};

export default function SaveTheCustomerPage() {
  const [plan, setPlan] = useState<SaveCustomerWorkflowPlan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activePhaseIndex, setActivePhaseIndex] = useState(0);
  const [awaitingApproval, setAwaitingApproval] = useState(false);
  const [approvalPhaseId, setApprovalPhaseId] = useState<string | null>(null);
  const [phaseStates, setPhaseStates] = useState<Record<string, PhaseState>>({});
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    async function load() {
      try {
        const response = await getSaveCustomerWorkflowPlan();
        const initialStates = Object.fromEntries(
          response.phases.map((p) => [p.id, "pending" as PhaseState])
        );
        setPlan(response);
        setPhaseStates(initialStates);
        setLogs([
          `Save Campaign Agent initialized (${response.high_risk_count} high-risk customers).`,
        ]);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load workflow.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  useEffect(() => {
    if (!plan || !plan.triggered || awaitingApproval) return;
    if (activePhaseIndex >= plan.phases.length) return;

    const phase = plan.phases[activePhaseIndex];
    const currentState = phaseStates[phase.id];
    if (!currentState) return;

    if (currentState === "pending") {
      setPhaseStates((prev) => ({ ...prev, [phase.id]: "in_progress" }));
      setLogs((prev) => [...prev, `Running "${phase.name}"...`]);
    }

    const timer = setTimeout(() => {
      const pendingApprovals = plan.approvals.filter(
        (item) => item.phase_id === phase.id && decisions[item.id] === undefined
      );
      if (pendingApprovals.length > 0) {
        setAwaitingApproval(true);
        setApprovalPhaseId(phase.id);
        setLogs((prev) => [
          ...prev,
          `Human approval required for phase "${phase.name}" before continuing.`,
        ]);
        return;
      }

      setPhaseStates((prev) => ({ ...prev, [phase.id]: "completed" }));
      setLogs((prev) => [...prev, `"${phase.name}" completed.`]);
      setActivePhaseIndex((prev) => prev + 1);
    }, 1800);

    return () => clearTimeout(timer);
  }, [activePhaseIndex, awaitingApproval, decisions, phaseStates, plan]);

  const approvalItems = useMemo(() => plan?.approvals ?? [], [plan]);
  const pendingForCurrentPhase = useMemo(() => {
    if (!approvalPhaseId) return [];
    return approvalItems.filter((item) => item.phase_id === approvalPhaseId);
  }, [approvalItems, approvalPhaseId]);
  const allCurrentPhaseDecided =
    pendingForCurrentPhase.length > 0 &&
    pendingForCurrentPhase.every((item) => decisions[item.id] !== undefined);

  function setDecision(approvalId: string, decision: Decision) {
    setDecisions((prev) => ({ ...prev, [approvalId]: decision }));
    setLogs((prev) => [...prev, `Approval ${approvalId}: ${decision}.`]);
  }

  function continueExecution() {
    if (!plan || !approvalPhaseId || !allCurrentPhaseDecided) return;
    const phaseIndex = plan.phases.findIndex((phase) => phase.id === approvalPhaseId);
    if (phaseIndex === -1) return;

    setPhaseStates((prev) => ({ ...prev, [approvalPhaseId]: "completed" }));
    setAwaitingApproval(false);
    setApprovalPhaseId(null);
    setActivePhaseIndex((prev) => Math.max(prev, phaseIndex + 1));
    setLogs((prev) => [...prev, "Human decisions recorded. Resuming agent execution."]);
  }

  if (loading) {
    return (
      <>
        <Header title="Save-the-Customer Agent" />
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
        <Header title="Save-the-Customer Agent" />
        <div className="p-4 text-sm text-destructive">Failed to load workflow: {error || "Unknown error"}</div>
      </>
    );
  }

  const approvedCount = approvalItems.filter((item) => decisions[item.id] === "approved").length;
  const rejectedCount = approvalItems.filter((item) => decisions[item.id] === "rejected").length;
  const done = activePhaseIndex >= plan.phases.length;

  return (
    <>
      <Header title="Save-the-Customer Agent" />
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
              High-risk customers:{" "}
              <span className="text-foreground font-semibold">{plan.high_risk_count}</span> ·
              Monthly threshold:{" "}
              <span className="text-foreground font-semibold"> {plan.monthly_threshold}</span> ·
              Escalation window:{" "}
              <span className="text-foreground font-semibold"> {plan.escalation_days} days</span>
            </p>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <div className="xl:col-span-2 space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Multi-Step Execution</CardTitle>
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
                  <CardTitle className="text-sm">Human Approval Queue</CardTitle>
                  <div className="flex items-center gap-2 text-xs">
                    <Badge variant="secondary">Approved: {approvedCount}</Badge>
                    <Badge variant="outline">Rejected: {rejectedCount}</Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Agent execution pauses at every required checkpoint. Human decisions are required to move forward.
                </p>
                {approvalItems.map((item) => {
                  const decision = decisions[item.id];
                  return (
                    <div key={item.id} className="rounded-2xl bg-muted/25 p-3 space-y-2 ring-1 ring-white/10">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="font-medium text-sm">{item.title}</div>
                        <Badge variant={decision === "approved" ? "secondary" : decision === "rejected" ? "destructive" : "outline"}>
                          {decision ? decision.toUpperCase() : "PENDING"}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">{item.description}</p>
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <p className="text-xs text-muted-foreground">
                          Owner: {item.owner} · Phase: {item.phase_id}
                        </p>
                        <div className="flex gap-2">
                          <Button size="xs" variant="secondary" onClick={() => setDecision(item.id, "approved")}>
                            Approve
                          </Button>
                          <Button size="xs" variant="destructive" onClick={() => setDecision(item.id, "rejected")}>
                            Reject
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div className="flex justify-end">
                  <Button onClick={continueExecution} disabled={!awaitingApproval || !allCurrentPhaseDecided}>
                    Continue Workflow
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
                  {done ? "Workflow completed" : awaitingApproval ? "Waiting for human approval" : "Automation running"}
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
                  Rescue mission finished. Review final save rate and ROI for rollout decisions.
                </div>
              )}
              {awaitingApproval && (
                <div className="flex items-center gap-2 rounded-2xl bg-amber-500/10 p-3 text-xs text-amber-300 ring-1 ring-amber-500/35">
                  <ShieldAlert className="size-4" />
                  Approval gate active: complete all decisions for current phase.
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
