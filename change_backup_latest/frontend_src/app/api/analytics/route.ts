import { NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(req: NextRequest) {
  const type = req.nextUrl.searchParams.get("type") || "summary";
  const res = await fetch(`${BACKEND}/analytics?type=${type}`);
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
