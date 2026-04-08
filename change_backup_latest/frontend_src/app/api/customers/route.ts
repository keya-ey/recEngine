import { NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams.toString();
  const res = await fetch(`${BACKEND}/customers?${sp}`);
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
