import { NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const topK = req.nextUrl.searchParams.get("top_k") || "20";
  const res = await fetch(`${BACKEND}/recommend/${id}?top_k=${topK}`);
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
