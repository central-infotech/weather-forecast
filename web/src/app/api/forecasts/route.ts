import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { getSupabaseAdmin } from "@/lib/supabase";

export async function GET(request: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { searchParams } = request.nextUrl;
  const dateFrom = searchParams.get("date_from");
  const dateTo = searchParams.get("date_to");
  const location = searchParams.get("location");

  const supabaseAdmin = getSupabaseAdmin();
  let query = supabaseAdmin.from("forecasts").select("*");

  if (dateFrom) {
    query = query.gte("date", dateFrom);
  }
  if (dateTo) {
    query = query.lte("date", dateTo);
  }
  if (location) {
    query = query.eq("location", location);
  }

  query = query.order("date", { ascending: true });

  const { data, error } = await query;

  if (error) {
    return NextResponse.json(
      { error: "Failed to fetch forecasts", details: error.message },
      { status: 500 },
    );
  }

  return NextResponse.json(data);
}
