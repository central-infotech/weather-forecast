import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { google } from "googleapis";
import { authOptions } from "@/lib/auth";
import { getSupabaseAdmin } from "@/lib/supabase";

/** Map weather text to Google Calendar color ID. */
function weatherColorId(weather: string): string {
  if (weather.includes("晴れ")) return "5";
  if (weather.includes("曇り")) return "8";
  if (weather.includes("雨")) return "9";
  if (weather.includes("雪")) return "1";
  return "0";
}

interface Forecast {
  id: string;
  date: string;
  weather: string;
  temp_max: number;
  temp_min?: number;
  precipitation_prob?: number;
  confidence?: number;
  model_agreement?: number;
  location?: string;
}

interface SyncResult {
  forecast_id: string;
  google_event_id: string | null;
  status: "synced" | "failed";
  error?: string;
}

export async function POST(request: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session?.accessToken) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: { forecast_ids: string[] };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!Array.isArray(body.forecast_ids) || body.forecast_ids.length === 0) {
    return NextResponse.json(
      { error: "forecast_ids must be a non-empty array" },
      { status: 400 },
    );
  }

  // Fetch the requested forecasts from Supabase
  const supabaseAdmin = getSupabaseAdmin();

  const { data: forecasts, error: fetchError } = await supabaseAdmin
    .from("forecasts")
    .select("*")
    .in("id", body.forecast_ids);

  if (fetchError) {
    return NextResponse.json(
      { error: "Failed to fetch forecasts", details: fetchError.message },
      { status: 500 },
    );
  }

  if (!forecasts || forecasts.length === 0) {
    return NextResponse.json(
      { error: "No forecasts found for the given IDs" },
      { status: 404 },
    );
  }

  // Set up Google Calendar API client
  const oauth2Client = new google.auth.OAuth2();
  oauth2Client.setCredentials({ access_token: session.accessToken });
  const calendar = google.calendar({ version: "v3", auth: oauth2Client });

  const results: SyncResult[] = [];

  for (const forecast of forecasts as Forecast[]) {
    try {
      const description = [
        `最高気温: ${forecast.temp_max}℃`,
        forecast.temp_min != null ? `最低気温: ${forecast.temp_min}℃` : null,
        forecast.precipitation_prob != null
          ? `降水確率: ${forecast.precipitation_prob}%`
          : null,
        forecast.confidence != null
          ? `信頼度: ${forecast.confidence}`
          : null,
        forecast.model_agreement != null
          ? `モデル一致度: ${forecast.model_agreement}`
          : null,
      ]
        .filter(Boolean)
        .join("\n");

      const event = await calendar.events.insert({
        calendarId: "primary",
        requestBody: {
          summary: `【AI予報】${forecast.weather}/${forecast.temp_max}℃`,
          description,
          start: { date: forecast.date },
          end: { date: forecast.date },
          colorId: weatherColorId(forecast.weather),
        },
      });

      results.push({
        forecast_id: forecast.id,
        google_event_id: event.data.id ?? null,
        status: "synced",
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unknown error";
      results.push({
        forecast_id: forecast.id,
        google_event_id: null,
        status: "failed",
        error: message,
      });
    }
  }

  // Log sync results to Supabase
  const logEntries = results.map((r) => ({
    forecast_id: r.forecast_id,
    google_event_id: r.google_event_id,
    status: r.status,
    error: r.error ?? null,
    synced_at: new Date().toISOString(),
  }));

  await supabaseAdmin.from("calendar_sync_log").insert(logEntries);

  const synced = results.filter((r) => r.status === "synced").length;
  const failed = results.filter((r) => r.status === "failed").length;

  return NextResponse.json({ synced, failed, events: results });
}
