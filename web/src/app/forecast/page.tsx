"use client";

import { useSession, signOut } from "next-auth/react";
import { useState } from "react";

interface Forecast {
  id: string;
  date: string;
  location: string;
  weather: string;
  temp_max: number;
  temp_min: number;
  precipitation_prob: number;
  confidence: number;
}

const CITIES = [
  "東京",
  "大阪",
  "名古屋",
  "札幌",
  "福岡",
  "仙台",
  "広島",
  "那覇",
];

const WEATHER_EMOJI: Record<string, string> = {
  晴れ: "\u2600\uFE0F",
  曇り: "\u2601\uFE0F",
  雨: "\uD83C\uDF27\uFE0F",
  雪: "\u2744\uFE0F",
  曇時々晴: "\u26C5",
  曇時々雨: "\uD83C\uDF26\uFE0F",
  晴時々曇: "\uD83C\uDF24\uFE0F",
};

type DateMode = "range" | "pinpoint";
type Tab = "forecast" | "calendar";

export default function ForecastPage() {
  const { data: session } = useSession();

  const [activeTab, setActiveTab] = useState<Tab>("forecast");
  const [location, setLocation] = useState(CITIES[0]);
  const [dateMode, setDateMode] = useState<DateMode>("range");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [pinpointDate, setPinpointDate] = useState("");
  const [forecasts, setForecasts] = useState<Forecast[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [syncing, setSyncing] = useState(false);
  const [syncMessage, setSyncMessage] = useState("");

  const fetchForecasts = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setForecasts([]);
    setLoading(true);
    setSyncMessage("");

    const params = new URLSearchParams({ location });
    if (dateMode === "range") {
      if (dateFrom) params.set("date_from", dateFrom);
      if (dateTo) params.set("date_to", dateTo);
    } else {
      if (pinpointDate) {
        params.set("date_from", pinpointDate);
        params.set("date_to", pinpointDate);
      }
    }

    try {
      const res = await fetch(`/api/forecasts?${params.toString()}`);
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error ?? `Error ${res.status}`);
      }
      const data: Forecast[] = await res.json();
      setForecasts(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "予測の取得に失敗しました");
    } finally {
      setLoading(false);
    }
  };

  const syncToCalendar = async () => {
    setSyncing(true);
    setSyncMessage("");
    try {
      const res = await fetch("/api/calendar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ forecasts }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error ?? `Error ${res.status}`);
      }
      setSyncMessage("Google カレンダーに登録しました");
    } catch (err) {
      setSyncMessage(
        err instanceof Error ? err.message : "カレンダー登録に失敗しました",
      );
    } finally {
      setSyncing(false);
    }
  };

  const confidenceColor = (c: number) => {
    if (c > 0.7) return "bg-green-500";
    if (c > 0.4) return "bg-yellow-400";
    return "bg-red-500";
  };

  const weatherEmoji = (weather: string) =>
    WEATHER_EMOJI[weather] ?? weather;

  return (
    <div className="flex flex-col flex-1 bg-gray-50">
      {/* Header */}
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <h1 className="text-lg font-semibold text-gray-900">
            AI Weather Forecast
          </h1>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600">
              {session?.user?.name}
            </span>
            <button
              onClick={() => signOut({ callbackUrl: "/" })}
              className="rounded-md bg-gray-100 px-3 py-1.5 text-sm font-medium text-gray-700 transition hover:bg-gray-200"
            >
              ログアウト
            </button>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="border-b bg-white">
        <div className="mx-auto flex max-w-5xl px-4">
          <button
            onClick={() => setActiveTab("forecast")}
            className={`border-b-2 px-4 py-2.5 text-sm font-medium transition ${
              activeTab === "forecast"
                ? "border-blue-600 text-blue-600"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            予測取得
          </button>
          <button
            onClick={() => setActiveTab("calendar")}
            className={`border-b-2 px-4 py-2.5 text-sm font-medium transition ${
              activeTab === "calendar"
                ? "border-blue-600 text-blue-600"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            カレンダー連携
          </button>
        </div>
      </div>

      {/* Main content */}
      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-6">
        {activeTab === "forecast" ? (
          <div className="space-y-6">
            {/* Input form */}
            <form
              onSubmit={fetchForecasts}
              className="rounded-lg border bg-white p-5 shadow-sm"
            >
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {/* Location */}
                <div>
                  <label className="mb-1 block text-sm font-medium text-gray-700">
                    地域
                  </label>
                  <select
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  >
                    {CITIES.map((city) => (
                      <option key={city} value={city}>
                        {city}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Date mode toggle */}
                <div>
                  <label className="mb-1 block text-sm font-medium text-gray-700">
                    日付指定
                  </label>
                  <div className="flex rounded-md border border-gray-300 shadow-sm">
                    <button
                      type="button"
                      onClick={() => setDateMode("range")}
                      className={`flex-1 rounded-l-md px-3 py-2 text-sm font-medium transition ${
                        dateMode === "range"
                          ? "bg-blue-600 text-white"
                          : "bg-white text-gray-600 hover:bg-gray-50"
                      }`}
                    >
                      期間
                    </button>
                    <button
                      type="button"
                      onClick={() => setDateMode("pinpoint")}
                      className={`flex-1 rounded-r-md px-3 py-2 text-sm font-medium transition ${
                        dateMode === "pinpoint"
                          ? "bg-blue-600 text-white"
                          : "bg-white text-gray-600 hover:bg-gray-50"
                      }`}
                    >
                      ピンポイント
                    </button>
                  </div>
                </div>

                {/* Date inputs */}
                {dateMode === "range" ? (
                  <>
                    <div>
                      <label className="mb-1 block text-sm font-medium text-gray-700">
                        開始日
                      </label>
                      <input
                        type="date"
                        value={dateFrom}
                        onChange={(e) => setDateFrom(e.target.value)}
                        className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                    </div>
                    <div>
                      <label className="mb-1 block text-sm font-medium text-gray-700">
                        終了日
                      </label>
                      <input
                        type="date"
                        value={dateTo}
                        onChange={(e) => setDateTo(e.target.value)}
                        className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                    </div>
                  </>
                ) : (
                  <div className="sm:col-span-2">
                    <label className="mb-1 block text-sm font-medium text-gray-700">
                      日付
                    </label>
                    <input
                      type="date"
                      value={pinpointDate}
                      onChange={(e) => setPinpointDate(e.target.value)}
                      className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                  </div>
                )}
              </div>

              <div className="mt-4">
                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-5 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-blue-700 disabled:opacity-50"
                >
                  {loading && (
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  )}
                  予測を取得
                </button>
              </div>
            </form>

            {/* Error */}
            {error && (
              <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}

            {/* Results */}
            {forecasts.length > 0 && (
              <div className="space-y-4">
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                  {forecasts.map((f) => (
                    <div
                      key={f.id}
                      className="relative overflow-hidden rounded-lg border bg-white shadow-sm transition hover:shadow-md"
                    >
                      {/* Confidence bar */}
                      <div
                        className={`h-1 ${confidenceColor(f.confidence)}`}
                      />
                      <div className="p-4">
                        <p className="text-sm font-medium text-gray-500">
                          {f.date}
                        </p>
                        <div className="mt-2 flex items-center gap-2">
                          <span className="text-3xl">
                            {weatherEmoji(f.weather)}
                          </span>
                          <span className="text-sm font-medium text-gray-700">
                            {f.weather}
                          </span>
                        </div>
                        <div className="mt-3 flex items-end justify-between">
                          <div>
                            <span className="text-lg font-semibold text-red-500">
                              {f.temp_max}°
                            </span>
                            <span className="mx-1 text-gray-400">/</span>
                            <span className="text-lg font-semibold text-blue-500">
                              {f.temp_min}°
                            </span>
                          </div>
                          <div className="text-sm text-gray-500">
                            降水 {Math.round(f.precipitation_prob * 100)}%
                          </div>
                        </div>
                        <div className="mt-2 flex items-center gap-1.5">
                          <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-200">
                            <div
                              className={`h-full rounded-full ${confidenceColor(f.confidence)}`}
                              style={{
                                width: `${Math.round(f.confidence * 100)}%`,
                              }}
                            />
                          </div>
                          <span className="text-xs text-gray-400">
                            {Math.round(f.confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <button
                  onClick={syncToCalendar}
                  disabled={syncing}
                  className="inline-flex items-center gap-2 rounded-md bg-green-600 px-5 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-green-700 disabled:opacity-50"
                >
                  {syncing && (
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  )}
                  Google カレンダーに登録
                </button>

                {syncMessage && (
                  <p className="text-sm text-gray-600">{syncMessage}</p>
                )}
              </div>
            )}
          </div>
        ) : (
          /* Calendar integration tab */
          <div className="rounded-lg border bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-gray-900">
              カレンダー連携
            </h2>
            <p className="mt-2 text-sm text-gray-600">
              予測結果を Google
              カレンダーに登録できます。まず「予測取得」タブで予測を取得してから、「Google
              カレンダーに登録」ボタンを押してください。
            </p>
            {forecasts.length > 0 ? (
              <div className="mt-4">
                <p className="text-sm text-gray-700">
                  現在 {forecasts.length} 件の予測データがあります。
                </p>
                <button
                  onClick={syncToCalendar}
                  disabled={syncing}
                  className="mt-3 inline-flex items-center gap-2 rounded-md bg-green-600 px-5 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-green-700 disabled:opacity-50"
                >
                  {syncing && (
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  )}
                  Google カレンダーに登録
                </button>
                {syncMessage && (
                  <p className="mt-2 text-sm text-gray-600">{syncMessage}</p>
                )}
              </div>
            ) : (
              <p className="mt-4 text-sm text-gray-400">
                予測データがありません。先に予測を取得してください。
              </p>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
