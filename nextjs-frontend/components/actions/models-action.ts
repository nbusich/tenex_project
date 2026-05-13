"use server";

import { cookies } from "next/headers";

const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:8000";

export interface ModelMetrics {
  f1: number;
  precision: number;
  recall: number;
  confusion_matrix: number[][];
  n: number;
  n_positive: number;
  n_predicted_positive: number;
  mean_score: number;
}

export interface ModelSummary {
  name: string;
  available: boolean;
  metrics: ModelMetrics | null;
  threshold: number;
  error?: string;
}

export interface ModelVerdict {
  available: boolean;
  prediction?: number;
  score?: number;
  correct?: boolean;
}

export interface ComparisonRow {
  timestamp: string | null;
  client_ip: string | null;
  user: string | null;
  method: string | null;
  url: string | null;
  label: number;
  verdicts: Record<string, ModelVerdict>;
}

export interface ComparisonResponse {
  sample_size: number;
  thresholds: Record<string, number>;
  page: number;
  size: number;
  total: number;
  pages: number;
  models: ModelSummary[];
  rows: ComparisonRow[];
}

export async function fetchModelComparison(
  page: number = 1,
  size: number = 25,
  sample: number = 0,
  threshold?: number,
): Promise<ComparisonResponse | { message: string }> {
  const cookieStore = await cookies();
  const token = cookieStore.get("accessToken")?.value;
  if (!token) {
    return { message: "No access token found" };
  }

  const qs = new URLSearchParams({
    page: String(page),
    size: String(size),
    sample: String(sample),
  });
  if (threshold !== undefined) {
    qs.set("threshold", String(threshold));
  }

  const res = await fetch(`${API_BASE_URL}/models/compare?${qs}`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });

  if (!res.ok) {
    let detail = `Backend returned ${res.status}`;
    try {
      const j = await res.json();
      if (typeof j.detail === "string") detail = j.detail;
    } catch {
      /* ignore */
    }
    return { message: detail };
  }

  return res.json();
}
