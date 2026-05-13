"use server";

import { cookies } from "next/headers";
import { revalidatePath } from "next/cache";

const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:8000";

export interface LogFile {
  id: string;
  filename: string;
  uploaded_at: string;
  total_entries: number;
  anomaly_count: number;
}

export interface LogEntry {
  id: string;
  log_file_id: string;
  timestamp: string | null;
  source_ip: string | null;
  user_agent: string | null;
  action: string | null;
  url: string | null;
  method: string | null;
  status_code: number | null;
  bytes_sent: number | null;
  url_category: string | null;
  threat_name: string | null;
  user_login: string | null;
  raw_line: string | null;
  is_anomaly: boolean;
  anomaly_score: number | null;
  anomaly_reason: string | null;
}

export interface PageResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

export interface LogFileSummary {
  file: LogFile;
  timeline: { bucket: string; count: number; anomaly_count: number }[];
  top_source_ips: { source_ip: string; count: number; anomaly_count: number }[];
  top_actions: Record<string, number>;
  top_categories: Record<string, number>;
  ai_explanation: string | null;
}

async function authedFetch(path: string, init: RequestInit = {}) {
  const cookieStore = await cookies();
  const token = cookieStore.get("accessToken")?.value;
  if (!token) {
    throw new Error("No access token found");
  }
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${token}`);
  return fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });
}

export async function fetchLogFiles(
  page: number = 1,
  size: number = 10,
): Promise<PageResponse<LogFile> | { message: string }> {
  const res = await authedFetch(`/logs/files?page=${page}&size=${size}`);
  if (!res.ok) {
    return { message: `Backend returned ${res.status}` };
  }
  return res.json();
}

export async function fetchLogFileSummary(
  fileId: string,
): Promise<LogFileSummary | { message: string }> {
  const res = await authedFetch(`/logs/files/${fileId}/summary`);
  if (!res.ok) {
    return { message: `Backend returned ${res.status}` };
  }
  return res.json();
}

export async function fetchLogEntries(
  fileId: string,
  page: number = 1,
  size: number = 25,
  onlyAnomalies: boolean = false,
): Promise<PageResponse<LogEntry> | { message: string }> {
  const qs = new URLSearchParams({
    page: String(page),
    size: String(size),
    only_anomalies: String(onlyAnomalies),
  });
  const res = await authedFetch(`/logs/files/${fileId}/entries?${qs}`);
  if (!res.ok) {
    return { message: `Backend returned ${res.status}` };
  }
  return res.json();
}

export async function uploadLogFile(formData: FormData) {
  const cookieStore = await cookies();
  const token = cookieStore.get("accessToken")?.value;
  if (!token) {
    return { error: "No access token found" };
  }

  const file = formData.get("file");
  if (!(file instanceof File)) {
    return { error: "No file provided" };
  }

  const upstream = new FormData();
  upstream.append("file", file, file.name);
  const model = formData.get("model");
  if (typeof model === "string" && model.length > 0) {
    upstream.append("model", model);
  }

  const res = await fetch(`${API_BASE_URL}/logs/upload`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: upstream,
  });

  if (!res.ok) {
    let detail = `Upload failed (${res.status})`;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : detail;
    } catch {
      /* ignore */
    }
    return { error: detail };
  }

  const json = await res.json();
  revalidatePath("/dashboard/logs");
  return { data: json };
}

export async function deleteLogFile(fileId: string) {
  const res = await authedFetch(`/logs/files/${fileId}`, { method: "DELETE" });
  if (!res.ok) {
    return { error: `Delete failed (${res.status})` };
  }
  revalidatePath("/dashboard/logs");
  return { ok: true };
}

export interface EntryExplanation {
  verdict: "anomaly" | "false_positive" | "normal";
  description: string;
  source: "llm" | "fallback";
}

export async function explainLogEntry(
  entryId: string,
): Promise<EntryExplanation | { error: string }> {
  const res = await authedFetch(`/logs/entries/${entryId}/explain`, {
    method: "POST",
  });
  if (!res.ok) {
    let detail = `Explain failed (${res.status})`;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : detail;
    } catch {
      /* ignore */
    }
    return { error: detail };
  }
  return res.json();
}
