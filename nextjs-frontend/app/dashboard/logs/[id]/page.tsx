import Link from "next/link";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TableHeader,
} from "@/components/ui/table";
import {
  fetchLogEntries,
  fetchLogFileSummary,
  type LogEntry,
  type LogFileSummary,
  type PageResponse,
} from "@/components/actions/logs-action";
import { PagePagination } from "@/components/page-pagination";
import { ExplainButton } from "./explain-button";

interface Props {
  params: Promise<{ id: string }>;
  searchParams: Promise<{
    page?: string;
    size?: string;
    only_anomalies?: string;
  }>;
}

function isSummary(
  r: LogFileSummary | { message: string },
): r is LogFileSummary {
  return (r as LogFileSummary).file !== undefined;
}

function isEntries(
  r: PageResponse<LogEntry> | { message: string },
): r is PageResponse<LogEntry> {
  return (r as PageResponse<LogEntry>).items !== undefined;
}

export default async function LogFilePage({ params, searchParams }: Props) {
  const { id } = await params;
  const sp = await searchParams;
  const page = Number(sp.page) || 1;
  const size = Number(sp.size) || 25;
  const onlyAnomalies = sp.only_anomalies === "true";

  const [summaryRes, entriesRes] = await Promise.all([
    fetchLogFileSummary(id),
    fetchLogEntries(id, page, size, onlyAnomalies),
  ]);

  if (!isSummary(summaryRes)) {
    return (
      <div>
        <h2 className="text-2xl font-semibold mb-4">Log file</h2>
        <p className="text-sm text-red-600">{summaryRes.message}</p>
      </div>
    );
  }

  const { file, top_source_ips, top_actions, top_categories, ai_explanation } =
    summaryRes;
  const totalPages = isEntries(entriesRes)
    ? Math.ceil(entriesRes.total / size)
    : 0;

  return (
    <div>
      <div className="flex items-baseline justify-between mb-6">
        <div>
          <h2 className="text-2xl font-semibold">{file.filename}</h2>
          <p className="text-sm text-muted-foreground">
            Uploaded {new Date(file.uploaded_at).toLocaleString()}
          </p>
        </div>
        <Link
          href="/dashboard/logs"
          className="text-sm text-blue-600 hover:underline"
        >
          ← Back to uploads
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card title="Total entries" value={file.total_entries.toString()} />
        <Card
          title="Anomalies"
          value={file.anomaly_count.toString()}
          accent={file.anomaly_count > 0 ? "danger" : "neutral"}
        />
        <Card
          title="Anomaly rate"
          value={`${
            file.total_entries === 0
              ? 0
              : ((file.anomaly_count / file.total_entries) * 100).toFixed(1)
          }%`}
        />
      </div>

      {ai_explanation && (
        <section className="p-6 bg-amber-50 border border-amber-200 rounded-lg mb-6">
          <h3 className="text-sm font-semibold text-amber-900 mb-2">
            AI summary
          </h3>
          <p className="text-sm text-amber-900 whitespace-pre-wrap">
            {ai_explanation}
          </p>
        </section>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <SmallList
          title="Top source IPs"
          rows={top_source_ips.map((r) => ({
            label: r.source_ip,
            value: `${r.count}${r.anomaly_count ? ` (${r.anomaly_count} anom)` : ""}`,
          }))}
        />
        <SmallList
          title="Top actions"
          rows={Object.entries(top_actions).map(([k, v]) => ({
            label: k,
            value: v.toString(),
          }))}
        />
        <SmallList
          title="Top categories"
          rows={Object.entries(top_categories).map(([k, v]) => ({
            label: k,
            value: v.toString(),
          }))}
        />
      </div>

      <section className="p-6 bg-white rounded-lg shadow">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Entries</h3>
          <div className="flex gap-3 text-sm">
            <Link
              href={`/dashboard/logs/${id}?page=1&size=${size}`}
              className={
                onlyAnomalies
                  ? "text-blue-600 hover:underline"
                  : "font-semibold"
              }
            >
              All
            </Link>
            <Link
              href={`/dashboard/logs/${id}?page=1&size=${size}&only_anomalies=true`}
              className={
                onlyAnomalies
                  ? "font-semibold"
                  : "text-blue-600 hover:underline"
              }
            >
              Anomalies only
            </Link>
          </div>
        </div>

        {!isEntries(entriesRes) ? (
          <p className="text-sm text-red-600">{entriesRes.message}</p>
        ) : (
          <>
            <Table className="min-w-full text-sm">
              <TableHeader>
                <TableRow>
                  <TableHead>Time</TableHead>
                  <TableHead>Source IP</TableHead>
                  <TableHead>Action</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>URL</TableHead>
                  <TableHead className="text-center">Score</TableHead>
                  <TableHead>Reason</TableHead>
                  <TableHead>AI verdict</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {entriesRes.items.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} className="text-center">
                      No entries.
                    </TableCell>
                  </TableRow>
                ) : (
                  entriesRes.items.map((e) => (
                    <TableRow
                      key={e.id}
                      className={
                        e.is_anomaly ? "bg-red-50 hover:bg-red-100" : undefined
                      }
                    >
                      <TableCell className="whitespace-nowrap">
                        {e.timestamp
                          ? new Date(e.timestamp).toLocaleString()
                          : "—"}
                      </TableCell>
                      <TableCell>{e.source_ip || "—"}</TableCell>
                      <TableCell>{e.action || "—"}</TableCell>
                      <TableCell>{e.status_code ?? "—"}</TableCell>
                      <TableCell
                        className="max-w-[24ch] truncate"
                        title={e.url || undefined}
                      >
                        {e.url || "—"}
                      </TableCell>
                      <TableCell className="text-center">
                        {e.is_anomaly ? (
                          <span className="inline-flex items-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800">
                            {(e.anomaly_score ?? 0).toFixed(2)}
                          </span>
                        ) : (
                          "—"
                        )}
                      </TableCell>
                      <TableCell
                        className="max-w-[32ch] truncate"
                        title={e.anomaly_reason || undefined}
                      >
                        {e.anomaly_reason || "—"}
                      </TableCell>
                      <TableCell>
                        {e.is_anomaly ? <ExplainButton entryId={e.id} /> : "—"}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>

            <PagePagination
              currentPage={page}
              totalPages={totalPages}
              pageSize={size}
              totalItems={entriesRes.total}
              basePath={`/dashboard/logs/${id}`}
            />
          </>
        )}
      </section>
    </div>
  );
}

function Card({
  title,
  value,
  accent,
}: {
  title: string;
  value: string;
  accent?: "danger" | "neutral";
}) {
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <p className="text-xs uppercase tracking-wide text-gray-500">{title}</p>
      <p
        className={`text-3xl font-semibold mt-1 ${
          accent === "danger" ? "text-red-600" : "text-gray-900"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function SmallList({
  title,
  rows,
}: {
  title: string;
  rows: { label: string; value: string }[];
}) {
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <p className="text-xs uppercase tracking-wide text-gray-500 mb-2">
        {title}
      </p>
      {rows.length === 0 ? (
        <p className="text-sm text-gray-500">No data.</p>
      ) : (
        <ul className="text-sm space-y-1">
          {rows.map((r) => (
            <li key={r.label} className="flex justify-between gap-2">
              <span className="truncate" title={r.label}>
                {r.label}
              </span>
              <span className="text-gray-700 whitespace-nowrap">
                {r.value}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
