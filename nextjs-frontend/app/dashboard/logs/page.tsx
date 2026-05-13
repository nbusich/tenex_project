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
  fetchLogFiles,
  type LogFile,
  type PageResponse,
} from "@/components/actions/logs-action";
import { UploadForm } from "./upload-form";

function isPage(
  result: PageResponse<LogFile> | { message: string },
): result is PageResponse<LogFile> {
  return (result as PageResponse<LogFile>).items !== undefined;
}

export default async function LogsPage() {
  const result = await fetchLogFiles(1, 25);

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-2">Log analysis</h2>
      <p className="text-sm text-muted-foreground mb-6">
        Upload ZScaler web proxy logs (CSV/TSV/.log). Pick which model
        scores the file (Transformer by default); flagged rows are stored
        against your account.
      </p>

      <div className="mb-6">
        <UploadForm />
      </div>

      <section className="p-6 bg-white rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Your uploads</h2>
        {!isPage(result) ? (
          <p className="text-sm text-red-600">{result.message}</p>
        ) : result.items.length === 0 ? (
          <p className="text-sm text-gray-600">No uploads yet.</p>
        ) : (
          <Table className="min-w-full text-sm">
            <TableHeader>
              <TableRow>
                <TableHead>Filename</TableHead>
                <TableHead>Uploaded</TableHead>
                <TableHead className="text-center">Entries</TableHead>
                <TableHead className="text-center">Anomalies</TableHead>
                <TableHead className="text-center">Open</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {result.items.map((f) => (
                <TableRow key={f.id}>
                  <TableCell className="font-medium">{f.filename}</TableCell>
                  <TableCell>
                    {new Date(f.uploaded_at).toLocaleString()}
                  </TableCell>
                  <TableCell className="text-center">
                    {f.total_entries}
                  </TableCell>
                  <TableCell className="text-center">
                    <span
                      className={
                        f.anomaly_count > 0
                          ? "inline-flex items-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800"
                          : "text-gray-500"
                      }
                    >
                      {f.anomaly_count}
                    </span>
                  </TableCell>
                  <TableCell className="text-center">
                    <Link
                      href={`/dashboard/logs/${f.id}`}
                      className="text-blue-600 hover:underline"
                    >
                      View
                    </Link>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </section>
    </div>
  );
}
