import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TableHeader,
} from "@/components/ui/table";
import {
  fetchModelComparison,
  type ComparisonResponse,
} from "@/components/actions/models-action";
import { PagePagination } from "@/components/page-pagination";

interface Props {
  searchParams: Promise<{
    page?: string;
    size?: string;
    sample?: string;
    threshold?: string;
  }>;
}

function isResponse(
  r: ComparisonResponse | { message: string },
): r is ComparisonResponse {
  return (r as ComparisonResponse).models !== undefined;
}

export default async function ModelsPage({ searchParams }: Props) {
  const sp = await searchParams;
  const page = Number(sp.page) || 1;
  const size = Number(sp.size) || 25;
  const sample = sp.sample !== undefined ? Number(sp.sample) : 0;
  const threshold = sp.threshold !== undefined ? Number(sp.threshold) : undefined;

  const result = await fetchModelComparison(page, size, sample, threshold);

  if (!isResponse(result)) {
    return (
      <div>
        <h2 className="text-2xl font-semibold mb-4">Model comparison</h2>
        <p className="text-sm text-red-600 mb-4">{result.message}</p>
        <p className="text-sm text-gray-500">
          The CSIC test set isn&apos;t prepared yet. From the backend
          container, run{" "}
          <code className="bg-gray-100 px-1 rounded">
            python -m app.anomaly.train.data.prepare
          </code>
          .
        </p>
      </div>
    );
  }

  const modelNames = result.models.map((m) => m.name);

  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <h2 className="text-2xl font-semibold">Model comparison</h2>
        <p className="text-sm text-muted-foreground">
          {result.sample_size} CSIC test rows
        </p>
      </div>
      <p className="text-sm text-muted-foreground mb-6">
        Each model is run against the CSIC-2010 held-out test set so the
        ground-truth label is known. Untrained models are skipped — train
        them with the scripts under{" "}
        <code className="bg-gray-100 px-1 rounded">
          app/anomaly/train/train/
        </code>
        .
      </p>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {result.models.map((m) => (
          <div key={m.name} className="p-4 bg-white rounded-lg shadow">
            <div className="flex items-center justify-between">
              <p className="text-lg font-semibold capitalize">
                {m.name.replace("_", " ")}
              </p>
              <span
                className={
                  m.available
                    ? "text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-800"
                    : "text-xs px-2 py-0.5 rounded-full bg-gray-200 text-gray-700"
                }
              >
                {m.available ? "trained" : "untrained"}
              </span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              threshold {formatThreshold(m.threshold)}
            </p>
            {m.metrics ? (
              <dl className="mt-3 grid grid-cols-3 gap-2 text-sm">
                <Metric label="F1" value={m.metrics.f1.toFixed(3)} />
                <Metric label="Precision" value={m.metrics.precision.toFixed(3)} />
                <Metric label="Recall" value={m.metrics.recall.toFixed(3)} />
              </dl>
            ) : (
              <p className="mt-3 text-sm text-gray-500">
                {m.error || "Train this model to populate metrics."}
              </p>
            )}
            {m.metrics && (
              <ConfusionMatrix matrix={m.metrics.confusion_matrix} />
            )}
          </div>
        ))}
      </section>

      <section className="p-6 bg-white rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Per-row breakdown</h3>
        <Table className="min-w-full text-sm">
          <TableHeader>
            <TableRow>
              <TableHead>Time</TableHead>
              <TableHead>Source IP</TableHead>
              <TableHead>URL</TableHead>
              <TableHead className="text-center">Truth</TableHead>
              {modelNames.map((n) => (
                <TableHead key={n} className="text-center capitalize">
                  {n.replace("_", " ")}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {result.rows.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={4 + modelNames.length}
                  className="text-center"
                >
                  No rows.
                </TableCell>
              </TableRow>
            ) : (
              result.rows.map((row, idx) => (
                <TableRow key={idx}>
                  <TableCell className="whitespace-nowrap">
                    {row.timestamp || "—"}
                  </TableCell>
                  <TableCell>{row.client_ip || "—"}</TableCell>
                  <TableCell
                    className="max-w-[28ch] truncate"
                    title={row.url || undefined}
                  >
                    {row.url || "—"}
                  </TableCell>
                  <TableCell className="text-center">
                    <TruthBadge label={row.label} />
                  </TableCell>
                  {modelNames.map((n) => {
                    const v = row.verdicts[n];
                    return (
                      <TableCell key={n} className="text-center">
                        <VerdictCell verdict={v} truth={row.label} />
                      </TableCell>
                    );
                  })}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>

        <PagePagination
          currentPage={result.page}
          totalPages={result.pages}
          pageSize={result.size}
          totalItems={result.total}
          basePath="/dashboard/models"
        />
      </section>
    </div>
  );
}

function formatThreshold(t: number): string {
  if (t === 0) return "0";
  if (t < 0.01) return t.toExponential(0);
  return t.toFixed(3);
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-gray-500">{label}</dt>
      <dd className="font-mono">{value}</dd>
    </div>
  );
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  if (!matrix || matrix.length !== 2) return null;
  const [[tn, fp], [fn, tp]] = matrix;
  return (
    <div className="mt-4 text-xs">
      <p className="uppercase tracking-wide text-gray-500 mb-1">
        Confusion matrix
      </p>
      <table className="border border-gray-200 w-full text-center">
        <thead>
          <tr className="bg-gray-50">
            <th className="p-1"></th>
            <th className="p-1">pred normal</th>
            <th className="p-1">pred anomaly</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="p-1 font-medium">true normal</td>
            <td className="p-1 bg-green-50">{tn}</td>
            <td className="p-1 bg-red-50">{fp}</td>
          </tr>
          <tr>
            <td className="p-1 font-medium">true anomaly</td>
            <td className="p-1 bg-red-50">{fn}</td>
            <td className="p-1 bg-green-50">{tp}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function TruthBadge({ label }: { label: number }) {
  return label === 1 ? (
    <span className="inline-flex items-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800">
      anomaly
    </span>
  ) : (
    <span className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">
      normal
    </span>
  );
}

function VerdictCell({
  verdict,
  truth,
}: {
  verdict: { available: boolean; prediction?: number; score?: number; correct?: boolean };
  truth: number;
}) {
  if (!verdict.available) {
    return <span className="text-gray-400">—</span>;
  }
  const label = verdict.prediction === 1 ? "anomaly" : "normal";
  const correct = verdict.correct;
  const className = correct
    ? "inline-flex items-center rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800"
    : "inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-900";
  const title = `score=${verdict.score?.toFixed(3)}  truth=${truth === 1 ? "anomaly" : "normal"}`;
  return (
    <span className={className} title={title}>
      {label} {correct ? "✓" : "✗"}
    </span>
  );
}
