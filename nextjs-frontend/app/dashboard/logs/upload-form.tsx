"use client";

import { useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { uploadLogFile } from "@/components/actions/logs-action";
import { Button } from "@/components/ui/button";

type ModelChoice = "transformer" | "autoencoder" | "random_forest" | "heuristic" | "mlp";

const MODEL_TABS: { id: ModelChoice; label: string; blurb: string }[] = [
  {
    id: "transformer",
    label: "Transformer",
    blurb: "Sequence autoencoder · default",
  },
  {
    id: "autoencoder",
    label: "AutoEncoder",
    blurb: "Per-row reconstruction error",
  },
  {
    id: "random_forest",
    label: "RandomForest",
    blurb: "Supervised · threshold 1e-3",
  },
  {
    id: "heuristic",
    label: "Heuristic",
    blurb: "Rule-based (no model needed)",
  },
  {
    id: "mlp",
    label: "mlp",
    blurb: "mlp",
  },
];

export function UploadForm() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [selected, setSelected] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<ModelChoice>("transformer");
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setSelected(files[0]);
    setError(null);
  };

  const submit = () => {
    if (!selected) {
      setError("Pick a file first.");
      return;
    }
    const fd = new FormData();
    fd.append("file", selected);
    fd.append("model", model);
    startTransition(async () => {
      {/*INVESTIGATION STEP 2: selected file and model go to uploadLogFile */}
      const result = await uploadLogFile(fd);
      if (result.error) {
        setError(result.error);
        return;
      }
      const fileId = result.data?.file?.id;
      if (fileId) {
        router.push(`/dashboard/logs/${fileId}`);
      } else {
        router.refresh();
      }
    });
  };

  return (
    <section className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Upload a ZScaler log</h2>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-10 cursor-pointer transition ${
          dragOver
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 hover:border-gray-400 bg-gray-50"
        }`}
      >
        <p className="text-sm text-gray-600">
          Drag &amp; drop a <code>.log</code>, <code>.txt</code>, or{" "}
          <code>.csv</code> ZScaler export here, or click to browse.
        </p>
        {selected && (
          <p className="mt-3 text-sm font-medium">
            Selected: {selected.name} (
            {(selected.size / 1024).toFixed(1)} KB)
          </p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".log,.txt,.csv,.tsv"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

      <div className="mt-5">
        <p className="text-sm font-medium text-gray-700 mb-2">
          Score with
        </p>
        <div
          role="tablist"
          aria-label="Anomaly detection model"
          className="flex flex-wrap gap-1 rounded-lg bg-gray-100 p-1"
        >
          {MODEL_TABS.map((t) => {
            const active = model === t.id;
            return (
              <button
                key={t.id}
                role="tab"
                aria-selected={active}
                type="button"
                onClick={() => setModel(t.id)}
                disabled={isPending}
                className={`flex-1 min-w-[140px] rounded-md px-3 py-2 text-left transition ${
                  active
                    ? "bg-white shadow text-gray-900"
                    : "text-gray-600 hover:text-gray-900"
                }`}
              >
                <span className="block text-sm font-semibold">{t.label}</span>
                <span className="block text-xs text-gray-500">{t.blurb}</span>
              </button>
            );
          })}
        </div>
        <p className="mt-2 text-xs text-gray-500">
          Each row is scored by exactly one model. Pick another model and
          re-upload to compare verdicts on the same file.
        </p>
      </div>

      <div className="mt-4 flex gap-2">
        <Button onClick={submit} disabled={!selected || isPending}>
          {isPending ? "Uploading…" : "Upload & analyze"}
        </Button>
        {selected && !isPending && (
          <Button variant="outline" onClick={() => setSelected(null)}>
            Clear
          </Button>
        )}
      </div>

      <div className="mt-6 pt-6 border-t border-gray-200 space-y-4">
        <p className="text-sm font-medium text-gray-700">
          No log handy? Try one of these:
        </p>
        <p className="text-xs text-gray-500">
          Each file is 20 rows pulled verbatim from the CSIC-2010 held-out
          test set — same columns and content the models were evaluated on.
        </p>

        <div>
          <p className="text-xs uppercase tracking-wide text-gray-500 mb-1">
            Normal traffic only
          </p>
          <ul className="grid grid-cols-1 sm:grid-cols-3 gap-x-6 gap-y-1 text-sm">
            {[
              { href: "/samples/csic-normal-1.csv", label: "Normal #1" },
              { href: "/samples/csic-normal-2.csv", label: "Normal #2" },
              { href: "/samples/csic-normal-3.csv", label: "Normal #3" },
            ].map((s) => (
              <li key={s.href}>
                <a href={s.href} download className="text-blue-600 hover:underline">
                  {s.label}
                </a>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <p className="text-xs uppercase tracking-wide text-gray-500 mb-1">
            Attack traffic only
          </p>
          <ul className="grid grid-cols-1 sm:grid-cols-3 gap-x-6 gap-y-1 text-sm">
            {[
              { href: "/samples/csic-attack-1.csv", label: "Attacks #1" },
              { href: "/samples/csic-attack-2.csv", label: "Attacks #2" },
              { href: "/samples/csic-attack-3.csv", label: "Attacks #3" },
            ].map((s) => (
              <li key={s.href}>
                <a href={s.href} download className="text-blue-600 hover:underline">
                  {s.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
