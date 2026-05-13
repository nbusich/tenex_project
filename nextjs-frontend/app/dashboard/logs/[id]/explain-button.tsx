"use client";

import { useState, useTransition } from "react";
import {
  explainLogEntry,
  type EntryExplanation,
} from "@/components/actions/logs-action";

interface Props {
  entryId: string;
}

export function ExplainButton({ entryId }: Props) {
  const [explanation, setExplanation] = useState<EntryExplanation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const onClick = () => {
    setError(null);
    startTransition(async () => {
      const result = await explainLogEntry(entryId);
      if ("error" in result) {
        setError(result.error);
        return;
      }
      setExplanation(result);
    });
  };

  if (explanation) {
    const isFP = explanation.verdict === "false_positive";
    return (
      <div
        className={`text-xs rounded px-2 py-1 max-w-[36ch] ${
          isFP
            ? "bg-green-50 text-green-900 border border-green-200"
            : "bg-amber-50 text-amber-900 border border-amber-200"
        }`}
      >
        <p className="font-semibold mb-0.5">
          {isFP ? "Likely false positive" : "Real anomaly"}
          {explanation.source === "fallback" && (
            <span className="ml-1 font-normal opacity-70">(no LLM key)</span>
          )}
        </p>
        <p className="leading-snug whitespace-pre-wrap">
          {explanation.description}
        </p>
      </div>
    );
  }

  return (
    <div>
      <button
        type="button"
        onClick={onClick}
        disabled={isPending}
        className="text-xs px-2 py-0.5 rounded border border-blue-300 text-blue-700 hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isPending ? "Asking AI…" : "Explain"}
      </button>
      {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
    </div>
  );
}
