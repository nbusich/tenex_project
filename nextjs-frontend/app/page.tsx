import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ShieldAlert } from "lucide-react";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-50 dark:bg-gray-900 p-8">
      <div className="text-center max-w-2xl">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-600/10 text-blue-700 dark:text-blue-300 mb-6">
          <ShieldAlert className="w-9 h-9" />
        </div>
        <h1 className="text-5xl font-bold text-gray-800 dark:text-white mb-4">
          Anomaly Sentinel
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-300 mb-2">
          Upload ZScaler web proxy logs and triage them in seconds.
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-8">
          Three machine-learning models — RandomForest, AutoEncoder, and a
          Transformer encoder — score every request, so SOC analysts can spot
          attacks the rule-based filters miss.
        </p>

        <div className="flex flex-col sm:flex-row gap-3 items-center justify-center">
          <Link href="/dashboard/logs">
            <Button className="px-8 py-4 text-lg font-semibold rounded-full shadow-lg bg-gradient-to-r from-blue-500 to-indigo-500 text-white hover:from-blue-600 hover:to-indigo-600 focus:ring-4 focus:ring-blue-300">
              Upload a log
            </Button>
          </Link>
          <Link href="/dashboard/models">
            <Button
              variant="outline"
              className="px-8 py-4 text-lg font-semibold rounded-full"
            >
              Compare models
            </Button>
          </Link>
        </div>
      </div>
    </main>
  );
}
