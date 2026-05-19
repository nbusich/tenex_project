"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useActionState } from "react";
import { SubmitButton } from "@/components/ui/submitButton";
import { launchTrainConfig } from "@/components/actions/train-action";

// Start with a null state so we can easily check if a submission has occurred
const initialState: any = null;

export default function TrainModelPage() {
  const [state, dispatch] = useActionState(launchTrainConfig, initialState);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-white dark:bg-gray-900 rounded-xl shadow-lg border border-gray-200 dark:border-gray-800 p-8">
        <header className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Configure Model
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Set your hyperparameters and initialize the training run.
          </p>
        </header>

        <form action={dispatch} className="space-y-6">
          {/* Model Selection */}
          <div className="space-y-2">
            <Label htmlFor="model_name">Model Architecture</Label>
            <select
              id="model_name"
              name="model_name"
              required
              className="flex h-10 w-full items-center justify-between rounded-md border border-gray-300 bg-transparent px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-900 dark:border-gray-700 dark:bg-gray-900 dark:text-white dark:focus:ring-gray-300"
            >
              <option value="autoencoder">Autoencoder</option>
              <option value="transformer">Transformer</option>
              <option value="rf">Random Forest (RF)</option>
            </select>
          </div>

          {/* Learning Rate */}
          <div className="space-y-2">
            <Label htmlFor="lr">Learning Rate</Label>
            <Input
              id="lr"
              name="lr"
              type="number"
              step="0.0001"
              defaultValue="0.001"
              required
              className="border-gray-300 dark:border-gray-700"
            />
          </div>

          {/* Batch Size */}
          <div className="space-y-2">
            <Label htmlFor="batch_size">Batch Size</Label>
            <Input
              id="batch_size"
              name="batch_size"
              type="number"
              step="1"
              defaultValue="128"
              required
              className="border-gray-300 dark:border-gray-700"
            />
          </div>

          {/* Epochs */}
          <div className="space-y-2">
            <Label htmlFor="epochs">Epochs</Label>
            <Input
              id="epochs"
              name="epochs"
              type="number"
              step="1"
              defaultValue="10"
              required
              className="border-gray-300 dark:border-gray-700"
            />
          </div>

          <div className="pt-2">
            <SubmitButton text="Start Training Run" />
          </div>
        </form>

        {/* --- RESULTS PANEL --- */}
        {state && (
          <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-800">
            {/* Handle Error State (from !res.ok) */}
            {state.error && (
              <div className="p-4 text-sm text-red-800 bg-red-50 dark:bg-red-900/20 dark:text-red-300 border border-red-200 dark:border-red-900 rounded-md">
                <p className="font-semibold">Training Failed</p>
                <p>{state.error}</p>
              </div>
            )}

            {/* Handle Auth Message State */}
            {state.message && !state.error && (
              <div className="p-4 text-sm text-yellow-800 bg-yellow-50 dark:bg-yellow-900/20 dark:text-yellow-300 border border-yellow-200 dark:border-yellow-900 rounded-md">
                {state.message}
              </div>
            )}

            {/* Handle Success State (Metrics returned from FastAPI) */}
            {state.metrics && (
              <div className="p-4 bg-green-50 dark:bg-green-900/10 border border-green-200 dark:border-green-900/50 rounded-md space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-green-900 dark:text-green-400">
                    Training Complete
                  </h3>
                  <span className="text-xs px-2 py-1 bg-green-200 dark:bg-green-800 text-green-900 dark:text-green-100 rounded-full font-medium">
                    {state.status}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider">F1 Score</p>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {state.metrics.f1?.toFixed(4) || "0.0000"}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider">Precision</p>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {state.metrics.precision?.toFixed(4) || "0.0000"}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider">Recall</p>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {state.metrics.recall?.toFixed(4) || "0.0000"}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider">Samples (n)</p>
                    <p className="font-medium text-gray-900 dark:text-gray-100">
                      {state.metrics.n || 0}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}