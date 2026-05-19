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
  metrics: ModelMetrics | null;
}


export interface TrainConfig {
    model_name: string;
    lr: number;
    epochs: number;
    batch_size: number;
}


export interface TrainResponse {
    f1: number;
    precision: number;
    recall: number;
    confusion_matrix: number[][];
    n: number;
    n_positive: number;
}


export async function launchTrainConfig(
  prevState: any,
  train_config: FormData,
) { // Removed strict return type here to avoid type clashing with useActionState, but you can keep it if your types perfectly align.
    const cookieStore = await cookies();
    const token = cookieStore.get("accessToken")?.value;
    
    if (!token) {
        return { message: "No access token found" };
    }

    // 2. Convert FormData to a standard object so it can be stringified
    const rawData = Object.fromEntries(train_config.entries());

    const res = await fetch(`${process.env.API_BASE_URL}/train`, {
        method: "POST",
        headers: { 
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json" // 3. Explicitly tell FastAPI this is JSON
        },
        body: JSON.stringify(rawData),
    });

    if (!res.ok) {
        let detail = `(Training failed: ${res.status})`;
        try {
        const j = await res.json();
        detail = typeof j.detail === "string" ? j.detail : detail;
        } catch {
        /* ignore */
        }
        return { error: detail };
    }
    const json = await res.json();
    return json;
}
