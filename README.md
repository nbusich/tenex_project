# Anomaly Sentinel — ZScaler Log Analyzer

Full-stack web app that lets a SOC analyst upload ZScaler web proxy logs,
parses them, scores each entry for anomalies with one of three ML models,
and renders a human-consumable dashboard. Bootstrapped from the
[vintasoftware/nextjs-fastapi-template](https://github.com/vintasoftware/nextjs-fastapi-template)
(Next.js + FastAPI + Postgres + fastapi-users auth); everything under
`app/anomaly/`, `app/log_parsing/`, and the dashboard pages was written
for this take-home.

## Quick start (Docker)

Prereqs: Docker, Docker Compose, Make.

```bash
# 1. Bootstrap env files
cp fastapi_backend/.env.example fastapi_backend/.env
cp nextjs-frontend/.env.example nextjs-frontend/.env

# 2. Start postgres + backend + frontend + mailhog
docker compose up --build

# 3. In a second terminal, run migrations once
make docker-migrate-db
```

UI is at <http://localhost:3000>, API at <http://localhost:8000>, Swagger
UI at <http://localhost:8000/docs>, mailhog (catches password-reset
emails in dev) at <http://localhost:8025>.

Register from `/register`, log in, and you land on `/dashboard/logs`. The
upload form lets you pick which model scores the file (Transformer by
default) and offers six downloadable sample logs you can try right away
(see [Example log files](#example-log-files)).

## Where AI is used

There are two distinct AI surfaces in this app — one classical-ML, one
LLM — and they're called out explicitly per the take-home rubric.

### 1. Anomaly detection (classical ML, three models)

`fastapi_backend/app/anomaly/` ships three trained models that score
uploaded ZScaler entries:

| Model           | Type                                | Where                                                          |
| --------------- | ----------------------------------- | -------------------------------------------------------------- |
| RandomForest    | Supervised classifier               | `app/anomaly/train/model/random_forest.py`                     |
| AutoEncoder     | Per-row reconstruction error (MLP)  | `app/anomaly/train/model/autoencoder.py`                       |
| TransformerEnc. | Per-window reconstruction error     | `app/anomaly/train/model/encodertransformer.py`                |

Training data: **CSIC-2010** HTTP attack dataset (~50k normal + ~25k
attack requests), augmented with synthetic timestamps / users / source
IPs to look like enterprise traffic. The shared sklearn preprocessor
(`app/anomaly/train/data/pipeline.py`) feeds all three models the same
67-feature representation: hand-engineered URL features (length, digit
ratio, entropy, SQL/XSS markers), `Content-Length` extraction, plus
one-hot encoded method and user, then `StandardScaler`. AE and
Transformer train only on normal rows so reconstruction error is the
anomaly signal; the score is squashed to `[0, 1]` with a soft saturation
calibrated against the 95th-percentile (Transformer) / 85th-percentile
(AE) of training-set errors.

On the upload form the analyst picks **one** model to score the file (a
prior multi-model ensemble was tried but disagreed too often, drowning
real signal). The user can re-upload the same file under a different
model to compare verdicts. `/dashboard/models` runs all three against
the held-out CSIC test set and reports F1 / precision / recall /
confusion matrix side-by-side, with RandomForest's threshold pinned at
`1e-3` (its positive-class probabilities cluster very low; see code
comments in `app/routes/models.py`).

### 2. LLM-backed per-entry explanation (Claude / Gemini)

`fastapi_backend/app/anomaly/explain.py` provides two LLM entry points:

* `explain_anomalies(...)` — file-level paragraph summary, shown above
  the entries table on a file detail page.
* `explain_entry(...)` — per-entry verdict + 1-2 sentence description,
  triggered by clicking **Explain** next to a flagged row. The model is
  asked to either confirm "anomaly" or call it "false_positive" and
  justify briefly. The response is forced to JSON via prompt and
  validated server-side.

Provider is auto-selected from environment: `ANTHROPIC_API_KEY` →
Claude (Haiku 4.5), else `GEMINI_API_KEY` → Gemini 2.5 Flash, else the
endpoint returns the rule-based reason text as a graceful fallback.
The HTTP call uses `httpx` directly — no SDK dependencies added to the
runtime image.

## Example log files

Six samples live under `nextjs-frontend/public/samples/` and are linked
from the upload form. Each is 20 rows pulled verbatim from the CSIC-2010
held-out test set so they exercise the model on its training
distribution:

* `csic-normal-{1,2,3}.csv` — 20 rows labelled normal in CSIC.
* `csic-attack-{1,2,3}.csv` — 20 rows labelled attack in CSIC (SQL
  injection, XSS, path traversal, command injection, recon).

A heterogeneous legacy sample is at
[`local-shared-data/sample_zscaler.csv`](local-shared-data/sample_zscaler.csv).

## Running the ML training pipeline (optional)

The repo ships with trained artifacts under
`fastapi_backend/app/anomaly/train/artifacts/`, so the app works
out-of-the-box. To reproduce / retrain:

```bash
docker compose run --rm --entrypoint sh backend -c "uv sync --group ml-train"
docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.data.prepare"
docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.train_all"

# Retune the AE / Transformer squash thresholds — does NOT touch weights:
docker compose run --rm --entrypoint sh backend -c \
    "python -m app.anomaly.train.train.recalibrate --pct 95"
# Or per-model:
docker compose run --rm --entrypoint sh backend -c \
    "python -m app.anomaly.train.train.recalibrate --model autoencoder --pct 85"
```

After recalibration, **restart the backend** (`docker compose restart
backend`) so the inference engine reloads the new threshold from disk.

## API surface added by this submission

All endpoints require a fastapi-users JWT bearer token and are scoped
to the calling user.

| Method | Path                                | Purpose                                                |
| ------ | ----------------------------------- | ------------------------------------------------------ |
| POST   | `/logs/upload`                      | Upload a CSV/TSV/log file; `model=` form field picks scorer |
| GET    | `/logs/files`                       | Paginated list of your uploads                          |
| GET    | `/logs/files/{id}/entries`          | Paginated entries (`?only_anomalies=`)                  |
| GET    | `/logs/files/{id}/summary`          | Counts, timeline buckets, top IPs, AI file summary       |
| DELETE | `/logs/files/{id}`                  | Delete an upload and its entries                         |
| POST   | `/logs/entries/{id}/explain`        | LLM per-entry verdict (anomaly / false_positive + text)  |
| GET    | `/models/compare`                   | Side-by-side per-model metrics on the CSIC test set     |

## Log format

Tested against ZScaler NSS-style web proxy exports. The parser sniffs
the delimiter (CSV / TSV / pipe), inspects the first row to decide if
it's a header, and maps known field aliases (`urlcategory`, `useragent`,
`responsesize`, `client_ip`, `content_length`, …) to the canonical
schema. Files without a header row fall back to a default column order.
See `fastapi_backend/app/log_parsing/zscaler.py`.

Persisted schema:

```
LogEntry(id, log_file_id, timestamp, source_ip, user_agent, action,
         url, method, status_code, bytes_sent, url_category, threat_name,
         user_login, raw_line, is_anomaly, anomaly_score, anomaly_reason)
```

## Environment variables

Most secrets ship as `.env.example` defaults. The two optional ones
worth knowing about:

```env
ANTHROPIC_API_KEY=sk-ant-...     # enables /logs/entries/{id}/explain
GEMINI_API_KEY=...               # ...or use Gemini instead
ANOMALY_THRESHOLD=0.5            # override per-row anomaly cutoff
```

If neither LLM key is set, the Explain button gracefully falls back to
the rule-based reason text.

## Architecture at a glance

```
┌───────────────┐    HTTPS      ┌────────────────────┐    SQL    ┌──────────┐
│  Next.js (UI) │ ────────────▶ │   FastAPI service  │ ────────▶ │ Postgres │
│  /dashboard   │   JWT cookie  │   /logs/* /models  │           └──────────┘
└───────────────┘               └─────┬────────┬─────┘
                                      │        │ optional
                                      │        ▼
                                      │   Claude / Gemini
                                      ▼
                              .joblib + .pt artifacts
                              (RF, AE, Transformer)
```

The FastAPI service is stateless: every upload is parsed and analyzed
in-memory inside a single request, then committed to Postgres. Trained
model artifacts are loaded once per process and cached. There is no
local disk dependency for request handling, so adding more replicas
works without changes.

## CI / linting

`.github/workflows/lint.yml` runs Super-Linter v8.6.0 on push and PR
with Python (Black / Flake8 / isort / Ruff), TypeScript/JS (ESLint /
Prettier), JSON, YAML, Markdown, Dockerfile, and GitHub Actions checks.
`ci.yml` runs the backend pytest suite.

## Deploying to GCP Cloud Run

```bash
gcloud run deploy tenex-backend \
    --source ./fastapi_backend --region us-central1 \
    --set-env-vars DATABASE_URL=...,ACCESS_SECRET_KEY=...,ANTHROPIC_API_KEY=...

gcloud run deploy tenex-frontend \
    --source ./nextjs-frontend --region us-central1 \
    --set-env-vars API_BASE_URL=https://tenex-backend-xxx.a.run.app
```

Use `--update-secrets` for production-grade secret handling. A
`cloudbuild.yaml` is included for trigger-based deploys.
