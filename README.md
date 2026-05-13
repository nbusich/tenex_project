# Anomaly Sentinel — ZScaler Log Analyzer

Upload ZScaler web proxy logs, score each entry for anomalies with one of
three ML models, and explore the results in a dashboard.

## Quick start

Prereqs: Docker + Docker Compose.

```bash
cp fastapi_backend/.env.example fastapi_backend/.env
cp nextjs-frontend/.env.example nextjs-frontend/.env

docker compose up --build -d
docker compose run --rm backend alembic upgrade head
```

Open <http://localhost:3000>, register an account, and upload a log from
`nextjs-frontend/public/samples/`.

## Optional: enable LLM explanations

Add either key to `fastapi_backend/.env` before `docker compose up`:

```env
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

Without a key, the **Explain** button falls back to rule-based text.
