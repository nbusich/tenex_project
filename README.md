
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

## ML Model Explination
Training Data: CSIC-2010 Web Attack dataset with synthetic timestamps and IP addresses. Features were processed into 67 dimensions with 1-hot encoding, scaling, and feature-engineering.
Random Forest: Supervised method that aggregates predictions from decision trees that have randomized inputs and features. Great inference speed but lacks ability to detect attacks that it isn't trained on.
Autoencoder: Trained on normal logs to minimize MSE between input and output. Compresses to bottleneck and decompresses to original input size. Reconstruction loss is used during training, and during inference to identify anomalies.
Transformer: Encoder only, given a sequence of logs, one is masked and the model reconstructs it, similar to BERT. Loss is measured between input and reconstruction, and is used to identify anomalies (high loss) during inference. 
