  docker compose run --rm --entrypoint sh backend -c "uv sync --group ml-train"
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.data.prepare"

  # Train every model in one go (recommended after changing any preprocessor):
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.train_all"

  # Or train individually:
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.train_rf"
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.train_autoencoder"
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.train_transformer"

  # Re-tune the AE / Transformer squash thresholds after training (no
  # weight changes — only rewrites calibration_threshold in config.json):
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.recalibrate --pct 95"