  
  ### Setup data and env
  docker compose run --rm --entrypoint sh backend -c "uv sync --group ml-train"
  \n

  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.data.prepare"
  \n

  ### Train all models:
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.train_all"
  \n
  ### Train individually:
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.train_rf"
  \n
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.train_autoencoder"
  \n
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.train_transformer"
  \n
  ### Re-tune the AE / Transformer thresholds
  docker compose run --rm --entrypoint sh backend -c "python -m app.anomaly.train.train.recalibrate --pct 99.9"