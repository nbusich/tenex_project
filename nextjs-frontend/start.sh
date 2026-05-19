#!/bin/bash
set -e

# Make sure the generated openapi-client exists before next dev starts
# compiling modules — otherwise webpack reports `./openapi-client` as
# "Module not found" until the watcher gets a chance to react to an
# openapi.json change (which may never happen if the backend reuses
# an existing schema file on startup).
if [ ! -f app/openapi-client/index.ts ] && [ -f "${OPENAPI_OUTPUT_FILE}" ]; then
  echo "[start.sh] No openapi-client found — running initial generate-client..."
  rm -rf app/openapi-client
  pnpm run generate-client || echo "[start.sh] generate-client failed; watcher will retry on openapi.json change"
fi

pnpm run dev &
node watcher.js
wait
