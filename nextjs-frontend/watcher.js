/* eslint-disable @typescript-eslint/no-require-imports */
const chokidar = require("chokidar");
const { exec } = require("child_process");
const { config } = require("dotenv");
const fs = require("fs");

config({ path: ".env.local" });

const openapiFile = process.env.OPENAPI_OUTPUT_FILE;

let running = false;
let pending = false;

const regen = () => {
  if (running) {
    pending = true;
    return;
  }
  running = true;
  // Wipe the output dir first so openapi-ts never trips over its own
  // non-atomic rmdir on Docker Desktop's macOS bind mount (ENOTEMPTY).
  exec(
    "rm -rf app/openapi-client && pnpm run generate-client",
    (error, stdout, stderr) => {
      running = false;
      if (error) console.error(`generate-client error: ${error.message}`);
      if (stderr) console.error(`stderr: ${stderr}`);
      if (stdout) console.log(stdout);
      if (pending) {
        pending = false;
        regen();
      }
    },
  );
};

// If openapi.json is already present at startup but openapi-client is
// missing, regen once so the dev server can resolve `./openapi-client`
// on its very first compile.
if (
  openapiFile &&
  fs.existsSync(openapiFile) &&
  !fs.existsSync("app/openapi-client/index.ts")
) {
  console.log(
    "[watcher] openapi.json present but openapi-client missing — running initial regen.",
  );
  regen();
}

chokidar
  .watch(openapiFile)
  .on("add", (path) => {
    console.log(`File ${path} appeared. Regenerating client...`);
    regen();
  })
  .on("change", (path) => {
    console.log(`File ${path} has been modified. Regenerating client...`);
    regen();
  });
