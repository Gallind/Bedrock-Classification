// Copies a curated set of repository visuals into demo/public/assets/ with
// clean, space-free names (Remotion can only serve files under public/).
// Re-run any time the source charts/maps/GIFs are regenerated.
//
//   npm run copy-assets
//
// Missing sources are warned about and skipped, so a partial dataset still
// produces a usable preview.

import { cp, mkdir, readFile } from "node:fs/promises";
import { existsSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const demoRoot = path.resolve(here, "..");
const repoRoot = path.resolve(demoRoot, "..");
const outDir = path.join(demoRoot, "public", "assets");

// The asset list is shared with the python-pptx deck: shared/assets.json maps a
// logical name (also the filename written here, so the scenes keep referencing
// it unchanged) to a path relative to the repo root. Edit that file once and
// both the video and the deck stay in sync.
const manifest = JSON.parse(
  await readFile(path.join(demoRoot, "shared", "assets.json"), "utf8")
);
const ASSETS = Object.entries(manifest.assets);

const fmt = (bytes) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;

await mkdir(outDir, { recursive: true });

let copied = 0;
const missing = [];
for (const [name, rel] of ASSETS) {
  const src = path.join(repoRoot, rel);
  if (!existsSync(src)) {
    missing.push(rel);
    continue;
  }
  await cp(src, path.join(outDir, name));
  copied += 1;
  console.log(`  + ${name.padEnd(26)} (${fmt(statSync(src).size)})`);
}

console.log(`\nCopied ${copied}/${ASSETS.length} assets -> ${path.relative(repoRoot, outDir)}`);
if (missing.length) {
  console.warn(`\nMissing (skipped):`);
  for (const m of missing) console.warn(`  - ${m}`);
  console.warn(
    "\nHint: run `git lfs pull` and regenerate reports/ if these are expected."
  );
  if (copied === 0) process.exitCode = 1;
}
