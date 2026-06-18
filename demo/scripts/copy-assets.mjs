// Copies a curated set of repository visuals into demo/public/assets/ with
// clean, space-free names (Remotion can only serve files under public/).
// Re-run any time the source charts/maps/GIFs are regenerated.
//
//   npm run copy-assets
//
// Missing sources are warned about and skipped, so a partial dataset still
// produces a usable preview.

import { cp, mkdir } from "node:fs/promises";
import { existsSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const demoRoot = path.resolve(here, "..");
const repoRoot = path.resolve(demoRoot, "..");
const outDir = path.join(demoRoot, "public", "assets");

// [ source relative to repo root, destination filename in public/assets ]
const ASSETS = [
  // Scene 2 — the three input bands + a hillshade
  ["DataBase/polygon1/bathymetry Grid 0.5m.jpg", "band_bathymetry.jpg"],
  ["DataBase/polygon1/Back scatter 0.2m.jpg", "band_backscatter.jpg"],
  ["DataBase/polygon3/HB_Poly3_Slope_1m.jpg", "band_slope.jpg"],
  ["DataBase/polygon3/HB_Poly3_HS_1m.jpg", "band_hillshade.jpg"],

  // Scene 6 — feature importance
  [
    "training/runs/forest_3band/feature_importance_random_forest.png",
    "feature_importance.png",
  ],

  // Scene 8 — results charts
  ["reports/learning_curves.png", "learning_curves.png"],
  ["reports/confusion_matrices.png", "confusion_matrices.png"],
  ["reports/metrics_by_type.png", "metrics_by_type.png"],

  // Scene 9 — classified maps (polygon1)
  [
    "reports/classified_maps/polygon1/polygon1__ground_truth__t128m_o50pct_r1m.png",
    "map_p1_ground_truth.png",
  ],
  [
    "reports/classified_maps/polygon1/polygon1__unet__experiment_3band__t128m_o50pct_r1m.png",
    "map_p1_unet.png",
  ],
  [
    "reports/classified_maps/polygon1/polygon1__random_forest_raw__forest_3band__t128m_o50pct_r1m.png",
    "map_p1_rf_raw.png",
  ],
  [
    "reports/classified_maps/polygon1/polygon1__random_forest_spatial__forest_3band__t128m_o50pct_r1m.png",
    "map_p1_rf_spatial.png",
  ],

  // Scene 10 — live watch viewer (hero)
  ["reports/watch_gifs/polygon3_watch_multi.gif", "watch_polygon3.gif"],
  ["reports/watch_gifs/polygon1_watch_multi.gif", "watch_polygon1.gif"],
];

const fmt = (bytes) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;

await mkdir(outDir, { recursive: true });

let copied = 0;
const missing = [];
for (const [rel, name] of ASSETS) {
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
