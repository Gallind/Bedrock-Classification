// Generates one ElevenLabs TTS clip per scene from src/narration.json, writes
// them to public/audio/scene-<id>.mp3, and records each clip's measured
// duration in public/audio/manifest.json. Remotion reads that manifest to
// re-time the composition so the visuals match the narration exactly.
//
//   cp .env.example .env   # then add ELEVENLABS_API_KEY
//   npm run voiceover           # idempotent: skips clips that already exist
//   npm run voiceover -- --force  # regenerate everything
//
// Decoupled from rendering on purpose: this is the only step that needs the
// API key / network; the render itself is deterministic and offline.

import "dotenv/config";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseFile } from "music-metadata";

const here = path.dirname(fileURLToPath(import.meta.url));
const demoRoot = path.resolve(here, "..");
const narrationPath = path.join(demoRoot, "src", "narration.json");
const audioDir = path.join(demoRoot, "public", "audio");

const force = process.argv.includes("--force");

const API_KEY = process.env.ELEVENLABS_API_KEY;
const VOICE_ID = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM"; // Rachel
const MODEL_ID = process.env.ELEVENLABS_MODEL_ID || "eleven_multilingual_v2";

if (!API_KEY) {
  console.error(
    "ELEVENLABS_API_KEY is not set.\n" +
      "  cp .env.example .env  and add your key, then re-run `npm run voiceover`."
  );
  process.exit(1);
}

const scenes = JSON.parse(await readFile(narrationPath, "utf8"));
await mkdir(audioDir, { recursive: true });

const synthesize = async (text) => {
  const res = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}?output_format=mp3_44100_128`,
    {
      method: "POST",
      headers: {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
        Accept: "audio/mpeg",
      },
      body: JSON.stringify({
        text,
        model_id: MODEL_ID,
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
          style: 0.0,
          use_speaker_boost: true,
        },
      }),
    }
  );
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`ElevenLabs ${res.status} ${res.statusText} — ${detail.slice(0, 300)}`);
  }
  return Buffer.from(await res.arrayBuffer());
};

const manifest = {};
console.log(`Voice ${VOICE_ID} · model ${MODEL_ID}${force ? " · --force" : ""}`);

for (const { id, narration } of scenes) {
  const file = path.join(audioDir, `scene-${id}.mp3`);
  if (existsSync(file) && !force) {
    console.log(`  = scene-${id}.mp3 (exists, skipping)`);
  } else {
    process.stdout.write(`  + scene-${id}.mp3 … `);
    const audio = await synthesize(narration);
    await writeFile(file, audio);
    console.log(`${(audio.length / 1024).toFixed(0)} KB`);
  }
  const { format } = await parseFile(file);
  const seconds = Number(format.duration?.toFixed(3) ?? 0);
  manifest[id] = { seconds };
}

await writeFile(
  path.join(audioDir, "manifest.json"),
  JSON.stringify(manifest, null, 2) + "\n"
);

const total = Object.values(manifest).reduce((a, m) => a + m.seconds, 0);
console.log(
  `\nWrote manifest.json — ${scenes.length} clips, ${total.toFixed(1)} s of narration.`
);
