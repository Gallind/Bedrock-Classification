# Seabed Classification — Demo Video (`demo/`)

A ~5-minute [Remotion](https://github.com/remotion-dev/remotion) (React → MP4) walkthrough of
the IOLR seabed-classification pipeline, for a Code4Good / academic audience: data → tiling →
augmentation contract → models → evaluation → results → live watch viewer → conclusions.

The narration is an [ElevenLabs](https://elevenlabs.io) text-to-speech voiceover. By design the
**voiceover generation is decoupled from the render**: the composition is fully previewable and
renderable *without* audio (it falls back to estimated per-scene timings), then re-times itself to
the narration once the voiceover manifest exists.

```
demo/
  src/
    Root.tsx        <Composition id="SeabedDemo"> + async calculateMetadata
    Video.tsx       TransitionSeries of all 12 scenes (crossfades) + one <Audio> per scene
    script.ts       pairs each scene id with its visual component
    narration.json  the script: { id, estSeconds, narration } — single source for the TTS
    timings.ts      reads public/audio/manifest.json (fallback to estSeconds)
    scenes/         S01_Intro … S12_Outro
    components/     AssetImage, AssetGif, MetricBar, StatBig, Figure, ClassLegend, ui
  scripts/
    copy-assets.mjs        curate repo charts/maps/GIFs → public/assets/
    generate-voiceover.mjs ElevenLabs TTS → public/audio/*.mp3 + manifest.json
  public/
    assets/         generated (gitignored) — run copy-assets
    audio/          generated (gitignored) — run voiceover
```

`public/assets/`, `public/audio/`, `out/`, and `.env` are gitignored — they are regenerated, never
committed (same policy as the pipeline's `outputs/`).

## Prerequisites

- **Node 18+** (developed on Node 25 / npm 11 — newer than Remotion's tested range; install and
  render work but may emit engine warnings. If a headless render misbehaves, use Node 20/22 via
  `nvm`).
- The repo's visuals must be real files, not Git LFS pointers: run `git lfs pull` at the repo root
  first if you only see pointer text.

## 1. Install

```bash
cd demo
npm install
```

## 2. Copy the source visuals

The demo reuses charts, classified maps, and watch GIFs already in the repo. Remotion can only
serve files under `public/`, so curate them once (re-run any time the source `reports/` are
regenerated):

```bash
npm run copy-assets
```

## 3. Preview (no audio needed)

```bash
npm run studio      # opens Remotion Studio → play "SeabedDemo"
```

All 12 scenes play with fallback timings; the watch GIFs animate and the charts/maps render.

## 4. Generate the voiceover (optional, needs an API key)

```bash
cp .env.example .env        # then add ELEVENLABS_API_KEY
npm run voiceover           # one mp3 per scene + manifest.json (idempotent)
npm run voiceover -- --force  # regenerate everything
```

Environment variables (`.env`):

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `ELEVENLABS_API_KEY` | yes | — | from your ElevenLabs account |
| `ELEVENLABS_VOICE_ID` | no | `21m00Tcm4TlvDq8ikWAM` (Rachel) | any voice id |
| `ELEVENLABS_MODEL_ID` | no | `eleven_multilingual_v2` | TTS model |

Reopen the studio after generating — the scenes now re-time to the narration and the audio plays
in sync. To change the script, edit `src/narration.json` and re-run `npm run voiceover --
--force`.

## 5. Render the MP4

```bash
npm run render      # → out/seabed-demo.mp4
```

The final duration is driven by the voiceover (≈4–5 min). Without a voiceover it renders at the
fallback timings (≈4.8 min). The MP4 is an output, not committed — share frozen exports via
external storage.
