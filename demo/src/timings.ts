import { staticFile } from "remotion";
import { SCENES } from "./script";
import { FPS } from "./theme";

/** Silence padding added after each narration clip (seconds). Kept short so
 * scenes cut on soon after the voiceover ends instead of lingering. */
const PAD_SECONDS = 0.35;

/** Playback speed for the no-audio fallback. 2 = twice as fast (scenes half as
 * long) so the silent demo lands near 2 min. Only the fallback is scaled — when
 * a voiceover manifest exists, scenes re-time to the real spoken length so the
 * narration is never clipped. */
const FALLBACK_SPEED = 2;

export type Manifest = Record<string, { seconds: number }>;

const fallbackFrames = (estSeconds: number): number =>
  Math.round((estSeconds * FPS) / FALLBACK_SPEED);

export const estDurationsFrames = (): number[] =>
  SCENES.map((s) => fallbackFrames(s.estSeconds));

export const durationsFromManifest = (manifest: Manifest | null): number[] =>
  SCENES.map((s) => {
    const m = manifest?.[s.id];
    if (m && m.seconds > 0) {
      return Math.ceil((m.seconds + PAD_SECONDS) * FPS);
    }
    return fallbackFrames(s.estSeconds);
  });

export const audioScenesFromManifest = (manifest: Manifest | null): string[] =>
  manifest
    ? SCENES.filter((s) => manifest[s.id]?.seconds > 0).map((s) => s.id)
    : [];

/** Fetch the generated voiceover manifest; null if it doesn't exist yet. */
export const loadManifest = async (): Promise<Manifest | null> => {
  try {
    const res = await fetch(staticFile("audio/manifest.json"));
    if (!res.ok) return null;
    return (await res.json()) as Manifest;
  } catch {
    return null;
  }
};
