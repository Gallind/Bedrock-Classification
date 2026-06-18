import { staticFile } from "remotion";
import { SCENES } from "./script";
import { FPS } from "./theme";

/** Silence padding added after each narration clip (seconds). Kept short so
 * scenes cut on soon after the voiceover ends instead of lingering. */
const PAD_SECONDS = 0.35;

export type Manifest = Record<string, { seconds: number }>;

export const estDurationsFrames = (): number[] =>
  SCENES.map((s) => Math.round(s.estSeconds * FPS));

export const durationsFromManifest = (manifest: Manifest | null): number[] =>
  SCENES.map((s) => {
    const m = manifest?.[s.id];
    if (m && m.seconds > 0) {
      return Math.ceil((m.seconds + PAD_SECONDS) * FPS);
    }
    return Math.round(s.estSeconds * FPS);
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
