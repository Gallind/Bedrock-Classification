import { loadFont } from "@remotion/google-fonts/Inter";
import palette from "../shared/palette.json";

export const { fontFamily } = loadFont("normal", {
  weights: ["400", "700", "800"],
  subsets: ["latin"],
});

// Video format
export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;

// Crossfade length between scenes (frames). Used by both Video.tsx (the actual
// transition) and Root.tsx (to subtract overlap from the total duration).
export const TRANSITION_FRAMES = 15;

// Deep-ocean palette + the four label classes (red / salmon / blue / grey),
// matching the classified-map colour scheme used across reports/. The values
// live in shared/palette.json so the python-pptx deck reads the same source.
export const COLORS = palette.colors;

export const CLASSES = palette.classes;
