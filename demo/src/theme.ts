import { loadFont } from "@remotion/google-fonts/Inter";

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
// matching the classified-map colour scheme used across reports/.
export const COLORS = {
  bg: "#081a24",
  bgAlt: "#0d2734",
  panel: "#103241",
  panelLine: "#1d4a5c",
  text: "#eef5f7",
  textDim: "#9fb8c2",
  accent: "#36c6cf",
  accent2: "#f6c75e",
  rock: "#d64545",
  shallow: "#f0a07a",
  sand: "#4a78c4",
  unlabeled: "#8a98a0",
  good: "#5fd08a",
  warn: "#e8a13a",
} as const;

export const CLASSES = [
  { key: "rock", label: "Rock", color: COLORS.rock },
  { key: "shallow_rock", label: "Shallow buried rock", color: COLORS.shallow },
  { key: "sand", label: "Sand", color: COLORS.sand },
  { key: "background", label: "Unlabeled", color: COLORS.unlabeled },
] as const;
