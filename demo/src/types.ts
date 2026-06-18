import type React from "react";

export type SceneSpec = {
  /** stable id; the voiceover clip is public/audio/scene-<id>.mp3 */
  id: string;
  /** narration text spoken over this scene (single source for TTS) */
  narration: string;
  /** fallback on-screen duration (seconds) used when no voiceover exists */
  estSeconds: number;
  /** the scene's visual component */
  Component: React.FC;
};

export type VideoProps = {
  /** per-scene durations in frames (from calculateMetadata) */
  durations: number[];
  /** scene ids that have a generated audio clip */
  audioScenes: string[];
};
