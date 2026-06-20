import type React from "react";
import type { SceneSpec } from "./types";
import narration from "../shared/narration.json";
import { S01_Intro } from "./scenes/S01_Intro";
import { S02_Data } from "./scenes/S02_Data";
import { S03_Pipeline } from "./scenes/S03_Pipeline";
import { S04_Modes } from "./scenes/S04_Modes";
import { S05_Augment } from "./scenes/S05_Augment";
import { S06_Models } from "./scenes/S06_Models";
import { S07_Eval } from "./scenes/S07_Eval";
import { S08_Results } from "./scenes/S08_Results";
import { S09_Maps } from "./scenes/S09_Maps";
import { S10_Watch } from "./scenes/S10_Watch";
import { S11_Conclusions } from "./scenes/S11_Conclusions";
import { S12_Outro } from "./scenes/S12_Outro";

// The visual component for each scene id. The narration text + fallback
// durations live in narration.json, which is the single source the ElevenLabs
// voiceover script reads too (so audio and video can never drift out of sync).
const COMPONENTS: Record<string, React.FC> = {
  intro: S01_Intro,
  data: S02_Data,
  pipeline: S03_Pipeline,
  modes: S04_Modes,
  augment: S05_Augment,
  models: S06_Models,
  eval: S07_Eval,
  results: S08_Results,
  maps: S09_Maps,
  watch: S10_Watch,
  conclusions: S11_Conclusions,
  outro: S12_Outro,
};

export const SCENES: SceneSpec[] = narration.map((s) => {
  const Component = COMPONENTS[s.id];
  if (!Component) throw new Error(`No component registered for scene "${s.id}"`);
  return { ...s, Component };
});
