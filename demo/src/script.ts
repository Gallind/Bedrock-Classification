import type { SceneSpec } from "./types";
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

// The ordered storyboard. `narration` is the single source for the ElevenLabs
// voiceover; `estSeconds` is the fallback on-screen duration used before any
// audio is generated. Numbers are quoted from training/README.md + reports/.
export const SCENES: SceneSpec[] = [
  {
    id: "intro",
    Component: S01_Intro,
    estSeconds: 22,
    narration:
      "Mapping the seafloor by hand is slow and inconsistent. Working with " +
      "the Israel Oceanographic and Limnological Research institute, we built " +
      "a pipeline that turns multibeam echosounder surveys from the research " +
      "vessel Bat-Galim into per-pixel maps of three seabed types — rock, " +
      "shallow rock, and sand — automatically, and reproducibly.",
  },
  {
    id: "data",
    Component: S02_Data,
    estSeconds: 28,
    narration:
      "Every survey gives us three physical layers on a one-metre grid. " +
      "Bathymetry is depth. Backscatter measures how acoustically hard the " +
      "seafloor is. Slope is the steepness of the terrain. Marine geologists " +
      "hand-draw the ground truth as polygons of rock, shallow rock, and sand. " +
      "But there's a catch: across four surveys we have only about one to two " +
      "square kilometres of labelled seabed — roughly a hundred times less " +
      "than the reference study. And backscatter is stored differently per " +
      "survey, from raw decibels to grayscale images — a real domain shift the " +
      "model has to survive.",
  },
  {
    id: "pipeline",
    Component: S03_Pipeline,
    estSeconds: 24,
    narration:
      "The pipeline is deterministic and config-driven. Every layer is " +
      "reprojected onto one shared grid in UTM zone 36 north. The expert " +
      "polygons are rasterised into per-pixel labels. Then each survey is cut " +
      "into overlapping tiles — 128 metres square, at one metre per pixel, " +
      "with fifty percent overlap — so every tile carries a three-band feature " +
      "stack and a matching label.",
  },
  {
    id: "modes",
    Component: S04_Modes,
    estSeconds: 22,
    narration:
      "Tiles are produced in three modes. The standard grid is axis-aligned. " +
      "The rotated grid aligns to the bounding box of the annotations, so " +
      "tiles follow the survey, not the compass. And an augmentation mode " +
      "re-extracts that rotated grid at jittered angles and offsets — turning " +
      "roughly 250 base tiles into over 900 for training.",
  },
  {
    id: "augment",
    Component: S05_Augment,
    estSeconds: 30,
    narration:
      "Because the pixels are physical measurements, augmentation follows a " +
      "strict contract. Only rigid geometry is allowed — rotations and flips, " +
      "the eight D-four operations, plus genuine re-extraction from the " +
      "source. Brightness, noise, and zoom are forbidden; they would corrupt " +
      "the physics. Splits are spatial only, never random, because " +
      "overlapping tiles would leak between train and test. Augmented tiles " +
      "stay in training. The loss ignores pixels with no data. And backscatter " +
      "is normalised per survey — which is what bridges that domain shift.",
  },
  {
    id: "models",
    Component: S06_Models,
    estSeconds: 24,
    narration:
      "We compare two model families on the same three bands. A compact " +
      "U-Net — about 1.9 million parameters — learns spatial context. Against " +
      "it, per-pixel tree baselines: a random forest and gradient boosting, " +
      "which see one pixel at a time. Feature importance confirms the " +
      "intuition — depth carries the strongest signal, followed by " +
      "backscatter, then slope.",
  },
  {
    id: "eval",
    Component: S07_Eval,
    estSeconds: 22,
    narration:
      "We evaluate two ways. The development split carves each survey into " +
      "separate train, validation, and test bands with buffers, so no pixel " +
      "is shared. The honest test is leave-one-polygon-out: train on three " +
      "surveys, predict the fourth, and repeat. That measures whether the " +
      "model generalises to a survey it has never seen.",
  },
  {
    id: "results",
    Component: S08_Results,
    estSeconds: 32,
    narration:
      "The results are clear. The three-band U-Net reaches a macro-Dice of " +
      "0.78 within surveys, and 0.61, plus or minus 0.08, across unseen " +
      "surveys — beating the two-band variant and the tree baselines on every " +
      "metric. Rock is robust everywhere, with a cross-survey Dice of 0.84. " +
      "Shallow rock is the persistent weak spot, around 0.37 — it's defined " +
      "partly by depth, and we simply have very little of it. The confusion " +
      "is almost always shallow rock mistaken for sand.",
  },
  {
    id: "maps",
    Component: S09_Maps,
    estSeconds: 22,
    narration:
      "Seen as full maps, the story holds. Against the ground truth, the " +
      "U-Net recovers the rock outcrops cleanly. The tree baselines capture " +
      "the broad structure but leave salt-and-pepper noise — which an " +
      "edge-aware spatial filter, guided by depth, smooths away for a small, " +
      "consistent gain.",
  },
  {
    id: "watch",
    Component: S10_Watch,
    estSeconds: 24,
    narration:
      "This is the pipeline running tile by tile. Each frame shows the input " +
      "bands and every model's prediction, while the full-survey map fills in " +
      "live. It's how we caught a labelling bug earlier — two of the largest " +
      "annotations were silently dropped, and the map made it obvious.",
  },
  {
    id: "conclusions",
    Component: S11_Conclusions,
    estSeconds: 26,
    narration:
      "So where does this land? Rock classification is reliable enough to be " +
      "useful for hazard and habitat mapping. Shallow rock is not yet there. " +
      "The ceiling isn't the architecture — it's data. The highest-value next " +
      "steps are more annotated shallow rock, a hillshade fourth band, " +
      "engineered multi-scale features, and ensembling. The infrastructure for " +
      "all of it already exists.",
  },
  {
    id: "outro",
    Component: S12_Outro,
    estSeconds: 16,
    narration:
      "From raw multibeam survey to a trained, honestly-evaluated seabed " +
      "classifier — fully reproducible. Built for IOLR, as part of Code4Good " +
      "at Reichman University.",
  },
];
