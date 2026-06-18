import React from "react";
import { Composition } from "remotion";
import { Video } from "./Video";
import { SCENES } from "./script";
import { FPS, WIDTH, HEIGHT, TRANSITION_FRAMES } from "./theme";
import {
  audioScenesFromManifest,
  durationsFromManifest,
  estDurationsFrames,
  loadManifest,
} from "./timings";

const totalFrames = (perScene: number[]): number =>
  // Crossfades overlap adjacent scenes, so the timeline is shorter than the sum.
  perScene.reduce((a, b) => a + b, 0) -
  TRANSITION_FRAMES * (perScene.length - 1);

export const RemotionRoot: React.FC = () => {
  const fallback = estDurationsFrames();
  return (
    <Composition
      id="SeabedDemo"
      component={Video}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      durationInFrames={totalFrames(fallback)}
      defaultProps={{ durations: fallback, audioScenes: [] as string[] }}
      calculateMetadata={async () => {
        const manifest = await loadManifest();
        const durations = durationsFromManifest(manifest);
        return {
          durationInFrames: totalFrames(durations),
          props: {
            durations,
            audioScenes: audioScenesFromManifest(manifest),
          },
        };
      }}
    />
  );
};
