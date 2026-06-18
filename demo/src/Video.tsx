import React from "react";
import { AbsoluteFill, Audio, staticFile } from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { SCENES } from "./script";
import { estDurationsFrames } from "./timings";
import { COLORS, TRANSITION_FRAMES, fontFamily } from "./theme";
import type { VideoProps } from "./types";

export const Video: React.FC<VideoProps> = ({ durations, audioScenes }) => {
  const frames =
    durations && durations.length === SCENES.length
      ? durations
      : estDurationsFrames();
  const hasAudio = new Set(audioScenes ?? []);

  // TransitionSeries requires a flat list of Sequence / Transition children.
  const children: React.ReactNode[] = [];
  SCENES.forEach((scene, i) => {
    const { Component } = scene;
    children.push(
      <TransitionSeries.Sequence
        key={scene.id}
        durationInFrames={frames[i]}
      >
        <Component />
        {hasAudio.has(scene.id) ? (
          <Audio src={staticFile(`audio/scene-${scene.id}.mp3`)} />
        ) : null}
      </TransitionSeries.Sequence>
    );
    if (i < SCENES.length - 1) {
      children.push(
        <TransitionSeries.Transition
          key={`t-${scene.id}`}
          timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
          presentation={fade()}
        />
      );
    }
  });

  return (
    <AbsoluteFill style={{ backgroundColor: COLORS.bg, fontFamily }}>
      <TransitionSeries>{children}</TransitionSeries>
    </AbsoluteFill>
  );
};
