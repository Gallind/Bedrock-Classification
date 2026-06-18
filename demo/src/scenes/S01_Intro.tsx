import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading, Body, Highlight } from "../components/ui";
import { ClassLegend } from "../components/ClassLegend";

export const S01_Intro: React.FC = () => (
  <SceneFrame>
    <Eyebrow>IOLR · R/V Bat-Galim · Multibeam survey</Eyebrow>
    <Heading size={96}>
      Classifying the Israeli seabed with deep learning
    </Heading>
    <Body size={44}>
      Turning multibeam echosounder surveys into per-pixel maps of{" "}
      <Highlight color={COLORS.rock}>rock</Highlight>,{" "}
      <Highlight color={COLORS.shallow}>shallow buried rock</Highlight> and{" "}
      <Highlight color={COLORS.sand}>sand</Highlight>, replacing slow,
      hand-drawn annotation with a reproducible pipeline.
    </Body>
    <div style={{ marginTop: "auto" }}>
      <ClassLegend delay={26} />
    </div>
  </SceneFrame>
);
