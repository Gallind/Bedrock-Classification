import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading, Body, Highlight } from "../components/ui";
import { AssetImage } from "../components/AssetImage";
import { Figure } from "../components/Figure";

const bands = [
  { name: "band_bathymetry.jpg", label: "Bathymetry", sub: "depth" },
  { name: "band_backscatter.jpg", label: "Backscatter", sub: "hardness" },
  { name: "band_slope.jpg", label: "Slope", sub: "steepness" },
];

export const S02_Data: React.FC = () => (
  <SceneFrame padding={90}>
    <Eyebrow>The data</Eyebrow>
    <Heading size={62}>Three physical bands on a shared 1&nbsp;m grid</Heading>

    <div style={{ flex: 1, display: "flex", gap: 30, marginTop: 40, minHeight: 0 }}>
      {bands.map((b, i) => (
        <Figure key={b.name} label={b.label} sublabel={b.sub} delay={10 + i * 6}>
          <AssetImage name={b.name} fit="cover" style={{ width: "100%", height: "100%" }} />
        </Figure>
      ))}
    </div>

    <Body size={34} delay={30}>
      Experts hand-draw the ground truth as polygons. But across four surveys we
      have only <Highlight>~1–2 km² of labelled seabed</Highlight> — about 100×
      less than the 576 km² reference of Garone et&nbsp;al. (2023) — and
      backscatter is stored differently per survey, from raw decibels to
      grayscale: <Highlight color={COLORS.warn}>a real domain shift</Highlight>.
    </Body>
  </SceneFrame>
);
