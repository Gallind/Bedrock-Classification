import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading } from "../components/ui";
import { useEntrance } from "../components/ui";

/** A small NxN grid of cells, optionally rotated, used as a tiling schematic. */
const MiniGrid: React.FC<{
  size: number;
  rotate?: number;
  color?: string;
  opacity?: number;
}> = ({ size, rotate = 0, color = COLORS.accent, opacity = 1 }) => (
  <div
    style={{
      position: "absolute",
      width: size,
      height: size,
      transform: `rotate(${rotate}deg)`,
      display: "grid",
      gridTemplateColumns: "repeat(3, 1fr)",
      gridTemplateRows: "repeat(3, 1fr)",
      gap: 4,
      opacity,
    }}
  >
    {Array.from({ length: 9 }).map((_, i) => (
      <div
        key={i}
        style={{
          border: `2px solid ${color}`,
          borderRadius: 4,
          background: `${color}22`,
        }}
      />
    ))}
  </div>
);

const ModeCard: React.FC<{
  title: string;
  sub: string;
  count: string;
  delay: number;
  children: React.ReactNode;
}> = ({ title, sub, count, delay, children }) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        flex: "1 1 0",
        background: COLORS.panel,
        border: `1px solid ${COLORS.panelLine}`,
        borderRadius: 18,
        padding: 32,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 20,
      }}
    >
      <div
        style={{
          position: "relative",
          width: 260,
          height: 260,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {children}
      </div>
      <div style={{ fontSize: 40, fontWeight: 800 }}>{title}</div>
      <div
        style={{
          fontSize: 28,
          color: COLORS.textDim,
          textAlign: "center",
          lineHeight: 1.35,
          minHeight: 76,
        }}
      >
        {sub}
      </div>
      <div style={{ fontSize: 30, fontWeight: 700, color: COLORS.accent2 }}>{count}</div>
    </div>
  );
};

export const S04_Modes: React.FC = () => (
  <SceneFrame>
    <Eyebrow>Tiling modes</Eyebrow>
    <Heading size={64}>One grid, three ways to cut it</Heading>

    <div style={{ flex: 1, display: "flex", gap: 30, marginTop: 30, alignItems: "stretch" }}>
      <ModeCard
        title="Standard"
        sub="axis-aligned grid, aligned to the compass"
        count="≈ 250 base tiles"
        delay={10}
      >
        <MiniGrid size={200} />
      </ModeCard>
      <ModeCard
        title="Rotated"
        sub="grid aligned to the annotation bounding box, follows the survey"
        count="252 tiles"
        delay={18}
      >
        <MiniGrid size={200} rotate={24} />
      </ModeCard>
      <ModeCard
        title="Augmented"
        sub="rotated grid re-extracted at jittered angles & offsets"
        count="909 tiles"
        delay={26}
      >
        <MiniGrid size={196} rotate={24} color={COLORS.unlabeled} opacity={0.45} />
        <MiniGrid size={196} rotate={10} color={COLORS.accent2} opacity={0.6} />
        <MiniGrid size={196} rotate={-6} color={COLORS.accent} opacity={0.9} />
      </ModeCard>
    </div>
  </SceneFrame>
);
