import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading } from "../components/ui";
import { useEntrance } from "../components/ui";

const steps = [
  { n: "1", title: "Align", sub: "reproject every layer onto one shared grid (EPSG:32636)" },
  { n: "2", title: "Rasterize labels", sub: "expert polygons → per-pixel class ids" },
  { n: "3", title: "Tile", sub: "cut overlapping feature + label tile pairs" },
];

const StepCard: React.FC<{ n: string; title: string; sub: string; delay: number }> = ({
  n,
  title,
  sub,
  delay,
}) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        flex: "1 1 0",
        background: COLORS.panel,
        border: `1px solid ${COLORS.panelLine}`,
        borderRadius: 18,
        padding: "34px 32px",
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      <div
        style={{
          width: 64,
          height: 64,
          borderRadius: 16,
          background: COLORS.accent,
          color: COLORS.bg,
          fontSize: 38,
          fontWeight: 800,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {n}
      </div>
      <div style={{ fontSize: 42, fontWeight: 800 }}>{title}</div>
      <div style={{ fontSize: 30, color: COLORS.textDim, lineHeight: 1.35 }}>{sub}</div>
    </div>
  );
};

const Arrow: React.FC<{ delay: number }> = ({ delay }) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        flex: "none",
        fontSize: 56,
        color: COLORS.accent,
        alignSelf: "center",
        fontWeight: 700,
      }}
    >
      →
    </div>
  );
};

const Spec: React.FC<{ children: React.ReactNode; delay: number }> = ({ children, delay }) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        fontSize: 32,
        fontWeight: 700,
        color: COLORS.text,
        background: COLORS.bgAlt,
        border: `1px solid ${COLORS.panelLine}`,
        borderRadius: 999,
        padding: "14px 28px",
      }}
    >
      {children}
    </div>
  );
};

export const S03_Pipeline: React.FC = () => (
  <SceneFrame>
    <Eyebrow>The pipeline</Eyebrow>
    <Heading size={64}>Deterministic and config-driven, end to end</Heading>

    <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 24 }}>
      <StepCard {...steps[0]} delay={10} />
      <Arrow delay={20} />
      <StepCard {...steps[1]} delay={26} />
      <Arrow delay={36} />
      <StepCard {...steps[2]} delay={42} />
    </div>

    <div style={{ display: "flex", gap: 18, flexWrap: "wrap", justifyContent: "center" }}>
      <Spec delay={54}>128 m tiles</Spec>
      <Spec delay={58}>1 m / pixel</Spec>
      <Spec delay={62}>50% overlap</Spec>
      <Spec delay={66}>3 feature bands + label</Spec>
    </div>
  </SceneFrame>
);
