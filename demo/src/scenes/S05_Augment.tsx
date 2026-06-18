import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading } from "../components/ui";
import { useEntrance } from "../components/ui";

const Item: React.FC<{ ok: boolean; children: React.ReactNode; delay: number }> = ({
  ok,
  children,
  delay,
}) => {
  const e = useEntrance(delay);
  const color = ok ? COLORS.good : "#e06a6a";
  return (
    <div style={{ ...e, display: "flex", gap: 18, alignItems: "flex-start", marginTop: 22 }}>
      <span style={{ color, fontSize: 38, fontWeight: 800, lineHeight: 1.1, flex: "none" }}>
        {ok ? "✓" : "✕"}
      </span>
      <span style={{ fontSize: 36, lineHeight: 1.3, color: COLORS.text }}>{children}</span>
    </div>
  );
};

const Column: React.FC<{
  title: string;
  color: string;
  children: React.ReactNode;
  delay: number;
}> = ({ title, color, children, delay }) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        flex: "1 1 0",
        background: COLORS.panel,
        border: `1px solid ${COLORS.panelLine}`,
        borderRadius: 18,
        padding: "30px 36px",
      }}
    >
      <div
        style={{
          fontSize: 30,
          fontWeight: 800,
          letterSpacing: 3,
          textTransform: "uppercase",
          color,
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
};

const Rule: React.FC<{ children: React.ReactNode; delay: number }> = ({ children, delay }) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        flex: "1 1 0",
        fontSize: 28,
        lineHeight: 1.35,
        color: COLORS.text,
        background: COLORS.bgAlt,
        border: `1px solid ${COLORS.panelLine}`,
        borderRadius: 14,
        padding: "20px 24px",
      }}
    >
      {children}
    </div>
  );
};

export const S05_Augment: React.FC = () => (
  <SceneFrame>
    <Eyebrow>Methods rigor · the augmentation contract</Eyebrow>
    <Heading size={60}>Pixels are physical measurements, so the rules are strict</Heading>

    <div style={{ display: "flex", gap: 30, marginTop: 36 }}>
      <Column title="Allowed: rigid geometry" color={COLORS.good} delay={12}>
        <Item ok delay={20}>The eight D4 rotations & flips</Item>
        <Item ok delay={26}>Genuine re-extraction from the source raster</Item>
      </Column>
      <Column title="Forbidden: photometric" color="#e06a6a" delay={16}>
        <Item ok={false} delay={22}>Brightness / contrast shifts</Item>
        <Item ok={false} delay={28}>Noise, blur, zoom: they corrupt the physics</Item>
      </Column>
    </div>

    <div style={{ display: "flex", gap: 20, marginTop: "auto" }}>
      <Rule delay={36}>Spatial splits only, so overlapping tiles never leak train to test</Rule>
      <Rule delay={42}>Loss & metrics ignore no-data and background pixels</Rule>
      <Rule delay={48}>Backscatter normalised per survey, bridging the domain shift</Rule>
    </div>
  </SceneFrame>
);
