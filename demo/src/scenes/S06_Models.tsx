import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading } from "../components/ui";
import { useEntrance } from "../components/ui";
import { AssetImage } from "../components/AssetImage";

const ModelCard: React.FC<{
  title: string;
  tag: string;
  desc: string;
  color: string;
  delay: number;
}> = ({ title, tag, desc, color, delay }) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        flex: "1 1 0",
        background: COLORS.panel,
        border: `1px solid ${COLORS.panelLine}`,
        borderLeft: `6px solid ${color}`,
        borderRadius: 16,
        padding: "26px 32px",
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 16 }}>
        <span style={{ fontSize: 44, fontWeight: 800 }}>{title}</span>
        <span style={{ fontSize: 30, fontWeight: 700, color }}>{tag}</span>
      </div>
      <div style={{ fontSize: 30, color: COLORS.textDim, lineHeight: 1.35 }}>{desc}</div>
    </div>
  );
};

export const S06_Models: React.FC = () => (
  <SceneFrame>
    <Eyebrow>The models</Eyebrow>
    <Heading size={62}>Spatial context vs. per-pixel baselines</Heading>

    <div style={{ flex: 1, display: "flex", gap: 30, marginTop: 36, minHeight: 0 }}>
      <div style={{ flex: "1.05 1 0", display: "flex", flexDirection: "column", gap: 24 }}>
        <ModelCard
          title="U-Net"
          tag="≈ 1.9M params"
          desc="A compact encoder-decoder that learns spatial context across each tile."
          color={COLORS.accent}
          delay={12}
        />
        <ModelCard
          title="Forest baselines"
          tag="RF · HGB"
          desc="Random forest (300 trees) and gradient boosting: one pixel at a time, on CPU."
          color={COLORS.accent2}
          delay={20}
        />
      </div>

      <div style={{ flex: "1 1 0", display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div style={{ fontSize: 28, color: COLORS.textDim, marginBottom: 14 }}>
          Same three bands. Depth carries the strongest signal:
        </div>
        <AssetImage
          name="feature_importance.png"
          fit="contain"
          card
          kenBurns={false}
          style={{ flex: 1, minHeight: 0 }}
        />
      </div>
    </div>
  </SceneFrame>
);
