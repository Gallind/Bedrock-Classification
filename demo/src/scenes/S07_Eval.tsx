import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading } from "../components/ui";
import { useEntrance } from "../components/ui";

const Panel: React.FC<{
  title: string;
  tag: string;
  tagColor: string;
  desc: string;
  delay: number;
  children: React.ReactNode;
}> = ({ title, tag, tagColor, desc, delay, children }) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        flex: "1 1 0",
        background: COLORS.panel,
        border: `1px solid ${COLORS.panelLine}`,
        borderRadius: 18,
        padding: 36,
        display: "flex",
        flexDirection: "column",
        gap: 22,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <span style={{ fontSize: 40, fontWeight: 800 }}>{title}</span>
        <span
          style={{
            fontSize: 24,
            fontWeight: 800,
            letterSpacing: 2,
            textTransform: "uppercase",
            color: COLORS.bg,
            background: tagColor,
            borderRadius: 999,
            padding: "6px 16px",
          }}
        >
          {tag}
        </span>
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
        {children}
      </div>
      <div style={{ fontSize: 30, color: COLORS.textDim, lineHeight: 1.4 }}>{desc}</div>
    </div>
  );
};

const Band: React.FC<{ label: string; color: string; grow: number }> = ({
  label,
  color,
  grow,
}) => (
  <div
    style={{
      flex: grow,
      background: `${color}33`,
      border: `2px solid ${color}`,
      borderRadius: 8,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: 28,
      fontWeight: 700,
      color: COLORS.text,
    }}
  >
    {label}
  </div>
);

const Poly: React.FC<{ label: string; test?: boolean }> = ({ label, test }) => (
  <div
    style={{
      width: 120,
      height: 120,
      borderRadius: 12,
      background: test ? `${COLORS.accent2}33` : `${COLORS.accent}22`,
      border: `2px solid ${test ? COLORS.accent2 : COLORS.accent}`,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      gap: 4,
    }}
  >
    <span style={{ fontSize: 24, color: COLORS.text }}>{label}</span>
    <span
      style={{
        fontSize: 22,
        fontWeight: 700,
        color: test ? COLORS.accent2 : COLORS.accent,
      }}
    >
      {test ? "test" : "train"}
    </span>
  </div>
);

export const S07_Eval: React.FC = () => (
  <SceneFrame>
    <Eyebrow>How we evaluate</Eyebrow>
    <Heading size={62}>Two protocols — one optimistic, one honest</Heading>

    <div style={{ flex: 1, display: "flex", gap: 30, marginTop: 30, minHeight: 0 }}>
      <Panel
        title="Within-survey"
        tag="dev / optimistic"
        tagColor={COLORS.accent}
        desc="Each survey is carved into train, validation and test bands with buffers — no pixel is shared."
        delay={12}
      >
        <div style={{ display: "flex", gap: 14, width: 420, height: 200 }}>
          <Band label="train" color={COLORS.good} grow={3} />
          <Band label="val" color={COLORS.accent} grow={1} />
          <Band label="test" color={COLORS.accent2} grow={1.4} />
        </div>
      </Panel>

      <Panel
        title="Leave-one-polygon-out"
        tag="LOPO / honest"
        tagColor={COLORS.accent2}
        desc="Train on three surveys, predict the fourth, repeat — does it generalise to a survey it has never seen?"
        delay={20}
      >
        <div style={{ display: "flex", gap: 18, alignItems: "center" }}>
          <Poly label="P1" />
          <Poly label="P3" />
          <Poly label="P4" />
          <span style={{ fontSize: 44, color: COLORS.textDim }}>→</span>
          <Poly label="P5" test />
        </div>
      </Panel>
    </div>
  </SceneFrame>
);
