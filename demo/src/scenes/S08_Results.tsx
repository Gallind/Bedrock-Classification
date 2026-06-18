import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading } from "../components/ui";
import { StatBig } from "../components/StatBig";
import { MetricBar } from "../components/MetricBar";
import { AssetImage } from "../components/AssetImage";

export const S08_Results: React.FC = () => (
  <SceneFrame padding={90}>
    <Eyebrow>Results</Eyebrow>
    <Heading size={58}>3-band U-Net wins — but shallow rock is the weak spot</Heading>

    <div style={{ flex: 1, display: "flex", gap: 36, marginTop: 30, minHeight: 0 }}>
      <div style={{ flex: "1 1 0", display: "flex", flexDirection: "column" }}>
        <div style={{ display: "flex", gap: 22 }}>
          <StatBig
            value="0.784"
            caption="Within-survey (dev)"
            note="macro-Dice"
            color={COLORS.accent}
            delay={10}
            size={84}
          />
          <StatBig
            value="0.608"
            caption="Cross-survey (LOPO)"
            note="macro-Dice · ± 0.084"
            color={COLORS.accent2}
            delay={16}
            size={84}
          />
        </div>

        <div style={{ fontSize: 28, color: COLORS.textDim, marginTop: 30 }}>
          Cross-survey per-class Dice (LOPO):
        </div>
        <MetricBar label="Rock" value={0.841} color={COLORS.rock} delay={26} />
        <MetricBar label="Shallow rock" value={0.371} color={COLORS.shallow} delay={32} />
        <MetricBar label="Sand" value={0.612} color={COLORS.sand} delay={38} />
      </div>

      <div style={{ flex: "1 1 0", display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div style={{ fontSize: 28, color: COLORS.textDim, marginBottom: 14 }}>
          Model comparison, within-survey test split:
        </div>
        <AssetImage
          name="metrics_by_type.png"
          fit="contain"
          card
          kenBurns={false}
          style={{ flex: 1, minHeight: 0 }}
        />
      </div>
    </div>
  </SceneFrame>
);
