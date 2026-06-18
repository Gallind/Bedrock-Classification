import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading, Highlight } from "../components/ui";
import { StatBig } from "../components/StatBig";
import { MetricBar } from "../components/MetricBar";
import { AssetImage } from "../components/AssetImage";

export const S08_Results: React.FC = () => (
  <SceneFrame padding={90}>
    <Eyebrow>Results</Eyebrow>
    <Heading size={58}>The 3-band U-Net wins, and rock maps reliably</Heading>

    <div style={{ flex: 1, display: "flex", gap: 36, marginTop: 30, minHeight: 0 }}>
      <div style={{ flex: "1 1 0", display: "flex", flexDirection: "column" }}>
        <div style={{ display: "flex", gap: 22 }}>
          <StatBig
            value="0.784"
            caption="Within-survey (dev)"
            note="macro-Dice"
            color={COLORS.accent}
            delay={10}
            size={80}
          />
          <StatBig
            value="0.976"
            caption="Rock, best survey"
            note="cross-survey Dice (polygon3)"
            color={COLORS.rock}
            delay={16}
            size={80}
          />
        </div>

        <div style={{ fontSize: 28, color: COLORS.textDim, marginTop: 30 }}>
          Cross-survey per-class Dice (LOPO mean):
        </div>
        <MetricBar label="Rock" value={0.841} color={COLORS.rock} delay={26} />
        <MetricBar label="Sand" value={0.612} color={COLORS.sand} delay={32} />
        <div style={{ fontSize: 26, color: COLORS.textDim, marginTop: 22, lineHeight: 1.4 }}>
          Overall <Highlight>0.608 ± 0.084</Highlight> macro-Dice across unseen
          surveys, beating the 2-band and tree baselines on every metric.
        </div>
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
