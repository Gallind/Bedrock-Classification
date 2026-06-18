import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading } from "../components/ui";
import { AssetImage } from "../components/AssetImage";
import { Figure } from "../components/Figure";
import { ClassLegend } from "../components/ClassLegend";

const maps = [
  { name: "map_p1_ground_truth.png", label: "Ground truth", sub: "expert" },
  { name: "map_p1_unet.png", label: "U-Net", sub: "3-band" },
  { name: "map_p1_rf_spatial.png", label: "Random forest", sub: "guided-spatial" },
];

export const S09_Maps: React.FC = () => (
  <SceneFrame padding={90}>
    <Eyebrow>Classified maps · polygon 1</Eyebrow>
    <Heading size={58}>The same story holds across the full survey</Heading>

    <div style={{ flex: 1, display: "flex", gap: 30, marginTop: 30, minHeight: 0 }}>
      {maps.map((m, i) => (
        <Figure key={m.name} label={m.label} sublabel={m.sub} delay={10 + i * 6}>
          <AssetImage
            name={m.name}
            fit="contain"
            kenBurns={false}
            style={{ width: "100%", height: "100%" }}
          />
        </Figure>
      ))}
    </div>

    <div style={{ marginTop: 26, display: "flex", justifyContent: "center" }}>
      <ClassLegend delay={30} />
    </div>
  </SceneFrame>
);
