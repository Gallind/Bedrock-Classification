import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading, Body } from "../components/ui";
import { useEntrance } from "../components/ui";

const TEAM = [
  "Gal Lind",
  "Adi Lind",
  "Eden Tsarfaty",
  "Adi Gotlib",
  "Tomer Sheffer",
];

export const S12_Outro: React.FC = () => {
  const team = useEntrance(24);
  const credit = useEntrance(34);
  return (
    <SceneFrame>
      <Eyebrow>Summary</Eyebrow>
      <Heading size={74}>
        A reproducible pipeline from raw MBES survey to a trained seabed
        classifier
      </Heading>
      <Body size={40}>
        Rock is mapped reliably across surveys; shallow rock and labelled area
        remain the bottlenecks. The next gains come from more annotation, not a
        bigger model.
      </Body>

      <div style={{ ...team, marginTop: 44 }}>
        <div
          style={{
            fontSize: 24,
            letterSpacing: 3,
            textTransform: "uppercase",
            color: COLORS.accent,
            fontWeight: 700,
            marginBottom: 14,
          }}
        >
          Team
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "14px 40px" }}>
          {TEAM.map((name) => (
            <span key={name} style={{ fontSize: 36, fontWeight: 700, color: COLORS.text }}>
              {name}
            </span>
          ))}
          <span style={{ fontSize: 36, fontWeight: 700, color: COLORS.accent2 }}>
            Asaf Giladi <span style={{ fontSize: 28, color: COLORS.textDim, fontWeight: 400 }}>(IOLR)</span>
          </span>
        </div>
      </div>

      <div
        style={{
          ...credit,
          marginTop: "auto",
          fontSize: 28,
          color: COLORS.textDim,
          lineHeight: 1.6,
        }}
      >
        <div>Code4Good · Reichman University</div>
        <div>
          For the Israel Oceanographic &amp; Limnological Research institute
          (IOLR)
        </div>
        <div>Method baseline: Garone et al., Frontiers in Earth Science, 2023</div>
      </div>
    </SceneFrame>
  );
};
