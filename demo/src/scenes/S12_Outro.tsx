import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading, Body } from "../components/ui";
import { useEntrance } from "../components/ui";

export const S12_Outro: React.FC = () => {
  const credit = useEntrance(30);
  return (
    <SceneFrame>
      <Eyebrow>Summary</Eyebrow>
      <Heading size={80}>
        A reproducible pipeline from raw MBES survey to a trained seabed
        classifier
      </Heading>
      <Body size={42}>
        Rock is mapped reliably across surveys; shallow rock and labelled area
        remain the bottlenecks. The next gains come from more annotation — not a
        bigger model.
      </Body>
      <div
        style={{
          ...credit,
          marginTop: "auto",
          fontSize: 30,
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
