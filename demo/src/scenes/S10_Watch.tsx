import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading } from "../components/ui";
import { useEntrance } from "../components/ui";
import { AssetGif } from "../components/AssetGif";

export const S10_Watch: React.FC = () => {
  const note = useEntrance(20);
  return (
    <SceneFrame padding={90}>
      <Eyebrow>The pipeline, live</Eyebrow>
      <Heading size={60}>Predicting tile by tile, every model at once</Heading>

      <div style={{ flex: 1, display: "flex", gap: 40, marginTop: 30, alignItems: "center", minHeight: 0 }}>
        <AssetGif
          name="watch_polygon3.gif"
          width={1080}
          height={574}
          style={{ flex: "none" }}
        />
        <div style={{ ...note, flex: 1 }}>
          <div style={{ fontSize: 36, lineHeight: 1.45, color: COLORS.text }}>
            Each frame shows the input bands and every model&rsquo;s prediction
            while the full-survey map fills in live.
          </div>
          <div
            style={{
              marginTop: 28,
              padding: "22px 26px",
              background: COLORS.panel,
              border: `1px solid ${COLORS.panelLine}`,
              borderLeft: `6px solid ${COLORS.accent2}`,
              borderRadius: 14,
              fontSize: 30,
              lineHeight: 1.4,
              color: COLORS.textDim,
            }}
          >
            It&rsquo;s how we caught a labelling bug: two of the largest
            annotations were silently dropped — and the map made it obvious.
          </div>
        </div>
      </div>
    </SceneFrame>
  );
};
