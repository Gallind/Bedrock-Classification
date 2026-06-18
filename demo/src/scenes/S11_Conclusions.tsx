import React from "react";
import { COLORS } from "../theme";
import { SceneFrame, Eyebrow, Heading, Bullet } from "../components/ui";
import { useEntrance } from "../components/ui";
import { AssetImage } from "../components/AssetImage";

const Step: React.FC<{ n: string; children: React.ReactNode; delay: number }> = ({
  n,
  children,
  delay,
}) => {
  const e = useEntrance(delay);
  return (
    <div style={{ ...e, display: "flex", gap: 18, alignItems: "center", marginTop: 20 }}>
      <span
        style={{
          flex: "none",
          width: 46,
          height: 46,
          borderRadius: 12,
          background: COLORS.accent2,
          color: COLORS.bg,
          fontSize: 26,
          fontWeight: 800,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {n}
      </span>
      <span style={{ fontSize: 34, color: COLORS.text }}>{children}</span>
    </div>
  );
};

export const S11_Conclusions: React.FC = () => {
  const steps = useEntrance(24);
  return (
    <SceneFrame>
      <Eyebrow>Conclusions & future work</Eyebrow>
      <Heading size={60}>The ceiling is data, not the architecture</Heading>

      <div style={{ flex: 1, display: "flex", gap: 50, marginTop: 30, minHeight: 0 }}>
        <div style={{ flex: "1 1 0" }}>
          <Bullet delay={12} color={COLORS.rock}>
            Rock classification is reliable across surveys, useful for hazard and
            habitat mapping.
          </Bullet>
          <Bullet delay={18} color={COLORS.shallow}>
            Shallow buried rock is the next frontier: defined partly by depth, and we
            simply have very little of it labelled.
          </Bullet>
          <Bullet delay={24} color={COLORS.accent}>
            More gains will come from more labelled data than from a bigger model.
          </Bullet>
        </div>

        <div style={{ flex: "1 1 0", display: "flex", flexDirection: "column" }}>
          <div
            style={{
              ...steps,
              fontSize: 28,
              fontWeight: 800,
              letterSpacing: 3,
              textTransform: "uppercase",
              color: COLORS.accent2,
            }}
          >
            Highest-value next steps
          </div>
          <Step n="1" delay={30}>More annotated shallow buried rock</Step>
          <Step n="2" delay={36}>Hillshade as a 4th band</Step>
          <Step n="3" delay={42}>Engineered multi-scale features</Step>
          <Step n="4" delay={48}>Model ensembling</Step>

          <div style={{ marginTop: "auto", display: "flex", alignItems: "center", gap: 18 }}>
            <AssetImage
              name="band_hillshade.jpg"
              fit="cover"
              kenBurns={false}
              style={{ width: 200, height: 120, flex: "none" }}
            />
            <span style={{ fontSize: 26, color: COLORS.textDim }}>
              candidate 4th band: hillshade
            </span>
          </div>
        </div>
      </div>
    </SceneFrame>
  );
};
