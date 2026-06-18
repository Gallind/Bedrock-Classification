import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from "remotion";
import { COLORS } from "../theme";

/**
 * An image from public/assets/ with an optional slow Ken Burns push-in. The
 * zoom runs over a fixed window then holds, so it reads as motion regardless of
 * the (audio-driven) scene length. `card` wraps light matplotlib charts on a
 * white panel; dark-background maps/renders sit straight on the scene.
 */
export const AssetImage: React.FC<{
  /** filename inside public/assets/ */
  name: string;
  /** object-fit; charts look best "contain", photos "cover" */
  fit?: "cover" | "contain";
  /** white rounded card behind the image (for charts on white bg) */
  card?: boolean;
  kenBurns?: boolean;
  radius?: number;
  style?: React.CSSProperties;
}> = ({ name, fit = "cover", card = false, kenBurns = true, radius = 16, style }) => {
  const frame = useCurrentFrame();
  const scale = kenBurns
    ? interpolate(frame, [0, 600], [1.0, 1.07], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      })
    : 1;

  return (
    <div
      style={{
        position: "relative",
        overflow: "hidden",
        borderRadius: radius,
        background: card ? "#ffffff" : COLORS.panel,
        border: `1px solid ${card ? "#ffffff" : COLORS.panelLine}`,
        boxShadow: "0 24px 60px rgba(0,0,0,0.45)",
        ...style,
      }}
    >
      <AbsoluteFill style={{ padding: card ? 18 : 0 }}>
        <Img
          src={staticFile(`assets/${name}`)}
          style={{
            width: "100%",
            height: "100%",
            objectFit: fit,
            transform: `scale(${scale})`,
            transformOrigin: "center",
          }}
        />
      </AbsoluteFill>
    </div>
  );
};
