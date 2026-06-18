import React from "react";
import { staticFile } from "remotion";
import { Gif } from "@remotion/gif";
import { COLORS } from "../theme";

/**
 * Animated GIF playback. A plain <img> does NOT animate during a headless
 * render; @remotion/gif decodes frames deterministically so the watch viewer
 * actually moves in the final MP4.
 */
export const AssetGif: React.FC<{
  name: string;
  width: number;
  height: number;
  radius?: number;
  style?: React.CSSProperties;
}> = ({ name, width, height, radius = 16, style }) => (
  <div
    style={{
      overflow: "hidden",
      borderRadius: radius,
      border: `1px solid ${COLORS.panelLine}`,
      boxShadow: "0 24px 60px rgba(0,0,0,0.5)",
      lineHeight: 0,
      ...style,
    }}
  >
    <Gif
      src={staticFile(`assets/${name}`)}
      width={width}
      height={height}
      fit="contain"
    />
  </div>
);
