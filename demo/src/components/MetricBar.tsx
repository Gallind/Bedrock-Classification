import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { COLORS } from "../theme";

/** A labelled horizontal bar that grows to `value` (0..max) with a spring. */
export const MetricBar: React.FC<{
  label: string;
  value: number;
  max?: number;
  color?: string;
  delay?: number;
  /** override the printed value (defaults to value.toFixed(3)) */
  display?: string;
}> = ({ label, value, max = 1, color = COLORS.accent, delay = 0, display }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const grow = spring({
    frame: frame - delay,
    fps,
    durationInFrames: 26,
    config: { damping: 200 },
  });
  const opacity = interpolate(grow, [0, 0.15], [0, 1], {
    extrapolateRight: "clamp",
  });
  const widthPct = (grow * value * 100) / max;

  return (
    <div style={{ opacity, marginTop: 22 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 10,
        }}
      >
        <span style={{ fontSize: 34, color: COLORS.text }}>{label}</span>
        <span style={{ fontSize: 34, fontWeight: 700, color }}>
          {display ?? value.toFixed(3)}
        </span>
      </div>
      <div
        style={{
          height: 18,
          borderRadius: 9,
          background: COLORS.panel,
          border: `1px solid ${COLORS.panelLine}`,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${widthPct}%`,
            background: color,
            borderRadius: 9,
          }}
        />
      </div>
    </div>
  );
};
