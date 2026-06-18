import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { COLORS } from "../theme";

/** Fade-up entrance driven by a spring; `delay` is in frames. */
export const useEntrance = (delay = 0, durationInFrames = 22) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({
    frame: frame - delay,
    fps,
    durationInFrames,
    config: { damping: 200 },
  });
  return {
    opacity: interpolate(s, [0, 1], [0, 1]),
    transform: `translateY(${interpolate(s, [0, 1], [22, 0])}px)`,
  };
};

/** Full-frame scene background with a subtle vertical gradient + padding. */
export const SceneFrame: React.FC<
  React.PropsWithChildren<{ padding?: number }>
> = ({ children, padding = 110 }) => (
  <AbsoluteFill
    style={{
      background: `linear-gradient(160deg, ${COLORS.bg} 0%, ${COLORS.bgAlt} 100%)`,
      color: COLORS.text,
      padding,
      display: "flex",
      flexDirection: "column",
    }}
  >
    {children}
  </AbsoluteFill>
);

export const Eyebrow: React.FC<{ children: React.ReactNode; delay?: number }> = ({
  children,
  delay = 0,
}) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        color: COLORS.accent,
        fontSize: 30,
        fontWeight: 700,
        letterSpacing: 6,
        textTransform: "uppercase",
        marginBottom: 18,
      }}
    >
      {children}
    </div>
  );
};

export const Heading: React.FC<{
  children: React.ReactNode;
  delay?: number;
  size?: number;
}> = ({ children, delay = 4, size = 86 }) => {
  const e = useEntrance(delay);
  return (
    <h1
      style={{
        ...e,
        margin: 0,
        fontSize: size,
        lineHeight: 1.05,
        fontWeight: 800,
        maxWidth: 1500,
      }}
    >
      {children}
    </h1>
  );
};

export const Body: React.FC<{
  children: React.ReactNode;
  delay?: number;
  size?: number;
  dim?: boolean;
}> = ({ children, delay = 10, size = 40, dim = true }) => {
  const e = useEntrance(delay);
  return (
    <p
      style={{
        ...e,
        margin: "26px 0 0",
        fontSize: size,
        lineHeight: 1.4,
        fontWeight: 400,
        color: dim ? COLORS.textDim : COLORS.text,
        maxWidth: 1450,
      }}
    >
      {children}
    </p>
  );
};

/** Accent-barred bullet row with staggered entrance. */
export const Bullet: React.FC<{
  children: React.ReactNode;
  delay?: number;
  color?: string;
}> = ({ children, delay = 0, color = COLORS.accent }) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        display: "flex",
        alignItems: "flex-start",
        gap: 22,
        fontSize: 40,
        lineHeight: 1.32,
        marginTop: 26,
      }}
    >
      <div
        style={{
          flex: "none",
          width: 14,
          height: 14,
          borderRadius: 4,
          background: color,
          marginTop: 18,
        }}
      />
      <div style={{ color: COLORS.text }}>{children}</div>
    </div>
  );
};

export const Highlight: React.FC<{ children: React.ReactNode; color?: string }> = ({
  children,
  color = COLORS.accent2,
}) => <span style={{ color, fontWeight: 700 }}>{children}</span>;
