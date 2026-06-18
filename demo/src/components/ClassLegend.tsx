import React from "react";
import { CLASSES, COLORS } from "../theme";
import { useEntrance } from "./ui";

/** Horizontal legend of the four label classes. */
export const ClassLegend: React.FC<{ delay?: number; size?: number }> = ({
  delay = 0,
  size = 34,
}) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        display: "flex",
        gap: 38,
        flexWrap: "wrap",
        alignItems: "center",
      }}
    >
      {CLASSES.map((c) => (
        <div
          key={c.key}
          style={{ display: "flex", alignItems: "center", gap: 14 }}
        >
          <div
            style={{
              width: size * 0.7,
              height: size * 0.7,
              borderRadius: 6,
              background: c.color,
              border: `1px solid ${COLORS.panelLine}`,
            }}
          />
          <span style={{ fontSize: size, color: COLORS.text }}>{c.label}</span>
        </div>
      ))}
    </div>
  );
};
