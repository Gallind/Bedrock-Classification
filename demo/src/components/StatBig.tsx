import React from "react";
import { COLORS } from "../theme";
import { useEntrance } from "./ui";

/** A big headline number with a small caption above and a note below. */
export const StatBig: React.FC<{
  value: string;
  caption: string;
  note?: string;
  color?: string;
  delay?: number;
  size?: number;
}> = ({ value, caption, note, color = COLORS.accent, delay = 0, size = 110 }) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        background: COLORS.panel,
        border: `1px solid ${COLORS.panelLine}`,
        borderRadius: 18,
        padding: "28px 36px",
      }}
    >
      <div
        style={{
          fontSize: 24,
          letterSpacing: 3,
          textTransform: "uppercase",
          color: COLORS.textDim,
          fontWeight: 700,
        }}
      >
        {caption}
      </div>
      <div style={{ fontSize: size, fontWeight: 800, color, lineHeight: 1.05 }}>
        {value}
      </div>
      {note ? (
        <div style={{ fontSize: 28, color: COLORS.textDim, marginTop: 4 }}>
          {note}
        </div>
      ) : null}
    </div>
  );
};
