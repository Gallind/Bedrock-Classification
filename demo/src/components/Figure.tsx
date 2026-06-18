import React from "react";
import { COLORS } from "../theme";
import { useEntrance } from "./ui";

/** A visual (image/gif/diagram) with a caption row beneath it. */
export const Figure: React.FC<
  React.PropsWithChildren<{
    label: React.ReactNode;
    sublabel?: React.ReactNode;
    color?: string;
    delay?: number;
    flex?: number;
    align?: "center" | "flex-start";
  }>
> = ({ children, label, sublabel, color, delay = 0, flex, align = "center" }) => {
  const e = useEntrance(delay);
  return (
    <div
      style={{
        ...e,
        flex: flex ?? "1 1 0",
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: align,
      }}
    >
      <div style={{ width: "100%", flex: 1, minHeight: 0, display: "flex" }}>
        {children}
      </div>
      <div
        style={{
          marginTop: 18,
          display: "flex",
          alignItems: "center",
          gap: 12,
          alignSelf: align,
        }}
      >
        {color ? (
          <span
            style={{
              width: 16,
              height: 16,
              borderRadius: 5,
              background: color,
              flex: "none",
            }}
          />
        ) : null}
        <span style={{ fontSize: 32, fontWeight: 700, color: COLORS.text }}>
          {label}
        </span>
        {sublabel ? (
          <span style={{ fontSize: 28, color: COLORS.textDim }}>{sublabel}</span>
        ) : null}
      </div>
    </div>
  );
};
