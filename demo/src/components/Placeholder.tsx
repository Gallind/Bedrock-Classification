import React from "react";
import { COLORS } from "../theme";
import { Eyebrow, Heading, Body, SceneFrame } from "./ui";

/** Temporary stub used by the scaffold until each scene is built out. */
export const Placeholder: React.FC<{ title: string; subtitle?: string }> = ({
  title,
  subtitle,
}) => (
  <SceneFrame>
    <Eyebrow>Scene</Eyebrow>
    <Heading>{title}</Heading>
    {subtitle ? <Body>{subtitle}</Body> : null}
    <div
      style={{
        marginTop: "auto",
        color: COLORS.textDim,
        fontSize: 28,
        letterSpacing: 2,
      }}
    >
      (placeholder — visuals pending)
    </div>
  </SceneFrame>
);

export const makePlaceholder =
  (title: string, subtitle?: string): React.FC =>
  () =>
    <Placeholder title={title} subtitle={subtitle} />;
