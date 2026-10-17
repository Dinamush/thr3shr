import React, { useId } from "react";
import { INK, TILE_BOTTOM, TILE_TOP, iconShapes, levelFor } from "./appIcon";

// The THR3SHR icon set ("Grain Bin"): rounded 1.5px line icons on a 16px
// grid.  Stroke follows currentColor; `f` parts are solid.
const ICONS = {
  // Navigation
  run: (
    <>
      <path d="M2.75 3.75a1 1 0 0 1 1-1h3.67a1 1 0 0 1 .7.29l5.3 5.3a1 1 0 0 1 0 1.42l-3.66 3.66a1 1 0 0 1-1.42 0l-5.3-5.3a1 1 0 0 1-.29-.7z" />
      <circle cx="5.75" cy="5.75" r="1.05" fill="currentColor" />
    </>
  ),
  review: (
    <>
      <path d="M5.25 2.75h7a1 1 0 0 1 1 1v7" />
      <rect x="2.75" y="5.25" width="8" height="8" rx="1.5" />
      <path d="m4.9 9.35 1.45 1.45 2.8-2.9" />
    </>
  ),
  runs: (
    <>
      <path d="M3.45 5.4A5.25 5.25 0 1 1 3.07 9.8" />
      <path d="M3.45 2.9v2.5h2.5" />
      <path d="M8 5.25V8l2 1.25" />
    </>
  ),
  settings: (
    <>
      <path d="M2.75 5H9M12.2 5h1.05M2.75 11h1.05M7 11h6.25" />
      <circle cx="10.6" cy="5" r="1.6" />
      <circle cx="5.4" cy="11" r="1.6" />
    </>
  ),
  debug: (
    <>
      <path d="M6 2.75h4M6.75 2.75v3.7L3.3 12.1a.8.8 0 0 0 .68 1.15h8.04a.8.8 0 0 0 .68-1.15L9.25 6.45v-3.7" />
      <path d="M4.7 10.25h6.6" />
    </>
  ),
  // Shell
  search: (
    <>
      <circle cx="7" cy="7" r="4.25" />
      <path d="m10.2 10.2 3.05 3.05" />
    </>
  ),
  collapse: (
    <>
      <rect x="2.75" y="2.75" width="10.5" height="10.5" rx="2" />
      <path d="M6.25 2.75v10.5" />
      <path className="icon-collapse-chevron" d="M10.25 6.25 8.5 8l1.75 1.75" />
    </>
  ),
  // Three separate bars so the phone "More" button can morph them into a cross.
  more: (
    <>
      <path className="icon-more-bar" d="M2.75 4.5h10.5" />
      <path className="icon-more-bar" d="M2.75 8h10.5" />
      <path className="icon-more-bar" d="M2.75 11.5h10.5" />
    </>
  ),
  appearance: (
    <>
      <circle cx="8" cy="8" r="5.25" />
      <path d="M8 2.75a5.25 5.25 0 0 1 0 10.5z" fill="currentColor" />
    </>
  ),
  close: <path d="M4.25 4.25l7.5 7.5M11.75 4.25l-7.5 7.5" />,
  back: <path d="M6.75 3.75 2.5 8l4.25 4.25M2.75 8h10.5" />,
  forward: <path d="M9.25 3.75 13.5 8l-4.25 4.25M13.25 8H2.75" />,
  check: <path d="m3.25 8.5 3 3 6.5-7" />,
  // Actions
  play: (
    <path
      d="M5 3.6v8.8a.75.75 0 0 0 1.14.64l7-4.4a.75.75 0 0 0 0-1.28l-7-4.4A.75.75 0 0 0 5 3.6z"
      fill="currentColor"
    />
  ),
  scan: (
    <>
      <path d="M2.75 5.5V4a1.25 1.25 0 0 1 1.25-1.25h1.5M10.5 2.75H12A1.25 1.25 0 0 1 13.25 4v1.5M13.25 10.5V12A1.25 1.25 0 0 1 12 13.25h-1.5M5.5 13.25H4A1.25 1.25 0 0 1 2.75 12v-1.5" />
      <path d="M5 8h6" />
    </>
  ),
  add: <path d="M8 3.25v9.5M3.25 8h9.5" />,
  approve: <path d="m3.25 8.5 3 3 6.5-7" />,
  reject: (
    <>
      <circle cx="8" cy="8" r="5.25" />
      <path d="M4.3 11.7l7.4-7.4" />
    </>
  ),
  migrate: (
    <>
      <path d="M2.75 4.75a1 1 0 0 1 1-1h2.8l1.5 1.5h4.2a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1h-8.5a1 1 0 0 1-1-1z" />
      <path d="M6 9h4.25M8.75 7.5 10.25 9l-1.5 1.5" />
    </>
  ),
  stop: <rect x="4" y="4" width="8" height="8" rx="1.5" fill="currentColor" />,
  reclassify: (
    <>
      <path d="M13 6.4A5.25 5.25 0 0 0 3.6 5M3 9.6A5.25 5.25 0 0 0 12.4 11" />
      <path d="M3.25 2.5V5h2.5M12.75 13.5V11h-2.5" />
    </>
  ),
  list: (
    <>
      <path d="M6 4.25h7.25M6 8h7.25M6 11.75h7.25" />
      <circle cx="3.2" cy="4.25" r="0.85" fill="currentColor" />
      <circle cx="3.2" cy="8" r="0.85" fill="currentColor" />
      <circle cx="3.2" cy="11.75" r="0.85" fill="currentColor" />
    </>
  ),
  grid: (
    <>
      <rect x="2.75" y="2.75" width="4.25" height="4.25" rx="1" />
      <rect x="9" y="2.75" width="4.25" height="4.25" rx="1" />
      <rect x="2.75" y="9" width="4.25" height="4.25" rx="1" />
      <rect x="9" y="9" width="4.25" height="4.25" rx="1" />
    </>
  ),
  unload: (
    <>
      <path d="M8 3.25 3.25 8.75h9.5z" />
      <path d="M3.25 12.75h9.5" />
    </>
  ),
  edit: (
    <>
      <path d="M10.25 3.25l2.5 2.5-7 7H3.25v-2.5z" />
      <path d="M8.75 4.75l2.5 2.5" />
    </>
  ),
  external: (
    <path d="M9.5 2.75h3.75V6.5M13.25 2.75 7.5 8.5M11.25 9.5v2.75a1 1 0 0 1-1 1h-6.5a1 1 0 0 1-1-1v-6.5a1 1 0 0 1 1-1H6.5" />
  ),
  // Media and engine
  video: (
    <>
      <rect x="2" y="3.75" width="12" height="8.5" rx="2" />
      <path d="M6.75 6.1v3.8L9.9 8z" fill="currentColor" />
    </>
  ),
  gpu: (
    <>
      <rect x="4.25" y="4.25" width="7.5" height="7.5" rx="1.5" />
      <path d="M6.5 2.25v2M9.5 2.25v2M6.5 11.75v2M9.5 11.75v2M2.25 6.5h2M2.25 9.5h2M11.75 6.5h2M11.75 9.5h2" />
    </>
  ),
  offline: <path d="M5.5 2.75v2.5M10.5 2.75v2.5M4 5.25h8v2.5a4 4 0 0 1-8 0zM8 11.75v1.5" />,
};

export const ICON_NAMES = Object.keys(ICONS);

// Below 14px the stroke thickens a touch so the line still reads.
export default function Icon({ name, size = 16, strokeWidth, className }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width={size}
      height={size}
      aria-hidden="true"
      focusable="false"
      className={className ? `icon ${className}` : "icon"}
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth ?? (size <= 13 ? 1.65 : 1.5)}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {ICONS[name]}
    </svg>
  );
}

// Status glyphs: a ring (a warning triangle for sensitive) and a mark, drawn
// on a 12px grid so the 12px chip size lands on whole pixels.  Straight
// strokes sit on x.625 / y.625 so one edge meets the pixel grid.
const RING = <circle cx="6" cy="6" r="4.45" />;
const TRIANGLE = <path d="M6 1.6 10.95 10.375H1.05z" />;
const CHECK = <path d="M3.9 6.1 5.35 7.55 8.1 4.7" />;

const STATUS_GLYPHS = {
  proposed: null,
  reviewed: <circle cx="6" cy="6" r="1.5" fill="currentColor" stroke="none" />,
  approved: CHECK,
  rejected: <path d="M4.4 4.4l3.2 3.2M7.6 4.4 4.4 7.6" />,
  migrated: <path d="M3.75 5.625h3.9M6.25 4.1l1.55 1.525L6.25 7.15" />,
  needs: (
    <>
      <path d="M4.7 4.75a1.35 1.35 0 1 1 1.95 1.2c-.42.22-.65.47-.65.88v.17" />
      <circle cx="6" cy="8.45" r="0.62" fill="currentColor" stroke="none" />
    </>
  ),
  sensitive: (
    <>
      <path d="M5.625 4.9v1.9" />
      <circle cx="5.625" cy="8.6" r="0.6" fill="currentColor" stroke="none" />
    </>
  ),
  fallback: <path d="M5.625 3.75v4.25M4 6.4l1.625 1.625L7.25 6.4" />,
  queued: <path d="M5.625 3.9v2.35h1.9" />,
  running: <path d="M4.9 4.05v3.9l3.2-1.95z" fill="currentColor" />,
  completed: CHECK,
  cancelled: <path d="M3.9 5.625h4.2" />,
  failed: (
    <>
      <path d="M5.625 3.6v2.75" />
      <circle cx="5.625" cy="8.2" r="0.66" fill="currentColor" stroke="none" />
    </>
  ),
};

export const STATUS_GLYPH_NAMES = Object.keys(STATUS_GLYPHS);

// An unknown name draws the bare ring, like "proposed".
export function StatusGlyph({ name, size = 12, className }) {
  return (
    <svg
      viewBox="0 0 12 12"
      width={size}
      height={size}
      aria-hidden="true"
      focusable="false"
      className={className ? `status-glyph ${className}` : "status-glyph"}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.25"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {name === "sensitive" ? TRIANGLE : RING}
      {STATUS_GLYPHS[name]}
    </svg>
  );
}

// The app icon in the shell: the same artwork as the favicon, at the level
// meant for its size.
export function BrandMark({ size = 22 }) {
  // useId's colons are not safe inside url(#...).
  const gradientId = `brand-tile${useId().replace(/:/g, "")}`;
  const s = iconShapes(levelFor(size));
  return (
    <svg
      viewBox={`0 0 ${s.grid} ${s.grid}`}
      width={size}
      height={size}
      aria-hidden="true"
      focusable="false"
      className="app-brand-mark"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={TILE_TOP} />
          <stop offset="1" stopColor={TILE_BOTTOM} />
        </linearGradient>
      </defs>
      <rect className="app-brand-tile" {...s.tile} fill={`url(#${gradientId})`} />
      {s.rim ? (
        <rect {...s.rim} fill="none" stroke="#fff" strokeOpacity=".32" strokeWidth="1.2" />
      ) : null}
      <path
        d={s.bin.d}
        fill={INK}
        stroke={s.bin.strokeWidth ? INK : undefined}
        strokeWidth={s.bin.strokeWidth || undefined}
        strokeLinejoin="round"
      />
      {s.grainPath ? (
        <path
          d={s.grainPath.d}
          fill="none"
          stroke={INK}
          strokeWidth={s.grainPath.strokeWidth}
          strokeLinecap="round"
        />
      ) : (
        s.grains.map((g) => <circle key={`${g.cx},${g.cy}`} {...g} fill={INK} />)
      )}
    </svg>
  );
}
