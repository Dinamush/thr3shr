import React from "react";

// 16px line icons used by the shell.  Stroke follows currentColor.
const PATHS = {
  tag: (
    <>
      <path d="M2.5 3.2v4.1l6.2 6.2 4.8-4.8-6.2-6.2H3.2z" />
      <circle cx="5.4" cy="5.4" r="1" />
    </>
  ),
  review: (
    <>
      <rect x="2.2" y="2.2" width="11.6" height="11.6" rx="2" />
      <path d="m5.2 8.2 1.9 1.9 3.8-4" />
    </>
  ),
  runs: (
    <>
      <circle cx="8" cy="8" r="5.6" />
      <path d="M8 5v3.2l2.2 1.4" />
    </>
  ),
  settings: (
    <>
      <circle cx="8" cy="8" r="2.1" />
      <path d="M8 1.8v1.6M8 12.6v1.6M14.2 8h-1.6M3.4 8H1.8M12.4 3.6l-1.1 1.1M4.7 11.3l-1.1 1.1M12.4 12.4l-1.1-1.1M4.7 4.7 3.6 3.6" />
    </>
  ),
  debug: (
    <>
      <path d="M6 2h4M6.8 2v4.2L3.4 12.4a1 1 0 0 0 .9 1.6h7.4a1 1 0 0 0 .9-1.6L9.2 6.2V2" />
      <path d="M4.9 10h6.2" />
    </>
  ),
  search: (
    <>
      <circle cx="7" cy="7" r="4.6" />
      <path d="m10.4 10.4 3.4 3.4" />
    </>
  ),
  chevron: <path d="M10 3 5 8l5 5" />,
  close: <path d="m4 4 8 8M12 4l-8 8" />,
  play: <path d="M5 3.2v9.6L12.6 8z" />,
  back: <path d="M9.5 3.5 5 8l4.5 4.5M5.5 8h7" />,
};

export default function Icon({ name, size = 16, strokeWidth = 1.6, className }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width={size}
      height={size}
      aria-hidden="true"
      focusable="false"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {PATHS[name]}
    </svg>
  );
}
