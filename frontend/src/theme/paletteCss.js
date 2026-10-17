// Turns the THEMES list into the [data-theme] CSS blocks tokens.css builds on.
// Runs at build time inside the palettes plugin (vite.config.js); it is plain
// data-in, text-out so tests can call it too.

import { DEFAULT_PALETTE, THEMES, TOKEN_NAMES } from "./themes.js";

function block(selector, declarations) {
  return `${selector} {\n${declarations.map((line) => `  ${line};`).join("\n")}\n}\n`;
}

export function paletteCss(themes = THEMES) {
  const out = [];
  // The default palette also sits on :root, which ties with [data-theme] on
  // specificity, so it goes first: a later :root block would override
  // whichever palettes were written before it.
  const ordered = [
    ...themes.filter((theme) => theme.slug === DEFAULT_PALETTE),
    ...themes.filter((theme) => theme.slug !== DEFAULT_PALETTE),
  ];
  for (const theme of ordered) {
    const selector =
      theme.slug === DEFAULT_PALETTE
        ? `:root,\n[data-theme="${theme.slug}"]`
        : `[data-theme="${theme.slug}"]`;
    out.push(
      block(selector, [
        `color-scheme: ${theme.scheme}`,
        ...TOKEN_NAMES.map((name) => `--${name}: ${theme.tokens[name]}`),
      ])
    );
  }

  const dark = themes.filter((theme) => theme.scheme === "dark");
  if (dark.length) {
    out.push(
      block(dark.map((theme) => `[data-theme="${theme.slug}"]`).join(",\n"), [
        "--scrim: rgba(0, 0, 0, 0.68)",
        "--shadow-1: 0 1px 2px rgba(0, 0, 0, 0.4)",
        "--shadow-2: 0 6px 20px rgba(0, 0, 0, 0.45)",
        "--shadow-3: 0 16px 48px rgba(0, 0, 0, 0.6)",
      ])
    );
  }

  const square = themes.filter((theme) => theme.square);
  if (square.length) {
    out.push(
      block(square.map((theme) => `[data-theme="${theme.slug}"]`).join(",\n"), [
        "--r-sm: 0",
        "--r: 0",
        "--r-lg: 0",
      ])
    );
  }
  return out.join("\n");
}
