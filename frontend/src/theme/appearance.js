// Appearance state outside React: saved choices, applying a palette, the
// reduced-motion preference and the palette cross-fade.  index.html repeats
// the reads before first paint so a reload never flashes the defaults.

import { REDUCE_MOTION, findTheme, resolveTheme } from "./themes.js";

export function readStored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeStored(key, value) {
  try {
    if (value == null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* storage blocked: the choice lasts for this page only */
  }
}

function matches(query) {
  return Boolean(window.matchMedia && window.matchMedia(query).matches);
}

export function prefersDark() {
  return matches("(prefers-color-scheme: dark)");
}

// Motion is reduced when the OS asks for it or the in-app switch is on.
export function motionReduced() {
  return (
    document.documentElement.dataset.motion === REDUCE_MOTION ||
    matches("(prefers-reduced-motion: reduce)")
  );
}

export function applyMotion(value) {
  const root = document.documentElement;
  if (value === REDUCE_MOTION) root.dataset.motion = REDUCE_MOTION;
  else delete root.dataset.motion;
}

let swapsRunning = 0;

// Applies a palette to <html> and the OS chrome.
export function applyPalette(value) {
  const root = document.documentElement;
  const resolved = resolveTheme(value, prefersDark());
  const swap = () => {
    root.dataset.theme = resolved;
    const theme = findTheme(resolved);
    document.getElementById("meta-theme-color")?.setAttribute("content", theme.tokens.bg);
    document.getElementById("meta-color-scheme")?.setAttribute("content", theme.scheme);
  };
  if (root.dataset.theme === resolved) {
    swap();
    return;
  }
  // Before the first paint (no .theme-ready yet), with reduced motion, or
  // without View Transitions the swap is direct; the app's colour
  // transitions (html.theme-ready) carry it where they apply.
  if (
    !root.classList.contains("theme-ready") ||
    motionReduced() ||
    typeof document.startViewTransition !== "function"
  ) {
    swap();
    return;
  }
  // A View Transition fades a snapshot of the old palette into the live new
  // one, opacity only.  The per-element colour transitions are paused for it
  // so the new page does not tween on top of the fade.
  // A newer swap skips the running one, so the pause lifts after the last.
  swapsRunning += 1;
  root.classList.add("theme-swapping");
  document.startViewTransition(swap).finished.finally(() => {
    swapsRunning -= 1;
    if (!swapsRunning) root.classList.remove("theme-swapping");
  });
}
