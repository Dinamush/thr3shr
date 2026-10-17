// Drives the theme picker in a DOM (jsdom): opening and closing, keyboard
// preview and selection, the focus trap, what persists, reduced motion and
// the Ctrl+K hand-off.  Layout and animation are not checked here: jsdom has
// no layout or rendering.

import assert from "node:assert/strict";
import { after, afterEach, before, beforeEach, test } from "node:test";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const dom = new JSDOM(
  `<!doctype html><html data-theme="graphite"><head>
    <meta id="meta-color-scheme" name="color-scheme" content="light dark">
    <meta id="meta-theme-color" name="theme-color" content="#f6f5f2">
  </head><body></body></html>`,
  { url: "http://localhost/", pretendToBeVisual: true }
);
const { window } = dom;
const media = { dark: true, reduce: false };
window.matchMedia = (query) => ({
  matches:
    (query.includes("prefers-color-scheme: dark") && media.dark) ||
    (query.includes("prefers-reduced-motion: reduce") && media.reduce),
  addEventListener() {},
  removeEventListener() {},
});
window.Element.prototype.scrollIntoView = function scrollIntoView() {};
for (const key of [
  "window",
  "document",
  "navigator",
  "localStorage",
  "HTMLElement",
  "Element",
  "Node",
  "KeyboardEvent",
  "MouseEvent",
  "requestAnimationFrame",
  "cancelAnimationFrame",
  "getComputedStyle",
]) {
  Object.defineProperty(globalThis, key, {
    configurable: true,
    writable: true,
    value: key === "window" ? window : window[key],
  });
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let server;
let load;
let createRoot;
let reactRoot;
let container;
let viewTransitions = 0;

before(async () => {
  ({ createRoot } = await import("react-dom/client"));
  server = await createServer({
    root,
    configFile: `${root}vite.config.js`,
    logLevel: "silent",
    server: { middlewareMode: true, hmr: false },
    appType: "custom",
  });
  load = (path) => server.ssrLoadModule(path);
});

after(async () => {
  await server?.close();
  window.close();
});

beforeEach(() => {
  localStorage.clear();
  media.dark = true;
  media.reduce = false;
  const html = document.documentElement;
  html.dataset.theme = "graphite";
  delete html.dataset.skin;
  delete html.dataset.motion;
  html.className = "";
  viewTransitions = 0;
  // Stands in for the browser's View Transitions API: runs the swap at once
  // and counts the cross-fades.
  document.startViewTransition = (swap) => {
    viewTransitions += 1;
    swap();
    return { finished: Promise.resolve() };
  };
  container = document.createElement("div");
  document.body.append(container);
});

afterEach(async () => {
  await act(async () => reactRoot?.unmount());
  reactRoot = null;
  container.remove();
});

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const html = () => document.documentElement;
const dialog = () => document.querySelector('[role="dialog"][aria-labelledby="themePickerTitle"]');
const radios = () => [...document.querySelectorAll('.theme-picker [role="radio"]')];
const checked = () => radios().find((el) => el.getAttribute("aria-checked") === "true");
const topButton = () => document.querySelector(".theme-picker-toggle");

async function press(key, options = {}) {
  const target = document.activeElement || document.body;
  await act(async () => {
    target.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...options }));
  });
}

// Lets the open frames and any close or pick timers run.  Time passes in
// short act() scopes: React holds updates from timers that fire inside one
// until it ends, so a single long scope would stall a timer chain such as
// pick, then close, then unmount.
async function settle(ms = 60) {
  const end = performance.now() + ms;
  do {
    await act(async () => {
      await wait(Math.min(10, Math.max(0, end - performance.now())));
    });
  } while (performance.now() < end);
}

// Compares DOM nodes by identity.  Never hand a node to assert.equal: on a
// mismatch it inspects both, and a jsdom node reaches the whole window, so
// building the message grows without bound.
function assertSame(actual, expected, message) {
  assert.ok(actual === expected, message);
}

function assertClosed(message = "picker closed") {
  assert.ok(dialog() === null, message);
}

async function mountPicker() {
  const { AppearanceProvider, AppearanceButton } = await load("/src/theme/Appearance.jsx");
  reactRoot = createRoot(container);
  await act(async () => {
    reactRoot.render(
      React.createElement(AppearanceProvider, null, React.createElement(AppearanceButton))
    );
  });
  await settle(); // html.theme-ready lands a frame after mount
}

async function openFromButton() {
  await act(async () => topButton().click());
  await settle();
}

test("the top-bar button opens an accessible modal on the saved palette", async () => {
  localStorage.setItem("thr3shr-theme", "blush");
  await mountPicker();
  assertClosed("closed until asked");
  assert.equal(topButton().getAttribute("aria-haspopup"), "dialog");
  assert.match(topButton().textContent, /Blush/);

  await openFromButton();
  const box = dialog();
  assert.ok(box, "dialog opens");
  assert.equal(box.getAttribute("aria-modal"), "true");
  assert.equal(document.getElementById("themePickerTitle").textContent, "Appearance");
  assert.ok(document.querySelector('[role="radiogroup"][aria-label="Theme"]'));
  assert.equal(topButton().getAttribute("aria-expanded"), "true");
  assert.ok(document.querySelector(".theme-picker").classList.contains("is-open"));
  assert.equal(radios().length, 16, "Auto plus all 15 palettes");
  assert.equal(checked().dataset.slug, "blush");
  assertSame(document.activeElement, checked(), "focus starts on the saved palette");
  // Roving tabindex: only the active palette is a Tab stop.
  assert.deepEqual(
    radios().filter((el) => el.tabIndex === 0).map((el) => el.dataset.slug),
    ["blush"]
  );
  for (const el of radios()) assert.ok(el.getAttribute("aria-label").includes(","), "name and description");
  assert.ok(document.body.classList.contains("theme-picker-open"), "page scroll locked");
});

test("arrows preview across the app and Esc puts the saved palette back", async () => {
  await mountPicker();
  await openFromButton();
  assert.equal(checked().dataset.slug, "auto");
  assert.equal(html().dataset.theme, "graphite", "Auto on a dark system");

  await press("ArrowRight"); // Graphite: same palette as Auto here
  await press("ArrowRight"); // Daylight
  assert.equal(html().dataset.theme, "daylight", "previewed live");
  assert.equal(document.activeElement.dataset.slug, "daylight");
  assert.equal(document.activeElement.tabIndex, 0);
  assert.ok(document.querySelector(".theme-picker").classList.contains("is-previewing"));
  assert.match(document.getElementById("themePickerNow").textContent, /Previewing Daylight/);
  assert.equal(
    document.getElementById("meta-theme-color").getAttribute("content"),
    "#f6f5f2",
    "browser chrome follows the preview"
  );
  assert.equal(localStorage.getItem("thr3shr-theme"), null, "a preview is not saved");

  await press("End");
  assert.equal(document.activeElement.dataset.slug, "onyx");
  await press("Home");
  assert.equal(document.activeElement.dataset.slug, "auto");
  await press("ArrowLeft");
  assert.equal(document.activeElement.dataset.slug, "auto", "stops at the first palette");
  await press("ArrowRight");
  await press("ArrowRight");

  await press("Escape");
  await settle(250);
  assertClosed("closed");
  assert.equal(html().dataset.theme, "graphite", "saved palette restored");
  assert.equal(localStorage.getItem("thr3shr-theme"), null);
  assertSame(document.activeElement, topButton(), "focus returns to the button");
  assert.equal(topButton().getAttribute("aria-expanded"), "false");
  assert.ok(!document.body.classList.contains("theme-picker-open"));
});

test("Enter applies the highlighted palette and saves it", async () => {
  await mountPicker();
  await openFromButton();
  await press("End"); // Onyx
  await press("Enter");
  assert.equal(checked().dataset.slug, "onyx", "checked straight away");
  assert.ok(checked().classList.contains("is-pressed"));
  await settle(700); // the check lands, then the picker closes
  assertClosed();
  assert.equal(html().dataset.theme, "onyx");
  assert.equal(localStorage.getItem("thr3shr-theme"), "onyx");
  assert.match(topButton().textContent, /Onyx/);
  assertSame(document.activeElement, topButton(), "focus returns to the button");

  // The pick survives a remount, as a reload would.
  await act(async () => reactRoot.unmount());
  await mountPicker();
  await openFromButton();
  assert.equal(checked().dataset.slug, "onyx");
});

test("a click applies a palette; a click outside or the close button cancels", async () => {
  await mountPicker();
  await openFromButton();
  const mahogany = radios().find((el) => el.dataset.slug === "mahogany");
  await act(async () => mahogany.click());
  await settle(700);
  assert.equal(localStorage.getItem("thr3shr-theme"), "mahogany");
  assert.equal(html().dataset.theme, "mahogany");

  await openFromButton();
  await press("Home");
  assert.equal(html().dataset.theme, "graphite");
  const layer = document.querySelector(".theme-picker");
  await act(async () => {
    layer.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
  });
  await settle(250);
  assertClosed("a click on the scrim closes");
  assert.equal(html().dataset.theme, "mahogany", "and restores the saved palette");

  await openFromButton();
  await press("Home");
  await act(async () => document.querySelector(".theme-picker-close").click());
  await settle(250);
  assertClosed();
  assert.equal(html().dataset.theme, "mahogany");
  assert.equal(localStorage.getItem("thr3shr-theme"), "mahogany");
});

test("Tab and Shift+Tab stay inside the picker", async () => {
  await mountPicker();
  await openFromButton();
  const stops = [
    document.querySelector(".theme-picker-close"),
    checked(),
    ...document.querySelectorAll(".theme-picker-foot input"),
  ];
  assert.equal(stops.length, 4);
  assertSame(document.activeElement, stops[1], "focus starts on the palette");
  await press("Tab");
  assertSame(document.activeElement, stops[2], "palette to the skin switch");
  await press("Tab");
  assertSame(document.activeElement, stops[3], "skin to the motion switch");
  await press("Tab");
  assertSame(document.activeElement, stops[0], "wraps to the close button");
  await press("Tab", { shiftKey: true });
  assertSame(document.activeElement, stops[3], "Shift+Tab wraps back");
  await press("Escape");
  await settle(250);
  assertClosed("Esc works from any stop");
});

test("the skin and reduced-motion switches apply at once and persist", async () => {
  await mountPicker();
  await openFromButton();
  const [skin, motion] = document.querySelectorAll(".theme-picker-foot input");
  await act(async () => skin.click());
  assert.equal(html().dataset.skin, "yorha");
  assert.equal(localStorage.getItem("thr3shr-skin"), "yorha");
  await act(async () => motion.click());
  assert.equal(html().dataset.motion, "reduce");
  assert.equal(localStorage.getItem("thr3shr-motion"), "reduce");
  await act(async () => motion.click());
  assert.equal(html().dataset.motion, undefined);
  assert.equal(localStorage.getItem("thr3shr-motion"), null);
});

test("palette swaps cross-fade, except under either reduced-motion setting", async () => {
  await mountPicker();
  await openFromButton();
  await press("ArrowRight");
  await press("ArrowRight");
  assert.equal(viewTransitions, 1, "a preview is a View Transition cross-fade");
  await press("Escape");
  await settle(250);
  assert.equal(viewTransitions, 2, "so is the revert");

  // The in-app switch: swaps are instant and the picker leaves at once.
  localStorage.setItem("thr3shr-motion", "reduce");
  await act(async () => reactRoot.unmount());
  await mountPicker();
  assert.equal(html().dataset.motion, "reduce", "saved switch applies on load");
  await openFromButton();
  await press("End");
  assert.equal(html().dataset.theme, "onyx");
  await press("Escape");
  await settle(0);
  assertClosed("no exit wait");
  assert.equal(html().dataset.theme, "graphite");
  assert.equal(viewTransitions, 2, "no cross-fades");

  // The OS setting alone does the same.
  localStorage.clear();
  media.reduce = true;
  await act(async () => reactRoot.unmount());
  delete html().dataset.motion;
  await mountPicker();
  await openFromButton();
  await press("End");
  await press("Enter");
  await settle(0);
  assertClosed("a pick closes without the settle delay");
  assert.equal(localStorage.getItem("thr3shr-theme"), "onyx");
  assert.equal(viewTransitions, 2);
});

test("Auto follows the system and the palette stays put without View Transitions", async () => {
  delete document.startViewTransition;
  media.dark = false;
  await mountPicker();
  assert.equal(html().dataset.theme, "daylight", "Auto on a light system");
  await openFromButton();
  await press("End");
  assert.equal(html().dataset.theme, "onyx", "swaps directly");
  await press("Escape");
  await settle(250);
  assert.equal(html().dataset.theme, "daylight");
});

test("Ctrl+K hands off to the picker through the Appearance entry", async () => {
  const { AppContext } = await load("/src/ui.jsx");
  const { AppearanceProvider } = await load("/src/theme/Appearance.jsx");
  const AppShell = (await load("/src/shell/AppShell.jsx")).default;
  const { staticJumpEntries } = await load("/src/shell/nav.js");
  const jumps = [];
  reactRoot = createRoot(container);
  await act(async () => {
    reactRoot.render(
      React.createElement(
        AppContext.Provider,
        { value: {} },
        React.createElement(
          AppearanceProvider,
          null,
          React.createElement(AppShell, {
            route: "run",
            badges: {},
            engine: null,
            version: "0.0.0",
            jumpEntries: staticJumpEntries(),
            onJump: (entry) => jumps.push(entry),
            banners: null,
            dock: null,
          })
        )
      )
    );
  });
  await settle();

  await act(async () => {
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true, bubbles: true }));
  });
  const input = document.querySelector(".jump-input");
  assert.ok(input, "jump bar open");
  const search = async (text) => {
    await act(async () => {
      const setValue = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
      setValue.call(input, text);
      input.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
    return document.querySelector('.jump-option[aria-selected="true"]').textContent;
  };
  // The Theme setting is the picker's button, so its entry opens it too.
  assert.match(await search("theme"), /^Theme/);
  assert.match(await search("appearance"), /^Appearance.*Change theme/);
  await press("Enter");
  await settle();
  assert.ok(document.querySelector(".jump-input") === null, "jump bar gone");
  assert.ok(dialog(), "picker open");
  assert.deepEqual(jumps, [], "no page navigation");

  // Ctrl+K does not stack the jump bar over the open picker.
  await act(async () => {
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true, bubbles: true }));
  });
  assert.ok(document.querySelector(".jump-input") === null, "no jump bar over the picker");
  await press("Escape");
  await settle(250);
  assertClosed();
  assertSame(document.activeElement, topButton(), "focus lands on the Appearance button");
});
