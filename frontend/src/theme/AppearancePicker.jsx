import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  AUTO_THEME,
  SKIN_STORAGE_KEY,
  THEME_STORAGE_KEY,
  findTheme,
  isKnownTheme,
  resolveTheme,
  themeMenu,
} from "./themes";

function readStored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStored(key, value) {
  try {
    if (value == null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* storage blocked: the choice lasts for this page only */
  }
}

function prefersDark() {
  return Boolean(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
}

// Applies a palette to <html> and the OS chrome.  index.html runs the same
// resolution before first paint so a reload never flashes the default.
function applyPalette(value) {
  const root = document.documentElement;
  const resolved = resolveTheme(value, prefersDark());
  root.dataset.theme = resolved;
  const theme = findTheme(resolved);
  document.getElementById("meta-theme-color")?.setAttribute("content", theme.tokens.bg);
  document.getElementById("meta-color-scheme")?.setAttribute("content", theme.scheme);
}

// Palette picker plus the opt-in YoRHa skin.  Persists to localStorage.
export default function AppearancePicker() {
  const [theme, setTheme] = useState(() => {
    const stored = readStored(THEME_STORAGE_KEY);
    return isKnownTheme(stored) ? stored : AUTO_THEME;
  });
  const [skin, setSkin] = useState(() => document.documentElement.dataset.skin === "yorha");
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const toggleRef = useRef(null);
  const menu = useMemo(() => themeMenu(), []);

  useEffect(() => {
    applyPalette(theme);
    if (theme !== AUTO_THEME || !window.matchMedia) return undefined;
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => applyPalette(AUTO_THEME);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [theme]);

  // Later picks crossfade; the first paint used the saved palette directly.
  useEffect(() => {
    const frame = requestAnimationFrame(() =>
      document.documentElement.classList.add("theme-ready")
    );
    return () => cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    const onClick = (ev) => {
      if (rootRef.current && !rootRef.current.contains(ev.target)) setOpen(false);
    };
    const onKey = (ev) => {
      if (ev.key === "Escape") {
        setOpen(false);
        toggleRef.current?.focus();
      }
    };
    document.addEventListener("click", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  function pick(value) {
    setTheme(value);
    writeStored(THEME_STORAGE_KEY, value);
    setOpen(false);
  }

  function toggleSkin(enabled) {
    setSkin(enabled);
    if (enabled) document.documentElement.dataset.skin = "yorha";
    else delete document.documentElement.dataset.skin;
    writeStored(SKIN_STORAGE_KEY, enabled ? "yorha" : null);
  }

  const currentName =
    menu.flatMap((group) => group.themes).find((row) => row.slug === theme)?.name || "Auto";

  return (
    <div className="theme-picker" data-open={open ? "true" : "false"} ref={rootRef}>
      <button
        type="button"
        className="theme-picker-toggle"
        ref={toggleRef}
        aria-haspopup="true"
        aria-expanded={open}
        aria-controls="themePickerPanel"
        title="Appearance"
        onClick={(ev) => {
          ev.stopPropagation();
          setOpen((prev) => !prev);
        }}
      >
        <span className="theme-picker-toggle-dot" aria-hidden="true" />
        <span className="theme-picker-toggle-label">{currentName}</span>
        <span className="sr-only">appearance</span>
      </button>
      <div className="theme-picker-panel" id="themePickerPanel" role="dialog" aria-label="Appearance">
        {menu.map((group) => (
          <div className="theme-picker-panel-group" key={group.id}>
            <p className="theme-picker-panel-group-label">{group.label}</p>
            <ul className="theme-swatches" aria-label={group.label}>
              {group.themes.map((row) => (
                <li key={row.slug}>
                  <button
                    type="button"
                    className="theme-swatch"
                    aria-pressed={row.slug === theme}
                    onClick={() => pick(row.slug)}
                  >
                    <span className="theme-swatch-chips" aria-hidden="true">
                      {row.chips.map((chip, idx) => (
                        <span key={idx} style={{ background: chip }} />
                      ))}
                    </span>
                    <span className="theme-swatch-text">
                      <span className="theme-swatch-name">{row.name}</span>
                      <span className="theme-swatch-sub">{row.sub}</span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
        <div className="theme-picker-panel-group theme-picker-skin">
          <label className="switch-row" htmlFor="skinToggle">
            <span className="switch-row-text">
              <span className="switch-row-label">YoRHa skin</span>
              <span className="switch-row-help">
                Boot intro, film grain and scanlines over any palette.
              </span>
            </span>
            <input
              type="checkbox"
              className="switch"
              id="skinToggle"
              checked={skin}
              onChange={(e) => toggleSkin(e.target.checked)}
            />
          </label>
        </div>
      </div>
    </div>
  );
}
