import React, { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  AUTO_THEME,
  MOTION_STORAGE_KEY,
  REDUCE_MOTION,
  SKIN_STORAGE_KEY,
  THEME_STORAGE_KEY,
  isKnownTheme,
  themeMenu,
} from "./themes";
import { applyMotion, applyPalette, readStored, writeStored } from "./appearance";
import Icon from "../shell/icons";
import ThemePicker from "./ThemePicker";

const AppearanceContext = createContext(null);

export function useAppearance() {
  return useContext(AppearanceContext);
}

// Palette, YoRHa skin and reduced-motion choices, shared by the top-bar
// button, the Settings section and the theme picker modal it opens.  Every
// choice persists to localStorage.
export function AppearanceProvider({ children }) {
  const [theme, setThemeState] = useState(() => {
    const stored = readStored(THEME_STORAGE_KEY);
    return isKnownTheme(stored) ? stored : AUTO_THEME;
  });
  const [skin, setSkinState] = useState(() => document.documentElement.dataset.skin === "yorha");
  const [reduceMotion, setReduceMotionState] = useState(
    () => readStored(MOTION_STORAGE_KEY) === REDUCE_MOTION
  );
  const [pickerOpen, setPickerOpen] = useState(false);
  const openerRef = useRef(null);
  const buttonRef = useRef(null);
  const menu = useMemo(() => themeMenu(), []);

  useEffect(() => {
    applyPalette(theme);
    if (theme !== AUTO_THEME || !window.matchMedia) return undefined;
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => applyPalette(AUTO_THEME);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [theme]);

  useEffect(() => applyMotion(reduceMotion ? REDUCE_MOTION : null), [reduceMotion]);

  // Later picks cross-fade; the first paint used the saved palette directly.
  useEffect(() => {
    const frame = requestAnimationFrame(() =>
      document.documentElement.classList.add("theme-ready")
    );
    return () => cancelAnimationFrame(frame);
  }, []);

  const value = useMemo(() => {
    const rows = menu.flatMap((group) => group.themes);
    return {
      menu,
      theme,
      current: rows.find((row) => row.slug === theme) || rows[0],
      skin,
      reduceMotion,
      pickerOpen,
      buttonRef,
      setTheme(next) {
        setThemeState(next);
        writeStored(THEME_STORAGE_KEY, next);
      },
      setSkin(enabled) {
        setSkinState(enabled);
        if (enabled) document.documentElement.dataset.skin = "yorha";
        else delete document.documentElement.dataset.skin;
        writeStored(SKIN_STORAGE_KEY, enabled ? "yorha" : null);
      },
      setReduceMotion(enabled) {
        setReduceMotionState(enabled);
        writeStored(MOTION_STORAGE_KEY, enabled ? REDUCE_MOTION : null);
      },
      // `from` is the control to hand focus back to on close; without one
      // (the jump bar has gone by then) it is the top-bar button.
      openPicker(from) {
        openerRef.current = from || buttonRef.current;
        setPickerOpen(true);
      },
      closePicker() {
        setPickerOpen(false);
        const opener = openerRef.current?.isConnected ? openerRef.current : buttonRef.current;
        opener?.focus?.({ preventScroll: true });
      },
      openerRef,
    };
  }, [menu, theme, skin, reduceMotion, pickerOpen]);

  return (
    <AppearanceContext.Provider value={value}>
      {children}
      <ThemePicker appearance={value} />
    </AppearanceContext.Provider>
  );
}

// Top-bar Appearance button: the current palette, opening the picker.
export function AppearanceButton() {
  const appearance = useAppearance();
  return (
    <button
      type="button"
      className="theme-picker-toggle"
      ref={appearance.buttonRef}
      aria-haspopup="dialog"
      aria-expanded={appearance.pickerOpen}
      title="Appearance"
      onClick={(ev) => appearance.openPicker(ev.currentTarget)}
    >
      <Icon name="appearance" size={14} />
      <span className="theme-picker-toggle-label">{appearance.current.name}</span>
      <span className="sr-only">: change appearance</span>
    </button>
  );
}
