import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import Icon from "../shell/icons";
import { AUTO_THEME } from "./themes";
import { applyPalette, motionReduced } from "./appearance";

// Hover must rest this long before it previews, so sweeping the pointer across
// the gallery does not flash every palette.
const HOVER_INTENT_MS = 110;
// Matches --t-close in styles.css: the picker unmounts after its exit.
const CLOSE_MS = 180;
// The check badge lands before the picker closes on a pick.
const PICK_SETTLE_MS = 380;

// Next card for an arrow key.  Left and right walk the list; up and down take
// the nearest card in the row above or below, across group boundaries.
// `rects` are the cards' bounding boxes, in list order.
export function nextCardIndex(rects, index, key) {
  if (key === "ArrowRight") return Math.min(index + 1, rects.length - 1);
  if (key === "ArrowLeft") return Math.max(index - 1, 0);
  if (key === "Home") return 0;
  if (key === "End") return rects.length - 1;
  if (key !== "ArrowUp" && key !== "ArrowDown") return index;
  const from = rects[index];
  const centre = from.left + from.width / 2;
  let best = index;
  let bestScore = Infinity;
  rects.forEach((rect, at) => {
    const gap = key === "ArrowDown" ? rect.top - from.bottom : from.top - rect.bottom;
    if (at === index || gap < -1) return;
    const score = gap * 4 + Math.abs(rect.left + rect.width / 2 - centre);
    if (score < bestScore) {
      bestScore = score;
      best = at;
    }
  });
  return best;
}

// A palette drawn as a miniature of the app shell in its own tokens:
// data-theme on the mini scopes that palette to it.  Auto splits the two
// palettes it switches between.
export function ThemeMini({ row }) {
  if (row.slug === AUTO_THEME) {
    return (
      <span className="theme-mini-auto" aria-hidden="true">
        <MiniShell slug="graphite" />
        <MiniShell slug="daylight" />
      </span>
    );
  }
  return <MiniShell slug={row.slug} square={row.square} />;
}

function MiniShell({ slug, square }) {
  return (
    <span
      className="theme-mini"
      data-theme={slug}
      data-square={square ? "" : undefined}
      aria-hidden="true"
    >
      <span className="theme-mini-side">
        <i />
        <i className="is-on" />
        <i />
        <i />
        <i />
      </span>
      <span className="theme-mini-main">
        <span className="theme-mini-top">
          <i />
          <b />
        </span>
        <span className="theme-mini-card">
          <i />
          <i />
          <span className="theme-mini-row">
            <span className="theme-mini-btn" />
            <span className="theme-mini-pill" />
          </span>
        </span>
      </span>
    </span>
  );
}

// Theme picker modal: a gallery of every palette as a miniature of the app.
// Arrowing or resting the pointer on a palette previews it across the whole
// app; Enter, Space or a click applies and saves it; Esc, the close button or
// a click outside puts the saved palette back.
export default function ThemePicker({ appearance }) {
  const { menu, theme, pickerOpen } = appearance;
  const rows = menu.flatMap((group) => group.themes);
  const [mounted, setMounted] = useState(pickerOpen);
  const [shown, setShown] = useState(false);
  const [active, setActive] = useState(0);
  const [previewing, setPreviewing] = useState(null);
  const [pressed, setPressed] = useState(null);
  const panelRef = useRef(null);
  const cardRefs = useRef([]);
  const hoverTimer = useRef(0);
  const settleTimer = useRef(0);

  // Mount, then show on the next frames so the closed state paints first and
  // the entrance transitions run; hide, then unmount after the exit.
  useEffect(() => {
    if (pickerOpen) {
      setMounted(true);
      setPreviewing(null);
      setPressed(null);
      setActive(Math.max(0, rows.findIndex((row) => row.slug === theme)));
      let second = 0;
      const first = requestAnimationFrame(() => {
        second = requestAnimationFrame(() => setShown(true));
      });
      return () => {
        cancelAnimationFrame(first);
        cancelAnimationFrame(second);
      };
    }
    setShown(false);
    clearTimeout(hoverTimer.current);
    clearTimeout(settleTimer.current);
    if (motionReduced()) {
      setMounted(false);
      return undefined;
    }
    const timer = setTimeout(() => setMounted(false), CLOSE_MS);
    return () => clearTimeout(timer);
  }, [pickerOpen]);

  useEffect(() => {
    if (!mounted) return undefined;
    document.body.classList.add("theme-picker-open");
    return () => document.body.classList.remove("theme-picker-open");
  }, [mounted]);

  // Grow out of the control that opened the picker.
  useLayoutEffect(() => {
    if (!pickerOpen || !mounted) return;
    const panel = panelRef.current;
    const from = appearance.openerRef.current?.getBoundingClientRect?.();
    if (!panel || !from || !from.width) return;
    const box = panel.getBoundingClientRect();
    panel.style.setProperty(
      "--picker-origin",
      `${from.left + from.width / 2 - box.left}px ${from.top + from.height / 2 - box.top}px`
    );
  }, [pickerOpen, mounted]);

  // Focus the saved palette once the picker is in the page.
  useEffect(() => {
    if (pickerOpen && mounted) cardRefs.current[active]?.focus({ preventScroll: true });
  }, [pickerOpen, mounted]);

  useEffect(() => () => {
    clearTimeout(hoverTimer.current);
    clearTimeout(settleTimer.current);
  }, []);

  if (!mounted) return null;

  const closing = !pickerOpen;
  const shownRow = rows.find((row) => row.slug === previewing) || appearance.current;

  function preview(index, { focus = true } = {}) {
    const row = rows[index];
    setActive(index);
    const card = cardRefs.current[index];
    if (focus) card?.focus({ preventScroll: true });
    card?.scrollIntoView?.({ block: "nearest", behavior: motionReduced() ? "auto" : "smooth" });
    applyPalette(row.slug);
    setPreviewing(row.slug === theme ? null : row.slug);
  }

  function cancel() {
    if (closing) return;
    applyPalette(theme);
    appearance.closePicker();
  }

  function pick(index) {
    if (closing || pressed) return;
    clearTimeout(hoverTimer.current);
    const row = rows[index];
    setActive(index);
    setPreviewing(null);
    setPressed(row.slug);
    appearance.setTheme(row.slug);
    settleTimer.current = setTimeout(appearance.closePicker, motionReduced() ? 0 : PICK_SETTLE_MS);
  }

  function onKeyDown(ev) {
    if (ev.key === "Escape") {
      ev.preventDefault();
      ev.stopPropagation();
      cancel();
      return;
    }
    if (ev.key === "Tab") {
      // Focus stays in the picker: close, the active palette, the switches.
      const stops = [...panelRef.current.querySelectorAll("button, input")].filter(
        (el) => el.tabIndex >= 0 && !el.disabled
      );
      if (!stops.length) return;
      const at = stops.indexOf(document.activeElement);
      const next = ev.shiftKey
        ? at <= 0 ? stops.length - 1 : at - 1
        : at === -1 || at === stops.length - 1 ? 0 : at + 1;
      ev.preventDefault();
      stops[next].focus();
      return;
    }
    if (ev.target.getAttribute("role") !== "radio") return;
    if (ev.key === "Enter") {
      ev.preventDefault();
      pick(active);
      return;
    }
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(ev.key)) {
      ev.preventDefault();
      const rects = cardRefs.current.map((card) => card.getBoundingClientRect());
      const next = nextCardIndex(rects, active, ev.key);
      if (next !== active) preview(next);
    }
  }

  let index = 0;
  return (
    <div
      className={[
        "theme-picker",
        shown && !closing ? "is-open" : "",
        previewing ? "is-previewing" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      onMouseDown={(ev) => {
        if (ev.target === ev.currentTarget) cancel();
      }}
    >
      <div
        className="theme-picker-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="themePickerTitle"
        aria-describedby="themePickerNow"
        tabIndex={-1}
        ref={panelRef}
        onKeyDown={onKeyDown}
      >
        <header className="theme-picker-head">
          <div className="theme-picker-head-text">
            <h2 className="theme-picker-title" id="themePickerTitle">
              Appearance
            </h2>
            <p className="theme-picker-now" id="themePickerNow">
              {previewing ? "Previewing " : "Current: "}
              <b>{shownRow.name}</b> · {shownRow.sub}
            </p>
          </div>
          <button type="button" className="theme-picker-close" aria-label="Close" onClick={cancel}>
            <Icon name="close" />
          </button>
        </header>
        <div className="theme-picker-body" role="radiogroup" aria-label="Theme">
          {menu.map((group) => (
            <section
              className="theme-picker-group"
              role="group"
              aria-labelledby={`themeGroup-${group.id}`}
              key={group.id}
            >
              <p className="theme-picker-group-label" id={`themeGroup-${group.id}`}>
                {group.label}
              </p>
              <div className="theme-picker-grid">
                {group.themes.map((row) => {
                  const at = index++;
                  return (
                    <button
                      key={row.slug}
                      type="button"
                      role="radio"
                      ref={(el) => {
                        cardRefs.current[at] = el;
                      }}
                      className={[
                        "theme-card",
                        at === active ? "is-active" : "",
                        pressed === row.slug ? "is-pressed" : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
                      style={{ "--i": at }}
                      tabIndex={at === active ? 0 : -1}
                      aria-checked={row.slug === theme}
                      aria-label={`${row.name}, ${row.sub}`}
                      data-slug={row.slug}
                      onClick={() => pick(at)}
                      onMouseEnter={() => {
                        clearTimeout(hoverTimer.current);
                        hoverTimer.current = setTimeout(() => {
                          if (!pressed) preview(at, { focus: false });
                        }, HOVER_INTENT_MS);
                      }}
                      onMouseLeave={() => clearTimeout(hoverTimer.current)}
                    >
                      <ThemeMini row={row} />
                      <span className="theme-card-text">
                        <span className="theme-card-name">{row.name}</span>
                        <span className="theme-card-sub">{row.sub}</span>
                      </span>
                      <span className="theme-card-check" aria-hidden="true">
                        <Icon name="check" size={13} strokeWidth={2.2} />
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>
          ))}
        </div>
        <footer className="theme-picker-foot">
          <label className="theme-picker-switch">
            <input
              type="checkbox"
              className="switch"
              checked={appearance.skin}
              onChange={(e) => appearance.setSkin(e.target.checked)}
            />
            YoRHa skin
          </label>
          <label className="theme-picker-switch">
            <input
              type="checkbox"
              className="switch"
              checked={appearance.reduceMotion}
              onChange={(e) => appearance.setReduceMotion(e.target.checked)}
            />
            Reduce motion
          </label>
          <span className="theme-picker-keys" aria-hidden="true">
            <span>
              <kbd className="app-kbd">← ↑ ↓ →</kbd> preview
            </span>
            <span>
              <kbd className="app-kbd">Enter</kbd> apply
            </span>
            <span>
              <kbd className="app-kbd">Esc</kbd> cancel
            </span>
          </span>
        </footer>
      </div>
    </div>
  );
}
