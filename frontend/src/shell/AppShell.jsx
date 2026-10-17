import React, { useEffect, useRef, useState } from "react";
import AppearancePicker from "../theme/AppearancePicker";
import { NAV_COLLAPSED_STORAGE_KEY } from "../theme/themes";
import Icon from "./icons";
import JumpBar from "./JumpBar";
import { PAGES, PAGE_GROUPS, pageById, pageHref } from "./nav";

// Phone tab bar: the four workflow pages, then "More" for the rest.
const TAB_PAGES = ["run", "review", "runs", "settings"];

function NavBadge({ badge }) {
  if (!badge) return null;
  return (
    <span className={`app-nav-count${badge.tone ? ` is-${badge.tone}` : ""}`} title={badge.title}>
      {badge.text}
    </span>
  );
}

// Console app shell: sidebar, top bar with the jump bar and Appearance menu,
// banner slot, run dock, and on phones a bottom tab bar with a "More" sheet.
export default function AppShell({
  route,
  badges,
  engine,
  version,
  jumpEntries,
  onJump,
  banners,
  dock,
  children,
}) {
  const [collapsed, setCollapsed] = useState(() =>
    document.documentElement.classList.contains("nav-collapsed")
  );
  const [jumpOpen, setJumpOpen] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  const sheetRef = useRef(null);
  const moreRef = useRef(null);
  const page = pageById(route);

  useEffect(() => {
    document.documentElement.classList.toggle("nav-collapsed", collapsed);
  }, [collapsed]);

  useEffect(() => setSheetOpen(false), [route]);

  // The sheet only exists at phone width; growing the window closes it.
  useEffect(() => {
    if (!sheetOpen) return undefined;
    const mql = window.matchMedia("(max-width: 720px)");
    const onChange = () => {
      if (!mql.matches) setSheetOpen(false);
    };
    const onKey = (ev) => {
      if (ev.key === "Escape") {
        ev.preventDefault();
        setSheetOpen(false);
        moreRef.current?.focus();
      }
    };
    const onClick = (ev) => {
      if (sheetRef.current?.contains(ev.target) || moreRef.current?.contains(ev.target)) return;
      setSheetOpen(false);
    };
    mql.addEventListener("change", onChange);
    document.addEventListener("keydown", onKey);
    document.addEventListener("click", onClick);
    document.body.classList.add("mobile-nav-open");
    sheetRef.current?.querySelector("a")?.focus({ preventScroll: true });
    return () => {
      mql.removeEventListener("change", onChange);
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("click", onClick);
      document.body.classList.remove("mobile-nav-open");
    };
  }, [sheetOpen]);

  function toggleCollapsed() {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(NAV_COLLAPSED_STORAGE_KEY, next ? "1" : "0");
      } catch {
        /* ignore */
      }
      return next;
    });
  }

  const collapseLabel = collapsed ? "Expand sidebar" : "Collapse sidebar";

  return (
    <>
      <a className="skip-link" href="#main-content" onClick={(ev) => {
        ev.preventDefault();
        document.getElementById("main-content")?.focus();
      }}>
        Skip to content
      </a>
      <div className="app">
        <aside className="app-sidebar" aria-label="Sidebar">
          <div className="app-sidebar-head">
            <a href="#/" className="app-brand" aria-label="THR3SHR home">
              <span className="app-brand-mark" aria-hidden="true" />
              <span className="app-brand-name">THR3SHR</span>
            </a>
            <button
              type="button"
              className="app-collapse"
              aria-pressed={collapsed}
              aria-label={collapseLabel}
              title={collapseLabel}
              onClick={toggleCollapsed}
            >
              <Icon name="chevron" />
            </button>
          </div>
          <nav className="app-nav" aria-label="Primary">
            {PAGE_GROUPS.map((group) => (
              <div className="app-nav-group" key={group}>
                <p className="app-nav-heading">
                  <span>{group}</span>
                </p>
                {PAGES.filter((item) => item.group === group).map((item) => (
                  <a
                    key={item.id}
                    href={pageHref(item.id)}
                    className="app-nav-link"
                    title={item.label}
                    aria-current={route === item.id ? "page" : undefined}
                  >
                    <span className="app-nav-icon" aria-hidden="true">
                      <Icon name={item.icon} />
                    </span>
                    <span className="app-nav-label">{item.label}</span>
                    <NavBadge badge={badges[item.id]} />
                  </a>
                ))}
              </div>
            ))}
          </nav>
          <div className="app-sidebar-foot">
            {engine ? (
              <p className={`engine-chip is-${engine.tone}`} title={engine.title}>
                <span className="engine-chip-dot" aria-hidden="true" />
                <span className="engine-chip-label">{engine.label}</span>
              </p>
            ) : null}
            <p className="app-version">v{version}</p>
          </div>
        </aside>

        <div className="app-main">
          <header className="app-topbar">
            <a href="#/" className="app-topbar-brand" aria-label="THR3SHR home">
              <span className="app-brand-mark" aria-hidden="true" />
            </a>
            <p className="app-crumb">
              <span className="app-crumb-section">{page.group}</span>
              <span className="app-crumb-sep" aria-hidden="true">
                /
              </span>
              <span className="app-crumb-page">{page.label}</span>
            </p>
            <button
              type="button"
              className="app-jump-trigger"
              aria-haspopup="dialog"
              aria-keyshortcuts="Control+K"
              onClick={() => setJumpOpen(true)}
            >
              <Icon name="search" size={15} />
              <span className="app-jump-trigger-text">Jump to a page, setting or run…</span>
              <kbd className="app-kbd">Ctrl K</kbd>
            </button>
            <AppearancePicker />
          </header>

          {banners ? <div className="app-banner-slot">{banners}</div> : null}

          <main id="main-content" className="layout-shell" tabIndex={-1}>
            {children}
          </main>

          <footer className="app-footer" aria-label="Attribution">
            <p>
              <span className="app-footer-name">
                THR3SHR <span className="app-footer-version">v{version}</span>
              </span>
              <span>
                © 2026 Dinamush · code MIT · creative{" "}
                <a
                  href="https://creativecommons.org/licenses/by/4.0/"
                  target="_blank"
                  rel="noreferrer"
                >
                  CC BY 4.0
                </a>
                {" · "}
                <a
                  href="https://github.com/Dinamush/thr3shr/blob/main/ATTRIBUTION.md"
                  target="_blank"
                  rel="noreferrer"
                >
                  model attributions
                </a>
              </span>
              <span>Local-first: images and runs stay on this machine.</span>
            </p>
          </footer>
        </div>
      </div>

      {dock}

      <nav className="app-tabbar" aria-label="Sections">
        {TAB_PAGES.map((id) => {
          const item = pageById(id);
          return (
            <a
              key={id}
              href={pageHref(id)}
              className="app-tab"
              aria-current={route === id ? "page" : undefined}
            >
              <Icon name={item.icon} size={20} strokeWidth={1.5} />
              <span>{id === "run" ? "Run" : item.label}</span>
              <NavBadge badge={badges[id]} />
            </a>
          );
        })}
        <button
          type="button"
          className="nav-hamburger"
          ref={moreRef}
          aria-expanded={sheetOpen}
          aria-controls="mobileNavPanel"
          aria-label="More pages"
          onClick={() => setSheetOpen((prev) => !prev)}
        >
          <span className="nav-hamburger-icon" aria-hidden="true">
            <span />
          </span>
          <span className="nav-hamburger-text" aria-hidden="true">
            More
          </span>
        </button>
      </nav>

      <div
        className="mobile-nav-panel"
        id="mobileNavPanel"
        ref={sheetRef}
        data-open={sheetOpen ? "true" : "false"}
        hidden={!sheetOpen}
      >
        <nav className="mobile-nav-panel-nav" aria-label="All pages">
          {PAGE_GROUPS.map((group) => (
            <React.Fragment key={group}>
              <p className="mobile-nav-panel-section">{group}</p>
              {PAGES.filter((item) => item.group === group).map((item) => (
                <a
                  key={item.id}
                  href={pageHref(item.id)}
                  className="mobile-nav-panel-link"
                  aria-current={route === item.id ? "page" : undefined}
                >
                  <Icon name={item.icon} size={18} />
                  <span>{item.label}</span>
                </a>
              ))}
            </React.Fragment>
          ))}
          <button
            type="button"
            className="mobile-nav-panel-link mobile-nav-panel-jump"
            onClick={() => {
              setSheetOpen(false);
              setJumpOpen(true);
            }}
          >
            <Icon name="search" size={18} />
            <span>Jump to a setting or run…</span>
          </button>
        </nav>
      </div>

      <JumpBar open={jumpOpen} onOpenChange={setJumpOpen} entries={jumpEntries} onGo={onJump} />
    </>
  );
}
