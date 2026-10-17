import React, { useEffect, useMemo, useRef, useState } from "react";
import Icon from "./icons";
import { rankJumpEntries } from "./nav";

// Ctrl+K jump bar over every page, settings section, setting and run.
export default function JumpBar({ open, onOpenChange, entries, onGo }) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);
  const lastFocusRef = useRef(null);
  const results = useMemo(() => rankJumpEntries(entries, query), [entries, query]);

  useEffect(() => {
    const onKey = (ev) => {
      if ((ev.ctrlKey || ev.metaKey) && !ev.altKey && (ev.key === "k" || ev.key === "K")) {
        ev.preventDefault();
        onOpenChange(!open);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onOpenChange]);

  useEffect(() => {
    if (open) {
      lastFocusRef.current = document.activeElement;
      setQuery("");
      setActive(0);
      document.body.classList.add("jump-open");
      inputRef.current?.focus();
      return () => document.body.classList.remove("jump-open");
    }
    return undefined;
  }, [open]);

  useEffect(() => setActive(0), [query]);

  useEffect(() => {
    listRef.current?.children[active]?.scrollIntoView({ block: "nearest" });
  }, [active, results]);

  function close(restoreFocus) {
    onOpenChange(false);
    if (restoreFocus) lastFocusRef.current?.focus?.();
  }

  function go(entry) {
    close(false);
    onGo(entry);
  }

  function onKeyDown(ev) {
    const count = results.length;
    if (ev.key === "ArrowDown") {
      ev.preventDefault();
      if (count) setActive((prev) => (prev + 1) % count);
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      if (count) setActive((prev) => (prev - 1 + count) % count);
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      if (results[active]) go(results[active]);
    } else if (ev.key === "Escape") {
      ev.preventDefault();
      close(true);
    } else if (ev.key === "Tab") {
      ev.preventDefault();
    }
  }

  if (!open) return null;
  return (
    <div
      className="jump"
      role="dialog"
      aria-modal="true"
      aria-label="Jump to a page, setting or run"
      onMouseDown={(ev) => {
        if (ev.target === ev.currentTarget) close(true);
      }}
    >
      <div className="jump-box">
        <div className="jump-field">
          <Icon name="search" />
          <input
            ref={inputRef}
            type="text"
            className="jump-input"
            autoComplete="off"
            spellCheck={false}
            placeholder="Jump to a page, setting or run…"
            role="combobox"
            aria-expanded="true"
            aria-controls="jumpList"
            aria-autocomplete="list"
            aria-activedescendant={results.length ? `jump-opt-${active}` : undefined}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <kbd className="app-kbd">Esc</kbd>
        </div>
        <ul className="jump-list" id="jumpList" role="listbox" aria-label="Results" ref={listRef}>
          {results.map((entry, idx) => (
            <li
              key={entry.key}
              id={`jump-opt-${idx}`}
              className="jump-option"
              role="option"
              aria-selected={idx === active}
              onMouseMove={() => setActive(idx)}
              onClick={() => go(entry)}
            >
              <span className="jump-option-label">{entry.label}</span>
              <span className="jump-option-group">{entry.group}</span>
            </li>
          ))}
        </ul>
        {results.length === 0 ? (
          <p className="jump-empty">No page, setting or run matches.</p>
        ) : null}
      </div>
    </div>
  );
}
