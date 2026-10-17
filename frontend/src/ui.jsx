import React, { createContext, useContext } from "react";

// App state and actions, provided by App.jsx to every page.
export const AppContext = createContext(null);

export function useApp() {
  return useContext(AppContext);
}

const ITEM_TONES = {
  proposed: "warn",
  reviewed: "info",
  approved: "ok",
  rejected: "bad",
  migrated: "done",
};

const RUN_TONES = {
  pending: "queued",
  running: "running",
  completed: "ok",
  cancelled: "warn",
  failed: "bad",
};

export function StatusChip({ status, kind = "item", children }) {
  const tones = kind === "run" ? RUN_TONES : ITEM_TONES;
  const tone = tones[status] || "neutral";
  return (
    <span className={`status-chip is-${tone} status-${status}`}>{children || status}</span>
  );
}

// Tag scores as labelled bars, highest first as the API sends them.
export function ScoreList({ entries, sensitive, empty = "None", label }) {
  if (!entries || entries.length === 0) return <p className="score-empty">{empty}</p>;
  return (
    <ul className="score-list" aria-label={label}>
      {entries.map((entry) => {
        const score = Number(entry.score) || 0;
        const isSensitive = sensitive?.has(entry.tag);
        return (
          <li
            key={entry.tag}
            className={`score-row${isSensitive ? " is-sensitive" : ""}`}
            title={`${entry.tag} (${score.toFixed(3)})`}
          >
            <span className="score-tag">
              {entry.tag}
              {isSensitive ? (
                <span className="score-flag" title="Sensitive - confirm in Final tag">
                  sensitive
                </span>
              ) : null}
            </span>
            <span className="score-bar" aria-hidden="true">
              <i style={{ width: `${Math.max(0, Math.min(1, score)) * 100}%` }} />
            </span>
            <span className="score-value">{score.toFixed(3)}</span>
          </li>
        );
      })}
    </ul>
  );
}

export function ProgressBar({ pct, label, indeterminate = false }) {
  const value = Math.max(0, Math.min(100, Number(pct) || 0));
  return (
    <div
      className={`progress-bar${indeterminate ? " is-indeterminate" : ""}`}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : Math.round(value)}
    >
      <div className="progress-fill" style={indeterminate ? undefined : { width: `${value}%` }} />
    </div>
  );
}

export function PageHead({ title, lede, children }) {
  return (
    <div className="page-head">
      <div className="page-head-copy">
        <h1>{title}</h1>
        {lede ? <p className="page-lede">{lede}</p> : null}
      </div>
      {children}
    </div>
  );
}

export function PageStats({ stats }) {
  return (
    <ul className="page-stats">
      {stats.map(([label, value, tone]) => (
        <li key={label} className={tone ? `is-${tone}` : undefined}>
          <span className="page-stat-value">{value}</span>
          <span className="page-stat-label">{label}</span>
        </li>
      ))}
    </ul>
  );
}

export function EmptyState({ title, children, action }) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      {children ? <p>{children}</p> : null}
      {action}
    </div>
  );
}

// Switch row: label with its help directly under it, control on the right.
export function SwitchRow({ id, label, help, checked, disabled, onChange }) {
  return (
    <label className={`switch-row${disabled ? " is-disabled" : ""}`} htmlFor={id}>
      <span className="switch-row-text">
        <span className="switch-row-label">{label}</span>
        {help ? <span className="switch-row-help">{help}</span> : null}
      </span>
      <input
        type="checkbox"
        className="switch"
        id={id}
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}
