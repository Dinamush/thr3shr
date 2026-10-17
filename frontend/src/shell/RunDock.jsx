import React from "react";
import Icon, { StatusGlyph } from "./icons";

// The state lamp: a status glyph, or the unplugged icon when offline.
function DockLamp({ state }) {
  if (state === "offline") return <Icon name="offline" size={14} className="run-dock-lamp" />;
  const glyph = { running: "running", migrating: "running", queued: "queued" }[state] || "proposed";
  return <StatusGlyph name={glyph} size={12} className="run-dock-lamp" />;
}

// Live run dock, on every page: what is tagging or migrating right now, how
// far it is, and how much is waiting for review.  App.jsx builds the model
// from the run status it already polls; the dock adds no requests.
export default function RunDock({ model }) {
  const { state, label, job, review } = model;
  return (
    <section className="run-dock" aria-label="Run activity" data-state={state}>
      <a className="run-dock-summary" href={job?.href || "#/"}>
        <DockLamp state={state} />
        <span className="run-dock-label" aria-live="polite">
          {label}
        </span>
      </a>
      {job ? (
        <a className="run-dock-job" href={job.href}>
          <span className="run-dock-name">{job.name}</span>
          <span
            className={`run-dock-bar${job.indeterminate ? " is-indeterminate" : ""}`}
            role="progressbar"
            aria-label={`${job.name} progress`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={job.indeterminate ? undefined : Math.round(job.pct)}
          >
            <i style={{ width: job.indeterminate ? undefined : `${Math.min(100, job.pct)}%` }} />
          </span>
          <span className="run-dock-count">{job.detail}</span>
        </a>
      ) : null}
      {review ? (
        <a className="run-dock-review" href="#/review">
          <StatusGlyph name="needs" />
          <span>
            <span className="run-dock-review-count">{review.count}</span> need review
          </span>
        </a>
      ) : null}
      <a className="run-dock-more" href="#/runs">
        <Icon name="runs" size={14} />
        Runs
      </a>
    </section>
  );
}
