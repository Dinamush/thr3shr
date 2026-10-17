import React from "react";

// Live run dock, on every page: what is tagging or migrating right now, how
// far it is, and how much is waiting for review.  App.jsx builds the model
// from the run status it already polls; the dock adds no requests.
export default function RunDock({ model }) {
  const { state, label, job, review } = model;
  return (
    <section className="run-dock" aria-label="Run activity" data-state={state}>
      <a className="run-dock-summary" href={job?.href || "#/"}>
        <span className="run-dock-dot" aria-hidden="true" />
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
          <span className="run-dock-review-count">{review.count}</span> need review
        </a>
      ) : null}
      <a className="run-dock-more" href="#/runs">
        Runs
      </a>
    </section>
  );
}
