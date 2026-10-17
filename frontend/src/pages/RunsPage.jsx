import React from "react";
import { ACTIVE_RUN_STATUSES, formatRunDate, pathTail, runPct, runProgressText, taggerLabel } from "../lib";
import Icon from "../shell/icons";
import { EmptyState, PageHead, PageStats, StatusChip, useApp } from "../ui";

export default function RunsPage() {
  const { runs, runId, openRunForReview } = useApp();
  const active = runs.filter((run) => ACTIVE_RUN_STATUSES.includes(run.status)).length;

  return (
    <>
      <PageHead title="Runs" lede="Reopen an earlier run to review or migrate its results.">
        <PageStats
          stats={[
            ["runs", runs.length],
            ["active", active, active ? "running" : null],
          ]}
        />
      </PageHead>
      {runs.length === 0 ? (
        <EmptyState
          title="No runs yet"
          action={
            <p className="empty-state-actions">
              <a className="btn btn-primary" href="#/">
                Start a run
              </a>
            </p>
          }
        >
          Runs you start are listed here, newest first.
        </EmptyState>
      ) : (
        <section className="layout-panel" aria-label="Run history">
          <div className="table-wrap">
            <table className="runs-table">
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Status</th>
                  <th scope="col">Progress</th>
                  <th scope="col">Approved / migrated</th>
                  <th scope="col">Root</th>
                  <th scope="col">Tagger</th>
                  <th scope="col">Created</th>
                  <th scope="col">
                    <span className="sr-only">Open</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const counts = run.item_counts || {};
                  const viewing = run.run_id === runId;
                  return (
                    <tr key={run.run_id} className={viewing ? "is-current" : undefined}>
                      <td className="mono">#{run.run_id}</td>
                      <td>
                        <StatusChip status={run.status} kind="run" />
                      </td>
                      <td>
                        <span className="runs-progress">
                          <span className="runs-progress-bar" aria-hidden="true">
                            <i style={{ width: `${Math.min(100, runPct(run))}%` }} />
                          </span>
                          <span className="mono">{runProgressText(run)}</span>
                        </span>
                      </td>
                      <td className="mono">
                        {counts.approved || 0} / {counts.migrated || 0}
                      </td>
                      <td className="mono runs-root" title={run.root_repo}>
                        {pathTail(run.root_repo)}
                      </td>
                      <td>
                        {taggerLabel(run.tagger_model)}
                        {run.tagging_domain === "real_life" ? (
                          <span className="muted"> · real life</span>
                        ) : null}
                      </td>
                      <td className="mono">{formatRunDate(run.created_at)}</td>
                      <td className="runs-open">
                        <button
                          type="button"
                          className={`btn btn-sm${viewing ? " btn-ghost" : ""}`}
                          aria-label={`Open run #${run.run_id}`}
                          onClick={() => openRunForReview(run.run_id)}
                        >
                          <Icon name="review" size={14} />
                          {viewing ? "Viewing" : "Open"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}
