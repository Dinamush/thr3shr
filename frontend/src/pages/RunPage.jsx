import React, { useState } from "react";
import {
  TAGGER_MODELS,
  formatDurationSeconds,
  formatEtaFinishAt,
  formatRunDate,
  pathTail,
  runProgressText,
  taggerLabel,
} from "../lib";
import { PageHead, PageStats, ProgressBar, StatusChip, SwitchRow, useApp } from "../ui";

const RUN_MODES = [
  {
    id: "classify",
    label: (realLife) => (realLife ? "Real-life tagging" : "Tag into destination folders"),
    start: (realLife) => (realLife ? "Start real-life tagging" : "Start run"),
    help: (realLife) =>
      realLife
        ? "Style gate → local adult VLM (+ optional position classifier) → isolated categories under Real Life/. All items start in needs-review. Sensitive tags require typing them into Final tag before approve."
        : "Only the selected destinations compete for folder assignment.",
  },
  {
    id: "real_life_filter",
    drawnOnly: true,
    label: () => "Filter real-life only",
    start: () => "Filter real-life only",
    help: () =>
      "Scan root (including GIF/video). Keep only confident real_life hits — nothing else. Style-first hybrid (skip WD on clear anime), capped GIF/video frames, stills stay batched.",
  },
  {
    id: "doujin_works",
    drawnOnly: true,
    label: () => "Doujin works",
    start: () => "Start Doujin works",
    help: () =>
      "Treat each child folder or top-level .cbz as one work. Sample pages → WD → primary + category tags (loli / shota / milf / fertilization / monster_girl / incest / bestiality / Pokemon / NTR / tentacles / furry / android). Migrate moves into Doujins/<tag>/ with junctions.",
  },
];

function Destinations() {
  const app = useApp();
  const {
    settings,
    sfwClassifyMode,
    runActive,
    opsLoading,
    tagQuery,
    setTagQuery,
    tagOptions,
    selectedTags,
    parkedNsfwTags,
  } = app;
  const realLife = settings.tagging_domain === "real_life";

  async function addFromInput() {
    if (sfwClassifyMode) return;
    app.setError("");
    try {
      await app.addTagsFromInput(tagQuery);
      setTagQuery("");
    } catch (err) {
      app.setError(err.message);
    }
  }

  return (
    <section className="layout-panel run-destinations" aria-labelledby="destinations-title">
      <div className="pane-head">
        <h2 className="pane-title" id="destinations-title">
          Destinations
        </h2>
        <span className="pane-meta">{selectedTags.length} selected</span>
      </div>
      <div className="pane-body">
        <p className="pane-lede">
          {realLife
            ? "Real-life categories only (isolated from anime tags.csv). Changes auto-save after settings save."
            : sfwClassifyMode
              ? "SFW mode: only SFW / scenery compete. Your NSFW tags stay parked."
              : "Only these tags compete for folder assignment. Changes auto-save."}
        </p>
        {!realLife ? (
          <SwitchRow
            id="sfw-classify-mode"
            label="SFW classify mode"
            help="Parks NSFW tags; they come back when it is off."
            checked={sfwClassifyMode}
            disabled={runActive || opsLoading.startingRun}
            onChange={(checked) => app.setSfwClassifyMode(checked)}
          />
        ) : null}
        <div className="tag-search">
          <label htmlFor="tag-match-search">
            {realLife ? "Real-life category search" : "Tag match search (destinations + tags.csv)"}
          </label>
          <div className="tag-search-row">
            <input
              id="tag-match-search"
              value={tagQuery}
              onChange={(e) => setTagQuery(e.target.value)}
              autoComplete="off"
              disabled={sfwClassifyMode}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  addFromInput();
                }
              }}
              placeholder={
                sfwClassifyMode
                  ? "Tag editing disabled in SFW mode"
                  : realLife
                    ? "Type creampie, BBC, hotwife…"
                    : "Type Voyeur, loli, Pokemon…"
              }
            />
            <button
              type="button"
              className="btn"
              disabled={sfwClassifyMode}
              onClick={addFromInput}
            >
              Add tag(s)
            </button>
          </div>
          <p className="help-text">Comma-separate several; Enter adds them.</p>
        </div>
        {tagQuery.trim() && !sfwClassifyMode ? (
          <div className="tag-suggestions" aria-label="Matching tags">
            {tagOptions.length === 0 ? (
              <span className="muted">No matching tags</span>
            ) : (
              tagOptions.slice(0, 30).map((tag) => (
                <button
                  key={tag}
                  type="button"
                  className="tag-suggestion"
                  disabled={selectedTags.includes(tag)}
                  onClick={() => {
                    app.addSelectedTag(tag);
                    setTagQuery("");
                    app.setError("");
                  }}
                >
                  {tag}
                </button>
              ))
            )}
          </div>
        ) : null}
        <div className={`tag-chips${sfwClassifyMode ? " tags-locked" : ""}`}>
          {selectedTags.length === 0 ? (
            <span className="muted">No tags selected</span>
          ) : (
            selectedTags.map((tag) => (
              <span key={tag} className="tag-chip">
                {tag}
                <button
                  type="button"
                  onClick={() => app.removeSelectedTag(tag)}
                  aria-label={`Remove ${tag}`}
                  disabled={sfwClassifyMode}
                >
                  ×
                </button>
              </span>
            ))
          )}
        </div>
        {sfwClassifyMode && parkedNsfwTags.length > 0 ? (
          <div className="parked-tags">
            <p className="help-text">Parked NSFW tags (inactive until SFW mode is off):</p>
            <div className="tag-chips tags-locked">
              {parkedNsfwTags.map((tag) => (
                <span key={`parked-${tag}`} className="tag-chip chip-parked">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}

export function ScanPreview() {
  const { scanPreview } = useApp();
  if (!scanPreview?.stats) return null;
  const s = scanPreview.stats;
  return (
    <div className="scan-preview" role="status">
      <div className="stat-row stat-row-lead">
        <span>Eligible images</span>
        <span className="stat-row-value">{s.eligible_images}</span>
      </div>
      <div className="stat-row">
        <span>Total files</span>
        <span className="stat-row-value">{s.total_files}</span>
      </div>
      <div className="stat-row">
        <span>Ignored GIF</span>
        <span className="stat-row-value">{s.ignored_gif}</span>
      </div>
      <div className="stat-row">
        <span>Unsupported</span>
        <span className="stat-row-value">{s.ignored_unsupported}</span>
      </div>
      <div className={`stat-row${s.failed_to_read ? " is-bad" : ""}`}>
        <span>Unreadable</span>
        <span className="stat-row-value">{s.failed_to_read}</span>
      </div>
      {scanPreview.recursive ? null : <p className="help-text">Top folder only.</p>}
    </div>
  );
}

export function PreviewScanButton() {
  const { scanPreviewLoading, handlePreviewScan } = useApp();
  return (
    <button
      type="button"
      className="btn"
      disabled={scanPreviewLoading}
      onClick={handlePreviewScan}
    >
      {scanPreviewLoading ? "Scanning…" : "Preview scan"}
    </button>
  );
}

function NewRun() {
  const app = useApp();
  const { settings, loading, opsLoading, anyRunActive, otherActiveRun, isDirty } = app;
  const realLife = settings.tagging_domain === "real_life";
  const [modeId, setModeId] = useState("classify");
  // Filter and Doujin runs are drawn-domain only.
  const mode =
    RUN_MODES.find((m) => m.id === modeId && !(m.drawnOnly && realLife)) || RUN_MODES[0];
  const startDisabled = loading || opsLoading.startingRun || anyRunActive;

  return (
    <section className="layout-panel run-new" aria-labelledby="new-run-title">
      <div className="pane-head">
        <h2 className="pane-title" id="new-run-title">
          New run
        </h2>
        <a className="pane-link" href="#/settings">
          Edit settings
        </a>
      </div>
      <div className="pane-body">
        <dl className="run-source">
          <div>
            <dt>Root repository</dt>
            <dd className={settings.root_repo ? "mono-path" : "is-missing"}>
              {settings.root_repo || "Not set"}
            </dd>
          </div>
          <div>
            <dt>Categories root</dt>
            <dd className={settings.categories_root ? "mono-path" : "is-missing"}>
              {settings.categories_root || "Not set"}
            </dd>
          </div>
          <div>
            <dt>Tagger</dt>
            <dd>
              {taggerLabel(settings.tagger_model)} · confidence{" "}
              <span className="mono-value">{Number(settings.confidence_threshold).toFixed(2)}</span>
              {" · "}
              {realLife ? "real-life domain" : "drawn domain"}
            </dd>
          </div>
        </dl>
        {isDirty ? (
          <p className="callout callout-warn">
            Settings have unsaved changes; starting a run or a preview scan saves them first.
          </p>
        ) : null}

        <fieldset className="run-modes">
          <legend className="field-label">Run mode</legend>
          {RUN_MODES.map((m) => {
            const disabled = Boolean(m.drawnOnly && realLife);
            return (
              <label
                key={m.id}
                className={`radio-option${disabled ? " is-disabled" : ""}`}
                title={disabled ? "Animated / drawn domain only" : undefined}
              >
                <input
                  type="radio"
                  name="run_mode"
                  value={m.id}
                  checked={mode.id === m.id}
                  disabled={disabled}
                  onChange={() => setModeId(m.id)}
                />
                <span className="radio-option-text">
                  <span className="radio-option-label">{m.label(realLife)}</span>
                  <span className="radio-option-help">
                    {disabled ? "Animated / drawn domain only. " : ""}
                    {m.help(realLife)}
                  </span>
                </span>
              </label>
            );
          })}
        </fieldset>

        {otherActiveRun ? (
          <div className="callout callout-warn run-blocked" role="status">
            <span>
              Run #{otherActiveRun.run_id} is still {otherActiveRun.status}. Wait for it to finish
              or cancel it before starting another run.
            </span>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => app.openRun(otherActiveRun.run_id)}
            >
              Open run #{otherActiveRun.run_id}
            </button>
          </div>
        ) : null}

        <ScanPreview />
      </div>
      <div className="pane-actions">
        <PreviewScanButton />
        <button
          type="button"
          className="btn btn-primary btn-lg"
          disabled={startDisabled}
          onClick={() => app.handleStartRun(mode.id)}
        >
          {opsLoading.startingRun ? "Starting…" : mode.start(realLife)}
        </button>
      </div>
    </section>
  );
}

export function RunDashboard({ compact = false }) {
  const app = useApp();
  const { runId, runStatus, runActive, stats, opsLoading, selectedTags, anyRunActive } = app;
  if (!runId || !runStatus) return null;
  const etaLabel =
    runActive && runStatus.eta_seconds_remaining != null
      ? formatDurationSeconds(runStatus.eta_seconds_remaining)
      : null;
  const finishLabel =
    runActive && runStatus.eta_finish_at ? formatEtaFinishAt(runStatus.eta_finish_at) : null;
  const pct = Number(runStatus.progress_pct || 0);
  const canReclassify =
    !runActive &&
    stats.reviewNeeded > 0 &&
    ["completed", "failed", "cancelled"].includes(runStatus.status);

  const metrics = [
    ["Total", runStatus.total_images],
    ["Processed", runStatus.processed_images],
    ["Failed", runStatus.failed_images, runStatus.failed_images ? "bad" : null],
    ["Needs review", stats.reviewNeeded, stats.reviewNeeded ? "warn" : null],
    ["Progress", `${pct.toFixed(1)}%`],
    ["ETA", etaLabel || (runActive ? "…" : "—")],
    ["Finish by", finishLabel || (runActive ? "…" : "—")],
  ];

  return (
    <section
      className={`layout-panel run-dashboard${compact ? " is-compact" : ""}`}
      aria-labelledby="run-dashboard-title"
    >
      <div className="pane-head">
        <h2 className="pane-title" id="run-dashboard-title">
          Run #{runId}
        </h2>
        <StatusChip status={runStatus.status} kind="run" />
        <span className="pane-meta">{runStatus.tagger_model || ""}</span>
      </div>
      <div className="pane-body">
        <dl className="metric-grid">
          {metrics.map(([label, value, tone]) => (
            <div key={label} className={`metric${tone ? ` is-${tone}` : ""}`}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        <ProgressBar
          pct={pct}
          label={`Run #${runId} progress`}
          indeterminate={runActive && !runStatus.total_images}
        />
        <p className="run-detail">
          {runStatus.avg_infer_ms_per_image != null ? (
            <span>Avg infer: {Number(runStatus.avg_infer_ms_per_image).toFixed(1)} ms/image</span>
          ) : null}
          {etaLabel ? (
            <span>
              Expected finish in {etaLabel}
              {finishLabel ? ` (~${finishLabel})` : ""}
            </span>
          ) : null}
          {runStatus.inference_mode ? <span>Mode: {runStatus.inference_mode}</span> : null}
          {runStatus.cancel_requested ? <span className="is-warn">Cancel requested</span> : null}
        </p>
        {canReclassify ? (
          <div className="callout callout-info reclassify-banner" role="region" aria-label="Reclassify needs review">
            <span className="callout-body">
              {stats.reviewNeeded} item{stats.reviewNeeded === 1 ? " needs" : "s need"} review — retry
              with another model (approved/rejected/migrated stay untouched). SwinV2 is much faster
              than EVA02.
            </span>
            <span className="reclassify-controls">
              <select
                value={app.reclassifyModel}
                onChange={(e) => app.setReclassifyModel(e.target.value)}
                disabled={opsLoading.reclassifying}
                aria-label="Model for reclassify"
              >
                {TAGGER_MODELS.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="btn"
                disabled={opsLoading.reclassifying || selectedTags.length === 0 || anyRunActive}
                onClick={app.handleReclassifyNeedsReview}
                aria-label="Reclassify needs review items"
              >
                {opsLoading.reclassifying ? "Queuing…" : "Reclassify needs review"}
              </button>
            </span>
          </div>
        ) : null}
        {runStatus.last_error ? (
          <p className="callout callout-danger" role="alert">
            {runStatus.last_error}
          </p>
        ) : null}
      </div>
      <div className="pane-actions">
        {runActive ? (
          <button type="button" className="btn btn-danger" onClick={app.cancelActiveRun}>
            Cancel run
          </button>
        ) : null}
        {!compact ? (
          <a className="btn btn-primary" href="#/review">
            Review results
            {stats.reviewNeeded ? ` · ${stats.reviewNeeded} need review` : ""}
          </a>
        ) : null}
      </div>
    </section>
  );
}

function EnginePanel() {
  const { providerInfo, offlineMode, settings, anyRunActive, opsLoading, handleUnloadModels } =
    useApp();
  const loaded = providerInfo?.loaded_models || [];
  const rows = providerInfo
    ? [
        ["Device", providerInfo.likely_device || "unknown"],
        ["CUDA", providerInfo.cuda_usable ? "usable" : "not usable", providerInfo.cuda_usable ? "ok" : null],
        ["Active", (providerInfo.active_providers || []).join(", ") || "n/a"],
        ["Tagger", providerInfo.tagger_model || settings.tagger_model],
        ["Forced CPU", providerInfo.forced_cpu ? "yes" : "no"],
        ["Loaded", loaded.length ? loaded.join(", ") : "none"],
      ]
    : [];
  return (
    <section className="layout-panel engine-panel" aria-labelledby="engine-title">
      <div className="pane-head">
        <h2 className="pane-title" id="engine-title">
          Engine
        </h2>
      </div>
      {offlineMode ? (
        <p className="pane-empty">Demo mode: no inference engine attached.</p>
      ) : !providerInfo ? (
        <p className="pane-empty">Engine status unavailable.</p>
      ) : (
        <>
          <div className="stat-rows">
            {rows.map(([label, value, tone]) => (
              <div key={label} className={`stat-row${tone ? ` is-${tone}` : ""}`}>
                <span>{label}</span>
                <span className="stat-row-value">{value}</span>
              </div>
            ))}
          </div>
          <div className="pane-foot">
            <button
              type="button"
              className="btn btn-sm btn-block"
              disabled={anyRunActive || opsLoading.unloadingModels}
              onClick={handleUnloadModels}
              title={
                anyRunActive
                  ? "Cancel the active run before unloading GPU models"
                  : "Drop cached tagger sessions to free GPU memory"
              }
            >
              {opsLoading.unloadingModels ? "Unloading…" : "Unload GPU models"}
            </button>
          </div>
        </>
      )}
    </section>
  );
}

function RecentRuns() {
  const { runs, runId, openRunForReview } = useApp();
  return (
    <section className="layout-panel recent-runs" aria-labelledby="recent-runs-title">
      <div className="pane-head">
        <h2 className="pane-title" id="recent-runs-title">
          Recent runs
        </h2>
        <a className="pane-link" href="#/runs">
          All runs
        </a>
      </div>
      {runs.length === 0 ? (
        <p className="pane-empty">No runs yet.</p>
      ) : (
        <ul className="recent-run-list">
          {runs.slice(0, 5).map((run) => (
            <li key={run.run_id}>
              <button
                type="button"
                className="recent-run"
                aria-current={run.run_id === runId ? "true" : undefined}
                onClick={() => openRunForReview(run.run_id)}
              >
                <span className="recent-run-head">
                  <span className="recent-run-name">Run #{run.run_id}</span>
                  <StatusChip status={run.status} kind="run" />
                </span>
                <span className="recent-run-sub">
                  {runProgressText(run)} images · {pathTail(run.root_repo)}
                  {run.created_at ? ` · ${formatRunDate(run.created_at)}` : ""}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default function RunPage() {
  const { selectedTags, scanPreview, settings, runs } = useApp();
  return (
    <>
      <PageHead
        title="Tag & run"
        lede="Thresh a media dump into destination bins: pick the destinations, start a run, then review and migrate."
      >
        <PageStats
          stats={[
            ["destinations", selectedTags.length],
            ["eligible", scanPreview?.stats ? scanPreview.stats.eligible_images : "—"],
            ["runs", runs.length],
            ["domain", settings.tagging_domain === "real_life" ? "real life" : "drawn"],
          ]}
        />
      </PageHead>
      <div className="run-grid">
        <Destinations />
        <div className="run-main">
          <NewRun />
          <RunDashboard />
        </div>
        <div className="run-side">
          <EnginePanel />
          <RecentRuns />
        </div>
      </div>
    </>
  );
}
