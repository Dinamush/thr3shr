import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { REAL_LIFE_SENSITIVE_TAGS, isVideoPreviewPath, pathTail } from "../lib";
import {
  EmptyState,
  PageHead,
  PageStats,
  ProgressBar,
  ScoreList,
  StatusChip,
  useApp,
} from "../ui";

const FILTERS = [
  { id: "all", label: "All", match: () => true },
  { id: "needs_review", label: "Needs review", match: (item) => item.needs_review },
  { id: "proposed", label: "Proposed", match: (item) => item.status === "proposed" },
  { id: "reviewed", label: "Reviewed", match: (item) => item.status === "reviewed" },
  { id: "approved", label: "Approved", match: (item) => item.status === "approved" },
  { id: "rejected", label: "Rejected", match: (item) => item.status === "rejected" },
  { id: "migrated", label: "Migrated", match: (item) => item.status === "migrated" },
];

const STATUS_KEYS = { a: "approved", r: "rejected", v: "reviewed" };

// List or Grid, remembered per browser.
const LAYOUT_STORAGE_KEY = "thr3shr-review-layout";
const LAYOUTS = [
  { id: "list", label: "List" },
  { id: "grid", label: "Grid" },
];

function readLayout() {
  try {
    return localStorage.getItem(LAYOUT_STORAGE_KEY) === "grid" ? "grid" : "list";
  } catch {
    return "list";
  }
}

function writeLayout(layout) {
  try {
    localStorage.setItem(LAYOUT_STORAGE_KEY, layout);
  } catch {
    /* storage blocked: the choice lasts for this page only */
  }
}

const NON_TYPING_INPUTS = new Set(["checkbox", "radio", "button", "submit"]);

function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  if (tag === "INPUT") return !NON_TYPING_INPUTS.has(el.type);
  return tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

function itemLabel(item) {
  return item.relative_path || item.file_path;
}

function fileName(item) {
  return String(itemLabel(item) || "").split(/[\\/]/).pop();
}

// size: "row" (list), "tile" (grid) or "large" (detail pane).
function Thumb({ item, size = "row" }) {
  const { previewErrors, setPreviewErrors } = useApp();
  const large = size === "large";
  const url = api.getItemPreviewUrl(item.id, { thumb: !large });
  const isVideo = isVideoPreviewPath(item.file_path || item.relative_path);
  const failed = previewErrors[item.id] || !url;
  return (
    <span className={`review-thumb${size === "row" ? "" : ` is-${size}`}`}>
      {failed ? (
        <span className="review-thumb-fallback">{url ? "No preview" : "Preview offline"}</span>
      ) : (
        <img
          src={url}
          alt={large ? itemLabel(item) : ""}
          loading="lazy"
          onError={() => setPreviewErrors((prev) => ({ ...prev, [item.id]: true }))}
        />
      )}
      {isVideo ? (
        <span className="media-badge" title="Video">
          ▶
        </span>
      ) : null}
    </span>
  );
}

function PrimaryTag({ item }) {
  if (!item.primary_tag) return <span className="primary-empty">needs review</span>;
  return (
    <span className="primary-tag">
      <span className="primary-tag-name">{item.primary_tag}</span>
      {REAL_LIFE_SENSITIVE_TAGS.has(item.primary_tag) ? (
        <span className="status-chip is-warn" title="Sensitive — confirm Final tag">
          sensitive
        </span>
      ) : null}
      {String(item.review_reason || "").includes("vlm_unavailable") ? (
        <span className="status-chip is-warn" title="Adult VLM unavailable">
          fallback
        </span>
      ) : null}
      {item.primary_score != null ? (
        <span className="primary-tag-score">{Number(item.primary_score).toFixed(3)}</span>
      ) : null}
    </span>
  );
}

function ItemTag({ item }) {
  if (item.final_tag && item.final_tag !== item.primary_tag) {
    return <span className="review-row-final">→ {item.final_tag}</span>;
  }
  return <PrimaryTag item={item} />;
}

function ItemDetail({ item, position, total, backLabel, onBack, finalTagRef }) {
  const app = useApp();
  const isMigrated = item.status === "migrated";
  const rowBusy = Boolean(app.pendingItemIds[item.id]);
  const label = itemLabel(item);
  const expanded = Boolean(app.expandedScoreRows[item.id]);

  return (
    <section className="layout-panel review-detail" aria-label="Item detail">
      <div className="pane-head">
        <button type="button" className="btn btn-ghost btn-sm review-back" onClick={onBack}>
          ← {backLabel}
        </button>
        <h2 className="pane-title">
          Item {position} of {total}
        </h2>
        <StatusChip status={item.status} />
      </div>
      <div className="review-preview" title={item.file_path}>
        <Thumb item={item} size="large" />
      </div>
      <div className="pane-body review-detail-body">
        <p className="review-path" title={item.file_path}>
          {label || "-"}
        </p>
        {item.review_reason ? <p className="callout callout-warn">{item.review_reason}</p> : null}

        <dl className="review-facts">
          <div>
            <dt>Primary</dt>
            <dd>
              <PrimaryTag item={item} />
            </dd>
          </div>
          <div>
            <dt>Status</dt>
            <dd>
              <StatusChip status={item.status} />
              {isMigrated && item.migrated_to ? (
                <span className="migrated-path" title={item.migrated_to}>
                  {"→ "}
                  {pathTail(item.migrated_to)}
                </span>
              ) : null}
            </dd>
          </div>
        </dl>

        <div className="form-group">
          <label htmlFor={`final-tag-${item.id}`}>Final tag</label>
          <input
            key={item.id}
            ref={finalTagRef}
            id={`final-tag-${item.id}`}
            className="final-tag-input"
            aria-label={`Final tag for ${label}`}
            list="final-tag-options"
            value={app.finalTagDrafts[item.id] ?? (item.final_tag || "")}
            placeholder={isMigrated ? "" : "Destination tag"}
            disabled={isMigrated}
            onChange={(e) => app.setFinalTagDraft(item.id, e.target.value)}
            onBlur={() => app.commitFinalTag(item)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                app.dropFinalTagDraft(item.id);
              }
            }}
          />
          {!isMigrated ? (
            <p className="help-text">Enter saves the destination; Esc drops the edit.</p>
          ) : null}
        </div>

        {isMigrated ? (
          <p className="muted review-done">Done — migrated items are locked.</p>
        ) : (
          <div className="review-actions">
            <button
              type="button"
              className="btn btn-primary"
              disabled={rowBusy || item.status === "approved"}
              onClick={() => app.updateStatus(item.id, "approved")}
            >
              Approve <kbd className="app-kbd">A</kbd>
            </button>
            <button
              type="button"
              className="btn"
              disabled={rowBusy || item.status === "rejected"}
              onClick={() => app.updateStatus(item.id, "rejected")}
            >
              Reject <kbd className="app-kbd">R</kbd>
            </button>
            <button
              type="button"
              className="btn"
              disabled={rowBusy || item.status === "reviewed"}
              onClick={() => app.updateStatus(item.id, "reviewed")}
            >
              Reviewed <kbd className="app-kbd">V</kbd>
            </button>
          </div>
        )}

        <h3 className="detail-heading">Category tags</h3>
        <ScoreList
          entries={item.secondary_suggestions || []}
          sensitive={REAL_LIFE_SENSITIVE_TAGS}
          label="Category tags"
          empty="No category tags"
        />
        <h3 className="detail-heading">Global top</h3>
        <ScoreList
          entries={(item.global_top_tags || []).slice(0, 5)}
          label="Global top tags"
          empty="No global tags"
        />

        <div className="debug-json-block">
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            aria-expanded={expanded}
            onClick={() => app.toggleScoreDebug(item.id)}
          >
            {expanded ? "Hide JSON" : "Show JSON"}
          </button>
          {expanded ? (
            <pre className="debug-json">
              {app.scoreLoadingByItem[item.id]
                ? "Loading..."
                : JSON.stringify(app.scoreDebugByItem[item.id] || {}, null, 2)}
            </pre>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function SelectionBar({ selectedCount }) {
  const app = useApp();
  const { opsLoading, runActive, stats, migrateMode } = app;
  return (
    <div className={`selection-bar${selectedCount ? " has-selection" : ""}`}>
      <span className="selection-count">
        <strong>{selectedCount}</strong> selected
      </span>
      {selectedCount ? (
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => app.setSelectedIds([])}>
          Clear
        </button>
      ) : null}
      <button
        type="button"
        className="btn btn-primary"
        disabled={opsLoading.applyingBatch || selectedCount === 0}
        onClick={() => app.applyBatch("approved")}
      >
        Approve selected{selectedCount ? ` (${selectedCount})` : ""}
      </button>
      <button
        type="button"
        className="btn"
        disabled={opsLoading.applyingBatch || selectedCount === 0}
        onClick={() => app.applyBatch("rejected")}
      >
        Reject selected{selectedCount ? ` (${selectedCount})` : ""}
      </button>
      <span className="selection-spacer" />
      <fieldset className="segmented" disabled={opsLoading.migrating}>
        <legend className="segmented-legend">Migrate mode</legend>
        {["copy", "move"].map((mode) => (
          <label key={mode} className="segmented-option">
            <input
              type="radio"
              name="migrate_mode"
              value={mode}
              checked={migrateMode === mode}
              onChange={() => app.setMigrateMode(mode)}
            />
            <span>{mode}</span>
          </label>
        ))}
      </fieldset>
      <button
        type="button"
        className="btn btn-migrate"
        disabled={opsLoading.migrating || runActive || stats.approved === 0}
        title={
          runActive
            ? "Wait for the run to finish (or cancel it) before migrating"
            : stats.approved === 0
              ? "Approve items first"
              : undefined
        }
        onClick={app.migrateApproved}
      >
        <span className="btn-migrate-arrow" aria-hidden="true">
          →
        </span>
        {opsLoading.migrating ? "Migrating…" : `Migrate approved (${stats.approved})`}
      </button>
    </div>
  );
}

// Server rendering (the tests) has no layout to measure.
const useIsomorphicLayoutEffect =
  typeof document !== "undefined" && typeof document.createElement === "function"
    ? useLayoutEffect
    : useEffect;

// Rows and tiles all share one height, so only those near the viewport are
// rendered and spacers stand in for the rest: rendering every item of a 10k
// run built 120k DOM nodes and froze the page for seconds.
const INITIAL_ITEMS = 60;
const OVERSCAN_ROWS = 6;

function sameRange(a, b) {
  return a.start === b.start && a.end === b.end && a.cols === b.cols && a.stride === b.stride;
}

// Returns the [start, end) slice of `count` items to render in listRef's list
// or grid, the column count and row stride, and scrollToIndex for items that
// are not rendered yet. The list scrolls itself on wide screens and with the
// page on narrow ones; measuring against the viewport covers both.
function useWindowedItems(listRef, count, layout) {
  const [range, setRange] = useState(() => ({
    start: 0,
    end: Math.min(count, INITIAL_ITEMS),
    cols: 1,
    stride: 0,
    gap: 0,
  }));
  const metricsRef = useRef(null);
  const measureRef = useRef(null);

  measureRef.current = () => {
    const list = listRef.current;
    if (!list) return;
    const style = window.getComputedStyle(list);
    const grid = layout === "grid";
    const cols = grid
      ? Math.max(1, style.gridTemplateColumns.split(" ").filter(Boolean).length)
      : 1;
    const listRect = list.getBoundingClientRect();
    const nodes = list.querySelectorAll("[data-index]");
    if (nodes.length) {
      const first = nodes[0];
      const firstRect = first.getBoundingClientRect();
      const nextRow = Array.prototype.find.call(
        nodes,
        (node) => node.getBoundingClientRect().top > firstRect.top + 0.5
      );
      const gap = grid ? parseFloat(style.rowGap) || 0 : 0;
      const stride = nextRow
        ? nextRow.getBoundingClientRect().top - firstRect.top
        : metricsRef.current?.layout === layout && metricsRef.current.cols === cols
          ? metricsRef.current.stride
          : firstRect.height + gap;
      if (stride > 0) {
        const firstRow = Math.floor(Number(first.dataset.index) / cols);
        const offset = firstRect.top - listRect.top + list.scrollTop - firstRow * stride;
        metricsRef.current = { layout, cols, stride, gap, offset };
      }
    }
    const metrics = metricsRef.current;
    if (!metrics || metrics.layout !== layout || metrics.cols !== cols) return;
    const toContent = (y) => y - listRect.top + list.scrollTop - metrics.offset;
    const viewTop = toContent(Math.max(listRect.top, 0));
    const viewBottom = toContent(Math.min(listRect.bottom, window.innerHeight));
    const rows = Math.ceil(count / cols);
    const firstRow = Math.min(rows, Math.max(0, Math.floor(viewTop / metrics.stride) - OVERSCAN_ROWS));
    const lastRow = Math.min(rows, Math.max(firstRow, Math.ceil(viewBottom / metrics.stride) + OVERSCAN_ROWS));
    const next = {
      start: firstRow * cols,
      end: Math.min(count, lastRow * cols),
      cols,
      stride: metrics.stride,
      gap: metrics.gap,
    };
    setRange((prev) => (sameRange(prev, next) ? prev : next));
  };

  // After every render: rows may have changed height, count or columns.
  useIsomorphicLayoutEffect(() => measureRef.current());

  const hasList = count > 0;
  useEffect(() => {
    const list = listRef.current;
    if (!list) return undefined;
    let frame = 0;
    const schedule = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        measureRef.current();
      });
    };
    list.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    const observer = typeof ResizeObserver === "function" ? new ResizeObserver(schedule) : null;
    observer?.observe(list);
    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      list.removeEventListener("scroll", schedule);
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      observer?.disconnect();
    };
  }, [listRef, layout, hasList]);

  // Brings an unrendered item into view the way scrollIntoView({ block:
  // "nearest" }) would; the scroll then renders it.
  function scrollToIndex(index) {
    const list = listRef.current;
    const metrics = metricsRef.current;
    if (!list || !metrics) return;
    const listRect = list.getBoundingClientRect();
    const top = metrics.offset + Math.floor(index / metrics.cols) * metrics.stride;
    const bottom = top + metrics.stride - metrics.gap;
    const scrollsItself = list.scrollHeight > list.clientHeight + 1;
    const viewTop = scrollsItself ? list.scrollTop : window.scrollY - (listRect.top + window.scrollY);
    const viewHeight = scrollsItself ? list.clientHeight : window.innerHeight;
    let delta = 0;
    if (top < viewTop) delta = top - viewTop;
    else if (bottom > viewTop + viewHeight) delta = bottom - viewTop - viewHeight;
    if (!delta) return;
    if (scrollsItself) list.scrollTop += delta;
    else window.scrollBy(0, delta);
  }

  const start = Math.min(range.start, count);
  const end = Math.max(start, Math.min(count, range.end || INITIAL_ITEMS));
  const rows = Math.ceil(count / range.cols);
  const rowsBefore = Math.floor(start / range.cols);
  const rowsAfter = Math.max(0, rows - Math.ceil(end / range.cols));
  // A grid spacer sits in its own row, so the row gap after it is already there.
  const spacer = (rowCount) =>
    range.stride && rowCount ? Math.max(0, rowCount * range.stride - range.gap) : 0;
  return {
    start,
    end,
    before: spacer(rowsBefore),
    after: spacer(rowsAfter),
    scrollToIndex,
  };
}

function Spacer({ height }) {
  if (!height) return null;
  return <li className="review-spacer" aria-hidden="true" style={{ height }} />;
}

export default function ReviewPage() {
  const app = useApp();
  const {
    runId,
    runStatus,
    runActive,
    otherActiveRun,
    items,
    stats,
    settings,
    selectedIds,
    selectedIdSet,
  } = app;
  const [filterId, setFilterId] = useState("all");
  const [query, setQuery] = useState("");
  const [focusedId, setFocusedId] = useState(null);
  // view: which pane shows on narrow screens; layout: List or Grid.
  const [view, setView] = useState("list");
  const [layout, setLayout] = useState(readLayout);
  const listRef = useRef(null);
  const finalTagRef = useRef(null);
  const scrollFocusRef = useRef(false);

  const counts = useMemo(() => {
    const out = {};
    for (const filter of FILTERS) out[filter.id] = items.filter(filter.match).length;
    return out;
  }, [items]);

  const filter = FILTERS.find((f) => f.id === filterId) || FILTERS[0];
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return items.filter((item) => {
      if (!filter.match(item)) return false;
      if (!q) return true;
      return [item.relative_path, item.file_path, item.primary_tag, item.final_tag]
        .filter(Boolean)
        .some((text) => String(text).toLowerCase().includes(q));
    });
  }, [items, filter, query]);

  const focusedIndex = Math.max(
    0,
    visible.findIndex((item) => item.id === focusedId)
  );
  const focused = visible[focusedIndex] || null;

  // Select-all covers the rows on screen that are not migrated yet.
  const visibleSelectable = useMemo(
    () => visible.filter((item) => item.status !== "migrated").map((item) => item.id),
    [visible]
  );
  const allVisibleSelected =
    visibleSelectable.length > 0 && visibleSelectable.every((id) => selectedIdSet.has(id));
  const someVisibleSelected = visibleSelectable.some((id) => selectedIdSet.has(id));

  function focusIndex(idx, fromKeyboard = false) {
    const next = visible[Math.max(0, Math.min(visible.length - 1, idx))];
    if (!next) return;
    scrollFocusRef.current = fromKeyboard;
    setFocusedId(next.id);
  }

  function toggleSelected(id, checked) {
    app.setSelectedIds((prev) =>
      checked ? (prev.includes(id) ? prev : [...prev, id]) : prev.filter((x) => x !== id)
    );
  }

  const windowed = useWindowedItems(listRef, visible.length, layout);

  useEffect(() => {
    if (!scrollFocusRef.current || !focused) return;
    scrollFocusRef.current = false;
    const row = listRef.current?.querySelector(`[data-item-id="${focused.id}"]`);
    if (row) row.scrollIntoView({ block: "nearest" });
    else windowed.scrollToIndex(focusedIndex);
  }, [focused]);

  // J/K move, A approve, R reject, V reviewed (each then moves on), X select,
  // F edits the final tag.  Ignored while typing or with a modifier held.
  useEffect(() => {
    const onKey = (ev) => {
      if (ev.defaultPrevented || ev.ctrlKey || ev.metaKey || ev.altKey) return;
      if (isTypingTarget(ev.target) || document.body.classList.contains("jump-open")) return;
      const key = ev.key.toLowerCase();
      if (key === "j" || key === "k") {
        ev.preventDefault();
        focusIndex(focusedIndex + (key === "j" ? 1 : -1), true);
        return;
      }
      if (!focused) return;
      const isMigrated = focused.status === "migrated";
      if (STATUS_KEYS[key]) {
        const status = STATUS_KEYS[key];
        if (isMigrated || app.pendingItemIds[focused.id] || focused.status === status) return;
        ev.preventDefault();
        app.updateStatus(focused.id, status);
        focusIndex(focusedIndex + 1, true);
      } else if (key === "x" && !isMigrated) {
        ev.preventDefault();
        toggleSelected(focused.id, !selectedIdSet.has(focused.id));
      } else if (key === "f" && !isMigrated) {
        ev.preventDefault();
        setView("detail");
        finalTagRef.current?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  });

  function pickLayout(next) {
    setLayout(next);
    writeLayout(next);
  }

  if (!runId) {
    // A run is tagging but none is open here: lead with that run.
    const live = otherActiveRun;
    return (
      <>
        <PageHead title="Review" lede="Approve, retag or reject what a run proposed, then migrate." />
        <EmptyState
          title="No run open"
          action={
            <p className="empty-state-actions">
              {live ? (
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => app.openRunForReview(live.run_id)}
                >
                  Open live run #{live.run_id}
                </button>
              ) : null}
              <a className="btn" href="#/runs">
                Open a run
              </a>
              <a className={live ? "btn" : "btn btn-primary"} href="#/">
                Start a run
              </a>
            </p>
          }
        >
          {live
            ? `Run #${live.run_id} is ${live.status === "pending" ? "queued" : "tagging"}; open it to review items as they arrive, or reopen an earlier run.`
            : "Reopen an earlier run or start a new one to review its results here."}
        </EmptyState>
      </>
    );
  }

  const realLife = settings.tagging_domain === "real_life";
  return (
    <>
      <PageHead
        title={`Review run #${runId}`}
        lede={
          <>
            Primary is the main destination folder; category tags (secondary) are extra labels for
            junctions on Doujin runs
            {realLife
              ? ", or supporting real-life categories. Sensitive suggestions are highlighted."
              : ". Global tops help spot mis-assignments."}
          </>
        }
      >
        <PageStats
          stats={[
            ["listed", stats.total],
            ["need review", stats.reviewNeeded, stats.reviewNeeded ? "warn" : null],
            ["approved", stats.approved, stats.approved ? "ok" : null],
            ["rejected", stats.rejected],
            ["migrated", stats.migrated],
          ]}
        />
      </PageHead>

      {runStatus ? (
        <div className="review-runline">
          <StatusChip status={runStatus.status} kind="run" />
          <span className="review-runline-text">
            {runStatus.processed_images}/{runStatus.total_images} processed
            {runStatus.tagger_model ? ` · ${runStatus.tagger_model}` : ""}
          </span>
          {runActive ? (
            <ProgressBar
              pct={runStatus.progress_pct}
              label={`Run #${runId} progress`}
              indeterminate={!runStatus.total_images}
            />
          ) : null}
          <a className="pane-link" href="#/">
            Run details
          </a>
        </div>
      ) : null}

      <div className="review-layout" data-view={view} data-layout={layout}>
        <section className="layout-panel review-list-pane" aria-label="Items">
          <div className="review-filters" role="tablist" aria-label="Filter items">
            {FILTERS.map((f) => (
              <button
                key={f.id}
                type="button"
                role="tab"
                aria-selected={filterId === f.id}
                className="review-filter"
                onClick={() => setFilterId(f.id)}
              >
                {f.label}
                <span className="review-filter-count">{counts[f.id]}</span>
              </button>
            ))}
          </div>
          <div className="review-list-tools">
            <input
              type="checkbox"
              aria-label="Select all shown items that are not migrated yet"
              checked={allVisibleSelected}
              ref={(el) => {
                if (el) el.indeterminate = someVisibleSelected && !allVisibleSelected;
              }}
              disabled={visibleSelectable.length === 0}
              onChange={(e) => {
                const ids = new Set(visibleSelectable);
                app.setSelectedIds((prev) =>
                  e.target.checked
                    ? [...prev, ...visibleSelectable.filter((id) => !prev.includes(id))]
                    : prev.filter((id) => !ids.has(id))
                );
              }}
            />
            <input
              type="search"
              className="review-search"
              placeholder="Filter by path or tag"
              aria-label="Filter items by path or tag"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <span className="pane-meta">
              {visible.length}/{items.length}
            </span>
            <span className="view-toggle" role="group" aria-label="Item layout">
              {LAYOUTS.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  aria-pressed={layout === option.id}
                  onClick={() => pickLayout(option.id)}
                >
                  {option.label}
                </button>
              ))}
            </span>
          </div>
          {visible.length === 0 ? (
            <p className="pane-empty">
              {items.length === 0
                ? runActive
                  ? "Classifying… items appear here as they are tagged."
                  : "This run has no items."
                : "No items match this filter."}
            </p>
          ) : layout === "grid" ? (
            <ul className="review-grid" ref={listRef}>
              <Spacer height={windowed.before} />
              {visible.slice(windowed.start, windowed.end).map((item, offset) => {
                const idx = windowed.start + offset;
                const isMigrated = item.status === "migrated";
                const isFocused = focused?.id === item.id;
                return (
                  <li
                    key={item.id}
                    data-item-id={item.id}
                    data-index={idx}
                    aria-setsize={visible.length}
                    aria-posinset={idx + 1}
                    className={[
                      "grid-tile",
                      `row-${item.status}`,
                      item.needs_review ? "needs-review" : "",
                      isFocused ? "is-focused" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    <input
                      type="checkbox"
                      className="grid-check"
                      aria-label={`Select ${itemLabel(item)}`}
                      checked={selectedIdSet.has(item.id)}
                      disabled={isMigrated}
                      onChange={(e) => toggleSelected(item.id, e.target.checked)}
                    />
                    <button
                      type="button"
                      className="grid-tile-main"
                      aria-current={isFocused ? "true" : undefined}
                      title={item.file_path}
                      onClick={() => {
                        focusIndex(idx);
                        setView("detail");
                      }}
                    >
                      <Thumb item={item} size="tile" />
                      <span className="grid-tile-text">
                        <span className="grid-tile-path">
                          {fileName(item) || "-"}
                        </span>
                        <span className="grid-tile-meta">
                          <ItemTag item={item} />
                          <StatusChip status={item.status} />
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
              <Spacer height={windowed.after} />
            </ul>
          ) : (
            <ul className="review-list" ref={listRef}>
              <Spacer height={windowed.before} />
              {visible.slice(windowed.start, windowed.end).map((item, offset) => {
                const idx = windowed.start + offset;
                const isMigrated = item.status === "migrated";
                const isFocused = focused?.id === item.id;
                return (
                  <li
                    key={item.id}
                    data-item-id={item.id}
                    data-index={idx}
                    aria-setsize={visible.length}
                    aria-posinset={idx + 1}
                    className={[
                      "review-row",
                      `row-${item.status}`,
                      item.needs_review ? "needs-review" : "",
                      isFocused ? "is-focused" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    <input
                      type="checkbox"
                      aria-label={`Select ${itemLabel(item)}`}
                      checked={selectedIdSet.has(item.id)}
                      disabled={isMigrated}
                      onChange={(e) => toggleSelected(item.id, e.target.checked)}
                    />
                    <button
                      type="button"
                      className="review-row-main"
                      aria-current={isFocused ? "true" : undefined}
                      title={item.file_path}
                      onClick={() => {
                        focusIndex(idx);
                        setView("detail");
                      }}
                    >
                      <Thumb item={item} />
                      <span className="review-row-text">
                        <span className="review-row-path">{itemLabel(item) || "-"}</span>
                        <span className="review-row-meta">
                          <ItemTag item={item} />
                        </span>
                      </span>
                      <StatusChip status={item.status} />
                    </button>
                  </li>
                );
              })}
              <Spacer height={windowed.after} />
            </ul>
          )}
          <p className="review-keys">
            <kbd className="app-kbd">J</kbd>/<kbd className="app-kbd">K</kbd> move ·{" "}
            <kbd className="app-kbd">A</kbd> approve · <kbd className="app-kbd">R</kbd> reject ·{" "}
            <kbd className="app-kbd">V</kbd> reviewed · <kbd className="app-kbd">X</kbd> select ·{" "}
            <kbd className="app-kbd">F</kbd> final tag
          </p>
        </section>

        {focused ? (
          <ItemDetail
            item={focused}
            position={focusedIndex + 1}
            total={visible.length}
            backLabel={layout === "grid" ? "Grid" : "List"}
            onBack={() => setView("list")}
            finalTagRef={finalTagRef}
          />
        ) : (
          <section className="layout-panel review-detail is-empty" aria-label="Item detail">
            <p className="pane-empty">Pick an item to see its preview and tags.</p>
          </section>
        )}
      </div>

      <SelectionBar selectedCount={selectedIds.length} />

      <datalist id="final-tag-options">
        {app.selectedTags.map((tag) => (
          <option key={tag} value={tag} />
        ))}
      </datalist>
    </>
  );
}
