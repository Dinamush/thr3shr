import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";

const DEFAULT_SETTINGS = {
  root_repo: "",
  categories_root: "",
  confidence_threshold: 0.6,
  default_migrate_mode: "copy",
  scan_recursive: true,
};
const ACTIVE_RUN_STORAGE_KEY = "imageClassifierActiveRunId";

function App() {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [offlineMode, setOfflineMode] = useState(false);
  const [runId, setRunId] = useState(null);
  const [runStatus, setRunStatus] = useState(null);
  const [runMeta, setRunMeta] = useState(null);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [providerInfo, setProviderInfo] = useState(null);
  const [migrateMode, setMigrateMode] = useState("copy");
  const [selectedIds, setSelectedIds] = useState([]);
  const [tagQuery, setTagQuery] = useState("");
  const [tagOptions, setTagOptions] = useState([]);
  const [selectedTags, setSelectedTags] = useState([]);
  const [previewErrors, setPreviewErrors] = useState({});
  const [expandedScoreRows, setExpandedScoreRows] = useState({});
  const [scoreDebugByItem, setScoreDebugByItem] = useState({});
  const [scoreLoadingByItem, setScoreLoadingByItem] = useState({});
  const [finalTagDrafts, setFinalTagDrafts] = useState({});
  const [opsLoading, setOpsLoading] = useState({
    saving: false,
    startingRun: false,
    updatingStatus: false,
    applyingBatch: false,
    migrating: false,
  });
  const finalTagTimersRef = useRef({});
  const pollTimerRef = useRef(null);

  async function refreshItems(currentRunId) {
    if (!currentRunId) return;
    try {
      const [runInfo, runItems] = await Promise.all([
        api.getRun(currentRunId),
        api.getRunItems(currentRunId),
      ]);
      setRunMeta(runInfo);
      setItems(runItems);
    } catch (err) {
      console.error("refreshItems failed", err);
      setError(`Failed to refresh items: ${err.message}`);
    }
  }

  useEffect(() => {
    api.getSettings()
      .then((data) => {
        setSettings(data);
        setMigrateMode(data.default_migrate_mode || "copy");
        setOfflineMode(api.isOfflineMode());
      })
      .catch((err) => setError(err.message));
    const storedRunId = Number(localStorage.getItem(ACTIVE_RUN_STORAGE_KEY) || 0);
    if (storedRunId > 0) {
      setRunId(storedRunId);
    }
    api.getProviders()
      .then((info) => setProviderInfo(info))
      .catch(() => setProviderInfo(null));
  }, []);

  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      api.getTags(tagQuery, 50)
        .then((data) => {
          if (cancelled) return;
          setTagOptions(data.items || []);
        })
        .catch(() => {
          if (cancelled) return;
          setTagOptions([]);
        });
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [tagQuery]);

  useEffect(() => {
    return () => {
      Object.values(finalTagTimersRef.current).forEach((timerId) => clearTimeout(timerId));
      finalTagTimersRef.current = {};
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, []);

  const stats = useMemo(() => {
    return {
      total: items.length,
      reviewNeeded: items.filter((x) => x.needs_review).length,
      approved: items.filter((x) => x.status === "approved").length,
      migrated: items.filter((x) => x.status === "migrated").length,
    };
  }, [items]);

  async function pollRunStatus(currentRunId) {
    try {
      const status = await api.getRunStatus(currentRunId);
      setRunStatus(status);
      await refreshItems(currentRunId);
      if (["completed", "cancelled", "failed"].includes(status.status)) {
        if (pollTimerRef.current) {
          clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
        }
        localStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
      }
    } catch (err) {
      setError(`Failed to poll run status: ${err.message}`);
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    }
  }

  function startStatusPolling(currentRunId) {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    pollRunStatus(currentRunId);
    pollTimerRef.current = setInterval(() => pollRunStatus(currentRunId), 1500);
  }

  useEffect(() => {
    if (!runId) return;
    localStorage.setItem(ACTIVE_RUN_STORAGE_KEY, String(runId));
    startStatusPolling(runId);
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [runId]);

  async function handleSaveSettings(e) {
    e.preventDefault();
    setLoading(true);
    setOpsLoading((prev) => ({ ...prev, saving: true }));
    setError("");
    try {
      const saved = await api.saveSettings(settings);
      setSettings(saved);
      setOfflineMode(api.isOfflineMode());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setOpsLoading((prev) => ({ ...prev, saving: false }));
    }
  }

  async function handleStartRun() {
    setLoading(true);
    setOpsLoading((prev) => ({ ...prev, startingRun: true }));
    setError("");
    try {
      const result = await api.startRun({
        ...settings,
        selected_folders: selectedTags.length > 0 ? selectedTags : null,
      });
      setOfflineMode(api.isOfflineMode());
      setRunId(result.run_id);
      setPreviewErrors({});
      setRunStatus({
        run_id: result.run_id,
        status: result.status || "pending",
        total_images: 0,
        processed_images: 0,
        failed_images: 0,
        progress_pct: 0,
        cancel_requested: false,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setOpsLoading((prev) => ({ ...prev, startingRun: false }));
    }
  }

  async function updateStatus(itemId, status) {
    setOpsLoading((prev) => ({ ...prev, updatingStatus: true }));
    setError("");
    try {
      await api.updateItem(itemId, { status });
      await refreshItems(runId);
    } catch (err) {
      console.error("updateStatus failed", err);
      setError(`Failed to update status: ${err.message}`);
    } finally {
      setOpsLoading((prev) => ({ ...prev, updatingStatus: false }));
    }
  }

  function queueFinalTagUpdate(itemId, finalTag) {
    setFinalTagDrafts((prev) => ({ ...prev, [itemId]: finalTag }));
    if (finalTagTimersRef.current[itemId]) {
      clearTimeout(finalTagTimersRef.current[itemId]);
    }
    finalTagTimersRef.current[itemId] = setTimeout(async () => {
      setError("");
      try {
        await api.updateItem(itemId, { final_tag: finalTag, status: "reviewed" });
        await refreshItems(runId);
      } catch (err) {
        console.error("queueFinalTagUpdate failed", err);
        setError(`Failed to update final tag: ${err.message}`);
      }
    }, 350);
  }

  async function applyBatch(status) {
    if (selectedIds.length === 0) return;
    if (status === "rejected") {
      const confirmed = window.confirm("Reject selected items? This may require manual recovery.");
      if (!confirmed) return;
    }
    setOpsLoading((prev) => ({ ...prev, applyingBatch: true }));
    setError("");
    try {
      await api.batchUpdate(runId, { item_ids: selectedIds, status });
      setSelectedIds([]);
      await refreshItems(runId);
    } catch (err) {
      console.error("applyBatch failed", err);
      setError(`Batch update failed: ${err.message}`);
    } finally {
      setOpsLoading((prev) => ({ ...prev, applyingBatch: false }));
    }
  }

  async function migrateApproved() {
    setLoading(true);
    setOpsLoading((prev) => ({ ...prev, migrating: true }));
    setError("");
    const confirmed = window.confirm(
      `Migrate approved items using ${migrateMode}? This can move/copy many files.`
    );
    if (!confirmed) {
      setLoading(false);
      setOpsLoading((prev) => ({ ...prev, migrating: false }));
      return;
    }
    try {
      await api.migrateRun(runId, { mode: migrateMode, create_missing_folders: true });
      setOfflineMode(api.isOfflineMode());
      await refreshItems(runId);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setOpsLoading((prev) => ({ ...prev, migrating: false }));
    }
  }

  async function cancelActiveRun() {
    if (!runId) return;
    setError("");
    try {
      const status = await api.cancelRun(runId);
      setRunStatus(status);
    } catch (err) {
      setError(`Failed to cancel run: ${err.message}`);
    }
  }

  async function toggleScoreDebug(itemId) {
    const isExpanded = Boolean(expandedScoreRows[itemId]);
    if (isExpanded) {
      setExpandedScoreRows((prev) => ({ ...prev, [itemId]: false }));
      return;
    }
    setExpandedScoreRows((prev) => ({ ...prev, [itemId]: true }));
    if (scoreDebugByItem[itemId]) return;
    setScoreLoadingByItem((prev) => ({ ...prev, [itemId]: true }));
    try {
      const response = await api.getItemScores(itemId);
      setScoreDebugByItem((prev) => ({ ...prev, [itemId]: response.full_scores || {} }));
    } catch (err) {
      setError(`Failed to load score debug JSON: ${err.message}`);
    } finally {
      setScoreLoadingByItem((prev) => ({ ...prev, [itemId]: false }));
    }
  }

  function addSelectedTag(value) {
    if (!value) return;
    setSelectedTags((prev) => (prev.includes(value) ? prev : [...prev, value]));
  }

  async function addValidatedTag(rawValue) {
    const value = rawValue.trim();
    if (!value) return;
    if (selectedTags.includes(value)) return;

    // Fast path when current dropdown options already include the tag.
    if (tagOptions.includes(value)) {
      addSelectedTag(value);
      return;
    }

    // Validate against backend tag index to avoid accidental typo tags.
    const result = await api.getTags(value, 200);
    if ((result.items || []).includes(value)) {
      addSelectedTag(value);
      return;
    }
    throw new Error(`Tag not found in tags.csv: ${value}`);
  }

  async function addTagsFromInput(raw) {
    const parts = raw
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
    if (parts.length === 0) return;

    for (const part of parts) {
      await addValidatedTag(part);
    }
    setTagQuery("");
  }

  function removeSelectedTag(value) {
    setSelectedTags((prev) => prev.filter((t) => t !== value));
  }

  return (
    <div className="container">
      <h1>Image Classifier Workflow</h1>
      <p>
        Configure source + category roots, run tagging, review predictions, then approve and
        migrate using move/copy.
      </p>
      {offlineMode && (
        <div className="error">
          Backend is offline. Running in browser-only demo mode with local mock data.
        </div>
      )}
      {providerInfo && !offlineMode && (
        <div className="stats">
          <span>Inference device: {providerInfo.likely_device || "unknown"}</span>
          <span>CUDA available: {String(Boolean(providerInfo.cuda_available))}</span>
          <span>Forced CPU: {String(Boolean(providerInfo.forced_cpu))}</span>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      <section className="card">
        <h2>Configuration</h2>
        <form onSubmit={handleSaveSettings} className="grid">
          <label>
            Root Repository
            <input
              value={settings.root_repo}
              onChange={(e) => setSettings({ ...settings, root_repo: e.target.value })}
            />
          </label>
          <label>
            Categories Root
            <input
              value={settings.categories_root}
              onChange={(e) => setSettings({ ...settings, categories_root: e.target.value })}
            />
          </label>
          <label>
            Confidence Threshold
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={settings.confidence_threshold}
              onChange={(e) =>
                setSettings({ ...settings, confidence_threshold: Number(e.target.value) })
              }
            />
          </label>
          <label>
            Default Migrate Mode
            <select
              value={settings.default_migrate_mode}
              onChange={(e) => setSettings({ ...settings, default_migrate_mode: e.target.value })}
            >
              <option value="copy">copy</option>
              <option value="move">move</option>
            </select>
          </label>
          <label style={{ flexDirection: "row", alignItems: "center", gap: "0.5rem" }}>
            <input
              type="checkbox"
              checked={Boolean(settings.scan_recursive)}
              onChange={(e) => setSettings({ ...settings, scan_recursive: e.target.checked })}
            />
            Scan subfolders recursively
          </label>
          <button disabled={loading || opsLoading.saving}>Save Settings</button>
        </form>
      </section>

      <section className="card">
        <h2>Run Classification</h2>
        <label>
          Tag match search (from tags.csv)
          <input
            value={tagQuery}
            onChange={(e) => setTagQuery(e.target.value)}
            onKeyDown={async (e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                setError("");
                try {
                  await addTagsFromInput(tagQuery);
                } catch (err) {
                  setError(err.message);
                }
              }
            }}
            placeholder="Type to search tags..."
          />
        </label>
        <div className="actions">
          <select defaultValue="" onChange={(e) => addSelectedTag(e.target.value)}>
            <option value="" disabled>
              Select matching tag
            </option>
            {tagOptions.map((tag) => (
              <option key={tag} value={tag}>
                {tag}
              </option>
            ))}
          </select>
          <button
            onClick={async () => {
              setError("");
              try {
                await addTagsFromInput(tagQuery);
              } catch (err) {
                setError(err.message);
              }
            }}
          >
            Add typed tag(s)
          </button>
        </div>
        <div className="stats">
          <span>Selected tags:</span>
          {selectedTags.length === 0 && <span>none</span>}
          {selectedTags.map((tag) => (
            <span key={tag}>
              {tag} <button onClick={() => removeSelectedTag(tag)}>x</button>
            </span>
          ))}
        </div>
        <button
          disabled={
            loading ||
            opsLoading.startingRun ||
            (runStatus && ["pending", "running"].includes(runStatus.status))
          }
          onClick={handleStartRun}
        >
          Start Run
        </button>
      </section>

      {runId && (
        <section className="card">
          <h2>Review Queue (Run #{runId})</h2>
          {runStatus && (
            <div className="stats">
              <span>Status: {runStatus.status}</span>
              <span>
                Progress: {runStatus.processed_images}/{runStatus.total_images} (
                {Number(runStatus.progress_pct || 0).toFixed(1)}%)
              </span>
              <span>Failed: {runStatus.failed_images}</span>
              {runStatus.inference_mode && <span>Inference mode: {runStatus.inference_mode}</span>}
              {runStatus.batch_size ? <span>Batch size: {runStatus.batch_size}</span> : null}
              {runStatus.avg_infer_ms_per_image !== null &&
              runStatus.avg_infer_ms_per_image !== undefined ? (
                <span>
                  Avg infer ms/image: {Number(runStatus.avg_infer_ms_per_image).toFixed(1)}
                </span>
              ) : null}
              {runStatus.queue_seed !== null && runStatus.queue_seed !== undefined ? (
                <span>Queue seed: {runStatus.queue_seed}</span>
              ) : null}
              {runStatus.cancel_requested && <span>Cancel requested</span>}
            </div>
          )}
          {runStatus && ["pending", "running"].includes(runStatus.status) && (
            <div className="actions">
              <button onClick={cancelActiveRun}>Cancel Run</button>
            </div>
          )}
          <div className="stats">
            <span>Total: {stats.total}</span>
            <span>Needs Review: {stats.reviewNeeded}</span>
            <span>Approved: {stats.approved}</span>
            <span>Migrated: {stats.migrated}</span>
          </div>
          <div className="actions">
            <button disabled={opsLoading.applyingBatch} onClick={() => applyBatch("approved")}>
              Approve Selected
            </button>
            <button disabled={opsLoading.applyingBatch} onClick={() => applyBatch("rejected")}>
              Reject Selected
            </button>
            <select value={migrateMode} onChange={(e) => setMigrateMode(e.target.value)}>
              <option value="copy">copy</option>
              <option value="move">move</option>
            </select>
            <button disabled={opsLoading.migrating} onClick={migrateApproved}>
              {opsLoading.migrating ? "Migrating..." : "Migrate Approved"}
            </button>
          </div>

          <table>
            <thead>
              <tr>
                <th>Select</th>
                <th>Image</th>
                <th>Primary (Selected)</th>
                <th>Score</th>
                <th>Secondary (Selected)</th>
                <th>Status</th>
                <th>Final Tag</th>
                <th>Debug Scores</th>
                <th>Review</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.id} className={item.needs_review ? "needs-review" : ""}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(item.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedIds([...selectedIds, item.id]);
                        } else {
                          setSelectedIds(selectedIds.filter((id) => id !== item.id));
                        }
                      }}
                    />
                  </td>
                  <td title={item.file_path}>
                    <div className="image-cell">
                      {!previewErrors[item.id] && api.getItemPreviewUrl(item.id) ? (
                        <img
                          className="image-thumb"
                          src={api.getItemPreviewUrl(item.id)}
                          alt={item.relative_path || item.file_path}
                          loading="lazy"
                          onError={() =>
                            setPreviewErrors((prev) => ({
                              ...prev,
                              [item.id]: true,
                            }))
                          }
                        />
                      ) : null}
                      <div className="image-path">{item.relative_path || item.file_path || "-"}</div>
                    </div>
                  </td>
                  <td>{item.primary_tag || "-"}</td>
                  <td>{item.primary_score?.toFixed(3) || "-"}</td>
                  <td>
                    {(item.secondary_suggestions || [])
                      .map((s) => `${s.tag} (${Number(s.score).toFixed(3)})`)
                      .join(", ") || "-"}
                  </td>
                  <td>{item.status}</td>
                  <td>
                    <input
                      value={finalTagDrafts[item.id] ?? (item.final_tag || "")}
                      onChange={(e) => queueFinalTagUpdate(item.id, e.target.value)}
                    />
                  </td>
                  <td>
                    <button onClick={() => toggleScoreDebug(item.id)}>
                      {expandedScoreRows[item.id] ? "Hide JSON" : "Show JSON"}
                    </button>
                    {expandedScoreRows[item.id] && (
                      <pre className="debug-json">
                        {scoreLoadingByItem[item.id]
                          ? "Loading..."
                          : JSON.stringify(scoreDebugByItem[item.id] || {}, null, 2)}
                      </pre>
                    )}
                  </td>
                  <td>
                    <button
                      disabled={opsLoading.updatingStatus}
                      onClick={() => updateStatus(item.id, "approved")}
                    >
                      Approve
                    </button>
                    <button
                      disabled={opsLoading.updatingStatus}
                      onClick={() => updateStatus(item.id, "rejected")}
                    >
                      Reject
                    </button>
                    <button
                      disabled={opsLoading.updatingStatus}
                      onClick={() => updateStatus(item.id, "reviewed")}
                    >
                      Reviewed
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {runMeta && <pre>{JSON.stringify(runMeta.counts, null, 2)}</pre>}
        </section>
      )}
    </div>
  );
}

export default App;
