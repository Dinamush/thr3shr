import React, { useEffect, useMemo, useState } from "react";
import { api } from "./api";

const DEFAULT_SETTINGS = {
  root_repo: "",
  categories_root: "",
  confidence_threshold: 0.6,
  default_migrate_mode: "copy",
};

function App() {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [offlineMode, setOfflineMode] = useState(false);
  const [runId, setRunId] = useState(null);
  const [runMeta, setRunMeta] = useState(null);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [migrateMode, setMigrateMode] = useState("copy");
  const [selectedIds, setSelectedIds] = useState([]);
  const [tagQuery, setTagQuery] = useState("");
  const [tagOptions, setTagOptions] = useState([]);
  const [selectedTags, setSelectedTags] = useState([]);

  async function refreshItems(currentRunId) {
    if (!currentRunId) return;
    const [runInfo, runItems] = await Promise.all([
      api.getRun(currentRunId),
      api.getRunItems(currentRunId),
    ]);
    setRunMeta(runInfo);
    setItems(runItems);
  }

  useEffect(() => {
    api.getSettings()
      .then((data) => {
        setSettings(data);
        setMigrateMode(data.default_migrate_mode || "copy");
        setOfflineMode(api.isOfflineMode());
      })
      .catch((err) => setError(err.message));
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

  const stats = useMemo(() => {
    return {
      total: items.length,
      reviewNeeded: items.filter((x) => x.needs_review).length,
      approved: items.filter((x) => x.status === "approved").length,
      migrated: items.filter((x) => x.status === "migrated").length,
    };
  }, [items]);

  async function handleSaveSettings(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const saved = await api.saveSettings(settings);
      setSettings(saved);
      setOfflineMode(api.isOfflineMode());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleStartRun() {
    setLoading(true);
    setError("");
    try {
      const result = await api.startRun({
        ...settings,
        selected_folders: selectedTags.length > 0 ? selectedTags : null,
      });
      setOfflineMode(api.isOfflineMode());
      setRunId(result.run_id);
      await refreshItems(result.run_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function updateStatus(itemId, status) {
    await api.updateItem(itemId, { status });
    await refreshItems(runId);
  }

  async function updateFinalTag(itemId, finalTag) {
    await api.updateItem(itemId, { final_tag: finalTag, status: "reviewed" });
    await refreshItems(runId);
  }

  async function applyBatch(status) {
    if (selectedIds.length === 0) return;
    await api.batchUpdate(runId, { item_ids: selectedIds, status });
    setSelectedIds([]);
    await refreshItems(runId);
  }

  async function migrateApproved() {
    setLoading(true);
    setError("");
    try {
      await api.migrateRun(runId, { mode: migrateMode, create_missing_folders: true });
      setOfflineMode(api.isOfflineMode());
      await refreshItems(runId);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function addSelectedTag(value) {
    if (!value) return;
    if (selectedTags.includes(value)) return;
    setSelectedTags([...selectedTags, value]);
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
    setSelectedTags(selectedTags.filter((t) => t !== value));
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
          <button disabled={loading}>Save Settings</button>
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
        <button disabled={loading} onClick={handleStartRun}>
          Start Run
        </button>
      </section>

      {runId && (
        <section className="card">
          <h2>Review Queue (Run #{runId})</h2>
          <div className="stats">
            <span>Total: {stats.total}</span>
            <span>Needs Review: {stats.reviewNeeded}</span>
            <span>Approved: {stats.approved}</span>
            <span>Migrated: {stats.migrated}</span>
          </div>
          <div className="actions">
            <button onClick={() => applyBatch("approved")}>Approve Selected</button>
            <button onClick={() => applyBatch("rejected")}>Reject Selected</button>
            <select value={migrateMode} onChange={(e) => setMigrateMode(e.target.value)}>
              <option value="copy">copy</option>
              <option value="move">move</option>
            </select>
            <button onClick={migrateApproved}>Migrate Approved</button>
          </div>

          <table>
            <thead>
              <tr>
                <th>Select</th>
                <th>Image</th>
                <th>Primary</th>
                <th>Score</th>
                <th>Secondary</th>
                <th>Status</th>
                <th>Final Tag</th>
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
                  <td title={item.file_path}>{item.relative_path}</td>
                  <td>{item.primary_tag || "-"}</td>
                  <td>{item.primary_score?.toFixed(3) || "-"}</td>
                  <td>
                    {item.secondary_suggestions
                      .map((s) => `${s.tag} (${Number(s.score).toFixed(3)})`)
                      .join(", ")}
                  </td>
                  <td>{item.status}</td>
                  <td>
                    <input
                      value={item.final_tag || ""}
                      onChange={(e) => updateFinalTag(item.id, e.target.value)}
                    />
                  </td>
                  <td>
                    <button onClick={() => updateStatus(item.id, "approved")}>Approve</button>
                    <button onClick={() => updateStatus(item.id, "rejected")}>Reject</button>
                    <button onClick={() => updateStatus(item.id, "reviewed")}>Reviewed</button>
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
