import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import ClassifierDebugPage from "./ClassifierDebugPage";
import { version as APP_VERSION } from "../package.json";

const DEFAULT_SETTINGS = {
  root_repo: "",
  categories_root: "",
  confidence_threshold: 0.6,
  default_migrate_mode: "copy",
  scan_recursive: true,
  experimental_media_enabled: false,
  experimental_style_detector_enabled: false,
  hybrid_ml_on_review: true,
  tagging_domain: "drawn",
  sfw_classify_mode: false,
  selected_tags: [],
  selected_tags_nsfw: [],
  max_inference_workers: 2,
  inference_batch_size: 8,
  force_cpu_inference: false,
  tagger_model: "wd_swinv2_v3",
  wd_general_threshold: 0.35,
};

const REAL_LIFE_SENSITIVE_TAGS = new Set(["BBC", "Ebony", "Asian"]);
const REAL_LIFE_DEFAULT_FOLDERS = [
  "creampie",
  "cumshot",
  "facial",
  "oral",
  "blowjob",
  "handjob",
  "anal",
  "vaginal",
  "masturbation",
  "threesome",
  "group",
  "lesbian",
  "POV",
  "interracial",
  "cuck",
  "hotwife",
  "BBC",
  "Ebony",
  "Asian",
];

const TAGGER_MODELS = [
  {
    id: "ml_danbooru",
    label: "ML-Danbooru",
    help: "Loli-only classifier — strong recall on that folder (not used for full multi-folder runs).",
  },
  {
    id: "wd_swinv2_v3",
    label: "WD SwinV2 v3",
    help: "Recommended default — strong general-tag accuracy on anime art.",
  },
  {
    id: "wd_eva02_large",
    label: "WD EVA02 Large",
    help: "Largest WD tagger; slower, often slightly more accurate.",
  },
];

const SFW_CLASSIFY_FOLDERS = ["SFW", "scenery"];
// Last run opened in the dashboard. Kept after the run finishes so a reload
// returns to its review table (key name predates that and stays for upgrades).
const LAST_RUN_STORAGE_KEY = "thr3shrActiveRunId";
const STATUS_POLL_INTERVAL_MS = 1500;
const VIDEO_PREVIEW_EXTS = new Set([
  ".mp4",
  ".m4v",
  ".webm",
  ".mov",
  ".mkv",
  ".avi",
  ".flv",
  ".wmv",
]);

function mediaExtFromPath(path) {
  if (!path || typeof path !== "string") return "";
  const clean = path.split(/[?#]/)[0];
  const idx = clean.lastIndexOf(".");
  return idx >= 0 ? clean.slice(idx).toLowerCase() : "";
}

function isVideoPreviewPath(path) {
  return VIDEO_PREVIEW_EXTS.has(mediaExtFromPath(path));
}

function settingsSnapshot(settings, selectedTags) {
  return JSON.stringify({
    ...settings,
    selected_tags: selectedTags,
  });
}

function sameList(a, b) {
  return a.length === b.length && a.every((value, idx) => value === b[idx]);
}

function formatTagScore(entry) {
  if (!entry) return "";
  return `${entry.tag} (${Number(entry.score).toFixed(3)})`;
}

function formatDurationSeconds(totalSeconds) {
  const secs = Math.max(0, Math.round(Number(totalSeconds) || 0));
  const hours = Math.floor(secs / 3600);
  const minutes = Math.floor((secs % 3600) / 60);
  const seconds = secs % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

function formatEtaFinishAt(iso) {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString(undefined, {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

// Number input that can be cleared and retyped. The text the user types is
// kept as a draft; valid in-range values commit as they are typed (so the
// unsaved-changes bar reacts), and on blur an empty or out-of-range draft is
// clamped or reverted instead of being coerced to 0 mid-edit.
function NumberField({ value, onCommit, min, max, step = 1, integer = false, ...inputProps }) {
  const [draft, setDraft] = useState(null);
  const normalize = (raw) => {
    const parsed = Number(raw);
    if (String(raw).trim() === "" || !Number.isFinite(parsed)) return null;
    return integer ? Math.round(parsed) : parsed;
  };
  return (
    <input
      {...inputProps}
      type="number"
      min={min}
      max={max}
      step={step}
      value={draft ?? String(value ?? "")}
      onChange={(e) => {
        const text = e.target.value;
        setDraft(text);
        const next = normalize(text);
        if (next !== null && next >= min && next <= max && next !== value) onCommit(next);
      }}
      onBlur={() => {
        if (draft === null) return;
        const next = normalize(draft);
        if (next !== null) {
          const clamped = Math.min(max, Math.max(min, next));
          if (clamped !== value) onCommit(clamped);
        }
        setDraft(null);
      }}
    />
  );
}

function readRoute() {
  const hash = window.location.hash.replace(/^#/, "") || "/";
  return hash.startsWith("/debug") ? "debug" : "home";
}

function pathTail(path) {
  const parts = String(path || "").split(/[\\/]+/).filter(Boolean);
  return parts.slice(-2).join("/") || String(path || "");
}

function formatRunDate(iso) {
  if (!iso) return "";
  // SQLite CURRENT_TIMESTAMP is UTC without a zone marker.
  const normalized = /[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso.replace(" ", "T")}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function runOptionLabel(run) {
  const progress = run.total_images
    ? `${run.processed_images}/${run.total_images}`
    : `${run.processed_images}`;
  const when = formatRunDate(run.created_at);
  return [`Run #${run.run_id}`, run.status, `${progress} images`, pathTail(run.root_repo), when]
    .filter(Boolean)
    .join(" · ");
}

function App() {
  const [route, setRoute] = useState(readRoute);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [savedSnapshot, setSavedSnapshot] = useState(
    settingsSnapshot(DEFAULT_SETTINGS, [])
  );
  const [offlineMode, setOfflineMode] = useState(false);
  const [backendBack, setBackendBack] = useState(false);
  const [runId, setRunId] = useState(null);
  const [runs, setRuns] = useState([]);
  const [runStatus, setRunStatus] = useState(null);
  const [runMeta, setRunMeta] = useState(null);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [providerInfo, setProviderInfo] = useState(null);
  const [realLifeStatus, setRealLifeStatus] = useState(null);
  const [realLifeStatusLoading, setRealLifeStatusLoading] = useState(false);
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
  const [pendingItemIds, setPendingItemIds] = useState({});
  const [scanPreview, setScanPreview] = useState(null);
  const [scanPreviewLoading, setScanPreviewLoading] = useState(false);
  const [opsLoading, setOpsLoading] = useState({
    saving: false,
    startingRun: false,
    applyingBatch: false,
    migrating: false,
    reclassifying: false,
    unloadingModels: false,
  });
  const [reclassifyModel, setReclassifyModel] = useState("wd_swinv2_v3");
  const itemUpdateQueueRef = useRef({});
  const pollTimerRef = useRef(null);
  // Bumped whenever polling stops or switches runs; stale responses check it.
  const pollGenerationRef = useRef(0);
  const viewedRunIdRef = useRef(null);
  const lastItemsRefreshAtRef = useRef(0);
  const lastItemsProcessedRef = useRef(-1);
  const tagsHydratedRef = useRef(false);
  const savedSettingsRef = useRef(null);
  const selectedTagsRef = useRef(selectedTags);
  selectedTagsRef.current = selectedTags;
  const skipNextTagPersistRef = useRef(false);

  useEffect(() => {
    const handleHashChange = () => setRoute(readRoute());
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const isDirty = useMemo(
    () => settingsSnapshot(settings, selectedTags) !== savedSnapshot,
    [settings, selectedTags, savedSnapshot]
  );

  const isWdModel = String(settings.tagger_model || "").startsWith("wd_");
  const runActive = Boolean(
    runStatus && ["pending", "running"].includes(runStatus.status)
  );
  const otherActiveRun = runs.find(
    (run) => run.run_id !== runId && ["pending", "running"].includes(run.status)
  );
  const anyRunActive = runActive || Boolean(otherActiveRun);

  async function refreshRuns() {
    try {
      const list = await api.listRuns();
      setRuns(Array.isArray(list) ? list : []);
    } catch (err) {
      console.error("refreshRuns failed", err);
    }
  }

  async function refreshItems(currentRunId, options = {}) {
    if (!currentRunId) return;
    try {
      const [runInfo, runItems] = await Promise.all([
        api.getRun(currentRunId),
        api.getRunItems(currentRunId, {
          timeoutMs: Number(options.timeoutMs) > 0 ? Number(options.timeoutMs) : 120000,
        }),
      ]);
      // The user may have opened another run while this request was in flight.
      if (viewedRunIdRef.current !== currentRunId) return;
      setRunMeta(runInfo);
      setItems(runItems);
      // Clear sticky thumb failures so a fixed preview endpoint can retry.
      setPreviewErrors({});
    } catch (err) {
      console.error("refreshItems failed", err);
      if (viewedRunIdRef.current !== currentRunId) return;
      setError(`Failed to refresh items: ${err.message}`);
    }
  }

  // Record what the server now holds; the dirty check compares against it.
  function rememberSavedSettings(saved, fallbackTags = []) {
    const merged = { ...DEFAULT_SETTINGS, ...saved };
    const tags = Array.isArray(saved?.selected_tags) ? saved.selected_tags : fallbackTags;
    savedSettingsRef.current = { ...merged, selected_tags: tags };
    setSavedSnapshot(settingsSnapshot(merged, tags));
    return { merged, tags };
  }

  useEffect(() => {
    api
      .getSettings()
      .then((data) => {
        const { merged, tags } = rememberSavedSettings(data);
        setSettings(merged);
        skipNextTagPersistRef.current = true;
        setSelectedTags(tags);
        tagsHydratedRef.current = true;
        setMigrateMode(data.default_migrate_mode || "copy");
        setOfflineMode(api.isOfflineMode());
      })
      .catch((err) => setError(err.message));
    const storedRunId = Number(localStorage.getItem(LAST_RUN_STORAGE_KEY) || 0);
    if (storedRunId > 0) {
      setRunId(storedRunId);
    }
    refreshRuns();
    api
      .getProviders()
      .then((info) => setProviderInfo(info))
      .catch(() => setProviderInfo(null));
  }, []);

  // Persist typed/selected tags (THR3SHR preference survival across reloads).
  useEffect(() => {
    if (!tagsHydratedRef.current) return;
    if (skipNextTagPersistRef.current) {
      skipNextTagPersistRef.current = false;
      return;
    }
    if (!Array.isArray(selectedTags)) return;
    const timer = setTimeout(() => {
      // Save only the tag list, on top of what the server already has. Other
      // fields may hold unsaved edits; they belong to the Save button, and a
      // stale copy of them must not overwrite what the user is typing.
      const sentTags = selectedTags;
      api
        .saveSettings({
          ...(savedSettingsRef.current || DEFAULT_SETTINGS),
          selected_tags: sentTags,
        })
        .then((saved) => {
          const { merged, tags } = rememberSavedSettings(saved, sentTags);
          setSettings((prev) => ({
            ...prev,
            selected_tags: tags,
            selected_tags_nsfw: merged.selected_tags_nsfw,
          }));
          // Adopt the server's canonical names unless the user changed the
          // selection while this save was in flight.
          if (selectedTagsRef.current === sentTags && !sameList(tags, sentTags)) {
            skipNextTagPersistRef.current = true;
            setSelectedTags(tags);
          }
        })
        .catch(() => {
          /* ignore transient save errors while typing; the dirty bar stays up */
        });
    }, 400);
    return () => clearTimeout(timer);
  }, [selectedTags]);

  // Demo mode sticks for the page's lifetime; watch for the backend returning
  // so the user can reconnect instead of working on mock data unawares.
  useEffect(() => {
    if (!offlineMode) return undefined;
    let cancelled = false;
    const timer = setInterval(async () => {
      if (await api.probeBackend()) {
        if (!cancelled) setBackendBack(true);
        clearInterval(timer);
      }
    }, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [offlineMode]);

  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      api
        .getTags(tagQuery, 50, settings.tagging_domain || "drawn")
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
  }, [tagQuery, settings.tagging_domain]);

  useEffect(() => {
    if (settings.tagging_domain !== "real_life" || offlineMode) {
      setRealLifeStatus(null);
      return;
    }
    let cancelled = false;
    setRealLifeStatusLoading(true);
    api
      .getRealLifeStatus()
      .then((data) => {
        if (!cancelled) setRealLifeStatus(data);
      })
      .catch(() => {
        if (!cancelled) setRealLifeStatus(null);
      })
      .finally(() => {
        if (!cancelled) setRealLifeStatusLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [settings.tagging_domain, offlineMode]);

  useEffect(() => {
    return () => stopStatusPolling();
  }, []);

  const stats = useMemo(() => {
    const fromList = {
      total: items.length,
      reviewNeeded: items.filter((x) => x.needs_review).length,
      approved: items.filter((x) => x.status === "approved").length,
      rejected: items.filter((x) => x.status === "rejected").length,
      migrated: items.filter((x) => x.status === "migrated").length,
    };
    // Prefer server status counts when present — list refresh can lag on big runs.
    const counts = Array.isArray(runMeta?.counts) ? runMeta.counts : [];
    if (!counts.length) return fromList;
    const byStatus = Object.fromEntries(
      counts.map((row) => [row.status, Number(row.count) || 0])
    );
    return {
      ...fromList,
      approved: byStatus.approved ?? fromList.approved,
      rejected: byStatus.rejected ?? fromList.rejected,
      migrated: byStatus.migrated ?? fromList.migrated,
    };
  }, [items, runMeta]);

  function forgetMissingRun(missingRunId) {
    stopStatusPolling();
    // Demo mode's mock store knows none of the real runs; keep the remembered
    // run so reconnecting to the backend reopens it.
    const offline = api.isOfflineMode();
    if (!offline && Number(localStorage.getItem(LAST_RUN_STORAGE_KEY) || 0) === missingRunId) {
      localStorage.removeItem(LAST_RUN_STORAGE_KEY);
    }
    viewedRunIdRef.current = null;
    setRunId(null);
    setRunStatus(null);
    setRunMeta(null);
    setItems([]);
    setSelectedIds([]);
    setError((prev) =>
      String(prev || "").startsWith("Failed to poll run status:") ? "" : prev
    );
    setNotice(
      offline
        ? `Run #${missingRunId} is not available in demo mode.`
        : `Run #${missingRunId} no longer exists on the backend.`
    );
    refreshRuns();
  }

  // Returns whether polling should continue for this run.
  const selectedIdSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const selectableIds = useMemo(
    () => items.filter((item) => item.status !== "migrated").map((item) => item.id),
    [items]
  );
  const allSelectableSelected =
    selectableIds.length > 0 && selectableIds.every((id) => selectedIdSet.has(id));
  const someSelectableSelected = selectableIds.some((id) => selectedIdSet.has(id));

  // Rows that just migrated (or left the list) drop out of the selection.
  useEffect(() => {
    const allowed = new Set(selectableIds);
    setSelectedIds((prev) => {
      const next = prev.filter((id) => allowed.has(id));
      return next.length === prev.length ? prev : next;
    });
  }, [selectableIds]);

  async function pollRunStatus(currentRunId, generation) {
    const isStale = () =>
      generation !== pollGenerationRef.current || viewedRunIdRef.current !== currentRunId;
    try {
      const status = await api.getRunStatus(currentRunId);
      if (isStale()) return false;
      setRunStatus(status);
      setError((prev) =>
        String(prev || "").startsWith("Failed to poll run status:") ? "" : prev
      );

      const terminal = ["completed", "cancelled", "failed"].includes(
        status.status
      );
      // Full item lists get heavy on large runs. Refresh when progress moves,
      // especially early (first refresh often hits 0 items during scan).
      const now = Date.now();
      const processed = Number(status.processed_images || 0);
      const sinceRefresh = now - lastItemsRefreshAtRef.current;
      const processedAdvanced = processed > lastItemsProcessedRef.current;
      const emptyTable = lastItemsProcessedRef.current <= 0;
      const earlyIntervalMs = emptyTable || processed < 80 ? 3000 : 20000;
      const dueForItems =
        terminal ||
        lastItemsRefreshAtRef.current === 0 ||
        (processedAdvanced && sinceRefresh >= earlyIntervalMs) ||
        sinceRefresh >= 20000;
      if (dueForItems) {
        lastItemsRefreshAtRef.current = now;
        lastItemsProcessedRef.current = processed;
        await refreshItems(currentRunId);
      }

      if (terminal) {
        refreshRuns();
        api.getProviders().then((info) => setProviderInfo(info)).catch(() => {});
        return false;
      }
      return true;
    } catch (err) {
      if (isStale()) return false;
      if (err?.status === 404) {
        forgetMissingRun(currentRunId);
        return false;
      }
      // Keep polling through transient API/load blips; a hard stop made the
      // UI look frozen while the backend was still classifying.
      setError(`Failed to poll run status: ${err.message}`);
      return true;
    }
  }

  function stopStatusPolling() {
    pollGenerationRef.current += 1;
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }

  // Chained timeouts instead of setInterval: a slow item refresh on a large run
  // must finish before the next poll, or requests pile up on the backend.
  function startStatusPolling(currentRunId) {
    stopStatusPolling();
    const generation = pollGenerationRef.current;
    const tick = async () => {
      const keepPolling = await pollRunStatus(currentRunId, generation);
      if (!keepPolling || generation !== pollGenerationRef.current) return;
      pollTimerRef.current = setTimeout(tick, STATUS_POLL_INTERVAL_MS);
    };
    tick();
  }

  useEffect(() => {
    viewedRunIdRef.current = runId;
    if (!runId) return;
    // Mock run ids would collide with real ones after reconnecting.
    if (!api.isOfflineMode()) {
      localStorage.setItem(LAST_RUN_STORAGE_KEY, String(runId));
    }
    // Drop sticky thumb failures from prior preview bugs / aborted video loads.
    setPreviewErrors({});
    lastItemsRefreshAtRef.current = 0;
    lastItemsProcessedRef.current = -1;
    startStatusPolling(runId);
    return () => stopStatusPolling();
  }, [runId]);

  useEffect(() => {
    if (!otherActiveRun) return;
    const timer = setInterval(refreshRuns, STATUS_POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [otherActiveRun?.run_id]);

  function openRun(nextRunId) {
    if (!nextRunId || nextRunId === runId) return;
    // Retire the old run's in-flight poll before React re-renders.
    stopStatusPolling();
    viewedRunIdRef.current = nextRunId;
    setError("");
    setNotice("");
    setRunStatus(null);
    setRunMeta(null);
    setItems([]);
    setSelectedIds([]);
    setRunId(nextRunId);
  }

  async function handleSaveSettings(e) {
    if (e) e.preventDefault();
    setLoading(true);
    setOpsLoading((prev) => ({ ...prev, saving: true }));
    setError("");
    try {
      const saved = await api.saveSettings({ ...settings, selected_tags: selectedTags });
      const { merged, tags } = rememberSavedSettings(saved, selectedTags);
      setSettings(merged);
      if (!sameList(tags, selectedTags)) {
        skipNextTagPersistRef.current = true;
        setSelectedTags(tags);
      }
      setOfflineMode(api.isOfflineMode());
      const providers = await api.getProviders();
      setProviderInfo(providers);
      return true;
    } catch (err) {
      setError(`Failed to save settings: ${err.message}`);
      return false;
    } finally {
      setLoading(false);
      setOpsLoading((prev) => ({ ...prev, saving: false }));
    }
  }

  async function handlePreviewScan() {
    setScanPreviewLoading(true);
    setError("");
    try {
      if (isDirty && !(await handleSaveSettings())) {
        setScanPreview(null);
        return;
      }
      const preview = await api.previewScan();
      setScanPreview(preview);
    } catch (err) {
      setError(err.message);
      setScanPreview(null);
    } finally {
      setScanPreviewLoading(false);
    }
  }

  async function handleStartRun(runMode = "classify") {
    setLoading(true);
    setOpsLoading((prev) => ({ ...prev, startingRun: true }));
    setError("");
    try {
      // A failed save would otherwise start a run on the old settings.
      if (isDirty && !(await handleSaveSettings())) return;
      const doujinFolders = [
        "loli",
        "shota",
        "milf",
        "fertilization",
        "monster_girl",
        "incest",
        "bestiality",
        "Pokemon",
        "NTR",
        "tentacles",
        "furry",
        "android",
      ];
      const domain = settings.tagging_domain === "real_life" ? "real_life" : "drawn";
      const effectiveMode =
        runMode === "classify" && domain === "real_life" ? "real_life_tag" : runMode;
      const result = await api.startRun({
        ...settings,
        selected_folders:
          effectiveMode === "real_life_filter"
            ? ["real_life"]
            : effectiveMode === "doujin_works"
              ? doujinFolders
              : effectiveMode === "real_life_tag"
                ? selectedTags.length > 0
                  ? selectedTags
                  : REAL_LIFE_DEFAULT_FOLDERS
                : selectedTags.length > 0
                  ? selectedTags
                  : null,
        run_mode: effectiveMode,
        // Filter / real-life tagging always enables GIF/video scanning server-side.
        experimental_media_enabled:
          effectiveMode === "real_life_filter" || effectiveMode === "real_life_tag"
            ? true
            : settings.experimental_media_enabled,
      });
      if (runMode === "doujin_works") {
        setMigrateMode("move");
      }
      setOfflineMode(api.isOfflineMode());
      stopStatusPolling();
      viewedRunIdRef.current = result.run_id;
      setSelectedIds([]);
      setItems([]);
      setRunMeta(null);
      setRunId(result.run_id);
      refreshRuns();
      setPreviewErrors({});
      setRunStatus({
        run_id: result.run_id,
        status: result.status || "pending",
        total_images: 0,
        processed_images: 0,
        failed_images: 0,
        progress_pct: 0,
        cancel_requested: false,
        tagger_model: settings.tagger_model,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setOpsLoading((prev) => ({ ...prev, startingRun: false }));
    }
  }

  // Updates to one item run in order: blurring the Final tag field commits the
  // tag just before the Approve click lands, and the approve must not race it.
  function enqueueItemUpdate(itemId, payload) {
    const previous = itemUpdateQueueRef.current[itemId] || Promise.resolve();
    const next = previous
      .catch(() => {})
      .then(() => api.updateItem(itemId, payload));
    itemUpdateQueueRef.current[itemId] = next;
    return next;
  }

  function settleItemUpdates(itemIds) {
    return Promise.allSettled(itemIds.map((id) => itemUpdateQueueRef.current[id]));
  }

  async function updateStatus(itemId, status) {
    setPendingItemIds((prev) => ({ ...prev, [itemId]: true }));
    setError("");
    try {
      await enqueueItemUpdate(itemId, { status });
      await refreshItems(runId);
    } catch (err) {
      console.error("updateStatus failed", err);
      setError(`Failed to update status: ${err.message}`);
    } finally {
      setPendingItemIds((prev) => {
        const { [itemId]: _done, ...rest } = prev;
        return rest;
      });
    }
  }

  function setFinalTagDraft(itemId, value) {
    setFinalTagDrafts((prev) => ({ ...prev, [itemId]: value }));
  }

  function dropFinalTagDraft(itemId) {
    setFinalTagDrafts((prev) => {
      const { [itemId]: _dropped, ...rest } = prev;
      return rest;
    });
  }

  // Commit on blur / Enter rather than per keystroke: a pause mid-word used to
  // save (and approve into) a partial folder name such as "scen".
  async function commitFinalTag(item) {
    const draft = finalTagDrafts[item.id];
    if (draft === undefined) return;
    const nextTag = draft.trim();
    if (!nextTag || nextTag === (item.final_tag || "")) {
      dropFinalTagDraft(item.id);
      return;
    }
    // Retagging keeps an approved item approved (now aimed at the new folder);
    // only untouched proposals move to reviewed.
    const payload = { final_tag: nextTag };
    if (item.status === "proposed") payload.status = "reviewed";
    setError("");
    try {
      await enqueueItemUpdate(item.id, payload);
      await refreshItems(runId);
      dropFinalTagDraft(item.id);
    } catch (err) {
      console.error("commitFinalTag failed", err);
      setError(`Failed to update final tag: ${err.message}`);
    }
  }

  async function applyBatch(status) {
    if (selectedIds.length === 0) return;
    if (status === "rejected") {
      const confirmed = window.confirm(
        `Reject ${selectedIds.length} selected item(s)? They stay on disk and are skipped by migrate/reclassify.`
      );
      if (!confirmed) return;
    }
    setOpsLoading((prev) => ({ ...prev, applyingBatch: true }));
    setError("");
    setNotice("");
    try {
      await settleItemUpdates(selectedIds);
      const result = await api.batchUpdate(runId, { item_ids: selectedIds, status });
      setSelectedIds([]);
      await refreshItems(runId);
      const skipped = Number(result?.skipped || 0);
      if (skipped > 0) {
        setNotice(
          `${Number(result?.updated || 0)} item(s) ${status}; ${skipped} skipped ` +
            (status === "approved"
              ? "(already migrated, no destination tag yet, or a sensitive tag that has to be typed as the Final tag first)."
              : "(already migrated).")
        );
      }
    } catch (err) {
      console.error("applyBatch failed", err);
      setError(`Batch update failed: ${err.message}`);
    } finally {
      setOpsLoading((prev) => ({ ...prev, applyingBatch: false }));
    }
  }

  async function migrateApproved() {
    setError("");
    setNotice("");
    const confirmed = window.confirm(
      `Migrate ${stats.approved} approved item(s) using ${migrateMode}? This can move/copy many files.`
    );
    if (!confirmed) return;
    setLoading(true);
    setOpsLoading((prev) => ({ ...prev, migrating: true }));
    try {
      await settleItemUpdates(Object.keys(itemUpdateQueueRef.current));
      const result = await api.migrateRun(runId, {
        mode: migrateMode,
        create_missing_folders: true,
      });
      setOfflineMode(api.isOfflineMode());
      const failed = Number(result?.failed_count || 0);
      const moved = Number(result?.migrated_count || 0);
      const candidates = Number(result?.total_candidates || 0);
      // Immediate UI update so Migrated/Approved counts don't wait on a huge refresh.
      if (moved > 0) {
        setItems((prev) =>
          prev.map((item) =>
            item.status === "approved" ? { ...item, status: "migrated" } : item
          )
        );
      }
      await refreshItems(runId, { timeoutMs: 180000 });
      if (failed > 0) {
        const sample = (result?.results || [])
          .filter((r) => !r.success && r.error)
          .slice(0, 3)
          .map((r) => r.error)
          .join("; ");
        setError(
          `Migrate finished: ${moved} ok, ${failed} failed${sample ? ` (${sample})` : ""}`
        );
      } else if (candidates === 0) {
        setNotice(
          "Nothing to migrate — no approved items left (already migrated or still in review)."
        );
      } else {
        setNotice(`Migrated ${moved} file(s) successfully (${migrateMode}).`);
      }
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
      // Cancel stops classifying but keeps already-tagged rows for review.
      await refreshItems(runId);
      setNotice(
        "Run cancelled — already classified images are still available for review."
      );
    } catch (err) {
      setError(`Failed to cancel run: ${err.message}`);
    }
  }

  async function handleUnloadModels() {
    if (anyRunActive || opsLoading.unloadingModels) return;
    setOpsLoading((prev) => ({ ...prev, unloadingModels: true }));
    setError("");
    try {
      const result = await api.unloadModels();
      const providers = await api.getProviders();
      setProviderInfo(providers);
      setNotice(result.message || "GPU models unloaded.");
    } catch (err) {
      setError(`Failed to unload GPU models: ${err.message}`);
    } finally {
      setOpsLoading((prev) => ({ ...prev, unloadingModels: false }));
    }
  }

  useEffect(() => {
    // Default reclassify to SwinV2 — EVA02 is accurate but ~10–20s/image on GPU.
    setReclassifyModel("wd_swinv2_v3");
  }, [runStatus?.tagger_model, settings.tagger_model]);

  async function handleReclassifyNeedsReview() {
    if (!runId || anyRunActive) return;
    setOpsLoading((prev) => ({ ...prev, reclassifying: true }));
    setError("");
    try {
      const result = await api.reclassifyRun(runId, {
        tagger_model: reclassifyModel,
      });
      setRunStatus((prev) => ({
        ...(prev || {}),
        run_id: runId,
        status: result.status || "running",
        cancel_requested: false,
        last_error: null,
      }));
      startStatusPolling(runId);
    } catch (err) {
      setError(`Failed to reclassify needs-review items: ${err.message}`);
    } finally {
      setOpsLoading((prev) => ({ ...prev, reclassifying: false }));
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

  const sfwClassifyMode =
    Boolean(settings.sfw_classify_mode) && settings.tagging_domain !== "real_life";
  const parkedNsfwTags = Array.isArray(settings.selected_tags_nsfw)
    ? settings.selected_tags_nsfw
    : [];

  async function setSfwClassifyMode(enabled) {
    if (settings.tagging_domain === "real_life") return;
    setError("");
    const unparked = selectedTags.filter((t) => !SFW_CLASSIFY_FOLDERS.includes(t));
    const nextTags = enabled
      ? [...SFW_CLASSIFY_FOLDERS]
      : parkedNsfwTags.length > 0
        ? parkedNsfwTags
        : unparked;
    const change = {
      sfw_classify_mode: enabled,
      selected_tags: nextTags,
      selected_tags_nsfw: enabled ? (unparked.length ? unparked : parkedNsfwTags) : nextTags,
    };
    const previous = {
      sfw_classify_mode: settings.sfw_classify_mode,
      selected_tags: selectedTags,
      selected_tags_nsfw: parkedNsfwTags,
    };
    skipNextTagPersistRef.current = true;
    setSelectedTags(nextTags);
    setSettings((prev) => ({ ...prev, ...change }));
    try {
      // Save only the SFW fields on top of the saved settings; unrelated
      // unsaved edits stay pending for the Save button.
      const saved = await api.saveSettings({
        ...(savedSettingsRef.current || DEFAULT_SETTINGS),
        ...change,
      });
      const { merged, tags } = rememberSavedSettings(saved, nextTags);
      setSettings((prev) => ({
        ...prev,
        sfw_classify_mode: merged.sfw_classify_mode,
        selected_tags: tags,
        selected_tags_nsfw: merged.selected_tags_nsfw,
      }));
      if (!sameList(tags, nextTags)) {
        skipNextTagPersistRef.current = true;
        setSelectedTags(tags);
      }
    } catch (err) {
      skipNextTagPersistRef.current = true;
      setSelectedTags(previous.selected_tags);
      setSettings((prev) => ({ ...prev, ...previous }));
      setError(`Failed to ${enabled ? "enable" : "disable"} SFW mode: ${err.message}`);
    }
  }

  function addSelectedTag(value) {
    if (!value || sfwClassifyMode) return;
    setSelectedTags((prev) => (prev.includes(value) ? prev : [...prev, value]));
  }

  async function addValidatedTag(rawValue) {
    const value = rawValue.trim();
    if (!value) return;
    const already = selectedTags.some((t) => t.toLowerCase() === value.toLowerCase());
    if (already) return;

    const fromOptions = tagOptions.find((t) => t.toLowerCase() === value.toLowerCase());
    if (fromOptions) {
      addSelectedTag(fromOptions);
      return;
    }

    const result = await api.getTags(value, 200, settings.tagging_domain || "drawn");
    const items = result.items || [];
    const match =
      items.find((t) => t === value) ||
      items.find((t) => t.toLowerCase() === value.toLowerCase());
    if (match) {
      addSelectedTag(match);
      return;
    }
    throw new Error(
      settings.tagging_domain === "real_life"
        ? `Unknown real-life category: ${value}. Try creampie, cumshot, oral, BBC…`
        : `Unknown destination: ${value}. Try a taxonomy folder (e.g. Voyeur) or a tags.csv tag.`
    );
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
    if (sfwClassifyMode) return;
    setSelectedTags((prev) => prev.filter((t) => t !== value));
  }

  if (route === "debug") {
    return <ClassifierDebugPage />;
  }

  return (
    <div className="container">
      <header className="app-header">
        <div className="app-nav">
          <a href="#/" className="nav-link active" aria-current="page">
            THR3SHR
          </a>
          <a href="#/debug" className="nav-link">
            Debug
          </a>
        </div>
        <h1>THR3SHR</h1>
        <p className="lede">
          Thresh a media dump into destination bins — configure paths and tagger, pick tags, run
          inference, then review and migrate.
        </p>
        {providerInfo && !offlineMode && (
          <div className="provider-strip">
            <span>
              Device: <strong>{providerInfo.likely_device || "unknown"}</strong>
            </span>
            <span>
              CUDA: <strong>{String(Boolean(providerInfo.cuda_usable))}</strong>
            </span>
            <span>
              Active: <strong>{(providerInfo.active_providers || []).join(", ") || "n/a"}</strong>
            </span>
            <span>
              Tagger: <strong>{providerInfo.tagger_model || settings.tagger_model}</strong>
            </span>
            <span>
              Forced CPU: <strong>{String(Boolean(providerInfo.forced_cpu))}</strong>
            </span>
            <span>
              Loaded:{" "}
              <strong>
                {(providerInfo.loaded_models || []).length
                  ? (providerInfo.loaded_models || []).join(", ")
                  : "none"}
              </strong>
            </span>
            <button
              type="button"
              className="unload-models-btn"
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
        )}
      </header>

      {offlineMode && (
        <div className={backendBack ? "notice offline-banner" : "error offline-banner"} role="status">
          {backendBack ? (
            <span>
              The backend is reachable again. Reconnect to leave demo mode and load your real
              settings and runs.
            </span>
          ) : (
            <span>
              Backend is offline. Running in browser-only demo mode with local mock data; nothing
              here touches your files.
            </span>
          )}
          {backendBack ? (
            <button type="button" onClick={() => window.location.reload()}>
              Reconnect
            </button>
          ) : null}
        </div>
      )}
      {providerInfo?.provider_error && !offlineMode && (
        <div className="error">{providerInfo.provider_error}</div>
      )}
      {error && <div className="error">{error}</div>}
      {notice && <div className="notice">{notice}</div>}

      <section className="panel">
        <h2>Settings</h2>
        <span className="kicker">Paths, model, thresholds, and performance. Save before runs.</span>
        <form onSubmit={handleSaveSettings}>
          <fieldset className="settings-section">
            <legend>Paths &amp; scan</legend>
            <div className="grid">
              <label>
                Root repository
                <input
                  value={settings.root_repo}
                  onChange={(e) => setSettings({ ...settings, root_repo: e.target.value })}
                  placeholder="Folder of unsorted images"
                />
              </label>
              <label>
                Categories root
                <input
                  value={settings.categories_root}
                  onChange={(e) => setSettings({ ...settings, categories_root: e.target.value })}
                  placeholder="Destination folders root (not tags.csv)"
                />
              </label>
              <label>
                Default migrate mode
                <select
                  value={settings.default_migrate_mode}
                  onChange={(e) =>
                    setSettings({ ...settings, default_migrate_mode: e.target.value })
                  }
                >
                  <option value="copy">copy</option>
                  <option value="move">move</option>
                </select>
              </label>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={Boolean(settings.scan_recursive)}
                  onChange={(e) => setSettings({ ...settings, scan_recursive: e.target.checked })}
                />
                Scan subfolders recursively
              </label>
            </div>
            <div className="actions">
              <button
                type="button"
                className="secondary"
                disabled={scanPreviewLoading}
                onClick={handlePreviewScan}
              >
                {scanPreviewLoading ? "Scanning…" : "Preview scan"}
              </button>
            </div>
            {scanPreview?.stats && (
              <div className="scan-preview" role="status">
                Eligible: <strong>{scanPreview.stats.eligible_images}</strong> · Total files:{" "}
                {scanPreview.stats.total_files} · Ignored GIF: {scanPreview.stats.ignored_gif} ·
                Unsupported: {scanPreview.stats.ignored_unsupported} · Unreadable:{" "}
                {scanPreview.stats.failed_to_read}
                {scanPreview.recursive ? "" : " · top folder only"}
              </div>
            )}
          </fieldset>

          <fieldset className="settings-section">
            <legend>Tagger model</legend>
            <div className="model-options">
              {TAGGER_MODELS.map((model) => (
                <label key={model.id} className="model-option">
                  <input
                    type="radio"
                    name="tagger_model"
                    value={model.id}
                    checked={settings.tagger_model === model.id}
                    onChange={() => setSettings({ ...settings, tagger_model: model.id })}
                  />
                  <span>
                    <strong>{model.label}</strong>
                    <span>{model.help}</span>
                  </span>
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="settings-section">
            <legend>Thresholds</legend>
            <div className="grid">
              <label>
                Assignment confidence
                <NumberField
                  step={0.01}
                  min={0}
                  max={1}
                  value={settings.confidence_threshold}
                  onCommit={(next) =>
                    setSettings((prev) => ({ ...prev, confidence_threshold: next }))
                  }
                />
                <span className="help">
                  Selected-tag scores below this are suggestions only (no primary assignment).
                </span>
              </label>
              {isWdModel && (
                <label>
                  WD general threshold
                  <NumberField
                    step={0.01}
                    min={0}
                    max={1}
                    value={settings.wd_general_threshold}
                    onCommit={(next) =>
                      setSettings((prev) => ({ ...prev, wd_general_threshold: next }))
                    }
                  />
                  <span className="help">
                    WD14 inference cutoff for general tags (default 0.35).
                  </span>
                </label>
              )}
            </div>
          </fieldset>

          <fieldset className="settings-section">
            <legend>Performance</legend>
            <div className="grid">
              <label>
                Max inference workers
                <NumberField
                  min={1}
                  max={16}
                  integer
                  value={settings.max_inference_workers ?? 2}
                  onCommit={(next) =>
                    setSettings((prev) => ({ ...prev, max_inference_workers: next }))
                  }
                />
                <span className="help">Keep at 2 for GPU ORT sessions (serialized lock).</span>
              </label>
              <label>
                Inference batch size
                <NumberField
                  min={1}
                  max={64}
                  integer
                  value={settings.inference_batch_size ?? 8}
                  onCommit={(next) =>
                    setSettings((prev) => ({ ...prev, inference_batch_size: next }))
                  }
                />
                <span className="help">
                  WD models: 4–8 is a good GPU default. 1 forces single-image mode.
                </span>
              </label>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={Boolean(settings.force_cpu_inference)}
                  onChange={(e) => {
                    const { checked } = e.target;
                    setSettings((prev) => ({ ...prev, force_cpu_inference: checked }));
                  }}
                />
                <span>Force CPU inference</span>
              </label>
            </div>
          </fieldset>

          <fieldset className="settings-section">
            <legend>Classification</legend>
            <div className="check-list">
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={Boolean(settings.experimental_media_enabled)}
                  onChange={(e) => {
                    const { checked } = e.target;
                    setSettings((prev) => ({ ...prev, experimental_media_enabled: checked }));
                  }}
                />
                <span>Experimental: classify GIF/videos via length-scaled multi-frame sampling</span>
              </label>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={Boolean(settings.experimental_style_detector_enabled)}
                  onChange={(e) => {
                    const { checked } = e.target;
                    setSettings((prev) => ({
                      ...prev,
                      experimental_style_detector_enabled: checked,
                    }));
                  }}
                />
                <span>
                  Experimental: real vs anime style gate (dedicated ONNX →{" "}
                  <code>real_life</code>; uncertain → needs review)
                </span>
              </label>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={Boolean(settings.hybrid_ml_on_review)}
                  onChange={(e) => {
                    const { checked } = e.target;
                    setSettings((prev) => ({ ...prev, hybrid_ml_on_review: checked }));
                  }}
                  disabled={settings.tagging_domain === "real_life"}
                />
                <span>
                  Hybrid: on needs-review (WD primary), re-run ML-Danbooru and merge allowlisted
                  high-recall tags only (skips Voyeur soft cues)
                </span>
              </label>
            </div>
            <fieldset className="domain-toggle">
              <legend>Tagging domain</legend>
              <label className="inline-check">
                <input
                  type="radio"
                  name="tagging_domain"
                  checked={settings.tagging_domain !== "real_life"}
                  onChange={() =>
                    setSettings((prev) => ({ ...prev, tagging_domain: "drawn" }))
                  }
                />
                <span>Animated / drawn (WD + ML-Danbooru taxonomy)</span>
              </label>
              <label className="inline-check">
                <input
                  type="radio"
                  name="tagging_domain"
                  checked={settings.tagging_domain === "real_life"}
                  onChange={() =>
                    setSettings({
                      ...settings,
                      tagging_domain: "real_life",
                      experimental_media_enabled: true,
                      sfw_classify_mode: false,
                    })
                  }
                />
                <span>Real-life photos / videos (local adult tagger)</span>
              </label>
              <span className="help">
                Real-life mode uses an isolated taxonomy under{" "}
                <code>Real Life/&lt;primary&gt;/</code>. Sensitive tags (BBC /
                Ebony / Asian) are suggestions only and need explicit Final-tag
                confirmation. Save settings after switching domains.
              </span>
              {settings.tagging_domain === "real_life" && !offlineMode ? (
                <div className="scan-preview real-life-status">
                  {realLifeStatusLoading ? (
                    <span className="muted">Checking local adult tagger…</span>
                  ) : realLifeStatus?.engine ? (
                    <>
                      <div>
                        VLM:{" "}
                        <strong>
                          {realLifeStatus.engine.vlm?.available ? "ready" : "unavailable"}
                        </strong>
                        {realLifeStatus.engine.vlm?.backend
                          ? ` (${realLifeStatus.engine.vlm.backend})`
                          : ""}
                      </div>
                      <div>
                        Position classifier:{" "}
                        <strong>
                          {realLifeStatus.engine.position?.available
                            ? "ready"
                            : "unavailable"}
                        </strong>
                      </div>
                      <div className="help">{realLifeStatus.engine.message}</div>
                      {realLifeStatus.engine.fallback ? (
                        <div className="help">
                          Fallback active: style gate + manual review only until{" "}
                          <code>llama-cpp-python</code> and the Qwen2.5-VL GGUF +
                          mmproj are installed.
                        </div>
                      ) : null}
                    </>
                  ) : (
                    <span className="muted">Real-life tagger status unavailable.</span>
                  )}
                </div>
              ) : null}
            </fieldset>
          </fieldset>

          <div className="sticky-save">
            <span className={isDirty ? "dirty" : "clean"}>
              {isDirty ? "Unsaved settings changes" : "Settings saved"}
            </span>
            <button type="submit" disabled={loading || opsLoading.saving || !isDirty}>
              {opsLoading.saving ? "Saving…" : "Save settings"}
            </button>
          </div>
        </form>
      </section>

      <section className="panel">
        <h2>Tag selection</h2>
        <span className="kicker">
          {settings.tagging_domain === "real_life"
            ? "Real-life categories only (isolated from anime tags.csv). Changes auto-save after settings save."
            : sfwClassifyMode
              ? "SFW mode: only SFW / scenery compete. Your NSFW tags stay parked."
              : "Only these tags compete for folder assignment. Changes auto-save."}
        </span>
        {settings.tagging_domain !== "real_life" ? (
          <label className="inline-check sfw-mode-toggle">
            <input
              type="checkbox"
              checked={sfwClassifyMode}
              disabled={runActive || opsLoading.startingRun}
              onChange={(e) => setSfwClassifyMode(e.target.checked)}
            />
            SFW classify mode (parks NSFW tags; restore when off)
          </label>
        ) : null}
        <div className="tag-search-row">
          <label>
            {settings.tagging_domain === "real_life"
              ? "Real-life category search"
              : "Tag match search (destinations + tags.csv)"}
            <input
              value={tagQuery}
              onChange={(e) => setTagQuery(e.target.value)}
              list="tag-match-suggestions"
              autoComplete="off"
              disabled={sfwClassifyMode}
              onKeyDown={async (e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  if (sfwClassifyMode) return;
                  setError("");
                  try {
                    await addTagsFromInput(tagQuery);
                    setTagQuery("");
                  } catch (err) {
                    setError(err.message);
                  }
                }
              }}
              placeholder={
                sfwClassifyMode
                  ? "Tag editing disabled in SFW mode"
                  : settings.tagging_domain === "real_life"
                    ? "Type creampie, BBC, hotwife…"
                    : "Type Voyeur, loli, Pokemon…"
              }
            />
            <datalist id="tag-match-suggestions">
              {tagOptions.map((tag) => (
                <option key={tag} value={tag} />
              ))}
            </datalist>
          </label>
          <button
            type="button"
            className="secondary"
            disabled={sfwClassifyMode}
            onClick={async () => {
              setError("");
              try {
                await addTagsFromInput(tagQuery);
              } catch (err) {
                setError(err.message);
              }
            }}
          >
            Add tag(s)
          </button>
        </div>
        {tagQuery.trim() && !sfwClassifyMode && (
          <div className="tag-suggestions">
            {tagOptions.length === 0 ? (
              <span className="muted">No matching tags</span>
            ) : (
              tagOptions.slice(0, 30).map((tag) => (
                <button
                  key={tag}
                  type="button"
                  className="tag-suggestion"
                  onClick={() => {
                    addSelectedTag(tag);
                    setTagQuery("");
                    setError("");
                  }}
                >
                  {tag}
                </button>
              ))
            )}
          </div>
        )}
        <div className={`stats${sfwClassifyMode ? " tags-locked" : ""}`}>
          {selectedTags.length === 0 ? (
            <span className="muted">No tags selected</span>
          ) : (
            selectedTags.map((tag) => (
              <span key={tag} className="chip">
                {tag}
                <button
                  type="button"
                  onClick={() => removeSelectedTag(tag)}
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
            <span className="help">Parked NSFW tags (inactive until SFW mode is off):</span>
            <div className="stats tags-locked">
              {parkedNsfwTags.map((tag) => (
                <span key={`parked-${tag}`} className="chip chip-parked">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ) : null}
        <div className="actions run-actions">
          <button
            type="button"
            disabled={loading || opsLoading.startingRun || anyRunActive}
            onClick={() => handleStartRun("classify")}
          >
            {opsLoading.startingRun
              ? "Starting…"
              : settings.tagging_domain === "real_life"
                ? "Start real-life tagging"
                : "Start run"}
          </button>
          <button
            type="button"
            className="secondary"
            title="Scan root (including GIF/video). Keep only confident real_life hits — nothing else."
            disabled={
              loading ||
              opsLoading.startingRun ||
              anyRunActive ||
              settings.tagging_domain === "real_life"
            }
            onClick={() => handleStartRun("real_life_filter")}
          >
            {opsLoading.startingRun ? "Starting…" : "Filter real-life only"}
          </button>
          <button
            type="button"
            className="secondary"
            title="Treat each child folder or top-level .cbz as one work. Sample pages → WD → primary + category tags. Migrate moves into Doujins/<tag>/ with junctions."
            disabled={
              loading ||
              opsLoading.startingRun ||
              anyRunActive ||
              settings.tagging_domain === "real_life"
            }
            onClick={() => handleStartRun("doujin_works")}
          >
            {opsLoading.startingRun ? "Starting…" : "Start Doujin works"}
          </button>
        </div>
        {otherActiveRun && (
          <p className="help run-blocked" role="status">
            Run #{otherActiveRun.run_id} is still {otherActiveRun.status}. Wait for it to
            finish or cancel it before starting another run.{" "}
            <button
              type="button"
              className="secondary"
              onClick={() => openRun(otherActiveRun.run_id)}
            >
              Open run #{otherActiveRun.run_id}
            </button>
          </p>
        )}
        <p className="help">
          {settings.tagging_domain === "real_life" ? (
            <>
              Real-life tagging: style gate → local adult VLM (+ optional position
              classifier) → isolated categories under <code>Real Life/</code>. All
              items start in needs-review. Sensitive tags require typing them into
              Final tag before approve.
            </>
          ) : (
            <>
              Filter real-life only: style-first hybrid (skip WD on clear anime), capped GIF/video
              frames, stills stay batched. Only blended <code>real_life</code> hits are kept.
              Doujin works: one review row per title folder/cbz; favourites loli / shota / milf /
              fertilization / monster_girl / incest / bestiality / Pokemon / NTR / tentacles /
              furry / android; approve then migrate (move + tag junctions).
            </>
          )}
        </p>
      </section>

      {runs.length > 0 && (
        <section className="panel run-history" aria-labelledby="run-history-heading">
          <h2 id="run-history-heading">Runs</h2>
          <span className="kicker">Reopen an earlier run to review or migrate its results.</span>
          <label className="run-picker">
            Open run
            <select
              value={runId ?? ""}
              onChange={(e) => openRun(Number(e.target.value))}
            >
              {runId == null && <option value="">Select a run…</option>}
              {runId != null && !runs.some((run) => run.run_id === runId) && (
                <option value={runId}>Run #{runId}</option>
              )}
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {runOptionLabel(run)}
                </option>
              ))}
            </select>
          </label>
        </section>
      )}

      {runId && runStatus && (
        <section className="panel">
          <h2>Run dashboard</h2>
          <span className="kicker">
            Run #{runId}
            {runStatus.tagger_model ? ` · ${runStatus.tagger_model}` : ""}
          </span>
          {(() => {
            const etaLabel =
              runActive && runStatus.eta_seconds_remaining != null
                ? formatDurationSeconds(runStatus.eta_seconds_remaining)
                : null;
            const finishLabel =
              runActive && runStatus.eta_finish_at
                ? formatEtaFinishAt(runStatus.eta_finish_at)
                : null;
            return (
              <>
          <div className="stats">
            <div className="metric">
              <span className="label">Status</span>
              <span className="value">{runStatus.status}</span>
            </div>
            <div className="metric">
              <span className="label">Total</span>
              <span className="value">{runStatus.total_images}</span>
            </div>
            <div className="metric">
              <span className="label">Processed</span>
              <span className="value">{runStatus.processed_images}</span>
            </div>
            <div className="metric">
              <span className="label">Failed</span>
              <span className="value">{runStatus.failed_images}</span>
            </div>
            <div className="metric">
              <span className="label">Needs review</span>
              <span className="value">{stats.reviewNeeded}</span>
            </div>
            <div className="metric">
              <span className="label">Progress</span>
              <span className="value">{Number(runStatus.progress_pct || 0).toFixed(1)}%</span>
            </div>
            <div className="metric">
              <span className="label">ETA</span>
              <span className="value">{etaLabel || (runActive ? "…" : "—")}</span>
            </div>
            <div className="metric">
              <span className="label">Finish by</span>
              <span className="value">{finishLabel || (runActive ? "…" : "—")}</span>
            </div>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${Math.min(100, Number(runStatus.progress_pct || 0))}%` }}
            />
          </div>
          <div className="stats">
            {runStatus.avg_infer_ms_per_image != null && (
              <span>
                Avg infer: {Number(runStatus.avg_infer_ms_per_image).toFixed(1)} ms/image
              </span>
            )}
            {etaLabel && (
              <span>
                Expected finish in {etaLabel}
                {finishLabel ? ` (~${finishLabel})` : ""}
              </span>
            )}
            {runStatus.inference_mode && <span>Mode: {runStatus.inference_mode}</span>}
            {runStatus.cancel_requested && <span>Cancel requested</span>}
          </div>
              </>
            );
          })()}
          {runActive && (
            <div className="actions">
              <button type="button" className="danger" onClick={cancelActiveRun}>
                Cancel run
              </button>
            </div>
          )}
          {!runActive &&
            stats.reviewNeeded > 0 &&
            ["completed", "failed", "cancelled"].includes(runStatus.status) && (
              <div className="actions reclassify-banner" role="region" aria-label="Reclassify needs review">
                <span>
                  {stats.reviewNeeded} item{stats.reviewNeeded === 1 ? " needs" : "s need"} review — retry with
                  another model (approved/rejected/migrated stay untouched). SwinV2 is much faster than
                  EVA02.
                </span>
                <label className="reclassify-model">
                  <span className="sr-only">Reclassify model</span>
                  <select
                    value={reclassifyModel}
                    onChange={(e) => setReclassifyModel(e.target.value)}
                    disabled={opsLoading.reclassifying}
                    aria-label="Model for reclassify"
                  >
                    {TAGGER_MODELS.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  disabled={
                    opsLoading.reclassifying || selectedTags.length === 0 || anyRunActive
                  }
                  onClick={handleReclassifyNeedsReview}
                  aria-label="Reclassify needs review items"
                >
                  {opsLoading.reclassifying ? "Queuing…" : "Reclassify needs review"}
                </button>
              </div>
            )}
          {runStatus.last_error && <div className="error">{runStatus.last_error}</div>}
        </section>
      )}

      {runId && (
        <section className="panel">
          <h2>Review</h2>
          <span className="kicker">
            Primary is the main destination folder; category tags (secondary) are extra labels
            for junctions on Doujin runs
            {settings.tagging_domain === "real_life"
              ? ", or supporting real-life categories. Sensitive suggestions are highlighted."
              : ". Global tops help spot mis-assignments."}
          </span>
          <div className="stats">
            <span>Listed: {stats.total}</span>
            <span>Needs review: {stats.reviewNeeded}</span>
            <span>Approved: {stats.approved}</span>
            <span>Rejected: {stats.rejected}</span>
            <span>Migrated: {stats.migrated}</span>
          </div>
          <div className="actions">
            <button
              type="button"
              disabled={opsLoading.applyingBatch || selectedIds.length === 0}
              onClick={() => applyBatch("approved")}
            >
              Approve selected{selectedIds.length ? ` (${selectedIds.length})` : ""}
            </button>
            <button
              type="button"
              className="secondary"
              disabled={opsLoading.applyingBatch || selectedIds.length === 0}
              onClick={() => applyBatch("rejected")}
            >
              Reject selected{selectedIds.length ? ` (${selectedIds.length})` : ""}
            </button>
            <label className="migrate-mode">
              <span className="sr-only">Migrate mode</span>
              <select
                value={migrateMode}
                onChange={(e) => setMigrateMode(e.target.value)}
                disabled={opsLoading.migrating}
              >
                <option value="copy">copy</option>
                <option value="move">move</option>
              </select>
            </label>
            <button
              type="button"
              disabled={opsLoading.migrating || runActive || stats.approved === 0}
              title={
                runActive
                  ? "Wait for the run to finish (or cancel it) before migrating"
                  : stats.approved === 0
                    ? "Approve items first"
                    : undefined
              }
              onClick={migrateApproved}
            >
              {opsLoading.migrating ? "Migrating…" : `Migrate approved (${stats.approved})`}
            </button>
          </div>

          <div className="table-wrap">
          <table className="review-table">
            <thead>
              <tr>
                <th scope="col" className="select-col">
                  <input
                    type="checkbox"
                    aria-label="Select all items that are not migrated yet"
                    checked={allSelectableSelected}
                    ref={(el) => {
                      if (el) el.indeterminate = someSelectableSelected && !allSelectableSelected;
                    }}
                    disabled={selectableIds.length === 0}
                    onChange={(e) => setSelectedIds(e.target.checked ? selectableIds : [])}
                  />
                </th>
                <th>Work / media</th>
                <th>Primary</th>
                <th>Category tags</th>
                <th>Global top</th>
                <th>Status</th>
                <th>Final tag</th>
                <th>Debug</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => {
                const isMigrated = item.status === "migrated";
                const rowBusy = Boolean(pendingItemIds[item.id]);
                const itemLabel = item.relative_path || item.file_path;
                return (
                <tr
                  key={item.id}
                  className={[
                    item.needs_review ? "needs-review" : "",
                    `row-${item.status}`,
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  <td className="select-col">
                    <input
                      type="checkbox"
                      aria-label={`Select ${itemLabel}`}
                      checked={selectedIdSet.has(item.id)}
                      disabled={isMigrated}
                      onChange={(e) => {
                        const { checked } = e.target;
                        setSelectedIds((prev) =>
                          checked
                            ? prev.includes(item.id)
                              ? prev
                              : [...prev, item.id]
                            : prev.filter((id) => id !== item.id)
                        );
                      }}
                    />
                  </td>
                  <td title={item.file_path}>
                    <div className="image-cell">
                      {!previewErrors[item.id] && api.getItemPreviewUrl(item.id) ? (
                        <div className="image-thumb-wrap">
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
                          {isVideoPreviewPath(item.file_path || item.relative_path) ? (
                            <span className="media-badge" title="Video">
                              ▶
                            </span>
                          ) : null}
                        </div>
                      ) : null}
                      <div className="image-path">{item.relative_path || item.file_path || "-"}</div>
                      {item.review_reason && (
                        <div className="help">{item.review_reason}</div>
                      )}
                    </div>
                  </td>
                  <td>
                    {item.primary_tag ? (
                      <>
                        {item.primary_tag}
                        {REAL_LIFE_SENSITIVE_TAGS.has(item.primary_tag) ? (
                          <span className="chip warn" title="Sensitive — confirm Final tag">
                            sensitive
                          </span>
                        ) : null}
                        {String(item.review_reason || "").includes("vlm_unavailable") ? (
                          <span className="chip warn" title="Adult VLM unavailable">
                            fallback
                          </span>
                        ) : null}
                        <div className="muted">
                          {item.primary_score != null
                            ? Number(item.primary_score).toFixed(3)
                            : ""}
                        </div>
                      </>
                    ) : (
                      <span className="primary-empty">needs review</span>
                    )}
                  </td>
                  <td>
                    <div className="tag-list">
                      {(item.secondary_suggestions || []).length === 0
                        ? "-"
                        : (item.secondary_suggestions || []).map((s) => (
                            <span
                              key={`${item.id}-${s.tag}`}
                              className={
                                REAL_LIFE_SENSITIVE_TAGS.has(s.tag) ? "chip warn" : undefined
                              }
                            >
                              {formatTagScore(s)}
                              {REAL_LIFE_SENSITIVE_TAGS.has(s.tag) ? " ⚠" : ""}
                            </span>
                          ))}
                    </div>
                  </td>
                  <td>
                    <div className="tag-list">
                      {(item.global_top_tags || []).length === 0
                        ? "-"
                        : (item.global_top_tags || []).slice(0, 5).map((s) => (
                            <span key={`${item.id}-g-${s.tag}`}>{formatTagScore(s)}</span>
                          ))}
                    </div>
                  </td>
                  <td>
                    <span className={`status-badge status-${item.status}`}>{item.status}</span>
                    {isMigrated && item.migrated_to ? (
                      <div className="help migrated-path" title={item.migrated_to}>
                        {"→\u00a0"}
                        {pathTail(item.migrated_to)}
                      </div>
                    ) : null}
                  </td>
                  <td>
                    <input
                      className="final-tag-input"
                      aria-label={`Final tag for ${itemLabel}`}
                      list="final-tag-options"
                      value={finalTagDrafts[item.id] ?? (item.final_tag || "")}
                      placeholder={isMigrated ? "" : "Destination tag"}
                      disabled={isMigrated}
                      onChange={(e) => setFinalTagDraft(item.id, e.target.value)}
                      onBlur={() => commitFinalTag(item)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          e.currentTarget.blur();
                        } else if (e.key === "Escape") {
                          dropFinalTagDraft(item.id);
                        }
                      }}
                    />
                  </td>
                  <td>
                    <button
                      type="button"
                      className="secondary"
                      aria-expanded={Boolean(expandedScoreRows[item.id])}
                      onClick={() => toggleScoreDebug(item.id)}
                    >
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
                    {isMigrated ? (
                      <span className="muted">Done</span>
                    ) : (
                      <div className="row-actions">
                        <button
                          type="button"
                          disabled={rowBusy || item.status === "approved"}
                          onClick={() => updateStatus(item.id, "approved")}
                        >
                          Approve
                        </button>
                        <button
                          type="button"
                          className="secondary"
                          disabled={rowBusy || item.status === "rejected"}
                          onClick={() => updateStatus(item.id, "rejected")}
                        >
                          Reject
                        </button>
                        <button
                          type="button"
                          className="secondary"
                          disabled={rowBusy || item.status === "reviewed"}
                          onClick={() => updateStatus(item.id, "reviewed")}
                        >
                          Reviewed
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
          </div>
          <datalist id="final-tag-options">
            {selectedTags.map((tag) => (
              <option key={tag} value={tag} />
            ))}
          </datalist>
        </section>
      )}

      <footer className="app-footer" aria-label="Attribution">
        <p>
          THR3SHR v{APP_VERSION} © 2026 Dinamush · code MIT · creative{" "}
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
        </p>
      </footer>
    </div>
  );
}

export default App;
