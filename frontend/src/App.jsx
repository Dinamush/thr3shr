import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import ClassifierDebugPage from "./ClassifierDebugPage";
import { version as APP_VERSION } from "../package.json";
import { formatDurationSeconds, runOptionLabel, runPct } from "./lib";
import ReviewPage from "./pages/ReviewPage";
import RunPage from "./pages/RunPage";
import RunsPage from "./pages/RunsPage";
import SettingsPage from "./pages/SettingsPage";
import AppShell from "./shell/AppShell";
import { pageHref, routeFromHash, staticJumpEntries } from "./shell/nav";
import RunDock from "./shell/RunDock";
import { AppearanceProvider } from "./theme/Appearance";
import YorhaBoot from "./theme/YorhaBoot";
import { AppContext } from "./ui";

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

const SFW_CLASSIFY_FOLDERS = ["SFW", "scenery"];
// Last run opened in the dashboard. Kept after the run finishes so a reload
// returns to its review table (key name predates that and stays for upgrades).
const LAST_RUN_STORAGE_KEY = "thr3shrActiveRunId";
const STATUS_POLL_INTERVAL_MS = 1500;

function settingsSnapshot(settings, selectedTags) {
  return JSON.stringify({
    ...settings,
    selected_tags: selectedTags,
  });
}

function sameList(a, b) {
  return a.length === b.length && a.every((value, idx) => value === b[idx]);
}

function readRoute() {
  return routeFromHash(window.location.hash);
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

  // ── Console shell ─────────────────────────────────────────────────────────

  // A jump-bar target waiting for its page to scroll to it; the page clears it.
  const [landing, setLanding] = useState(null);

  function navigate(pageId) {
    const href = pageHref(pageId);
    if (window.location.hash !== href) window.location.hash = href;
  }

  function openRunForReview(nextRunId) {
    openRun(nextRunId);
    navigate("review");
  }

  function handleJump(entry) {
    if (entry.run) {
      openRunForReview(entry.run);
      return;
    }
    navigate(entry.route);
    if (entry.target) setLanding({ target: entry.target, at: Date.now() });
  }

  const dirtyCount = useMemo(() => {
    if (!isDirty) return 0;
    let saved = {};
    try {
      saved = JSON.parse(savedSnapshot);
    } catch {
      saved = {};
    }
    const current = { ...settings, selected_tags: selectedTags };
    let changed = 0;
    new Set([...Object.keys(saved), ...Object.keys(current)]).forEach((key) => {
      if (JSON.stringify(saved[key]) !== JSON.stringify(current[key])) changed += 1;
    });
    return Math.max(1, changed);
  }, [isDirty, savedSnapshot, settings, selectedTags]);

  // Back to what the server holds.  Destination tags save on their own, so
  // they are not part of the revert.
  function revertSettings() {
    if (!savedSettingsRef.current) return;
    setSettings((prev) => ({
      ...prev,
      ...savedSettingsRef.current,
      selected_tags: prev.selected_tags,
    }));
  }

  const jumpEntries = useMemo(
    () => [
      ...staticJumpEntries(),
      ...runs.map((run) => ({
        key: `run-${run.run_id}`,
        label: runOptionLabel(run),
        group: "Run",
        run: run.run_id,
        quiet: true,
      })),
    ],
    [runs]
  );

  function dockModel() {
    const review = runId && stats.reviewNeeded > 0 ? { count: stats.reviewNeeded } : null;
    if (opsLoading.migrating) {
      return {
        state: "migrating",
        label: `Migrating ${stats.approved} approved (${migrateMode})`,
        job: {
          name: `Run #${runId}`,
          indeterminate: true,
          detail: migrateMode,
          href: "#/review",
        },
        review,
      };
    }
    const live =
      runActive && runStatus
        ? { ...runStatus, run_id: runId, href: "#/" }
        : otherActiveRun
          ? { ...otherActiveRun, href: "#/runs" }
          : null;
    if (live) {
      const total = Number(live.total_images) || 0;
      const processed = Number(live.processed_images) || 0;
      const pct = runPct(live);
      const eta =
        live.eta_seconds_remaining != null
          ? formatDurationSeconds(live.eta_seconds_remaining)
          : null;
      const queued = live.status === "pending";
      return {
        state: queued ? "queued" : "running",
        label: queued ? `Run #${live.run_id} queued` : `Tagging run #${live.run_id}`,
        job: {
          name: `Run #${live.run_id}`,
          pct,
          indeterminate: !total,
          detail: total
            ? `${processed}/${total} · ${pct.toFixed(0)}%${eta ? ` · ETA ${eta}` : ""}`
            : "starting",
          href: live.href,
        },
        review,
      };
    }
    if (offlineMode) return { state: "offline", label: "Demo mode · backend offline", review };
    const last = runs[0];
    return {
      state: "idle",
      label: last ? `Idle · last run #${last.run_id} ${last.status}` : "Idle · no runs yet",
      review,
    };
  }

  const badges = {
    review:
      runId && stats.reviewNeeded > 0
        ? {
            text: String(stats.reviewNeeded),
            tone: "warn",
            title: `${stats.reviewNeeded} items need review`,
          }
        : null,
    runs: anyRunActive ? { text: "live", tone: "running", title: "A run is active" } : null,
    settings: isDirty
      ? { text: String(dirtyCount), tone: "warn", title: "Unsaved settings changes" }
      : null,
  };

  const engine = offlineMode
    ? { tone: "bad", label: "Demo mode", title: "Backend offline: browser-only mock data" }
    : providerInfo
      ? {
          tone: providerInfo.cuda_usable && !providerInfo.forced_cpu ? "ok" : "warn",
          label: `${providerInfo.likely_device || "unknown"} · ${providerInfo.tagger_model || settings.tagger_model}`,
          title: `Active: ${(providerInfo.active_providers || []).join(", ") || "n/a"}`,
        }
      : null;

  const banners =
    offlineMode || (providerInfo?.provider_error && !offlineMode) ? (
      <>
        {offlineMode ? (
          <div
            className={`app-banner offline-banner ${backendBack ? "is-info" : "is-bad"}`}
            role="status"
          >
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
              <button type="button" className="btn btn-sm" onClick={() => window.location.reload()}>
                Reconnect
              </button>
            ) : null}
          </div>
        ) : null}
        {providerInfo?.provider_error && !offlineMode ? (
          <div className="app-banner is-bad" role="alert">
            {providerInfo.provider_error}
          </div>
        ) : null}
      </>
    ) : null;

  const app = {
    settings,
    setSettings,
    isDirty,
    dirtyCount,
    isWdModel,
    loading,
    opsLoading,
    offlineMode,
    providerInfo,
    realLifeStatus,
    realLifeStatusLoading,
    runs,
    runId,
    runStatus,
    runActive,
    otherActiveRun,
    anyRunActive,
    items,
    stats,
    selectedIds,
    setSelectedIds,
    selectedIdSet,
    previewErrors,
    setPreviewErrors,
    pendingItemIds,
    finalTagDrafts,
    expandedScoreRows,
    scoreDebugByItem,
    scoreLoadingByItem,
    migrateMode,
    setMigrateMode,
    reclassifyModel,
    setReclassifyModel,
    tagQuery,
    setTagQuery,
    tagOptions,
    selectedTags,
    sfwClassifyMode,
    parkedNsfwTags,
    scanPreview,
    scanPreviewLoading,
    landing,
    setLanding,
    setError,
    openRun,
    openRunForReview,
    handleSaveSettings,
    revertSettings,
    handlePreviewScan,
    handleStartRun,
    updateStatus,
    setFinalTagDraft,
    dropFinalTagDraft,
    commitFinalTag,
    applyBatch,
    migrateApproved,
    cancelActiveRun,
    handleUnloadModels,
    handleReclassifyNeedsReview,
    toggleScoreDebug,
    setSfwClassifyMode,
    addSelectedTag,
    addTagsFromInput,
    removeSelectedTag,
  };

  let page;
  if (route === "review") page = <ReviewPage />;
  else if (route === "runs") page = <RunsPage />;
  else if (route === "settings") page = <SettingsPage />;
  else if (route === "debug") page = <ClassifierDebugPage landing={landing} onLanded={() => setLanding(null)} />;
  else page = <RunPage />;

  return (
    <AppContext.Provider value={app}>
      <AppearanceProvider>
        <AppShell
          route={route}
          badges={badges}
          engine={engine}
          version={APP_VERSION}
          jumpEntries={jumpEntries}
          onJump={handleJump}
          banners={banners}
          dock={<RunDock model={dockModel()} />}
        >
          {page}
        </AppShell>
        <div className="toasts">
          {error ? (
            <div className="toast is-bad" role="alert">
              <span className="toast-text">{error}</span>
              <button
                type="button"
                className="toast-close"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                ×
              </button>
            </div>
          ) : null}
          {notice ? (
            <div className="toast is-ok" role="status">
              <span className="toast-text">{notice}</span>
              <button
                type="button"
                className="toast-close"
                aria-label="Dismiss notice"
                onClick={() => setNotice("")}
              >
                ×
              </button>
            </div>
          ) : null}
        </div>
        <YorhaBoot />
      </AppearanceProvider>
    </AppContext.Provider>
  );
}

export default App;
