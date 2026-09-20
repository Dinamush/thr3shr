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
const ACTIVE_RUN_STORAGE_KEY = "thr3shrActiveRunId";
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

function readRoute() {
  const hash = window.location.hash.replace(/^#/, "") || "/";
  return hash.startsWith("/debug") ? "debug" : "home";
}

function App() {
  const [route, setRoute] = useState(readRoute);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [savedSnapshot, setSavedSnapshot] = useState(
    settingsSnapshot(DEFAULT_SETTINGS, [])
  );
  const [offlineMode, setOfflineMode] = useState(false);
  const [runId, setRunId] = useState(null);
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
  const [scanPreview, setScanPreview] = useState(null);
  const [scanPreviewLoading, setScanPreviewLoading] = useState(false);
  const [opsLoading, setOpsLoading] = useState({
    saving: false,
    startingRun: false,
    updatingStatus: false,
    applyingBatch: false,
    migrating: false,
    reclassifying: false,
    unloadingModels: false,
  });
  const [reclassifyModel, setReclassifyModel] = useState("wd_swinv2_v3");
  const finalTagTimersRef = useRef({});
  const pollTimerRef = useRef(null);
  const lastItemsRefreshAtRef = useRef(0);
  const lastItemsProcessedRef = useRef(-1);
  const tagsHydratedRef = useRef(false);
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

  async function refreshItems(currentRunId, options = {}) {
    if (!currentRunId) return;
    try {
      const [runInfo, runItems] = await Promise.all([
        api.getRun(currentRunId),
        api.getRunItems(currentRunId, {
          timeoutMs: Number(options.timeoutMs) > 0 ? Number(options.timeoutMs) : 120000,
        }),
      ]);
      setRunMeta(runInfo);
      setItems(runItems);
      // Clear sticky thumb failures so a fixed preview endpoint can retry.
      setPreviewErrors({});
    } catch (err) {
      console.error("refreshItems failed", err);
      setError(`Failed to refresh items: ${err.message}`);
    }
  }

  useEffect(() => {
    api
      .getSettings()
      .then((data) => {
        const merged = { ...DEFAULT_SETTINGS, ...data };
        const tags = Array.isArray(data.selected_tags) ? data.selected_tags : [];
        setSettings(merged);
        skipNextTagPersistRef.current = true;
        setSelectedTags(tags);
        tagsHydratedRef.current = true;
        setMigrateMode(data.default_migrate_mode || "copy");
        setSavedSnapshot(settingsSnapshot(merged, tags));
        setOfflineMode(api.isOfflineMode());
      })
      .catch((err) => setError(err.message));
    const storedRunId = Number(localStorage.getItem(ACTIVE_RUN_STORAGE_KEY) || 0);
    if (storedRunId > 0) {
      setRunId(storedRunId);
    }
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
      const next = { ...settings, selected_tags: selectedTags };
      setSettings(next);
      api
        .saveSettings(next)
        .then((saved) => {
          const merged = { ...DEFAULT_SETTINGS, ...saved };
          setSettings(merged);
          setSavedSnapshot(
            settingsSnapshot(merged, Array.isArray(saved.selected_tags) ? saved.selected_tags : selectedTags)
          );
        })
        .catch(() => {
          /* ignore transient save errors while typing */
        });
    }, 400);
    return () => clearTimeout(timer);
    // Intentionally depend on selectedTags only to avoid save loops from settings edits.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTags]);

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

  async function pollRunStatus(currentRunId) {
    try {
      const status = await api.getRunStatus(currentRunId);
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
        if (pollTimerRef.current) {
          clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
        }
        localStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
        api.getProviders().then((info) => setProviderInfo(info)).catch(() => {});
      }
    } catch (err) {
      // Keep polling through transient API/load blips; a hard stop made the
      // UI look frozen while the backend was still classifying.
      setError(`Failed to poll run status: ${err.message}`);
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
    // Drop sticky thumb failures from prior preview bugs / aborted video loads.
    setPreviewErrors({});
    lastItemsRefreshAtRef.current = 0;
    lastItemsProcessedRef.current = -1;
    startStatusPolling(runId);
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [runId]);

  async function handleSaveSettings(e) {
    if (e) e.preventDefault();
    setLoading(true);
    setOpsLoading((prev) => ({ ...prev, saving: true }));
    setError("");
    try {
      const saved = await api.saveSettings({ ...settings, selected_tags: selectedTags });
      const merged = { ...DEFAULT_SETTINGS, ...saved };
      setSettings(merged);
      const tags = Array.isArray(saved.selected_tags) ? saved.selected_tags : selectedTags;
      if (Array.isArray(saved.selected_tags)) {
        skipNextTagPersistRef.current = true;
        setSelectedTags(saved.selected_tags);
      }
      setSavedSnapshot(settingsSnapshot(merged, tags));
      setOfflineMode(api.isOfflineMode());
      const providers = await api.getProviders();
      setProviderInfo(providers);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setOpsLoading((prev) => ({ ...prev, saving: false }));
    }
  }

  async function handlePreviewScan() {
    setScanPreviewLoading(true);
    setError("");
    try {
      if (isDirty) {
        await handleSaveSettings();
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
      if (isDirty) {
        await handleSaveSettings();
      }
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
        tagger_model: settings.tagger_model,
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
      const confirmed = window.confirm(
        "Reject selected items? They stay on disk and are skipped by migrate/reclassify."
      );
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
    setNotice("");
    const confirmed = window.confirm(
      `Migrate approved items using ${migrateMode}? This can move/copy many files.`
    );
    if (!confirmed) {
      setLoading(false);
      setOpsLoading((prev) => ({ ...prev, migrating: false }));
      return;
    }
    try {
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
    if (runActive || opsLoading.unloadingModels) return;
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
    if (!runId || runActive) return;
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
    skipNextTagPersistRef.current = true;
    if (enabled) {
      const parked = selectedTags.filter((t) => !SFW_CLASSIFY_FOLDERS.includes(t));
      const nextTags = [...SFW_CLASSIFY_FOLDERS];
      const next = {
        ...settings,
        sfw_classify_mode: true,
        selected_tags: nextTags,
        selected_tags_nsfw: parked.length ? parked : parkedNsfwTags,
      };
      setSelectedTags(nextTags);
      setSettings(next);
      try {
        const saved = await api.saveSettings(next);
        const merged = { ...DEFAULT_SETTINGS, ...saved };
        setSettings(merged);
        setSelectedTags(
          Array.isArray(saved.selected_tags) ? saved.selected_tags : nextTags
        );
        setSavedSnapshot(
          settingsSnapshot(
            merged,
            Array.isArray(saved.selected_tags) ? saved.selected_tags : nextTags
          )
        );
      } catch (err) {
        setError(`Failed to enable SFW mode: ${err.message}`);
      }
      return;
    }

    const restored =
      parkedNsfwTags.length > 0
        ? parkedNsfwTags
        : selectedTags.filter((t) => !SFW_CLASSIFY_FOLDERS.includes(t));
    const next = {
      ...settings,
      sfw_classify_mode: false,
      selected_tags: restored,
      selected_tags_nsfw: restored,
    };
    setSelectedTags(restored);
    setSettings(next);
    try {
      const saved = await api.saveSettings(next);
      const merged = { ...DEFAULT_SETTINGS, ...saved };
      setSettings(merged);
      setSelectedTags(
        Array.isArray(saved.selected_tags) ? saved.selected_tags : restored
      );
      setSavedSnapshot(
        settingsSnapshot(
          merged,
          Array.isArray(saved.selected_tags) ? saved.selected_tags : restored
        )
      );
    } catch (err) {
      setError(`Failed to disable SFW mode: ${err.message}`);
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
              disabled={runActive || opsLoading.unloadingModels}
              onClick={handleUnloadModels}
              title={
                runActive
                  ? "Cancel the current run before unloading GPU models"
                  : "Drop cached tagger sessions to free GPU memory"
              }
            >
              {opsLoading.unloadingModels ? "Unloading…" : "Unload GPU models"}
            </button>
          </div>
        )}
      </header>

      {offlineMode && (
        <div className="error">
          Backend is offline. Running in browser-only demo mode with local mock data.
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
              <div className="scan-preview">
                Eligible: {scanPreview.stats.eligible_images} · Total files:{" "}
                {scanPreview.stats.total_files} · Ignored GIF: {scanPreview.stats.ignored_gif} ·
                Unsupported: {scanPreview.stats.ignored_unsupported}
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
                <span className="help">
                  Selected-tag scores below this are suggestions only (no primary assignment).
                </span>
              </label>
              {isWdModel && (
                <label>
                  WD general threshold
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={settings.wd_general_threshold}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        wd_general_threshold: Number(e.target.value),
                      })
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
                <input
                  type="number"
                  min="1"
                  max="16"
                  value={settings.max_inference_workers ?? 2}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      max_inference_workers: Math.max(
                        1,
                        Math.min(16, Number(e.target.value) || 1)
                      ),
                    })
                  }
                />
                <span className="help">Keep at 2 for GPU ORT sessions (serialized lock).</span>
              </label>
              <label>
                Inference batch size
                <input
                  type="number"
                  min="1"
                  max="64"
                  value={settings.inference_batch_size ?? 8}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      inference_batch_size: Math.max(
                        1,
                        Math.min(64, Number(e.target.value) || 8)
                      ),
                    })
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
                  onChange={(e) =>
                    setSettings({ ...settings, force_cpu_inference: e.target.checked })
                  }
                />
                Force CPU inference
              </label>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={Boolean(settings.experimental_media_enabled)}
                  onChange={(e) =>
                    setSettings({ ...settings, experimental_media_enabled: e.target.checked })
                  }
                />
                Experimental: classify GIF/videos via length-scaled multi-frame sampling
              </label>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={Boolean(settings.experimental_style_detector_enabled)}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      experimental_style_detector_enabled: e.target.checked,
                    })
                  }
                />
                Experimental: real vs anime style gate (dedicated ONNX →{" "}
                <code>real_life</code>; uncertain → needs review)
              </label>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={Boolean(settings.hybrid_ml_on_review)}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      hybrid_ml_on_review: e.target.checked,
                    })
                  }
                  disabled={settings.tagging_domain === "real_life"}
                />
                Hybrid: on needs-review (WD primary), re-run ML-Danbooru and merge
                allowlisted high-recall tags only (skips Voyeur soft cues)
              </label>
              <fieldset className="domain-toggle">
                <legend>Tagging domain</legend>
                <label className="inline-check">
                  <input
                    type="radio"
                    name="tagging_domain"
                    checked={settings.tagging_domain !== "real_life"}
                    onChange={() =>
                      setSettings({ ...settings, tagging_domain: "drawn" })
                    }
                  />
                  Animated / drawn (WD + ML-Danbooru taxonomy)
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
                  Real-life photos / videos (local adult tagger)
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
            </div>
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
        <div className="actions">
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
            Add typed tag(s)
          </button>
          <button
            type="button"
            disabled={loading || opsLoading.startingRun || runActive}
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
              runActive ||
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
              runActive ||
              settings.tagging_domain === "real_life"
            }
            onClick={() => handleStartRun("doujin_works")}
          >
            {opsLoading.startingRun ? "Starting…" : "Start Doujin works"}
          </button>
        </div>
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
      </section>

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
                  {stats.reviewNeeded} item{stats.reviewNeeded === 1 ? "" : "s"} need review — retry with
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
                  disabled={opsLoading.reclassifying || selectedTags.length === 0}
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
              disabled={opsLoading.applyingBatch}
              onClick={() => applyBatch("approved")}
            >
              Approve selected
            </button>
            <button
              type="button"
              className="secondary"
              disabled={opsLoading.applyingBatch}
              onClick={() => applyBatch("rejected")}
            >
              Reject selected
            </button>
            <select value={migrateMode} onChange={(e) => setMigrateMode(e.target.value)}>
              <option value="copy">copy</option>
              <option value="move">move</option>
            </select>
            <button type="button" disabled={opsLoading.migrating} onClick={migrateApproved}>
              {opsLoading.migrating ? "Migrating…" : "Migrate approved"}
            </button>
          </div>

          <table>
            <thead>
              <tr>
                <th>Select</th>
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
                  <td>{item.status}</td>
                  <td>
                    <input
                      value={finalTagDrafts[item.id] ?? (item.final_tag || "")}
                      onChange={(e) => queueFinalTagUpdate(item.id, e.target.value)}
                    />
                  </td>
                  <td>
                    <button type="button" className="secondary" onClick={() => toggleScoreDebug(item.id)}>
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
                    <div className="actions">
                      <button
                        type="button"
                        disabled={opsLoading.updatingStatus}
                        onClick={() => updateStatus(item.id, "approved")}
                      >
                        Approve
                      </button>
                      <button
                        type="button"
                        className="secondary"
                        disabled={opsLoading.updatingStatus}
                        onClick={() => updateStatus(item.id, "rejected")}
                      >
                        Reject
                      </button>
                      <button
                        type="button"
                        className="secondary"
                        disabled={opsLoading.updatingStatus}
                        onClick={() => updateStatus(item.id, "reviewed")}
                      >
                        Reviewed
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {runMeta && <pre className="debug-json">{JSON.stringify(runMeta.counts, null, 2)}</pre>}
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
